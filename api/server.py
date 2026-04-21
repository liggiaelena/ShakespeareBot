"""
ShakespeareBot — FastAPI Backend
Run with:  uvicorn api.server:app --reload
           (from the ShakespeareBot/ directory)
"""
import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Make sure imports resolve from ShakespeareBot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WHISPER_MODEL
from llm.agent import Agent
from rag.retriever import Retriever
from src.classifier import classify_query
from stt.transcriber import Transcriber
from tts.synthesizer import Synthesizer

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ShakespeareBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load heavy models once ──────────────────────────────────────────

_retriever: Retriever | None = None
_agent: Agent | None = None
_synthesizer: Synthesizer | None = None
_transcriber: Transcriber | None = None


@app.on_event("startup")
async def _startup():
    global _retriever, _agent, _synthesizer, _transcriber
    _retriever = Retriever()
    _agent = Agent()
    _synthesizer = Synthesizer()
    _transcriber = Transcriber(model_name=WHISPER_MODEL)


# ── Conversation history (in-memory, resets on restart) ──────────────────────

_conversation_history: list[dict] = []
_history_lock = threading.Lock()
_MAX_HISTORY = 6  # 3 turns: each turn = 1 user message + 1 assistant message


def _append_history(question: str, answer: str) -> None:
    global _conversation_history
    with _history_lock:
        _conversation_history.append({"role": "user", "content": question})
        _conversation_history.append({"role": "assistant", "content": answer})
        if len(_conversation_history) > _MAX_HISTORY:
            _conversation_history = _conversation_history[-_MAX_HISTORY:]


def _get_history() -> list[dict]:
    with _history_lock:
        return list(_conversation_history)


# ── Helpers ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    message: str


def _detect_language(text: str) -> str:
    """Best-effort language detection; falls back to English."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for incremental TTS.

    Splits on sentence-ending punctuation followed by whitespace, then merges
    any fragment shorter than 20 chars with the next sentence to avoid
    synthesising isolated abbreviations or single words.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    merged: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if merged and len(merged[-1]) < 20:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return [s for s in merged if s]


def _build_response(question: str, language: str) -> dict:
    """Non-streaming path — kept for fallback/testing. Agentic LLM → TTS."""
    # 1. Classify the query
    classification = classify_query(question)

    # 2. Enrich the search query toward the right part of the corpus
    search_query = question
    if classification["label"] == "tragedy" and classification["confidence"] > 0.6:
        search_query = question + " tragedy themes characters"
    elif classification["label"] == "non-tragedy" and classification["confidence"] > 0.6:
        search_query = question + " history comedy sonnets"

    # 3. Search with the enriched query
    history = _get_history()
    text_response = _agent.ask(
        question=question,
        search_fn=lambda q: _retriever.search(search_query),
        history=history,
        language=language,
        query_type=classification["label"],
        confidence=classification["confidence"],
    )
    _append_history(question, text_response)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    _synthesizer.speak(text_response, language=language, output_path=tmp_path)
    with open(tmp_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    result = {"text": text_response, "audio_base64": audio_b64}
    result["query_type"] = classification["label"]
    result["confidence"] = round(classification["confidence"], 3)
    return result


async def _stream_response(question: str, language: str, user_text: str):
    """Async generator: agentic loop → sentence-by-sentence TTS → SSE events.

    Yields SSE lines:
        data: {"type": "audio_chunk", "text": "...", "audio_base64": "..."}
        data: {"type": "done", "full_text": "...", "user_text": "...",
               "query_type": "...", "confidence": ...}
    """
    # 1. Classify the query
    classification = classify_query(question)

    # 2. Enrich the search query toward the right part of the corpus
    search_query = question
    if classification["label"] == "tragedy" and classification["confidence"] > 0.6:
        search_query = question + " tragedy themes characters"
    elif classification["label"] == "non-tragedy" and classification["confidence"] > 0.6:
        search_query = question + " history comedy sonnets"

    history = _get_history()

    # Run the agentic loop in a thread (it calls blocking Anthropic SDK + embedder)
    text_response = await asyncio.to_thread(
        _agent.ask,
        question=question,
        search_fn=lambda q: _retriever.search(search_query),
        history=history,
        language=language,
        query_type=classification["label"],
        confidence=classification["confidence"],
    )
    _append_history(question, text_response)

    sentences = _split_sentences(text_response)
    if not sentences:
        sentences = [text_response]

    for sentence in sentences:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            # speak_async works inside the running event loop
            await _synthesizer.speak_async(sentence, language=language, output_path=tmp_path)
            with open(tmp_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows file-lock; file will be cleaned up by OS

        event = json.dumps({"type": "audio_chunk", "text": sentence, "audio_base64": audio_b64})
        yield f"data: {event}\n\n"

    done = json.dumps({
        "type":       "done",
        "full_text":  text_response,
        "user_text":  user_text,
        "query_type": classification["label"],
        "confidence": round(classification["confidence"], 3),
    })
    yield f"data: {done}\n\n"


# ── Routes — non-streaming (kept for fallback) ────────────────────────────────

@app.post("/api/chat/text")
async def chat_text(req: TextRequest):
    """Non-streaming: receive typed message, return full text + audio."""
    language = await asyncio.to_thread(_detect_language, req.message)
    try:
        result = await asyncio.to_thread(_build_response, req.message, language)
        result["user_text"] = req.message
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """Non-streaming: receive audio blob, transcribe, return full text + audio."""
    content = await audio.read()
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_audio = f.name

    try:
        user_text, language = await asyncio.to_thread(
            _transcriber.transcribe, tmp_audio
        )
    except Exception as exc:
        os.unlink(tmp_audio)
        raise HTTPException(status_code=422, detail=f"Transcription failed: {exc}")
    finally:
        if os.path.exists(tmp_audio):
            try:
                os.unlink(tmp_audio)
            except PermissionError:
                pass

    try:
        result = await asyncio.to_thread(_build_response, user_text, language)
        result["user_text"] = user_text
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Routes — streaming ────────────────────────────────────────────────────────

@app.post("/api/chat/text/stream")
async def chat_text_stream(req: TextRequest):
    """Streaming: sentence-by-sentence SSE — audio starts before full response is ready."""
    language = await asyncio.to_thread(_detect_language, req.message)
    return StreamingResponse(
        _stream_response(req.message, language, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/audio/stream")
async def chat_audio_stream(audio: UploadFile = File(...)):
    """Streaming: transcribe audio then stream SSE sentence-by-sentence."""
    content = await audio.read()
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_audio = f.name

    try:
        user_text, language = await asyncio.to_thread(
            _transcriber.transcribe, tmp_audio
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_audio)
        except PermissionError:
            pass

    return StreamingResponse(
        _stream_response(user_text, language, user_text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
