"""
ShakespeareBot — FastAPI Backend
Run with:  uvicorn api.server:app --reload
           (from the Agribot/ directory)
"""
import asyncio
import base64
import os
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure imports resolve from Agribot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WHISPER_MODEL
from llm.agent import Agent
from rag.retriever import Retriever
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


def _build_response(question: str, language: str) -> dict:
    """RAG → LLM → TTS. Blocking — runs inside a thread pool."""
    chunks = _retriever.search(question)
    text_response = _agent.ask(question=question, context=chunks, language=language)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    _synthesizer.speak(text_response, language=language, output_path=tmp_path)

    with open(tmp_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    return {"text": text_response, "audio_base64": audio_b64}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/chat/text")
async def chat_text(req: TextRequest):
    """Receive a typed message and return Shakespeare's text + audio reply."""
    language = await asyncio.to_thread(_detect_language, req.message)
    try:
        result = await asyncio.to_thread(_build_response, req.message, language)
        result["user_text"] = req.message
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """Receive a recorded audio blob, transcribe it, and return Shakespeare's reply."""
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
            os.unlink(tmp_audio)

    try:
        result = await asyncio.to_thread(_build_response, user_text, language)
        result["user_text"] = user_text
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
