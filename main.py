"""
ShakespeareBot — Voice-Powered Historical Persona Chatbot
Usage: python main.py <audio_path> [output_path.mp3]
"""
import sys

from config import WHISPER_MODEL
from stt.transcriber import Transcriber
from rag.retriever import Retriever
from llm.agent import Agent
from tts.synthesizer import Synthesizer


def run(audio_path: str, output_path: str = "output.mp3") -> str:
    # Step 1 — STT: audio → text + language
    print(f"[1/4] Transcribing {audio_path}...")
    transcriber = Transcriber(model_name=WHISPER_MODEL)
    text, language = transcriber.transcribe(audio_path)
    print(f"      Text    : {text!r}")
    print(f"      Language: {language}")

    # Step 2 — RAG: retrieve relevant Shakespeare context
    print("[2/4] Searching vector store for context...")
    retriever = Retriever()
    chunks = retriever.search(text)
    print(f"      {len(chunks)} chunks found")

    # Step 3 — LLM: generate response with Claude
    print("[3/4] Generating response with Claude...")
    agent = Agent()
    response = agent.ask(question=text, context=chunks, language=language)
    print(f"      Response: {response[:120]}...")

    # Step 4 — TTS: response → audio
    print(f"[4/4] Synthesizing audio → {output_path}")
    synthesizer = Synthesizer()
    synthesizer.speak(response, language=language, output_path=output_path)

    print("\nDone! Response saved to:", output_path)
    return response


if __name__ == "__main__":
    audio_input = sys.argv[1] if len(sys.argv) > 1 else "input.ogg"
    audio_output = sys.argv[2] if len(sys.argv) > 2 else "output.mp3"
    run(audio_input, audio_output)
