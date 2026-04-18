"""Tests the STT module with a synthetic audio file."""
import os
import wave
import struct
import tempfile
import pytest
from stt.transcriber import Transcriber


def _create_silent_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def test_transcriber_returns_tuple():
    transcriber = Transcriber(model_name="tiny")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    _create_silent_wav(tmp_path)
    try:
        text, language = transcriber.transcribe(tmp_path)
        assert isinstance(text, str)
        assert isinstance(language, str)
        assert len(language) == 2  # ISO code e.g. "en"
    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass  # Windows: Whisper may still hold the file handle; safe to ignore
