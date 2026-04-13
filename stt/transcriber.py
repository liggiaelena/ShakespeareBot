import whisper
from config import WHISPER_MODEL


class Transcriber:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Receives audio path, returns (text, language_code)."""
        result = self.model.transcribe(audio_path)
        text = result["text"].strip()
        language = result["language"]   # e.g. "es", "fr", "sw", "en"
        return text, language
