import asyncio
import edge_tts

# Neural voices by language (edge-tts free library)
# en-GB-RyanNeural — British male, expressive and theatrical; period-appropriate for Shakespeare.
_VOICES: dict[str, str] = {
    "en": "en-GB-RyanNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "pt": "pt-BR-AntonioNeural",
    "sw": "sw-KE-RafikiNeural",
    "ar": "ar-SA-HamedNeural",
    "hi": "hi-IN-MadhurNeural",
    "ha": "ha-NE-AbdullahNeural",
}
_DEFAULT_VOICE = "en-GB-RyanNeural"


class Synthesizer:
    def speak(
        self,
        text: str,
        language: str = "en",
        output_path: str = "output.mp3",
    ) -> str:
        """Sync: converts text to audio and saves to output_path. Returns the path."""
        voice = _VOICES.get(language, _DEFAULT_VOICE)
        asyncio.run(self._synthesize(text, voice, output_path))
        return output_path

    async def speak_async(
        self,
        text: str,
        language: str = "en",
        output_path: str = "output.mp3",
    ) -> str:
        """Async: for use inside a running event loop (streaming endpoints)."""
        voice = _VOICES.get(language, _DEFAULT_VOICE)
        await self._synthesize(text, voice, output_path)
        return output_path

    @staticmethod
    async def _synthesize(text: str, voice: str, output_path: str) -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
