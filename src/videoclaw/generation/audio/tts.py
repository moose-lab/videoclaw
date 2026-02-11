"""Text-to-Speech module -- synthesises voice-over audio from text.

Provides a :class:`TTSProvider` protocol for pluggable backends and a
production-ready :class:`EdgeTTSProvider` that uses Microsoft Edge TTS
(free, no API key required).  The :class:`TTSManager` is the main entry
point for the rest of the pipeline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTSProvider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSProvider(Protocol):
    """Structural interface for any TTS backend."""

    async def synthesize(
        self,
        text: str,
        voice: str,
        language: str,
    ) -> bytes:
        """Convert *text* to audio bytes (MP3 or WAV).

        Parameters
        ----------
        text:
            The text to speak.
        voice:
            A provider-specific voice identifier.
        language:
            ISO language code (e.g. ``"zh"``, ``"en"``).

        Returns
        -------
        bytes
            Raw audio data.
        """
        ...


# ---------------------------------------------------------------------------
# EdgeTTSProvider
# ---------------------------------------------------------------------------


class EdgeTTSProvider:
    """Free TTS provider using Microsoft Edge's online TTS API.

    Requires the ``edge-tts`` package (``pip install edge-tts``).
    """

    # Curated subset of high-quality neural voices
    AVAILABLE_VOICES: dict[str, list[str]] = {
        "zh": [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-YunyangNeural",
            "zh-CN-XiaoyiNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-XiaochenNeural",
            "zh-TW-HsiaoChenNeural",
            "zh-TW-YunJheNeural",
        ],
        "en": [
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-US-AriaNeural",
            "en-US-DavisNeural",
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural",
        ],
        "ja": [
            "ja-JP-NanamiNeural",
            "ja-JP-KeitaNeural",
        ],
        "ko": [
            "ko-KR-SunHiNeural",
            "ko-KR-InJoonNeural",
        ],
    }

    DEFAULT_VOICES: dict[str, str] = {
        "zh": "zh-CN-XiaoxiaoNeural",
        "en": "en-US-JennyNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
    }

    async def synthesize(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        language: str = "zh",
    ) -> bytes:
        """Synthesize speech using Edge TTS.

        Parameters
        ----------
        text:
            The text to convert to speech.
        voice:
            Edge TTS voice identifier.
        language:
            Language code (used as fallback for voice selection).

        Returns
        -------
        bytes
            MP3 audio data.
        """
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError(
                "edge-tts is required for EdgeTTSProvider. "
                "Install it with: pip install edge-tts"
            ) from exc

        # Fall back to a default voice if the requested one seems wrong
        if not voice or voice == "default":
            voice = self.DEFAULT_VOICES.get(language, "zh-CN-XiaoxiaoNeural")

        logger.info(
            "EdgeTTS synthesizing %d chars with voice=%s",
            len(text),
            voice,
        )

        communicate = edge_tts.Communicate(text, voice)

        # Collect all audio chunks into a buffer
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_data = audio_buffer.getvalue()
        if not audio_data:
            raise RuntimeError(
                f"Edge TTS returned empty audio for voice={voice!r}"
            )

        logger.info("EdgeTTS produced %d bytes of audio", len(audio_data))
        return audio_data


# ---------------------------------------------------------------------------
# TTSManager
# ---------------------------------------------------------------------------


class TTSManager:
    """High-level TTS manager used by the generation pipeline.

    Parameters
    ----------
    provider:
        A :class:`TTSProvider` implementation.  Defaults to
        :class:`EdgeTTSProvider` when *None*.
    """

    def __init__(self, provider: TTSProvider | None = None) -> None:
        self._provider: TTSProvider = provider or EdgeTTSProvider()

    @property
    def provider(self) -> TTSProvider:
        """The active TTS provider."""
        return self._provider

    async def generate_voiceover(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        language: str = "zh",
    ) -> Path:
        """Generate a voice-over audio file from text.

        Parameters
        ----------
        text:
            The narration text to speak.
        output_path:
            Where to write the audio file.
        voice:
            Optional voice identifier.  When *None*, a sensible default is
            chosen based on *language*.
        language:
            ISO language code.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        if not text.strip():
            raise ValueError("Cannot generate voiceover from empty text")

        # Resolve voice
        resolved_voice = voice
        if resolved_voice is None:
            if isinstance(self._provider, EdgeTTSProvider):
                resolved_voice = EdgeTTSProvider.DEFAULT_VOICES.get(
                    language, "zh-CN-XiaoxiaoNeural"
                )
            else:
                resolved_voice = "default"

        logger.info(
            "Generating voiceover: %d chars, voice=%s, output=%s",
            len(text),
            resolved_voice,
            output_path,
        )

        audio_data = await self._provider.synthesize(
            text=text,
            voice=resolved_voice,
            language=language,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_data)

        logger.info(
            "Voiceover saved: %s (%d bytes)",
            output_path,
            len(audio_data),
        )
        return output_path
