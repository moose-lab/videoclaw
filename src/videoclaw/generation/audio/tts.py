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


class WaveSpeedTTSProvider:
    """TTS provider using MiniMax speech-02-hd via WaveSpeed API.

    Optimal for Chinese short drama with fine-grained voice control
    (speed, pitch, emotion, volume per character).

    Pricing: $0.05 per 1,000 characters.
    """

    API_BASE = "https://api.wavespeed.ai/api/v3"
    SUBMIT_URL = f"{API_BASE}/minimax/speech-02-hd"
    RESULT_URL = f"{API_BASE}/predictions/{{request_id}}/result"

    # Defaults per language when no voice_id is specified
    DEFAULT_VOICES: dict[str, str] = {
        "zh": "Friendly_Person",
        "en": "Friendly_Person",
        "ja": "Calm_Woman",
        "ko": "Calm_Woman",
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> None:
        from videoclaw.config import get_config
        import os

        self._api_key = (
            api_key
            or os.environ.get("WAVESPEED_API_KEY")
            or get_config().wavespeed_api_key
        )
        if not self._api_key:
            raise ValueError(
                "WaveSpeed API key is required. Set WAVESPEED_API_KEY or "
                "VIDEOCLAW_WAVESPEED_API_KEY environment variable."
            )
        self._timeout = timeout
        self._poll_interval = poll_interval

    async def synthesize(
        self,
        text: str,
        voice: str = "",
        language: str = "zh",
        *,
        speed: float = 1.0,
        pitch: int = 0,
        emotion: str = "neutral",
        volume: float = 1.0,
    ) -> bytes:
        """Synthesize speech via MiniMax speech-02-hd.

        Parameters
        ----------
        text:
            Text to convert (max 10,000 chars).
        voice:
            MiniMax voice_id (e.g. ``"Friendly_Person"``).
        language:
            ISO language code (used for default voice selection).
        speed:
            Speech speed 0.50-2.00, default 1.0.
        pitch:
            Pitch adjustment -12 to 12, default 0.
        emotion:
            One of: happy, sad, angry, fearful, disgusted, surprised, neutral.
        volume:
            Volume 0.10-10.00, default 1.0.
        """
        import httpx

        if not voice or voice == "default":
            voice = self.DEFAULT_VOICES.get(language, "Friendly_Person")

        logger.info(
            "WaveSpeed TTS: %d chars, voice=%s, speed=%.2f, pitch=%d, emotion=%s",
            len(text), voice, speed, pitch, emotion,
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voice_id": voice,
            "speed": max(0.50, min(2.00, speed)),
            "pitch": max(-12, min(12, pitch)),
            "emotion": emotion,
            "volume": max(0.10, min(10.00, volume)),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Submit synthesis job
            resp = await client.post(self.SUBMIT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            # Check if result is returned synchronously
            if isinstance(result, bytes):
                return result

            # Async mode: poll for result
            request_id = result.get("id") or result.get("requestId") or result.get("request_id")
            if not request_id:
                # Response might contain the audio URL directly
                output = result.get("output") or result.get("result")
                if isinstance(output, str) and output.startswith("http"):
                    audio_resp = await client.get(output, headers=headers)
                    audio_resp.raise_for_status()
                    return audio_resp.content
                raise RuntimeError(f"Unexpected WaveSpeed response: {result}")

            # Poll for completion
            result_url = self.RESULT_URL.format(request_id=request_id)
            import asyncio

            elapsed = 0.0
            while elapsed < self._timeout:
                await asyncio.sleep(self._poll_interval)
                elapsed += self._poll_interval

                poll_resp = await client.get(result_url, headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("status", "")
                if status in ("completed", "succeeded"):
                    # Download the audio
                    output = poll_data.get("output") or poll_data.get("result")
                    if isinstance(output, dict):
                        audio_url = output.get("audio") or output.get("url")
                    elif isinstance(output, str):
                        audio_url = output
                    else:
                        raise RuntimeError(f"Unexpected output format: {poll_data}")

                    audio_resp = await client.get(audio_url)
                    audio_resp.raise_for_status()
                    audio_data = audio_resp.content
                    logger.info("WaveSpeed TTS produced %d bytes", len(audio_data))
                    return audio_data

                if status in ("failed", "error", "canceled"):
                    error_msg = poll_data.get("error") or poll_data.get("message") or str(poll_data)
                    raise RuntimeError(f"WaveSpeed TTS failed: {error_msg}")

                logger.debug("WaveSpeed TTS polling... status=%s elapsed=%.1fs", status, elapsed)

            raise TimeoutError(f"WaveSpeed TTS timed out after {self._timeout}s")

    async def synthesize_with_profile(
        self,
        text: str,
        voice_profile: dict,
        language: str = "zh",
    ) -> bytes:
        """Synthesize using a VoiceProfile dict (convenience wrapper)."""
        return await self.synthesize(
            text=text,
            voice=voice_profile.get("voice_id", ""),
            language=language,
            speed=voice_profile.get("speed", 1.0),
            pitch=voice_profile.get("pitch", 0),
            emotion=voice_profile.get("emotion", "neutral"),
            volume=voice_profile.get("volume", 1.0),
        )


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
