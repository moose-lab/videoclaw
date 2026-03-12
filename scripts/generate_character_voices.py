"""Generate character voice samples for 这个王妃太狂野.

Uses EdgeTTS (free, no API key) to produce voice samples for each character.
Each character gets a representative line spoken in their assigned voice.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from videoclaw.generation.audio.tts import EdgeTTSProvider

# Character → Edge TTS voice mapping + sample dialogue
CHARACTER_VOICES = {
    "林薇": {
        "voice": "zh-CN-XiaoxiaoNeural",      # Young woman, warm yet capable
        "sample_text": "你确定？在这王府里，没人能动我。我不是从前的废物王妃了。",
    },
    "萧衍": {
        "voice": "zh-CN-YunjianNeural",         # Deep male, authoritative
        "sample_text": "本王的事，不劳你费心。退下。",
    },
    "慕容雪": {
        "voice": "zh-CN-XiaoyiNeural",          # Sweet female, playful/cunning
        "sample_text": "姐姐说笑了，妹妹怎敢造次呢？不过是替王爷分忧罢了。",
    },
    "老太君": {
        "voice": "zh-CN-XiaochenNeural",        # Mature female, dramatic/commanding
        "sample_text": "放肆！萧家的规矩，还轮不到你来说！给我把人带下去！",
    },
}


async def main():
    output_dir = Path("docs/deliverables/这个王妃太狂野/audio/voice_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = EdgeTTSProvider()

    for name, config in CHARACTER_VOICES.items():
        output_path = output_dir / f"{name}_voice_sample.mp3"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", name)
            continue

        try:
            audio_data = await provider.synthesize(
                text=config["sample_text"],
                voice=config["voice"],
                language="zh",
            )
            output_path.write_bytes(audio_data)
            logger.info(
                "Generated: %s — voice=%s (%.1f KB)",
                output_path.name, config["voice"], len(audio_data) / 1024,
            )
        except Exception as e:
            logger.error("Failed for %s: %s", name, e)

    # Write voice mapping metadata
    metadata = {}
    for name, config in CHARACTER_VOICES.items():
        metadata[name] = {
            "edge_tts_voice": config["voice"],
            "sample_text": config["sample_text"],
            "sample_audio": f"audio/voice_samples/{name}_voice_sample.mp3",
        }

    meta_path = output_dir / "voice_mapping.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Voice mapping saved: %s", meta_path)

    logger.info("Done! Voice samples saved to %s", output_dir)


if __name__ == "__main__":
    asyncio.run(main())
