"""Generate character reference images for 这个王妃太狂野.

Uses Seedream 5.0 via the existing EvolinkImageGenerator.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ensure the Evolink API key is available
os.environ.setdefault(
    "VIDEOCLAW_EVOLINK_API_KEY",
    "sk-dE0PoRCh6RqChi7xbh8UfwSiArKfHx3IyYdjyMippo9M4jYG",
)

from videoclaw.generation.image import EvolinkImageGenerator

CHARACTERS = {
    "林薇": "Full body portrait of a young Chinese woman, early 20s, delicate oval face, phoenix eyes with sharp gaze, long black hair with jade hairpin, wearing tattered but elegant white hanfu with red inner lining, slender build, subtle scar on left wrist. Standing in ancient Chinese courtyard, confident posture despite worn clothing. Cinematic lighting, 9:16 vertical composition, photorealistic, high detail, ancient China setting.",
    "萧衍": "Full body portrait of a tall Chinese man, late 20s, sharp angular jawline, intense dark eyes, long black hair in high ponytail with silver crown, wearing black brocade robe with dragon embroidery, broad shoulders, commanding presence, a sheathed sword at waist that is never drawn. Standing in dark palace corridor, imposing stance. Cinematic lighting, 9:16 vertical composition, photorealistic, high detail, ancient China setting.",
    "慕容雪": "Full body portrait of a beautiful Chinese woman, mid 20s, round face with dimples, gentle almond eyes hiding cunning intent, elaborate updo with golden phoenix hairpins, wearing pink silk hanfu with floral patterns, holding folding fan with mandala flower painting. Standing in luxurious bedroom, graceful but calculating pose. Cinematic lighting, 9:16 vertical composition, photorealistic, high detail, ancient China setting.",
    "老太君": "Full body portrait of an elderly Chinese woman, 70s, silver hair in neat bun, wise piercing eyes, dignified posture, wearing deep purple silk robe with gold trim, leaning on an ornate dragon-headed walking cane, jade bracelet on wrist. Standing in grand hall, authoritative presence. Cinematic lighting, 9:16 vertical composition, photorealistic, high detail, ancient China setting.",
}


async def main():
    output_dir = Path("docs/deliverables/这个王妃太狂野/images/characters")
    generator = EvolinkImageGenerator()

    for name, prompt in CHARACTERS.items():
        output_path = output_dir / f"{name}.png"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", name)
            continue
        try:
            path = await generator.generate(
                prompt,
                output_dir=output_dir,
                filename=f"{name}.png",
                size="3:4",  # vertical portrait
            )
            logger.info("Generated: %s (%.1f KB)", path, path.stat().st_size / 1024)
        except Exception as e:
            logger.error("Failed for %s: %s", name, e)

    logger.info("Done! Character images saved to %s", output_dir)


if __name__ == "__main__":
    asyncio.run(main())
