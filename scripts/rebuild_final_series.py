"""Rebuild final_series.json with all episodes' scenes from individual script files.

Also generates per-episode executor data files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videoclaw.drama.models import DramaSeries, DramaScene, Episode, assign_voice_profile

BASE = Path("docs/deliverables/这个王妃太狂野")

# Load series outline
series_data = json.loads((BASE / "00_series_outline.json").read_text(encoding="utf-8"))
series = DramaSeries.from_dict(series_data)

# Assign voice profiles
for c in series.characters:
    assign_voice_profile(c)

# Load episode 1 mock data from e2e test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_drama_e2e import MOCK_EPISODE_SCRIPTS
ep1_script = MOCK_EPISODE_SCRIPTS[1]

# Merge all episode scripts into series
for ep in series.episodes:
    if ep.number == 1:
        script_data = ep1_script
    else:
        ep_path = BASE / f"ep{ep.number:02d}_{ep.title}_script.json"
        if not ep_path.exists():
            print(f"WARNING: Missing script for episode {ep.number}")
            continue
        script_data = json.loads(ep_path.read_text(encoding="utf-8"))

    scenes = script_data.get("scenes", [])
    ep.scenes = [DramaScene.from_dict(s) for s in scenes]
    ep.script = json.dumps(script_data, ensure_ascii=False)

    print(f"  Episode {ep.number}: {ep.title} — {len(ep.scenes)} scenes loaded")

# Save complete final_series.json
final_path = BASE / "final_series.json"
final_path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {final_path} ({final_path.stat().st_size / 1024:.1f} KB)")

# Generate per-episode executor data
executor_dir = BASE / "executor"
executor_dir.mkdir(parents=True, exist_ok=True)

for ep in series.episodes:
    ep_data = {
        "episode_number": ep.number,
        "episode_title": ep.title,
        "duration_seconds": ep.duration_seconds,
        "scenes": [s.to_dict() for s in ep.scenes],
        "characters": [c.to_dict() for c in series.characters],
        "series_metadata": {
            "title": series.title,
            "genre": series.genre,
            "style": series.style,
            "language": series.language,
            "aspect_ratio": series.aspect_ratio,
        },
    }

    # Parse script for music/voice_over/cliffhanger
    if ep.script:
        script_parsed = json.loads(ep.script)
        ep_data["music"] = script_parsed.get("music", {})
        ep_data["voice_over"] = script_parsed.get("voice_over", {})
        ep_data["cliffhanger"] = script_parsed.get("cliffhanger", "")

    ep_path = executor_dir / f"ep{ep.number:02d}_executor_data.json"
    ep_path.write_text(json.dumps(ep_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Executor data: {ep_path.name} ({len(ep.scenes)} scenes)")

# Generate characters reference data
chars_data = []
for c in series.characters:
    chars_data.append({
        "name": c.name,
        "description": c.description,
        "visual_prompt": c.visual_prompt,
        "voice_style": c.voice_style,
        "voice_profile": c.voice_profile.to_dict() if c.voice_profile else None,
        "reference_image": f"images/characters/{c.name}.png",
    })

chars_path = executor_dir / "characters.json"
chars_path.write_text(json.dumps(chars_data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  Characters: {chars_path.name} ({len(chars_data)} characters)")

print("\nDone!")
