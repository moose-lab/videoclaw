# Seedance 2.0 Prompt Engineering Reference

## Five-Part Prompt Anatomy (Director-Style)

```
Subject + Action + Camera + Style + Constraints
```

- Keep under 200 words (sweet spot: 60-120 for simple, 120-200 for complex)
- Actions in present tense ("walks", not "walked")
- ONE motion verb per shot — never stack multiple camera movements

## Camera Movement Vocabulary

| Category | Verbs |
|----------|-------|
| Dolly/Track | dolly in, push in, dolly out, pull back, truck left/right, tracking shot |
| Handheld | handheld following, slight sway, micro-shake, phone perspective |
| Gimbal | gimbal-smooth, steadicam glide |
| Pan/Tilt | pan left/right, tilt up/down (avoid whip pan) |
| Crane | crane up/down, crane up and over |
| Special | 360-degree orbit, dolly zoom/vertigo, rack focus, POV switch |

### Shot Size Pairing
- Wide: slow dolly, locked/static, slow crane
- Medium: handheld (personal), gimbal (polished)
- Close-up: tiny push-ins, rack focus — avoid pans

### Lens Buckets (not mm)
- Wide: 24-28mm feel
- Normal: 35-50mm feel
- Telephoto: 85mm+ feel (compression, shallow DoF)

## Character Consistency Rules
1. Image-to-Video > Text-to-Video
2. Repeat critical traits in EVERY prompt ("same female, brown hair, white jacket")
3. Maintain stable environmental conditions across shots
4. Use gradual transitions — sudden changes cause identity drift
5. Generate longer clips when possible (avoid AI reset)
6. Keep creativity/consistency sliders identical across clips

## Realism Enhancement
- Physics-aware: "tires smoke, weight shifting" instead of "car turns"
- Sound descriptors: "metallic clink", "muffled reverb" → triggers native audio
- Style mods: film grain, natural lighting, shallow DoF, muted palette, light motion blur
- Explicit: "anatomically correct hands"

## Constraints (3-5 max)
- Visual noise: text overlays, watermarks, lens flares
- Identity drift: extra characters, crowd, mirrors
- Camera chaos: snap zooms, whip pans, jump cuts
- Body artifacts: extra fingers, warped edges
- Color: neon, cartoon saturation

## Reference Priority
@Audio (locks rhythm) > @Video (locks movement) > @Image (locks appearance)

## Scene Parameter Config
| Param | Options | Drama Recommendation |
|-------|---------|---------------------|
| Duration | 4-15s | 5s per shot (TikTok pacing) |
| Resolution | 720p/1080p | 720p draft, 1080p final |
| Ratio | 9:16, 16:9, 1:1 | 9:16 for TikTok |
| generate_audio | true/false | true (native audio) |

## Material Budget
- Max: 9 images + 3 videos (15s) + 3 audio
- Recommended: 3-5 images, 1-2 videos, 1 audio
- "Less is more" — overload confuses the model
