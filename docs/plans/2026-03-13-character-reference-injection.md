# Character Reference Image Injection (Task 2.2)

## Problem

Character consistency is the biggest technical bottleneck in Chinese AI short dramas. `CharacterDesigner` generates PNG reference images for each character (stored in `Character.reference_image`), but these were discarded during the `DramaScene → Shot` conversion in `build_episode_dag()`. All shots were generated as pure TEXT_TO_VIDEO with no facial/character consistency.

## Data Flow (After Fix)

```
Character.reference_image ──┐
                             ├──► build_episode_dag() ──► Shot.reference_images
DramaScene.characters_present┘                                    │
                                                                  ▼
                                                    TaskNode.params["reference_images"]
                                                    TaskNode.params["speaking_character"]
                                                                  │
                                                                  ▼
                                                    _handle_video_gen()
                                                    reads files → bytes
                                                                  │
                                                                  ▼
                                                    VideoGenerator.generate_shot(
                                                        reference_image=primary_bytes,
                                                        extra_references={...}
                                                    )
                                                                  │
                                                                  ▼
                                                    GenerationRequest(
                                                        reference_image=bytes,
                                                        extra={"additional_references": ...}
                                                    )
                                                                  │
                                                                  ▼
                                                    Router auto-infers IMAGE_TO_VIDEO
                                                                  │
                                                                  ▼
                                                    Adapter encodes per protocol:
                                                    ZhipuAI: base64 image_url
                                                    MiniMax: first_frame_image / subject_reference
                                                    Kling: image2video endpoint
```

## Default Production Model Suite

| Stage | Model | Rationale |
|-------|-------|-----------|
| Character image gen | Evolink Seedream 5.0 | Already integrated, high quality for Chinese-style characters |
| Video gen (default) | MiniMax Hailuo 2.3 | Best quality/cost ratio, supports first_frame_image |
| Video gen (close-up) | MiniMax S2V-01 | Only model with subject_reference, strongest face consistency |
| Video gen (free alt) | ZhipuAI CogVideoX-Flash | Free tier, supports base64 image_url |
| TTS (production) | WaveSpeed MiniMax speech-02-hd | Multi-character, emotion control |
| TTS (dev/test) | EdgeTTS | Free, no API key needed |
| LLM | GPT-4o | Best screenplay quality |

## Multi-Character Scene Handling

When multiple characters appear in a scene:

1. **Primary character**: The `speaking_character` is used as the primary reference image (passed as `reference_image` in `GenerationRequest`).
2. **Fallback**: If `speaking_character` has no reference image, the first character with a reference image is used as primary.
3. **Extra references**: Additional characters are passed in `extra["additional_references"]` for adapters that support multi-subject references (e.g., MiniMax S2V-01).

## Adapter Reference Image Encoding

| Adapter | Encoding | Field |
|---------|----------|-------|
| ZhipuAI | bytes → base64 → data URI | `image_url` |
| MiniMax (Hailuo) | bytes → base64 → data URI | `first_frame_image` |
| MiniMax (S2V-01) | bytes → base64 → data URI | `subject_reference[].image` |
| Kling | bytes → base64 → data URI | `image` (uses image2video endpoint) |
| Mock | Ignored | N/A |

## Degradation Strategy

- If a character reference image file does not exist on disk, a warning is logged and the character is skipped.
- If no reference images are loadable, the shot falls back to TEXT_TO_VIDEO generation.
- The pipeline is never blocked by missing reference images.

## Files Modified

| File | Change |
|------|--------|
| `src/videoclaw/core/state.py` | `Shot.reference_images` field |
| `src/videoclaw/drama/models.py` | Character, DramaScene, DramaEpisode, DramaSeries models |
| `src/videoclaw/drama/runner.py` | `build_episode_dag()` + `_build_drama_dag()` |
| `src/videoclaw/core/executor.py` | `_handle_video_gen()` loads ref image files |
| `src/videoclaw/generation/video.py` | `generate_shot()` forwards reference_image |
| `src/videoclaw/models/adapters/minimax.py` | bytes→base64, S2V-01 subject_reference |
| `src/videoclaw/models/adapters/kling.py` | bytes→base64, image2video endpoint |
| `src/videoclaw/config.py` | default_video_model → minimax-hailuo-2.3 |
| `tests/test_reference_image_injection.py` | Comprehensive test suite |
