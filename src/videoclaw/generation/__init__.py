"""Content generation — images, video, scripts, storyboards, and composition.

Submodules
----------
base_image : Abstract base class for all image generators.
evolink_image : Evolink Seedream 5.0 image generation.
byteplus_image : BytePlus image generation and editing.
gemini_image : Gemini-based image generation.
video : Video clip generation from prompts/images.
script : Script parsing and generation.
storyboard : Storyboard planning from scripts.
compose : Audio/video composition and assembly.
render : Final rendering profiles and export.
subtitle : SRT/ASS subtitle generation.
audio : TTS, voice casting, and audio post-processing (subpackage).
"""

__all__ = [
    # Base
    "BaseImageGenerator",
    "BytePlusImageEditor",
    "BytePlusImageGenerator",
    # Image generators
    "EvolinkImageGenerator",
    "GeminiImageGenerator",
    "RenderProfile",
    # Script
    "Script",
    "ScriptGenerator",
    "ScriptSection",
    # Storyboard
    "StoryboardGenerator",
    # Subtitles
    "SubtitleGenerator",
    # Composition & rendering
    "VideoComposer",
    # Video
    "VideoGenerator",
    "VideoRenderer",
    "generate_srt",
]


def __getattr__(name: str):
    """Lazy imports to avoid heavy dependency loading at package import time."""
    _import_map = {
        "BaseImageGenerator": "videoclaw.generation.base_image",
        "EvolinkImageGenerator": "videoclaw.generation.evolink_image",
        "BytePlusImageGenerator": "videoclaw.generation.byteplus_image",
        "BytePlusImageEditor": "videoclaw.generation.byteplus_image",
        "GeminiImageGenerator": "videoclaw.generation.gemini_image",
        "VideoGenerator": "videoclaw.generation.video",
        "Script": "videoclaw.generation.script",
        "ScriptSection": "videoclaw.generation.script",
        "ScriptGenerator": "videoclaw.generation.script",
        "StoryboardGenerator": "videoclaw.generation.storyboard",
        "VideoComposer": "videoclaw.generation.compose",
        "RenderProfile": "videoclaw.generation.render",
        "VideoRenderer": "videoclaw.generation.render",
        "SubtitleGenerator": "videoclaw.generation.subtitle",
        "generate_srt": "videoclaw.generation.subtitle",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
