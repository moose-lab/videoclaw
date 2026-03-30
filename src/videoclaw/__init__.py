"""VideoClaw — The Agent OS for AI Video Generation.

Key subpackages
---------------
core : State management, DAG planning, execution, and event bus.
drama : AI short-drama orchestration (multi-episode series).
generation : Image, video, script, storyboard, and audio generation.
models : Model protocol, registry, routing, and cloud adapters.
flow : YAML-based visual pipeline definitions (ClawFlow).
storage : Storage backends for generated assets.
publishers : Platform publishing (YouTube, Bilibili).
cost : Cost tracking and budget management.
agents : Autonomous video production agents.
server : REST API server.
cli : Command-line interface.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "agents",
    # Subpackages (importable via `from videoclaw import <pkg>`)
    "core",
    "cost",
    "drama",
    "flow",
    "generation",
    "models",
    "publishers",
    "server",
    "storage",
]
