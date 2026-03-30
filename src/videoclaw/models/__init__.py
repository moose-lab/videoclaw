"""Model management — protocol, registry, routing, and adapters.

Submodules
----------
protocol : Shared types (ModelCapability, GenerationRequest/Result, VideoModelAdapter).
registry : ModelRegistry for discovering and managing available models.
router : ModelRouter for intelligent model selection and routing.
adapters : Cloud video model adapters (Seedance, Kling, MiniMax, etc.).
llm : LLM client wrappers for text generation.
"""

__all__ = [
    "ExecutionMode",
    "GenerationRequest",
    "GenerationResult",
    # Protocol & types
    "ModelCapability",
    # Registry
    "ModelRegistry",
    # Router
    "ModelRouter",
    "ProgressEvent",
    "RoutingStrategy",
    "VideoModelAdapter",
    "get_registry",
]


def __getattr__(name: str):
    """Lazy imports to avoid heavy dependency loading at package import time."""
    _import_map = {
        "ModelCapability": "videoclaw.models.protocol",
        "ExecutionMode": "videoclaw.models.protocol",
        "GenerationRequest": "videoclaw.models.protocol",
        "GenerationResult": "videoclaw.models.protocol",
        "ProgressEvent": "videoclaw.models.protocol",
        "VideoModelAdapter": "videoclaw.models.protocol",
        "ModelRegistry": "videoclaw.models.registry",
        "get_registry": "videoclaw.models.registry",
        "ModelRouter": "videoclaw.models.router",
        "RoutingStrategy": "videoclaw.models.router",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
