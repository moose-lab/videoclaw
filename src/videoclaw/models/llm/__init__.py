"""LLM client wrappers for text generation tasks."""

__all__ = [
    "LLMClient",
    "TokenUsage",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading litellm at package import time."""
    _import_map = {
        "LLMClient": "videoclaw.models.llm.litellm_wrapper",
        "TokenUsage": "videoclaw.models.llm.litellm_wrapper",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
