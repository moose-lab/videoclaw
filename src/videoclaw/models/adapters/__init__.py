"""Video model adapters — cloud API integrations for video generation.

Each adapter implements the BaseCloudVideoAdapter interface and handles
authentication, request submission, polling, and result retrieval for a
specific cloud video generation API.

Available adapters: Seedance (BytePlus), Kling, MiniMax, ZhiPu, OpenAI, Mock.
"""

__all__ = [
    "BaseCloudVideoAdapter",
]


def __getattr__(name: str):
    """Lazy imports — adapters pull in heavy SDK dependencies."""
    _import_map = {
        "BaseCloudVideoAdapter": "videoclaw.models.adapters.base",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
