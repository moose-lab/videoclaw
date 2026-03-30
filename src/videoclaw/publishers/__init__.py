"""Platform publishers — upload finished videos to distribution platforms."""

from videoclaw.publishers.base import Publisher, PublishRequest, PublishResult, PublishStatus

__all__ = [
    # Platform implementations
    "BilibiliPublisher",
    "PublishRequest",
    "PublishResult",
    "PublishStatus",
    # Base types
    "Publisher",
    "YouTubePublisher",
]


def __getattr__(name: str):
    """Lazy imports for platform publishers with heavy SDK dependencies."""
    _import_map = {
        "BilibiliPublisher": "videoclaw.publishers.bilibili",
        "YouTubePublisher": "videoclaw.publishers.youtube",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
