"""Tests for model registry."""

import pytest

from videoclaw.models.registry import ModelRegistry
from videoclaw.models.adapters.mock import MockVideoAdapter


def test_register_and_get():
    registry = ModelRegistry()
    adapter = MockVideoAdapter()
    registry.register(adapter)
    assert registry.get("mock") is adapter
    assert any(m["model_id"] == "mock" for m in registry.list_models())


def test_get_unknown_raises():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")
