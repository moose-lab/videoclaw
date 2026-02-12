"""Tests for the Director agent."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from videoclaw.core.director import Director, DIRECTOR_SYSTEM_PROMPT
from videoclaw.core.state import ProjectState, ShotStatus


MOCK_PLAN_RESPONSE = json.dumps({
    "title": "Test Video",
    "description": "A test video about cats",
    "scenes": [
        {
            "scene_id": "scene_01",
            "description": "A cat sitting on a windowsill, golden hour light",
            "duration": 5.0,
            "visual_style": "cinematic",
            "camera_movement": "dolly_in",
        },
        {
            "scene_id": "scene_02",
            "description": "Cat jumping gracefully, slow motion",
            "duration": 4.0,
            "visual_style": "cinematic",
            "camera_movement": "tracking",
        },
    ],
    "voice_over": {
        "text": "Cats are nature's most graceful creatures.",
        "tone": "warm",
        "language": "en",
    },
    "music": {
        "style": "acoustic",
        "tempo": 90,
        "mood": "serene",
    },
})


@pytest.mark.asyncio
async def test_plan_from_prompt():
    """Director.plan() with a string prompt should call LLM and return ProjectState."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=MOCK_PLAN_RESPONSE)

    director = Director(llm=mock_llm)
    state = await director.plan("A video about cats", duration=10.0)

    assert isinstance(state, ProjectState)
    assert state.prompt == "A video about cats"
    assert len(state.storyboard) == 2
    assert state.storyboard[0].shot_id == "scene_01"
    assert state.storyboard[0].duration_seconds == 5.0
    assert state.storyboard[0].status == ShotStatus.PENDING
    assert state.script == "Cats are nature's most graceful creatures."
    assert state.metadata["music"]["mood"] == "serene"

    # Verify LLM was called.
    mock_llm.chat.assert_called_once()
    call_args = mock_llm.chat.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "cats" in messages[1]["content"].lower()


@pytest.mark.asyncio
async def test_plan_from_project_state():
    """Director.plan() with a ProjectState should update it in-place."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=MOCK_PLAN_RESPONSE)

    existing = ProjectState(prompt="existing project about cats")

    director = Director(llm=mock_llm)
    result = await director.plan(existing, duration=10.0)

    assert result is existing  # same object, updated in-place
    assert len(result.storyboard) == 2
    assert result.script is not None


@pytest.mark.asyncio
async def test_plan_with_preferred_model():
    """preferred_model should propagate to all shots."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=MOCK_PLAN_RESPONSE)

    director = Director(llm=mock_llm)
    state = await director.plan("cats", preferred_model="sora")

    for shot in state.storyboard:
        assert shot.model_id == "sora"


@pytest.mark.asyncio
async def test_plan_handles_markdown_fenced_json():
    """Director should strip markdown fences from LLM response."""
    fenced = "```json\n" + MOCK_PLAN_RESPONSE + "\n```"
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=fenced)

    director = Director(llm=mock_llm)
    state = await director.plan("cats")

    assert len(state.storyboard) == 2


@pytest.mark.asyncio
async def test_plan_invalid_json_raises():
    """Director should raise ValueError on invalid JSON from LLM."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value="This is not JSON at all")

    director = Director(llm=mock_llm)
    with pytest.raises(ValueError, match="invalid JSON"):
        await director.plan("cats")


@pytest.mark.asyncio
async def test_refine_prompt():
    """Director.refine_prompt() should call LLM and return improved text."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value="  An improved prompt about cats in sunlight  ")

    director = Director(llm=mock_llm)
    result = await director.refine_prompt(
        original_prompt="A cat video",
        feedback="Make it warmer and more golden",
    )

    assert result == "An improved prompt about cats in sunlight"
    mock_llm.chat.assert_called_once()
