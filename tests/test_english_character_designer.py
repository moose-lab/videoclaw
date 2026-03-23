"""Tests for locale-aware character image prompt generation (B4)."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from videoclaw.drama.character_designer import CharacterDesigner, CHARACTER_IMAGE_PROMPT, CHARACTER_IMAGE_PROMPT_SINGLE
from videoclaw.drama.models import Character, DramaManager, DramaSeries


def _make_series(language: str) -> DramaSeries:
    series = DramaSeries(
        series_id="test_series",
        title="Test Drama",
        language=language,
        style="cinematic",
    )
    series.characters = [
        Character(
            name="Alice",
            visual_prompt="Tall woman with brown hair, wearing a red coat",
        )
    ]
    return series


@pytest.fixture
def mock_drama_manager(tmp_path):
    mgr = MagicMock(spec=DramaManager)
    mgr.base_dir = tmp_path
    mgr.save = MagicMock()
    return mgr


@pytest.fixture
def mock_image_generator():
    gen = MagicMock()
    gen.generate = AsyncMock(return_value=Path("/tmp/alice.png"))
    return gen


@pytest.mark.asyncio
async def test_english_series_gets_western_drama_prompt(mock_image_generator, mock_drama_manager):
    """English series generates a prompt containing 'Western drama'."""
    designer = CharacterDesigner(
        image_generator=mock_image_generator,
        drama_manager=mock_drama_manager,
    )
    series = _make_series("en")

    await designer.design_characters(series)

    assert mock_image_generator.generate.called
    call_args = mock_image_generator.generate.call_args
    prompt = call_args[0][0]  # first positional arg

    assert "Western drama" in prompt
    assert "Hollywood cinematography aesthetics" in prompt
    assert "Chinese drama" not in prompt


@pytest.mark.asyncio
async def test_chinese_series_gets_chinese_drama_prompt(mock_image_generator, mock_drama_manager):
    """Chinese series generates a prompt containing 'Chinese drama'."""
    designer = CharacterDesigner(
        image_generator=mock_image_generator,
        drama_manager=mock_drama_manager,
    )
    series = _make_series("zh")

    await designer.design_characters(series)

    assert mock_image_generator.generate.called
    call_args = mock_image_generator.generate.call_args
    prompt = call_args[0][0]

    assert "Chinese drama" in prompt
    assert "modern Asian aesthetics" in prompt
    assert "Western drama" not in prompt


@pytest.mark.asyncio
async def test_prompt_contains_appearance(mock_image_generator, mock_drama_manager):
    """The character appearance text appears in the generated prompt."""
    designer = CharacterDesigner(
        image_generator=mock_image_generator,
        drama_manager=mock_drama_manager,
    )
    series = _make_series("en")

    await designer.design_characters(series)

    prompt = mock_image_generator.generate.call_args[0][0]
    assert "brown hair" in prompt
    assert "red coat" in prompt


@pytest.mark.asyncio
async def test_character_image_prompt_uses_style_line_placeholder():
    """CHARACTER_IMAGE_PROMPT template uses {style_line} not {style}."""
    assert "{style_line}" in CHARACTER_IMAGE_PROMPT
    assert "{style_line}" in CHARACTER_IMAGE_PROMPT_SINGLE
    assert "{style} Chinese drama" not in CHARACTER_IMAGE_PROMPT


@pytest.mark.asyncio
async def test_skips_character_with_existing_image(mock_image_generator, mock_drama_manager):
    """Characters with existing reference_image are skipped."""
    designer = CharacterDesigner(
        image_generator=mock_image_generator,
        drama_manager=mock_drama_manager,
    )
    series = _make_series("en")
    series.characters[0].reference_image = "/existing/path.png"
    series.characters[0].reference_images = ["/existing/front.png", "/existing/3q.png", "/existing/full.png"]

    await designer.design_characters(series)

    mock_image_generator.generate.assert_not_called()
