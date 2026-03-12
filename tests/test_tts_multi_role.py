import pytest
from unittest.mock import AsyncMock
from pathlib import Path
from videoclaw.drama.models import DialogueLine, LineType, VoiceProfile, AudioSegment, AudioType
from videoclaw.generation.audio.tts import TTSManager


@pytest.mark.asyncio
async def test_generate_multi_role_basic(tmp_path):
    mock_provider = AsyncMock()
    mock_provider.synthesize.return_value = b"\xff" * 100
    mgr = TTSManager(provider=mock_provider)
    lines = [
        DialogueLine(text="黑夜降临", speaker="narrator", line_type=LineType.NARRATION, scene_id="s01"),
        DialogueLine(text="你来了", speaker="林薇", line_type=LineType.DIALOGUE, scene_id="s02"),
    ]
    voice_map = {
        "narrator": VoiceProfile(voice_id="Calm_Woman", role_name="narrator", line_type=LineType.NARRATION),
        "林薇": VoiceProfile(voice_id="Lively_Girl", role_name="林薇", speed=1.05, pitch=2, emotion="happy"),
    }
    segments = await mgr.generate_multi_role(lines, voice_map, tmp_path)
    assert len(segments) == 2
    assert segments[0].audio_type == AudioType.NARRATION
    assert segments[1].audio_type == AudioType.DIALOGUE
    assert mock_provider.synthesize.call_count == 2


@pytest.mark.asyncio
async def test_generate_multi_role_skips_empty(tmp_path):
    mock_provider = AsyncMock()
    mock_provider.synthesize.return_value = b"\xff" * 100
    mgr = TTSManager(provider=mock_provider)
    lines = [
        DialogueLine(text="", speaker="narrator", line_type=LineType.NARRATION, scene_id="s01"),
        DialogueLine(text="有话说", speaker="林薇", line_type=LineType.DIALOGUE, scene_id="s02"),
    ]
    voice_map = {"narrator": VoiceProfile(voice_id="V1", role_name="narrator"), "林薇": VoiceProfile(voice_id="V2", role_name="林薇")}
    segments = await mgr.generate_multi_role(lines, voice_map, tmp_path)
    assert len(segments) == 1
    assert mock_provider.synthesize.call_count == 1


@pytest.mark.asyncio
async def test_generate_multi_role_inner_monologue(tmp_path):
    mock_provider = AsyncMock()
    mock_provider.synthesize.return_value = b"\xff" * 100
    mgr = TTSManager(provider=mock_provider)
    lines = [DialogueLine(text="他不可能知道", speaker="萧衍", line_type=LineType.INNER_MONOLOGUE, scene_id="s01")]
    voice_map = {"萧衍": VoiceProfile(voice_id="Determined_Man", role_name="萧衍")}
    segments = await mgr.generate_multi_role(lines, voice_map, tmp_path)
    assert segments[0].audio_type == AudioType.INNER_MONOLOGUE
    assert segments[0].line_type == LineType.INNER_MONOLOGUE
