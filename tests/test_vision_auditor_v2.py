"""Tests for the refactored 3-layer pragmatic audit pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from videoclaw.drama.models import DramaScene
from videoclaw.drama.vision_auditor import (
    ShotAuditResult,
    VisionAuditor,
)


class TestShotAuditResultGrading:
    def test_result_with_fatals_is_regen(self):
        r = ShotAuditResult(
            shot_id="s01", passed=False, regen_required=True,
            fatals=["hand_anatomy"], tolerables=[],
        )
        assert r.regen_required is True
        assert r.passed is False

    def test_result_with_two_tolerables_passes(self):
        r = ShotAuditResult(
            shot_id="s01", passed=True, regen_required=False,
            fatals=[], tolerables=["minor_flicker_f3", "minor_flicker_f7"],
        )
        assert r.passed is True
        assert r.regen_required is False

    def test_result_with_three_tolerables_is_regen(self):
        r = ShotAuditResult(
            shot_id="s01", passed=False, regen_required=True,
            fatals=[], tolerables=["a", "b", "c"],
        )
        assert r.regen_required is True

    def test_to_dict_includes_fatals_tolerables(self):
        r = ShotAuditResult(
            shot_id="s01", passed=True, fatals=["x"], tolerables=["y"],
        )
        d = r.to_dict()
        assert d["fatals"] == ["x"]
        assert d["tolerables"] == ["y"]


class TestLayer0MetadataRules:
    @pytest.mark.asyncio
    async def test_zero_duration_is_fatal(self):
        auditor = VisionAuditor()
        scene = DramaScene(scene_id="s01")
        with patch(
            "videoclaw.drama.vision_auditor.get_video_info",
            new_callable=AsyncMock,
            return_value={"format": {"duration": "0", "size": "100"}},
        ), patch.object(auditor, "_layer1_temporal", new_callable=AsyncMock, return_value=([], [])), \
           patch.object(auditor, "_layer2_vision_llm", new_callable=AsyncMock, return_value=([], [])), \
           patch("pathlib.Path.exists", return_value=True):
            result = await auditor.audit_shot(scene, Path("/fake/clip.mp4"))
        assert any("duration" in f for f in result.fatals)
        assert result.regen_required is True


class TestShortCircuit:
    @pytest.mark.asyncio
    async def test_layer1_fatal_skips_layer2(self):
        auditor = VisionAuditor()
        scene = DramaScene(scene_id="s01")
        mock_l2 = AsyncMock(return_value=([], []))
        with patch(
            "videoclaw.drama.vision_auditor.get_video_info",
            new_callable=AsyncMock,
            return_value={"format": {"duration": "5.0", "size": "100000"}},
        ), patch.object(
            auditor, "_layer1_temporal", new_callable=AsyncMock,
            return_value=(["temporal_break_f4_f5_ssim0.50"], []),
        ), patch.object(auditor, "_layer2_vision_llm", mock_l2), \
           patch("pathlib.Path.exists", return_value=True):
            result = await auditor.audit_shot(scene, Path("/fake/clip.mp4"))
        assert result.regen_required is True
        mock_l2.assert_not_called()


class TestIncrementalAudit:
    @pytest.mark.asyncio
    async def test_skips_previously_passed(self):
        auditor = VisionAuditor()
        passed_scene = DramaScene(
            scene_id="s01",
            audit_result={
                "passed": True, "regen_required": False,
                "fatals": [], "tolerables": [],
            },
        )
        failed_scene = DramaScene(
            scene_id="s02",
            audit_result={
                "passed": False, "regen_required": True,
                "fatals": ["hand_anatomy"], "tolerables": [],
            },
        )
        mock_audit = AsyncMock(return_value=ShotAuditResult(
            shot_id="s02", passed=True, fatals=[], tolerables=[],
        ))
        with patch.object(auditor, "audit_shot", mock_audit), \
             patch("videoclaw.drama.vision_auditor.resolve_clip", return_value=Path("/fake.mp4")):
            report = await auditor.audit_clip_dir(
                [passed_scene, failed_scene],
                clip_dir=Path("/clips"),
                incremental=True,
            )
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0].scene_id == "s02"
        assert report.passed_shots == 2


class TestBuildVerdict:
    def test_no_defects_passes(self):
        auditor = VisionAuditor()
        r = auditor._build_verdict("s01", [], [])
        assert r.passed is True
        assert r.regen_required is False

    def test_one_fatal_regens(self):
        auditor = VisionAuditor()
        r = auditor._build_verdict("s01", ["hand_anatomy"], [])
        assert r.passed is False
        assert r.regen_required is True

    def test_two_tolerables_passes(self):
        auditor = VisionAuditor()
        r = auditor._build_verdict("s01", [], ["a", "b"])
        assert r.passed is True
        assert r.regen_required is False

    def test_three_tolerables_regens(self):
        auditor = VisionAuditor()
        r = auditor._build_verdict("s01", [], ["a", "b", "c"])
        assert r.passed is False
        assert r.regen_required is True
