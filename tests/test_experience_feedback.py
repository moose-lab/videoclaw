"""Tests for experience feedback — audit defects → prompt constraints."""
import pytest
from pathlib import Path

from videoclaw.drama.prompt_enhancer import PromptEnhancer
from videoclaw.drama.vision_auditor import AuditLog, EpisodeAuditReport, ShotAuditResult


class TestLoadAuditConstraints:
    def test_hand_defect_becomes_constraint(self, tmp_path):
        log = AuditLog(tmp_path / "audit_logs" / "ep01_audit.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["hand_anatomy_error"], tolerables=[])],
            )
            log.append_round(report, round_num=i + 1)

        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert any("finger" in c.lower() or "hand" in c.lower() for c in constraints)

    def test_temporal_break_becomes_constraint(self, tmp_path):
        log = AuditLog(tmp_path / "audit_logs" / "ep01_audit.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["temporal_break_f3_f4"], tolerables=[])],
            )
            log.append_round(report, round_num=i + 1)

        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert any("stability" in c.lower() or "visual" in c.lower() for c in constraints)

    def test_character_missing_becomes_constraint(self, tmp_path):
        log = AuditLog(tmp_path / "audit_logs" / "ep01_audit.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["character missing from frame"], tolerables=[])],
            )
            log.append_round(report, round_num=i + 1)

        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert any("character" in c.lower() and "visible" in c.lower() for c in constraints)

    def test_scene_mismatch_becomes_constraint(self, tmp_path):
        log = AuditLog(tmp_path / "audit_logs" / "ep01_audit.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["scene mismatch - wrong location"], tolerables=[])],
            )
            log.append_round(report, round_num=i + 1)

        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert any("scene" in c.lower() and "match" in c.lower() for c in constraints)

    def test_unknown_defect_becomes_avoid(self, tmp_path):
        log = AuditLog(tmp_path / "audit_logs" / "ep01_audit.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["weird_artifact_xyz"], tolerables=[])],
            )
            log.append_round(report, round_num=i + 1)

        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert any("AVOID" in c and "weird_artifact_xyz" in c for c in constraints)

    def test_no_log_returns_empty(self, tmp_path):
        enhancer = PromptEnhancer()
        constraints = enhancer.load_audit_constraints(tmp_path, min_count=3)
        assert constraints == []

    def test_inject_learned_constraints(self):
        enhancer = PromptEnhancer()
        enhancer.inject_learned_constraints(["No extra fingers", "Stable lighting"])
        assert enhancer._learned_constraints == ["No extra fingers", "Stable lighting"]

    def test_learned_constraints_appear_in_enhanced_prompt(self):
        """Learned constraints should appear in the final enhanced prompt."""
        from videoclaw.drama.models import DramaSeries, Episode, DramaScene, Character, ShotScale

        series = DramaSeries(
            series_id="test",
            title="Test",
            style="cinematic",
            aspect_ratio="9:16",
            language="en",
            model_id="seedance-2.0",
            characters=[
                Character(name="Alice", description="A 25-year-old woman", visual_prompt="blonde hair, blue eyes"),
            ],
            episodes=[],
        )
        scene = DramaScene(
            scene_id="s01",
            description="Alice walks into the room",
            visual_prompt="Alice enters a modern office",
            dialogue="Hello there",
            characters_present=["Alice"],
            shot_scale=ShotScale.MEDIUM,
        )

        enhancer = PromptEnhancer()
        enhancer.inject_learned_constraints(["No extra fingers", "Stable lighting"])
        prompt = enhancer.enhance_scene_prompt(scene, series)
        assert "No extra fingers" in prompt
        assert "Stable lighting" in prompt
