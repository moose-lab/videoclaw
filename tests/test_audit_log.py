"""Tests for AuditLog persistence and experience feedback."""
import json
import pytest
from pathlib import Path

from videoclaw.drama.vision_auditor import AuditLog, EpisodeAuditReport, ShotAuditResult


class TestAuditLog:
    def test_append_and_read(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        report = EpisodeAuditReport(
            series_id="test", episode_number=1, total_shots=2, passed_shots=1,
            shot_results=[
                ShotAuditResult(shot_id="s01", passed=True, fatals=[], tolerables=["minor_flicker"]),
                ShotAuditResult(shot_id="s02", passed=False, regen_required=True, fatals=["hand_anatomy"], tolerables=[]),
            ],
            regen_required=["s02"],
        )
        log.append_round(report, round_num=1)

        rounds = log.read_all()
        assert len(rounds) == 1
        assert rounds[0]["round"] == 1
        assert rounds[0]["summary"]["passed"] == 1
        assert rounds[0]["summary"]["total"] == 2

    def test_multiple_rounds_appended(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=1,
                shot_results=[ShotAuditResult(shot_id="s01", passed=True)],
            )
            log.append_round(report, round_num=i + 1)
        assert len(log.read_all()) == 3

    def test_get_frequent_defects(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        # 3 rounds, each with "hand_anatomy" fatal
        for i in range(3):
            report = EpisodeAuditReport(
                series_id="test", episode_number=1, total_shots=1, passed_shots=0,
                shot_results=[
                    ShotAuditResult(shot_id="s01", passed=False, fatals=["hand_anatomy"], tolerables=["minor_flicker"]),
                ],
            )
            log.append_round(report, round_num=i + 1)

        frequent = log.get_frequent_defects(min_count=3)
        assert "hand_anatomy" in frequent
        assert "minor_flicker" in frequent

    def test_get_frequent_defects_filters_below_threshold(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        report = EpisodeAuditReport(
            series_id="test", episode_number=1, total_shots=1, passed_shots=0,
            shot_results=[ShotAuditResult(shot_id="s01", passed=False, fatals=["rare_defect"], tolerables=[])],
        )
        log.append_round(report, round_num=1)

        frequent = log.get_frequent_defects(min_count=3)
        assert frequent == []

    def test_empty_log_returns_empty(self, tmp_path):
        log = AuditLog(tmp_path / "nonexistent.jsonl")
        assert log.read_all() == []
        assert log.get_frequent_defects() == []

    def test_save_to_log_creates_directory(self, tmp_path):
        """EpisodeAuditReport.save_to_log creates audit_logs dir if missing."""
        report = EpisodeAuditReport(
            series_id="test", episode_number=1, total_shots=1, passed_shots=1,
            shot_results=[ShotAuditResult(shot_id="s01", passed=True)],
        )
        series_dir = tmp_path / "dramas" / "test_series"
        report.save_to_log(series_dir, round_num=1)

        log_path = series_dir / "audit_logs" / "ep01_audit.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["round"] == 1
        assert data["episode"] == 1

    def test_round_record_has_timestamp(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        report = EpisodeAuditReport(
            series_id="test", episode_number=1, total_shots=1, passed_shots=1,
            shot_results=[ShotAuditResult(shot_id="s01", passed=True)],
        )
        log.append_round(report, round_num=1)
        rounds = log.read_all()
        assert "timestamp" in rounds[0]
        # Should be a valid ISO format timestamp
        from datetime import datetime
        datetime.fromisoformat(rounds[0]["timestamp"])

    def test_round_record_has_regen_ids(self, tmp_path):
        log = AuditLog(tmp_path / "test.jsonl")
        report = EpisodeAuditReport(
            series_id="test", episode_number=1, total_shots=2, passed_shots=1,
            shot_results=[
                ShotAuditResult(shot_id="s01", passed=True),
                ShotAuditResult(shot_id="s02", passed=False, regen_required=True, fatals=["bad"]),
            ],
            regen_required=["s02"],
        )
        log.append_round(report, round_num=1)
        rounds = log.read_all()
        assert rounds[0]["summary"]["regen_ids"] == ["s02"]
