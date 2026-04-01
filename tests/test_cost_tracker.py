"""Tests for cost tracker — covers tracking, budgeting, ledger persistence,
optimization hints, task-type bucketing, and the O(1) running total."""

import json
from datetime import UTC, datetime

import pytest

from videoclaw.cost.tracker import CostRecord, CostTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    task_id: str = "t1",
    model_id: str = "sora",
    execution_mode: str = "cloud",
    api_cost: float = 0.50,
    compute_cost: float = 0.0,
    duration: float = 10.0,
    task_type: str = "video_gen",
    video_seconds: float = 5.0,
    retries: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> CostRecord:
    return CostRecord(
        task_id=task_id,
        model_id=model_id,
        execution_mode=execution_mode,
        api_cost_usd=api_cost,
        compute_cost_usd=compute_cost,
        duration_seconds=duration,
        task_type=task_type,
        video_seconds=video_seconds,
        retries=retries,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Core tracking
# ---------------------------------------------------------------------------

class TestCostRecord:
    def test_total_usd(self):
        r = _make_record(api_cost=0.50, compute_cost=0.02)
        assert r.total_usd == pytest.approx(0.52)

    def test_default_timestamp_is_utc(self):
        r = CostRecord(
            task_id="t", model_id="m", execution_mode="cloud",
            api_cost_usd=0.0, compute_cost_usd=0.0, duration_seconds=0.0,
        )
        assert r.timestamp.tzinfo is not None

    def test_default_task_type(self):
        r = CostRecord(
            task_id="t", model_id="m", execution_mode="cloud",
            api_cost_usd=0.0, compute_cost_usd=0.0, duration_seconds=0.0,
        )
        assert r.task_type == "video_gen"


class TestCostTracker:
    def test_track_and_summary(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(task_id="t1", api_cost=0.50, task_type="video_gen"))
        tracker.record(_make_record(
            task_id="t2", model_id="cogvideo", execution_mode="local",
            api_cost=0.0, compute_cost=0.02, task_type="video_gen",
        ))
        summary = tracker.get_summary()
        assert summary.total_usd == pytest.approx(0.52)
        assert summary.cloud_usd == pytest.approx(0.50)
        assert summary.local_usd == pytest.approx(0.02)
        assert len(summary.records) == 2

    def test_by_model_bucketing(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(model_id="sora", api_cost=0.30))
        tracker.record(_make_record(model_id="sora", api_cost=0.20))
        tracker.record(_make_record(model_id="seedance-2.0", api_cost=0.10))
        summary = tracker.get_summary()
        assert summary.by_model["sora"] == pytest.approx(0.50)
        assert summary.by_model["seedance-2.0"] == pytest.approx(0.10)

    def test_by_task_type_bucketing(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(task_type="video_gen", api_cost=0.40))
        tracker.record(_make_record(task_type="llm", api_cost=0.05))
        tracker.record(_make_record(task_type="tts", api_cost=0.02))
        summary = tracker.get_summary()
        assert summary.by_task_type["video_gen"] == pytest.approx(0.40)
        assert summary.by_task_type["llm"] == pytest.approx(0.05)
        assert summary.by_task_type["tts"] == pytest.approx(0.02)

    def test_running_total_is_o1(self):
        """Verify the accumulator stays consistent after multiple records."""
        tracker = CostTracker(project_id="test")
        for i in range(100):
            tracker.record(_make_record(api_cost=0.01))
        assert tracker._running_total() == pytest.approx(1.0)
        assert tracker._total_usd == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_within_budget(self):
        tracker = CostTracker(project_id="test", budget_usd=5.0)
        tracker.record(_make_record(api_cost=4.50))
        within, remaining = tracker.check_budget()
        assert within is True
        assert remaining == pytest.approx(0.50)

    def test_exceeds_budget(self):
        tracker = CostTracker(project_id="test", budget_usd=4.0)
        tracker.record(_make_record(api_cost=4.50))
        within, _ = tracker.check_budget()
        assert within is False

    def test_no_budget_set(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(api_cost=100.0))
        within, remaining = tracker.check_budget()
        assert within is True
        assert remaining == float("inf")


# ---------------------------------------------------------------------------
# Persistent ledger
# ---------------------------------------------------------------------------

class TestLedgerPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        tracker = CostTracker(project_id="proj1", budget_usd=10.0)
        tracker.record(_make_record(task_id="s01", api_cost=0.25, task_type="video_gen"))
        tracker.record(_make_record(
            task_id="llm1", model_id="gpt-4o", api_cost=0.01,
            task_type="llm", input_tokens=500, output_tokens=200,
        ))

        ledger_path = tmp_path / "cost.json"
        tracker.save_ledger(ledger_path)

        assert ledger_path.exists()
        loaded = CostTracker.load_ledger(ledger_path)
        assert loaded.project_id == "proj1"
        assert loaded.budget_usd == 10.0
        assert loaded._total_usd == pytest.approx(0.26)
        assert len(loaded._records) == 2
        assert loaded._records[0].task_type == "video_gen"
        assert loaded._records[1].task_type == "llm"
        assert loaded._records[1].input_tokens == 500

    def test_ledger_json_structure(self, tmp_path):
        tracker = CostTracker(project_id="proj2")
        tracker.record(_make_record(api_cost=0.10))

        ledger_path = tmp_path / "cost.json"
        tracker.save_ledger(ledger_path)

        data = json.loads(ledger_path.read_text())
        assert data["project_id"] == "proj2"
        assert data["total_usd"] == pytest.approx(0.10)
        assert data["record_count"] == 1
        assert len(data["records"]) == 1
        assert "timestamp" in data["records"][0]

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CostTracker.load_ledger(tmp_path / "nope.json")

    def test_atomic_write(self, tmp_path):
        """Verify no .tmp file remains after save."""
        tracker = CostTracker(project_id="atomic")
        tracker.record(_make_record())

        ledger_path = tmp_path / "cost.json"
        tracker.save_ledger(ledger_path)

        assert ledger_path.exists()
        assert not ledger_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Optimization hints
# ---------------------------------------------------------------------------

class TestOptimizationHints:
    def test_switch_to_local_hint(self):
        tracker = CostTracker(project_id="test")
        # Add expensive cloud shots
        for i in range(3):
            tracker.record(_make_record(
                task_id=f"s{i:02d}", model_id="sora",
                api_cost=0.50, video_seconds=5.0,
            ))
        hints = tracker.get_optimization_hints()
        local_hints = [h for h in hints if h.action == "switch_to_local"]
        assert len(local_hints) >= 1
        assert local_hints[0].potential_savings_usd > 0

    def test_retry_waste_hint(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(api_cost=1.0, retries=5))
        hints = tracker.get_optimization_hints()
        retry_hints = [h for h in hints if h.action == "reduce_retries"]
        assert len(retry_hints) == 1

    def test_batch_hint(self):
        tracker = CostTracker(project_id="test")
        for i in range(5):
            tracker.record(_make_record(
                task_id=f"s{i:02d}", model_id="seedance-2.0",
                api_cost=0.05, video_seconds=5.0,
            ))
        hints = tracker.get_optimization_hints()
        batch_hints = [h for h in hints if h.action == "batch_shots"]
        assert len(batch_hints) >= 1

    def test_no_hints_for_cheap_project(self):
        tracker = CostTracker(project_id="test")
        tracker.record(_make_record(api_cost=0.001, video_seconds=1.0))
        hints = tracker.get_optimization_hints()
        # Should not suggest switching to local for tiny costs
        assert all(h.action != "switch_to_local" for h in hints)


# ---------------------------------------------------------------------------
# Estimate remaining
# ---------------------------------------------------------------------------

class TestEstimateRemaining:
    def test_estimate_cloud(self):
        tracker = CostTracker(project_id="test")
        estimate = tracker.estimate_remaining([
            {"model_id": "mock", "duration_seconds": 10.0},
        ])
        assert estimate == pytest.approx(0.0)

    def test_estimate_local(self):
        tracker = CostTracker(project_id="test")
        estimate = tracker.estimate_remaining([
            {"model_id": "sora", "duration_seconds": 10.0, "execution_mode": "local"},
        ])
        assert estimate == pytest.approx(0.02)  # 10 * 0.002


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class TestFormatTable:
    def test_format_table_renders(self):
        tracker = CostTracker(project_id="test", budget_usd=5.0)
        tracker.record(_make_record(api_cost=0.50))
        output = tracker.format_table()
        assert "Cost Summary" in output
        assert "TOTAL" in output
        assert "BUDGET" in output

    def test_format_table_empty(self):
        tracker = CostTracker(project_id="empty")
        output = tracker.format_table()
        assert "TOTAL" in output
        assert "$0.0000" in output


# ---------------------------------------------------------------------------
# ProjectState cost_total property
# ---------------------------------------------------------------------------

class TestProjectStateCost:
    def test_cost_total_is_computed(self):
        from videoclaw.core.state import ProjectState, Shot
        state = ProjectState()
        state.storyboard = [
            Shot(cost=0.10),
            Shot(cost=0.25),
            Shot(cost=0.05),
        ]
        assert state.cost_total == pytest.approx(0.40)

    def test_cost_total_empty_storyboard(self):
        from videoclaw.core.state import ProjectState
        state = ProjectState()
        assert state.cost_total == 0.0

    def test_serialization_roundtrip(self):
        from videoclaw.core.state import ProjectState, Shot
        state = ProjectState()
        state.storyboard = [Shot(cost=0.15)]
        d = state.to_dict()
        assert d["cost_total"] == pytest.approx(0.15)

        restored = ProjectState.from_dict(d)
        assert restored.cost_total == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_estimate_cost(self):
        from videoclaw.models.llm.litellm_wrapper import TokenUsage
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = usage.estimate_cost_usd("gpt-4o")
        # 1000/1000 * 0.0025 + 500/1000 * 0.01 = 0.0025 + 0.005 = 0.0075
        assert cost == pytest.approx(0.0075)

    def test_estimate_cost_unknown_model(self):
        from videoclaw.models.llm.litellm_wrapper import TokenUsage
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
        cost = usage.estimate_cost_usd("unknown-model-xyz")
        assert cost > 0  # uses default pricing

    def test_reset(self):
        from videoclaw.models.llm.litellm_wrapper import TokenUsage
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage.reset()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_export_csv_to_file(self, tmp_path):
        tracker = CostTracker(project_id="csv_test")
        tracker.record(_make_record(task_id="v1", task_type="video_gen", api_cost=0.50))
        tracker.record(_make_record(task_id="l1", task_type="llm", api_cost=0.01,
                                    input_tokens=500, output_tokens=200))

        csv_path = tmp_path / "cost.csv"
        result = tracker.export_csv(csv_path)
        assert result == str(csv_path)
        assert csv_path.exists()

        import csv as _csv
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["task_type"] == "video_gen"
        assert rows[1]["task_type"] == "llm"
        assert rows[1]["input_tokens"] == "500"

    def test_export_csv_to_string(self):
        tracker = CostTracker(project_id="csv_str")
        tracker.record(_make_record(task_id="s1", api_cost=0.10))
        csv_text = tracker.export_csv()
        assert "timestamp" in csv_text
        assert "s1" in csv_text
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

    def test_export_csv_empty(self):
        tracker = CostTracker(project_id="empty")
        csv_text = tracker.export_csv()
        lines = csv_text.strip().split("\n")
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_load_ledger_without_task_type(self, tmp_path):
        """Ledgers from before task_type was added should still load."""
        old_ledger = {
            "project_id": "old_proj",
            "budget_usd": None,
            "total_usd": 0.25,
            "record_count": 1,
            "records": [{
                "task_id": "t1",
                "model_id": "seedance-2.0",
                "execution_mode": "cloud",
                "api_cost_usd": 0.25,
                "compute_cost_usd": 0.0,
                "duration_seconds": 10.0,
                "task_type": "video_gen",
                "input_tokens": 0,
                "output_tokens": 0,
                "video_seconds": 5.0,
                "retries": 0,
                "timestamp": "2026-03-15T10:00:00+00:00",
            }],
        }
        ledger_path = tmp_path / "cost.json"
        ledger_path.write_text(json.dumps(old_ledger))

        tracker = CostTracker.load_ledger(ledger_path)
        assert tracker._total_usd == pytest.approx(0.25)
        assert tracker._records[0].task_type == "video_gen"
