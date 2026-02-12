"""Tests for cost tracker."""

import pytest
from datetime import datetime, timezone

from videoclaw.cost.tracker import CostRecord, CostTracker


def test_track_and_summary():
    tracker = CostTracker(project_id="test")
    tracker.record(CostRecord(
        task_id="t1",
        model_id="sora",
        execution_mode="cloud",
        api_cost_usd=0.50,
        compute_cost_usd=0.0,
        duration_seconds=10.0,
        video_seconds=5.0,
        timestamp=datetime.now(timezone.utc),
    ))
    tracker.record(CostRecord(
        task_id="t2",
        model_id="cogvideo",
        execution_mode="local",
        api_cost_usd=0.0,
        compute_cost_usd=0.02,
        duration_seconds=30.0,
        video_seconds=5.0,
        timestamp=datetime.now(timezone.utc),
    ))
    summary = tracker.get_summary()
    assert summary.total_usd == pytest.approx(0.52)
    assert sum(r.video_seconds for r in summary.records) == pytest.approx(10.0)


def test_budget_check():
    tracker = CostTracker(project_id="test", budget_usd=5.0)
    tracker.record(CostRecord(
        task_id="t1",
        model_id="sora",
        execution_mode="cloud",
        api_cost_usd=4.50,
        compute_cost_usd=0.0,
        duration_seconds=10.0,
        timestamp=datetime.now(timezone.utc),
    ))
    within, remaining = tracker.check_budget()
    assert within is True
    assert remaining == pytest.approx(0.50)

    # Exceed budget
    tracker2 = CostTracker(project_id="test2", budget_usd=4.0)
    tracker2.record(CostRecord(
        task_id="t1",
        model_id="sora",
        execution_mode="cloud",
        api_cost_usd=4.50,
        compute_cost_usd=0.0,
        duration_seconds=10.0,
        timestamp=datetime.now(timezone.utc),
    ))
    within2, _ = tracker2.check_budget()
    assert within2 is False
