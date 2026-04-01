"""Tests for lib.data — build_best_available_times and the difficulty hierarchy.

Tests use synthetic data to verify the priority ordering (completion >
censored > first-blood > estimate) and the outlier exclusion gate.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.data import build_best_available_times


def _make_completion(task_id, seconds, session_id="s1", passed=True):
    return {
        "task_id": task_id,
        "server_elapsed_seconds": seconds,
        "session_id": session_id,
        "passed": passed,
    }


def _make_estimation(task_id, seconds, session_id="e1"):
    return {
        "task_id": task_id,
        "estimated_seconds": seconds,
        "session_id": session_id,
    }


class TestBuildBestAvailableTimes:
    def test_completion_takes_priority_over_estimation(self):
        bat = build_best_available_times(
            completions=[_make_completion("t1", 300)],  # 5 min
            censored=[],
            first_blood_minutes={},
            estimations=[_make_estimation("t1", 60)],  # 1 min estimate
        )
        assert "t1" in bat
        minutes, source = bat["t1"]
        assert source == "completion"
        assert np.isclose(minutes, 5.0)

    def test_estimation_used_as_fallback(self):
        bat = build_best_available_times(
            completions=[],
            censored=[],
            first_blood_minutes={},
            estimations=[_make_estimation("t1", 120)],  # 2 min estimate
        )
        assert "t1" in bat
        minutes, source = bat["t1"]
        assert source == "expert_estimate"
        assert np.isclose(minutes, 2.0)

    def test_first_blood_used_when_no_completion_or_censored(self):
        bat = build_best_available_times(
            completions=[],
            censored=[],
            first_blood_minutes={"t1": 15.0},
            estimations=[],
        )
        assert "t1" in bat
        minutes, source = bat["t1"]
        assert source == "first_blood"
        assert minutes == 15.0

    def test_excluded_task_ids_removes_tasks(self):
        bat = build_best_available_times(
            completions=[_make_completion("t1", 300), _make_completion("t2", 600)],
            censored=[],
            first_blood_minutes={},
            estimations=[],
            excluded_task_ids={"t1"},
        )
        assert "t1" not in bat
        assert "t2" in bat

    def test_timing_corrections_applied(self):
        corrections = {"corrected-session": 120.0}  # 2 minutes
        bat = build_best_available_times(
            completions=[_make_completion("t1", 99999, session_id="corrected-session")],
            censored=[],
            first_blood_minutes={},
            estimations=[],
            timing_corrections=corrections,
        )
        assert "t1" in bat
        minutes, _ = bat["t1"]
        assert np.isclose(minutes, 2.0)  # 120 seconds = 2 minutes

    def test_geometric_mean_for_multiple_completions(self):
        bat = build_best_available_times(
            completions=[
                _make_completion("t1", 60, session_id="s1"),   # 1 min
                _make_completion("t1", 240, session_id="s2"),  # 4 min
            ],
            censored=[],
            first_blood_minutes={},
            estimations=[],
        )
        minutes, source = bat["t1"]
        assert source == "completion"
        # geometric mean of 1 and 4 = 2
        assert np.isclose(minutes, 2.0)

    def test_censored_skipped_on_short_horizon_benchmarks(self):
        """Censored completions on cybashbench/nl2bash should be ignored."""
        bat = build_best_available_times(
            completions=[],
            censored=[_make_completion("cybashbench_test/t1", 3600, passed=False)],
            first_blood_minutes={},
            estimations=[_make_estimation("cybashbench_test/t1", 30)],
            task_bench={"cybashbench_test/t1": "cybashbench"},
        )
        minutes, source = bat["cybashbench_test/t1"]
        # Should use the estimation, not the censored completion
        assert source == "expert_estimate"

    def test_empty_inputs_returns_empty(self):
        bat = build_best_available_times(
            completions=[], censored=[], first_blood_minutes={}, estimations=[],
        )
        assert len(bat) == 0


class TestOutlierExclusion:
    """Verify that KNOWN_OUTLIERS are excluded from best_available_times."""

    def test_glacier_exchange_excluded(self):
        """glacier-exchange is in KNOWN_OUTLIERS and should be filtered."""
        from lib.corrections import KNOWN_OUTLIERS

        excluded = {o.task_id for o in KNOWN_OUTLIERS}
        assert "glacier-exchange" in excluded

        bat = build_best_available_times(
            completions=[_make_completion("glacier-exchange", 39000)],  # 10.9h
            censored=[],
            first_blood_minutes={},
            estimations=[],
            excluded_task_ids=excluded,
        )
        assert "glacier-exchange" not in bat


class TestDeriveHumanVariants:
    """Tests for results.derive_human_variants."""

    def test_runs_best_overrides_human_minutes(self):
        from lib.results import derive_human_variants

        runs = pd.DataFrame({
            "task_id": ["t1", "t2"],
            "task_family": ["bench", "bench"],
            "human_minutes": [10.0, 20.0],
            "log2_human_minutes": [np.log2(10), np.log2(20)],
            "score_binarized": [1, 0],
        })
        bat = {"t1": (5.0, "completion")}  # override t1 from 10 to 5

        runs_best, runs_human = derive_human_variants(runs, bat)

        assert runs_best.loc[runs_best["task_id"] == "t1", "human_minutes"].iloc[0] == 5.0
        assert runs_best.loc[runs_best["task_id"] == "t2", "human_minutes"].iloc[0] == 20.0

    def test_runs_human_filtered_to_bat_tasks(self):
        from lib.results import derive_human_variants

        runs = pd.DataFrame({
            "task_id": ["t1", "t2", "t3"],
            "task_family": ["bench", "bench", "bench"],
            "human_minutes": [10.0, 20.0, 30.0],
            "log2_human_minutes": [np.log2(10), np.log2(20), np.log2(30)],
            "score_binarized": [1, 0, 1],
        })
        bat = {"t1": (5.0, "completion"), "t3": (15.0, "estimate")}

        _, runs_human = derive_human_variants(runs, bat)

        assert set(runs_human["task_id"]) == {"t1", "t3"}

    def test_runs_human_weights_sum_to_one(self):
        from lib.results import derive_human_variants

        runs = pd.DataFrame({
            "task_id": [f"t{i}" for i in range(10)],
            "task_family": ["a"] * 5 + ["b"] * 5,
            "human_minutes": [float(i + 1) for i in range(10)],
            "log2_human_minutes": [np.log2(i + 1) for i in range(10)],
            "score_binarized": [1, 0] * 5,
        })
        bat = {f"t{i}": (float(i + 1), "est") for i in range(10)}

        _, runs_human = derive_human_variants(runs, bat)

        assert np.isclose(runs_human["invsqrt_task_weight"].sum(), 1.0)

    def test_does_not_mutate_input(self):
        from lib.results import derive_human_variants

        runs = pd.DataFrame({
            "task_id": ["t1"],
            "task_family": ["bench"],
            "human_minutes": [10.0],
            "log2_human_minutes": [np.log2(10)],
            "score_binarized": [1],
        })
        original_val = runs["human_minutes"].iloc[0]
        derive_human_variants(runs, {"t1": (5.0, "completion")})
        assert runs["human_minutes"].iloc[0] == original_val
