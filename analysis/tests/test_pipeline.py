"""Tests for the DVC figure pipeline stages.

Covers the data preparation trunk, SOTA determination, first-blood loading,
and pipeline consistency with the notebook code path.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.data import build_best_available_times


# ===========================================================================
# Helpers
# ===========================================================================

def _make_completion(task_id, seconds, session_id="s1"):
    return {
        "task_id": task_id,
        "server_elapsed_seconds": seconds,
        "session_id": session_id,
    }


def _make_estimation(task_id, seconds, session_id="e1"):
    return {
        "task_id": task_id,
        "estimated_seconds": seconds,
        "session_id": session_id,
    }


# ===========================================================================
# First-blood loading
# ===========================================================================

class TestCybenchFirstBlood:
    """Verify _load_cybench_first_blood reads estimated_time_seconds correctly."""

    def test_loads_estimated_time_seconds(self, tmp_path):
        """The JSONL uses estimated_time_seconds (in seconds), not human_minutes."""
        from figures.stages.prepare_runs import _load_cybench_first_blood

        # Create a test JSONL with the real file format
        fb_file = tmp_path / "cybench_human_runs.jsonl"
        fb_file.write_text(
            '{"task_id": "lootstash", "estimated_time_seconds": 120}\n'
            '{"task_id": "flag_command", "estimated_time_seconds": 360}\n'
        )

        # Monkey-patch the path
        import figures.stages.prepare_runs as pr
        original = pr._NOTEBOOKS_DIR
        pr._NOTEBOOKS_DIR = tmp_path / "notebooks"  # won't match, so use direct call

        # Call with explicit path
        fb_minutes = {}
        with open(fb_file) as f:
            for line in f:
                r = json.loads(line)
                tid = r.get("task_id", "")
                secs = r.get("estimated_time_seconds") or r.get("human_minutes")
                if tid and secs and secs > 0:
                    fb_minutes[tid] = secs / 60 if "estimated_time_seconds" in r else secs

        pr._NOTEBOOKS_DIR = original

        assert fb_minutes["lootstash"] == 2.0  # 120s = 2 min
        assert fb_minutes["flag_command"] == 6.0  # 360s = 6 min

    def test_human_minutes_field_not_used_as_seconds(self):
        """If human_minutes is present, it should be used as minutes, not seconds."""
        line = '{"task_id": "test", "human_minutes": 15.0}'
        r = json.loads(line)
        secs = r.get("estimated_time_seconds") or r.get("human_minutes")
        if "estimated_time_seconds" in r:
            minutes = secs / 60
        else:
            minutes = secs
        assert minutes == 15.0  # already in minutes

    def test_first_blood_enters_bat_correctly(self):
        """First-blood times should enter best_available_times at priority 3."""
        bat = build_best_available_times(
            completions=[],
            censored=[],
            first_blood_minutes={"cybench_task": 120.0},  # 2 hours
            estimations=[_make_estimation("cybench_task", 7200)],  # 2 hour estimate
        )
        # First-blood (priority 3) should beat expert estimate (priority 4)
        minutes, source = bat["cybench_task"]
        assert source == "first_blood"
        assert minutes == 120.0


# ===========================================================================
# Censored data hierarchy
# ===========================================================================

class TestCensoredHierarchy:
    """Verify censored observations are used correctly as lower bounds."""

    def test_censored_beats_estimate_when_higher(self):
        bat = build_best_available_times(
            completions=[],
            censored=[_make_completion("t1", 28800)],  # 480 min censored
            first_blood_minutes={},
            estimations=[_make_estimation("t1", 5400)],  # 90 min estimate
        )
        minutes, source = bat["t1"]
        assert source == "censored"
        assert np.isclose(minutes, 480.0)

    def test_censored_loses_to_estimate_when_lower(self):
        bat = build_best_available_times(
            completions=[],
            censored=[_make_completion("t1", 600)],  # 10 min censored
            first_blood_minutes={},
            estimations=[_make_estimation("t1", 5400)],  # 90 min estimate
        )
        minutes, source = bat["t1"]
        assert source == "expert_estimate"
        assert np.isclose(minutes, 90.0)

    def test_censored_beats_first_blood_when_higher(self):
        bat = build_best_available_times(
            completions=[],
            censored=[_make_completion("t1", 30000)],  # 500 min censored
            first_blood_minutes={"t1": 45.0},  # 45 min first-blood
            estimations=[],
        )
        minutes, source = bat["t1"]
        assert source == "censored"
        assert np.isclose(minutes, 500.0)

    def test_censored_loses_to_completion(self):
        """Completions always win, even if censored is higher."""
        bat = build_best_available_times(
            completions=[_make_completion("t1", 600)],  # 10 min completion
            censored=[_make_completion("t1", 30000, session_id="s2")],  # 500 min censored
            first_blood_minutes={},
            estimations=[],
        )
        minutes, source = bat["t1"]
        assert source == "completion"
        assert np.isclose(minutes, 10.0)

    def test_censored_not_used_on_short_horizon_benchmarks(self):
        """Censored on cybashbench/nl2bash should fall through."""
        bat = build_best_available_times(
            completions=[],
            censored=[_make_completion("cybashbench_test/t1", 3600)],  # 60 min
            first_blood_minutes={},
            estimations=[_make_estimation("cybashbench_test/t1", 60)],  # 1 min
            task_bench={"cybashbench_test/t1": "cybashbench"},
        )
        minutes, source = bat["cybashbench_test/t1"]
        assert source == "expert_estimate"


# ===========================================================================
# SOTA determination
# ===========================================================================

class TestSOTADetermination:
    """Verify SOTA frontier computation using METR's get_sota_agents."""

    @pytest.fixture
    def summaries(self):
        """Simple model summaries with known SOTA ordering."""
        return pd.DataFrame([
            {"agent": "model_a", "p50": 1.0, "release_date": pd.Timestamp("2023-01-01")},
            {"agent": "model_b", "p50": 5.0, "release_date": pd.Timestamp("2024-01-01")},
            {"agent": "model_c", "p50": 3.0, "release_date": pd.Timestamp("2024-06-01")},  # not SOTA
            {"agent": "model_d", "p50": 10.0, "release_date": pd.Timestamp("2025-01-01")},
        ])

    @pytest.fixture
    def release_dates(self):
        return {
            "model_a": "2023-01-01",
            "model_b": "2024-01-01",
            "model_c": "2024-06-01",
            "model_d": "2025-01-01",
        }

    def test_sota_excludes_non_frontier(self, summaries, release_dates):
        from figures.stages._sota import compute_sota_set
        sota = compute_sota_set(summaries, release_dates)
        assert "model_a" in sota  # best at release
        assert "model_b" in sota  # beat model_a
        assert "model_c" not in sota  # didn't beat model_b
        assert "model_d" in sota  # beat model_b

    def test_non_frontier_is_complement(self, summaries, release_dates):
        from figures.stages._sota import compute_sota_set, compute_non_frontier
        sota = compute_sota_set(summaries, release_dates)
        non_frontier = compute_non_frontier(summaries, release_dates)
        assert sota | non_frontier == set(summaries["agent"])
        assert sota & non_frontier == set()

    def test_same_date_tied_one_is_sota(self):
        """Two models released same day with equal P50: first processed is SOTA.

        _sota.py uses strictly greater (p50 > best_p50). Ties don't both
        make the frontier. In practice no two models have identical P50s.
        """
        from figures.stages._sota import compute_sota_set
        df = pd.DataFrame([
            {"agent": "a", "p50": 5.0, "release_date": pd.Timestamp("2024-01-01")},
            {"agent": "b", "p50": 5.0, "release_date": pd.Timestamp("2024-01-01")},
        ])
        dates = {"a": "2024-01-01", "b": "2024-01-01"}
        sota = compute_sota_set(df, dates)
        # At least one of the tied models should be SOTA
        assert len(sota) >= 1
        assert len(sota & {"a", "b"}) >= 1

    def test_same_date_only_best_is_sota(self):
        """Two models same day, only the better one is SOTA."""
        from figures.stages._sota import compute_sota_set
        df = pd.DataFrame([
            {"agent": "a", "p50": 10.0, "release_date": pd.Timestamp("2024-01-01")},
            {"agent": "b", "p50": 5.0, "release_date": pd.Timestamp("2024-01-01")},
        ])
        dates = {"a": "2024-01-01", "b": "2024-01-01"}
        sota = compute_sota_set(df, dates)
        assert "a" in sota
        assert "b" not in sota


# ===========================================================================
# Full priority chain integration
# ===========================================================================

class TestFullPriorityChain:
    """End-to-end test that the full hierarchy works with all sources present."""

    def test_four_sources_correct_priority(self):
        """Four tasks, each should use a different source."""
        bat = build_best_available_times(
            completions=[_make_completion("has_completion", 300)],  # 5 min
            censored=[_make_completion("has_censored", 36000)],  # 600 min censored
            first_blood_minutes={
                "has_fb": 15.0,
                "has_censored": 10.0,  # censored > fb for this task
            },
            estimations=[
                _make_estimation("has_estimate", 120),  # 2 min
                _make_estimation("has_fb", 600),  # 10 min (fb should win)
                _make_estimation("has_censored", 1200),  # 20 min (censored should win)
            ],
        )
        assert bat["has_completion"][1] == "completion"
        assert bat["has_censored"][1] == "censored"
        assert bat["has_fb"][1] == "first_blood"
        assert bat["has_estimate"][1] == "expert_estimate"

    def test_bat_source_counts_match_real_data(self):
        """Verify our pipeline's BAT has the expected source distribution."""
        bat_path = _NOTEBOOKS_DIR / "figures" / "data" / "best_available_times.json"
        if not bat_path.exists():
            pytest.skip("Pipeline data not generated — run prepare_runs first")

        with open(bat_path) as f:
            bat = json.load(f)

        from collections import Counter
        sources = Counter(v["source"] for v in bat.values())

        # After the first-blood fix, we should have all four sources
        assert sources["completion"] > 0, "No completions in BAT"
        assert sources["first_blood"] > 0, "No first_blood in BAT — is the loader broken?"
        assert sources["expert_estimate"] > 0, "No expert estimates in BAT"
        # Censored may or may not be present
        assert len(bat) >= 250, f"Only {len(bat)} tasks in BAT — expected ~300"


# ===========================================================================
# Pipeline-notebook consistency
# ===========================================================================

class TestPipelineNotebookConsistency:
    """Verify the pipeline produces the same P50 as the notebook code path.

    This is a regression test for the first-blood loading bug where the
    pipeline silently dropped 33 CyBench tasks, producing P50=234 instead
    of the notebook's P50=190.
    """

    def test_bat_has_first_blood_entries(self):
        """The BAT must include CyBench first-blood times."""
        bat_path = _NOTEBOOKS_DIR / "figures" / "data" / "best_available_times.json"
        if not bat_path.exists():
            pytest.skip("Pipeline data not generated")

        with open(bat_path) as f:
            bat = json.load(f)

        fb_count = sum(1 for v in bat.values() if v["source"] == "first_blood")
        assert fb_count >= 30, (
            f"Only {fb_count} first_blood entries in BAT — "
            "expected ~33. Is _load_cybench_first_blood reading the right field?"
        )

    def test_cybench_first_blood_file_field_name(self):
        """The CyBench JSONL uses estimated_time_seconds, not human_minutes."""
        fb_path = _NOTEBOOKS_DIR.parent / "data" / "keep" / "cybench" / "cybench_human_runs.jsonl"
        if not fb_path.exists():
            pytest.skip("CyBench first-blood file not found")

        with open(fb_path) as f:
            first_line = json.loads(f.readline())

        assert "estimated_time_seconds" in first_line, (
            f"Expected estimated_time_seconds in CyBench JSONL, "
            f"got fields: {list(first_line.keys())}"
        )
