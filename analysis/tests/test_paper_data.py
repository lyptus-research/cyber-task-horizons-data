"""Tests for paper data pipeline: JSONs, figure stages, and consistency.

These tests verify that:
1. Pipeline JSONs are internally consistent (paper_stats agrees with datasets_table)
2. Figure companion JSONs match the data they were computed from
3. Model estimates come from the correct source (data/keep/)
4. ICC computation includes the correct sessions (fails included for completions)
5. All Liquid-referenced numbers in the paper resolve to the same values as
   direct computation from the pipeline data
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

# ---------------------------------------------------------------------------
# Fixtures: load pipeline outputs once
# ---------------------------------------------------------------------------

DATA_DIR = _NOTEBOOKS_DIR / "figures" / "data"
OUT_DIR = _NOTEBOOKS_DIR / "figures" / "out"
KEEP_DIR = _NOTEBOOKS_DIR.parent / "data" / "keep"


def _skip_if_missing(path):
    if not path.exists():
        pytest.skip(f"Pipeline output not found: {path}")


@pytest.fixture(scope="module")
def paper_stats():
    path = DATA_DIR / "paper_stats.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def datasets_table():
    path = DATA_DIR / "datasets_table.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def task_difficulties():
    path = DATA_DIR / "task_difficulties.parquet"
    _skip_if_missing(path)
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def model_runs():
    path = DATA_DIR / "model_runs.parquet"
    _skip_if_missing(path)
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def best_available():
    path = DATA_DIR / "best_available_times.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def human_snapshot():
    path = DATA_DIR / "human_snapshot.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def icc_json():
    path = OUT_DIR / "icc_agreement.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cross_source_json():
    path = OUT_DIR / "cross_source_grid.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Cross-JSON consistency
# ---------------------------------------------------------------------------


class TestCrossJSONConsistency:
    """paper_stats and datasets_table should agree on shared numbers."""

    def test_headline_task_count_matches(self, paper_stats, datasets_table):
        """The total task count in datasets_table should match paper_stats."""
        assert paper_stats["headline_tasks"] == datasets_table["total_tasks"]

    def test_benchmark_count_matches(self, paper_stats, datasets_table):
        assert paper_stats["n_benchmarks"] == len(datasets_table["benchmarks"])

    def test_headline_tasks_match_task_difficulties(
        self, paper_stats, task_difficulties
    ):
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        model_runs = pd.read_parquet(model_runs_path)
        eval_tasks = set(model_runs["task_id"].astype(str))
        headline = task_difficulties.dropna(subset=["best_available_minutes"])
        headline = headline[headline["task_id"].astype(str).isin(eval_tasks)]
        assert paper_stats["headline_tasks"] == len(headline)

    def test_eval_set_tasks_match_model_runs(self, paper_stats, model_runs):
        assert paper_stats["eval_set_tasks"] == model_runs["task_id"].nunique()

    def test_n_models_matches_summaries(self, paper_stats):
        assert (
            paper_stats["n_models"] == paper_stats["n_sota"] + paper_stats["n_non_sota"]
        )

    def test_model_lists_consistent(self, paper_stats):
        all_models = set(paper_stats["all_models"])
        sota = set(paper_stats["sota_models"])
        non_sota = set(paper_stats["non_sota_models"])
        assert all_models == sota | non_sota
        assert len(sota & non_sota) == 0


# ---------------------------------------------------------------------------
# 2. Model estimates come from the correct source
# ---------------------------------------------------------------------------


class TestModelEstimatesSource:
    """Model estimates must come from data/keep/, not runs.parquet."""

    def test_task_difficulties_model_estimates_match_canonical(self, task_difficulties):
        """task_difficulties.model_estimate_minutes should match load_model_time_estimates."""
        from lib.data import load_model_time_estimates

        model_est = load_model_time_estimates()

        td = task_difficulties.dropna(subset=["model_estimate_minutes"])
        mismatches = 0
        for _, row in td.iterrows():
            tid = str(row["task_id"])
            if tid in model_est:
                if abs(row["model_estimate_minutes"] - model_est[tid]) > 0.01:
                    mismatches += 1
        assert (
            mismatches == 0
        ), f"{mismatches} tasks have model_estimate_minutes != load_model_time_estimates()"

    def test_model_estimates_files_exist(self):
        """Every benchmark should have a model_estimates.jsonl file."""
        benchmarks = [
            "cybashbench",
            "nl2bash",
            "intercode-ctf",
            "nyuctf",
            "cybench",
            "cvebench",
            "cybergym",
        ]
        for bench in benchmarks:
            path = KEEP_DIR / bench / f"{bench}_model_estimates.jsonl"
            assert path.exists(), f"Missing model estimates: {path}"


# ---------------------------------------------------------------------------
# 3. ICC computation correctness
# ---------------------------------------------------------------------------


class TestICCComputation:
    """ICC should include the correct sessions and filter to headline tasks."""

    def test_completion_icc_includes_fails(
        self, icc_json, human_snapshot, task_difficulties
    ):
        """Completion ICC N should be larger than passes-only paired count."""
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        eval_tasks = set(pd.read_parquet(model_runs_path)["task_id"].astype(str))
        td_headline = task_difficulties.dropna(subset=["best_available_minutes"])
        headline = set(
            td_headline[td_headline["task_id"].astype(str).isin(eval_tasks)][
                "task_id"
            ].astype(str)
        )

        # Count paired tasks from passes only
        pass_by_task = defaultdict(set)
        for p in human_snapshot["passes"]:
            if p["task_id"] in headline and p.get("server_elapsed_seconds"):
                pass_by_task[p["task_id"]].add(p.get("user_id", ""))
        passes_only_paired = sum(
            1 for experts in pass_by_task.values() if len(experts) >= 2
        )

        # Count paired tasks from passes + fails
        all_by_task = defaultdict(set)
        for key in ("passes", "fails"):
            for s in human_snapshot[key]:
                if s["task_id"] in headline and s.get("server_elapsed_seconds"):
                    all_by_task[s["task_id"]].add(s.get("user_id", ""))
        all_paired = sum(1 for experts in all_by_task.values() if len(experts) >= 2)

        # The JSON should match the passes+fails count
        assert icc_json["completion"]["n_tasks"] == all_paired
        # And it should be strictly more than passes-only
        assert all_paired > passes_only_paired

    def test_estimation_icc_filtered_to_headline(
        self, icc_json, human_snapshot, task_difficulties
    ):
        """Estimation ICC should only include headline tasks."""
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        eval_tasks = set(pd.read_parquet(model_runs_path)["task_id"].astype(str))
        td_headline = task_difficulties.dropna(subset=["best_available_minutes"])
        headline = set(
            td_headline[td_headline["task_id"].astype(str).isin(eval_tasks)][
                "task_id"
            ].astype(str)
        )

        est_by_task = defaultdict(set)
        for e in human_snapshot["estimations"]:
            if e["task_id"] in headline and e.get("estimated_seconds"):
                est_by_task[e["task_id"]].add(e.get("user_id", ""))
        expected = sum(1 for experts in est_by_task.values() if len(experts) >= 2)

        assert icc_json["estimation"]["n_tasks"] == expected

    def test_icc_values_in_range(self, icc_json):
        """ICC values should be between -1 and 1."""
        for key in ("estimation", "completion"):
            icc = icc_json[key]["icc"]
            assert -1 <= icc <= 1, f"{key} ICC out of range: {icc}"

    def test_icc_enforces_k2(self):
        """ICC computation should truncate to exactly 2 raters per task.

        If a task has 3+ raters and k=2 isn't enforced, the ANOVA
        decomposition is wrong (MS_within denominator assumes k=2).
        """
        from lib.icc import compute_icc

        # Construct data with 3 raters on one task
        rows = pd.DataFrame(
            [
                {"task_id": "t1", "expert": "A", "log2_min": 3.0},
                {"task_id": "t1", "expert": "B", "log2_min": 3.5},
                {"task_id": "t1", "expert": "C", "log2_min": 4.0},
                # Need enough tasks to pass min_tasks threshold
                *[
                    {"task_id": f"t{i}", "expert": "A", "log2_min": float(i)}
                    for i in range(2, 15)
                ],
                *[
                    {"task_id": f"t{i}", "expert": "B", "log2_min": float(i) + 0.3}
                    for i in range(2, 15)
                ],
            ]
        )
        icc_val, ci_lo, ci_hi, n_tasks, sigma_w = compute_icc(rows)

        # With k=2 enforcement, t1 should use only 2 of 3 raters.
        # The ICC should still compute successfully.
        assert icc_val is not None
        assert n_tasks >= 13  # t1 through t14

        # Verify the same result as if we pre-truncated to 2 raters
        truncated = rows.sort_values(["task_id", "expert"]).groupby("task_id").head(2)
        icc2, _, _, n2, _ = compute_icc(truncated)
        assert (
            icc_val == icc2
        ), "ICC with 3 raters should equal ICC with truncated 2 raters"


# ---------------------------------------------------------------------------
# 4. Cross-source grid correctness
# ---------------------------------------------------------------------------


class TestCrossSourceGrid:
    """Cross-source comparison stats should match direct computation."""

    def test_cross_source_n_within_headline(
        self, cross_source_json, model_runs, task_difficulties
    ):
        """Cross-source panel N values should not exceed the headline task count.

        If the cross-source stage doesn't filter to tasks with model evaluations,
        it could include tasks outside the headline set.
        """
        from lib.data import assemble_runs

        headline = assemble_runs(
            model_runs, task_difficulties, "best_available_minutes"
        )
        headline_tasks = headline["task_id"].nunique()

        for panel_name, stats in cross_source_json.items():
            assert (
                stats["n"] <= headline_tasks
            ), f"{panel_name} has N={stats['n']} but headline has only {headline_tasks} tasks"

    def test_model_estimates_not_from_runs_parquet(self, cross_source_json):
        """Panel (b) model_vs_comp should use real model estimates, not runs.parquet values.

        If model estimates were loaded from runs.parquet, CyBench tasks would
        have identical model_est and first-blood values (slope ≈ 1.0).
        The firstblood_vs_study panel slope should NOT be exactly 1.0.
        """
        fb_stats = cross_source_json.get("firstblood_vs_study")
        if fb_stats is None:
            pytest.skip("No firstblood_vs_study panel")
        # If model estimates were tautological, OLS slope would be ~1.0
        # With real model estimates, it should deviate
        assert fb_stats["ols_slope"] != 1.0

    def test_panel_n_counts_positive(self, cross_source_json):
        for panel_name, stats in cross_source_json.items():
            assert stats["n"] > 0, f"{panel_name} has no data"

    def test_r2_in_range(self, cross_source_json):
        for panel_name, stats in cross_source_json.items():
            assert 0 <= stats["ols_r2"] <= 1, f"{panel_name} OLS R² out of range"
            assert (
                0 <= stats["yx_r2"] <= 1 or stats["yx_r2"] < 0
            ), f"{panel_name} y=x+b R² unexpected: {stats['yx_r2']}"

    def test_residual_sd_positive(self, cross_source_json):
        for panel_name, stats in cross_source_json.items():
            assert stats["residual_sd"] > 0

    def test_calibration_sigma_uses_unbiased_estimator(self):
        """OLS residual SD should use ddof=2 (unbiased for 2-parameter regression)."""
        from lib.calibration import fit_ols

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = (
            x * 0.8
            + 0.5
            + np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2])
        )

        result = fit_ols(x, y)
        residuals = y - (result.intercept + result.slope * x)
        expected_sigma = float(np.std(residuals, ddof=2))

        assert abs(result.sigma - expected_sigma) < 1e-10, (
            f"fit_ols sigma ({result.sigma}) != np.std(residuals, ddof=2) ({expected_sigma}). "
            f"Likely using ddof=0 (biased estimator)."
        )


# ---------------------------------------------------------------------------
# 5. Paper stats human study numbers
# ---------------------------------------------------------------------------


class TestPaperStatsHumanStudy:
    """Human study numbers should match direct computation from snapshot."""

    def test_completion_hours_reasonable(self, paper_stats):
        """Completion hours should be > 0 and < 500 (sanity)."""
        assert 0 < paper_stats["completion_hours"] < 500

    def test_estimation_hours_reasonable(self, paper_stats):
        assert 0 < paper_stats["estimation_hours"] < 500

    def test_total_hours_is_sum(self, paper_stats):
        """Total should equal comp + est within rounding tolerance."""
        expected = paper_stats["completion_hours"] + paper_stats["estimation_hours"]
        assert abs(paper_stats["total_expert_hours"] - expected) <= 1

    def test_efficiency_ratio_positive(self, paper_stats):
        assert paper_stats["estimation_efficiency_ratio"] > 1

    def test_completion_equivalent_hours_exceeds_estimation_hours(self, paper_stats):
        """Estimates should cover more completion-equivalent hours than they cost."""
        assert (
            paper_stats["completion_equivalent_hours"] > paper_stats["estimation_hours"]
        )

    def test_difficulty_range(self, paper_stats, task_difficulties):
        """Min/max difficulty should match task_difficulties."""
        headline = task_difficulties.dropna(subset=["best_available_minutes"])
        tasks = headline["best_available_minutes"]
        assert paper_stats["difficulty_min_seconds"] == round(tasks.min() * 60)
        assert paper_stats["difficulty_max_hours"] == round(tasks.max() / 60, 1)

    def test_completion_session_counts(self, paper_stats):
        """Completion session counts should be internally consistent."""
        assert paper_stats["n_completion_sessions"] >= paper_stats["n_completion_tasks"]
        assert (
            paper_stats["n_completion_paired_tasks"]
            <= paper_stats["n_completion_tasks"]
        )
        assert paper_stats["n_completion_paired_tasks"] > 0

    def test_bootstrap_probabilities(self, paper_stats):
        """Bootstrap model-absence probabilities should be in [0, 1]."""
        assert 0 < paper_stats["prob_model_absent"] < 1
        assert 0 < paper_stats["prob_either_frontier_absent"] < 1
        assert (
            paper_stats["prob_either_frontier_absent"]
            > paper_stats["prob_model_absent"]
        )

    def test_frontier_above_trend(self, paper_stats):
        """Frontier models should sit above the trendline (ratio > 1)."""
        for f in paper_stats["frontier_above_trend"]:
            assert f["ratio"] > 1, f"{f['name']} ratio should be > 1"
        assert paper_stats["frontier_above_trend_avg"] > 1


# ---------------------------------------------------------------------------
# 6. Sensitivity dual JSON consistency
# ---------------------------------------------------------------------------


class TestSensitivityDualConsistency:
    """sensitivity_dual.json should be consistent with paper_stats."""

    @pytest.fixture(scope="class")
    def sensitivity_dual(self):
        path = DATA_DIR / "sensitivity_dual.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    def test_headline_dt_in_reasonable_range_of_paper_stats(
        self, sensitivity_dual, paper_stats
    ):
        """Headline multiverse median DT should be within 3 months of the point estimate.

        paper_stats.doubling_time_months is the point estimate (fixed weighting
        and regularisation). sensitivity_dual headline median is the multiverse
        median (sweeping weighting and regularisation). They measure different
        things but should be in the same ballpark.
        """
        sd_dt = sensitivity_dual["headline_best_available"]["dt_months"]["median"]
        ps_dt = paper_stats["doubling_time_months"]
        assert (
            abs(sd_dt - ps_dt) <= 3.0
        ), f"sensitivity_dual headline DT ({sd_dt}) too far from paper_stats DT ({ps_dt})"

    def test_all_treatments_have_both_metrics(self, sensitivity_dual):
        """Every treatment should have dt_months. Summary keys (prefixed _) are excluded."""
        for key, entry in sensitivity_dual.items():
            if key.startswith("_"):
                continue
            assert "dt_months" in entry, f"{key} missing dt_months"

    def test_dt_values_reasonable(self, sensitivity_dual):
        """All DT medians should be between 1 and 30 months."""
        for key, entry in sensitivity_dual.items():
            if key.startswith("_"):
                continue
            dt = entry["dt_months"]["median"]
            assert 1 <= dt <= 30, f"{key} DT median out of range: {dt}"


# ---------------------------------------------------------------------------
# 7. Regularisation comparison JSON consistency
# ---------------------------------------------------------------------------


class TestRegularisationConsistency:
    """regularisation_comparison.json should be consistent with paper_stats."""

    @pytest.fixture(scope="class")
    def reg_json(self):
        path = DATA_DIR / "regularisation_comparison.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    def test_target_p50_matches_paper_stats(self, reg_json, paper_stats):
        """Target model headline P50 should match paper_stats frontier P50."""
        target_name = reg_json["target_model"]
        target_p50 = reg_json["target"]["p50_headline"]
        # Find matching frontier model in paper_stats
        for fm in paper_stats["frontier_models"]:
            if fm["name"] == target_name:
                assert (
                    abs(target_p50 - fm["p50_minutes"]) < 1
                ), f"reg target P50 ({target_p50}) != paper_stats P50 ({fm['p50_minutes']})"
                break

    def test_strong_reg_increases_p50(self, reg_json):
        """Strong regularisation should increase P50 for frontier models."""
        target = reg_json["target"]
        assert target["p50_strong"] > target["p50_headline"]

    def test_ratio_consistent(self, reg_json):
        """Ratio should equal p50_strong / p50_headline."""
        target = reg_json["target"]
        expected = round(target["p50_strong"] / target["p50_headline"], 1)
        assert target["ratio"] == expected


# ---------------------------------------------------------------------------
# 8. Core pipeline function tests
# ---------------------------------------------------------------------------


class TestAssembleRuns:
    """assemble_runs() should produce correct DataFrames."""

    def test_assembles_with_best_available(self, task_difficulties):
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        model_runs = pd.read_parquet(model_runs_path)
        from lib.data import assemble_runs

        result = assemble_runs(model_runs, task_difficulties, "best_available_minutes")
        assert "human_minutes" in result.columns
        assert "log2_human_minutes" in result.columns
        assert "invsqrt_task_weight" in result.columns
        assert result["human_minutes"].notna().all()
        assert len(result) > 0

    def test_assembles_with_model_estimates(self, task_difficulties):
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        model_runs = pd.read_parquet(model_runs_path)
        from lib.data import assemble_runs

        result = assemble_runs(model_runs, task_difficulties, "model_estimate_minutes")
        assert len(result) >= len(
            assemble_runs(model_runs, task_difficulties, "best_available_minutes")
        ), "Model estimates should cover at least as many tasks as best-available"

    def test_different_columns_produce_different_values(self, task_difficulties):
        model_runs_path = DATA_DIR / "model_runs.parquet"
        _skip_if_missing(model_runs_path)
        model_runs = pd.read_parquet(model_runs_path)
        from lib.data import assemble_runs

        headline = assemble_runs(
            model_runs, task_difficulties, "best_available_minutes"
        )
        model_est = assemble_runs(
            model_runs, task_difficulties, "model_estimate_minutes"
        )

        # For tasks in both, at least some should differ
        hl_vals = headline.drop_duplicates("task_id").set_index("task_id")[
            "human_minutes"
        ]
        me_vals = model_est.drop_duplicates("task_id").set_index("task_id")[
            "human_minutes"
        ]
        common = set(hl_vals.index) & set(me_vals.index)
        diffs = sum(1 for t in common if abs(hl_vals[t] - me_vals[t]) > 0.01)
        assert diffs > 0, "Headline and model-estimate should differ on some tasks"


# ---------------------------------------------------------------------------
# 9. Task set consistency across models
# ---------------------------------------------------------------------------


class TestTaskSetConsistency:
    """Every model's task set must be a subset of a single canonical task set.

    The headline task count should equal the maximum per-model task count.
    No model should have tasks that no other current-campaign model has.
    Legacy models (GPT-2, GPT-3, GPT-3.5) should only contain tasks that
    at least one current-campaign model also evaluated.
    """

    LEGACY_ALIASES = {"GPT-2", "GPT-3", "GPT-3.5"}

    def test_headline_equals_max_model_tasks(self, model_runs, task_difficulties):
        """Headline task count == max per-model task count.

        If headline > max, there are tasks with human times but no model
        evaluated them — a pipeline leak (the bug this test was written for).
        """
        from lib.data import assemble_runs

        headline = assemble_runs(
            model_runs, task_difficulties, "best_available_minutes"
        )
        total = headline["task_id"].nunique()

        # Max across current (non-legacy) models
        current = headline[~headline["alias"].isin(self.LEGACY_ALIASES)]
        max_per_model = current.groupby("alias")["task_id"].nunique().max()

        assert total == max_per_model, (
            f"Headline has {total} tasks but max per-model is {max_per_model}. "
            f"{total - max_per_model} tasks exist only in legacy data."
        )

    def test_no_legacy_only_tasks(self, model_runs):
        """Legacy models should not introduce tasks absent from all current campaigns."""
        current = model_runs[~model_runs["alias"].isin(self.LEGACY_ALIASES)]
        legacy = model_runs[model_runs["alias"].isin(self.LEGACY_ALIASES)]

        current_tasks = set(current["task_id"].astype(str))
        legacy_tasks = set(legacy["task_id"].astype(str))
        legacy_only = legacy_tasks - current_tasks

        assert len(legacy_only) == 0, (
            f"{len(legacy_only)} tasks appear only in legacy models: "
            f"{sorted(list(legacy_only))[:5]}..."
        )

    def test_all_model_tasks_subset_of_headline(self, model_runs, task_difficulties):
        """Every model's task set should be a subset of the headline task set."""
        from lib.data import assemble_runs

        headline = assemble_runs(
            model_runs, task_difficulties, "best_available_minutes"
        )
        headline_tasks = set(headline["task_id"].astype(str))

        for alias in headline["alias"].unique():
            model_tasks = set(
                headline[headline["alias"] == alias]["task_id"].astype(str)
            )
            extra = model_tasks - headline_tasks
            assert (
                len(extra) == 0
            ), f"{alias} has {len(extra)} tasks not in headline: {extra}"

    def test_datasets_table_matches_headline(
        self, datasets_table, model_runs, task_difficulties
    ):
        """datasets_table.total_tasks must equal the headline task count."""
        from lib.data import assemble_runs

        headline = assemble_runs(
            model_runs, task_difficulties, "best_available_minutes"
        )
        assert datasets_table["total_tasks"] == headline["task_id"].nunique()

    def test_per_benchmark_tasks_sum_to_total(self, datasets_table):
        """Sum of per-benchmark task counts should equal total_tasks."""
        bench_sum = sum(b["tasks"] for b in datasets_table["benchmarks"])
        assert (
            bench_sum == datasets_table["total_tasks"]
        ), f"Benchmark sum ({bench_sum}) != total ({datasets_table['total_tasks']})"


class TestTaskDifficulties:
    """task_difficulties.parquet should have correct structure."""

    def test_has_required_columns(self, task_difficulties):
        required = [
            "task_id",
            "task_family",
            "completion_minutes",
            "estimate_minutes",
            "firstblood_minutes",
            "model_estimate_minutes",
            "best_available_minutes",
            "best_available_source",
        ]
        for col in required:
            assert col in task_difficulties.columns, f"Missing column: {col}"

    def test_best_available_source_valid(self, task_difficulties):
        valid_sources = {
            "completion",
            "censored",
            "first_blood",
            "expert_estimate",
            None,
        }
        actual = set(task_difficulties["best_available_source"].dropna().unique())
        assert actual <= valid_sources, f"Unexpected sources: {actual - valid_sources}"

    def test_model_estimates_complete(self, task_difficulties):
        """Every task should have a model estimate."""
        n_total = len(task_difficulties)
        n_with_model_est = task_difficulties["model_estimate_minutes"].notna().sum()
        assert (
            n_with_model_est == n_total
        ), f"Only {n_with_model_est}/{n_total} tasks have model estimates"


# ---------------------------------------------------------------------------
# 10. Regression tests: known-good headline values
# ---------------------------------------------------------------------------
# These are intentionally brittle. If the pipeline produces different numbers,
# either the code changed (investigate) or the data changed (update the
# expected values after confirming the change is correct).
#
# WHEN A REGRESSION TEST FAILS:
#   1. Do NOT just update the expected value to make the test pass.
#   2. Identify what changed: code, data, or configuration.
#   3. Determine whether the change is correct or a bug.
#   4. If correct: update the expected value AND commit with a message
#      explaining what changed and why (e.g. "n=291→296: added 5 new
#      CyberGym expert estimations in batch 12").
#   5. If a bug: fix the bug. The test did its job.
#
# To generate fresh expected values from the current pipeline:
#   DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
#     .venv/bin/python -m pytest tests/test_paper_data.py \
#     -k "Regression" --tb=long 2>&1 | grep "expected\|got\|shifted"


class TestRegressionHeadlineValues:
    """Snapshot tests for the exact numbers cited in the paper.

    These catch silent drift in P50 values, doubling time, task counts,
    and ICC — the numbers most consequential to the paper's claims.
    """

    # Known-good values (updated 2026-04-01 after 10M data fix + source coverage bug fixes)
    EXPECTED_HEADLINE_TASKS = 291
    EXPECTED_EVAL_SET_TASKS = 630
    EXPECTED_N_MODELS = 15
    EXPECTED_N_SOTA = 9
    EXPECTED_DOUBLING_TIME_MONTHS = 9.8
    EXPECTED_R_SQUARED = 0.95

    # P50 values in minutes (±5% tolerance for floating point / regularisation)
    EXPECTED_P50S = {
        "GPT-2": 0.5,
        "GPT-3": 0.7,
        "GPT-3.5": 2.0,
        "Claude 3 Opus": 6.3,
        "o3": 29.2,
        "Opus 4": 36.3,
        "GPT-5.1 Codex Max": 51.1,
        "GPT-5.3 Codex": 186.1,
        "Opus 4.6": 190.3,
    }
    P50_TOLERANCE = 0.05  # 5%

    # ICC values (±0.02 absolute tolerance)
    EXPECTED_ESTIMATION_ICC = 0.806
    EXPECTED_COMPLETION_ICC = 0.641
    ICC_TOLERANCE = 0.02

    @pytest.fixture(scope="class")
    def summaries(self):
        path = DATA_DIR / "model_summaries_human_2M.parquet"
        _skip_if_missing(path)
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def trendline(self):
        path = DATA_DIR / "trendline_params_human_2M.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    def test_headline_task_count(self, paper_stats):
        assert paper_stats["headline_tasks"] == self.EXPECTED_HEADLINE_TASKS

    def test_eval_set_task_count(self, paper_stats):
        assert paper_stats["eval_set_tasks"] == self.EXPECTED_EVAL_SET_TASKS

    def test_n_models(self, paper_stats):
        assert paper_stats["n_models"] == self.EXPECTED_N_MODELS

    def test_n_sota(self, paper_stats):
        assert paper_stats["n_sota"] == self.EXPECTED_N_SOTA

    def test_doubling_time(self, trendline):
        assert (
            trendline["doubling_time_months"] == self.EXPECTED_DOUBLING_TIME_MONTHS
        ), (
            f"Doubling time shifted: expected {self.EXPECTED_DOUBLING_TIME_MONTHS}, "
            f"got {trendline['doubling_time_months']}"
        )

    def test_r_squared(self, trendline):
        assert (
            abs(trendline["r_squared"] - self.EXPECTED_R_SQUARED) < 0.01
        ), f"R² shifted: expected {self.EXPECTED_R_SQUARED}, got {trendline['r_squared']}"

    def test_sota_model_p50s(self, summaries):
        """Every SOTA model's P50 should match the known-good value within tolerance."""
        for agent, expected_p50 in self.EXPECTED_P50S.items():
            row = summaries[summaries["agent"] == agent]
            assert len(row) == 1, f"Model {agent} not found in summaries"
            actual = float(row.iloc[0]["p50"])
            tol = max(expected_p50 * self.P50_TOLERANCE, 0.1)  # at least 0.1m absolute
            assert abs(actual - expected_p50) < tol, (
                f"{agent} P50 shifted: expected {expected_p50:.1f}m, got {actual:.1f}m "
                f"(tolerance {tol:.1f}m)"
            )

    def test_estimation_icc(self):
        path = OUT_DIR / "icc_agreement.json"
        _skip_if_missing(path)
        with open(path) as f:
            icc = json.load(f)
        actual = icc["estimation"]["icc"]
        assert (
            abs(actual - self.EXPECTED_ESTIMATION_ICC) < self.ICC_TOLERANCE
        ), f"Estimation ICC shifted: expected {self.EXPECTED_ESTIMATION_ICC}, got {actual}"

    def test_completion_icc(self):
        path = OUT_DIR / "icc_agreement.json"
        _skip_if_missing(path)
        with open(path) as f:
            icc = json.load(f)
        actual = icc["completion"]["icc"]
        assert (
            abs(actual - self.EXPECTED_COMPLETION_ICC) < self.ICC_TOLERANCE
        ), f"Completion ICC shifted: expected {self.EXPECTED_COMPLETION_ICC}, got {actual}"


class TestRegressionSensitivity:
    """Regression tests for sensitivity analysis values cited in the paper."""

    @pytest.fixture(scope="class")
    def sensitivity_dual(self):
        path = DATA_DIR / "sensitivity_dual.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    # Source treatment DT medians (months) — ±0.5 tolerance
    EXPECTED_SOURCE_DT_MEDIANS = {
        "headline_best_available": 7.9,
        "study_completions_only": 5.4,
        "actuals_only_completions_first_blood_2_4": 5.9,
        "no_first_blood_times": 7.3,
        "model_estimates_human_task_set": 5.8,
        "model_estimates_full_task_set": 6.5,
    }
    DT_TOLERANCE = 0.5

    def test_source_treatment_dt_medians(self, sensitivity_dual):
        for key, expected in self.EXPECTED_SOURCE_DT_MEDIANS.items():
            actual = sensitivity_dual[key]["dt_months"]["median"]
            assert (
                abs(actual - expected) < self.DT_TOLERANCE
            ), f"{key} DT median shifted: expected {expected}, got {actual}"

    def test_source_range(self, sensitivity_dual):
        sr = sensitivity_dual["_source_range"]
        assert abs(sr["min"] - 5.4) < 0.5
        assert abs(sr["max"] - 7.5) < 0.5

    def test_loo_range(self, sensitivity_dual):
        lr = sensitivity_dual["_loo_range"]
        assert abs(lr["min"] - 4.5) < 1.0
        assert abs(lr["max"] - 8.3) < 1.0


class TestRegressionCrossSource:
    """Regression tests for cross-source grid values cited in Appendix D."""

    @pytest.fixture(scope="class")
    def cross_source(self):
        path = OUT_DIR / "cross_source_grid.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    TOLERANCE = 0.05  # absolute on slopes/R²

    def test_est_vs_comp(self, cross_source):
        s = cross_source["est_vs_comp"]
        assert s["n"] == 39
        assert abs(s["ols_slope"] - 0.59) < self.TOLERANCE
        assert abs(s["ols_r2"] - 0.53) < self.TOLERANCE

    def test_model_vs_comp(self, cross_source):
        s = cross_source["model_vs_comp"]
        assert s["n"] == 100
        assert abs(s["ols_slope"] - 0.60) < self.TOLERANCE
        assert abs(s["ols_r2"] - 0.72) < self.TOLERANCE

    def test_expert_vs_model(self, cross_source):
        s = cross_source["expert_vs_model"]
        assert s["n"] == 224
        assert abs(s["ols_slope"] - 0.82) < self.TOLERANCE
        assert abs(s["ols_r2"] - 0.81) < self.TOLERANCE
        assert abs(s["residual_sd"] - 1.10) < 0.1

    def test_firstblood_vs_study(self, cross_source):
        s = cross_source["firstblood_vs_study"]
        assert s["n"] == 31
        assert abs(s["ols_slope"] - 0.96) < 0.15


class TestRegressionTrendlineAlternatives:
    """Regression tests for trendline functional form R² values."""

    @pytest.fixture(scope="class")
    def trendline_alts(self):
        path = DATA_DIR / "trendline_alternatives.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    def test_full_range_exponential(self, trendline_alts):
        r2 = trendline_alts["full_range"]["exponential"]["r_squared"]
        assert abs(r2 - 0.949) < 0.01

    def test_full_range_linear(self, trendline_alts):
        r2 = trendline_alts["full_range"]["linear"]["r_squared"]
        assert abs(r2 - 0.415) < 0.05

    def test_full_range_hyperbolic_singularity_in_past(self, trendline_alts):
        sing = trendline_alts["full_range"]["hyperbolic"]["singularity"]
        assert (
            "2026" in sing or "2025" in sing
        ), f"Singularity should be in the past: {sing}"

    def test_zoomed_2024_exponential(self, trendline_alts):
        r2 = trendline_alts["zoomed_2024"]["exponential"]["r_squared"]
        assert abs(r2 - 0.89) < 0.02


class TestRegressionRegularisation:
    """Regression tests for regularisation comparison values."""

    @pytest.fixture(scope="class")
    def reg_comp(self):
        path = DATA_DIR / "regularisation_comparison.json"
        _skip_if_missing(path)
        with open(path) as f:
            return json.load(f)

    def test_target_model(self, reg_comp):
        assert reg_comp["target_model"] == "GPT-5.3 Codex"

    def test_strong_reg_shifts_p50_upward(self, reg_comp):
        assert reg_comp["target"]["p50_strong"] > reg_comp["target"]["p50_headline"]
        assert abs(reg_comp["target"]["ratio"] - 1.4) < 0.2

    def test_n_stable_models(self, reg_comp):
        assert reg_comp["n_stable_models"] >= 8


class TestRegressionDatasetsTable:
    """Regression tests for per-benchmark task counts."""

    EXPECTED_BENCHMARKS = {
        "cybashbench": 51,
        "nl2bash": 9,
        "intercode_ctf": 45,
        "nyuctf": 33,
        "cybench": 37,
        "cvebench": 14,
        "cybergym": 102,
    }

    def test_per_benchmark_counts(self, datasets_table):
        for b in datasets_table["benchmarks"]:
            key = b["key"]
            if key in self.EXPECTED_BENCHMARKS:
                assert b["tasks"] == self.EXPECTED_BENCHMARKS[key], (
                    f"{key} task count shifted: expected {self.EXPECTED_BENCHMARKS[key]}, "
                    f"got {b['tasks']}"
                )

    def test_total(self, datasets_table):
        assert datasets_table["total_tasks"] == 291


class TestRegressionHumanStudy:
    """Regression tests for human study numbers in paper_stats."""

    def test_expert_count(self, paper_stats):
        assert paper_stats["n_participants"] == 10

    def test_completion_hours(self, paper_stats):
        assert abs(paper_stats["completion_hours"] - 88) <= 2

    def test_estimation_hours(self, paper_stats):
        assert abs(paper_stats["estimation_hours"] - 61) <= 2

    def test_total_expert_hours(self, paper_stats):
        assert abs(paper_stats["total_expert_hours"] - 149) <= 3

    def test_n_completion_tasks(self, paper_stats):
        assert paper_stats["n_completion_tasks"] == 105

    def test_n_estimation_tasks(self, paper_stats):
        assert paper_stats["n_estimation_tasks"] == 224

    def test_n_firstblood_tasks(self, paper_stats):
        assert paper_stats["n_firstblood_tasks"] == 37

    def test_difficulty_range(self, paper_stats):
        assert paper_stats["difficulty_min_seconds"] == 28
        assert paper_stats["difficulty_max_hours"] == 36.0

    def test_adaptation_buffer(self, paper_stats):
        assert abs(paper_stats["adaptation_buffer_lower_months"] - 5.7) < 1.0
        assert abs(paper_stats["adaptation_buffer_upper_months"] - 13.1) < 2.0
