"""Tests for lib.trendline — P50 trendline computation and plotting.

Tests verify integration with METR's eval-analysis-public without
requiring S3 access or .eval files.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.trendline import (
    LEGACY_MODELS,
    RELEASE_DATES,
    build_agent_summaries,
    merge_legacy_models,
    plot_horizon_graph,
)


def _make_runs(agent, alias, n_tasks=50, accuracy=0.5, min_minutes=1.0,
               max_minutes=480.0, seed=42):
    rng = np.random.RandomState(seed)
    log2_range = np.linspace(np.log2(min_minutes), np.log2(max_minutes), n_tasks)
    human_minutes = 2 ** log2_range
    threshold = min_minutes * (max_minutes / min_minutes) ** accuracy
    log2_threshold = np.log2(threshold)
    probs = 1.0 / (1.0 + np.exp(0.5 * (log2_range - log2_threshold)))
    scores = (rng.random(n_tasks) < probs).astype(int)
    df = pd.DataFrame({
        "task_id": [f"task_{i}" for i in range(n_tasks)],
        "task_family": "synthetic",
        "score_binarized": scores,
        "total_tokens": 0,
        "human_minutes": human_minutes,
        "task_source": "synthetic",
        "agent": agent,
        "alias": alias,
    })
    df["log2_human_minutes"] = np.log2(df["human_minutes"])
    n = len(df)
    df["equal_task_weight"] = 1.0 / n
    df["invsqrt_task_weight"] = 1.0 / n
    return df


@pytest.fixture
def synthetic_campaign_data():
    weak = _make_runs("model/weak", "Weak Model", accuracy=0.3, seed=1)
    strong = _make_runs("model/strong", "Strong Model", accuracy=0.7, seed=2)
    return {
        "Weak Model": {"runs": weak, "runs_best": weak.copy(), "runs_human": weak.copy()},
        "Strong Model": {"runs": strong, "runs_best": strong.copy(), "runs_human": strong.copy()},
    }


class TestBuildAgentSummaries:
    def test_returns_one_row_per_model(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        assert len(summaries) == 2

    def test_strong_model_has_higher_p50(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        weak_p50 = summaries.loc[summaries["agent"] == "Weak Model", "p50"].iloc[0]
        strong_p50 = summaries.loc[summaries["agent"] == "Strong Model", "p50"].iloc[0]
        assert strong_p50 > weak_p50

    def test_has_required_columns(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        required = {"agent", "p50", "p80", "coefficient", "intercept", "release_date"}
        assert required.issubset(set(summaries.columns))

    def test_release_dates_are_timestamps(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        assert pd.api.types.is_datetime64_any_dtype(summaries["release_date"])

    def test_empty_data_returns_empty(self):
        summaries = build_agent_summaries({}, time_source="runs")
        assert summaries.empty

    def test_all_zero_scores(self):
        df = _make_runs("model/zero", "Zero", accuracy=0.0, seed=99)
        df["score_binarized"] = 0
        data = {"Zero": {"runs": df, "runs_best": df.copy(), "runs_human": df.copy()}}
        summaries = build_agent_summaries(data, "runs", release_dates={"Zero": "2024-01-01"})
        assert summaries.loc[summaries["agent"] == "Zero", "p50"].iloc[0] == 0


class TestPlotHorizonGraph:
    def test_renders_without_error(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(111)
        plot_horizon_graph(summaries, release_dates=release_dates, non_frontier=set(), success_percent=50, fig=fig)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_p80_renders(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(111)
        plot_horizon_graph(summaries, release_dates=release_dates, non_frontier=set(), success_percent=80, fig=fig)
        plt.close(fig)

    def test_linear_scale(self, synthetic_campaign_data):
        release_dates = {"Weak Model": "2024-01-01", "Strong Model": "2025-01-01"}
        summaries = build_agent_summaries(synthetic_campaign_data, "runs", release_dates=release_dates)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(111)
        plot_horizon_graph(summaries, release_dates=release_dates, non_frontier=set(), y_scale="linear", fig=fig)
        plt.close(fig)


class TestDoublingTime:
    def test_positive_doubling_time(self):
        models = {}
        for i, (name, acc) in enumerate([("Early", 0.2), ("Mid", 0.4), ("Late", 0.7)]):
            df = _make_runs(f"model/{name}", name, accuracy=acc, seed=i + 10)
            models[name] = {"runs": df, "runs_best": df.copy(), "runs_human": df.copy()}
        release_dates = {"Early": "2023-01-01", "Mid": "2024-01-01", "Late": "2025-01-01"}
        summaries = build_agent_summaries(models, "runs", release_dates=release_dates)
        # Verify P50s are increasing (later models are stronger)
        p50s = summaries.sort_values("release_date")["p50"].values
        assert p50s[-1] > p50s[0], "Latest model should have highest P50"


class TestMergeLegacyModels:
    def test_adds_legacy_to_campaign_data(self, synthetic_campaign_data):
        merged = merge_legacy_models(synthetic_campaign_data, legacy_models=LEGACY_MODELS[:2])
        assert len(merged) >= len(synthetic_campaign_data)

    def test_does_not_mutate_input(self, synthetic_campaign_data):
        original_keys = set(synthetic_campaign_data.keys())
        merge_legacy_models(synthetic_campaign_data, legacy_models=LEGACY_MODELS[:1])
        assert set(synthetic_campaign_data.keys()) == original_keys


class TestConstants:
    def test_all_release_dates_are_valid(self):
        for agent, d in RELEASE_DATES.items():
            parsed = pd.to_datetime(d)
            assert parsed is not pd.NaT, f"Invalid date for {agent}: {d}"

    def test_legacy_aliases_are_in_release_dates(self):
        for cfg in LEGACY_MODELS:
            assert cfg["alias"] in RELEASE_DATES

    def test_legacy_models_are_pre_chat_era_only(self):
        allowed = {"GPT-2", "GPT-3", "GPT-3.5"}
        actual = {cfg["alias"] for cfg in LEGACY_MODELS}
        assert actual == allowed
