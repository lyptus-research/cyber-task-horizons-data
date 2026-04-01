"""Trendline plotting: P50/P80 time horizons vs model release date.

Wraps METR's eval-analysis-public for all statistics and plotting. This
module handles data preparation and provides thin wrappers that construct
METR's config objects programmatically.

Usage: trendline stages read pre-fitted model_summaries and pass them
to plot_horizon_graph. SOTA is determined from the is_sota column in
model_summaries, not from a static list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .constants import DEFAULT_REGULARIZATION
from .results import load_legacy_runs, derive_human_variants

# ---------------------------------------------------------------------------
# Import METR's canonical code
# ---------------------------------------------------------------------------
# Monorepo-level checkout (third-party/ at repo root, not cyber-task-horizons/third-party/).
# The monorepo-level checkout tracks upstream METR which restructured under horizon/ in mid-2025.
)
)

# Add both src/ (for `import horizon.*`) and the parent (for `import src.*`).
# METR's code uses both conventions internally. Without both paths, imports
# break depending on which file is loaded first.

from horizon.wrangle.logistic import agent_regression  # noqa: E402
from horizon.wrangle.bootstrap import compute_bootstrap_regressions  # noqa: E402
from horizon.plot.logistic import (  # noqa: E402
    plot_horizon_graph as _metr_plot_horizon_graph,
)
from horizon.plot.bootstrap_ci import (  # noqa: E402
    compute_bootstrap_confidence_region,
)

# =============================================================================
# Constants
# =============================================================================


def _load_release_dates() -> dict[str, str]:
    """Load release dates from model config JSONs (single source of truth).

    Delegates to figures.stages._common_data.load_release_dates() which is
    the canonical implementation shared across the pipeline.
    """
    # Import here to avoid circular dependency at module level
    import importlib

    try:
        mod = importlib.import_module("figures.stages._common_data")
        return mod.load_release_dates()
    except (ModuleNotFoundError, ImportError):
        # Fallback for contexts where figures.stages is not on sys.path
        # (e.g. interactive notebook use). Use the same logic inline.
        import json

        from analysis.config import MODELS_DIR
        models_dir = MODELS_DIR
        _ALIAS_MAP = {
            "Claude Haiku 4.5": "Haiku 4.5",
            "Claude Opus 4": "Opus 4",
            "Claude Sonnet 4.6": "Sonnet 4.6",
            "Claude Opus 4.6": "Opus 4.6",
            "Gemini 2.5 Pro (June 2025)": "Gemini 2.5 Pro",
        }
        dates: dict[str, str] = {
            "GPT-2": "2019-11-05",
            "GPT-3": "2020-07-11",
            "GPT-3.5": "2022-03-15",
        }
        if models_dir.exists():
            for json_file in sorted(models_dir.glob("*.json")):
                with open(json_file) as f:
                    data = json.load(f)
                for model in data.get("models", []):
                    alias = model.get("alias", "")
                    release = model.get("release_date", "")
                    if alias and release:
                        dates[_ALIAS_MAP.get(alias, alias)] = release
        return dates


RELEASE_DATES: dict[str, str] = _load_release_dates()


LEGACY_MODELS: list[dict[str, str]] = [
    {"alias_filter": "GPT 2", "agent": "openai/gpt2-xl", "alias": "GPT-2"},
    {"alias_filter": "GPT 3", "agent": "openai/davinci-002", "alias": "GPT-3"},
    {"alias_filter": "GPT 3.5", "agent": "openai/gpt-3.5-turbo", "alias": "GPT-3.5"},
]

_WRANGLE_PARAMS: dict[str, Any] = {
    "weighting": "invsqrt_task_weight",
    "regularization": DEFAULT_REGULARIZATION,
    "success_percents": [50, 80],
    "confidence_level": 0.95,
}

DEFAULT_N_BOOTSTRAP = 1000

# Provider colors using the Lyptus palette
try:
    from .lyptus_style import COLORS as _C

    _ANTHROPIC_COLOR = _C["coral"]  # #ff5b5b
    _OPENAI_COLOR = _C["teal"]  # #00897b
    _GOOGLE_COLOR = _C["slate"]  # #457b9d
    _OTHER_COLOR = _C["plum"]  # #6d597a
    _GRID_COLOR = _C["grid"]  # #e5dfd6
    _MUTED_COLOR = _C["text_muted"]  # #888888
    _EDGE_COLOR = "white"
    _TRENDLINE_COLOR = _C["teal_dark"]  # #264653
except (ImportError, KeyError):
    _ANTHROPIC_COLOR = "#d97757"
    _OPENAI_COLOR = "#18a683"
    _GOOGLE_COLOR = "#4285F4"
    _OTHER_COLOR = "#888888"
    _GRID_COLOR = "grey"
    _MUTED_COLOR = "grey"
    _EDGE_COLOR = "black"
    _TRENDLINE_COLOR = "blue"

# =============================================================================
# METR config construction
# =============================================================================


def _build_plot_params(agents: list[str]) -> dict:
    """Build METR PlotParams config programmatically for our models."""
    # Try loading METR's params.yaml as base

    # Build agent styling with explicit per-model markers and provider colors.
    # Models are grouped by provider in the legend for visual clarity.

    def _classify(agent):
        a = agent.lower()
        if any(k in a for k in ("claude", "opus", "sonnet", "haiku")):
            return "anthropic"
        if any(k in a for k in ("gpt", "o3", "o4", "o1")):
            return "openai"
        if "gemini" in a:
            return "google"
        return "other"

    _PROVIDER_COLORS = {
        "anthropic": _ANTHROPIC_COLOR,
        "openai": _OPENAI_COLOR,
        "google": _GOOGLE_COLOR,
        "other": _OTHER_COLOR,
    }

    # Each provider gets a distinct marker family.
    # Within each provider, markers cycle so adjacent models differ.
    _PROVIDER_MARKERS = {
        "anthropic": ["o", "D", "p", "h"],
        "openai": ["s", "^", "v", "<", ">", "P"],
        "google": ["*", "X"],
        "other": ["d", "H", "8"],
    }
    _provider_iters = {k: iter(v) for k, v in _PROVIDER_MARKERS.items()}

    agent_styling = {}
    for agent in agents:
        provider = _classify(agent)
        color = _PROVIDER_COLORS[provider]
        marker = next(_provider_iters[provider], "o")
        agent_styling[agent] = {
            "lab_color": color,
            "marker": marker,
            "unique_color": color,
        }

    agent_styling["default"] = {
        "lab_color": _MUTED_COLOR,
        "marker": "o",
        "unique_color": _MUTED_COLOR,
    }
    plot_params["agent_styling"] = agent_styling

    # Sort legend by provider, then chronologically within provider.
    # This groups Anthropic models together, OpenAI together, etc.
    _PROVIDER_ORDER = {"anthropic": 0, "openai": 1, "google": 2, "other": 3}
    sorted_agents = sorted(
        agents,
        key=lambda a: (_PROVIDER_ORDER.get(_classify(a), 9), agents.index(a)),
    )
    plot_params["legend_order"] = sorted_agents

    return plot_params


def _build_script_params(
    success_percent: int = 50,
    title: str | None = None,
    exclude_agents: list[str] | None = None,
    trendline_fit_type: str = "exponential",
) -> dict:
    """Build METR ScriptParams for plot_horizon_graph."""
    if title is None:
        title = f"P{success_percent} Time Horizons vs Release Date"
    return {
        "parameter_group_name": "cth",
        "lower_y_lim": 0.0666,  # ~4 seconds
        "upper_y_lim": 3840,  # 64 hours
        "exclude": [],
        "title": title,
        "subtitle": "",
        "weighting": "invsqrt_task_weight",
        "include_task_distribution": "none",
        "weight_key": "invsqrt_task_weight",
        "trendlines": []
        if trendline_fit_type == "none"
        else [
            {
                "fit_type": trendline_fit_type,
                "after_date": "2019-01-01",
                "color": _TRENDLINE_COLOR,
                "line_start_date": "2019-01-01",
                "line_end_date": "2027-06-01",
                "display_r_squared": True,
                "data_file": None,
                "styling": None,
                "caption": None,
                "skip_annotation": True,
            }
        ],
        "exclude_agents": exclude_agents or [],
        "xlabel": "Model release date",
        "ylabel": f"Task time that model completes with {success_percent}% success rate",
        "show_y_label": True,
        "ax_label_fontsize": 14,
        "title_fontsize": 16,
        "legend_fontsize": 11,
        "annotation_fontsize": 14,
        "legend_frameon": True,
        "show_watermark": False,
        "show_example_tasks": False,
        "show_minor_xticks": False,
        "show_grid": True,
    }


# =============================================================================
# Data loading
# =============================================================================


def merge_legacy_models(
    campaign_data: dict[str, dict[str, pd.DataFrame]],
    best_available_times: dict[str, tuple[float, str]] | None = None,
    legacy_models: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Extend campaign_data with legacy model runs from the June 2025 study."""
    if legacy_models is None:
        legacy_models = LEGACY_MODELS

    result = dict(campaign_data)

    for cfg in legacy_models:
        alias = cfg["alias"]
        runs = load_legacy_runs(
            alias_filter=cfg["alias_filter"],
            agent=cfg["agent"],
            alias=alias,
        )
        if runs.empty:
            continue

        if best_available_times:
            runs_best, runs_human = derive_human_variants(runs, best_available_times)
        else:
            runs_best = runs.copy()
            runs_human = pd.DataFrame(columns=runs.columns)

        result[alias] = {"runs": runs, "runs_best": runs_best, "runs_human": runs_human}

    return result


def _concat_runs(
    model_data: dict[str, dict[str, pd.DataFrame]],
    time_source: str,
    token_budget: int | None = None,
) -> pd.DataFrame:
    """Concatenate runs from all models, using alias as the agent identifier.

    If token_budget is set, only include runs with total_tokens <= token_budget.
    Scores for excluded runs are set to 0 (failed) — they didn't solve it
    within the budget. Weights are renormalized after filtering.
    """
    frames = []
    for alias, data in model_data.items():
        df = data.get(time_source)
        if df is None or df.empty:
            continue
        frame = df.copy()
        frame["agent"] = alias
        if "run_id" not in frame.columns:
            frame["run_id"] = frame["task_id"] + "_" + alias

        if token_budget is not None and "total_tokens" in frame.columns:
            # Runs over budget count as failures (score=0), not excluded
            over_budget = frame["total_tokens"] > token_budget
            frame.loc[over_budget, "score_binarized"] = 0

        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Agent summaries
# =============================================================================


def build_agent_summaries(
    model_data: dict[str, dict[str, pd.DataFrame]],
    time_source: str = "runs",
    regularization: float = _WRANGLE_PARAMS["regularization"],
    release_dates: dict[str, str] | None = None,
    bootstrap_results: pd.DataFrame | None = None,
    token_budget: int | None = None,
) -> pd.DataFrame:
    """Fit IRT logistic curves for all models via METR's agent_regression().

    If token_budget is set, runs over that budget are scored as failures.
    """
    if release_dates is None:
        release_dates = RELEASE_DATES

    all_runs = _concat_runs(model_data, time_source, token_budget=token_budget)
    if all_runs.empty:
        return pd.DataFrame()

    results = []
    for agent_name, agent_df in all_runs.groupby("agent"):
        weights = agent_df["invsqrt_task_weight"].values
        regression = agent_regression(
            x=agent_df["human_minutes"].values,
            y=agent_df["score_binarized"].values,
            weights=weights,
            agent_name=agent_name,
            regularization=regularization,
            success_percents=_WRANGLE_PARAMS["success_percents"],
            confidence_level=_WRANGLE_PARAMS["confidence_level"],
            bootstrap_results=bootstrap_results,
            include_empirical_rates=False,
            ensure_weights_sum_to_1=True,
        )
        regression["agent"] = agent_name
        results.append(regression)

    summaries = pd.DataFrame([s.to_dict() for s in results])
    summaries["release_date"] = pd.to_datetime(summaries["agent"].map(release_dates))
    return summaries


# =============================================================================
# Bootstrap
# =============================================================================


def run_bootstrap(
    model_data: dict[str, dict[str, pd.DataFrame]],
    time_source: str = "runs",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    regularization: float = _WRANGLE_PARAMS["regularization"],
    token_budget: int | None = None,
) -> pd.DataFrame:
    """Run hierarchical bootstrap for P50/P80 CIs via METR's code."""
    all_runs = _concat_runs(model_data, time_source, token_budget=token_budget)
    if all_runs.empty:
        return pd.DataFrame()

    return compute_bootstrap_regressions(
        data=all_runs,
        categories=["task_family", "task_id"],
        n_bootstrap=n_bootstrap,
        regularization=regularization,
        weights_col="invsqrt_task_weight",
        success_percents=_WRANGLE_PARAMS["success_percents"],
        score_col="score_binarized",
    )


# =============================================================================
# Plotting — thin wrappers around METR's plot_horizon_graph
# =============================================================================


def plot_horizon_graph(
    agent_summaries: pd.DataFrame,
    runs_df: pd.DataFrame | None = None,
    release_dates: dict[str, str] | None = None,
    non_frontier: set[str] | None = None,
    success_percent: int = 50,
    title: str | None = None,
    y_scale: str = "log",
    bootstrap_results: pd.DataFrame | None = None,
    fig=None,
    x_lim_start: str | None = None,
    x_lim_end: str | None = None,
    trendline_fit_type: str = "exponential",
) -> None:
    """Plot P50 or P80 horizon graph using METR's plot_horizon_graph.

    Constructs METR's config objects programmatically and delegates to
    their plotting code for correct error bars, trendlines, and formatting.
    """
    if release_dates is None:
        release_dates = RELEASE_DATES
    if non_frontier is None:
        raise ValueError(
            "non_frontier must be passed explicitly. Use is_sota from "
            "model_summaries instead of the deprecated NON_FRONTIER_AGENTS."
        )

    agents = agent_summaries["agent"].tolist()
    plot_params = _build_plot_params(agents)
    script_params = _build_script_params(
        success_percent=success_percent,
        title=title,
        exclude_agents=list(non_frontier),
        trendline_fit_type=trendline_fit_type,
    )

    summaries = agent_summaries.copy()
    # Ensure release_date is Timestamp (build_agent_summaries already does this,
    # but re-mapping here in case caller passed custom release_dates)
    if release_dates:
        summaries["release_date"] = pd.to_datetime(
            summaries["agent"].map(release_dates)
        )

    # Empty runs_df if not provided (no task distribution histogram)
    if runs_df is None:
        runs_df = pd.DataFrame(
            columns=["human_minutes", "task_family", "invsqrt_task_weight"]
        )

    # METR's _process_agent_summaries maps release_dates["date"] onto the
    # DataFrame and compares with pd.Timestamp().date(). The values must be
    # date-comparable, not strings.
    rd_dates = {k: pd.Timestamp(v).date() for k, v in release_dates.items()}

    _metr_plot_horizon_graph(
        plot_params=plot_params,
        all_agent_summaries=summaries,
        runs_df=runs_df,
        release_dates={"date": rd_dates},
        lower_y_lim=script_params["lower_y_lim"],
        x_lim_start=x_lim_start or "2019-06-01",
        x_lim_end=x_lim_end or "2026-07-01",
        subtitle=script_params["subtitle"],
        title=script_params.get("title", f"P{success_percent} Time Horizons"),
        weight_key=script_params["weight_key"],
        exclude_agents=script_params["exclude_agents"],
        success_percent=success_percent,
        script_params=script_params,
        trendlines=script_params["trendlines"],
        upper_y_lim=script_params["upper_y_lim"],
        fig=fig,
        y_scale=y_scale,
    )

    # Add bootstrap CI band if available
    if bootstrap_results is not None and fig is not None:
        try:
            # compute_bootstrap_confidence_region hardcodes _p50 column
            # filtering. For P80, drop _p50 columns and rename _p80 -> _p50.
            boot = bootstrap_results
            if success_percent != 50:
                suffix_from = f"_p{success_percent}"
                # Drop existing _p50 columns to avoid duplicates after rename
                p50_cols = [c for c in boot.columns if c.endswith("_p50")]
                boot = boot.drop(columns=p50_cols)
                rename_map = {
                    c: c.replace(suffix_from, "_p50")
                    for c in boot.columns
                    if c.endswith(suffix_from)
                }
                boot = boot.rename(columns=rename_map)
                # Also need p50 in summaries for point estimate
                sum_for_ci = summaries.copy()
                sum_for_ci["p50"] = sum_for_ci[f"p{success_percent}"]
            else:
                sum_for_ci = summaries

            frontier_summaries = sum_for_ci[~sum_for_ci["agent"].isin(non_frontier)]
            if len(frontier_summaries) >= 2:
                earliest = (
                    pd.to_datetime(frontier_summaries["release_date"])
                    .min()
                    .strftime("%Y-%m-%d")
                )
                latest = (
                    pd.to_datetime(frontier_summaries["release_date"])
                    .max()
                    .strftime("%Y-%m-%d")
                )

                dt_stats, time_points, lower, upper = (
                    compute_bootstrap_confidence_region(
                        agent_summaries=frontier_summaries,
                        bootstrap_results=boot,
                        release_dates={
                            "date": {
                                k: pd.Timestamp(v).date()
                                for k, v in release_dates.items()
                            }
                        },
                        after_date=earliest,
                        sota_before_date=latest,
                        trendline_end_date="2027-01-01",
                        confidence_level=0.95,
                        filter_sota=False,
                    )
                )
                ax = fig.axes[0]
                ax.fill_between(
                    time_points,
                    lower,
                    upper,
                    alpha=0.15,
                    color=_TRENDLINE_COLOR,
                    zorder=1,
                )
        except Exception as e:
            import traceback

            print(f"Bootstrap CI band failed (P{success_percent}): {e}")
            traceback.print_exc()


def plot_trendline_dashboard(
    model_data: dict[str, dict[str, pd.DataFrame]],
    time_source: str = "runs",
    non_frontier: set[str] | None = None,
    release_dates: dict[str, str] | None = None,
    n_bootstrap: int = 200,
    token_budgets: list[int | None] | None = None,
) -> list:
    """Generate trendline plots: P50/P80 × log/linear × token budgets.

    Each token budget gets a group heading. METR's plot_horizon_graph
    manages its own figure, so this returns a list of figures.

    Args:
        token_budgets: list of token limits, e.g. [None, 1_000_000].
            None means no limit (full 2M budget).
    """
    import matplotlib.pyplot as plt

    if token_budgets is None:
        token_budgets = [None]

    source_label = {
        "runs": "Model-Estimated",
        "runs_best": "Best Available",
        "runs_human": "Human-Derived",
    }.get(time_source, time_source)

    figs = []
    for budget in token_budgets:
        budget_label = (
            f"{budget // 1_000_000}M tokens" if budget else "2M tokens (full budget)"
        )
        print(f"\n{'='*60}")
        print(f"  {source_label} — {budget_label}")
        print(f"{'='*60}")

        boot = run_bootstrap(
            model_data,
            time_source=time_source,
            n_bootstrap=n_bootstrap,
            token_budget=budget,
        )
        summaries = build_agent_summaries(
            model_data,
            time_source,
            release_dates=release_dates,
            bootstrap_results=boot,
            token_budget=budget,
        )

        for pct in [50, 80]:
            for scale in ["log", "linear"]:
                fig = plt.figure(figsize=(10, 7))
                fig.add_subplot(111)
                plot_horizon_graph(
                    summaries,
                    release_dates=release_dates,
                    non_frontier=non_frontier,
                    success_percent=pct,
                    title=f"P{pct} — {source_label} — {budget_label} ({scale})",
                    y_scale=scale,
                    bootstrap_results=boot,
                    fig=fig,
                )
                figs.append(fig)

            # Zoomed views (log scale only)
            for zoom_label, zoom_start in [
                ("2024+", "2024-01-01"),
                ("6-month", "2025-09-01"),
            ]:
                fig_zoom = plt.figure(figsize=(10, 7))
                fig_zoom.add_subplot(111)
                plot_horizon_graph(
                    summaries,
                    release_dates=release_dates,
                    non_frontier=non_frontier,
                    success_percent=pct,
                    title=f"P{pct} — {source_label} — {budget_label} ({zoom_label} zoom)",
                    y_scale="log",
                    bootstrap_results=boot,
                    fig=fig_zoom,
                    x_lim_start=zoom_start,
                    x_lim_end="2026-07-01",
                )
                plt.show()
                figs.append(fig_zoom)

    return figs
