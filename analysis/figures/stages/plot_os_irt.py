"""Stage: Open-source model IRT comparison figures.

Produces four supplementary figures comparing open-weight models against
the closed-source frontier using IRT-based analysis from lib/irt.py:

  - trendline_buffer:     Two-panel P50 trendline + per-benchmark buffer
  - scurve_overlay:       IRT S-curves for selected models overlaid
  - gap_by_difficulty:    Frontier vs OS accuracy gap by benchmark
  - benchmark_comparison: Grouped bar chart, models x benchmarks

Each figure is produced by a separate DVC stage invocation using the
--figure selector. compute() dispatches to the appropriate figure type
and returns chart_data with figure-specific structure.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
from pathlib import Path

import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    build_campaign_data,
    load_params,
    save_chart_json,
    save_png,
)
from figures.stages._common_data import load_release_dates  # noqa: E402
from lib.data import assemble_runs  # noqa: E402
from lib.irt import (  # noqa: E402
    plot_gap_by_difficulty,
    plot_os_trendline_buffer,
    plot_per_benchmark_comparison,
    plot_scurve_overlay,
)

FIGURE_CHOICES = [
    "trendline_buffer",
    "scurve_overlay",
    "gap_by_difficulty",
    "benchmark_comparison",
]


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, run the appropriate IRT analysis, return chart_data dict.

    The chart_data structure varies by figure type:
    - trendline_buffer: model points + buffer values + trendline data
    - scurve_overlay: per-model IRT curve points (x=log2_minutes, y=P(success))
    - gap_by_difficulty: per-benchmark accuracy gap (frontier vs OS)
    - benchmark_comparison: per-model per-benchmark accuracy matrix
    """
    os_params = params.get("os_models", {})
    irt_params = params.get("irt", {})
    regularization = irt_params.get("regularization")

    open_weight = os_params.get("open_weight", ["GLM-5", "DeepSeek V3.1"])
    comparison_set = os_params.get(
        "comparison_set",
        ["GLM-5", "DeepSeek V3.1", "o3", "Opus 4", "GPT-5.3 Codex", "Opus 4.6"],
    )
    frontier_for_gap = os_params.get("frontier_for_gap", ["GPT-5.3 Codex", "Opus 4.6"])

    # Load parquets
    model_runs = pd.read_parquet(args.model_runs)
    task_difficulties = pd.read_parquet(args.task_difficulties)
    assembled = assemble_runs(model_runs, task_difficulties, args.difficulty_col)
    campaign_data = build_campaign_data(model_runs, assembled)
    release_dates = load_release_dates()

    figure = args.figure

    if figure == "trendline_buffer":
        if args.summaries is None:
            raise ValueError("--summaries required for trendline_buffer figure")
        summaries = pd.read_parquet(args.summaries)
        sota_agents = set(
            summaries[summaries["is_sota"].fillna(False)]["agent"].tolist()
        )
        open_set = set(open_weight)
        closed_sota = sorted(
            (a for a in sota_agents if a not in open_set),
            key=lambda a: release_dates.get(a, "9999"),
        )

        # Extract data for chart JSON
        chart_models = []
        for alias in list(closed_sota) + list(open_weight):
            if alias not in campaign_data:
                continue
            rd = release_dates.get(alias, "")
            chart_models.append(
                {
                    "name": alias,
                    "release": rd,
                    "is_os": alias in open_set,
                    "is_sota": alias in sota_agents,
                }
            )

        return {
            "chart_type": "osTrendlineBuffer",
            "version": 1,
            "data": {
                "models": chart_models,
                "closed_sota": closed_sota,
                "open_models": open_weight,
            },
            "options": {
                "title": "OS Model Trendline Buffer",
                "regularization": regularization,
            },
            # Store enough info to reproduce the figure from lib/irt functions
            "_figure": figure,
            "_campaign_data": campaign_data,
            "_release_dates": release_dates,
            "_regularization": regularization,
        }

    elif figure == "scurve_overlay":
        return {
            "chart_type": "scurveOverlay",
            "version": 1,
            "data": {"models": comparison_set},
            "options": {
                "title": "Open-Source vs Frontier: IRT S-Curve Comparison",
                "regularization": regularization,
            },
            "_figure": figure,
            "_campaign_data": campaign_data,
            "_regularization": regularization,
        }

    elif figure == "gap_by_difficulty":
        return {
            "chart_type": "gapByDifficulty",
            "version": 1,
            "data": {
                "os_models": open_weight,
                "frontier_models": frontier_for_gap,
            },
            "options": {"title": "Frontier vs OS Accuracy Gap by Benchmark"},
            "_figure": figure,
            "_campaign_data": campaign_data,
        }

    elif figure == "benchmark_comparison":
        return {
            "chart_type": "benchmarkComparison",
            "version": 1,
            "data": {"models": comparison_set},
            "options": {"title": "Per-Benchmark Model Comparison"},
            "_figure": figure,
            "_campaign_data": campaign_data,
        }

    raise ValueError(f"Unknown figure type: {figure}")


# =============================================================================
# Render: matplotlib figure from chart_data dict
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render the appropriate figure from chart_data.

    These figures delegate to lib/irt.py plotting functions which return
    matplotlib Figure objects. The chart_data carries the campaign_data
    and parameters needed for the call.
    """
    figure = chart_data["_figure"]
    campaign_data = chart_data["_campaign_data"]

    fig = None

    if figure == "trendline_buffer":
        release_dates = chart_data["_release_dates"]
        regularization = chart_data.get("_regularization")
        closed_sota = chart_data["data"]["closed_sota"]
        open_models = chart_data["data"]["open_models"]
        fig = plot_os_trendline_buffer(
            campaign_data,
            closed_sota=closed_sota,
            open_models=open_models,
            release_dates=release_dates,
            regularization=regularization,
        )

    elif figure == "scurve_overlay":
        regularization = chart_data.get("_regularization")
        models = chart_data["data"]["models"]
        fig = plot_scurve_overlay(
            campaign_data,
            models=models,
            title=chart_data["options"]["title"],
            regularization=regularization,
        )

    elif figure == "gap_by_difficulty":
        fig = plot_gap_by_difficulty(
            campaign_data,
            os_models=chart_data["data"]["os_models"],
            frontier_models=chart_data["data"]["frontier_models"],
        )

    elif figure == "benchmark_comparison":
        fig = plot_per_benchmark_comparison(
            campaign_data,
            models=chart_data["data"]["models"],
        )

    if fig is not None:
        save_png(fig, output, params)
    else:
        print(f"Warning: {figure} produced no figure")


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate OS model IRT comparison figure")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument(
        "--summaries",
        default=None,
        help="model_summaries parquet (needed for trendline_buffer to derive closed_sota)",
    )
    parser.add_argument(
        "--figure",
        required=True,
        choices=FIGURE_CHOICES,
        help="Which figure to produce",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)

    # Save a cleaned version of chart_data (without internal keys)
    json_data = {k: v for k, v in chart_data.items() if not k.startswith("_")}
    save_chart_json(json_data, args.output)

    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
