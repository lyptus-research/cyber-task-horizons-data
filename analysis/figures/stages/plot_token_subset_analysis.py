"""Token budget sensitivity by task subset.

Two panels:
  Left: Accuracy vs token budget for different task subsets
  Right: Cost per success vs token budget for different task subsets

Uses GPT-5.3 Codex as the reference model (highest capability + 10M data).

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    load_params,
    save_chart_json,
    save_png,
)
from lib.lyptus_style import apply_style, COLORS  # noqa: E402

apply_style()

MODEL = "GPT-5.3 Codex"
PRICE_PER_TOKEN = 4.5 / 1e6  # GPT-5.1 Codex tier

BUDGETS = [
    50_000,
    100_000,
    200_000,
    500_000,
    750_000,
    1_000_000,
    1_250_000,
    1_500_000,
    1_750_000,
    2_000_000,
    3_000_000,
    4_000_000,
    5_000_000,
    6_000_000,
    7_000_000,
    8_000_000,
    9_000_000,
    10_000_000,
]

SUBSETS = [
    {"key": "all", "label": "All tasks", "filter": lambda df: df},
    {
        "key": "cybergym",
        "label": "CyberGym",
        "filter": lambda df: df[df["task_family"] == "cybergym"],
    },
    {
        "key": "hard_30m",
        "label": ">30m tasks",
        "filter": lambda df: df[df["human_minutes"] > 30],
    },
    {
        "key": "hard_2h",
        "label": ">2h tasks",
        "filter": lambda df: df[df["human_minutes"] > 120],
    },
]


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute accuracy and cost at each budget for each subset."""
    from lib.data import assemble_runs

    mr = pd.read_parquet(
        Path(args.model_runs)
        if Path(args.model_runs).is_absolute()
        else _NOTEBOOKS_DIR / args.model_runs
    )
    td = pd.read_parquet(
        Path(args.task_difficulties)
        if Path(args.task_difficulties).is_absolute()
        else _NOTEBOOKS_DIR / args.task_difficulties
    )
    runs_df = assemble_runs(mr, td, args.difficulty_col)
    model_runs = runs_df[runs_df["alias"] == MODEL].copy()

    if len(model_runs) == 0:
        return {
            "chart_type": "tokenSubsetAnalysis",
            "version": 1,
            "data": {"model": MODEL, "subsets": [], "study_budget_m": 2.0},
            "options": {"title": f"{MODEL} token budget analysis by task subset"},
        }

    # Load 10M re-run data if available
    task_10m = {}
    if args.ten_m_cache:
        import pickle

        cache_path = (
            Path(args.ten_m_cache)
            if Path(args.ten_m_cache).is_absolute()
            else _NOTEBOOKS_DIR / args.ten_m_cache
        )
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                ten_m_data = pickle.load(f)
            for s in ten_m_data:
                task_10m[s["task_id"]] = {
                    "score": s["score"],
                    "tokens": s["total_tokens"],
                }
            print(f"Loaded {len(task_10m)} 10M re-run samples")

    # Compute accuracy and cost-per-success at each budget for each subset
    chart_subsets = []

    for subset_def in SUBSETS:
        subset_runs = subset_def["filter"](model_runs)
        n_tasks = subset_runs["task_id"].nunique()

        budget_points = []
        for b in BUDGETS:
            # For budgets <= 2M, use the standard approach
            if b <= 2_000_000:
                successes = int(
                    (
                        (subset_runs["score_binarized"] == 1)
                        & (subset_runs["total_tokens"] <= b)
                    ).sum()
                )
                tokens_used = subset_runs["total_tokens"].clip(upper=b)
            else:
                # For extended budgets, incorporate 10M re-run data
                successes = 0
                tokens_used_list = []
                for _, row in subset_runs.iterrows():
                    tok = min(row["total_tokens"], b)
                    tokens_used_list.append(tok)
                    if row["score_binarized"] == 1 and row["total_tokens"] <= b:
                        successes += 1
                    elif row["score_binarized"] == 0 and row["task_id"] in task_10m:
                        r10 = task_10m[row["task_id"]]
                        if r10["score"] > 0 and r10["tokens"] <= b:
                            successes += 1
                            extra = min(r10["tokens"], b) - row["total_tokens"]
                            if extra > 0:
                                tokens_used_list[-1] += extra
                tokens_used = pd.Series(tokens_used_list)

            total = len(subset_runs)
            accuracy = successes / total * 100 if total > 0 else 0
            total_cost = float((tokens_used * PRICE_PER_TOKEN).sum())
            cps = total_cost / successes if successes > 0 else float("inf")

            budget_points.append(
                {
                    "budget": b,
                    "budget_m": b / 1e6,
                    "budget_label": f"{b // 1000}K"
                    if b < 1_000_000
                    else f"{b / 1_000_000:.0f}M",
                    "accuracy": round(accuracy, 1),
                    "cost_per_success": round(cps, 2) if np.isfinite(cps) else None,
                    "successes": successes,
                    "total": total,
                }
            )

        chart_subsets.append(
            {
                "key": subset_def["key"],
                "label": f"{subset_def['label']} (n={n_tasks})",
                "n_tasks": n_tasks,
                "points": budget_points,
            }
        )

    return {
        "chart_type": "tokenSubsetAnalysis",
        "version": 1,
        "data": {
            "model": MODEL,
            "subsets": chart_subsets,
            "study_budget_m": 2.0,
        },
        "options": {"title": f"{MODEL} token budget analysis by task subset"},
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render token budget sensitivity plots from chart JSON data."""
    chart_subsets = chart_data["data"]["subsets"]
    model = chart_data["data"]["model"]
    study_budget_m = chart_data["data"].get("study_budget_m", 2.0)

    if not chart_subsets:
        print(f"No data for {model}, skipping render")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    subset_colors = [
        COLORS["teal_dark"],
        COLORS["coral"],
        COLORS["teal_light"],
        COLORS["gold"],
    ]
    subset_styles = ["-", "-", "-", "-"]

    for i, subset in enumerate(chart_subsets):
        budgets_m = [p["budget_m"] for p in subset["points"]]
        accuracies = [p["accuracy"] for p in subset["points"]]
        costs = [p["cost_per_success"] for p in subset["points"]]

        ax1.plot(
            budgets_m,
            accuracies,
            marker="o",
            color=subset_colors[i],
            linewidth=2,
            markersize=5,
            label=subset["label"],
            linestyle=subset_styles[i],
        )

        valid_costs = [(b, c) for b, c in zip(budgets_m, costs) if c is not None]
        if valid_costs:
            ax2.plot(
                [x[0] for x in valid_costs],
                [x[1] for x in valid_costs],
                marker="o",
                color=subset_colors[i],
                linewidth=2,
                markersize=5,
                label=subset["label"],
                linestyle=subset_styles[i],
            )

    # Study budget line
    for ax in [ax1, ax2]:
        ax.axvline(
            study_budget_m, color=COLORS["coral"], linestyle=":", alpha=0.5, linewidth=1
        )
        ax.text(
            study_budget_m,
            ax.get_ylim()[0],
            "Study budget",
            color=COLORS["coral"],
            fontsize=8,
            alpha=0.7,
            ha="right",
            va="bottom",
            rotation=90,
        )

    ax1.set_xlabel("Token budget")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"{model} accuracy by task subset")
    ax1.set_xscale("log")
    ax1.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax1.set_xticklabels(["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M"])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    ax2.set_xlabel("Token budget")
    ax2.set_ylabel("Cost per success ($)")
    ax2.set_title(f"{model} cost per success by task subset")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 50)
    ax2.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax2.set_xticklabels(["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M"])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot token budget sensitivity by task subset")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument("--ten-m-cache", default=None, help="Path to 10M re-run cache")
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
