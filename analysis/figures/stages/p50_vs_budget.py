"""Stage 3c: P50 horizon vs token budget curves.

Uses assemble_runs() to join model_runs with difficulty source.
Produces a two-panel figure: P50 horizon (left) and accuracy (right)
vs token budget for all models.

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
    coerce_floats,
    load_params,
    save_chart_json,
    save_png,
)
from lib.data import assemble_runs  # noqa: E402
from lib.irt import p50_vs_token_budget  # noqa: E402


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load runs, compute P50 and accuracy at each budget, return chart_data dict."""
    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)
    runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)

    campaign_data = {
        alias: {"runs": group.copy()} for alias, group in runs_df.groupby("alias")
    }

    chart_models = []
    for alias, data in campaign_data.items():
        result = p50_vs_token_budget(data["runs"])
        budgets_k = [float(b / 1000) for b in result["budget"]]
        p50_minutes = [round(float(v), 2) for v in result["p50_minutes"]]
        accuracy_pct = [round(float(v) * 100, 2) for v in result["accuracy"]]
        chart_models.append(
            {
                "alias": alias,
                "budgets_k": budgets_k,
                "p50_minutes": p50_minutes,
                "accuracy_pct": accuracy_pct,
            }
        )

    # Print summary table
    for alias, data in campaign_data.items():
        result = p50_vs_token_budget(data["runs"])
        print(f"\n{alias}:")
        for _, row in result.iterrows():
            print(
                f"  {row['budget']/1000:.0f}k tokens: "
                f"P50 = {row['p50_minutes']:.1f} min, "
                f"accuracy = {row['accuracy']:.1%}"
            )

    chart_data = {
        "chart_type": "p50VsBudget",
        "version": 1,
        "data": {"models": chart_models},
        "options": {"title": "P50 Time Horizon vs Token Budget"},
    }

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render two-panel P50 vs budget figure from chart_data."""
    import matplotlib.pyplot as plt

    models = chart_data["data"]["models"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for m in models:
        budgets_k = coerce_floats(m["budgets_k"])
        p50 = coerce_floats(m["p50_minutes"])
        acc = coerce_floats(m["accuracy_pct"])

        ax1.plot(budgets_k, p50, marker="o", label=m["alias"], linewidth=2)
        ax2.plot(budgets_k, acc, marker="o", label=m["alias"], linewidth=2)

    ax1.set_xlabel("Token budget (thousands)")
    ax1.set_ylabel("P50 horizon (minutes)")
    ax1.set_title("P50 Time Horizon vs Token Budget")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")

    ax2.set_xlabel("Token budget (thousands)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs Token Budget")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    save_png(fig, output, params)
    plt.close(fig)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate P50 vs token budget figure")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
