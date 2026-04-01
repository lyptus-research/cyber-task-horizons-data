"""Stage: Side-by-side IRT S-curve for a frontier model, headline vs completions-only.

Uses assemble_runs() with best_available_minutes (headline) and
completion_minutes (completions-only) from task_difficulties.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
from lib.data import assemble_runs  # noqa: E402
from lib.irt import compute_scurve_data  # noqa: E402


# =============================================================================
# Compute: all statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute IRT curves for headline vs completions-only, return chart_data."""
    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)

    headline_df = assemble_runs(model_runs, task_diff, "best_available_minutes")
    completions_df = assemble_runs(model_runs, task_diff, "completion_minutes")

    models = [m.strip() for m in args.model.split(",")]

    panels = []
    for model in models:
        hl_model = headline_df[headline_df["alias"] == model]
        co_model = completions_df[completions_df["alias"] == model]
        hl_tasks = hl_model["task_id"].nunique()
        co_tasks = co_model["task_id"].nunique()
        panels.append((hl_model, f"{model} — Headline ({hl_tasks} tasks)"))
        panels.append((co_model, f"{model} — Completions only ({co_tasks} tasks)"))

    # Compute shared bin edges across all panels
    all_log2 = np.concatenate([df["log2_human_minutes"].values for df, _ in panels])
    global_bin_edges = np.arange(
        np.floor(all_log2.min()) - 0.5,
        np.ceil(all_log2.max()) + 1.5,
        1.0,
    )

    chart_models = []
    for df, label in panels:
        panel_data = compute_scurve_data(df, global_bin_edges)
        chart_models.append(
            {
                "alias": label,
                "provider": "comparison",
                "release_date": "",
                "release_year": "",
                **panel_data,
            }
        )

    return {
        "chart_type": "irtScurve",
        "version": 1,
        "data": {
            "models": chart_models,
            "global_bin_edges": [round(float(e), 2) for e in global_bin_edges],
        },
        "options": {
            "title": "Headline vs study completions only",
            "defaultModels": [label for _, label in panels],
            "ncols": 2,
            "noControls": True,
        },
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render side-by-side IRT S-curves from chart JSON data.

    Reproduces the same layout as plot_scurve_grid with ncols=2, by
    re-assembling the data and calling the existing grid renderer. The
    chart_data is authoritative but the matplotlib rendering delegates
    to the lib/irt.py grid function for visual fidelity.
    """
    # Re-assemble panels from chart_data for plot_scurve_grid
    # plot_scurve_grid needs DataFrames, but render_png should work from JSON.
    # We reconstruct minimal DataFrames from the chart JSON bin data.
    import matplotlib.pyplot as plt

    models = chart_data["data"]["models"]
    global_bin_edges = chart_data["data"]["global_bin_edges"]
    ncols = chart_data["options"].get("ncols", 2)
    suptitle = chart_data["options"]["title"]

    try:
        from lib.lyptus_style import FONT_SANS
    except (ImportError, KeyError):
        FONT_SANS = "Helvetica Neue"

    from lib.irt import _TICK_LOG2, _TICK_LABELS

    n = len(models)
    import math

    nrows = math.ceil(n / ncols)
    col_width = 6
    row_height = 3.5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(col_width * ncols, row_height * nrows),
        squeeze=False,
        sharey=True,
    )

    x_lo = global_bin_edges[0]
    x_hi = global_bin_edges[-1]

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # Extract data from chart JSON
        bin_centers = model["bin_centers"]
        empirical_rates = model["empirical_rates"]
        bin_counts = model["bin_counts"]
        standard_errors = model.get("standard_errors", [None] * len(bin_centers))
        curve_x = model["curve_x"]
        curve_y = model["curve_y"]
        p50_label = model.get("p50_label", "")
        min_n = model.get("min_n", 5)
        coef = model.get("coef", 0)
        _ = model.get("intercept", 0)  # kept in JSON for JS charts
        alias = model["alias"]

        # Draw empirical rate bars
        for i, (center, rate, count) in enumerate(
            zip(bin_centers, empirical_rates, bin_counts)
        ):
            if rate is None:
                continue
            alpha = 0.3 if count <= min_n else 0.55
            ax.bar(
                center,
                rate * 100,
                width=0.8,
                color="#2a9d8f",
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5,
            )
            # Error bars
            se = standard_errors[i] if i < len(standard_errors) else None
            if se is not None and count > min_n:
                ax.errorbar(
                    center,
                    rate * 100,
                    yerr=se * 100,
                    fmt="none",
                    ecolor="#264653",
                    capsize=3,
                    capthick=1,
                    alpha=0.5,
                )

        # Draw S-curve
        ax.plot(
            curve_x,
            curve_y,
            color="#e76f51",
            linewidth=2.5,
            zorder=4,
        )

        # P50 vertical line
        if coef != 0 and float(coef) != float("-inf"):
            p50_log2 = model.get("p50_log2", 0)
            if p50_log2 is not None and p50_log2 > float("-inf"):
                ax.axvline(
                    p50_log2,
                    color="#e76f51",
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1,
                )

        # Title with P50 annotation
        title_text = alias
        if p50_label:
            title_text += f"\nP50 = {p50_label}"
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(axis="both", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Axis labels
        xlabel = "Human time (log\u2082 scale)"
        if r == nrows - 1 or (idx + ncols >= n):
            ax.set_xlabel(xlabel, fontsize=10, fontfamily=FONT_SANS)
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Success rate (%)", fontsize=10, fontfamily=FONT_SANS)
        else:
            ax.set_ylabel("")

    # Apply clean x-ticks
    visible_ticks = [
        (pos, label)
        for pos, label in zip(_TICK_LOG2, _TICK_LABELS)
        if x_lo <= pos <= x_hi
    ]
    if visible_ticks:
        tick_positions, tick_labels_list = zip(*visible_ticks)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(x_lo, x_hi)
                ax.set_xticks(list(tick_positions))
                ax.set_xticklabels(
                    list(tick_labels_list), fontsize=10, fontfamily=FONT_SANS
                )
                ax.tick_params(axis="y", labelsize=10)
                for label in ax.get_yticklabels():
                    label.set_fontfamily(FONT_SANS)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot headline vs completions-only S-curve comparison")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--model", default="GPT-5.3 Codex", help="Comma-separated model aliases"
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
