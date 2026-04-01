"""Stage 3a: Generate IRT logistic fit grid figures.

One grid per difficulty source, showing per-model S-curves ordered by
release date. Uses assemble_runs() to join model_runs with the named
difficulty column.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
from pathlib import Path

import numpy as np
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
from lib.irt import compute_scurve_data  # noqa: E402
from lib.data import assemble_runs  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402


def _provider(alias):
    a = alias.lower()
    if any(k in a for k in ("claude", "opus", "sonnet", "haiku")):
        return "anthropic"
    if any(k in a for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in a:
        return "google"
    return "other"


# =============================================================================
# Compute: all data loading + statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, fit IRT curves for each model, return chart_data dict."""
    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)

    runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)

    exclude = set(params.get("irt_grid", {}).get("exclude_models", []))
    panels = [
        (group, alias)
        for alias, group in runs_df.groupby("alias")
        if alias not in exclude and len(group) >= 5
    ]

    release_dates = load_release_dates()
    source_label = args.title_label or args.difficulty_col.replace("_", " ").title()
    suptitle = f"IRT Logistic Fits \u2014 {source_label}"

    # Sort panels by release date (chronological order)
    sorted_panels = sorted(
        panels,
        key=lambda p: pd.to_datetime(release_dates.get(p[1], "2099-01-01")),
    )

    # Compute shared bin edges across all panels
    all_log2 = np.concatenate(
        [df["log2_human_minutes"].values for df, _ in sorted_panels]
    )
    global_bin_edges = np.arange(
        np.floor(all_log2.min()) - 0.5,
        np.ceil(all_log2.max()) + 1.5,
        1.0,
    )

    chart_models = []
    for df, alias in sorted_panels:
        panel_data = compute_scurve_data(df, global_bin_edges)
        rd = release_dates.get(alias, "")
        chart_models.append(
            {
                "alias": alias,
                "provider": _provider(alias),
                "release_date": rd,
                "release_year": rd[:4] if rd else "",
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
            "title": suptitle,
            "defaultModels": ["GPT-5.3 Codex", "Opus 4.6", "GLM-5", "Sonnet 4.6"],
        },
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render IRT S-curve grid from chart JSON data.

    Reproduces the same visual output as the old plot_scurve_grid() call:
    a grid of panels with shared axes, empirical histograms, and fitted
    S-curves, ordered by release date.
    """
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from lib.lyptus_style import COLORS as _C, FONT_SANS as _sans

        _bar_color = _C["teal"]
        _curve_color = _C["coral"]
    except (ImportError, KeyError):
        _sans = "Helvetica Neue"
        _bar_color = "#4a90d9"
        _curve_color = "#e74c3c"

    _bar_color_full = _bar_color + "d9"  # ~85% opacity
    _bar_color_pale = _bar_color + "40"  # ~25% opacity for low-n bins

    models = chart_data["data"]["models"]
    global_bin_edges = chart_data["data"]["global_bin_edges"]
    suptitle = chart_data["options"]["title"]

    ncols = 3
    col_width = 5.5
    row_height = 3.2
    n = len(models)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(col_width * ncols, row_height * nrows),
        squeeze=False,
        sharey=True,
    )

    x_lo = global_bin_edges[0]
    x_hi = global_bin_edges[-1]

    # Human-readable time axis ticks
    _TICK_LOG2 = [
        np.log2(v) for v in [1 / 60, 5 / 60, 15 / 60, 1, 5, 15, 60, 240, 960, 3840]
    ]
    _TICK_LABELS = ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "4h", "16h", "64h"]

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        alias = model["alias"]
        bin_centers = model["bin_centers"]
        empirical_rates = coerce_floats(model["empirical_rates"])
        bin_counts = model["bin_counts"]
        standard_errors = coerce_floats(model.get("standard_errors", []))
        min_n = model.get("min_n", 5)
        curve_x = coerce_floats(model["curve_x"])
        curve_y = coerce_floats(model["curve_y"])
        p50_label = model["p50_label"]
        p50_log2 = model["p50_log2"]

        # Bar colors: full opacity for bins with n > min_n, pale for low-n
        bar_colors = [
            _bar_color_full if cnt > min_n else _bar_color_pale for cnt in bin_counts
        ]

        # Plot empirical bars (rates as percentage)
        rates_pct = [r * 100 if r == r else 0 for r in empirical_rates]
        ax.bar(
            bin_centers,
            rates_pct,
            width=0.85,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # Error bars: 2*SE
        valid_se = []
        valid_centers = []
        valid_rates = []
        for bc, rate, se in zip(bin_centers, empirical_rates, standard_errors):
            if rate == rate and se == se and se > 0:  # not NaN
                valid_se.append(se * 200)  # 2*SE * 100 for percentage
                valid_centers.append(bc)
                valid_rates.append(rate * 100)
        if valid_se:
            ax.errorbar(
                valid_centers,
                valid_rates,
                yerr=valid_se,
                fmt="none",
                color=_bar_color,
                alpha=0.9,
                capsize=3,
                zorder=4,
            )

        # Per-bin n= labels
        for bc, rate, cnt in zip(bin_centers, empirical_rates, bin_counts):
            if cnt > 0 and rate == rate:  # not NaN
                ax.text(
                    bc,
                    rate * 100 + 3,
                    f"n={cnt}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontfamily=_sans,
                    color="#666",
                    alpha=0.75,
                )

        # Fitted logistic curve
        ax.plot(curve_x, curve_y, color=_curve_color, linewidth=2.5, zorder=3)

        # P50 vertical line
        if p50_log2 is not None and p50_log2 > -1e6:
            ax.axvline(
                p50_log2,
                color=_curve_color,
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                zorder=3,
            )

        ax.set_title(f"{alias} (P50 = {p50_label})", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", alpha=0.3)

        # Axis labels: only edge subplots
        xlabel = "Human time (log\u2082 scale)"
        if r == nrows - 1 or (idx + ncols >= n):
            ax.set_xlabel(xlabel, fontsize=10, fontfamily=_sans)
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Success rate (%)", fontsize=10, fontfamily=_sans)
        else:
            ax.set_ylabel("")

    # Apply clean x-ticks to all axes
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
                    list(tick_labels_list), fontsize=10, fontfamily=_sans
                )
                ax.tick_params(axis="y", labelsize=10)
                for label in ax.get_yticklabels():
                    label.set_fontfamily(_sans)

    # Hide unused axes in the last row
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    for model in models:
        alias = model["alias"]
        p50_log2 = model["p50_log2"]
        if p50_log2 is not None and p50_log2 > -1e6:
            p50 = 2**p50_log2
            print(f"  {alias}: P50 = {p50:.1f} min ({p50 / 60:.1f} h)")
        else:
            print(f"  {alias}: insufficient data")

    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate IRT S-curve grid")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--difficulty-col",
        required=True,
        help="Column from task_difficulties to use as difficulty axis",
    )
    parser.add_argument(
        "--title-label",
        default=None,
        help="Label for the difficulty source in the title",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
