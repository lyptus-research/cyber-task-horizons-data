"""Stage: Plot source-sensitivity multiverse boxplot.

Reads the multiverse records JSON (same format as METR's wrangle stage)
and produces a horizontal boxplot showing the DT distribution for each
source variant.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
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
from lib.lyptus_style import COLORS  # noqa: E402


def _darken(color: str | tuple, factor: float = 0.6) -> tuple:
    from matplotlib.colors import to_rgb

    return tuple(c * factor for c in to_rgb(color))


METRIC_CONFIG = {
    "dt_months": {
        "xlabel": "Doubling time (months)",
        "title": "Sensitivity to human timing source and analytical choices",
        "extract": lambda r: (
            np.log(2) / r["coef"] / 30.44 if r.get("coef", 0) > 0 else float("nan")
        ),
    },
    "frontier_p50_hours": {
        "xlabel": "Time horizon (hours)",
        "title": "Frontier model sensitivity to human timing source and analytical choices",
        "extract": lambda r: (
            r["frontier_p50"] / 60
            if pd.notna(r.get("frontier_p50")) and r.get("frontier_p50", 0) > 0
            else float("nan")
        ),
    },
}


# Variant group classification
_SOURCE_LABELS = {
    "Study completions only",
    "Actuals only (completions + first-blood \u00d72.4)",
    "No first-blood times",
    "First-blood \u00d72.4 adjusted",
    "Model estimates (human task set)",
    "Model estimates (full task set)",
}
_LOO_LABELS = {
    "Leave-one-benchmark-out",
    "No CyBashBench",
    "No NL2Bash",
    "No InterCode-CTF",
    "No NYUCTF",
    "No CyBench",
    "No CVEBench",
    "No CyberGym",
}
_ANALYTICAL_LABELS = {
    "Bootstrap (tasks)",
    "Bootstrap (models)",
    "Weighting / regularisation",
    "Overall",
}

PREFERRED_ORDER = [
    "Headline (best-available)",
    "Study completions only",
    "Actuals only (completions + first-blood \u00d72.4)",
    "No first-blood times",
    "First-blood \u00d72.4 adjusted",
    "Model estimates (human task set)",
    "Model estimates (full task set)",
    "Leave-one-benchmark-out",
    "No CyBashBench",
    "No NL2Bash",
    "No InterCode-CTF",
    "No NYUCTF",
    "No CyBench",
    "No CVEBench",
    "No CyberGym",
    "Bootstrap (tasks)",
    "Bootstrap (models)",
    "Weighting / regularisation",
    "Overall",
]


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load multiverse records, extract metric values, return chart_data dict."""
    metric_key = args.metric
    metric = METRIC_CONFIG[metric_key]

    records_path = Path(args.records)
    if not records_path.is_absolute():
        records_path = _NOTEBOOKS_DIR / records_path

    with open(records_path) as f:
        record_dict = json.load(f)

    # Extract metric from records and compute box stats
    available = list(record_dict.keys())
    variant_order = [v for v in PREFERRED_ORDER if v in available]
    for v in available:
        if v not in variant_order:
            variant_order.append(v)

    # Classify variants into groups for coloring
    def _group(v):
        if v.startswith("Headline"):
            return "headline"
        if v in _SOURCE_LABELS:
            return "source"
        if v in _LOO_LABELS:
            return "loo"
        if v in _ANALYTICAL_LABELS:
            return "analytical"
        return "other"

    chart_variants = []
    for v in variant_order:
        records = record_dict.get(v, [])
        values = []
        for r in records:
            val = metric["extract"](r)
            if not np.isnan(val):
                values.append(float(val))

        if not values:
            continue

        # Clip extreme outliers (1st-99th percentile)
        arr = np.array(values)
        q01, q99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, max(q01, 0), q99)
        values = arr.tolist()

        q25, med, q75 = np.percentile(values, [25, 50, 75])
        q10, q90 = np.percentile(values, [10, 90])

        chart_variants.append(
            {
                "label": v,
                "group": _group(v),
                "values": [round(x, 3) for x in values],
                "median": round(float(med), 3),
                "q25": round(float(q25), 3),
                "q75": round(float(q75), 3),
                "q10": round(float(q10), 3),
                "q90": round(float(q90), 3),
                "n": len(values),
            }
        )

    # Find separator positions (where groups transition)
    sep_positions = []
    for i in range(1, len(chart_variants)):
        if chart_variants[i]["group"] != chart_variants[i - 1]["group"]:
            sep_positions.append(i + 0.5)

    # Print summary stats
    print(f"\n{'Variant':<45} {'Median':<10} {'IQR':<15} {'10-90%'}")
    print("-" * 85)
    for cv in chart_variants:
        print(
            f"{cv['label']:<45} {cv['median']:<10.1f} "
            f"[{cv['q25']:.1f}, {cv['q75']:.1f}]{'':>3} "
            f"[{cv['q10']:.1f}, {cv['q90']:.1f}]"
        )

    chart_data = {
        "chart_type": "sensitivityMultiverse",
        "version": 1,
        "data": {
            "variants": chart_variants,
            "sep_positions": sep_positions,
        },
        "options": {
            "title": metric["title"],
            "xlabel": metric["xlabel"],
            "metric": metric_key,
        },
    }

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render horizontal boxplot from chart_data."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    variants = chart_data["data"]["variants"]
    sep_positions = chart_data["data"]["sep_positions"]
    options = chart_data.get("options", {})

    color_headline = COLORS["teal"]
    color_source = COLORS.get("coral", "#E8927C")
    color_loo = COLORS.get("slate", "#6B7B8D")
    color_analytical = COLORS.get("teal_dark", "#3A7D7E")

    GROUP_COLORS = {
        "headline": color_headline,
        "source": color_source,
        "loo": color_loo,
        "analytical": color_analytical,
        "other": "#7F8C8D",
    }

    LINE_WIDTH = 1.75
    MEDIAN_LINE_WIDTH = 2.5

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(variants) + 1.5))

    box_data = [coerce_floats(v["values"]) for v in variants]
    colors = [GROUP_COLORS.get(v["group"], "#7F8C8D") for v in variants]

    boxplot = ax.boxplot(
        box_data,
        vert=False,
        whis=(10, 90),
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"linewidth": MEDIAN_LINE_WIDTH},
        boxprops={"alpha": 0.9},
        whiskerprops={"linewidth": LINE_WIDTH, "linestyle": "-"},
        capprops={"linewidth": LINE_WIDTH},
        widths=0.6,
    )

    for i, (box, color) in enumerate(zip(boxplot["boxes"], colors)):
        dark = _darken(color)
        box.set(facecolor=color, edgecolor=dark, linewidth=LINE_WIDTH)
        boxplot["whiskers"][i * 2].set(color=dark, linewidth=LINE_WIDTH)
        boxplot["whiskers"][i * 2 + 1].set(color=dark, linewidth=LINE_WIDTH)
        boxplot["caps"][i * 2].set(color=dark, linewidth=LINE_WIDTH)
        boxplot["caps"][i * 2 + 1].set(color=dark, linewidth=LINE_WIDTH)
        boxplot["medians"][i].set(color=dark, linewidth=MEDIAN_LINE_WIDTH)

    # Separator lines between groups
    for sep_y in sep_positions:
        ax.axhline(
            y=sep_y, linestyle=":", color="black", linewidth=1.5, alpha=0.5, zorder=0
        )

    # Labels
    ax.set_yticks(range(1, len(variants) + 1))
    ax.set_yticklabels([v["label"] for v in variants], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel(options.get("xlabel", ""), fontsize=12)
    ax.set_title(options.get("title", ""), fontsize=13, pad=12)

    # Legend
    present_groups = {v["group"] for v in variants}
    GROUP_LABELS = {
        "headline": "Headline",
        "source": "Source treatment",
        "loo": "Leave-one-benchmark-out",
        "analytical": "Analytical robustness",
    }
    legend_handles = []
    for group_key, label in GROUP_LABELS.items():
        if group_key in present_groups:
            c = GROUP_COLORS[group_key]
            legend_handles.append(Patch(facecolor=c, edgecolor=_darken(c), label=label))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)

    # Clip x-axis if extreme whiskers dominate
    all_medians = [v["median"] for v in variants]
    if all_medians:
        med_q75 = np.percentile(all_medians, 75)
        x_clip = max(med_q75 * 4, np.percentile(all_medians, 90) * 2.5)
        x_max_data = max(v["q90"] for v in variants)
        if x_max_data > x_clip * 1.5:
            ax.set_xlim(left=0, right=x_clip)
            try:
                from lib.lyptus_style import FONT_SANS
            except (ImportError, KeyError):
                FONT_SANS = "Helvetica Neue"
            for i, v in enumerate(variants):
                if v["q90"] > x_clip:
                    label = f"{v['q90']:.0f}" if v["q90"] >= 10 else f"{v['q90']:.1f}"
                    ax.annotate(
                        label,
                        xy=(x_clip * 0.97, i + 1),
                        fontsize=7,
                        fontfamily=FONT_SANS,
                        color="#666",
                        fontweight="bold",
                        ha="right",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="#fffaf0",
                            ec="#ccc",
                            alpha=0.85,
                        ),
                    )

    # Grid
    ax.grid(True, axis="x", linestyle="-", color="grey", alpha=0.3)
    ax.grid(True, axis="x", which="minor", linestyle="-", color="grey", alpha=0.1)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    save_png(fig, output, params)
    plt.close(fig)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot source-sensitivity multiverse boxplot")
    parser.add_argument("--records", required=True, help="multiverse_records.json")
    parser.add_argument(
        "--metric",
        default="dt_months",
        choices=list(METRIC_CONFIG.keys()),
        help="Which metric to plot",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
