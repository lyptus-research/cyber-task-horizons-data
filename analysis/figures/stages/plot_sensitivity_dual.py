"""Stage: Dual sensitivity boxplot (doubling time + frontier model P50).

Two panels side by side with shared y-axis labels. Left panel shows
doubling time distributions, right panel shows frontier model P50
distributions. Same treatments on the y-axis, same styling.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    load_params,
    save_chart_json,
    save_png,
)
from lib.lyptus_style import COLORS  # noqa: E402


PREFERRED_ORDER = [
    "Headline (best-available)",
    "Study completions only",
    "Actuals only (completions + first-blood \u00d72.4)",
    "No first-blood times",
    "First-blood \u00d72.4 adjusted",
    "Model estimates (human task set)",
    "Model estimates (full task set)",
    "Leave-one-benchmark-out",
    "Leave-one-expert-out",
    "No CyBashBench",
    "No NL2Bash",
    "No InterCode-CTF",
    "No NYUCTF",
    "No CyBench",
    "No CVEBench",
    "No CyberGym",
    "Bootstrap (models)",
    "Weighting / regularisation",
    "Overall",
]

# Excluded from the plot
_EXCLUDED = {
    "Bootstrap (tasks)",
    "First-blood \u00d72.4 adjusted",
}
# Bootstrap (models) excluded from P50 only (meaningless for absolute P50)
_EXCLUDED_P50 = _EXCLUDED | {"Bootstrap (models)"}

_SOURCE_LABELS = {
    "Study completions only",
    "Actuals only (completions + first-blood \u00d72.4)",
    "No first-blood times",
    "First-blood \u00d72.4 adjusted",
    "Model estimates (human task set)",
    "Model estimates (full task set)",
    "Leave-one-benchmark-out",
    "Leave-one-expert-out",
    "No CyBashBench",
    "No NL2Bash",
    "No InterCode-CTF",
    "No NYUCTF",
    "No CyBench",
    "No CVEBench",
    "No CyberGym",
}
# Keep for separator logic (visual gap before analytical section)
_LOO_LABELS = {
    "Leave-one-benchmark-out",
    "Leave-one-expert-out",
}
_ANALYTICAL_LABELS = {
    "Bootstrap (tasks)",
    "Bootstrap (models)",
    "Weighting / regularisation",
    "Overall",
}

# Display names for y-axis labels (data keys stay unchanged)
_DISPLAY_NAMES = {
    "Headline (best-available)": "Headline",
    "Study completions only": "Drop estimates + first-blood",
    "Actuals only (completions + first-blood \u00d72.4)": "Drop estimates only",
    "No first-blood times": "Drop first-blood times",
    "Model estimates (human task set)": "Model estimates (same tasks)",
    "Model estimates (full task set)": "Model estimates (all tasks)",
    "Bootstrap (models)": "Model resampling",
}

# Treatment descriptions for interactive tooltip annotations (matching Table 5)
_DESCRIPTIONS = {
    "Headline (best-available)": "Full best-available source hierarchy: expert completions, censored observations, CTF first-blood competition times, and expert estimates.",
    "Study completions only": "Only the expert completion times collected in our study. No estimates, no first-blood times. The strictest test of our own data.",
    "Actuals only (completions + first-blood \u00d72.4)": "Study completions plus CTF first-blood times, adjusted to expert-equivalent scale (\u00d72.4). No expert estimates.",
    "No first-blood times": "Study completions and expert estimates, but no CTF first-blood competition times. Tests whether removing the structurally different timing source matters.",
    "First-blood \u00d72.4 adjusted": "Full hierarchy, but first-blood times scaled by 2.4\u00d7 to approximate individual expert completion times.",
    "Model estimates (human task set)": "Same task set as the headline, but with model-estimated difficulty labels instead of human-derived ones. Isolates the estimation method.",
    "Model estimates (full task set)": "All evaluated tasks with model-estimated difficulty labels. Tests whether the larger task set changes the result.",
    "Leave-one-benchmark-out": "Each of the seven benchmarks excluded in turn, testing whether any single dataset drives the result.",
    "Leave-one-expert-out": "Each of the study's expert participants excluded in turn, testing whether any single rater's estimates or completions drive the result.",
    "Bootstrap (models)": "Resampling SOTA models with replacement, testing sensitivity to which models anchor the trendline.",
    "Model resampling": "Resampling SOTA models with replacement, testing sensitivity to which models anchor the trendline.",
    "Weighting / regularisation": "All combinations of two weighting schemes and six regularisation strengths spanning four orders of magnitude.",
}

# Color constants used in render_png
_COLOR_HEADLINE = COLORS["teal"]
_COLOR_SOURCE = COLORS.get("coral", "#E8927C")
_COLOR_ANALYTICAL = COLORS.get("teal_dark", "#3A7D7E")


def _extract_dt(r):
    coef = r.get("coef", 0)
    if coef > 0:
        return np.log(2) / coef / 30.44
    return np.nan


def _extract_p50(r):
    p = r.get("frontier_p50")
    if p is not None and not np.isnan(p) and p > 0:
        return p / 60  # minutes to hours
    return np.nan


def _resolve_path(p, base):
    path = Path(p)
    return path if path.is_absolute() else base / path


def _load_records(path):
    with open(path) as f:
        return json.load(f)


def _build_variant_order(record_dict, excluded=None):
    excl = excluded if excluded is not None else _EXCLUDED
    available = list(record_dict.keys())
    order = [v for v in PREFERRED_ORDER if v in available and v not in excl]
    for v in available:
        if v not in order and v not in excl:
            order.append(v)
    return order


def _build_colors(variant_order):
    colors = []
    for v in variant_order:
        if v.startswith("Headline"):
            colors.append(_COLOR_HEADLINE)
        elif v in _SOURCE_LABELS:
            colors.append(_COLOR_SOURCE)
        elif v in _ANALYTICAL_LABELS:
            colors.append(_COLOR_ANALYTICAL)
        else:
            colors.append("#7F8C8D")
    return colors


def _build_separators(variant_order):
    sep_positions = []
    for i, v in enumerate(variant_order):
        if i == 0:
            continue
        prev = variant_order[i - 1]
        if (
            (prev.startswith("Headline") and v in _SOURCE_LABELS)
            or (prev in _SOURCE_LABELS and v in _ANALYTICAL_LABELS)
        ):
            sep_positions.append(i + 0.5)
    return sep_positions


def _extract_metric(record_dict, variant_order, extract_fn, clip_at_zero=False):
    data = []
    for v in variant_order:
        records = record_dict.get(v, [])
        vals = [extract_fn(r) for r in records]
        vals = [x for x in vals if not np.isnan(x)]
        if vals:
            q01, q99 = np.percentile(vals, [1, 99])
            lo = max(q01, 0) if clip_at_zero else q01
            vals = np.clip(vals, lo, q99).tolist()
        data.append(vals)
    return data


def _treatment_category(name):
    """Classify a treatment name into source/analytical."""
    if name.startswith("Headline"):
        return "source"
    if name in _SOURCE_LABELS:
        return "source"
    if name in _ANALYTICAL_LABELS:
        return "analytical"
    return "other"


def _build_treatment(name, vals):
    """Build a single treatment dict with summary, outliers, and points."""
    if not vals:
        return {
            "name": name,
            "display_name": _DISPLAY_NAMES.get(name, name),
            "description": _DESCRIPTIONS.get(
                name, _DESCRIPTIONS.get(_DISPLAY_NAMES.get(name, ""), "")
            ),
            "category": _treatment_category(name),
            "is_headline": name.startswith("Headline"),
            "summary": None,
            "outliers": [],
            "points": [],
        }
    arr = np.array(vals)
    q1 = float(np.percentile(arr, 25))
    median = float(np.median(arr))
    q3 = float(np.percentile(arr, 75))
    whisker_lo = float(np.percentile(arr, 10))
    whisker_hi = float(np.percentile(arr, 90))
    outliers = [float(v) for v in arr if v < whisker_lo or v > whisker_hi]
    return {
        "name": name,
        "display_name": _DISPLAY_NAMES.get(name, name),
        "description": _DESCRIPTIONS.get(
            name, _DESCRIPTIONS.get(_DISPLAY_NAMES.get(name, ""), "")
        ),
        "category": _treatment_category(name),
        "is_headline": name.startswith("Headline"),
        "summary": {
            "median": round(median, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "whisker_lo": round(whisker_lo, 2),
            "whisker_hi": round(whisker_hi, 2),
        },
        "outliers": [round(v, 2) for v in sorted(outliers)],
        "points": [round(v, 4) for v in vals],
    }


def _build_panel(title, variant_order, box_data):
    """Build a panel dict from variant names and their data lists."""
    treatments = []
    for name, data in zip(variant_order, box_data):
        treatments.append(_build_treatment(name, data))
    return {"title": title, "treatments": treatments}


def _variant_key(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _summarise(data):
    if len(data) == 0:
        return None
    return {
        "median": round(float(np.median(data)), 1),
        "q25": round(float(np.percentile(data, 25)), 1),
        "q75": round(float(np.percentile(data, 75)), 1),
        "p10": round(float(np.percentile(data, 10)), 1),
        "p90": round(float(np.percentile(data, 90)), 1),
        "n": len(data),
    }


# =============================================================================
# Compute: all data loading + statistical work, returns JSON-serializable dict
# =============================================================================

DT_XLIM = (0, 12)
P50_XLIM = (0.5, 128)


def compute(args, params) -> dict:
    """Load multiverse records, compute all metrics, return chart_data dict."""
    record_dict = _load_records(_resolve_path(args.records, _NOTEBOOKS_DIR))

    variant_order = _build_variant_order(record_dict)
    colors = _build_colors(variant_order)
    sep_positions = _build_separators(variant_order)
    display_labels = [_DISPLAY_NAMES.get(v, v) for v in variant_order]

    dt_data_full = _extract_metric(record_dict, variant_order, _extract_dt)
    p50_data = _extract_metric(
        record_dict, variant_order, _extract_p50, clip_at_zero=True
    )

    # --- DT panels ---
    dt_panels = []
    dt_data_2024 = None
    if args.records_2024plus:
        record_dict_2024 = _load_records(
            _resolve_path(args.records_2024plus, _NOTEBOOKS_DIR)
        )
        dt_data_2024 = _extract_metric(record_dict_2024, variant_order, _extract_dt)
        dt_panels.append(
            _build_panel("Doubling time (all SOTA, 2019+)", variant_order, dt_data_full)
        )
        dt_panels.append(
            _build_panel("Doubling time (2024+ only)", variant_order, dt_data_2024)
        )
    else:
        dt_panels.append(
            _build_panel("Doubling time (all SOTA, 2019+)", variant_order, dt_data_full)
        )

    chart_data_dt = {
        "chart_type": "boxplotH",
        "version": 1,
        "data": {"panels": dt_panels},
        "options": {
            "title": "Sensitivity of doubling time to analytical choices",
            "xlim": [float(DT_XLIM[0]), float(DT_XLIM[1])],
        },
    }

    # --- P50 panels (separate variant order, excludes Bootstrap (models)) ---
    frontier_labels = [s.strip() for s in args.frontier_label.split(",")]
    variant_order_p50 = _build_variant_order(record_dict, excluded=_EXCLUDED_P50)
    p50_panels = []
    p50_data_per_model = {}
    for fl in frontier_labels:
        p50_key = f"frontier_p50_{fl}"

        def _extract_p50_for(r, _key=p50_key):
            p = r.get(_key, r.get("frontier_p50"))
            if p is not None and not np.isnan(p) and p > 0:
                return p / 60
            return np.nan

        p50_for_model = _extract_metric(
            record_dict, variant_order_p50, _extract_p50_for, clip_at_zero=True
        )
        p50_data_per_model[fl] = p50_for_model
        p50_panels.append(
            _build_panel(f"{fl} P50 time horizon (hours)", variant_order_p50, p50_for_model)
        )

    chart_data_p50 = {
        "chart_type": "boxplotH",
        "version": 1,
        "data": {"panels": p50_panels},
        "options": {
            "title": "Sensitivity of frontier P50 to analytical choices",
            "xlim": [float(P50_XLIM[0]), float(P50_XLIM[1])],
            "logX": True,
            "legendRight": True,
        },
    }

    # --- Companion stats ---
    def _variant_key_fn(name):
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    stats = {}
    source_keys = []
    loo_keys = []
    analytical_keys = []

    for i, v in enumerate(variant_order):
        key = _variant_key_fn(v)
        entry = {}
        dt_summary = _summarise(dt_data_full[i])
        if dt_summary:
            entry["dt_months"] = dt_summary
        p50_summary = _summarise(p50_data[i])
        if p50_summary:
            entry["frontier_p50_hours"] = p50_summary
        if entry:
            stats[key] = entry

        if v in _SOURCE_LABELS:
            source_keys.append(key)
        elif v in _LOO_LABELS:
            loo_keys.append(key)
        elif v in _ANALYTICAL_LABELS:
            analytical_keys.append(key)

    def _group_range(keys, metric="dt_months"):
        medians = [
            stats[k][metric]["median"]
            for k in keys
            if k in stats and metric in stats[k]
        ]
        if not medians:
            return None
        return {"min": min(medians), "max": max(medians)}

    stats["_source_range"] = _group_range(source_keys)
    stats["_analytical_range"] = _group_range(analytical_keys)
    stats["_source_p50_range"] = _group_range(source_keys, "frontier_p50_hours")

    # Individual LOO medians from source records (before pooling)
    if args.source_records:
        src_path = (
            Path(args.source_records)
            if Path(args.source_records).is_absolute()
            else _NOTEBOOKS_DIR / args.source_records
        )
        if src_path.exists():
            with open(src_path) as f:
                src_records = json.load(f)
            loo_medians = []
            for label, recs in src_records.items():
                if label.startswith("No "):
                    dts = [
                        np.log(2) / r["coef"] / 30.44
                        for r in recs
                        if r.get("coef", 0) > 0
                    ]
                    if dts:
                        loo_medians.append(round(float(np.median(dts)), 1))
            if loo_medians:
                stats["_loo_range"] = {
                    "min": min(loo_medians),
                    "max": max(loo_medians),
                }
            else:
                stats["_loo_range"] = _group_range(loo_keys)
        else:
            stats["_loo_range"] = _group_range(loo_keys)
    else:
        stats["_loo_range"] = _group_range(loo_keys)

    return {
        "_dt": chart_data_dt,
        "_p50": chart_data_p50,
        "_companion_stats": stats,
        "_render": {
            "variant_order": variant_order,
            "variant_order_p50": variant_order_p50,
            "display_labels": display_labels,
            "colors": colors,
            "sep_positions": sep_positions,
            "dt_data_full": dt_data_full,
            "dt_data_2024": dt_data_2024,
            "p50_data_per_model": {k: v for k, v in p50_data_per_model.items()},
            "frontier_labels": frontier_labels,
            "has_2024plus": args.records_2024plus is not None,
        },
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def _darken(color, factor=0.6):
    from matplotlib.colors import to_rgb

    return tuple(c * factor for c in to_rgb(color))


def _draw_boxplot(
    ax, box_data, variant_order, colors, sep_positions, xlabel, xlim=None, log_x=False
):
    """Draw a single horizontal boxplot on the given axes."""
    import numpy as np
    import matplotlib.ticker as mticker

    LINE_WIDTH = 1.75
    MEDIAN_LINE_WIDTH = 2.5

    # Convert lists to numpy arrays for boxplot
    np_data = [np.array(d) for d in box_data]

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}")
        )

    boxplot = ax.boxplot(
        np_data,
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

    for sep_y in sep_positions:
        ax.axhline(
            y=sep_y, linestyle=":", color="black", linewidth=1.5, alpha=0.5, zorder=0
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, axis="x", linestyle="-", color="grey", alpha=0.3)
    ax.grid(True, axis="x", which="minor", linestyle="-", color="grey", alpha=0.1)
    if not log_x:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if xlim is not None:
        ax.set_xlim(*xlim)

        # Annotate whiskers that extend past the x-axis limit
        x_max = xlim[1]
        for i, data in enumerate(np_data):
            if len(data) == 0:
                continue
            p90 = float(np.percentile(data, 90))
            if p90 > x_max:
                label = f"{p90:.0f}" if p90 >= 10 else f"{p90:.1f}"
                ax.annotate(
                    label,
                    xy=(x_max * 0.92, i + 1),
                    fontsize=6,
                    color="#666",
                    fontweight="bold",
                    ha="right",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="#fffaf0", ec="#ccc", alpha=0.85
                    ),
                )


def _add_legend(ax, variant_order, loc="upper right"):
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=_COLOR_HEADLINE,
            edgecolor=_darken(_COLOR_HEADLINE),
            label="Headline",
        ),
        Patch(
            facecolor=_COLOR_SOURCE,
            edgecolor=_darken(_COLOR_SOURCE),
            label="Data ablation",
        ),
        Patch(
            facecolor=_COLOR_ANALYTICAL,
            edgecolor=_darken(_COLOR_ANALYTICAL),
            label="Analytical",
        ),
    ]
    present_groups = set()
    for v in variant_order:
        if v.startswith("Headline"):
            present_groups.add("Headline")
        elif v in _SOURCE_LABELS:
            present_groups.add("Data ablation")
        elif v in _ANALYTICAL_LABELS:
            present_groups.add("Analytical")
    legend_handles = [h for h in legend_handles if h.get_label() in present_groups]
    ax.legend(handles=legend_handles, loc=loc, fontsize=8, framealpha=0.9)


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render dual sensitivity boxplots from chart_data dict.

    Produces two PNGs: the main DT figure at `output`, and the P50 figure
    at the path stored in chart_data (or derived from output).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = chart_data["_render"]
    variant_order = rd["variant_order"]
    display_labels = rd["display_labels"]
    colors = rd["colors"]
    sep_positions = rd["sep_positions"]
    dt_data_full = rd["dt_data_full"]
    dt_data_2024 = rd["dt_data_2024"]
    frontier_labels = rd["frontier_labels"]
    has_2024plus = rd["has_2024plus"]
    p50_data_per_model = rd["p50_data_per_model"]

    # --- Figure 1: Doubling time ---
    if has_2024plus and dt_data_2024 is not None:
        fig, (ax_full, ax_2024) = plt.subplots(
            1,
            2,
            figsize=(16, 0.5 * len(variant_order) + 1.5),
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        _draw_boxplot(
            ax_full,
            dt_data_full,
            variant_order,
            colors,
            sep_positions,
            "Doubling time (months)",
            xlim=DT_XLIM,
        )
        _draw_boxplot(
            ax_2024,
            dt_data_2024,
            variant_order,
            colors,
            sep_positions,
            "Doubling time (months)",
            xlim=DT_XLIM,
        )
        ax_full.set_xlim(*DT_XLIM)
        ax_2024.set_xlim(*DT_XLIM)
        ax_full.invert_yaxis()

        ax_full.set_yticks(range(1, len(variant_order) + 1))
        ax_full.set_yticklabels(display_labels, fontsize=8.5)
        ax_2024.tick_params(axis="y", labelleft=False)

        ax_full.set_title("2019+", fontsize=11, fontweight="bold", pad=8)
        ax_2024.set_title("2024+", fontsize=11, fontweight="bold", pad=8)
        _add_legend(ax_full, variant_order, loc="lower left")
        fig.subplots_adjust(left=0.22, right=0.97, wspace=0.05)
    else:
        fig, ax_dt = plt.subplots(
            1,
            1,
            figsize=(9, 0.5 * len(variant_order) + 1.5),
        )
        _draw_boxplot(
            ax_dt,
            dt_data_full,
            variant_order,
            colors,
            sep_positions,
            "Doubling time (months)",
            xlim=DT_XLIM,
        )
        ax_dt.invert_yaxis()
        ax_dt.set_yticks(range(1, len(variant_order) + 1))
        ax_dt.set_yticklabels(display_labels, fontsize=8.5)
        ax_dt.set_title("Doubling time", fontsize=11, fontweight="bold", pad=8)
        _add_legend(ax_dt, variant_order, loc="lower left")
        fig.subplots_adjust(left=0.28, right=0.97)

    # Save DT chart JSON and PNG
    save_chart_json(chart_data["_dt"], output)
    save_png(fig, output, params)
    plt.close("all")

    # --- Figure 2: Frontier P50 ---
    output_p50 = chart_data.get("_output_p50")
    if output_p50 and frontier_labels:
        variant_order_p50 = rd["variant_order_p50"]
        display_labels_p50 = [_DISPLAY_NAMES.get(v, v) for v in variant_order_p50]
        colors_p50 = _build_colors(variant_order_p50)
        sep_positions_p50 = _build_separators(variant_order_p50)
        n_frontiers = len(frontier_labels)

        fig_p50, axes = plt.subplots(
            1,
            n_frontiers,
            figsize=(9 * n_frontiers, 0.5 * len(variant_order_p50) + 1.5),
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        if n_frontiers == 1:
            axes = [axes]

        for fi, (ax, fl) in enumerate(zip(axes, frontier_labels)):
            p50_for_model = p50_data_per_model[fl]
            _draw_boxplot(
                ax,
                p50_for_model,
                variant_order_p50,
                colors_p50,
                sep_positions_p50,
                "P50 time horizon (hours)",
                xlim=P50_XLIM,
                log_x=True,
            )
            ax.set_title(f"{fl} (2M tokens)", fontsize=11, fontweight="bold", pad=8)
            if fi > 0:
                ax.tick_params(axis="y", labelleft=False)

        axes[0].invert_yaxis()
        axes[0].set_yticks(range(1, len(variant_order_p50) + 1))
        axes[0].set_yticklabels(display_labels_p50, fontsize=8.5)
        _add_legend(axes[-1], variant_order_p50, loc="lower right")
        fig_p50.subplots_adjust(
            left=0.22 if n_frontiers > 1 else 0.28, right=0.97, wspace=0.05
        )

        save_chart_json(chart_data["_p50"], output_p50)
        save_png(fig_p50, output_p50, params)
        plt.close("all")


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot sensitivity boxplots (DT full/2024+ and frontier P50)")
    parser.add_argument(
        "--records", required=True, help="Full-range combined_multiverse_records.json"
    )
    parser.add_argument(
        "--records-2024plus", default=None, help="2024+ multiverse_records.json"
    )
    parser.add_argument("--frontier-label", default="GPT-5.3 Codex")
    parser.add_argument("--stats-output", default=None)
    parser.add_argument(
        "--source-records",
        default=None,
        help="Source multiverse_records.json (for individual LOO medians)",
    )
    parser.add_argument(
        "--output-p50",
        default=None,
        help="Separate output path for the frontier P50 figure",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)

    # Stash output_p50 so render_png knows where to write the second figure
    chart_data["_output_p50"] = args.output_p50

    render_png(chart_data, args.output, params)

    # Write companion stats
    if args.stats_output:
        out_path = (
            Path(args.stats_output)
            if Path(args.stats_output).is_absolute()
            else _NOTEBOOKS_DIR / args.stats_output
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(chart_data["_companion_stats"], f, indent=2)
        print(f"Wrote stats: {out_path}")


if __name__ == "__main__":
    main()
