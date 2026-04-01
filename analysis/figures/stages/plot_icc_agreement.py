"""ICC rater agreement plots for Appendix D (Measurement Quality).

Generates a two-panel figure: estimation agreement (left) and completion
agreement (right). Each panel shows rater A vs rater B in log2 space with
the identity line and ICC annotation.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    load_params,
    save_chart_json,
    save_png,
)
from lib.icc import compute_icc  # noqa: E402

# Palette
_TEAL_DARK = "#264653"
_TEAL = "#2a9d8f"
_TEXT = "#222222"
_TEXT_MUTED = "#888888"
_BG = "#fffaf0"

BENCH_COLORS = {
    "cybashbench": "#264653",
    "nl2bash": "#287271",
    "intercode_ctf": "#2a9d8f",
    "nyuctf": "#8ab17d",
    "cybench": "#e9c46a",
    "cvebench": "#f4a261",
    "cybergym": "#e76f51",
}

_TIME_TICKS_LOG2 = [
    -5.9069,  # 1s = log2(1/60)
    -3.5850,  # 5s = log2(5/60)
    -2.0000,  # 15s = log2(15/60)
    0.0000,  # 1m = log2(1)
    2.3219,  # 5m = log2(5)
    3.9069,  # 15m = log2(15)
    5.9069,  # 1h = log2(60)
    7.9069,  # 4h = log2(240)
    9.9069,  # 16h = log2(960)
    11.9069,  # 64h = log2(3840)
]
_TIME_TICK_LABELS = ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "4h", "16h", "64h"]


def _extract_pairs(rows_df):
    """Extract exactly two raters per task.

    Returns DataFrame with columns: va, vb, benchmark, passed_a, passed_b,
    task_id, expert_a, expert_b.
    """
    rows = []
    for tid, group in rows_df.groupby("task_id"):
        experts = group.sort_values("expert")
        if len(experts) < 2:
            continue
        rows.append(
            {
                "va": experts.iloc[0]["log2_min"],
                "vb": experts.iloc[1]["log2_min"],
                "benchmark": experts.iloc[0].get("benchmark", "unknown"),
                "passed_a": experts.iloc[0].get("passed", True),
                "passed_b": experts.iloc[1].get("passed", True),
                "task_id": tid,
                "expert_a": experts.iloc[0]["expert"],
                "expert_b": experts.iloc[1]["expert"],
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _passfail_symbol(passed_a, passed_b):
    """Map pass/fail combination to a marker symbol key for the JS renderer."""
    if passed_a and passed_b:
        return "both_passed"
    if not passed_a and passed_b:
        return "a_failed"
    if passed_a and not passed_b:
        return "b_failed"
    return "both_failed"


def _panel_series(pairs_df, expert_map):
    """Build per-benchmark scatter series from a pairs DataFrame.

    Returns list of series dicts with x/y in log2 minutes, anonymized labels,
    and markerSymbols for pass/fail shape encoding when the data includes it.
    """
    if pairs_df.empty:
        return []

    has_passfail = "passed_a" in pairs_df.columns and "passed_b" in pairs_df.columns

    series = []
    for bench in sorted(pairs_df["benchmark"].unique()):
        bdf = pairs_df[pairs_df["benchmark"] == bench]
        x_vals = bdf["va"].tolist()
        y_vals = bdf["vb"].tolist()
        labels = []
        marker_symbols = []
        for _, row in bdf.iterrows():
            labels.append(
                {
                    "task_id": row["task_id"],
                    "expert_a": expert_map.get(row["expert_a"], "Expert A"),
                    "expert_b": expert_map.get(row["expert_b"], "Expert B"),
                    "benchmark": row["benchmark"],
                }
            )
            if has_passfail:
                marker_symbols.append(
                    _passfail_symbol(row["passed_a"], row["passed_b"])
                )
        entry = {
            "name": bench,
            "x": [round(float(v), 4) for v in x_vals],
            "y": [round(float(v), 4) for v in y_vals],
            "color": BENCH_COLORS.get(bench, _TEXT_MUTED),
            "labels": labels,
        }
        if marker_symbols:
            entry["markerSymbols"] = marker_symbols
        series.append(entry)
    return series


# =============================================================================
# Compute: all data loading + statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute ICCs, build chart_data and companion stats."""

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    with open(resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)
    from lib.data import headline_task_set

    task_diff = pd.read_parquet(resolve(args.task_difficulties))
    model_runs = pd.read_parquet(resolve(args.model_runs)) if args.model_runs else None
    headline_tasks, task_family = headline_task_set(task_diff, model_runs)

    from lib.corrections import corrected_elapsed, TIMING_CORRECTIONS

    def build_icc_df(sessions, time_field, time_divisor=60, pass_set=None):
        """Build a DataFrame for ICC computation.

        pass_set: if provided, a set of session_ids that passed.
        When None (estimations), all rows get passed=True.
        Uses corrected_elapsed() for server_elapsed_seconds to apply
        timing corrections (e.g. rev-rock idle-overnight fix).
        """
        by_task = defaultdict(list)
        for s in sessions:
            tid = s["task_id"]
            if tid not in headline_tasks:
                continue
            if time_field == "server_elapsed_seconds":
                val = corrected_elapsed(s, TIMING_CORRECTIONS)
            else:
                val = s.get(time_field)
            if val and val > 0:
                passed = True
                if pass_set is not None:
                    passed = s.get("session_id", "") in pass_set
                by_task[tid].append(
                    {
                        "task_id": tid,
                        "expert": s.get("user_id", "unknown"),
                        "log2_min": np.log2(val / time_divisor),
                        "benchmark": task_family.get(tid, "unknown"),
                        "passed": passed,
                    }
                )
        rows = []
        for tid, vals in by_task.items():
            experts = {v["expert"] for v in vals}
            if len(experts) >= 2:
                rows.extend(vals)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    est_df = build_icc_df(snapshot["estimations"], "estimated_seconds")

    pass_session_ids = {p.get("session_id", "") for p in snapshot.get("passes", [])}
    all_completions = snapshot.get("passes", []) + snapshot.get("fails", [])
    comp_df = build_icc_df(
        all_completions, "server_elapsed_seconds", pass_set=pass_session_ids
    )

    # Compute ICC
    est_icc = compute_icc(est_df)
    comp_icc = compute_icc(comp_df)

    # Extract pairs
    est_pairs = _extract_pairs(est_df)
    comp_pairs = _extract_pairs(comp_df)

    # Shared axis range
    all_vals_list = []
    if not est_pairs.empty:
        all_vals_list.extend([est_pairs["va"].values, est_pairs["vb"].values])
    if not comp_pairs.empty:
        all_vals_list.extend([comp_pairs["va"].values, comp_pairs["vb"].values])
    all_vals = np.concatenate(all_vals_list)
    pad = 1.0
    shared_lim = (float(all_vals.min() - pad), float(all_vals.max() + pad))

    # Build a single anonymization map across both panels for consistency
    all_expert_ids = set()
    for df in [est_pairs, comp_pairs]:
        if not df.empty:
            all_expert_ids.update(df["expert_a"].unique())
            all_expert_ids.update(df["expert_b"].unique())
    expert_map = {
        eid: f"Expert {chr(65 + i)}" for i, eid in enumerate(sorted(all_expert_ids))
    }

    def _icc_annotation(icc_result):
        icc_val, ci_lo, ci_hi, n_tasks, sigma_w = icc_result
        if icc_val is None:
            return {"icc": None, "n_tasks": int(n_tasks)}
        return {
            "icc": round(float(icc_val), 3),
            "ci_lo": round(float(ci_lo), 3),
            "ci_hi": round(float(ci_hi), 3),
            "n_tasks": int(n_tasks),
            "within_rater_sd": round(float(sigma_w), 2),
        }

    # Build per-panel data for both chart_data and render_png
    def _pairs_to_list(pairs_df):
        """Convert pairs DataFrame to JSON-serializable list of dicts."""
        if pairs_df.empty:
            return []
        records = []
        for _, row in pairs_df.iterrows():
            records.append(
                {
                    "va": round(float(row["va"]), 4),
                    "vb": round(float(row["vb"]), 4),
                    "benchmark": row["benchmark"],
                    "passed_a": bool(row.get("passed_a", True)),
                    "passed_b": bool(row.get("passed_b", True)),
                    "task_id": row["task_id"],
                    "expert_a": row["expert_a"],
                    "expert_b": row["expert_b"],
                }
            )
        return records

    def _to_json_scalar(v):
        """Convert numpy scalar to Python scalar for JSON serialization."""
        if v is None:
            return None
        if hasattr(v, "item"):
            return v.item()
        return v

    est_icc_result = [_to_json_scalar(v) for v in est_icc]
    comp_icc_result = [_to_json_scalar(v) for v in comp_icc]

    # Build interactive chart tabs
    tabs = [
        {
            "label": "Estimation Agreement",
            "series": _panel_series(est_pairs, expert_map),
            "yEqualsX": True,
            "stats": _icc_annotation(est_icc),
        },
        {
            "label": "Completion Agreement",
            "series": _panel_series(comp_pairs, expert_map),
            "yEqualsX": True,
            "stats": _icc_annotation(comp_icc),
        },
    ]

    # All benchmarks present (for legend in render_png)
    all_benchmarks = sorted(
        set(
            list(est_pairs["benchmark"].unique() if not est_pairs.empty else [])
            + list(comp_pairs["benchmark"].unique() if not comp_pairs.empty else [])
        )
    )

    # Companion stats (written separately to the .json output)
    def _icc_dict(icc_result):
        icc_val, ci_lo, ci_hi, n_tasks, sigma_w = icc_result
        if icc_val is None:
            return {"icc": None, "n_tasks": int(n_tasks)}
        return {
            "icc": round(float(icc_val), 3),
            "ci_lo": round(float(ci_lo), 3),
            "ci_hi": round(float(ci_hi), 3),
            "n_tasks": int(n_tasks),
            "within_rater_sd": round(float(sigma_w), 2),
        }

    companion_stats = {
        "estimation": _icc_dict(est_icc),
        "completion": _icc_dict(comp_icc),
    }

    return {
        "chart_type": "tabbedScatter",
        "version": 1,
        "data": {
            "tabs": tabs,
            "panels": {
                "estimation": {
                    "title": "(a) Estimation Agreement",
                    "pairs": _pairs_to_list(est_pairs),
                    "icc_result": est_icc_result,
                },
                "completion": {
                    "title": "(b) Completion Agreement",
                    "pairs": _pairs_to_list(comp_pairs),
                    "icc_result": comp_icc_result,
                },
            },
            "shared_lim": list(shared_lim),
            "all_benchmarks": all_benchmarks,
        },
        "options": {
            "title": "Within-source rater agreement",
            "xLabel": "Rater A (log\u2082 minutes)",
            "yLabel": "Rater B (log\u2082 minutes)",
            "logAxes": True,
            "referenceLine": True,
            "timeAxis": True,
        },
        "_companion_stats": companion_stats,
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def _format_time_axis(ax, which="both"):
    for axis_name in ["x", "y"] if which == "both" else [which]:
        lo, hi = getattr(ax, f"get_{axis_name}lim")()
        visible = [
            (t, lab)
            for t, lab in zip(_TIME_TICKS_LOG2, _TIME_TICK_LABELS)
            if lo <= t <= hi
        ]
        if visible:
            ticks, labels = zip(*visible)
            getattr(ax, f"set_{axis_name}ticks")(list(ticks))
            getattr(ax, f"set_{axis_name}ticklabels")(list(labels))


def _stat_box(ax, text, loc="upper left"):
    va = "top" if "upper" in loc else "bottom"
    ha = "left" if "left" in loc else "right"
    x = 0.03 if "left" in loc else 0.97
    y = 0.97 if "upper" in loc else 0.03
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9,
        va=va,
        ha=ha,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor=_BG, alpha=0.85, edgecolor="#e5dfd6"
        ),
    )


def _render_agreement_panel(ax, panel_data, shared_lim):
    """Draw a rater agreement scatter on a single axes from chart_data."""
    title = panel_data["title"]
    pairs = panel_data["pairs"]
    icc_result = panel_data["icc_result"]

    if not pairs:
        ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    data_min, data_max = shared_lim

    # Identity line
    ax.plot(
        [data_min, data_max],
        [data_min, data_max],
        "--",
        color=_TEXT_MUTED,
        alpha=0.3,
        linewidth=0.8,
    )

    # Plot points by pass/fail category
    for p in pairs:
        color = BENCH_COLORS.get(p["benchmark"], _TEXT_MUTED)
        pa, pb = p["passed_a"], p["passed_b"]
        if pa and pb:
            marker = "o"
        elif not pa and pb:
            marker = "v"
        elif pa and not pb:
            marker = "<"
        else:
            marker = "D"
        ax.scatter(
            p["va"],
            p["vb"],
            c=color,
            s=55,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
            marker=marker,
        )

    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Rater A", fontsize=10)
    ax.set_ylabel("Rater B", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15)
    _format_time_axis(ax, which="both")

    # ICC stats box
    icc_val, ci_lo, ci_hi, n_tasks, sigma_w = icc_result
    if icc_val is not None:
        stat_text = (
            f"ICC = {icc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"within-rater SD = {sigma_w:.2f}\n"
            f"N = {n_tasks} paired tasks"
        )
    else:
        stat_text = f"Insufficient data\nN = {n_tasks} paired tasks"
    _stat_box(ax, stat_text, loc="lower right")


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render ICC agreement two-panel figure from chart JSON data."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    panels_data = chart_data["data"]["panels"]
    shared_lim = chart_data["data"]["shared_lim"]
    all_benchmarks = chart_data["data"]["all_benchmarks"]

    fig, (ax_est, ax_comp) = plt.subplots(1, 2, figsize=(14, 6.5))

    _render_agreement_panel(ax_est, panels_data["estimation"], shared_lim)
    _render_agreement_panel(ax_comp, panels_data["completion"], shared_lim)

    # Single shared legend at bottom: marker shapes + benchmark colors
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=_TEXT_MUTED,
            linestyle="",
            markersize=8,
            label="Both passed",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color=_TEXT_MUTED,
            linestyle="",
            markersize=8,
            label="Rater A failed",
        ),
        Line2D(
            [0],
            [0],
            marker="<",
            color=_TEXT_MUTED,
            linestyle="",
            markersize=8,
            label="Rater B failed",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=_TEXT_MUTED,
            linestyle="",
            markersize=8,
            label="Both failed",
        ),
    ]

    bench_handles = [
        mpatches.Patch(facecolor=BENCH_COLORS.get(b, _TEXT_MUTED), label=b)
        for b in all_benchmarks
    ]

    fig.legend(
        handles=marker_handles + bench_handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate ICC agreement plots")
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument(
        "--model-runs",
        default=None,
        help="model_runs.parquet (filters headline to evaluated tasks)",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)

    # Write companion JSON with all stats for paper references
    companion_stats = chart_data.get("_companion_stats", {})
    json_path = Path(args.output).with_suffix(".json")
    if not json_path.is_absolute():
        json_path = _NOTEBOOKS_DIR / json_path
    with open(json_path, "w") as f:
        json.dump(companion_stats, f, indent=2)
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
