"""ICC gauge plot for Appendix D (Measurement Quality).

Shows estimation and completion ICC point estimates with confidence intervals
on a shared horizontal scale with Koo & Li (2016) interpretive thresholds.

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

import matplotlib.pyplot as plt  # noqa: E402

# Palette
_TEAL_DARK = "#264653"
_CORAL = "#e76f51"
_TEXT_MUTED = "#888888"

# Koo & Li (2016) threshold colors
_ZONE_POOR = "#e76f51"
_ZONE_MODERATE = "#f4a261"
_ZONE_GOOD = "#2a9d8f"
_ZONE_EXCELLENT = "#264653"

_COLOR_MAP = {"tealDark": _TEAL_DARK, "coral": _CORAL}


# =============================================================================
# Compute: all statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute ICCs, return chart_data dict."""

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    with open(resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)
    from lib.data import headline_task_set

    task_diff = pd.read_parquet(resolve(args.task_difficulties))
    model_runs = pd.read_parquet(resolve(args.model_runs)) if args.model_runs else None
    headline_tasks, task_family = headline_task_set(task_diff, model_runs)

    from lib.corrections import corrected_elapsed, TIMING_CORRECTIONS

    def build_icc_df(sessions, time_field, time_divisor=60):
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
                by_task[tid].append(
                    {
                        "task_id": tid,
                        "expert": s.get("user_id", "unknown"),
                        "log2_min": np.log2(val / time_divisor),
                        "benchmark": task_family.get(tid, "unknown"),
                    }
                )
        rows = []
        for tid, vals in by_task.items():
            experts = {v["expert"] for v in vals}
            if len(experts) >= 2:
                rows.extend(vals)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    est_df = build_icc_df(snapshot["estimations"], "estimated_seconds")
    all_completions = snapshot.get("passes", []) + snapshot.get("fails", [])
    comp_df = build_icc_df(all_completions, "server_elapsed_seconds")

    est_icc = compute_icc(est_df)
    comp_icc = compute_icc(comp_df)

    chart_items = []
    if est_icc[0] is not None:
        icc_val, ci_lo, ci_hi, n_tasks, _ = est_icc
        chart_items.append(
            {
                "label": "Estimations",
                "icc": round(float(icc_val), 3),
                "ci_lo": round(float(ci_lo), 3),
                "ci_hi": round(float(ci_hi), 3),
                "n": int(n_tasks),
                "color": "tealDark",
            }
        )
    if comp_icc[0] is not None:
        icc_val, ci_lo, ci_hi, n_tasks, _ = comp_icc
        chart_items.append(
            {
                "label": "Completions",
                "icc": round(float(icc_val), 3),
                "ci_lo": round(float(ci_lo), 3),
                "ci_hi": round(float(ci_hi), 3),
                "n": int(n_tasks),
                "color": "coral",
            }
        )

    return {
        "chart_type": "iccGauge",
        "version": 1,
        "data": {"items": chart_items},
        "options": {"title": "Inter-rater reliability (ICC)"},
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render ICC gauge matplotlib figure from chart JSON data."""
    items = chart_data["data"]["items"]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Threshold zones
    ax.axvspan(-0.1, 0.50, alpha=0.08, color=_ZONE_POOR)
    ax.axvspan(0.50, 0.75, alpha=0.08, color=_ZONE_MODERATE)
    ax.axvspan(0.75, 0.90, alpha=0.08, color=_ZONE_GOOD)
    ax.axvspan(0.90, 1.05, alpha=0.08, color=_ZONE_EXCELLENT)

    # Threshold labels at top
    for x_pos, label in [
        (0.25, "Poor"),
        (0.625, "Moderate"),
        (0.825, "Good"),
        (0.975, "Excellent"),
    ]:
        ax.text(
            x_pos,
            2.3,
            label,
            ha="center",
            fontsize=8,
            color=_TEXT_MUTED,
            style="italic",
        )

    # Threshold lines
    for t in [0.50, 0.75, 0.90]:
        ax.axvline(t, color=_TEXT_MUTED, linestyle=":", alpha=0.4, linewidth=0.8)

    y_positions = [0.6, -0.6]
    text_y = [1.3, -1.3]
    text_va = ["bottom", "top"]

    for i, item in enumerate(items):
        icc_val = item["icc"]
        ci_lo = item["ci_lo"]
        ci_hi = item["ci_hi"]
        color = _COLOR_MAP.get(item["color"], _TEAL_DARK)

        ax.errorbar(
            icc_val,
            y_positions[i],
            xerr=[[icc_val - ci_lo], [ci_hi - icc_val]],
            fmt="D",
            color=color,
            capsize=8,
            capthick=2,
            markersize=10,
            zorder=5,
        )
        ax.text(
            icc_val,
            text_y[i],
            f"{item['label']}\nICC = {icc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], N = {item['n']}",
            ha="center",
            va=text_va[i],
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2.5, 2.8)
    ax.set_yticks([])
    ax.set_xlabel("ICC(1,1)", fontsize=10)
    ax.grid(axis="x", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate ICC gauge plot")
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--model-runs", default=None, help="model_runs.parquet")
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
