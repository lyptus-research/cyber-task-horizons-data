"""Expert estimation effort plot for Appendix C (Dataset Characterisation).

Box plot showing how much time experts spent on estimation sessions,
grouped by task difficulty bucket.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import sys
from pathlib import Path

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
from lib.constants import CALIBRATION_BUCKET_EDGES_MIN, bucket_labels  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Palette
_TEAL = "#2a9d8f"
_TEAL_DARK = "#264653"
_TEXT_MUTED = "#888888"

EST_CAP_SECONDS = 45 * 60


def _assign_bucket(minutes, edges):
    for i in range(len(edges) - 1):
        if minutes < edges[i + 1]:
            return i
    return len(edges) - 1


# =============================================================================
# Compute: all statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute bucketed effort durations, return chart_data dict."""

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    with open(resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)
    with open(resolve(args.best_available)) as f:
        best_available = json.load(f)
    from lib.data import headline_task_set

    task_diff = pd.read_parquet(resolve(args.task_difficulties))
    headline_tasks, _ = headline_task_set(task_diff)

    # Task difficulty lookup
    task_difficulty = {}
    for tid, info in best_available.items():
        if tid in headline_tasks:
            task_difficulty[tid] = info["minutes"]

    # Estimation sessions: collect (difficulty_bucket, session_duration_minutes)
    edges = CALIBRATION_BUCKET_EDGES_MIN
    labels = bucket_labels(edges)
    n_buckets = len(labels)

    bucket_durations = [[] for _ in range(n_buckets)]

    for e in snapshot.get("estimations", []):
        tid = e["task_id"]
        t = e.get("server_elapsed_seconds")
        if tid in task_difficulty and t and t > 0:
            bucket = _assign_bucket(task_difficulty[tid], edges)
            duration_min = min(t, EST_CAP_SECONDS) / 60
            bucket_durations[bucket].append(duration_min)

    # Confidence counts per bucket
    CONF_LEVELS = ["high", "medium", "low"]
    bucket_confidence = [{c: 0 for c in CONF_LEVELS} for _ in range(n_buckets)]

    for e in snapshot.get("estimations", []):
        tid = e["task_id"]
        conf = e.get("estimation_confidence")
        t = e.get("server_elapsed_seconds")
        if tid in task_difficulty and t and t > 0 and conf in CONF_LEVELS:
            bucket = _assign_bucket(task_difficulty[tid], edges)
            bucket_confidence[bucket][conf] += 1

    # Filter to buckets with data
    chart_buckets = []
    for i in range(n_buckets):
        if bucket_durations[i]:
            vals = sorted(bucket_durations[i])
            chart_buckets.append(
                {
                    "label": labels[i],
                    "n": len(vals),
                    "values": [round(float(v), 2) for v in vals],
                    "confidence": bucket_confidence[i],
                }
            )

    return {
        "chart_type": "boxplotV",
        "version": 1,
        "data": {"buckets": chart_buckets},
        "options": {
            "xlabel": "Task difficulty (human_minutes)",
            "ylabel": "Estimation session duration (minutes)",
        },
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render expert effort box plot from chart JSON data."""
    buckets = chart_data["data"]["buckets"]

    plot_data = [b["values"] for b in buckets]
    plot_labels = [b["label"] for b in buckets]
    plot_n = [b["n"] for b in buckets]
    plot_conf = [b.get("confidence", {}) for b in buckets]

    fig, (ax, ax_conf) = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.06},
    )

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=_TEAL_DARK, linewidth=1.5),
        whiskerprops=dict(color=_TEXT_MUTED, linewidth=0.7),
        capprops=dict(color=_TEXT_MUTED, linewidth=0.7),
        flierprops=dict(
            marker="o",
            markerfacecolor=_TEAL,
            markersize=3,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.4,
        ),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(_TEAL)
        patch.set_alpha(0.3)
        patch.set_edgecolor(_TEAL_DARK)
        patch.set_linewidth(0.7)

    ax.set_ylabel("Estimation session duration (minutes)", fontsize=10)
    ax.grid(axis="y", alpha=0.12, color="#c8c3b9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c8c3b9")
    ax.spines["bottom"].set_color("#c8c3b9")
    ax.tick_params(colors=_TEXT_MUTED, labelsize=9)

    # Annotate n per bucket
    for i, n in enumerate(plot_n):
        ax.text(
            i + 1,
            ax.get_ylim()[1] * 0.95,
            f"n={n}",
            ha="center",
            fontsize=8,
            color=_TEXT_MUTED,
        )

    # Confidence stacked bar (bottom panel)
    CONF_LEVELS = ["high", "medium", "low"]
    CONF_COLORS = {"high": "#5cb8ad", "medium": "#dfc078", "low": "#d98a7e"}
    x_pos = range(1, len(plot_conf) + 1)
    for i, pos in enumerate(x_pos):
        total = sum(plot_conf[i].get(c, 0) for c in CONF_LEVELS)
        if total == 0:
            continue
        bottom = 0
        for conf in CONF_LEVELS:
            pct = plot_conf[i].get(conf, 0) / total
            ax_conf.bar(
                pos,
                pct,
                bottom=bottom,
                width=0.45,
                color=CONF_COLORS[conf],
                edgecolor="none",
            )
            bottom += pct

    ax_conf.set_ylim(0, 1)
    ax_conf.set_ylabel("Confidence", fontsize=10)
    ax_conf.set_xlabel("Task difficulty (human_minutes)", fontsize=10)
    ax_conf.set_yticks([0, 0.5, 1.0])
    ax_conf.set_yticklabels(["0%", "50%", "100%"], fontsize=8)
    ax_conf.set_xticks(list(x_pos))
    ax_conf.set_xticklabels(plot_labels, fontsize=9)
    ax_conf.spines["top"].set_visible(False)
    ax_conf.spines["right"].set_visible(False)
    ax_conf.spines["left"].set_visible(False)
    ax_conf.spines["bottom"].set_color("#c8c3b9")
    ax_conf.tick_params(left=False, colors=_TEXT_MUTED, labelsize=8)

    from matplotlib.patches import Patch

    conf_legend = [
        Patch(facecolor=CONF_COLORS[c], label=c.capitalize()) for c in CONF_LEVELS
    ]
    ax_conf.legend(handles=conf_legend, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot expert estimation effort by difficulty")
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--best-available", required=True)
    parser.add_argument("--task-difficulties", required=True)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
