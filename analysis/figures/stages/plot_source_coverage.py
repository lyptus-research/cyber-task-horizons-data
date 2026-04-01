"""Plot human timing source coverage as a waffle chart across difficulty buckets.

Each task is a small square in its difficulty column. Squares are colored
to show which timing sources are available for that task, with multi-source
tasks shown as split squares. This makes overlap visible without needing
a combinatorial legend.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import sys
from collections import defaultdict
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
import matplotlib.patches as mpatches  # noqa: E402

# Source colors
COLOR_COMPLETION = "#264653"
COLOR_ESTIMATE = "#2a9d8f"
COLOR_FIRSTBLOOD = "#e76f51"
COLOR_CENSORED = "#6d597a"

# Square layout
SQUARE_SIZE = 0.7
SQUARE_GAP = 0.15
COLS_PER_BUCKET = 4  # wrap tasks into columns of this width within each bucket


def _assign_bucket(minutes: float, edges: list[float]) -> int:
    """Return bucket index for a given difficulty in minutes."""
    for i in range(len(edges) - 1):
        if minutes < edges[i + 1]:
            return i
    return len(edges) - 1  # last bucket (>max edge)


SOURCE_COLORS = {
    "completion": COLOR_COMPLETION,
    "censored": COLOR_CENSORED,
    "first_blood": COLOR_FIRSTBLOOD,
    "estimate": COLOR_ESTIMATE,
}

# Ordered source list (controls band stacking order, bottom to top)
# Estimate on top since it's the most dominant color.
SOURCE_ORDER = ["estimate", "first_blood", "censored", "completion"]


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, build per-task source sets, return chart_data dict."""

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    # Load data
    task_diff = pd.read_parquet(resolve(args.task_difficulties))
    model_runs = pd.read_parquet(resolve(args.model_runs)) if args.model_runs else None
    eval_tasks = (
        set(model_runs["task_id"].astype(str)) if model_runs is not None else None
    )
    headline = task_diff.dropna(subset=["best_available_minutes"])
    if eval_tasks is not None:
        headline = headline[headline["task_id"].astype(str).isin(eval_tasks)]

    with open(resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)

    # Build best_available dict from task_difficulties
    best_available = {
        row["task_id"]: {
            "minutes": row["best_available_minutes"],
            "source": row["best_available_source"],
        }
        for _, row in headline.iterrows()
    }

    # Per-task benchmark lookup from task_family column
    task_benchmark = headline.set_index("task_id")["task_family"].to_dict()

    # Per-task difficulty from headline set
    task_difficulty = headline.set_index("task_id")["best_available_minutes"].to_dict()
    # Per-task estimate (for effective hours computation)
    task_estimate_minutes = headline.set_index("task_id")["estimate_minutes"].to_dict()

    # Build per-task source sets (availability, not hierarchy winners)
    completion_tasks = {p["task_id"] for p in snapshot.get("passes", [])}
    # Censored = explicit censored sessions + failed attempts (both are
    # right-censored lower bounds on task difficulty).
    # Apply the same 30-min floor used in build_best_available_times so the
    # plot reflects what actually enters the analytical pipeline.
    _CENSORED_MIN_SECONDS = 30 * 60
    censored_tasks = {
        c["task_id"]
        for c in snapshot.get("censored", []) + snapshot.get("fails", [])
        if c.get("server_elapsed_seconds", 0) >= _CENSORED_MIN_SECONDS
    }
    estimation_tasks = {e["task_id"] for e in snapshot.get("estimations", [])}
    # All tasks with first-blood data (from task_difficulties), not just
    # where first-blood won the source hierarchy
    firstblood_tasks = set(
        headline.loc[headline["firstblood_minutes"].notna(), "task_id"]
    )

    headline_tasks = set(task_difficulty.keys())

    # Build per-task source availability
    task_sources = {}
    for task_id in headline_tasks:
        sources = set()
        if task_id in completion_tasks:
            sources.add("completion")
        if task_id in censored_tasks:
            sources.add("censored")
        if task_id in firstblood_tasks:
            sources.add("first_blood")
        if task_id in estimation_tasks:
            sources.add("estimate")
        task_sources[task_id] = sources

    # Build per-task rater counts
    est_raters = defaultdict(set)
    for e in snapshot.get("estimations", []):
        if e.get("estimated_seconds") and e["task_id"] in headline_tasks:
            est_raters[e["task_id"]].add(e.get("user_id", ""))

    comp_raters = defaultdict(set)
    for key in ("passes", "fails"):
        for s in snapshot.get(key, []):
            if s.get("server_elapsed_seconds") and s["task_id"] in headline_tasks:
                comp_raters[s["task_id"]].add(s.get("user_id", ""))

    # Per-task k counts per source
    task_k = {}
    for task_id in headline_tasks:
        k = {}
        if task_id in estimation_tasks:
            k["estimate"] = len(est_raters.get(task_id, set()))
        if task_id in completion_tasks:
            k["completion"] = len(comp_raters.get(task_id, set()))
        if task_id in firstblood_tasks:
            k["first_blood"] = 1
        if task_id in censored_tasks:
            k["censored"] = len(comp_raters.get(task_id, set()))
        task_k[task_id] = k

    # Build per-task k>=2 source sets (for visual k2 markers when --show-k2)
    task_k2_sources = {tid: set() for tid in headline_tasks}
    if args.show_k2:
        for task_id in headline_tasks:
            k2 = set()
            if len(est_raters.get(task_id, set())) >= 2:
                k2.add("estimate")
            if len(comp_raters.get(task_id, set())) >= 2:
                k2.add("completion")
            task_k2_sources[task_id] = k2

    # Bucket tasks
    edges = CALIBRATION_BUCKET_EDGES_MIN
    labels = bucket_labels(edges)
    n_buckets = len(labels)

    buckets = [[] for _ in range(n_buckets)]
    for task_id in headline_tasks:
        bucket_idx = _assign_bucket(task_difficulty[task_id], edges)
        buckets[bucket_idx].append(task_id)

    # Sort tasks within each bucket by visual similarity
    def _sort_key(t):
        sources = task_sources[t]
        k2 = task_k2_sources[t]
        source_sig = tuple(s in sources for s in SOURCE_ORDER)
        k2_sig = tuple(s in k2 for s in SOURCE_ORDER)
        return (source_sig, k2_sig, task_difficulty[t])

    for i, bucket in enumerate(buckets):
        bucket.sort(key=_sort_key)

    # Compute stats
    n_comp = sum(1 for t in headline_tasks if "completion" in task_sources[t])
    n_est = sum(1 for t in headline_tasks if "estimate" in task_sources[t])
    n_fb = sum(1 for t in headline_tasks if "first_blood" in task_sources[t])
    n_cens = sum(1 for t in headline_tasks if "censored" in task_sources[t])

    EST_CAP_SECONDS = 45 * 60

    from lib.corrections import corrected_elapsed, TIMING_CORRECTIONS

    # Filter hours to headline tasks only (this figure is about the headline set)
    comp_hours = 0.0
    for key in ("passes", "fails", "censored"):
        comp_hours += sum(
            corrected_elapsed(s, TIMING_CORRECTIONS) / 3600
            for s in snapshot.get(key, [])
            if s["task_id"] in headline_tasks
            and corrected_elapsed(s, TIMING_CORRECTIONS) > 0
        )

    est_times = [
        corrected_elapsed(s, TIMING_CORRECTIONS)
        for s in snapshot.get("estimations", [])
        if s["task_id"] in headline_tasks
        and corrected_elapsed(s, TIMING_CORRECTIONS) > 0
    ]
    est_capped = [min(t, EST_CAP_SECONDS) for t in est_times]
    est_hours = sum(est_capped) / 3600
    n_missing = len(snapshot.get("estimations", [])) - len(est_times)
    if est_capped and n_missing > 0:
        est_hours += n_missing * (sum(est_capped) / len(est_capped)) / 3600

    # Build chart JSON export
    chart_tasks = []
    for task_id in headline_tasks:
        bucket_idx = _assign_bucket(task_difficulty[task_id], edges)
        sources_list = sorted(
            task_sources.get(task_id, set()),
            key=lambda s: SOURCE_ORDER.index(s)
            if s in SOURCE_ORDER
            else len(SOURCE_ORDER),
        )
        k2_list = sorted(task_k2_sources.get(task_id, set()))
        chart_tasks.append(
            {
                "id": task_id,
                "bucket": labels[bucket_idx],
                "source": best_available[task_id]["source"],
                "sources": sources_list,
                "k": task_k.get(task_id, {}),
                "k2_sources": k2_list,
                "benchmark": task_benchmark.get(task_id, ""),
                "human_minutes": round(task_difficulty[task_id], 2),
            }
        )

    return {
        "chart_type": "waffle",
        "version": 2,
        "data": chart_tasks,
        "stats": {
            "completions": {"tasks": n_comp, "hours": round(comp_hours)},
            "estimates": {
                "tasks": n_est,
                "hours": round(est_hours),
                "effective_hours": round(
                    sum(
                        v / 60
                        for t, v in task_estimate_minutes.items()
                        if t in headline_tasks
                        and "estimate" in task_sources.get(t, set())
                        and pd.notna(v)
                    )
                ),
            },
            "first_blood": {"tasks": n_fb},
            "censored": {"tasks": n_cens},
            "total": {
                "tasks": len(headline_tasks),
                "hours": round(comp_hours + est_hours),
                "effective_hours": round(
                    comp_hours
                    + sum(
                        v / 60
                        for t, v in task_estimate_minutes.items()
                        if t in headline_tasks
                        and "estimate" in task_sources.get(t, set())
                        and pd.notna(v)
                    )
                ),
            },
        },
        "options": {
            "title": "Source coverage across the difficulty spectrum",
            "show_k2": args.show_k2,
        },
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def _draw_task_square(ax, x, y, sources, k2_sources=None, size=SQUARE_SIZE):
    """Draw a single task square, split by source type.

    For single-source tasks: solid color fill.
    For multi-source tasks: split horizontally into equal bands.
    Bands with k>=2 raters get a thick border in a darker shade.

    sources: list of source names present for this task.
    k2_sources: list of source names with k>=2 raters (optional).
    """
    if k2_sources is None:
        k2_sources = []

    k2_set = set(k2_sources)
    bands = [(s, SOURCE_COLORS[s]) for s in SOURCE_ORDER if s in sources]
    if not bands:
        return

    n = len(bands)
    band_height = size / n

    for i, (source_name, color) in enumerate(bands):
        has_k2 = source_name in k2_set
        band_y = y + i * band_height
        rect = mpatches.FancyBboxPatch(
            (x, band_y),
            size,
            band_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.add_patch(rect)

        # k>=2 indicator: inner border (inset rectangle)
        if has_k2:
            inset = size * 0.15
            inner = mpatches.FancyBboxPatch(
                (x + inset, band_y + inset),
                size - 2 * inset,
                band_height - 2 * inset,
                boxstyle="round,pad=0.01",
                facecolor="none",
                edgecolor="white",
                linewidth=1.0,
                alpha=0.85,
            )
            ax.add_patch(inner)


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render source coverage waffle chart from chart JSON data."""
    tasks = chart_data["data"]
    stats = chart_data["stats"]
    show_k2 = chart_data.get("options", {}).get("show_k2", False)

    edges = CALIBRATION_BUCKET_EDGES_MIN
    labels = bucket_labels(edges)
    n_buckets = len(labels)

    # Build per-bucket task lists from chart data, sorted for visual grouping
    # Each task dict has: id, bucket, sources, k2_sources, human_minutes
    bucket_tasks = [[] for _ in range(n_buckets)]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for t in tasks:
        idx = label_to_idx.get(t["bucket"])
        if idx is not None:
            bucket_tasks[idx].append(t)

    # Sort tasks within each bucket by visual similarity (same logic as compute)
    def _sort_key(t):
        sources = set(t.get("sources", []))
        k2 = set(t.get("k2_sources", []))
        source_sig = tuple(s in sources for s in SOURCE_ORDER)
        k2_sig = tuple(s in k2 for s in SOURCE_ORDER)
        return (source_sig, k2_sig, t["human_minutes"])

    for bucket in bucket_tasks:
        bucket.sort(key=_sort_key)

    # Compute layout dimensions
    max_tasks = max((len(b) for b in bucket_tasks), default=0)
    max_rows = (max_tasks + COLS_PER_BUCKET - 1) // COLS_PER_BUCKET
    bucket_width = COLS_PER_BUCKET * (SQUARE_SIZE + SQUARE_GAP) + SQUARE_GAP * 2
    total_width = n_buckets * bucket_width
    total_height = max_rows * (SQUARE_SIZE + SQUARE_GAP) + 2.5  # room for labels

    fig_w = min(14, total_width * 0.6)
    fig_h = min(6, total_height * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw waffle squares
    for bucket_idx, bt in enumerate(bucket_tasks):
        base_x = bucket_idx * bucket_width + SQUARE_GAP
        for task_i, task in enumerate(bt):
            col = task_i % COLS_PER_BUCKET
            row = task_i // COLS_PER_BUCKET
            x = base_x + col * (SQUARE_SIZE + SQUARE_GAP)
            y = row * (SQUARE_SIZE + SQUARE_GAP)
            sources = set(task.get("sources", []))
            k2_sources = task.get("k2_sources", []) if show_k2 else []
            _draw_task_square(ax, x, y, sources, k2_sources=k2_sources)

    # X-axis: bucket labels
    for i, label in enumerate(labels):
        center_x = i * bucket_width + bucket_width / 2
        ax.text(
            center_x,
            -0.8,
            label,
            ha="center",
            va="top",
            fontsize=13,
            color="#444",
        )
        # Task count annotation
        n = len(bucket_tasks[i])
        if n > 0:
            top_y = ((n - 1) // COLS_PER_BUCKET + 1) * (SQUARE_SIZE + SQUARE_GAP)
            ax.text(
                center_x,
                top_y + 0.15,
                str(n),
                ha="center",
                va="bottom",
                fontsize=12,
                color="#888",
                fontweight="bold",
            )

    # Legend
    has_censored = any("censored" in t.get("sources", []) for t in tasks)
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_COMPLETION, label="Expert completions"),
        mpatches.Patch(facecolor=COLOR_ESTIMATE, label="Expert estimates"),
        mpatches.Patch(facecolor=COLOR_FIRSTBLOOD, label="First-blood times"),
    ]
    if has_censored:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_CENSORED, label="Censored completions")
        )
    if show_k2:
        legend_elements.append(
            mpatches.Patch(
                facecolor=COLOR_ESTIMATE,
                edgecolor="white",
                linewidth=2.0,
                label="Inner border = k>=2 raters",
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=False,
        fontsize=12,
        handlelength=1.2,
        handleheight=1,
    )

    # Stats annotation box
    n_comp = stats["completions"]["tasks"]
    comp_hours = stats["completions"]["hours"]
    n_est = stats["estimates"]["tasks"]
    est_hours = stats["estimates"]["hours"]
    n_fb = stats["first_blood"]["tasks"]
    total_tasks = stats["total"]["tasks"]
    total_hours = stats["total"]["hours"]

    stats_text = (
        f"Completions: {n_comp} tasks ({comp_hours:.0f}h)\n"
        f"Estimates: {n_est} tasks ({est_hours:.0f}h)\n"
        f"First-blood: {n_fb} tasks\n"
        f"Total: {total_tasks} tasks ({total_hours:.0f}h)"
    )
    ax.text(
        0.01,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        color="#666",
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fffaf0",
            alpha=0.85,
            edgecolor="#e5dfd6",
        ),
    )

    # Clean up axes
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-1.5, max_rows * (SQUARE_SIZE + SQUARE_GAP) + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot source coverage waffle chart")
    parser.add_argument(
        "--show-k2",
        action="store_true",
        default=False,
        help="Show k>=2 rater indicators (inner borders)",
    )
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument(
        "--model-runs",
        required=True,
        help="model_runs.parquet (for headline filtering)",
    )
    parser.add_argument("--human-snapshot", required=True)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
