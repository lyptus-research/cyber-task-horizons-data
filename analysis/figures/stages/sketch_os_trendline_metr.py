"""Sketch: existing os_main_trendline + METR overlay elements.

Wraps plot_os_trendline.compute() and reuses its data, then renders with
METR additions: grey 16h unreliable band, real CTH anchor tasks, cleaner
2026 label de-collision.

Outputs to figures/out/sketches/ so we can iterate without touching committed
plot. Re-run cheap (no fitting).
"""
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    coerce_floats,
    load_params,
    save_png,
)
from figures.stages.plot_os_trendline import compute  # noqa: E402
from lib.lyptus_style import COLORS  # noqa: E402


# METR-style elements
UNRELIABLE_MIN = 12 * 60  # 12h - principled cutoff (last marginally-resolved bin)
# Band height = log(Y_CAP_MIN / UNRELIABLE_MIN); doubling visual height on the
# log y-axis means squaring the multiplier (1.25 -> 1.5625).
Y_CAP_MIN = UNRELIABLE_MIN * 1.5625
Y_TICKS = [0.25, 1, 6, 30, 60, 240, 720]
Y_LABELS = ["15s", "1m", "6m", "30m", "1h", "4h", "12h"]

# Real CTH anchor tasks (representative descriptions, chosen at canonical y tiers)
ANCHOR_TASKS = [
    # Bottom anchor pushed up to 1m so it doesn't collide with GPT-2's marker
    (1.0, "Cybersec MCQ"),
    (4.0, "Intro CTF challenge"),
    (15.0, "Reverse simple binary"),
    (60.0, "Write memory-safety fuzzer"),
    (240.0, "Pickle deserialization exploit"),
    (720.0, "Multi-step memory corruption"),
]

INCLUDED_COLOR = "#264653"  # dark slate - in 2024+ trendline
EXCLUDED_COLOR = "#b0b0b0"  # grey - excluded (pre-2024 or non-SOTA)
OS_COLOR = COLORS.get("orange", "#f4a261")
HERO_COLOR = COLORS.get("coral", "#ff5b5b")

# GPT-5.5 @ 2M only (no 50M overlay): P50 = 5.47h, CI [3.45h, 10.45h].
# Computed via bootstrap on 2M-baseline scoring set (1000 reps).
GPT55_2M_P50_MIN = 328.0       # 5.47h
GPT55_2M_CI_LO = 207.0         # 3.45h
GPT55_2M_CI_HI = 627.0         # 10.45h


def _label_de_collide(
    points: list[tuple[float, float, str]],
    min_gap: float = 0.30,
    x_window_days: float = 90,
):
    """Two-phase label placement: assign a fixed offset, then x-pad collisions.

    Returns (x_label, label_y) tuples. label_y is log2(y_marker) +/- offset
    in log2 minutes; x_label is x_marker pushed right (clockwise) when same
    date+y collision detected.
    """
    points_sorted = sorted(points, key=lambda t: (t[0], -t[1]))  # newer first, higher first
    placed: list[tuple[float, float]] = []
    out = []
    for i, (x, p50, name) in enumerate(points_sorted):
        log2_y = np.log2(max(p50, 0.2))
        candidate_y = log2_y
        candidate_x = x
        direction = 1 if i % 2 == 0 else -1
        attempts = 0
        while attempts < 15:
            collision = False
            for px, plog2 in placed:
                if abs(candidate_x - px) < x_window_days and abs(candidate_y - plog2) < min_gap:
                    collision = True
                    break
            if not collision:
                break
            # First try vertical shifts; if too many attempts, jitter x too
            if attempts < 8:
                candidate_y += direction * min_gap * 1.05
            else:
                candidate_x += 18  # ~18 days to the right
                candidate_y = log2_y
            attempts += 1
        placed.append((candidate_x, candidate_y))
        out.append((x, p50, name, candidate_x, 2**candidate_y))
    return out


def render(chart_data: dict, output: str, params: dict) -> None:
    models = chart_data["data"]["models"]
    os_models = chart_data["data"]["os_models"]
    trendline = chart_data.get("trendline", {})
    trendline_2024 = chart_data.get("trendline_2024", {})

    os_set = {m["name"] for m in os_models}

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.5))

    # ------- Grey unreliable band (METR-style) -------
    # Clip the band to the trendline's x range so it doesn't extend past
    # the rightmost projected point.
    band_xmin = mdates.date2num(datetime(2022, 1, 1))
    band_xmax = mdates.date2num(datetime(2027, 2, 1))
    if "dates" in trendline and trendline["dates"]:
        t_dates_all = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        band_xmin = mdates.date2num(min(t_dates_all))
        band_xmax = mdates.date2num(max(t_dates_all))
    ax.fill_between(
        [band_xmin, band_xmax],
        UNRELIABLE_MIN,
        Y_CAP_MIN,
        color="#aaa",
        alpha=0.18,
        zorder=0,
    )
    # Caption: anchor on left side of visible chart (~2023) so it doesn't
    # collide with GPT-5.5's saturated star on the right.
    ax.text(
        mdates.date2num(datetime(2023, 6, 1)),
        np.sqrt(UNRELIABLE_MIN * Y_CAP_MIN),
        "P50 above 12h is unreliable — only 3 tasks in our suite extend beyond 12h",
        fontsize=9,
        color="#555",
        ha="center",
        va="center",
        style="italic",
        zorder=1,
    )

    # ------- Bootstrap CI band around the 2019+ trendline -------
    if "dates" in trendline and "ci_lower" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        ci_lo = [max(v, 0.18) for v in coerce_floats(trendline["ci_lower"])]
        ci_hi = [min(v, Y_CAP_MIN) for v in coerce_floats(trendline["ci_upper"])]
        ax.fill_between(
            t_mpl,
            ci_lo,
            ci_hi,
            color="#264653",
            alpha=0.08,
            zorder=1,
            label="2019+ 95% CI",
        )

    # ------- 2019+ full-range trendline (subtle, in slate grey) -------
    if "dates" in trendline and "fit" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        t_fit = [min(v, Y_CAP_MIN) for v in coerce_floats(trendline["fit"])]
        dt_2019 = trendline.get("doubling_time_days", 0)
        label_2019 = (
            f"2019+ trend (DT = {dt_2019 / 30.44:.1f} months)"
            if dt_2019 else "2019+ trend"
        )
        ax.plot(
            t_mpl,
            t_fit,
            color="#264653",
            linestyle="-",
            linewidth=1.8,
            alpha=0.45,
            zorder=3,
            label=label_2019,
        )

    # ------- 2024+ trendline (METR uses a single dashed line) -------
    if "dates" in trendline_2024 and "fit" in trendline_2024:
        t2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline_2024["dates"]]
        t2_mpl = [mdates.date2num(d) for d in t2_dates]
        t2_fit = [min(v, Y_CAP_MIN) for v in coerce_floats(trendline_2024["fit"])]
        dt_2024 = trendline_2024.get("doubling_time_days", 0)
        ax.plot(
            t2_mpl,
            t2_fit,
            color=HERO_COLOR,
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            zorder=4,
            label=f"2024+ trend (DT = {dt_2024 / 30.44:.1f} months)",
        )

    # (Anchor task labels removed per design feedback.)

    # ------- Model scatter -------
    # Highlight all state-of-the-art models. Any SOTA model post-2024 is in
    # the 2024+ trendline fit by definition; any pre-2024 SOTA model is in
    # the 2019+ trendline fit; distinguishing them in the legend was redundant.
    label_points = []
    for m in models:
        if m["name"] in os_set:
            continue
        rd_str = m.get("release", "")
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        if m["name"] == "GPT-5.5":
            p50 = GPT55_2M_P50_MIN
        else:
            p50 = m["p50"]
        saturated = p50 > UNRELIABLE_MIN
        p50_display = min(p50, UNRELIABLE_MIN * 0.96)

        is_hero = m["name"] == "GPT-5.5"
        is_sota = bool(m.get("frontier"))
        if is_hero:
            color = HERO_COLOR
        elif is_sota:
            color = INCLUDED_COLOR
        else:
            color = EXCLUDED_COLOR
        alpha = 1.0 if is_hero else (0.45 if saturated else (0.95 if is_sota else 0.5))
        ms = 220 if is_hero else (90 if is_sota else 60)
        marker = "*" if is_hero else "o"

        ax.scatter(
            rd_mpl,
            p50_display,
            s=ms,
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolor="white",
            linewidth=1.5 if is_hero else 0.9,
            zorder=15 if is_hero else 5,
        )

        # CI bars
        if is_hero:
            # GPT-5.5 @ 2M bootstrap CI: [3.45h, 10.45h] - sits inside chart
            ax.plot(
                [rd_mpl, rd_mpl],
                [GPT55_2M_CI_LO, GPT55_2M_CI_HI],
                color=color,
                alpha=0.6,
                linewidth=2,
                zorder=14,
            )
        elif not saturated:
            ci_lo = max(m.get("ci_lo", p50 * 0.5), 0.2)
            ci_hi = min(m.get("ci_hi", p50 * 2.0), UNRELIABLE_MIN * 0.95)
            ax.plot(
                [rd_mpl, rd_mpl],
                [ci_lo, ci_hi],
                color=color,
                alpha=alpha * 0.35,
                linewidth=1.2,
                zorder=4,
            )

        # Curate labels: too many late-2025/2026 models cause pileup.
        # Show only headline frontier + hero + a few historical anchors.
        HEADLINE_LABELS = {
            "GPT-2", "GPT-3", "GPT-3.5", "GPT-4o", "Claude 3 Opus",
            "Opus 4", "o3", "GPT-5.3 Codex", "Opus 4.6", "Sonnet 4.6",
            "GPT-5.5",
        }
        if m["name"] in HEADLINE_LABELS or is_hero:
            label_points.append((rd_mpl, p50_display, m["name"], color, is_hero))

    # De-collide labels (separate from markers; markers stay at p50_display)
    de_coll_input = [(x, p, n) for (x, p, n, _, _) in label_points]
    de_collided = _label_de_collide(de_coll_input, min_gap=0.40, x_window_days=150)
    color_lookup = {n: (c, h) for (_, _, n, c, h) in label_points}

    # Decide left-vs-right text alignment based on neighbour proximity.
    # If the next marker (chronologically) is < 200 days away AND at similar
    # y level, place THIS label to the left of its marker so it doesn't
    # collide with the next marker's text.
    marker_lookup = {n: (x, p) for (x, p, n, _, _) in label_points}

    def _place_left(name: str) -> bool:
        x_self, y_self = marker_lookup[name]
        for other_name, (x_o, y_o) in marker_lookup.items():
            if other_name == name:
                continue
            if 0 < x_o - x_self < 200 and abs(np.log2(y_self) - np.log2(y_o)) < 0.7:
                return True
        return False

    # Render labels with leader lines from marker to label position.
    for x_marker, p_marker, name, x_label, label_y in de_collided:
        color, is_hero = color_lookup[name]
        if is_hero:
            two_m_h = GPT55_2M_P50_MIN / 60
            ax.annotate(
                f"GPT-5.5 @ 2M: {two_m_h:.1f}h",
                (x_marker, p_marker),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=9.5,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                zorder=16,
            )
            # Arrow from the GPT-5.5 marker up into the band, with the
            # off-scale caption sitting at the geometric centre of the band
            # so it reads cleanly inside the (now taller) unreliable zone.
            band_centre_y = np.sqrt(UNRELIABLE_MIN * Y_CAP_MIN)
            ax.annotate(
                "",
                xy=(x_marker, band_centre_y),
                xytext=(x_marker, p_marker * 1.15),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.4,
                    alpha=0.85,
                ),
                zorder=15,
            )
            ax.annotate(
                "10M / 50M push off-scale",
                xy=(x_marker, band_centre_y),
                textcoords="offset points",
                xytext=(-8, 0),
                fontsize=8.5,
                color=color,
                ha="right",
                va="center",
                alpha=0.9,
                zorder=16,
            )
            continue

        on_left = _place_left(name)
        label_anchor_x = x_label - 6 if on_left else x_label
        moved = abs(np.log2(p_marker) - np.log2(label_y)) > 0.05 or abs(x_marker - x_label) > 1
        if moved:
            ax.plot(
                [x_marker, label_anchor_x],
                [p_marker, label_y],
                color=color,
                alpha=0.55,
                linewidth=0.9,
                zorder=5,
            )
        ax.annotate(
            name,
            (label_anchor_x, label_y),
            textcoords="offset points",
            xytext=(-7 if on_left else 7, 0),
            fontsize=8.5,
            color=color,
            alpha=0.95,
            va="center",
            ha="right" if on_left else "left",
            zorder=6,
        )

    # OS model markers removed per design feedback — chart limits to closed-weight frontier.

    # ------- Axes -------
    ax.set_yscale("log")
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_LABELS)
    ax.set_ylim(0.18, Y_CAP_MIN)
    ax.set_ylabel(
        "Task duration (for humans)\nwhere logistic fit predicts AI has 50% chance of succeeding",
        fontsize=9,
        color="#444",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # Chart starts 2022 onwards (pre-2022 models compressed into prehistory;
    # the 2024+ trend is the relevant story).
    ax.set_xlim(
        mdates.date2num(datetime(2022, 1, 1)),
        mdates.date2num(datetime(2027, 2, 1)),
    )
    ax.set_xlabel("LLM release date", fontsize=10)
    ax.grid(alpha=0.15)

    ax.set_title(
        "Time horizon of cybersecurity tasks\ndifferent LLMs can complete 50% of the time",
        fontsize=13,
        pad=10,
    )

    # (Brand removed per design feedback.)

    # Legend - marker colors (SOTA vs non-SOTA vs hero) + trendlines
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=INCLUDED_COLOR,
               markersize=10, label='State of the art'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=EXCLUDED_COLOR,
               markersize=8, label='Non-SOTA'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=HERO_COLOR,
               markersize=15, label='GPT-5.5'),
    ]
    # Pull trendline labels from existing axes legend entries
    line_handles, line_labels = ax.get_legend_handles_labels()
    legend_handles.extend(line_handles)
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#ddd",
        bbox_to_anchor=(0.99, 0.04),
    )

    plt.tight_layout()
    save_png(fig, output, params)


def main():
    parser = base_parser("Sketch METR-overlay on existing trendline")
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--x-lim-start", default=None)
    parser.add_argument("--x-lim-end", default=None)
    args = parser.parse_args()
    params = load_params(args.params)
    chart_data = compute(args, params)
    render(chart_data, args.output, params)


if __name__ == "__main__":
    main()
