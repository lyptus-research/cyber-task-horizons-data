"""Sketch: token_budget_extended_10m with METR-style overlay.

Wraps plot_token_budget.compute() and re-renders the extended figure with:
- Grey 16h unreliable band on Y axis
- Anchor task labels on left
- Sharper GPT-5.5 milestone markers + leader-line labels
- Lyptus brand
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
from figures.stages.plot_token_budget import compute, STILL_RISING, PLATEAU_COLORS  # noqa: E402
from lib.lyptus_style import COLORS  # noqa: E402


UNRELIABLE_MIN = 12 * 60  # 12h - principled cutoff (last marginally-resolved bin)
# Band height = log(Y_CAP_MIN / UNRELIABLE_MIN); doubling visual height on the
# log y-axis means squaring the multiplier (1.25 -> 1.5625).
Y_CAP_MIN = UNRELIABLE_MIN * 1.5625
Y_TICKS = [1, 5, 10, 30, 60, 240, 720]
Y_LABELS = ["1m", "5m", "10m", "30m", "1h", "4h", "12h"]

ANCHOR_TASKS = [
    (4, "Intro CTF challenge"),
    (15, "Reverse simple binary"),
    (60, "Memory-safety fuzzer"),
    (240, "Pickle exploit"),
    (720, "Multi-step memory corruption"),
]

HERO = COLORS.get("coral", "#ff5b5b")
TEAL_DARK = COLORS.get("teal_dark", "#264653")
TEAL_LIGHT = COLORS.get("teal_light", "#5ba8a0")


def _draw_offscale(
    ax,
    variant: str,
    last_x: float,
    last_y: float,
    band_y: float,
    first_offscale_bm: float,
    ext_bm: list,
    ext_p50: list,
    final_h: float,
    ten_m_h: float | None,
) -> None:
    """Render one of 4 off-scale visualisation styles for GPT-5.5 beyond 2M."""

    if variant == "A":
        # Short stub arrow straight up from 2M ring + caption beside it.
        ax.annotate(
            "",
            xy=(last_x, band_y),
            xytext=(last_x, last_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=HERO,
                linewidth=2.5,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=8,
        )
        ax.annotate(
            f"saturates beyond 2M\ntested to 50M, all off-scale",
            xy=(last_x, band_y),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=9,
            fontweight="bold",
            color=HERO,
            ha="left",
            va="top",
            zorder=10,
        )

    elif variant == "B":
        # Vertical arrow at 2M + small dots at every tested budget at band_y.
        ax.annotate(
            "",
            xy=(last_x, band_y),
            xytext=(last_x, last_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=HERO,
                linewidth=2.5,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=8,
        )
        offscale_budgets = [b for b in ext_bm if b >= first_offscale_bm]
        for b in offscale_budgets:
            ax.plot(
                b,
                band_y,
                marker="o",
                markersize=7,
                markerfacecolor=HERO,
                markeredgecolor="white",
                markeredgewidth=1,
                zorder=9,
            )
        ax.annotate(
            "tested 5M – 50M, all off-scale",
            xy=(50.0, band_y),
            textcoords="offset points",
            xytext=(-6, -10),
            fontsize=9,
            fontweight="bold",
            color=HERO,
            ha="right",
            va="top",
            zorder=10,
        )

    elif variant == "C":
        # Faded coral line extension from 2M to 5M (just into grey band) with
        # arrow at end, then text. No traversal across whole band.
        ax.annotate(
            "",
            xy=(first_offscale_bm, band_y),
            xytext=(last_x, last_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=HERO,
                linewidth=2.5,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=8,
        )
        ax.annotate(
            f"tested to 50M\nall off-scale (IRT P50 ≈ {final_h:.0f}h)",
            xy=(first_offscale_bm, band_y),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=9,
            fontweight="bold",
            color=HERO,
            ha="left",
            va="top",
            zorder=10,
        )

    elif variant == "D":
        # Arrow up at 2M ring + a coral shaded zone that overlaps the entire
        # grey unreliable band across 2M-50M (the "off-scale tested range").
        ax.annotate(
            "↑",
            xy=(last_x, band_y),
            textcoords="data",
            xytext=(last_x, band_y),
            fontsize=20,
            fontweight="bold",
            color=HERO,
            ha="center",
            va="center",
            zorder=10,
        )
        # Coral overlay covering the FULL grey unreliable band over 2M-50M.
        ax.fill_between(
            [2.0, 50.0],
            UNRELIABLE_MIN,
            Y_CAP_MIN,
            color=HERO,
            alpha=0.22,
            zorder=2,
        )
        # Label placed INSIDE the coral overlay band, centered, in dark text.
        ax.annotate(
            "GPT-5.5 saturates beyond 2M — tested to 50M, all off-scale",
            xy=((2.0 * 50.0) ** 0.5, np.sqrt(UNRELIABLE_MIN * Y_CAP_MIN)),
            xycoords="data",
            fontsize=9.5,
            fontweight="bold",
            color="#222",
            ha="center",
            va="center",
            zorder=11,
        )


_DARKER_PLATEAU = {
    "o3": "#7a7a7a",
    "Opus 4": "#7a7a7a",
    "Gemini 2.5 Pro": "#7a7a7a",
    "DeepSeek V3.1": "#7a7a7a",
    "Claude 3 Opus": "#7a7a7a",
    "GPT-4o": "#7a7a7a",
}


def _get_color(alias: str) -> str:
    if alias == "GPT-5.5":
        return HERO
    if alias in STILL_RISING:
        if alias in {"Opus 4.6", "GPT-5.3 Codex"}:
            return TEAL_DARK
        return TEAL_LIGHT
    return _DARKER_PLATEAU.get(alias, PLATEAU_COLORS.get(alias, "#888888"))


def _line_params(alias: str) -> dict:
    if alias == "GPT-5.5":
        return {"linewidth": 3.5, "markersize": 7, "zorder": 7}
    if alias in STILL_RISING:
        lw = 3 if alias in {"Opus 4.6", "GPT-5.3 Codex"} else 2
        ms = 6 if alias in {"Opus 4.6", "GPT-5.3 Codex"} else 4
        return {"linewidth": lw, "markersize": ms, "zorder": 5}
    # plateau models: thicker than before so they're visible against cream BG
    return {"linewidth": 1.8, "markersize": 4, "zorder": 3, "alpha": 0.8}


def render(chart_data: dict, output: str, params: dict) -> None:
    ext = chart_data["extended"]
    if ext is None:
        print("No extended data")
        return

    models = ext["data"]["models"]
    gpt53_ext = ext["data"]["gpt53_extended"]
    gpt55_ext = ext["data"].get("gpt55_extended")

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Grey unreliable band - clip to data range (50K to 50M) so it doesn't
    # extend past the rightmost tested budget into the chart's right padding.
    ax.fill_between(
        [0.05, 50.0],
        UNRELIABLE_MIN,
        Y_CAP_MIN,
        color="#aaa",
        alpha=0.18,
        zorder=0,
    )
    ax.text(
        0.32,  # ~500K budget
        np.sqrt(UNRELIABLE_MIN * Y_CAP_MIN),
        "P50 above 12h is unreliable — only 3 tasks in our suite extend beyond 12h",
        fontsize=9,
        color="#555",
        ha="center",
        va="center",
        style="italic",
        zorder=1,
    )

    # (Anchor task labels removed - distract from the budget story.)

    # 1M-2M green shade (default budget range)
    ax.axvspan(1.0, 2.0, color=COLORS["teal"], alpha=0.06, zorder=0)
    # 2M-50M coral shade (extended GPT-5.5 zone)
    ax.axvspan(2.0, 50.0, color=HERO, alpha=0.04, zorder=0)

    # Plot all model lines (50K -> 2M)
    for m in models:
        alias = m["alias"]
        bm = coerce_floats(m["budgets_m"])
        p50 = [min(v, UNRELIABLE_MIN * 0.96) for v in coerce_floats(m["p50_minutes"])]
        c = _get_color(alias)
        p = _line_params(alias)
        marker = "o" if alias in STILL_RISING else "."

        if alias == "GPT-5.3 Codex":
            ax.plot(bm, p50, marker="o", color=c, **p)
            # Extended 2M -> 10M dashed
            ext_bm = coerce_floats(gpt53_ext["budgets_m"])
            ext_p50 = [min(v, UNRELIABLE_MIN * 0.96) for v in coerce_floats(gpt53_ext["p50_minutes"])]
            ax.plot(
                [2.0] + ext_bm,
                [min(gpt53_ext["last_2m_p50"], UNRELIABLE_MIN * 0.96)] + ext_p50,
                marker="s",
                color=TEAL_DARK,
                linewidth=2.5,
                markersize=6,
                linestyle="--",
                zorder=5,
            )
            # Inline "pass@2" label on the dashed extension. Place it on a
            # mid-segment point (not the first one, which sits next to the
            # 2M endpoint labels for Opus 4.6 / GPT-5.3 Codex / Sonnet 4.6),
            # so the callout sits clearly within the dashed extension zone.
            if len(ext_bm) >= 3:
                _label_idx = len(ext_bm) // 2  # middle of the extension
                ax.annotate(
                    "pass@2",
                    xy=(ext_bm[_label_idx], ext_p50[_label_idx]),
                    textcoords="offset points",
                    xytext=(2, -16),
                    fontsize=8,
                    fontstyle="italic",
                    color=TEAL_DARK,
                    zorder=10,
                    ha="left",
                    va="top",
                )
        elif alias == "GPT-5.5":
            ax.plot(bm, p50, marker="o", color=c, **p)
            if gpt55_ext is not None:
                ext_bm = coerce_floats(gpt55_ext["budgets_m"])
                ext_p50 = coerce_floats(gpt55_ext["p50_minutes"])
                # Only plot extension squares while P50 is BELOW the 16h band.
                # Squares plotted flat at 16h read as a plateau, not saturation.
                last_2m = gpt55_ext["last_2m_p50"]
                onscale_bm = [2.0]
                onscale_p50 = [min(last_2m, UNRELIABLE_MIN * 0.96)]
                first_offscale_bm = None
                first_offscale_p = None
                for b, ppt in zip(ext_bm, ext_p50):
                    if ppt < UNRELIABLE_MIN:
                        onscale_bm.append(b)
                        onscale_p50.append(ppt)
                    else:
                        first_offscale_bm = b
                        first_offscale_p = ppt
                        break

                if len(onscale_bm) > 1:
                    # Dashed: this segment is pass@2 by construction (2M baseline
                    # + retry-at-higher-budget overlay), not pass@1 like the
                    # solid line below 2M.
                    ax.plot(
                        onscale_bm,
                        onscale_p50,
                        marker="s",
                        color=HERO,
                        linewidth=3,
                        markersize=6,
                        linestyle="--",
                        zorder=7,
                    )
                    # Inline "pass@2" label below the dashed segment so it
                    # doesn't collide with the GPT-5.5 ring label or the
                    # off-scale up-arrow.
                    label_x = onscale_bm[1]
                    label_y = onscale_p50[1]
                    ax.annotate(
                        "pass@2",
                        xy=(label_x, label_y),
                        textcoords="offset points",
                        xytext=(2, -16),
                        fontsize=8,
                        fontstyle="italic",
                        color=HERO,
                        zorder=10,
                        ha="left",
                        va="top",
                    )

                # Off-scale visual - variant-controlled via OFFSCALE_VARIANT.
                if first_offscale_bm is not None:
                    last_x = onscale_bm[-1]
                    last_y = onscale_p50[-1]
                    band_y = UNRELIABLE_MIN * 1.05
                    final_h = ext_p50[-1] / 60
                    ten_m_h = next(
                        (p / 60 for b, p in zip(ext_bm, ext_p50) if b == 10.0),
                        None,
                    )
                    variant = globals().get("OFFSCALE_VARIANT", "A")
                    _draw_offscale(
                        ax,
                        variant=variant,
                        last_x=last_x,
                        last_y=last_y,
                        band_y=band_y,
                        first_offscale_bm=first_offscale_bm,
                        ext_bm=ext_bm,
                        ext_p50=ext_p50,
                        final_h=final_h,
                        ten_m_h=ten_m_h,
                    )
        else:
            ax.plot(bm, p50, marker=marker, color=c, **p)

    # Endpoint labels at 2M for all visible models (including plateau).
    HEADLINE = {
        "GPT-5.5", "Opus 4.6", "GPT-5.3 Codex", "Sonnet 4.6", "o3",
        "GPT-5.2 Codex", "GPT-5.1 Codex Max", "GLM-5", "Opus 4",
        "Gemini 2.5 Pro", "DeepSeek V3.1", "Claude 3 Opus", "GPT-4o",
    }
    endpoint_pts = []
    for m in models:
        if m["alias"] not in HEADLINE:
            continue
        last_p50 = min(m["p50_2m"], UNRELIABLE_MIN * 0.96)
        endpoint_pts.append((m["alias"], last_p50))
    endpoint_pts.sort(key=lambda t: -t[1])
    # Greedy sweep: each label sits strictly below the previous by MIN_GAP on
    # the log2 axis so frontier labels (5.5 / Opus 4.6 / 5.3 Codex) do not
    # collide with each other or with the lines that converge near 2M.
    MIN_GAP = 0.38
    placed_y_log = []
    last_placed = None
    for alias, p50 in endpoint_pts:
        log_y = np.log2(p50)
        if last_placed is None:
            target = log_y
        else:
            target = min(log_y, last_placed - MIN_GAP)
        placed_y_log.append(target)
        last_placed = target
        display_y = 2**target
        color = _get_color(alias)
        fw = "bold" if alias in {"GPT-5.5", "Opus 4.6", "GPT-5.3 Codex"} else "normal"
        if alias == "GPT-5.5":
            # GPT-5.5 endpoint at 2M sits inside the grey band and the dashed
            # extension goes up-right from here. Push the label DOWN-LEFT so
            # the coral line does not cross it.
            xytext = (-8, -12)
            ha = "right"
            va = "top"
        else:
            xytext = (6, 0)
            ha = "left"
            va = "center"
        ax.annotate(
            alias,
            (2.0, display_y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=9,
            fontweight=fw,
            color=color,
            va=va,
            ha=ha,
            zorder=6,
        )

    # GPT-5.3 at 10M label
    if gpt53_ext.get("p50_minutes"):
        last_ext_p50 = min(gpt53_ext["p50_minutes"][-1], UNRELIABLE_MIN * 0.96)
        ax.annotate(
            f"GPT-5.3 at 10M\n({gpt53_ext['p50_minutes'][-1] / 60:.1f}h)",
            (10.0, last_ext_p50),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            fontweight="bold",
            color=TEAL_DARK,
            va="center",
            zorder=6,
        )

    # Axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    ax.set_xticklabels(["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M", "20M", "50M"])
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_LABELS)
    ax.set_ylim(0.8, Y_CAP_MIN)
    # Tight x-limit: stop right at 50M so the grey band doesn't leave an
    # orphan rectangle in the top-right past 50M.
    ax.set_xlim(0.04, 55)
    ax.set_xlabel("Token budget", fontsize=10)
    ax.set_ylabel(
        "P50 time horizon\nwhere logistic fit predicts 50% success",
        fontsize=9,
        color="#444",
    )
    ax.set_title(
        "P50 time horizon vs token budget — GPT-5.5 extends past previous frontier",
        fontsize=13,
        pad=10,
    )
    ax.grid(alpha=0.15)

    # (Brand removed per design feedback.)

    plt.subplots_adjust(right=0.95)
    save_png(fig, output, params)


def main():
    parser = base_parser("Sketch METR-style token-budget extended figure")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument(
        "--10m-samples", dest="ten_m_samples", default=None,
    )
    parser.add_argument("--os-cache", default=None)
    parser.add_argument(
        "--gpt55-50m-cache",
        default="../data/keep/gpt55_50m_reruns.json",
    )
    parser.add_argument(
        "--variants",
        default="A,B,C,D",
        help="Comma-separated variants to render (A/B/C/D); output= path's parent gets variant suffix.",
    )
    args = parser.parse_args()
    params = load_params(args.params)
    chart = compute(args, params)

    output_path = Path(args.output)
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    if len(variants) == 1:
        globals()["OFFSCALE_VARIANT"] = variants[0]
        render(chart, str(output_path), params)
    else:
        for v in variants:
            globals()["OFFSCALE_VARIANT"] = v
            variant_out = output_path.with_name(
                f"{output_path.stem}_{v}{output_path.suffix}"
            )
            render(chart, str(variant_out), params)


if __name__ == "__main__":
    main()
