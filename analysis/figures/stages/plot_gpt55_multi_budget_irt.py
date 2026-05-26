"""IRT grid: GPT-5.5 at 2M / 10M / 50M token budgets vs frontier comparators.

Two rows. Top: GPT-5.5 at three token budgets, overlaying 50M failure-retry
data progressively. Bottom: 2M pass@1 comparators (GPT-5.3 Codex, Opus 4.6,
Sonnet 4.6) for context.

Designed to mirror the irt_grid.py aesthetic from the March study: pale bars
for low-n bins, ±2SE error bars, human-time x ticks, 12h unreliable band,
off-scale P50 arrows when the fitted P50 exceeds the reliable measurement
range.
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import base_parser, load_params, save_png  # noqa: E402
from lib.data import assemble_runs  # noqa: E402
from lib.irt import compute_scurve_data  # noqa: E402
from lib.lyptus_style import COLORS, FONT_SERIF  # noqa: E402


REPO_CTH = _NOTEBOOKS_DIR.parent
RERUNS_PATH = _NOTEBOOKS_DIR / "figures" / "data" / "gpt55_50m_reruns.json"
GPT53_10M_PATH = REPO_CTH / "data" / "keep" / "10m_samples.pkl"

UNRELIABLE_HOURS = 12.0
UNRELIABLE_LOG2 = np.log2(UNRELIABLE_HOURS * 60)  # in log2(minutes)

# Human-readable x ticks (matches irt_grid.py palette)
_TICK_LOG2 = [np.log2(v) for v in [1 / 60, 5 / 60, 15 / 60, 1, 5, 15, 60, 240, 960, 3840]]
_TICK_LABELS = ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "4h", "16h", "64h"]


def _build_gpt55_at_budget(template: pd.DataFrame, reruns: dict, budget_tokens: int) -> pd.DataFrame:
    """Per-task scores at the given budget: 2M baseline (from parquet) +
    retry overlay where the rerun beat the budget."""
    out = template.copy().reset_index(drop=True)
    out["score_binarized"] = out["score_binarized"].astype(int)
    for i, row in out.iterrows():
        if out.at[i, "score_binarized"] == 1:
            continue
        rr = reruns.get(str(row["task_id"]))
        if rr is None:
            continue
        if rr["score"] >= 0.7 and rr["tokens"] <= budget_tokens:
            out.at[i, "score_binarized"] = 1
    return out


def _build_gpt53_10m(template: pd.DataFrame) -> pd.DataFrame:
    """GPT-5.3 Codex 2M baseline + 10M retry overlay (pass@2 by construction).

    10m_samples.pkl is a list of {benchmark, task_id, score, total_tokens}
    covering the 83 failures we re-ran at 10M tokens. Score >= 0.7 = pass.
    """
    with open(GPT53_10M_PATH, "rb") as f:
        samples = pickle.load(f)
    flips = {s["task_id"]: s for s in samples if s.get("score", 0) >= 0.7}
    out = template.copy().reset_index(drop=True)
    out["score_binarized"] = out["score_binarized"].astype(int)
    for i, row in out.iterrows():
        if out.at[i, "score_binarized"] == 1:
            continue
        if row["task_id"] in flips:
            out.at[i, "score_binarized"] = 1
    return out


def compute(args, params):
    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)
    runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)

    gpt55_template = runs_df[runs_df["alias"] == "GPT-5.5"].copy()
    if len(gpt55_template) == 0:
        raise RuntimeError("GPT-5.5 not in runs - re-run prepare_runs")

    reruns = json.loads(RERUNS_PATH.read_text())

    # 2M panel is pass@1 baseline from the canonical parquet (no overlay).
    # 10M and 50M panels apply the retry overlay (pass@2 by construction).
    panels_top = [
        ("GPT-5.5 @ 2M", gpt55_template.copy()),
        ("GPT-5.5 @ 10M", _build_gpt55_at_budget(gpt55_template, reruns, 10_000_000)),
        ("GPT-5.5 @ 50M", _build_gpt55_at_budget(gpt55_template, reruns, 50_000_000)),
    ]
    # Bottom row order: GPT-5.3 Codex (2M) | GPT-5.3 Codex @ 10M | Opus 4.6.
    panels_bot = []
    gpt53_template = runs_df[runs_df["alias"] == "GPT-5.3 Codex"]
    if len(gpt53_template):
        panels_bot.append(("GPT-5.3 Codex", gpt53_template))
    if len(gpt53_template) and GPT53_10M_PATH.exists():
        panels_bot.append(("GPT-5.3 Codex @ 10M", _build_gpt53_10m(gpt53_template)))
    opus = runs_df[runs_df["alias"] == "Opus 4.6"]
    if len(opus):
        panels_bot.append(("Opus 4.6", opus))

    # Global bin edges across all panels for consistent x range
    all_dfs = [df for _, df in panels_top + panels_bot]
    all_log2 = np.concatenate([df["log2_human_minutes"].values for df in all_dfs])
    bin_edges = np.arange(np.floor(all_log2.min()) - 0.5, np.ceil(all_log2.max()) + 1.5, 1.0)

    def _to_panel(alias, df, is_hero):
        s = compute_scurve_data(df, bin_edges)
        n_total = len(df)
        n_solved = int(df["score_binarized"].sum())
        return {
            "alias": alias,
            "n_total": n_total,
            "n_solved": n_solved,
            "scurve": s,
            "is_hero": is_hero,
        }

    return {
        "top": [_to_panel(a, d, True) for a, d in panels_top],
        "bot": [_to_panel(a, d, False) for a, d in panels_bot],
        "bin_edges": bin_edges.tolist(),
    }


def _format_p50_label(p50_log2: float | None) -> tuple[str, bool]:
    """Returns (label, off_scale) — off_scale True iff P50 > 12h."""
    if p50_log2 is None or not np.isfinite(p50_log2):
        return "P50 = n/a", False
    p50_min = 2 ** p50_log2
    if p50_min / 60 > UNRELIABLE_HOURS:
        return "P50 off-scale", True
    if p50_min >= 60:
        return f"P50 = {p50_min / 60:.1f}h", False
    return f"P50 = {p50_min:.0f}m", False


def _draw_panel(ax, panel, bin_edges, color):
    s = panel["scurve"]
    is_hero = panel["is_hero"]

    bcenters = s.get("bin_centers", [])
    rates_raw = s.get("empirical_rates", [])
    ns = s.get("bin_counts", [])
    ses = s.get("standard_errors", []) or [None] * len(bcenters)

    bar_full = color + "d9"
    bar_pale = color + "40"
    MIN_N = 5

    # Bars
    centers, rates_pct, counts, sevals = [], [], [], []
    for cx, r, n, se in zip(bcenters, rates_raw, ns, ses):
        if r is None:
            continue
        centers.append(cx)
        rates_pct.append(float(r) * 100)
        counts.append(n)
        sevals.append(se)
    bar_colors = [bar_full if n > MIN_N else bar_pale for n in counts]
    ax.bar(
        centers,
        rates_pct,
        width=0.82,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )

    # ±2SE error bars on bins with valid SE
    valid = [(c, r, se) for c, r, se in zip(centers, rates_pct, sevals) if se is not None and se > 0]
    if valid:
        cx_v, r_v, se_v = zip(*valid)
        ax.errorbar(
            cx_v, r_v, yerr=[2 * s * 100 for s in se_v], fmt="none",
            ecolor=color, alpha=0.85, capsize=2.5, linewidth=1.1, zorder=4,
        )

    # n=N labels above bars
    for cx, rp, n in zip(centers, rates_pct, counts):
        if n <= 0:
            continue
        ax.text(
            cx, min(rp + 4, 104), f"n={n}", ha="center", va="bottom",
            fontsize=6.8, color="#666", alpha=0.75, fontfamily=FONT_SERIF,
        )

    # Fitted curve
    curve_x = s.get("curve_x", [])
    curve_y = s.get("curve_y", [])
    if curve_x:
        ax.plot(curve_x, curve_y, color=color, linewidth=2.2, zorder=5)

    # 12h unreliable band (light grey vertical shading)
    ax.axvspan(UNRELIABLE_LOG2, bin_edges[-1], color="#aaa", alpha=0.10, zorder=1)

    # P50: vertical dashed line if on-scale, else right-edge arrow
    p50_log2 = s.get("p50_log2")
    p50_label, off_scale = _format_p50_label(p50_log2)
    if p50_log2 is not None and np.isfinite(p50_log2):
        if not off_scale:
            ax.axvline(p50_log2, color=color, linestyle="--", alpha=0.6, linewidth=1.4, zorder=4)
        else:
            # Arrow at right edge indicating off-scale P50.
            x_right = bin_edges[-1] - 0.4
            ax.annotate(
                "",
                xy=(x_right, 50),
                xytext=(UNRELIABLE_LOG2 + 0.2, 50),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
                zorder=4,
            )

    # 50% reference line
    ax.axhline(50, color="#999", linestyle=":", linewidth=0.9, alpha=0.55, zorder=1)

    # Title block
    n_solved = panel["n_solved"]
    n_total = panel["n_total"]
    acc = 100 * n_solved / n_total if n_total else 0
    title_main = panel["alias"]
    title_sub = f"{acc:.1f}% ({n_solved}/{n_total})    {p50_label}"
    ax.set_title(
        title_main,
        fontsize=11,
        fontweight="bold",
        color=color if is_hero else "#333",
        loc="left",
        pad=18,
        fontfamily=FONT_SERIF,
    )
    ax.text(
        0.0,
        1.015,
        title_sub,
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
        ha="left",
        va="bottom",
        fontfamily=FONT_SERIF,
    )

    ax.set_ylim(-5, 110)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.grid(axis="y", alpha=0.25, zorder=0)


def render_png(chart_data, output, params):
    """Transposed layout: one column per model, budget variants stacked vertically.

    Matches the interactive gpt55IrtGridByModel renderer so the static PNG
    (served on mobile) and the interactive desktop view tell the same story.

    Layout:
      Col 0 (GPT-5.5):       row 0 = @ 2M, row 1 = @ 10M, row 2 = @ 50M
      Col 1 (GPT-5.3 Codex): row 0 = @ 2M, row 1 = @ 10M
      Col 2 (Opus 4.6):      row 0 = @ 2M
    """
    panels_top = chart_data["top"]
    panels_bot = chart_data["bot"]
    bin_edges = chart_data["bin_edges"]

    hero_color = COLORS["coral"]
    cmp_color = COLORS["teal_dark"]

    def _base_alias(p):
        return p["alias"].split(" @ ")[0]

    columns = []  # list of (model_name, is_hero, [panels])
    by_base = {}
    for p in panels_top + panels_bot:
        base = _base_alias(p)
        if base not in by_base:
            by_base[base] = {"hero": False, "panels": []}
            columns.append(base)
        by_base[base]["panels"].append(p)
    for p in panels_top:
        by_base[_base_alias(p)]["hero"] = True

    n_cols = len(columns)
    n_rows = max(len(by_base[c]["panels"]) for c in columns)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.6 * n_cols, 3.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    visible = [(p, l) for p, l in zip(_TICK_LOG2, _TICK_LABELS) if bin_edges[0] <= p <= bin_edges[-1]]

    for c_idx, model_name in enumerate(columns):
        col = by_base[model_name]
        color = hero_color if col["hero"] else cmp_color
        for r_idx in range(n_rows):
            ax = axes[r_idx][c_idx]
            if r_idx < len(col["panels"]):
                _draw_panel(ax, col["panels"][r_idx], bin_edges, color)
                if visible:
                    pos, lab = zip(*visible)
                    ax.set_xticks(list(pos))
                    ax.set_xticklabels(list(lab), fontsize=9, fontfamily=FONT_SERIF)
                    ax.tick_params(axis="y", labelsize=9)
                # X-label only on the last filled row of this column.
                if r_idx == len(col["panels"]) - 1:
                    ax.set_xlabel("Human time (log scale)", fontsize=10, fontfamily=FONT_SERIF)
                # Y-label only on the leftmost column.
                if c_idx == 0:
                    ax.set_ylabel("Success rate (%)", fontsize=10, fontfamily=FONT_SERIF)
            else:
                ax.axis("off")

    fig.suptitle(
        "Per-bin success rate vs human-time difficulty",
        fontsize=13, fontweight="bold", y=0.995, fontfamily=FONT_SERIF,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    save_png(fig, output, params)


def main():
    parser = base_parser("GPT-5.5 multi-budget IRT grid")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    args = parser.parse_args()
    params = load_params(args.params)
    chart = compute(args, params)
    render_png(chart, args.output, params)


if __name__ == "__main__":
    main()
