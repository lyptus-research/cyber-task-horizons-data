"""Sketch: METR-style P50 trendline variants.

Three different aesthetics for the 5.5 follow-up post. Outputs to
figures/out/sketches/ so we can iterate without overwriting committed plots.

  v1_metr_style.png - close clone of METR's chart (grey unreliable zone +
                      anchor task labels + faded saturated models).
  v2_lyptus_clean.png - Lyptus-coloured variant, less METR-specific tropes.
  v3_multibudget_overlay.png - METR-style with GPT-5.5 shown at 2M/10M/50M
                               as three stacked markers connected by line.

Run from notebooks/ with DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
"""
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import base_parser, load_params, save_png  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402
from lib.lyptus_style import COLORS  # noqa: E402


# Unreliable measurement zone - tasks above ~16h are extrapolation
UNRELIABLE_MIN = 16 * 60  # 16h in minutes

# Y-axis ticks/labels (METR-style)
Y_TICKS = [0.25, 1, 6, 60, 240, 960]  # 15s, 1m, 6m, 1h, 4h, 16h
Y_LABELS = ["15s", "1m", "6m", "1h", "4h", "16h"]

# Cyber-themed anchor tasks (placeholder examples)
ANCHOR_TASKS = [
    (5,    "Identify hash format"),
    (30,   "Read flag from open file"),
    (180,  "Decode obfuscated string"),
    (900,  "Reverse small binary"),
    (3600, "Craft RCE payload"),
    (14400, "Reproduce CVE end-to-end"),
]

PROVIDER_COLORS = {
    "anthropic": "#e76f51",
    "openai": "#2a9d8f",
    "google": "#e9c46a",
    "other": "#999999",
}


def _provider(agent: str) -> str:
    a = agent.lower()
    if any(k in a for k in ("claude", "opus", "sonnet", "haiku")):
        return "anthropic"
    if any(k in a for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in a:
        return "google"
    return "other"


def _load_models() -> pd.DataFrame:
    s = pd.read_parquet("figures/data/model_summaries_human_2M.parquet")
    rd = load_release_dates()
    s["release"] = s["agent"].map(rd)
    s = s.dropna(subset=["release", "p50"]).copy()
    s["provider"] = s["agent"].apply(_provider)
    return s.sort_values("p50")


def _common_setup(ax, title: str, y_cap: float, x_lim_start: str | None = None):
    """Apply METR-style chrome: grey unreliable band, axes, gridlines."""
    # Grey unreliable-measurement band
    ax.axhspan(UNRELIABLE_MIN, y_cap, color="#888", alpha=0.10, zorder=0)
    ax.text(
        mdates.date2num(datetime(2022, 1, 1)),
        np.sqrt(UNRELIABLE_MIN * y_cap),
        "Measurements above 16h are unreliable\nwith our current task suite",
        fontsize=9,
        color="#666",
        ha="center",
        va="center",
        style="italic",
        zorder=1,
    )

    # Y-axis
    ax.set_yscale("log")
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_LABELS)
    ax.set_ylim(0.15, y_cap)
    ax.set_ylabel(
        "Task duration (for humans)\nwhere the logistic fit predicts the AI\nhas a 50% chance of succeeding",
        fontsize=9,
        color="#444",
    )

    # X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_ha("center")
    ax.set_xlabel("LLM release date", fontsize=10)

    # X limits
    x_start = datetime.strptime(x_lim_start, "%Y-%m-%d") if x_lim_start else datetime(2019, 6, 1)
    x_end = datetime(2026, 8, 1)
    ax.set_xlim(mdates.date2num(x_start), mdates.date2num(x_end))

    ax.set_title(title, fontsize=13, fontweight="normal", color="#222")
    ax.grid(alpha=0.15)


def _plot_anchor_tasks(ax, x_lim_start: str | None = None):
    """METR-style anchor task labels on the LEFT side at appropriate y."""
    if x_lim_start:
        x_start = datetime.strptime(x_lim_start, "%Y-%m-%d")
    else:
        x_start = datetime(2019, 7, 1)
    x_pos = mdates.date2num(x_start)
    for minutes, label in ANCHOR_TASKS:
        if minutes < UNRELIABLE_MIN:  # don't anchor inside grey band
            ax.text(
                x_pos,
                minutes,
                label,
                fontsize=8.5,
                color="#666",
                ha="left",
                va="center",
                zorder=2,
            )


def _scatter_models(ax, df: pd.DataFrame, hero_alias: str = "GPT-5.5"):
    """Scatter points; saturated/hero models get faded with annotation."""
    for _, m in df.iterrows():
        agent = m["agent"]
        provider = m["provider"]
        rd = datetime.strptime(m["release"], "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = float(m["p50"])
        saturated = p50 > UNRELIABLE_MIN
        p50_display = min(p50, UNRELIABLE_MIN * 0.96)

        is_hero = agent == hero_alias
        color = PROVIDER_COLORS.get(provider, "#999")
        alpha = 0.45 if saturated else (1.0 if is_hero else 0.85)

        ax.scatter(
            rd_mpl,
            p50_display,
            s=80 if not is_hero else 130,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=1.2,
            zorder=10 if is_hero else 5,
        )

        # CI error bars (skip if saturated - meaningless)
        if not saturated:
            ci_lo = max(float(m.get("p50q0.025", p50 * 0.5)), 0.2)
            ci_hi = min(float(m.get("p50q0.975", p50 * 2.0)), UNRELIABLE_MIN * 0.95)
            ax.plot(
                [rd_mpl, rd_mpl],
                [ci_lo, ci_hi],
                color=color,
                alpha=alpha * 0.4,
                linewidth=1.2,
                zorder=4,
            )

        # Label
        offset = (6, 8) if not is_hero else (8, -12)
        fs = 9 if is_hero else 7.5
        fw = "bold" if is_hero else "normal"
        label = agent
        if saturated:
            label = f"{agent}\n(saturated)"
        ax.annotate(
            label,
            (rd_mpl, p50_display),
            textcoords="offset points",
            xytext=offset,
            fontsize=fs,
            fontweight=fw,
            color=color,
            alpha=0.7 if saturated else 1.0,
            zorder=11 if is_hero else 6,
        )


def _fit_trendline(df: pd.DataFrame, after: str = "2024-01-01"):
    """Return (dates, fit_values, dt_months) for the 2024+ SOTA trendline."""
    from sklearn.linear_model import LinearRegression

    sub = df[df["is_sota"]].copy()
    sub["release_dt"] = pd.to_datetime(sub["release"])
    sub = sub[sub["release_dt"] >= pd.Timestamp(after)]
    # Cap saturated P50 before fitting
    sub["p50_capped"] = np.minimum(sub["p50"], UNRELIABLE_MIN)

    epoch = datetime(2019, 1, 1)
    days = np.array([(r["release_dt"].to_pydatetime() - epoch).days for _, r in sub.iterrows()])
    log2_p50 = np.log2(sub["p50_capped"].values)

    reg = LinearRegression().fit(days.reshape(-1, 1), log2_p50)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    dt_months = (1 / slope) / 30.44 if slope > 0 else float("inf")

    plot_dates = pd.date_range(after, "2026-09-01", freq="MS")
    plot_days = np.array([(d.to_pydatetime() - epoch).days for d in plot_dates])
    fit = 2 ** (slope * plot_days + intercept)
    return plot_dates, fit, dt_months


def render_v1_metr_style(df: pd.DataFrame, out: str, params: dict):
    """Close clone of METR's chart aesthetic."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    _common_setup(
        ax,
        title="Time horizon of cybersecurity tasks\ndifferent LLMs can complete 50% of the time",
        y_cap=UNRELIABLE_MIN * 1.3,
    )
    _plot_anchor_tasks(ax)
    _scatter_models(ax, df)

    # 2024+ trendline
    dates, fit, dt_m = _fit_trendline(df, "2024-01-01")
    ax.plot(
        [mdates.date2num(d) for d in dates],
        fit,
        color="#264653",
        linestyle="--",
        linewidth=1.8,
        alpha=0.7,
        zorder=3,
    )
    ax.text(
        mdates.date2num(dates[-1]),
        fit[-1] * 1.15,
        f"2024+ trend: DT = {dt_m:.1f} months",
        fontsize=9,
        color="#264653",
        ha="right",
        alpha=0.85,
    )

    # Top-right brand
    ax.text(
        0.99,
        0.97,
        "Lyptus · Cyber Task Horizons",
        transform=ax.transAxes,
        fontsize=10,
        color="#888",
        ha="right",
        va="top",
        fontweight="bold",
    )

    plt.tight_layout()
    save_png(fig, out, params)


def render_v2_lyptus_clean(df: pd.DataFrame, out: str, params: dict):
    """Same data, fewer text annotations, Lyptus palette."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    _common_setup(
        ax,
        title="P50 horizon vs release date — cyber task suite (2M token budget)",
        y_cap=UNRELIABLE_MIN * 1.3,
    )
    _scatter_models(ax, df)

    dates, fit, dt_m = _fit_trendline(df, "2024-01-01")
    ax.plot(
        [mdates.date2num(d) for d in dates],
        fit,
        color=COLORS["coral"],
        linewidth=2.5,
        alpha=0.9,
        zorder=3,
        label=f"2024+ trend (DT = {dt_m:.1f} months)",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    plt.tight_layout()
    save_png(fig, out, params)


def render_v3_multibudget_overlay(df: pd.DataFrame, out: str, params: dict):
    """METR-style with GPT-5.5 plotted at 2M/10M/50M as three stacked markers."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    _common_setup(
        ax,
        title="GPT-5.5 saturates cyber suite at every token budget tested",
        y_cap=UNRELIABLE_MIN * 1.3,
    )
    _plot_anchor_tasks(ax)
    _scatter_models(ax, df)

    # Three GPT-5.5 stacked markers, all in the grey zone
    gpt55_release = datetime.strptime(
        df[df["agent"] == "GPT-5.5"].iloc[0]["release"], "%Y-%m-%d"
    )
    rd_mpl = mdates.date2num(gpt55_release)
    # All saturated → display at top of grey band, offset slightly for visibility
    y_levels = [UNRELIABLE_MIN * 1.05, UNRELIABLE_MIN * 1.12, UNRELIABLE_MIN * 1.20]
    labels = ["@ 2M", "@ 10M", "@ 50M"]
    for y, lbl in zip(y_levels, labels):
        ax.scatter(
            rd_mpl,
            y,
            marker="*",
            s=180,
            color=COLORS["coral"],
            edgecolor="white",
            linewidth=1.5,
            zorder=15,
        )
        ax.annotate(
            f"GPT-5.5 {lbl} (off-scale)",
            (rd_mpl, y),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["coral"],
            va="center",
            zorder=16,
        )
    # Connecting line
    ax.plot(
        [rd_mpl] * 3,
        y_levels,
        color=COLORS["coral"],
        linewidth=2,
        alpha=0.5,
        zorder=14,
    )

    dates, fit, dt_m = _fit_trendline(df, "2024-01-01")
    ax.plot(
        [mdates.date2num(d) for d in dates],
        fit,
        color="#264653",
        linestyle="--",
        linewidth=1.8,
        alpha=0.7,
        zorder=3,
    )

    plt.tight_layout()
    save_png(fig, out, params)


def main():
    parser = base_parser("Sketch METR-style trendline variants")
    parser.add_argument("--output-dir", default="figures/out/sketches")
    # base_parser declares --output as required - satisfy it but ignore
    args = parser.parse_args(sys.argv[1:] + ["--output", "_unused"])
    params = load_params(args.params)

    df = _load_models()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_v1_metr_style(df, str(out_dir / "v1_metr_style.png"), params)
    render_v2_lyptus_clean(df, str(out_dir / "v2_lyptus_clean.png"), params)
    render_v3_multibudget_overlay(df, str(out_dir / "v3_multibudget_overlay.png"), params)
    print(f"Wrote 3 sketches to {out_dir}")


if __name__ == "__main__":
    main()
