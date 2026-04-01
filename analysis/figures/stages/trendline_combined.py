"""Stage: Combined trendline figure - linear (left) + log (right).

Renders two panels side-by-side as a single publication figure. compute()
produces the chart_data dict with model points, trendlines, and CI bands.
render_png() draws both panels from the dict using plain matplotlib.

Non-frontier set is derived from model_summaries (is_sota column).

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

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
from lib.trendline import _load_release_dates  # noqa: E402


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load summaries and bootstrap, compute trendlines, return chart_data dict."""
    from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region
    from horizon.plot.logistic_alternative_fits import fit_trendline as _metr_fit
    from matplotlib.dates import date2num

    model_summaries = pd.read_parquet(args.summaries)
    bootstrap_df = pd.read_parquet(args.bootstrap)
    pct = args.success_percent

    # Build title
    source_label = args.title_label or "Human-Derived"
    budget_label = ", 2M Token Budget"
    if args.token_budget != "null":
        budget_label = f", {int(args.token_budget) // 1_000_000}M Token Budget"
    title = args.title or (
        f"P50 Time Horizon - {source_label} Task Difficulty{budget_label}"
    )

    # Provider classification
    def _provider(agent: str) -> str:
        a = agent.lower()
        if any(k in a for k in ("claude", "opus", "sonnet", "haiku")):
            return "anthropic"
        if any(k in a for k in ("gpt", "o1", "o3", "o4")):
            return "openai"
        if "gemini" in a:
            return "google"
        return "other"

    p_col = f"p{pct}"
    release_dates = _load_release_dates()

    # Per-model points
    chart_models = []
    for _, row in model_summaries.iterrows():
        p50_val = row.get(p_col, row.get("p50"))
        if pd.isna(p50_val) or p50_val <= 0:
            continue
        chart_models.append(
            {
                "name": row["agent"],
                "release": release_dates.get(
                    row["agent"], str(pd.Timestamp(row["release_date"]).date())
                ),
                "p50": float(p50_val),
                "ci_lo": float(row.get(f"{p_col}q0.025", p50_val * 0.5)),
                "ci_hi": float(row.get(f"{p_col}q0.975", p50_val * 2.0)),
                "provider": _provider(row["agent"]),
                "frontier": bool(row.get("is_sota", False)),
                "score": float(row.get("average", 0)),
            }
        )

    # Trendline regression and bootstrap CI (full range, frontier only)
    frontier_summaries = model_summaries[model_summaries["is_sota"]].copy()
    rd_dates = {k: pd.Timestamp(v).date() for k, v in release_dates.items()}

    boot_df = bootstrap_df.copy()
    if pct != 50:
        suffix_from = f"_p{pct}"
        p50_cols = [c for c in boot_df.columns if c.endswith("_p50")]
        boot_df = boot_df.drop(columns=p50_cols)
        rename_map = {
            c: c.replace(suffix_from, "_p50")
            for c in boot_df.columns
            if c.endswith(suffix_from)
        }
        boot_df = boot_df.rename(columns=rename_map)
        frontier_summaries["p50"] = frontier_summaries[p_col]

    trendline_data = {}
    dt_all_text = ""
    if len(frontier_summaries) >= 2:
        earliest = (
            pd.to_datetime(frontier_summaries["release_date"])
            .min()
            .strftime("%Y-%m-%d")
        )
        latest = (
            pd.to_datetime(frontier_summaries["release_date"])
            .max()
            .strftime("%Y-%m-%d")
        )

        try:
            dt_stats, time_points, lower, upper = compute_bootstrap_confidence_region(
                agent_summaries=frontier_summaries,
                bootstrap_results=boot_df,
                release_dates={"date": rd_dates},
                after_date=earliest,
                sota_before_date=latest,
                trendline_end_date="2027-01-01",
                confidence_level=0.95,
                filter_sota=False,
            )

            reg, r2 = _metr_fit(
                frontier_summaries["p50"],
                pd.to_datetime(frontier_summaries["release_date"]),
                log_scale=True,
            )
            dt_days = (
                1.0 / reg.coef_[0] * np.log(2) if reg.coef_[0] > 0 else float("inf")
            )

            tp_dates = [str(t.date()) for t in time_points]
            tp_x = np.array([date2num(t) for t in time_points])
            fit_values = [float(np.exp(v)) for v in reg.predict(tp_x.reshape(-1, 1))]

            trendline_data = {
                "dates": tp_dates,
                "fit": fit_values,
                "ci_lower": [float(v) for v in lower],
                "ci_upper": [float(v) for v in upper],
                "doubling_time_days": round(dt_days),
                "r2": round(r2, 2),
                "data_start": earliest,
            }
            dt_all_text = f"2019+ DT: {dt_days / 30.44:.1f} months ($R^2$={r2:.2f})"
        except Exception as e:
            print(f"Warning: bootstrap CI for chart JSON failed: {e}")

    # 2024+ accelerated trendline
    trendline_2024 = {}
    cutoff_2024 = pd.Timestamp("2024-01-01")
    sota_recent = frontier_summaries[
        pd.to_datetime(frontier_summaries["release_date"]) >= cutoff_2024
    ]
    dt_recent_text = ""

    if len(sota_recent) >= 2:
        reg2, r2_2024 = _metr_fit(
            sota_recent["p50"],
            pd.to_datetime(sota_recent["release_date"]),
            log_scale=True,
        )
        dt2_days = (
            1.0 / reg2.coef_[0] * np.log(2) if reg2.coef_[0] > 0 else float("inf")
        )

        date_range_2024 = pd.date_range("2024-01-01", "2026-09-01", freq="MS")
        x_2024 = np.array([date2num(d.to_pydatetime()) for d in date_range_2024])
        trendline_2024 = {
            "dates": [str(d.date()) for d in date_range_2024],
            "fit": [float(np.exp(v)) for v in reg2.predict(x_2024.reshape(-1, 1))],
            "doubling_time_days": round(dt2_days),
            "r2": round(r2_2024, 2),
        }
        dt_recent_text = (
            f"2024+ DT: {dt2_days / 30.44:.1f} months ($R^2$={r2_2024:.2f})"
        )

    chart_data = {
        "chart_type": "trendline",
        "version": 2,
        "data": chart_models,
        "trendline": trendline_data,
        "trendline_2024": trendline_2024,
        "options": {
            "title": title,
            "success_percent": pct,
            "dt_all_text": dt_all_text,
            "dt_recent_text": dt_recent_text,
        },
    }

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def _render_panel(ax, chart_data, y_scale, show_legend=True):
    """Render a single trendline panel onto the given axis from chart_data.

    Uses plain matplotlib: scatter for model points, ax.plot for trendline,
    ax.fill_between for CI band.
    """
    import matplotlib.dates as mdates
    from datetime import datetime

    try:
        from lib.lyptus_style import COLORS as _C, FONT_SANS

        teal = _C.get("teal_dark", "#264653")
        accent = _C.get("coral", "#ff5b5b")
        slate = _C.get("slate", "#555")
    except (ImportError, KeyError):
        teal = "#264653"
        accent = "#ff5b5b"
        slate = "#555"
        FONT_SANS = "Helvetica Neue"

    PROVIDER_COLORS = {
        "anthropic": "#e76f51",
        "openai": "#2a9d8f",
        "google": "#e9c46a",
        "other": "#999999",
    }

    models = chart_data["data"]
    trendline = chart_data.get("trendline", {})
    trendline_2024 = chart_data.get("trendline_2024", {})
    options = chart_data.get("options", {})

    # Draw trendline CI band
    if "dates" in trendline and "ci_lower" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        ci_lo = coerce_floats(trendline["ci_lower"])
        ci_hi = coerce_floats(trendline["ci_upper"])
        ax.fill_between(t_mpl, ci_lo, ci_hi, color=teal, alpha=0.1, zorder=1)

    # Draw main trendline
    if "dates" in trendline and "fit" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        t_fit = coerce_floats(trendline["fit"])
        ax.plot(t_mpl, t_fit, color=teal, linewidth=2, alpha=0.5, zorder=2)

    # Draw 2024+ trendline
    if "dates" in trendline_2024 and "fit" in trendline_2024:
        t2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline_2024["dates"]]
        t2_mpl = [mdates.date2num(d) for d in t2_dates]
        t2_fit = coerce_floats(trendline_2024["fit"])
        ax.plot(
            t2_mpl,
            t2_fit,
            color=accent,
            linewidth=3,
            linestyle="-",
            alpha=0.9,
            zorder=2,
        )

    # Plot model points
    for m in models:
        rd_str = m.get("release", "")
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = m["p50"]

        if m.get("frontier"):
            color = PROVIDER_COLORS.get(m["provider"], "#999")
            alpha = 1.0
            ms = 80
            zorder = 5
        else:
            color = "#cccccc"
            alpha = 0.5
            ms = 40
            zorder = 3

        ax.scatter(
            [rd_mpl],
            [p50],
            s=ms,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.8,
            zorder=zorder,
        )

        ci_lo = m.get("ci_lo", p50 * 0.5)
        ci_hi = m.get("ci_hi", p50 * 2.0)
        ax.plot(
            [rd_mpl, rd_mpl],
            [ci_lo, ci_hi],
            color=color,
            alpha=alpha * 0.5,
            linewidth=1.5,
            zorder=zorder - 1,
        )

        if m.get("frontier"):
            ax.annotate(
                m["name"],
                (rd_mpl, p50),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=7,
                color=color,
                alpha=0.8,
                zorder=zorder + 1,
            )

    # Y-scale specific formatting
    if y_scale == "log":
        ax.set_yscale("log")
        yticks = [1, 5, 10, 30, 60, 120, 240, 480]
        ylabels = ["1m", "5m", "10m", "30m", "1h", "2h", "4h", "8h"]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        # DT annotations
        dt_all_text = options.get("dt_all_text", "")
        dt_recent_text = options.get("dt_recent_text", "")
        line_idx = 0
        if dt_all_text:
            ax.text(
                0.98,
                0.08 + line_idx * 0.07,
                dt_all_text,
                transform=ax.transAxes,
                fontsize=12,
                fontfamily=FONT_SANS,
                color=slate,
                ha="right",
                va="bottom",
            )
            line_idx += 1
        if dt_recent_text:
            ax.text(
                0.98,
                0.08 + line_idx * 0.07,
                dt_recent_text,
                transform=ax.transAxes,
                fontsize=12,
                fontfamily=FONT_SANS,
                color=accent,
                ha="right",
                va="bottom",
            )

        # Annotate clipped error bars
        y_upper = ax.get_ylim()[1]
        for m in models:
            ci_hi_val = m.get("ci_hi", 0)
            if ci_hi_val > y_upper and m.get("release"):
                rd = datetime.strptime(m["release"], "%Y-%m-%d")
                rd_mpl = mdates.date2num(rd)
                hours = ci_hi_val / 60
                ax.annotate(
                    f"{hours:.0f}h",
                    xy=(rd_mpl, y_upper * 0.92),
                    fontsize=8,
                    fontfamily=FONT_SANS,
                    color="#666",
                    fontweight="bold",
                    ha="center",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="#fffaf0", ec="#ccc", alpha=0.85
                    ),
                )

        # Remove legend on log panel
        if not show_legend:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    elif y_scale == "linear":
        ax.set_ylim(-5, 480)
        tick_mins = [0, 30, 60, 120, 240, 480]
        tick_labels = ["0", "30m", "1h", "2h", "4h", "8h"]
        ax.set_yticks(tick_mins)
        ax.set_yticklabels(tick_labels)

        # Annotate clipped error bars
        y_upper = 480
        clipped = []
        for m in models:
            ci_hi_val = m.get("ci_hi", 0)
            if ci_hi_val > y_upper and m.get("release"):
                rd = datetime.strptime(m["release"], "%Y-%m-%d")
                hours = ci_hi_val / 60
                clipped.append((rd, hours, m["name"]))

        clipped.sort(key=lambda c: c[0])
        offsets_xy = [(-20, 12), (20, 12)]
        for i, (rd, hours, _agent) in enumerate(clipped):
            x_pos = mdates.date2num(rd)
            ox, oy = offsets_xy[i % len(offsets_xy)]
            ax.annotate(
                f"{hours:.0f}h",
                xy=(x_pos, y_upper),
                fontsize=8,
                fontfamily=FONT_SANS,
                color="#666",
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(ox, oy),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.2),
                bbox=dict(
                    boxstyle="round,pad=0.15", fc="#fffaf0", ec="#ccc", alpha=0.85
                ),
            )

    # Common formatting
    ax.set_ylabel("P50 time horizon (minutes)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(alpha=0.15)


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render combined linear + log trendline figure from chart_data."""
    import matplotlib.pyplot as plt

    title = chart_data.get("options", {}).get("title", "")

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: linear scale (with legend)
    _render_panel(ax_lin, chart_data, "linear", show_legend=True)

    # Right panel: log scale (no legend)
    _render_panel(ax_log, chart_data, "log", show_legend=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_png(fig, output, params)
    plt.close(fig)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate combined trendline figure")
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet (canonical P50s and SOTA)",
    )
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--token-budget", default="null")
    parser.add_argument("--success-percent", type=int, default=50)
    parser.add_argument("--x-lim-start", default=None)
    parser.add_argument("--x-lim-end", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--title-label", default=None)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
