"""Stage 3b: Generate P50/P80 trendline figures.

Reads the pre-fitted model_summaries parquet (from fit_summaries.py)
and plots the horizon-vs-release-date trendline with confidence bands.

Does NOT refit IRT curves. The canonical P50 values, SOTA determination,
and bootstrap CIs all come from model_summaries + bootstrap parquet.

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
    from datetime import datetime
    from sklearn.linear_model import LinearRegression
    from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region
    from horizon.plot.logistic_alternative_fits import fit_trendline as _metr_fit
    from matplotlib.dates import date2num

    summaries = pd.read_parquet(args.summaries)
    bootstrap_df = pd.read_parquet(args.bootstrap)

    # Build descriptive title
    source_label = args.title_label or "Time Horizon"
    budget_label = ""
    if args.token_budget != "null":
        budget_label = f", {int(args.token_budget) // 1_000_000}M Token Budget"
    pct = args.success_percent
    ordinal = {50: "50th", 80: "80th"}.get(pct, f"{pct}th")
    title = (
        args.title
        or f"{ordinal} Percentile Time Horizon - {source_label}{budget_label}"
    )

    release_dates = _load_release_dates()

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
    epoch = datetime(2019, 1, 1)

    # Per-model points
    chart_models = []
    for _, row in summaries.iterrows():
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
    frontier_summaries = summaries[summaries["is_sota"]].copy()
    rd_dates = {k: pd.Timestamp(v).date() for k, v in release_dates.items()}

    trendline_data = {}
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

        # Handle non-p50 percentiles
        boot_df = bootstrap_df.copy()
        fs_work = frontier_summaries.copy()
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
            fs_work["p50"] = fs_work[p_col]

        try:
            dt_stats, time_points, lower, upper = compute_bootstrap_confidence_region(
                agent_summaries=fs_work,
                bootstrap_results=boot_df,
                release_dates={"date": rd_dates},
                after_date=earliest,
                sota_before_date=latest,
                trendline_end_date="2027-01-01",
                confidence_level=0.95,
                filter_sota=False,
            )

            reg, r2 = _metr_fit(
                fs_work["p50"],
                pd.to_datetime(fs_work["release_date"]),
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
        except Exception as e:
            print(f"Warning: trendline computation failed: {e}")

    # 2024+ trendline
    trendline_2024 = {}
    cutoff_2024 = datetime(2024, 1, 1)
    sota_set = set(frontier_summaries["agent"].tolist())
    recent_days, recent_log2 = [], []
    for _, row in summaries.iterrows():
        agent = row.get("agent", "")
        rd = row.get("release_date")
        p50_val = row.get(p_col)
        if agent not in sota_set or pd.isna(rd) or pd.isna(p50_val) or p50_val <= 0:
            continue
        rd_dt = pd.Timestamp(rd).to_pydatetime()
        if rd_dt >= cutoff_2024:
            recent_days.append((rd_dt - epoch).days)
            recent_log2.append(np.log2(p50_val))

    if len(recent_days) >= 2:
        reg_2024 = LinearRegression()
        reg_2024.fit(np.array(recent_days).reshape(-1, 1), np.array(recent_log2))
        slope_recent = reg_2024.coef_[0]
        dt_recent_days = 1.0 / slope_recent if slope_recent > 0 else float("inf")

        date_range_2024 = pd.date_range("2024-01-01", "2026-09-01", freq="MS")
        trend_days = np.array(
            [(d.to_pydatetime() - epoch).days for d in date_range_2024]
        )
        trend_p50 = [
            float(2 ** (slope_recent * d + reg_2024.intercept_)) for d in trend_days
        ]

        trendline_2024 = {
            "dates": [str(d.date()) for d in date_range_2024],
            "fit": trend_p50,
            "doubling_time_days": round(dt_recent_days),
        }

    chart_data = {
        "chart_type": "trendline",
        "version": 2,
        "data": chart_models,
        "trendline": trendline_data,
        "trendline_2024": trendline_2024,
        "options": {
            "title": title,
            "y_scale": args.y_scale,
            "success_percent": pct,
        },
    }

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render single-panel trendline figure from chart_data.

    Uses plain matplotlib: scatter for model points, ax.plot for trendline,
    ax.fill_between for CI band.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    try:
        from lib.lyptus_style import COLORS as _C

        teal = _C.get("teal_dark", "#264653")
        accent = _C.get("coral", "#ff5b5b")
    except (ImportError, KeyError):
        teal = "#264653"
        accent = "#ff5b5b"

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
    y_scale = options.get("y_scale", "log")
    title = options.get("title", "")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

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

        # Annotation (log panel only)
        dt_2024_days = trendline_2024.get("doubling_time_days", 0)
        if y_scale == "log" and dt_2024_days > 0:
            ax.text(
                0.98,
                0.17,
                f"2024+ doubling time: {dt_2024_days:.0f} days",
                transform=ax.transAxes,
                fontsize=14,
                color=accent,
                ha="right",
                va="bottom",
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

    # Y-scale formatting
    if y_scale == "log":
        ax.set_yscale("log")
        yticks = [1, 5, 10, 30, 60, 120, 240, 480]
        ylabels = ["1m", "5m", "10m", "30m", "1h", "2h", "4h", "8h"]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
    elif y_scale == "linear":
        ax.set_ylim(-5, 480)
        ax.set_yticks([0, 60, 120, 180, 240, 300, 360, 420, 480])
        ax.set_yticklabels(["0", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h"])

    ax.set_ylabel("P50 time horizon (minutes)")
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(alpha=0.15)

    legend = ax.get_legend()
    if legend:
        legend.set_loc("upper left")

    fig.tight_layout()
    save_png(fig, output, params)
    plt.close(fig)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate trendline figure")
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet (canonical P50s and SOTA)",
    )
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--token-budget", default="null")
    parser.add_argument("--success-percent", type=int, default=50)
    parser.add_argument("--y-scale", default="log", choices=["log", "linear"])
    parser.add_argument("--title", default=None)
    parser.add_argument("--title-label", default=None)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
