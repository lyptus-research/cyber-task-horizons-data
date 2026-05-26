"""Stage: P50 trendline with open-source model annotations.

Overlays OS-specific analysis on top of the standard METR trendline:
buffer arrows showing how far behind the closed-source frontier each
open-weight model sits, a 2024+ trendline fit, and open/closed visual
distinction.

Two DVC stages invoke this script with different x-axis limits:
  - os_main_trendline (full date range)
  - os_main_trendline_zoomed (2024-01-01 to 2026-07-01)

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
from datetime import datetime, timedelta
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
from figures.stages._common_data import load_release_dates  # noqa: E402

# Trigger METR path setup so horizon.* imports work
import lib.trendline  # noqa: E402, F401

try:
    from lib.lyptus_style import COLORS, FONT_SANS  # noqa: E402
except (ImportError, KeyError):
    COLORS = {"orange": "#f4a261", "coral": "#ff5b5b", "grid": "#e0dcd0"}
    FONT_SANS = "Helvetica Neue"


# -- Visual layout constants -------------------------------------------------

OS_COLOR = COLORS.get("orange", "#f4a261")
CLOSED_DIM = "#b0c4b0"

_OS_LAYOUT = {
    "GLM-5": {
        "marker": "s",
        "name_offset": (8, 16),
        "buf_offset": (0, 20),
        "name_ha": "left",
    },
    "DeepSeek V3.1": {
        "marker": "D",
        "name_offset": (8, -16),
        "buf_offset": (0, -22),
        "name_ha": "left",
    },
}


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load summaries, compute trendlines and OS buffers, return chart_data dict."""
    from sklearn.linear_model import LinearRegression
    from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region

    summaries = pd.read_parquet(args.summaries)
    bootstrap_df = pd.read_parquet(args.bootstrap)

    non_frontier = set(summaries[~summaries["is_sota"]]["agent"].tolist())

    os_params = params.get("os_models", {})
    os_models = os_params.get("open_weight", ["GLM-5", "DeepSeek V3.1"])

    release_dates = load_release_dates()

    # Build title
    is_zoomed = bool(args.x_lim_start)
    title = "P50 Time Horizon - Human-Derived Task Difficulty, 2M Token Budget"
    if is_zoomed:
        title = "P50 Time Horizon - 2024 onward (OS models highlighted)"

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

    epoch = datetime(2019, 1, 1)

    # Extract P50 and release dates for all models
    p50s = {}
    for _, row in summaries.iterrows():
        agent = row["agent"]
        rd_str = release_dates.get(agent)
        if rd_str and "p50" in row and np.isfinite(row["p50"]):
            p50_min = row["p50"]
            rd = datetime.strptime(rd_str, "%Y-%m-%d")
            p50s[agent] = (rd, p50_min)

    # Compute 2024+ trendline (used for buffer arrows)
    # Cap individual P50s at SATURATION_CAP_MIN to prevent IRT-saturated
    # models (e.g. GPT-5.5 with extrapolated 800+ h) from dragging the fit.
    SATURATION_CAP_MIN = 1440.0  # 1 day
    cutoff_2024 = datetime(2024, 1, 1)
    sota_recent_days, sota_recent_log2 = [], []
    for agent, (rd, p50) in p50s.items():
        if agent not in non_frontier and p50 > 0 and rd >= cutoff_2024:
            sota_recent_days.append((rd - epoch).days)
            sota_recent_log2.append(np.log2(min(p50, SATURATION_CAP_MIN)))

    trendline_2024 = {}
    dt_recent = float("inf")
    if len(sota_recent_days) >= 2:
        reg_recent = LinearRegression()
        reg_recent.fit(
            np.array(sota_recent_days).reshape(-1, 1), np.array(sota_recent_log2)
        )
        slope_recent = reg_recent.coef_[0]
        dt_recent = 1.0 / slope_recent if slope_recent > 0 else float("inf")

        date_range_2024 = pd.date_range("2024-01-01", "2026-09-01", freq="MS")
        trendline_2024 = {
            "dates": [str(d.date()) for d in date_range_2024],
            "fit": [
                float(
                    np.exp(
                        np.log(2) * slope_recent * ((d.to_pydatetime() - epoch).days)
                        + np.log(2) * reg_recent.intercept_
                    )
                )
                for d in date_range_2024
            ],
            "doubling_time_days": round(dt_recent),
            "r2": round(
                float(
                    reg_recent.score(
                        np.array(sota_recent_days).reshape(-1, 1),
                        np.array(sota_recent_log2),
                    )
                ),
                2,
            ),
        }
        # Recompute fit values properly: 2^(slope*days + intercept) since we used log2
        trendline_2024["fit"] = [
            float(
                2
                ** (
                    slope_recent * ((d.to_pydatetime() - epoch).days)
                    + reg_recent.intercept_
                )
            )
            for d in date_range_2024
        ]

        print(
            f"  2024+ trendline: DT={dt_recent:.0f} days ({dt_recent / 30.44:.1f} months)"
        )

    # Build model point list and compute OS buffers
    chart_models = []
    os_chart_models = []
    for _, row in summaries.iterrows():
        p50_val = row.get("p50")
        if pd.isna(p50_val) or p50_val <= 0:
            continue
        agent = row["agent"]
        rd_str = release_dates.get(agent)
        entry = {
            "name": agent,
            "release": rd_str or str(pd.Timestamp(row["release_date"]).date()),
            "p50": float(p50_val),
            "ci_lo": float(row.get("p50q0.025", p50_val * 0.5)),
            "ci_hi": float(row.get("p50q0.975", p50_val * 2.0)),
            "provider": _provider(agent),
            "frontier": bool(row.get("is_sota", False)),
            "score": float(row.get("average", 0)),
        }
        chart_models.append(entry)

        # Compute buffer for OS models using 2024+ fit
        if agent in os_models and rd_str and len(sota_recent_days) >= 2:
            try:
                rd = datetime.strptime(rd_str, "%Y-%m-%d")
                effective_days = (
                    np.log2(p50_val) - reg_recent.intercept_
                ) / slope_recent
                effective_date = epoch + timedelta(days=effective_days)
                buffer_months = (rd - effective_date).days / 30.44
                os_chart_models.append(
                    {
                        **entry,
                        "is_os": True,
                        "buffer_months": round(buffer_months, 1),
                        "matched_date": effective_date.strftime("%Y-%m-%d"),
                        "marker": _OS_LAYOUT.get(agent, {}).get("marker", "s"),
                    }
                )
                print(
                    f"  {agent}: P50={p50_val:.1f}m, buffer={buffer_months:.1f}mo "
                    f"(effective {effective_date.strftime('%Y-%m')})"
                )
            except (ValueError, ZeroDivisionError):
                pass

    # Full-range trendline + bootstrap CI
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

        try:
            dt_stats, time_points, lower, upper = compute_bootstrap_confidence_region(
                agent_summaries=frontier_summaries,
                bootstrap_results=bootstrap_df.copy(),
                release_dates={"date": rd_dates},
                after_date=earliest,
                sota_before_date=latest,
                trendline_end_date="2027-01-01",
                confidence_level=0.95,
                filter_sota=False,
            )

            # Point-estimate regression (ln-space)
            days_arr = np.array(
                [
                    (pd.Timestamp(rd_dates[a]).to_pydatetime() - epoch).days
                    for a in frontier_summaries["agent"]
                ]
            )
            # Cap saturated P50s before fitting (same rationale as 2024+ above)
            capped_p50 = np.minimum(
                frontier_summaries["p50"].values, SATURATION_CAP_MIN
            )
            ln_p50 = np.log(capped_p50)
            reg_full = LinearRegression().fit(days_arr.reshape(-1, 1), ln_p50)
            slope_full = reg_full.coef_[0]
            dt_days = np.log(2) / slope_full if slope_full > 0 else float("inf")
            ss_res = np.sum((ln_p50 - reg_full.predict(days_arr.reshape(-1, 1))) ** 2)
            ss_tot = np.sum((ln_p50 - np.mean(ln_p50)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            tp_dates = [str(t.date()) for t in time_points]
            fit_values = [
                float(
                    np.exp(
                        slope_full * ((pd.Timestamp(t).to_pydatetime() - epoch).days)
                        + reg_full.intercept_
                    )
                )
                for t in time_points
            ]

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
            print(f"Warning: bootstrap CI for OS chart JSON failed: {e}")

    chart_data = {
        "chart_type": "osTrendline",
        "version": 2,
        "data": {
            "models": chart_models,
            "os_models": os_chart_models,
        },
        "trendline": trendline_data,
        "trendline_2024": trendline_2024,
        "options": {
            "title": title,
            "x_lim_start": args.x_lim_start,
            "x_lim_end": args.x_lim_end,
        },
    }

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render trendline with OS annotations from chart_data.

    Custom matplotlib renderer using scatter + trendline line + CI fill_between
    + OS buffer arrows. Does NOT call METR's plot_horizon_graph.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    PROVIDER_COLORS = {
        "anthropic": "#e76f51",
        "openai": "#2a9d8f",
        "google": "#e9c46a",
        "other": "#999999",
    }

    models = chart_data["data"]["models"]
    os_models = chart_data["data"]["os_models"]
    trendline = chart_data.get("trendline", {})
    trendline_2024 = chart_data.get("trendline_2024", {})
    options = chart_data.get("options", {})
    title = options.get("title", "")

    os_set = {m["name"] for m in os_models}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Y-axis cap: IRT P50 explodes for near-saturated models (e.g. GPT-5.5)
    # because the logistic crosses 50% only after the observed-task range.
    # Cap display at 1440 min (1 day) and tag any model whose P50 exceeds that.
    Y_CAP_MINUTES = 1440.0

    # Draw trendline CI band
    if "dates" in trendline and "ci_lower" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        ci_lo = coerce_floats(trendline["ci_lower"])
        ci_hi = coerce_floats(trendline["ci_upper"])
        ax.fill_between(t_mpl, ci_lo, ci_hi, color="#264653", alpha=0.1, zorder=1)

    # Draw main trendline
    if "dates" in trendline and "fit" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        t_fit = coerce_floats(trendline["fit"])
        ax.plot(t_mpl, t_fit, color="#264653", linewidth=2, alpha=0.5, zorder=2)

    # Draw 2024+ trendline
    if "dates" in trendline_2024 and "fit" in trendline_2024:
        t2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline_2024["dates"]]
        t2_mpl = [mdates.date2num(d) for d in t2_dates]
        t2_fit = coerce_floats(trendline_2024["fit"])
        dt_2024 = trendline_2024.get("doubling_time_days", 0)
        ax.plot(
            t2_mpl,
            t2_fit,
            color=OS_COLOR,
            linewidth=2,
            linestyle="-",
            alpha=0.6,
            zorder=2,
            label=f"2024+ trend (DT={dt_2024:.0f}d)",
        )

    # Plot closed-source model points (dimmed)
    for m in models:
        if m["name"] in os_set:
            continue
        rd_str = m.get("release", "")
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = m["p50"]
        saturated = p50 > Y_CAP_MINUTES
        p50_display = min(p50, Y_CAP_MINUTES)

        color = PROVIDER_COLORS.get(m["provider"], "#999")

        # GPT-5.5 gets coral highlight (the campaign hero)
        is_hero = m["name"] == "GPT-5.5"
        marker_color = COLORS.get("coral", "#ff5b5b") if is_hero else color
        marker_size = 240 if is_hero else 80
        marker_alpha = 1.0 if is_hero else 0.5
        marker_zorder = 15 if is_hero else 3
        marker = "*" if is_hero else "o"
        edge_color = "white" if is_hero else np.array([1, 1, 1, 0.7])
        edge_lw = 2 if is_hero else 0.8

        ax.scatter(
            [rd_mpl],
            [p50_display],
            s=marker_size,
            color=marker_color,
            marker=marker,
            alpha=marker_alpha,
            edgecolor=edge_color,
            linewidth=edge_lw,
            zorder=marker_zorder,
        )

        # Error bars (skip for saturated/hero — meaningless when extrapolated)
        if not saturated and not is_hero:
            ci_lo = m.get("ci_lo", p50 * 0.5)
            ci_hi = m.get("ci_hi", p50 * 2.0)
            ax.plot(
                [rd_mpl, rd_mpl],
                [ci_lo, min(ci_hi, Y_CAP_MINUTES)],
                color=color,
                alpha=0.3,
                linewidth=1.5,
                zorder=2,
            )

        # Model label
        if m.get("frontier") or is_hero:
            if is_hero:
                hours = p50 / 60
                label = (
                    f"{m['name']} — saturated at 2M / 10M / 50M budgets\n"
                    f"(P50 ≈ {hours:.0f}h, off-scale ↑)"
                )
                fs = 9
                fw = "bold"
                lc = COLORS.get("coral", "#ff5b5b")
                offset = (-10, -18)
                ha = "right"
            else:
                label = m["name"]
                fs = 7
                fw = "normal"
                lc = color
                offset = (5, 8)
                ha = "left"
            ax.annotate(
                label,
                (rd_mpl, p50_display),
                textcoords="offset points",
                xytext=offset,
                fontsize=fs,
                fontweight=fw,
                color=lc,
                ha=ha,
                alpha=1.0 if is_hero else 0.6,
                zorder=16 if is_hero else 4,
            )

    # Plot OS models with large markers and buffer arrows
    for osm in os_models:
        rd_str = osm.get("release", "")
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = osm["p50"]
        layout = _OS_LAYOUT.get(osm["name"], {})
        marker = layout.get("marker", "s")

        # Large highlighted marker
        ax.scatter(
            [rd_mpl],
            [p50],
            s=220,
            color=OS_COLOR,
            marker=marker,
            edgecolor="white",
            linewidth=2,
            zorder=15,
        )

        # Model name label
        nx, ny = layout.get("name_offset", (0, 14))
        ax.annotate(
            osm["name"],
            (rd_mpl, p50),
            textcoords="offset points",
            xytext=(nx, ny),
            ha=layout.get("name_ha", "center"),
            fontsize=10,
            fontweight="bold",
            color=OS_COLOR,
            zorder=16,
        )

        # Buffer arrow
        buffer_months = osm.get("buffer_months", 0)
        matched_date_str = osm.get("matched_date", "")
        if matched_date_str and buffer_months != 0:
            eff_date = datetime.strptime(matched_date_str, "%Y-%m-%d")
            eff_mpl = mdates.date2num(eff_date)

            xlim = ax.get_xlim()
            arrow_target = max(eff_mpl, xlim[0])

            ax.annotate(
                "",
                xy=(arrow_target, p50),
                xytext=(rd_mpl, p50),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=OS_COLOR,
                    linewidth=2.5,
                    shrinkA=10,
                    shrinkB=2,
                ),
                zorder=13,
            )

            if eff_mpl >= xlim[0]:
                ax.plot(
                    eff_mpl,
                    p50,
                    "o",
                    color=OS_COLOR,
                    markersize=5,
                    zorder=14,
                    alpha=0.7,
                )

            mid_mpl = (rd_mpl + arrow_target) / 2
            bx, by = layout.get("buf_offset", (0, 14))
            ax.annotate(
                f"{buffer_months:.0f} mo",
                (mid_mpl, p50),
                textcoords="offset points",
                xytext=(bx, by),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=OS_COLOR,
                    edgecolor="none",
                    alpha=0.9,
                ),
                zorder=16,
            )

    # Axes formatting
    ax.set_yscale("log")
    ax.set_ylabel("P50 time horizon (minutes)")
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(alpha=0.15)

    yticks = [1, 5, 10, 30, 60, 120, 240, 480, 1440]
    ylabels = ["1m", "5m", "10m", "30m", "1h", "2h", "4h", "8h", "1d"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0.3, Y_CAP_MINUTES * 1.2)

    # Apply x-limits if zoomed
    x_start = options.get("x_lim_start")
    x_end = options.get("x_lim_end")
    if x_start:
        ax.set_xlim(left=mdates.date2num(datetime.strptime(x_start, "%Y-%m-%d")))
    if x_end:
        ax.set_xlim(right=mdates.date2num(datetime.strptime(x_end, "%Y-%m-%d")))

    # Legend
    dt_2024 = trendline_2024.get("doubling_time_days", 0)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CLOSED_DIM,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1,
            label="Closed-source models",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=OS_COLOR,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Open-weight models",
        ),
        Line2D(
            [0],
            [0],
            color=OS_COLOR,
            linewidth=2,
            linestyle="-",
            alpha=0.6,
            label=f"2024+ trend (DT={dt_2024:.0f}d)",
        ),
        Line2D([0], [0], color=OS_COLOR, linewidth=2.5, label="Adaptation buffer"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        edgecolor=COLORS.get("grid", "#e0dcd0"),
    )

    fig.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate trendline with OS model annotations")
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet (canonical P50s and SOTA)",
    )
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--x-lim-start", default=None)
    parser.add_argument("--x-lim-end", default=None)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
