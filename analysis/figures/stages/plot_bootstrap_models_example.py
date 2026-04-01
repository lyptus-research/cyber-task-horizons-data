"""Stage: Side-by-side trendline showing bootstrap-over-models sensitivity.

Left panel: headline trendline with all SOTA models.
Right panel: the same data but with frontier models removed from the SOTA
set (simulating an egregious bootstrap sample), showing the flatter trendline.

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
from lib.trendline import build_agent_summaries  # noqa: E402


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, find egregious bootstrap sample, return chart_data dict.

    Returns chart_data with model points, trendlines for both panels
    (headline and egregious sample), and bootstrap CI bands.
    """
    import matplotlib.dates as mdates
    from datetime import datetime
    from sklearn.linear_model import LinearRegression
    from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region
    from lib.data import assemble_runs
    from lib.trendline import RELEASE_DATES as _rd

    model_runs_df = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)
    runs_df = assemble_runs(model_runs_df, task_diff, args.difficulty_col)

    bootstrap_df = pd.read_parquet(args.bootstrap)
    summaries_df = pd.read_parquet(args.summaries)
    non_frontier = set(summaries_df[~summaries_df["is_sota"]]["agent"].tolist())

    _TIME_SOURCE = "assembled"

    # Build model_data and summaries
    model_data = {}
    for alias, group in runs_df.groupby("alias"):
        model_data[alias] = {_TIME_SOURCE: group.copy()}

    summaries = build_agent_summaries(
        model_data,
        time_source=_TIME_SOURCE,
        bootstrap_results=bootstrap_df,
    )

    # Find the most egregious bootstrap sample by resampling SOTA models
    sota_aliases = [a for a in summaries["agent"].unique() if a not in non_frontier]
    sota_summaries = summaries[summaries["agent"].isin(sota_aliases)].copy()
    sota_summaries = sota_summaries[sota_summaries["release_date"].notna()]
    sota_summaries = sota_summaries[sota_summaries["p50"] > 0]

    X = mdates.date2num(sota_summaries["release_date"].values).reshape(-1, 1)
    y = np.log(sota_summaries["p50"].values)
    agents = sota_summaries["agent"].values

    rng = np.random.default_rng(args.seed)
    worst_dt = 0
    worst_missing = set()

    for _ in range(5000):
        idx = rng.choice(len(agents), size=len(agents), replace=True)
        try:
            reg = LinearRegression().fit(X[idx], y[idx])
            dt = np.log(2) / reg.coef_[0] / 30.44
            if dt > worst_dt and dt < 50:
                worst_dt = dt
                present = set(agents[idx])
                worst_missing = set(agents) - present
        except Exception:
            pass

    print(f"Headline SOTA: {sorted(sota_aliases)}")
    print(f"Egregious sample drops: {sorted(worst_missing)}")
    print(f"Egregious DT: {worst_dt:.1f} months")

    non_frontier_boot = non_frontier | worst_missing

    # Provider classification
    def _provider(alias):
        a = alias.lower()
        if any(k in a for k in ("claude", "opus", "sonnet", "haiku")):
            return "anthropic"
        if any(k in a for k in ("gpt", "o1", "o3", "o4")):
            return "openai"
        if "gemini" in a:
            return "google"
        return "other"

    # Model points
    chart_models = []
    for _, row in summaries.iterrows():
        agent = row["agent"]
        p50_val = row.get("p50")
        if pd.isna(p50_val) or p50_val <= 0:
            continue
        rd_str = _rd.get(
            agent,
            str(pd.Timestamp(row["release_date"]).date())
            if pd.notna(row.get("release_date"))
            else "",
        )
        is_frontier = agent not in non_frontier_boot
        chart_models.append(
            {
                "name": agent,
                "release": rd_str,
                "p50": float(p50_val),
                "ci_lo": float(row.get("p50q0.025", p50_val * 0.5)),
                "ci_hi": float(row.get("p50q0.975", p50_val * 2.0)),
                "provider": _provider(agent),
                "frontier": is_frontier,
                "dropped": agent in worst_missing,
                "score": float(row.get("average", 0)),
                "is_sota_headline": agent not in non_frontier,
            }
        )

    # Compute trendlines and CI for BOTH panels
    rd_dates = {k: pd.Timestamp(v).date() for k, v in _rd.items()}
    epoch = datetime(2019, 1, 1)

    def _compute_trendline(agent_set, label):
        """Compute trendline + CI for a given set of frontier agents."""
        panel_sota = summaries[
            summaries["agent"].isin(agent_set)
            & summaries["release_date"].notna()
            & (summaries["p50"] > 0)
        ].copy()

        if len(panel_sota) < 2:
            return {}

        days_arr = np.array(
            [
                (pd.Timestamp(row["release_date"]).to_pydatetime() - epoch).days
                for _, row in panel_sota.iterrows()
            ]
        )
        ln_p50 = np.log(panel_sota["p50"].values)
        reg_fit = LinearRegression().fit(days_arr.reshape(-1, 1), ln_p50)
        slope = reg_fit.coef_[0]
        dt_days = np.log(2) / slope if slope > 0 else float("inf")

        ss_res = np.sum((ln_p50 - reg_fit.predict(days_arr.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((ln_p50 - np.mean(ln_p50)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        earliest = pd.to_datetime(panel_sota["release_date"]).min().strftime("%Y-%m-%d")
        latest = pd.to_datetime(panel_sota["release_date"]).max().strftime("%Y-%m-%d")

        try:
            _, time_points, lower, upper = compute_bootstrap_confidence_region(
                agent_summaries=panel_sota,
                bootstrap_results=bootstrap_df.copy(),
                release_dates={"date": rd_dates},
                after_date=earliest,
                sota_before_date=latest,
                trendline_end_date="2027-01-01",
                confidence_level=0.95,
                filter_sota=False,
            )
            tp_dates = [str(t.date()) for t in time_points]
            fit_values = [
                float(
                    np.exp(
                        slope * ((pd.Timestamp(t).to_pydatetime() - epoch).days)
                        + reg_fit.intercept_
                    )
                )
                for t in time_points
            ]
            return {
                "dates": tp_dates,
                "fit": fit_values,
                "ci_lower": [float(v) for v in lower],
                "ci_upper": [float(v) for v in upper],
                "doubling_time_days": round(dt_days),
                "doubling_time_months": round(dt_days / 30.44, 1),
                "r2": round(r2, 2),
                "data_start": "2019-01-01",
                "n_models": len(agent_set),
            }
        except Exception as e:
            print(f"Warning: bootstrap CI for {label} failed: {e}")
            return {
                "doubling_time_days": round(dt_days),
                "doubling_time_months": round(dt_days / 30.44, 1),
                "r2": round(r2, 2),
                "n_models": len(agent_set),
            }

    headline_agents = set(sota_aliases)
    boot_agents = headline_agents - worst_missing

    headline_trendline = _compute_trendline(headline_agents, "headline")
    boot_trendline = _compute_trendline(boot_agents, "bootstrap")

    # Compute headline DT for the title
    headline_dt = headline_trendline.get("doubling_time_months", 0)
    reg_headline = LinearRegression().fit(X, y)
    headline_dt = np.log(2) / reg_headline.coef_[0] / 30.44

    chart_data = {
        "chart_type": "trendline",
        "version": 2,
        "data": chart_models,
        "trendline": boot_trendline,
        "trendline_headline": headline_trendline,
        "trendline_2024": {},
        "panels": {
            "headline": {
                "title": f"Headline ({len(sota_aliases)} SOTA models)",
                "subtitle": f"DT = {headline_dt:.1f} months",
                "n_sota": len(sota_aliases),
                "sota_agents": sorted(sota_aliases),
                "non_frontier": sorted(non_frontier),
            },
            "bootstrap": {
                "title": f"Bootstrap sample (drops {len(worst_missing)} SOTA models)",
                "subtitle": f"DT = {worst_dt:.1f} months",
                "dropped": sorted(worst_missing),
                "non_frontier": sorted(non_frontier_boot),
            },
        },
        "options": {
            "title": f"Bootstrap sample (drops {len(worst_missing)} SOTA models)",
        },
    }

    return chart_data


# =============================================================================
# Render: matplotlib figures from chart_data dict (no DataFrames)
# =============================================================================


def _render_trendline_panel(ax, chart_data, panel_key, show_legend=True):
    """Render a single trendline panel from chart_data onto the given axis.

    Uses scatter for model points, ax.plot for trendline, ax.fill_between
    for CI band. Does NOT call METR's plot_horizon_graph.
    """
    import matplotlib.dates as mdates
    from datetime import datetime

    try:
        from lib.lyptus_style import COLORS as _C

        teal = _C.get("teal_dark", "#264653")
        non_frontier_color = "#cccccc"
    except (ImportError, KeyError):
        teal = "#264653"
        non_frontier_color = "#cccccc"

    PROVIDER_COLORS = {
        "anthropic": "#e76f51",
        "openai": "#2a9d8f",
        "google": "#e9c46a",
        "other": "#999999",
    }

    panel_info = chart_data["panels"][panel_key]
    non_frontier_set = set(panel_info.get("non_frontier", []))
    trendline_key = "trendline_headline" if panel_key == "headline" else "trendline"
    trendline = chart_data.get(trendline_key, {})

    models = chart_data["data"]

    # Plot model points
    for m in models:
        rd_str = m["release"]
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = m["p50"]

        is_frontier = m["name"] not in non_frontier_set

        if is_frontier:
            color = PROVIDER_COLORS.get(m["provider"], "#999")
            alpha = 1.0
            ms = 80
            zorder = 5
        else:
            color = non_frontier_color
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

        # Error bars
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

        # Label frontier models
        if is_frontier:
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

    # Draw dropped model X markers (for bootstrap panel only)
    if panel_key == "bootstrap":
        dropped = set(panel_info.get("dropped", []))
        for m in models:
            if m["name"] in dropped and m["release"]:
                rd = datetime.strptime(m["release"], "%Y-%m-%d")
                rd_mpl = mdates.date2num(rd)
                ax.scatter(
                    [rd_mpl],
                    [m["p50"]],
                    marker="x",
                    s=200,
                    c="red",
                    linewidths=3,
                    zorder=10,
                )

    # Draw trendline
    if "dates" in trendline and "fit" in trendline:
        t_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trendline["dates"]]
        t_mpl = [mdates.date2num(d) for d in t_dates]
        t_fit = coerce_floats(trendline["fit"])
        ax.plot(t_mpl, t_fit, color=teal, linewidth=2, alpha=0.7, zorder=4)

        # CI band
        if "ci_lower" in trendline and "ci_upper" in trendline:
            ci_lo = coerce_floats(trendline["ci_lower"])
            ci_hi = coerce_floats(trendline["ci_upper"])
            ax.fill_between(
                t_mpl,
                ci_lo,
                ci_hi,
                color=teal,
                alpha=0.12,
                zorder=1,
            )

    # Title
    title = panel_info.get("title", "")
    subtitle = panel_info.get("subtitle", "")
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11)

    # Formatting
    ax.set_yscale("log")
    ax.set_ylabel("P50 time horizon (minutes)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(alpha=0.15)

    # Y-axis formatting
    yticks = [1, 5, 10, 30, 60, 120, 240, 480]
    ylabels = ["1m", "5m", "10m", "30m", "1h", "2h", "4h", "8h"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    if show_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PROVIDER_COLORS["anthropic"],
                markersize=8,
                label="Anthropic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PROVIDER_COLORS["openai"],
                markersize=8,
                label="OpenAI",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PROVIDER_COLORS["google"],
                markersize=8,
                label="Google",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=non_frontier_color,
                markersize=8,
                label="Non-frontier",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render both panels (headline + egregious bootstrap) as separate figures.

    Main output: the bootstrap panel (the interesting one).
    Headline companion: saved with _headline suffix.
    """
    import matplotlib.pyplot as plt

    # Render the main (bootstrap) panel
    fig_boot, ax_boot = plt.subplots(1, 1, figsize=(10, 7))
    _render_trendline_panel(ax_boot, chart_data, "bootstrap", show_legend=False)
    fig_boot.tight_layout()
    save_png(fig_boot, output, params)
    plt.close(fig_boot)

    # Render the headline companion panel
    fig_head, ax_head = plt.subplots(1, 1, figsize=(10, 7))
    _render_trendline_panel(ax_head, chart_data, "headline", show_legend=True)
    fig_head.tight_layout()

    out_path = Path(output)
    headline_path = out_path.with_stem(out_path.stem + "_headline")
    save_png(fig_head, str(headline_path), params)
    plt.close(fig_head)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot bootstrap-over-models trendline comparison")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet (for data-driven SOTA)",
    )
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
