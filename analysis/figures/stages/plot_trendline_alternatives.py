"""Stage: Compare exponential, linear, hyperbolic, and logistic trendline fits.

Produces a 4x2 grid (four fit types, two zoom levels). Fits are computed
once per zoom level in compute() and stored in the chart JSON. render_png()
composites individual panels from the pre-computed data using plain
matplotlib (scatter + trendline line + CI fill_between).

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
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


# -- Fit functions (used in compute only) ------------------------------------


def _format_minutes(minutes):
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    if minutes < 60:
        return f"{minutes:.1f}m"
    if minutes < 1440:
        return f"{minutes / 60:.1f}h"
    if minutes < 1440 * 365.25:
        return f"{minutes / 1440:.1f}d"
    return f"{minutes / (1440 * 365.25):.1f}y"


def _hyperbolic_func(x, a, t_agi):
    return np.log(a / (t_agi - x))


def _fit_hyperbolic(X, y_log):
    from scipy.optimize import curve_fit

    try:
        bounds = ([0, X.max().item()], [np.inf, np.inf])
        params, _ = curve_fit(
            _hyperbolic_func,
            X.flatten(),
            y_log,
            bounds=bounds,
            p0=[5, X.max().item() + 365],
            maxfev=5000,
        )
        y_pred = _hyperbolic_func(X.flatten(), *params)
        ss_res = np.sum((y_log - y_pred) ** 2)
        ss_tot = np.sum((y_log - y_log.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return params, r2
    except Exception as e:
        print(f"Hyperbolic fit failed: {e}")
        return None, None


def _logistic_func(x, L, k, t0):
    return L / (1 + np.exp(-k * (x - t0)))


def _fit_logistic(X, y_log):
    """Fit logistic growth in log-P50 space: log(P50(t)) = L / (1 + exp(-k*(t - t0))).

    Consistent with _fit_hyperbolic which also works in log space. The ceiling L
    is in log(minutes); convert to minutes via np.exp(L) for display.
    R² is computed in log space, making it comparable to the exponential R².
    """
    from scipy.optimize import curve_fit

    try:
        bounds = (
            [y_log.max(), 0, X.min().item()],
            [y_log.max() + 10, np.inf, X.max().item() + 365 * 5],
        )
        params, _ = curve_fit(
            _logistic_func,
            X.flatten(),
            y_log,
            bounds=bounds,
            p0=[y_log.max() + 2, 0.01, X.mean().item()],
            maxfev=20000,
        )
        y_pred = _logistic_func(X.flatten(), *params)
        ss_res = np.sum((y_log - y_pred) ** 2)
        ss_tot = np.sum((y_log - y_log.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return params, r2
    except Exception as e:
        print(f"Logistic fit failed: {e}")
        return None, None


def _bootstrap_ci_band(bootstrap_df, sota_agents, release_dates_num, fit_type, x_range):
    """Compute bootstrap CI band for a given fit type from bootstrap P50s."""
    from sklearn.linear_model import LinearRegression

    n_iter = len(bootstrap_df)
    predictions = np.full((n_iter, len(x_range)), np.nan)

    for i in range(n_iter):
        p50s = []
        for agent in sota_agents:
            col = f"{agent}_p50"
            if col in bootstrap_df.columns:
                val = bootstrap_df.iloc[i][col]
                p50s.append(val if pd.notna(val) and val > 0 else np.nan)
            else:
                p50s.append(np.nan)

        p50s = np.array(p50s)
        valid = ~np.isnan(p50s)
        if valid.sum() < 3:
            continue

        X = release_dates_num[valid].reshape(-1, 1)
        y = p50s[valid]

        try:
            if fit_type == "linear":
                reg = LinearRegression().fit(X, y)
                predictions[i] = np.clip(
                    reg.predict(x_range.reshape(-1, 1)), 1e-3, None
                )
            elif fit_type == "hyperbolic":
                y_log = np.log(y.clip(1e-3))
                params, _ = _fit_hyperbolic(X, y_log)
                if params is not None:
                    safe = x_range < params[1] - 5
                    pred = np.full_like(x_range, np.nan)
                    pred[safe] = np.exp(_hyperbolic_func(x_range[safe], *params))
                    predictions[i] = pred
            elif fit_type == "logistic":
                y_log = np.log(y.clip(1e-3))
                params, _ = _fit_logistic(X, y_log)
                if params is not None:
                    predictions[i] = np.clip(
                        np.exp(_logistic_func(x_range, *params)), 1e-3, None
                    )
        except Exception:
            continue

    lo = np.nanpercentile(predictions, 2.5, axis=0)
    hi = np.nanpercentile(predictions, 97.5, axis=0)
    return lo, hi


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute all fit types across zoom levels, return chart_data dict."""
    from matplotlib.dates import date2num, num2date
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

    model_data = {}
    for alias, group in runs_df.groupby("alias"):
        model_data[alias] = {_TIME_SOURCE: group.copy()}

    summaries = build_agent_summaries(
        model_data,
        time_source=_TIME_SOURCE,
        bootstrap_results=bootstrap_df,
    )

    fit_types = ["exponential", "linear", "hyperbolic", "logistic"]
    fit_labels = ["Exponential", "Linear", "Hyperbolic", "Logistic (sigmoid)"]

    zooms = [
        ("Full range", "2019-06-01", "2026-07-01", None),
        ("2024 onward", "2023-10-01", "2026-07-01", pd.Timestamp("2024-01-01")),
    ]

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

    # Model scatter points (shared across all panels)
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
        chart_models.append(
            {
                "name": agent,
                "release": rd_str,
                "p50": float(p50_val),
                "ci_lo": float(row.get("p50q0.025", p50_val * 0.5)),
                "ci_hi": float(row.get("p50q0.975", p50_val * 2.0)),
                "provider": _provider(agent),
                "frontier": bool(agent not in non_frontier),
                "score": float(row.get("average", 0)),
            }
        )

    rd_dates = {k: pd.Timestamp(v).date() for k, v in _rd.items()}

    # Build fit data for each panel
    chart_panels = []
    stats_output = {}

    for zoom_label, x_start, x_end, fit_after in zooms:
        # Compute SOTA subset for this zoom
        sota = summaries[~summaries["agent"].isin(non_frontier)].copy()
        sota = sota[sota["release_date"].notna() & (sota["p50"] > 0)]
        if fit_after is not None:
            sota = sota[pd.to_datetime(sota["release_date"]) >= fit_after]

        X_sota = date2num(sota["release_date"].values)
        y_vals = sota["p50"].values
        y_log = np.log(y_vals.clip(1e-3))

        # Pre-compute fits for this zoom level
        zoom_fits = {}

        # Exponential
        reg_exp = LinearRegression().fit(X_sota.reshape(-1, 1), y_log)
        r2_exp = float(reg_exp.score(X_sota.reshape(-1, 1), y_log))
        dt_days_exp = np.log(2) / reg_exp.coef_[0]
        zoom_fits["exponential"] = {
            "r_squared": round(r2_exp, 3),
            "doubling_time_months": round(dt_days_exp / 30.44, 1),
        }

        # Linear
        reg_lin = LinearRegression().fit(X_sota.reshape(-1, 1), y_vals)
        r2_lin = float(reg_lin.score(X_sota.reshape(-1, 1), y_vals))
        zoom_fits["linear"] = {"r_squared": round(r2_lin, 3)}

        # Hyperbolic
        params_hyp, r2_hyp = _fit_hyperbolic(X_sota.reshape(-1, 1), y_log)
        if params_hyp is not None:
            zoom_fits["hyperbolic"] = {
                "r_squared": round(float(r2_hyp), 3),
                "singularity": num2date(params_hyp[1]).strftime("%B %Y"),
            }
        else:
            zoom_fits["hyperbolic"] = {"r_squared": None, "singularity": None}

        # Logistic (fit in log space — L is the log-minutes ceiling)
        params_log, r2_log = _fit_logistic(X_sota.reshape(-1, 1), y_log)
        if params_log is not None:
            ceiling_minutes = float(np.exp(params_log[0]))
            zoom_fits["logistic"] = {
                "r_squared": round(float(r2_log), 3),
                "ceiling_minutes": ceiling_minutes,
                "ceiling_display": _format_minutes(ceiling_minutes),
                "inflection_date": num2date(params_log[2]).strftime("%B %Y"),
            }
        else:
            zoom_fits["logistic"] = {
                "r_squared": None,
                "ceiling_minutes": None,
                "ceiling_display": None,
                "inflection_date": None,
            }

        zoom_key = "full_range" if zoom_label == "Full range" else "zoomed_2024"
        stats_output[zoom_key] = zoom_fits

        # Build panel data for each fit type
        for fit_type, fit_label in zip(fit_types, fit_labels):
            title = f"{fit_label} - {zoom_label}"
            sota_dates = pd.to_datetime(sota["release_date"])
            data_start = sota_dates.min().strftime("%Y-%m-%d")
            data_end = sota_dates.max().strftime("%Y-%m-%d")

            r2_val = zoom_fits.get(fit_type, {}).get("r_squared")
            panel = {
                "title": title,
                "fit_type": fit_type,
                "zoom": zoom_label,
                "x_start": x_start,
                "x_end": x_end,
                "data_start": data_start,
                "data_end": data_end,
                "r_squared": r2_val,
            }

            x_start_num = date2num(pd.Timestamp(x_start))
            x_end_num = date2num(pd.Timestamp(x_end))
            x_range = np.linspace(x_start_num - 30, x_end_num + 30, 200)

            if fit_type == "exponential":
                fit_y = np.exp(reg_exp.predict(x_range.reshape(-1, 1)))
                fit_dates = [num2date(x).strftime("%Y-%m-%d") for x in x_range]
                panel["fit"] = {"dates": fit_dates, "values": [float(v) for v in fit_y]}

                try:
                    earliest = sota_dates.min().strftime("%Y-%m-%d")
                    latest = sota_dates.max().strftime("%Y-%m-%d")
                    _, tp, lo, hi = compute_bootstrap_confidence_region(
                        agent_summaries=sota,
                        bootstrap_results=bootstrap_df.copy(),
                        release_dates={"date": rd_dates},
                        after_date=earliest,
                        sota_before_date=latest,
                        trendline_end_date=x_end,
                        confidence_level=0.95,
                        filter_sota=False,
                    )
                    panel["ci"] = {
                        "dates": [str(t.date()) for t in tp],
                        "lower": [float(v) for v in lo],
                        "upper": [float(v) for v in hi],
                    }
                except Exception as e:
                    print(f"  CI failed for {title}: {e}")

                dt = zoom_fits["exponential"].get("doubling_time_months")
                r2 = zoom_fits["exponential"].get("r_squared")
                panel["stats"] = (
                    f"Doubling time: {int(dt * 30.44)} days\n{x_start}+ data\nR\u00b2: {r2}"
                )

            elif fit_type == "linear":
                fit_y = np.clip(reg_lin.predict(x_range.reshape(-1, 1)), 1e-3, None)
                fit_dates = [num2date(x).strftime("%Y-%m-%d") for x in x_range]
                panel["fit"] = {"dates": fit_dates, "values": [float(v) for v in fit_y]}

                lo, hi = _bootstrap_ci_band(
                    bootstrap_df,
                    sota["agent"].tolist(),
                    X_sota,
                    "linear",
                    x_range,
                )
                valid = ~np.isnan(lo) & ~np.isnan(hi) & (lo > 0) & (hi > 0)
                if valid.any():
                    panel["ci"] = {
                        "dates": [
                            num2date(x).strftime("%Y-%m-%d") for x in x_range[valid]
                        ],
                        "lower": [float(v) for v in lo[valid]],
                        "upper": [float(v) for v in hi[valid]],
                    }

                r2 = zoom_fits["linear"].get("r_squared")
                panel["stats"] = f"{x_start}+ data\nR\u00b2: {r2}"

            elif fit_type == "hyperbolic":
                r2 = zoom_fits["hyperbolic"].get("r_squared")
                sing = zoom_fits["hyperbolic"].get("singularity")
                if params_hyp is not None:
                    x_range_h = np.linspace(
                        X_sota.min() - 30,
                        min(X_sota.max() + 60, params_hyp[1] - 5),
                        200,
                    )
                    fit_y = np.exp(_hyperbolic_func(x_range_h, *params_hyp))
                    fit_dates = [num2date(x).strftime("%Y-%m-%d") for x in x_range_h]
                    panel["fit"] = {
                        "dates": fit_dates,
                        "values": [float(v) for v in fit_y],
                    }

                    lo, hi = _bootstrap_ci_band(
                        bootstrap_df,
                        sota["agent"].tolist(),
                        X_sota,
                        "hyperbolic",
                        x_range_h,
                    )
                    valid = ~np.isnan(lo) & ~np.isnan(hi) & (lo > 0) & (hi > 0)
                    if valid.any():
                        panel["ci"] = {
                            "dates": [
                                num2date(x).strftime("%Y-%m-%d")
                                for x in x_range_h[valid]
                            ],
                            "lower": [float(v) for v in lo[valid]],
                            "upper": [float(v) for v in hi[valid]],
                        }

                    panel["stats"] = f"R\u00b2: {r2}\nSingularity: {sing}"

            elif fit_type == "logistic":
                r2 = zoom_fits["logistic"].get("r_squared")
                ceiling = zoom_fits["logistic"].get("ceiling_minutes")
                infl = zoom_fits["logistic"].get("inflection_date")
                if params_log is not None:
                    fit_y = np.exp(_logistic_func(x_range, *params_log))
                    fit_dates = [num2date(x).strftime("%Y-%m-%d") for x in x_range]
                    panel["fit"] = {
                        "dates": fit_dates,
                        "values": [float(v) for v in fit_y],
                    }

                    lo, hi = _bootstrap_ci_band(
                        bootstrap_df,
                        sota["agent"].tolist(),
                        X_sota,
                        "logistic",
                        x_range,
                    )
                    valid = ~np.isnan(lo) & ~np.isnan(hi) & (lo > 0) & (hi > 0)
                    if valid.any():
                        panel["ci"] = {
                            "dates": [
                                num2date(x).strftime("%Y-%m-%d") for x in x_range[valid]
                            ],
                            "lower": [float(v) for v in lo[valid]],
                            "upper": [float(v) for v in hi[valid]],
                        }

                    ceiling_str = _format_minutes(ceiling)
                    panel["stats"] = (
                        f"R\u00b2: {r2}\nCeiling: {ceiling_str}\nInflection: {infl}"
                    )

            chart_panels.append(panel)
            print(f"  Computed: {title}")

    chart_data = {
        "chart_type": "trendlineGrid",
        "version": 1,
        "data": {"models": chart_models, "panels": chart_panels},
        "options": {"title": "Trendline Functional Form Comparison"},
        "_stats": stats_output,
    }

    # Write stats JSON if requested
    if args.stats_output:
        out_path = (
            Path(args.stats_output)
            if Path(args.stats_output).is_absolute()
            else _NOTEBOOKS_DIR / args.stats_output
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats_output, f, indent=2)
        print(f"Wrote stats: {out_path}")

    return chart_data


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames)
# =============================================================================


def _render_panel(ax, chart_data, panel, show_legend=True):
    """Render a single trendline panel onto the given axis from chart_data."""
    import matplotlib.dates as mdates
    from datetime import datetime

    try:
        from lib.lyptus_style import COLORS as _C

        teal = _C.get("teal_dark", "#264653")
    except (ImportError, KeyError):
        teal = "#264653"

    PROVIDER_COLORS = {
        "anthropic": "#e76f51",
        "openai": "#2a9d8f",
        "google": "#e9c46a",
        "other": "#999999",
    }

    models = chart_data["data"]["models"]

    # Parse zoom range
    x_start = panel.get("x_start", "2019-06-01")
    x_end = panel.get("x_end", "2026-07-01")
    x_start_dt = datetime.strptime(x_start, "%Y-%m-%d")
    x_end_dt = datetime.strptime(x_end, "%Y-%m-%d")

    # Plot model points
    for m in models:
        rd_str = m.get("release", "")
        if not rd_str:
            continue
        rd = datetime.strptime(rd_str, "%Y-%m-%d")
        rd_mpl = mdates.date2num(rd)
        p50 = m["p50"]

        if m["frontier"]:
            color = PROVIDER_COLORS.get(m["provider"], "#999")
            alpha = 1.0
            ms = 60
            zorder = 5
        else:
            color = "#cccccc"
            alpha = 0.5
            ms = 30
            zorder = 3

        ax.scatter(
            [rd_mpl],
            [p50],
            s=ms,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.5,
            zorder=zorder,
        )

        # Error bars
        ci_lo = m.get("ci_lo", p50 * 0.5)
        ci_hi = m.get("ci_hi", p50 * 2.0)
        ax.plot(
            [rd_mpl, rd_mpl],
            [ci_lo, ci_hi],
            color=color,
            alpha=alpha * 0.4,
            linewidth=1,
            zorder=zorder - 1,
        )

    # Draw fit line
    if "fit" in panel:
        fit_dates = [datetime.strptime(d, "%Y-%m-%d") for d in panel["fit"]["dates"]]
        fit_mpl = [mdates.date2num(d) for d in fit_dates]
        fit_vals = coerce_floats(panel["fit"]["values"])
        ax.plot(fit_mpl, fit_vals, color=teal, linewidth=2, alpha=0.5, zorder=4)

    # Draw CI band
    if "ci" in panel:
        ci_dates = [datetime.strptime(d, "%Y-%m-%d") for d in panel["ci"]["dates"]]
        ci_mpl = [mdates.date2num(d) for d in ci_dates]
        ci_lo = [max(v, 0.05) for v in coerce_floats(panel["ci"]["lower"])]
        ci_hi = [min(v, 5000) for v in coerce_floats(panel["ci"]["upper"])]
        valid_mask = [lo > 0 and hi > 0 for lo, hi in zip(ci_lo, ci_hi)]
        if any(valid_mask):
            ax.fill_between(
                [x for x, v in zip(ci_mpl, valid_mask) if v],
                [lo for lo, v in zip(ci_lo, valid_mask) if v],
                [hi for hi, v in zip(ci_hi, valid_mask) if v],
                color=teal,
                alpha=0.15,
                zorder=1,
            )

    # Stats annotation
    stats = panel.get("stats", "")
    if stats:
        ax.annotate(
            stats,
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85),
        )

    # Title
    ax.set_title(panel.get("title", ""), fontsize=11)

    # Axes
    ax.set_yscale("log")
    ax.set_xlim(mdates.date2num(x_start_dt) - 15, mdates.date2num(x_end_dt) + 15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(alpha=0.15)

    yticks = [1, 5, 10, 30, 60, 120, 240, 480]
    ylabels = ["1m", "5m", "10m", "30m", "1h", "2h", "4h", "8h"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0.05, 5000)

    if not show_legend:
        legend = ax.get_legend()
        if legend:
            legend.remove()


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render 4x2 composite grid from chart_data."""
    import matplotlib.pyplot as plt

    panels = chart_data["data"]["panels"]

    fit_types = ["exponential", "linear", "hyperbolic", "logistic"]
    zoom_labels = ["Full range", "2024 onward"]

    # Create 4x2 grid: rows = fit types, cols = zoom levels
    n_rows = len(fit_types)
    n_cols = len(zoom_labels)

    fig, axes = plt.subplots(
        n_cols,
        n_rows,
        figsize=(9 * n_rows, 7 * n_cols),
    )

    # Build lookup for panels
    panel_lookup = {}
    for p in panels:
        key = (p["fit_type"], p["zoom"])
        panel_lookup[key] = p

    first_panel = True
    for col_idx, zoom in enumerate(zoom_labels):
        for row_idx, fit_type in enumerate(fit_types):
            ax = axes[col_idx, row_idx]
            panel = panel_lookup.get((fit_type, zoom))
            if panel:
                _render_panel(ax, chart_data, panel, show_legend=first_panel)
                first_panel = False
            else:
                ax.set_visible(False)

    fig.suptitle(
        "Trendline Functional Form Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_png(fig, output, params)
    plt.close(fig)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot trendline functional form comparison (4x2 grid)")
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
    parser.add_argument(
        "--stats-output", default=None, help="Output JSON with fit statistics"
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
