"""Cross-source comparison plots for Appendix D (Measurement Quality).

Generates a 2x2 grid of scatter plots comparing different human timing sources.
Each panel shows OLS regression and a y=x constrained fit (intercept only),
with benchmark-colored points and censored observations where applicable.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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

from scipy import stats as sp_stats  # noqa: E402

# Lyptus palette
_TEAL_DARK = "#264653"
_TEAL = "#2a9d8f"
_CORAL = "#e76f51"
_PLUM = "#6d597a"
_TEXT = "#222222"
_TEXT_MUTED = "#888888"
_BG = "#fffaf0"

BENCH_COLORS = {
    "cybashbench": "#264653",
    "nl2bash": "#287271",
    "intercode_ctf": "#2a9d8f",
    "nyuctf": "#8ab17d",
    "cybench": "#e9c46a",
    "cvebench": "#f4a261",
    "cybergym": "#e76f51",
}

_TIME_TICKS_LOG2 = [
    -5.9069,
    -3.5850,
    -2.0000,
    0.0000,
    2.3219,
    3.9069,
    5.9069,
    7.9069,
    9.9069,
    11.9069,
]
_TIME_TICK_LABELS = ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "4h", "16h", "64h"]


def _yx_r_squared(x, y):
    """R-squared for the y=x+b model (slope forced to 1, intercept free)."""
    b = np.mean(y - x)
    ss_res = np.sum((y - (x + b)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, b


def _compute_panel_stats(
    x, y, benchmarks, censored_x=None, censored_y=None, censored_benchmarks=None
):
    """Compute all regression stats for a single panel. Returns JSON-serializable dict."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    has_censored = censored_x is not None and len(censored_x) > 0
    cx = np.array(censored_x, dtype=float) if has_censored else np.array([])
    cy = np.array(censored_y, dtype=float) if has_censored else np.array([])

    # OLS regression (uncensored data only)
    slope, intercept, r_value, p_value, se = sp_stats.linregress(x, y)
    ols_r2 = r_value**2
    residuals = y - (intercept + slope * x)
    sigma = np.std(residuals, ddof=2)

    # Censored residuals
    cens_residuals = (cy - (intercept + slope * cx)).tolist() if has_censored else []

    # Tobit regression (when censored data exists)
    tobit_result = None
    if has_censored and len(cx) > 0:
        from lib.calibration import tobit_regression

        all_x = np.concatenate([x, cx])
        all_y = np.concatenate([y, cy])
        censored_mask = np.concatenate(
            [np.zeros(len(x), dtype=bool), np.ones(len(cx), dtype=bool)]
        )
        try:
            tobit_result = tobit_regression(all_x, all_y, censored_mask)
        except Exception:
            pass

    # y = x + b (slope forced to 1)
    yx_r2, yx_bias = _yx_r_squared(x, y)

    stats = {
        "n": int(len(x)),
        "n_censored": int(len(cx)),
        "ols_slope": round(float(slope), 2),
        "ols_intercept": round(float(intercept), 2),
        "ols_se": round(float(se), 2),
        "ols_r2": round(float(ols_r2), 2),
        "yx_bias": round(float(yx_bias), 2),
        "yx_r2": round(float(yx_r2), 2),
        "residual_sd": round(float(sigma), 2),
    }
    if tobit_result is not None:
        stats["tobit_slope"] = round(float(tobit_result.slope), 2)
        stats["tobit_intercept"] = round(float(tobit_result.intercept), 2)
        stats["tobit_se"] = round(float(tobit_result.se_slope), 2)
        stats["tobit_sigma"] = round(float(tobit_result.sigma), 2)

    return stats, residuals.tolist(), cens_residuals


def _pairs_to_series(pairs, censored=None):
    """Convert a list of pair dicts to per-benchmark scatter series."""
    by_bench = defaultdict(lambda: {"x": [], "y": [], "labels": []})
    for p in pairs:
        b = p["bench"]
        by_bench[b]["x"].append(round(float(p["x"]), 4))
        by_bench[b]["y"].append(round(float(p["y"]), 4))
        by_bench[b]["labels"].append({"benchmark": b})

    series = []
    for bench in sorted(by_bench):
        entry = {
            "name": bench,
            "x": by_bench[bench]["x"],
            "y": by_bench[bench]["y"],
            "color": BENCH_COLORS.get(bench, _TEXT_MUTED),
            "labels": by_bench[bench]["labels"],
        }
        series.append(entry)

    if censored:
        cens_by_bench = defaultdict(lambda: {"x": [], "y": [], "labels": []})
        for p in censored:
            b = p["bench"]
            cens_by_bench[b]["x"].append(round(float(p["x"]), 4))
            cens_by_bench[b]["y"].append(round(float(p["y"]), 4))
            cens_by_bench[b]["labels"].append({"benchmark": b, "censored": True})
        for bench in sorted(cens_by_bench):
            series.append(
                {
                    "name": f"{bench} (censored)",
                    "x": cens_by_bench[bench]["x"],
                    "y": cens_by_bench[bench]["y"],
                    "color": BENCH_COLORS.get(bench, _TEXT_MUTED),
                    "labels": cens_by_bench[bench]["labels"],
                    "marker": "triangle",
                }
            )

    return series


def _build_interactive_tabs(
    pairs_a, cens_a, pairs_b, cens_b, pairs_c, pairs_d, panel_stats, axis_range
):
    """Build tabbedScatter tabs for the interactive chart."""
    shared_min, shared_max = axis_range or (-5, 15)
    tabs = []

    def _add_regression_and_stats(tab, stats_key):
        if stats_key not in panel_stats:
            return
        s = panel_stats[stats_key]
        slope = s["ols_slope"]
        intercept = s.get("ols_intercept", 0)
        tab["regressionLine"] = {
            "slope": slope,
            "intercept": intercept,
            "color": _TEAL_DARK,
            "label": f"OLS (slope={slope}, R\u00b2={s['ols_r2']})",
            "points": {
                "x": [round(shared_min, 4), round(shared_max, 4)],
                "y": [
                    round(slope * shared_min + intercept, 4),
                    round(slope * shared_max + intercept, 4),
                ],
            },
        }
        if "yx_bias" in s:
            bias = s["yx_bias"]
            tab["forcedSlopeLine"] = {
                "intercept": bias,
                "color": _CORAL,
                "points": {
                    "x": [round(shared_min, 4), round(shared_max, 4)],
                    "y": [round(shared_min + bias, 4), round(shared_max + bias, 4)],
                },
            }
        if s.get("tobit_slope") is not None:
            t_slope = s["tobit_slope"]
            t_int = s["tobit_intercept"]
            tab["tobitLine"] = {
                "slope": t_slope,
                "intercept": t_int,
                "points": {
                    "x": [round(shared_min, 4), round(shared_max, 4)],
                    "y": [
                        round(t_slope * shared_min + t_int, 4),
                        round(t_slope * shared_max + t_int, 4),
                    ],
                },
            }
        for series in tab.get("series", []):
            series["residuals"] = [
                round(y - (slope * x + intercept), 4)
                for x, y in zip(series["x"], series["y"])
            ]
        tab["stats"] = s

    if pairs_a:
        tab = {
            "label": "Estimates vs Completions",
            "series": _pairs_to_series(pairs_a, cens_a),
            "yEqualsX": True,
        }
        _add_regression_and_stats(tab, "est_vs_comp")
        tabs.append(tab)

    if pairs_b:
        tab = {
            "label": "Model vs Completions",
            "series": _pairs_to_series(pairs_b, cens_b),
            "yEqualsX": True,
        }
        _add_regression_and_stats(tab, "model_vs_comp")
        tabs.append(tab)

    if pairs_c:
        tab = {
            "label": "Expert vs Model Estimates",
            "series": _pairs_to_series(pairs_c),
            "yEqualsX": True,
        }
        _add_regression_and_stats(tab, "expert_vs_model")
        tabs.append(tab)

    if pairs_d:
        tab = {
            "label": "CTF First-Blood vs Expert Study",
            "series": _pairs_to_series(pairs_d),
            "yEqualsX": True,
        }
        _add_regression_and_stats(tab, "firstblood_vs_study")
        tabs.append(tab)

    return tabs


# =============================================================================
# Compute: all data loading + statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute regressions, return chart_data dict."""

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    with open(resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)
    with open(resolve(args.best_available)) as f:
        best_available = json.load(f)  # noqa: F841 - loaded for future use, arg is a declared dep
    from lib.data import headline_task_set

    task_diff = pd.read_parquet(resolve(args.task_difficulties))
    model_runs_df = (
        pd.read_parquet(resolve(args.model_runs)) if args.model_runs else None
    )
    headline_tasks, task_family = headline_task_set(task_diff, model_runs_df)

    # Expert completions: geometric mean of server_elapsed per task (passes only)
    from lib.corrections import corrected_elapsed, TIMING_CORRECTIONS

    comp_times = defaultdict(list)
    for p in snapshot.get("passes", []):
        tid = p["task_id"]
        if tid not in headline_tasks:
            continue
        s = corrected_elapsed(p, TIMING_CORRECTIONS)
        if s > 0:
            comp_times[tid].append(s / 60)

    comp_geomean = {}
    for tid, times in comp_times.items():
        comp_geomean[tid] = float(np.exp(np.mean(np.log(times))))

    # Censored completions (censored sessions + failed attempts)
    # Both are right-censored lower bounds: the true completion time is at
    # least the observed duration.
    cens_times = {}
    for c in snapshot.get("censored", []) + snapshot.get("fails", []):
        tid = c["task_id"]
        if tid not in headline_tasks:
            continue
        s = corrected_elapsed(c, TIMING_CORRECTIONS)
        if s > 0:
            # Keep the longest censored observation per task
            cens_times[tid] = max(cens_times.get(tid, 0), s / 60)

    # Expert estimates: geometric mean per task
    est_times_raw = defaultdict(list)
    for e in snapshot.get("estimations", []):
        tid = e["task_id"]
        if tid in headline_tasks and e.get("estimated_seconds"):
            est_times_raw[tid].append(e["estimated_seconds"] / 60)

    est_geomean = {}
    for tid, times in est_times_raw.items():
        est_geomean[tid] = float(np.exp(np.mean(np.log(times))))

    # Model estimates
    from lib.data import load_model_time_estimates

    model_est = load_model_time_estimates()

    # === Build pair data ===

    # Panel A: Expert estimates vs expert completions
    pairs_a, cens_a = [], []
    for tid in comp_geomean:
        if tid in est_geomean:
            pairs_a.append(
                {
                    "x": round(float(np.log2(est_geomean[tid])), 4),
                    "y": round(float(np.log2(comp_geomean[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )
    for tid in cens_times:
        if tid in est_geomean:
            cens_a.append(
                {
                    "x": round(float(np.log2(est_geomean[tid])), 4),
                    "y": round(float(np.log2(cens_times[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )

    # Panel B: Model estimates vs expert completions
    pairs_b, cens_b = [], []
    for tid in comp_geomean:
        if tid in model_est:
            pairs_b.append(
                {
                    "x": round(float(np.log2(model_est[tid])), 4),
                    "y": round(float(np.log2(comp_geomean[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )
    for tid in cens_times:
        if tid in model_est:
            cens_b.append(
                {
                    "x": round(float(np.log2(model_est[tid])), 4),
                    "y": round(float(np.log2(cens_times[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )

    # Panel C: Expert estimates vs model estimates
    pairs_c = []
    for tid in est_geomean:
        if tid in model_est:
            pairs_c.append(
                {
                    "x": round(float(np.log2(model_est[tid])), 4),
                    "y": round(float(np.log2(est_geomean[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )

    # Panel D: CTF first-blood times vs expert study data
    fb_tasks = {}
    for _, row in task_diff.iterrows():
        tid = str(row["task_id"])
        fb_min = row.get("firstblood_minutes")
        if tid in headline_tasks and pd.notna(fb_min) and fb_min > 0:
            fb_tasks[tid] = float(fb_min)

    pairs_d = []
    for tid in fb_tasks:
        expert_time = comp_geomean.get(tid) or est_geomean.get(tid)
        if expert_time is not None:
            pairs_d.append(
                {
                    "x": round(float(np.log2(expert_time)), 4),
                    "y": round(float(np.log2(fb_tasks[tid])), 4),
                    "bench": task_family.get(tid, "unknown"),
                }
            )

    # === Compute shared axis range ===
    all_log2_vals = []
    for pairs in [pairs_a, pairs_b, pairs_c]:
        for p in pairs:
            all_log2_vals.extend([p["x"], p["y"]])
    for pairs in [cens_a, cens_b]:
        for p in pairs:
            all_log2_vals.extend([p["x"], p["y"]])
    for p in pairs_d:
        all_log2_vals.extend([p["x"], p["y"]])
    pad = 1.0
    shared_min = float(min(all_log2_vals) - pad)
    shared_max = float(max(all_log2_vals) + pad)

    # === Compute panel stats ===
    panel_stats = {}

    if pairs_a:
        xa = [p["x"] for p in pairs_a]
        ya = [p["y"] for p in pairs_a]
        ba = [p["bench"] for p in pairs_a]
        cxa = [p["x"] for p in cens_a] if cens_a else None
        cya = [p["y"] for p in cens_a] if cens_a else None
        cba = [p["bench"] for p in cens_a] if cens_a else None
        stats_a, resid_a, cens_resid_a = _compute_panel_stats(xa, ya, ba, cxa, cya, cba)
        panel_stats["est_vs_comp"] = stats_a

    if pairs_b:
        xb = [p["x"] for p in pairs_b]
        yb = [p["y"] for p in pairs_b]
        bb = [p["bench"] for p in pairs_b]
        cxb = [p["x"] for p in cens_b] if cens_b else None
        cyb = [p["y"] for p in cens_b] if cens_b else None
        cbb = [p["bench"] for p in cens_b] if cens_b else None
        stats_b, resid_b, cens_resid_b = _compute_panel_stats(xb, yb, bb, cxb, cyb, cbb)
        panel_stats["model_vs_comp"] = stats_b

    if pairs_c:
        xc = [p["x"] for p in pairs_c]
        yc = [p["y"] for p in pairs_c]
        bc = [p["bench"] for p in pairs_c]
        stats_c, resid_c, _ = _compute_panel_stats(xc, yc, bc)
        panel_stats["expert_vs_model"] = stats_c

    if pairs_d:
        xd = [p["x"] for p in pairs_d]
        yd = [p["y"] for p in pairs_d]
        bd = [p["bench"] for p in pairs_d]
        stats_d, resid_d, _ = _compute_panel_stats(xd, yd, bd)
        panel_stats["firstblood_vs_study"] = stats_d

    # Build interactive chart tabs
    tabs = _build_interactive_tabs(
        pairs_a,
        cens_a,
        pairs_b,
        cens_b,
        pairs_c,
        pairs_d,
        panel_stats,
        (shared_min, shared_max),
    )

    # All benchmarks for legend
    all_benchmarks = sorted(set(task_family.values()))

    # Panel configs for render_png
    panel_configs = [
        {
            "key": "est_vs_comp",
            "title": "(a) Expert Estimates vs Completions",
            "xlabel": "Expert estimate",
            "ylabel": "Expert completion",
            "pairs": pairs_a,
            "censored": cens_a,
            "row_pair": 0,
            "col": 0,
        },
        {
            "key": "model_vs_comp",
            "title": "(b) Model Estimates vs Completions",
            "xlabel": "Model estimate",
            "ylabel": "Expert completion",
            "pairs": pairs_b,
            "censored": cens_b,
            "row_pair": 0,
            "col": 1,
        },
        {
            "key": "expert_vs_model",
            "title": "(c) Expert vs Model Estimates",
            "xlabel": "Model estimate",
            "ylabel": "Expert estimate",
            "pairs": pairs_c,
            "censored": [],
            "row_pair": 1,
            "col": 0,
        },
        {
            "key": "firstblood_vs_study",
            "title": "(d) CTF First-Blood vs Expert Study",
            "xlabel": "Expert study",
            "ylabel": "CTF first-blood time",
            "pairs": pairs_d,
            "censored": [],
            "row_pair": 1,
            "col": 1,
        },
    ]

    return {
        "chart_type": "tabbedScatter",
        "version": 1,
        "data": {
            "tabs": tabs,
            "panel_configs": panel_configs,
            "panel_stats": panel_stats,
            "shared_lim": [shared_min, shared_max],
            "all_benchmarks": all_benchmarks,
        },
        "options": {
            "title": "Cross-source comparisons",
            "xLabel": "Source A (log\u2082 minutes)",
            "yLabel": "Source B (log\u2082 minutes)",
            "logAxes": True,
            "timeAxis": True,
        },
        "_companion_stats": panel_stats,
    }


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def _format_time_axis(ax, which="both"):
    for axis_name in ["x", "y"] if which == "both" else [which]:
        lo, hi = getattr(ax, f"get_{axis_name}lim")()
        visible = [
            (t, lab)
            for t, lab in zip(_TIME_TICKS_LOG2, _TIME_TICK_LABELS)
            if lo <= t <= hi
        ]
        if visible:
            ticks, labels = zip(*visible)
            getattr(ax, f"set_{axis_name}ticks")(list(ticks))
            getattr(ax, f"set_{axis_name}ticklabels")(list(labels))


def _stat_box(ax, text, loc="lower right"):
    va = "bottom" if "lower" in loc else "top"
    ha = "right" if "right" in loc else "left"
    x = 0.97 if "right" in loc else 0.03
    y = 0.03 if "lower" in loc else 0.97
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va=va,
        ha=ha,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor=_BG, alpha=0.85, edgecolor="#e5dfd6"
        ),
    )


def _render_scatter_panel(ax_scatter, ax_resid, panel_cfg, stats, shared_lim):
    """Draw a scatter + residual panel pair from chart_data dicts."""
    import numpy as np

    pairs = panel_cfg["pairs"]
    censored = panel_cfg.get("censored", [])
    title = panel_cfg["title"]
    xlabel = panel_cfg["xlabel"]
    ylabel = panel_cfg["ylabel"]

    if not pairs:
        ax_scatter.text(
            0.5, 0.5, "No data", transform=ax_scatter.transAxes, ha="center"
        )
        ax_scatter.set_title(title, fontsize=10, fontweight="bold")
        return

    x = np.array([p["x"] for p in pairs], dtype=float)
    y = np.array([p["y"] for p in pairs], dtype=float)
    bench = [p["bench"] for p in pairs]
    colors = [BENCH_COLORS.get(b, _TEXT_MUTED) for b in bench]

    has_censored = len(censored) > 0
    cx = (
        np.array([p["x"] for p in censored], dtype=float)
        if has_censored
        else np.array([])
    )
    cy = (
        np.array([p["y"] for p in censored], dtype=float)
        if has_censored
        else np.array([])
    )
    cb = [p["bench"] for p in censored] if has_censored else []
    cc = [BENCH_COLORS.get(b, _TEXT_MUTED) for b in cb]

    data_min, data_max = shared_lim

    # Get stats
    slope = stats["ols_slope"]
    intercept = stats["ols_intercept"]
    se = stats["ols_se"]
    ols_r2 = stats["ols_r2"]
    yx_bias = stats["yx_bias"]
    yx_r2 = stats["yx_r2"]
    sigma = stats["residual_sd"]

    # Compute residuals from the stored regression parameters
    residuals = y - (intercept + slope * x)
    cens_residuals = cy - (intercept + slope * cx) if has_censored else np.array([])

    # --- Scatter panel ---
    ax_scatter.plot(
        [data_min, data_max],
        [data_min, data_max],
        "--",
        color=_TEXT_MUTED,
        alpha=0.3,
        linewidth=0.8,
    )

    ax_scatter.scatter(
        x, y, c=colors, s=40, edgecolors="white", linewidth=0.4, zorder=3
    )

    if has_censored:
        ax_scatter.scatter(
            cx, cy, c=cc, s=55, edgecolors="white", linewidth=0.4, zorder=3, marker="^"
        )

    xline = np.linspace(data_min, data_max, 100)
    ax_scatter.plot(
        xline, slope * xline + intercept, color=_TEAL_DARK, linewidth=1.5, alpha=0.8
    )
    ax_scatter.plot(xline, xline + yx_bias, color=_CORAL, linewidth=1.5, alpha=0.8)

    if stats.get("tobit_slope") is not None:
        t_slope = stats["tobit_slope"]
        t_int = stats["tobit_intercept"]
        ax_scatter.plot(
            xline,
            t_slope * xline + t_int,
            color=_PLUM,
            linewidth=1.5,
            alpha=0.8,
        )

    ax_scatter.set_xlim(data_min, data_max)
    ax_scatter.set_ylim(data_min, data_max)
    ax_scatter.set_aspect("equal", adjustable="box")
    if ylabel:
        ax_scatter.set_ylabel(ylabel, fontsize=9)
    ax_scatter.set_title(title, fontsize=10, fontweight="bold")
    ax_scatter.grid(alpha=0.15)
    ax_scatter.tick_params(labelbottom=False)
    _format_time_axis(ax_scatter, which="y")

    # Stats box
    n = stats["n"]
    n_cens = stats["n_censored"]
    n_str = f"{n}" if n_cens == 0 else f"{n}+{n_cens}cens"
    stat_lines = [
        f"OLS   slope={slope:.2f}\u00b1{se:.2f}  R\u00b2={ols_r2:.2f}",
    ]
    if stats.get("tobit_slope") is not None:
        stat_lines.append(
            f"Tobit slope={stats['tobit_slope']:.2f}\u00b1{stats['tobit_se']:.2f}"
        )
    stat_lines.extend(
        [
            f"y=x+b bias={yx_bias:+.2f}       R\u00b2={yx_r2:.2f}",
            f"residual SD={sigma:.2f}  N={n_str}",
        ]
    )
    _stat_box(ax_scatter, "\n".join(stat_lines), loc="lower right")

    # --- Residual panel ---
    ax_resid.scatter(
        x, residuals, c=colors, s=30, edgecolors="white", linewidth=0.3, zorder=3
    )

    if has_censored:
        ax_resid.scatter(
            cx,
            cens_residuals,
            c=cc,
            s=40,
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
            marker="^",
        )

    ax_resid.axhline(0, color=_TEXT_MUTED, linestyle="--", alpha=0.4, linewidth=0.8)

    ax_resid.axhline(sigma, color=_CORAL, linestyle=":", alpha=0.3, linewidth=0.8)
    ax_resid.axhline(-sigma, color=_CORAL, linestyle=":", alpha=0.3, linewidth=0.8)
    ax_resid.axhspan(-sigma, sigma, alpha=0.03, color=_CORAL)

    if xlabel:
        ax_resid.set_xlabel(xlabel, fontsize=9)
    ax_resid.set_ylabel("Residual", fontsize=8)
    ax_resid.set_xlim(data_min, data_max)
    ax_resid.grid(alpha=0.15)
    _format_time_axis(ax_resid, which="x")


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render cross-source 2x2 grid from chart JSON data."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    panel_configs = chart_data["data"]["panel_configs"]
    panel_stats = chart_data["data"]["panel_stats"]
    shared_lim = chart_data["data"]["shared_lim"]
    all_benchmarks = chart_data["data"]["all_benchmarks"]

    # Create 2x2 grid: 4 rows x 2 columns, height ratios [3,1,3,1]
    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.12, wspace=0.35)

    for cfg in panel_configs:
        if not cfg["pairs"]:
            continue
        key = cfg["key"]
        rp = cfg["row_pair"]
        col = cfg["col"]
        r = rp * 2
        ax_s = fig.add_subplot(gs[r, col])
        ax_r = fig.add_subplot(gs[r + 1, col], sharex=ax_s)

        stats = panel_stats.get(key, {})
        _render_scatter_panel(ax_s, ax_r, cfg, stats, shared_lim)

    # Shared residual y-limits
    resid_axes = [fig.axes[i] for i in range(len(fig.axes)) if i % 2 == 1]
    max_resid = 0
    for ax in resid_axes:
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                max_resid = max(max_resid, np.abs(offsets[:, 1]).max())
    resid_pad = max_resid * 1.15
    for ax in resid_axes:
        ax.set_ylim(-resid_pad, resid_pad)

    # Re-apply time axis formatting
    scatter_axes = [fig.axes[i] for i in range(len(fig.axes)) if i % 2 == 0]
    for ax in scatter_axes:
        _format_time_axis(ax, which="both")
    for ax in resid_axes:
        _format_time_axis(ax, which="x")

    # Shared legend at bottom
    bench_handles = [
        mpatches.Patch(facecolor=BENCH_COLORS.get(b, _TEXT_MUTED), label=b)
        for b in all_benchmarks
    ]
    line_handles = [
        Line2D([0], [0], color=_TEAL_DARK, linewidth=1.5, alpha=0.8, label="OLS fit"),
        Line2D([0], [0], color=_PLUM, linewidth=1.5, alpha=0.8, label="Tobit fit"),
        Line2D([0], [0], color=_CORAL, linewidth=1.5, alpha=0.8, label="y = x + b"),
        Line2D([0], [0], color=_TEXT_MUTED, linestyle="--", alpha=0.3, label="y = x"),
        Line2D(
            [0],
            [0],
            marker="^",
            color=_TEXT_MUTED,
            linestyle="",
            markersize=7,
            label="censored",
        ),
    ]
    fig.legend(
        handles=line_handles + bench_handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Generate cross-source comparison grid")
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--best-available", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument(
        "--model-runs",
        default=None,
        help="model_runs.parquet (filters headline to evaluated tasks)",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)

    # Write companion JSON
    companion_stats = chart_data.get("_companion_stats", {})
    json_path = Path(args.output).with_suffix(".json")
    if not json_path.is_absolute():
        json_path = _NOTEBOOKS_DIR / json_path
    with open(json_path, "w") as f:
        json.dump(companion_stats, f, indent=2)
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
