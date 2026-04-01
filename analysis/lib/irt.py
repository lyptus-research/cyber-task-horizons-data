"""IRT fitting, simulation, and plotting functions for CTH sensitivity analysis.

Core logistic regression uses METR's eval-analysis-public (the same code that
produces the published results). Simulation helpers and S-curve grid plotting
are built on top.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression

from .constants import DEFAULT_REGULARIZATION, JUNE_SOTA_FULL_NAMES

# --- Import METR's canonical fitting code from the monorepo-level checkout ---
# Uses the upstream METR checkout (horizon package) at third-party/ in repo root.
)

try:
    from horizon.utils.logistic import (  # noqa: E402
        get_x_for_quantile,
        logistic_regression,
    )
except ModuleNotFoundError:
    # Older checkout has utils.logistic at top level (no horizon package)
    from utils.logistic import (  # noqa: E402
        get_x_for_quantile,
        logistic_regression,
    )


# =============================================================================
# Core IRT fitting (thin wrappers around METR)
# =============================================================================


def fit_p50(log2_times, scores, weights=None, regularization=DEFAULT_REGULARIZATION):
    """Fit 2-PL IRT logistic regression, return (p50_log2, coef, intercept).

    Uses METR's logistic_regression() which handles fractional y values and
    applies regularization as C = 1/regularization.

    Args:
        log2_times: array of log₂(human_minutes) values
        scores: array of binary scores (0/1)
        weights: optional sample weights
        regularization: METR regularization param (C = 1/regularization)

    Returns:
        (p50_log2, coefficient, intercept) tuple.
        p50_log2 is -inf if all scores are 0.
    """
    X = np.asarray(log2_times, dtype=float).reshape(-1, 1)
    y = np.asarray(scores, dtype=float)

    if weights is None:
        w = np.ones(len(y))
    else:
        w = np.asarray(weights, dtype=float)

    # METR 1.1 asserts weights sum to 1.0 per agent
    w = w / w.sum()

    if np.all(y == 0):
        return -np.inf, -np.inf, 0.0

    model = logistic_regression(X, y, sample_weight=w, regularization=regularization)
    p50_log2 = get_x_for_quantile(model, 0.5)
    return float(p50_log2), float(model.coef_[0][0]), float(model.intercept_[0])


def fit_all_p50s(
    runs_df,
    log2_col="log2_human_minutes",
    alias_map=None,
    release_dates=None,
    regularization=DEFAULT_REGULARIZATION,
):
    """Fit P50 for each model in runs_df. Returns DataFrame.

    Columns: agent, alias, p50_log2, p50_minutes, coefficient, intercept, release_date
    """
    if alias_map is None:
        alias_map = {}
    if release_dates is None:
        release_dates = {}

    results = []
    for agent, group in runs_df.groupby("agent"):
        p50_log2, coef, intercept = fit_p50(
            group[log2_col].values,
            group["score_binarized"].values,
            weights=group["weight"].values if "weight" in group.columns else None,
            regularization=regularization,
        )
        results.append(
            {
                "agent": agent,
                "alias": alias_map.get(agent, agent),
                "p50_log2": p50_log2,
                "p50_minutes": 2**p50_log2 if np.isfinite(p50_log2) else 0.0,
                "coefficient": coef,
                "intercept": intercept,
                "release_date": release_dates.get(agent),
            }
        )
    return pd.DataFrame(results)


def fit_doubling_time(p50s_df, sota_list=None):
    """Fit ln-space trendline to SOTA models.

    Returns (doubling_time_months, r_squared, regression_model).
    Delegates to METR's fit_trendline (OLS in log-space).

    Note: this is used by perturbation_sensitivity.ipynb for sensitivity
    analysis. The headline pipeline uses fit_trendline.py (which writes
    trendline_params.json) instead.
    """
    from horizon.plot.logistic import fit_trendline

    if sota_list is None:
        # Use is_sota column if available (current pipeline), fall back to
        # static list for backwards compatibility with older DataFrames
        if "is_sota" in p50s_df.columns:
            sota = p50s_df[p50s_df["is_sota"]].dropna(subset=["release_date"])
        else:
            sota = p50s_df[p50s_df["agent"].isin(JUNE_SOTA_FULL_NAMES)].dropna(
                subset=["release_date"]
            )
    else:
        sota = p50s_df[p50s_df["agent"].isin(sota_list)].dropna(subset=["release_date"])

    if len(sota) < 2:
        return np.nan, 0.0, None

    reg, r2 = fit_trendline(
        agent_horizons=sota["p50_minutes"],
        release_dates=sota["release_date"],
        log_scale=True,
    )
    dt_days = np.log(2) / reg.coef_[0]
    return dt_days / 30.44, r2, reg


# =============================================================================
# S-curve plotting
# =============================================================================


def _fmt_p50(p50_min):
    """Format P50 value for display in title."""
    if p50_min < 1:
        return f"{p50_min * 60:.0f}s"
    elif p50_min < 60:
        return f"{p50_min:.1f}m"
    else:
        return f"{p50_min / 60:.1f}h"


# Clean tick positions in log2(minutes) space and their labels.
# Chosen for readability: round human-friendly durations.
_TICK_MINUTES = [
    1 / 60,  # 1s
    5 / 60,  # 5s
    15 / 60,  # 15s
    1,  # 1m
    5,  # 5m
    15,  # 15m
    60,  # 1h
    240,  # 4h
    960,  # 16h
    3840,  # 64h
]
_TICK_LABELS = ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "4h", "16h", "64h"]
_TICK_LOG2 = [np.log2(m) for m in _TICK_MINUTES]


def compute_scurve_data(
    df,
    global_bin_edges,
    weight_col="invsqrt_task_weight",
    regularization=DEFAULT_REGULARIZATION,
    min_n=5,
):
    """Extract per-panel IRT data without plotting. Used for interactive chart JSON.

    Returns the same data that plot_scurve() computes internally, but as a
    serializable dict instead of matplotlib artists.
    """
    log2_t = df["log2_human_minutes"].values
    scores = df["score_binarized"].values
    w = df[weight_col].values

    # Fit using METR's logistic_regression directly (not fit_p50) so we
    # retain the sklearn model for predict_proba. No manual sigmoid eval.
    X = log2_t.reshape(-1, 1)
    w_norm = w / w.sum()

    if np.all(scores == 0):
        p50_log2 = -np.inf
        coef = -np.inf
        intercept = 0.0
        model = None
    else:
        model = logistic_regression(
            X, scores, sample_weight=w_norm, regularization=regularization
        )
        p50_log2 = float(get_x_for_quantile(model, 0.5))
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])

    p50_min = 2**p50_log2

    bin_edges = global_bin_edges
    n_bins = len(bin_edges) - 1
    empirical_rates = [None] * n_bins
    bin_counts = [0] * n_bins
    standard_errors = [None] * n_bins
    bin_centers = [
        float(0.5 * (bin_edges[i] + bin_edges[i + 1])) for i in range(n_bins)
    ]

    for i in range(n_bins):
        mask = (log2_t >= bin_edges[i]) & (log2_t < bin_edges[i + 1])
        n = int(mask.sum())
        bin_counts[i] = n
        if n >= 1:
            rate = float(scores[mask].mean())
            empirical_rates[i] = round(rate, 4)
            weights_in_bin = w[mask]
            n_eff = np.sum(weights_in_bin) ** 2 / np.sum(weights_in_bin**2)
            if n_eff > 0 and rate * (1 - rate) > 0:
                standard_errors[i] = round(float(np.sqrt(rate * (1 - rate) / n_eff)), 4)

    # Pre-sample the S-curve using METR's fitted model (predict_proba)
    n_curve_pts = 200
    ax_min = bin_edges[0] - 0.5
    ax_max = bin_edges[-1] + 0.5
    curve_x = np.linspace(ax_min, ax_max, n_curve_pts)
    if model is not None:
        curve_y = model.predict_proba(curve_x.reshape(-1, 1))[:, 1] * 100.0
    else:
        curve_y = np.zeros(n_curve_pts)

    return {
        "bin_centers": bin_centers,
        "empirical_rates": empirical_rates,
        "bin_counts": bin_counts,
        "standard_errors": standard_errors,
        "p50_log2": round(float(p50_log2), 4),
        "coef": round(float(coef), 6),
        "intercept": round(float(intercept), 4),
        "p50_label": _fmt_p50(p50_min),
        "min_n": min_n,
        "curve_x": [round(float(x), 4) for x in curve_x],
        "curve_y": [round(float(y), 2) for y in curve_y],
    }


def plot_scurve(
    ax,
    df,
    title_label,
    weight_col="invsqrt_task_weight",
    min_n=5,
    global_bin_edges=None,
    regularization=DEFAULT_REGULARIZATION,
):
    """Plot histogram + IRT S-curve on a given axis.

    Args:
        ax: matplotlib Axes to plot on.
        df: DataFrame with log2_human_minutes, score_binarized, and weight_col.
        title_label: title string for the subplot.
        weight_col: column name for sample weights.
        min_n: bins with n <= min_n are drawn pale.
        global_bin_edges: if provided, use these bin edges instead of
            computing from the data. Required for shared x-axes.
        regularization: regularization parameter for logistic regression
            (λ; C = 1/λ). Default matches METR 1.1.

    Returns:
        (p50_minutes, coefficient, intercept) tuple.
    """
    log2_t = df["log2_human_minutes"].values
    scores = df["score_binarized"].values
    w = df[weight_col].values

    p50_log2, coef, intercept = fit_p50(
        log2_t, scores, weights=w, regularization=regularization
    )
    p50_min = 2**p50_log2

    # Histogram bins (1-doubling wide)
    if global_bin_edges is not None:
        bin_edges = global_bin_edges
    else:
        bin_edges = np.arange(
            np.floor(log2_t.min()) - 0.5, np.ceil(log2_t.max()) + 1.5, 1.0
        )
    n_bins = len(bin_edges) - 1
    empirical_rates = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i in range(n_bins):
        mask = (log2_t >= bin_edges[i]) & (log2_t < bin_edges[i + 1])
        n = mask.sum()
        bin_counts[i] = n
        if n >= 1:
            empirical_rates[i] = scores[mask].mean()

    # Theme-aware colors
    try:
        from .lyptus_style import COLORS as _C

        _bar_color = _C["teal"]
        _curve_color = _C["coral"]
    except (ImportError, KeyError):
        _bar_color = "#4a90d9"
        _curve_color = "#e74c3c"

    # Standard errors for weighted binary data
    standard_errors = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (log2_t >= bin_edges[i]) & (log2_t < bin_edges[i + 1])
        weights_in_bin = w[mask]
        p = empirical_rates[i]
        if np.isfinite(p) and len(weights_in_bin) > 0:
            n_eff = np.sum(weights_in_bin) ** 2 / np.sum(weights_in_bin**2)
            if n_eff > 0:
                variance = (p * (1 - p)) / n_eff
                if variance > 0:
                    standard_errors[i] = np.sqrt(variance)

    _bar_color_full = _bar_color + "d9"  # ~85% opacity
    _bar_color_pale = _bar_color + "40"  # ~25% opacity — low-n bins
    bar_colors = [_bar_color_full if n > min_n else _bar_color_pale for n in bin_counts]
    ax.bar(
        bin_centers,
        empirical_rates * 100,
        width=0.85,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )

    # Error bars: 2*SE
    se_mask = np.isfinite(standard_errors)
    if se_mask.any():
        ax.errorbar(
            bin_centers[se_mask],
            empirical_rates[se_mask] * 100,
            yerr=2 * standard_errors[se_mask] * 100,
            fmt="none",
            color=_bar_color,
            alpha=0.9,
            capsize=3,
            zorder=4,
        )

    # Per-bin n= labels
    try:
        from .lyptus_style import FONT_SANS as _sans
    except (ImportError, KeyError):
        _sans = "Helvetica Neue"
    for bc, rate, n in zip(bin_centers, empirical_rates, bin_counts):
        if n > 0 and np.isfinite(rate):
            ax.text(
                bc,
                rate * 100 + 3,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontfamily=_sans,
                color="#666",
                alpha=0.75,
            )

    # Fitted logistic curve
    x_smooth = np.linspace(bin_edges[0] - 0.5, bin_edges[-1] + 0.5, 300)
    y_smooth = expit(coef * x_smooth + intercept) * 100
    ax.plot(
        x_smooth,
        y_smooth,
        color=_curve_color,
        linewidth=2.5,
        zorder=3,
    )

    # P50 vertical line
    ax.axvline(
        p50_log2, color=_curve_color, linestyle="--", alpha=0.6, linewidth=1.5, zorder=3
    )

    ax.set_ylim(-5, 105)
    ax.set_title(
        f"{title_label} (P50 = {_fmt_p50(p50_min)})", fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    return p50_min, coef, intercept


def plot_scurve_grid(
    panels,
    suptitle,
    xlabel="Human time (log\u2082 scale)",
    ncols=3,
    col_width=5.5,
    row_height=4,
    release_dates=None,
):
    """Plot a grid of S-curves — panels flow left-to-right, top-to-bottom.

    Designed to scale to 10-15 models. Shared y-axis and shared x-axis
    across all panels. Only edge subplots get axis labels. X-ticks use
    clean human-readable positions (1s, 5s, 1m, 5m, 1h, 4h, etc.).

    Args:
        panels: list of (df, label) tuples. Each gets one cell in the grid.
        suptitle: figure title.
        xlabel: shared x-axis label.
        ncols: number of columns in the grid.
        col_width: width per column in inches.
        row_height: height per row in inches.
        release_dates: {label: date_string} for ordering panels by release date.
            If provided, panels are sorted chronologically (oldest top-left).

    Returns:
        (fig, results) where results is list of (label, p50_minutes_or_None) tuples.
    """
    try:
        from .lyptus_style import FONT_SANS as _sans
    except (ImportError, KeyError):
        _sans = "Helvetica Neue"

    if release_dates:
        import pandas as pd

        panels = sorted(
            panels,
            key=lambda p: pd.to_datetime(release_dates.get(p[1], "2099-01-01")),
        )
    import math

    # Compute global bin edges across all panels so x-axes are shared
    all_log2 = np.concatenate([df["log2_human_minutes"].values for df, _ in panels])
    global_lo = np.floor(all_log2.min()) - 0.5
    global_hi = np.ceil(all_log2.max()) + 1.5
    global_bin_edges = np.arange(global_lo, global_hi, 1.0)

    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(col_width * ncols, row_height * nrows),
        squeeze=False,
        sharey=True,
    )

    results = []
    for idx, (df, label) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if len(df) >= 5:
            p50, _, _ = plot_scurve(ax, df, label, global_bin_edges=global_bin_edges)
            results.append((label, p50))
        else:
            ax.text(
                0.5,
                0.5,
                f"Too few runs ({len(df)} tasks)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#999",
            )
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_ylim(-5, 105)
            results.append((label, None))

        # Only label edge axes
        if r == nrows - 1 or (idx + ncols >= n):
            ax.set_xlabel(xlabel, fontsize=10, fontfamily=_sans)
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Success rate (%)", fontsize=10, fontfamily=_sans)
        else:
            ax.set_ylabel("")

    # Apply clean x-ticks and consistent xlim to every subplot
    x_lo = global_bin_edges[0]
    x_hi = global_bin_edges[-1]
    visible_ticks = [
        (pos, label)
        for pos, label in zip(_TICK_LOG2, _TICK_LABELS)
        if x_lo <= pos <= x_hi
    ]
    if visible_ticks:
        tick_positions, tick_labels_list = zip(*visible_ticks)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(x_lo, x_hi)
                ax.set_xticks(list(tick_positions))
                ax.set_xticklabels(
                    list(tick_labels_list), fontsize=10, fontfamily=_sans
                )
                ax.tick_params(axis="y", labelsize=10)
                for label in ax.get_yticklabels():
                    label.set_fontfamily(_sans)

    # Hide unused axes in the last row
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    for label, p50 in results:
        if p50 is not None:
            print(f"  {label}: P50 = {p50:.1f} min ({p50 / 60:.1f} h)")
        else:
            print(f"  {label}: insufficient data")

    return fig, results


def apply_token_budget(df, budget):
    """Return a copy of df with runs over budget reclassified as failures.

    Any run where total_tokens > budget has score_binarized set to 0.
    Runs at or under budget keep their original score.
    """
    out = df.copy()
    over = out["total_tokens"] > budget
    out.loc[over, "score_binarized"] = 0
    return out


def p50_vs_token_budget(runs_df, budgets=None, weight_col="invsqrt_task_weight"):
    """Compute P50 horizon at each token budget.

    Args:
        runs_df: DataFrame with total_tokens, score_binarized, log2_human_minutes.
        budgets: list of token budget thresholds. Defaults to a log-spaced range
            up to 10M (extended from 2M to support higher-budget sensitivity runs).
        weight_col: column for IRT weighting.

    Returns:
        DataFrame with columns: budget, p50_minutes, accuracy.
    """
    if budgets is None:
        budgets = [
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            1_500_000,
            2_000_000,
            5_000_000,
            10_000_000,
        ]

    rows = []
    for b in budgets:
        capped = apply_token_budget(runs_df, b)
        acc = capped["score_binarized"].mean()
        try:
            p50_log2, _, _ = fit_p50(
                capped["log2_human_minutes"].values,
                capped["score_binarized"].values,
                weights=capped[weight_col].values,
            )
            p50_min = 2**p50_log2 if np.isfinite(p50_log2) else 0.0
        except Exception:
            p50_min = 0.0
        rows.append({"budget": b, "p50_minutes": p50_min, "accuracy": acc})

    return pd.DataFrame(rows)


def plot_p50_vs_token_budget(campaign_data, budgets=None):
    """Plot P50 horizon vs token budget for all models.

    Args:
        campaign_data: dict of alias -> {runs, ...} (as built by the notebook).
        budgets: list of token budgets to evaluate. Extended to 10M to support
            higher-budget sensitivity runs (points beyond observed data are omitted).
    """
    if budgets is None:
        budgets = [
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            1_500_000,
            2_000_000,
            5_000_000,
            10_000_000,
        ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for alias, data in campaign_data.items():
        result = p50_vs_token_budget(data["runs"], budgets=budgets)
        budget_k = [b / 1000 for b in result["budget"]]
        ax1.plot(budget_k, result["p50_minutes"], marker="o", label=alias, linewidth=2)
        ax2.plot(
            budget_k, result["accuracy"] * 100, marker="o", label=alias, linewidth=2
        )

    ax1.set_xlabel("Token budget (thousands)")
    ax1.set_ylabel("P50 horizon (minutes)")
    ax1.set_title("P50 Time Horizon vs Token Budget")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")

    ax2.set_xlabel("Token budget (thousands)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs Token Budget")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.show()

    # Print summary table
    for alias, data in campaign_data.items():
        result = p50_vs_token_budget(data["runs"], budgets=budgets)
        print(f"\n{alias}:")
        for _, row in result.iterrows():
            print(
                f"  {row['budget']/1000:.0f}k tokens: P50 = {row['p50_minutes']:.1f} min, accuracy = {row['accuracy']:.1%}"
            )


# =============================================================================
# Cost per success (AISI-style inference scaling analysis)
# =============================================================================

# Blended $/token estimates (input + output averaged).
# For reasoning models, output tokens (including thinking) dominate cost.
# These are rough estimates for curve shape - exact values are secondary.
# Sources: campaign docs, provider pricing pages as of Mar 2026.
MODEL_PRICING_PER_TOKEN = {
    "Claude 3 Opus": 45.0 / 1e6,  # $15/$75, ~60/40 in/out split
    "Haiku 4.5": 2.0 / 1e6,  # $0.80/$4, cheap baseline
    "Opus 4": 22.5 / 1e6,  # $15/$75, fixed 16K reasoning
    "Sonnet 4.6": 9.0 / 1e6,  # $3/$15
    "Opus 4.6": 22.5 / 1e6,  # $15/$75, adaptive thinking
    "o1": 30.0 / 1e6,  # $15/$60
    "o3": 20.0 / 1e6,  # $10/$40
    "GPT-5.1 Codex Max": 4.5 / 1e6,  # $1.25/$10
    "GPT-5.2 Codex": 6.0 / 1e6,  # $1.75/$14
    "GPT-5.3 Codex": 4.5 / 1e6,  # est. similar to 5.1
    "GLM-5": 1.0 / 1e6,  # $0.60/$1.70
    "DeepSeek V3.1": 1.0 / 1e6,  # $0.60/$1.70
    "Gemini 2.5 Pro": 5.0 / 1e6,  # $1.25/$10 (Vertex)
    "Gemini 3.1 Pro": 5.0 / 1e6,  # est. similar to 2.5
}


def cost_per_success(runs_df, price_per_token, budgets=None):
    """Compute cost per success at each token budget threshold.

    For each budget, counts successes among runs that finished within
    that budget, and divides total cost (tokens used, capped at budget)
    by the number of successes. Produces the U-shaped curve described
    in UK AISI / Irregular (2026): high at low budgets (few successes),
    falling to a minimum, then rising as extra tokens go to unproductive
    runs.

    Args:
        runs_df: DataFrame with total_tokens and score_binarized columns.
        price_per_token: blended $/token for this model.
        budgets: list of token budget thresholds.

    Returns:
        DataFrame with columns: budget, cost_per_success, total_cost,
        successes, n_runs.
    """
    if budgets is None:
        budgets = [
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            1_500_000,
            2_000_000,
            5_000_000,
            10_000_000,
        ]

    rows = []
    for b in budgets:
        tokens_used = runs_df["total_tokens"].clip(upper=b)
        total_cost = (tokens_used * price_per_token).sum()
        successes = int(
            ((runs_df["score_binarized"] == 1) & (runs_df["total_tokens"] <= b)).sum()
        )
        cps = total_cost / successes if successes > 0 else np.inf
        rows.append(
            {
                "budget": b,
                "cost_per_success": cps,
                "total_cost": total_cost,
                "successes": successes,
                "n_runs": len(runs_df),
            }
        )

    return pd.DataFrame(rows)


def plot_cost_per_success(campaign_data, budgets=None, pricing=None):
    """Plot cost per success vs token budget for all models.

    Produces the AISI-style U-curve showing the economically optimal
    evaluation budget. The dip in each curve indicates where additional
    tokens stop buying enough extra successes to justify the cost.

    Args:
        campaign_data: dict of alias -> {runs, ...}.
        budgets: list of token budgets to evaluate.
        pricing: dict of alias -> $/token. Falls back to MODEL_PRICING_PER_TOKEN.
    """
    if budgets is None:
        budgets = [
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            1_500_000,
            2_000_000,
            5_000_000,
            10_000_000,
        ]
    if pricing is None:
        pricing = MODEL_PRICING_PER_TOKEN

    fig, ax = plt.subplots(figsize=(10, 6))

    for alias, data in campaign_data.items():
        ppt = pricing.get(alias)
        if ppt is None:
            print(f"  Skipping {alias}: no pricing data")
            continue

        result = cost_per_success(data["runs"], ppt, budgets=budgets)
        # Only plot budgets where we have data (total_tokens <= budget for some runs)
        max_observed = data["runs"]["total_tokens"].max()
        plot_mask = result["budget"] <= max(max_observed * 1.2, result["budget"].min())
        plot_data = result[plot_mask]

        budget_m = plot_data["budget"] / 1e6
        cps = plot_data["cost_per_success"].replace([np.inf], np.nan)
        ax.plot(budget_m, cps, marker="o", label=alias, linewidth=2)

    ax.set_xlabel("Token budget (millions)")
    ax.set_ylabel("Cost per success ($)")
    ax.set_title("Cost per Success vs Token Budget")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.show()

    # Print summary
    for alias, data in campaign_data.items():
        ppt = pricing.get(alias)
        if ppt is None:
            continue
        result = cost_per_success(data["runs"], ppt, budgets=budgets)
        print(f"\n{alias} (${ppt * 1e6:.2f}/M tokens):")
        for _, row in result.iterrows():
            cps_str = (
                f"${row['cost_per_success']:.2f}"
                if np.isfinite(row["cost_per_success"])
                else "inf"
            )
            print(
                f"  {row['budget']/1e6:.1f}M tokens: "
                f"cost/success = {cps_str}, "
                f"successes = {row['successes']}/{row['n_runs']}"
            )


# =============================================================================
# Open-source model analysis
# =============================================================================

# Benchmark ordering by approximate difficulty (human time range)
_BENCH_ORDER = [
    "cybashbench",
    "nl2bash",
    "intercode_ctf",
    "nyuctf",
    "cybench",
    "cvebench",
    "cybergym",
]
_BENCH_LABELS = {
    "cybashbench": "CyBashBench\n(1s-30s)",
    "nl2bash": "NL2Bash\n(4s-4m)",
    "intercode_ctf": "InterCode\n(10s-10m)",
    "nyuctf": "NYUCTF\n(2m-6h)",
    "cybench": "CyBench\n(2m-25h)",
    "cvebench": "CVEBench\n(15m-8h)",
    "cybergym": "CyberGym\n(30m-8h)",
}


def _per_benchmark_accuracy(runs_df):
    """Compute per-benchmark accuracy from a runs DataFrame.

    Returns dict of {benchmark: accuracy} where accuracy is the mean
    of score_binarized for that benchmark. Uses task_family column
    (set during prepare step) which contains clean benchmark names.
    """
    col = "task_family" if "task_family" in runs_df.columns else "benchmark"
    if col not in runs_df.columns:
        raise ValueError(
            f"No benchmark/task_family column in DataFrame. Columns: {list(runs_df.columns)}"
        )
    return runs_df.groupby(col)["score_binarized"].mean().to_dict()


def plot_adaptation_buffer(
    campaign_data,
    closed_models,
    open_models,
    release_dates,
    title="Adaptation buffer: time for open-source to match closed-source capability",
    weight_col="invsqrt_task_weight",
    regularization=None,
):
    """Plot P50 time horizon vs release date with horizontal arrows showing
    the adaptation buffer between closed-source and open-source models.

    Directly inspired by UK AISI's Frontier AI Trends Report Figure 25.
    Each closed-source model gets a horizontal arrow to the first open-source
    model that matches or exceeds its P50 horizon. Unmatched models get a
    dotted line extending to the present.

    Args:
        campaign_data: dict of alias -> {runs, runs_human, ...}.
        closed_models: list of closed-source model aliases (plotted as circles).
        open_models: list of open-source model aliases (plotted as squares).
        release_dates: dict of alias -> "YYYY-MM-DD" release date string.
        title: plot title.
        weight_col: weighting column for IRT fitting.
        regularization: IRT regularization. Defaults to DEFAULT_REGULARIZATION.
    """
    from datetime import datetime

    if regularization is None:
        regularization = DEFAULT_REGULARIZATION

    # Fit P50 for all models
    p50s = {}  # alias -> (release_date, p50_minutes)
    for alias in closed_models + open_models:
        data = campaign_data.get(alias)
        if data is None:
            continue
        runs = data.get("runs_human", data.get("runs"))
        if runs is None or len(runs) == 0:
            continue

        log2_times = runs["log2_human_minutes"].values
        scores = runs["score_binarized"].values
        weights = runs[weight_col].values if weight_col in runs.columns else None

        p50_log2, _, _ = fit_p50(
            log2_times, scores, weights=weights, regularization=regularization
        )
        p50_min = 2**p50_log2 if np.isfinite(p50_log2) else 0

        rd = release_dates.get(alias)
        if rd:
            dt = datetime.strptime(rd, "%Y-%m-%d")
            p50s[alias] = (dt, p50_min)

    if not p50s:
        print("No models with both P50 and release date.")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort open models by release date for matching
    open_sorted = sorted(
        [(a, p50s[a]) for a in open_models if a in p50s],
        key=lambda x: x[1][0],
    )

    # Plot closed-source models and find matches
    for alias in closed_models:
        if alias not in p50s:
            continue
        dt, p50 = p50s[alias]

        ax.plot(
            dt,
            p50,
            "o",
            color="#ff5b5b",
            markersize=10,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        # Find first open-source model that matches or exceeds this P50
        match = None
        for os_alias, (os_dt, os_p50) in open_sorted:
            if os_dt > dt and os_p50 >= p50:
                match = (os_alias, os_dt, os_p50)
                break

        if match:
            os_alias, os_dt, os_p50 = match
            # Solid arrow to the matching OS model
            months_gap = (os_dt - dt).days / 30.44
            mid_dt = dt + (os_dt - dt) / 2

            ax.annotate(
                "",
                xy=(os_dt, p50),
                xytext=(dt, p50),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#457b9d",
                    linewidth=1.5,
                    shrinkA=6,
                    shrinkB=6,
                ),
            )
            ax.text(
                mid_dt,
                p50 * 1.15,
                f"{months_gap:.1f} mo",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#457b9d",
                fontweight="bold",
            )
        else:
            # Dotted line to "now" (rightmost date + 1 month)
            now = datetime(2026, 3, 25)
            months_gap = (now - dt).days / 30.44
            ax.plot(
                [dt, now],
                [p50, p50],
                linestyle=":",
                color="#ff5b5b",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.text(
                now,
                p50,
                f"  >{months_gap:.0f} mo",
                ha="left",
                va="center",
                fontsize=8,
                color="#ff5b5b",
                alpha=0.7,
            )

    # Plot open-source models
    for alias in open_models:
        if alias not in p50s:
            continue
        dt, p50 = p50s[alias]
        ax.plot(
            dt,
            p50,
            "s",
            color="#457b9d",
            markersize=10,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    # Label all models
    for alias in closed_models + open_models:
        if alias not in p50s:
            continue
        dt, p50 = p50s[alias]
        # Offset label below or above depending on crowding
        ax.text(
            dt,
            p50 * 0.75,
            alias,
            ha="center",
            va="top",
            fontsize=8,
            color="#555",
            rotation=0,
        )

    # Formatting
    ax.set_yscale("log")
    ax.set_ylabel("P50 time horizon (minutes)")
    ax.set_xlabel("Model release date")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Y-axis time labels
    tick_vals = [1, 2, 5, 10, 30, 60, 120, 240]
    tick_labels = ["1m", "2m", "5m", "10m", "30m", "1h", "2h", "4h"]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(0.5, 500)

    ax.grid(alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff5b5b",
            markersize=10,
            label="Closed-source model",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#457b9d",
            markersize=10,
            label="Open-source model",
        ),
        Line2D(
            [0],
            [0],
            color="#ff5b5b",
            linestyle=":",
            alpha=0.6,
            label="Not yet matched by open-source",
        ),
        Line2D(
            [0],
            [0],
            color="#457b9d",
            linewidth=1.5,
            label="First open-source model to match",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(
        f"\n{'Closed model':<25} {'P50':>10} {'Release':>12} {'Matched by':<20} {'Buffer':>10}"
    )
    print("-" * 80)
    for alias in closed_models:
        if alias not in p50s:
            continue
        dt, p50 = p50s[alias]
        match = None
        for os_alias, (os_dt, os_p50) in open_sorted:
            if os_dt > dt and os_p50 >= p50:
                match = (os_alias, os_dt)
                break
        if match:
            months = (match[1] - dt).days / 30.44
            print(
                f"{alias:<25} {_fmt_time(p50):>10} {dt.strftime('%Y-%m'):>12} {match[0]:<20} {months:>7.1f} mo"
            )
        else:
            now = datetime(2026, 3, 25)
            months = (now - dt).days / 30.44
            print(
                f"{alias:<25} {_fmt_time(p50):>10} {dt.strftime('%Y-%m'):>12} {'not yet matched':<20} {'>' + f'{months:.0f}':>7} mo"
            )

    return fig


def plot_buffer_by_difficulty(
    campaign_data,
    os_models,
    closed_models,
    release_dates,
    title="Adaptation buffer by benchmark difficulty",
):
    """Plot adaptation buffer (months behind frontier) per benchmark.

    For each benchmark, finds the closest-matching closed-source model
    for each open-source model and computes the date lag. This directly
    shows whether the open-closed gap widens on harder tasks.

    Args:
        campaign_data: dict of alias -> {runs, ...}.
        os_models: list of (alias, release_date_str) for open-source models.
        closed_models: list of (alias, release_date_str) for closed-source models.
        release_dates: dict of alias -> "YYYY-MM-DD".
    """
    from datetime import datetime

    # Compute per-benchmark accuracy for all models
    all_accs = {}
    for alias in [a for a, _ in os_models] + [a for a, _ in closed_models]:
        data = campaign_data.get(alias)
        if data is None:
            continue
        all_accs[alias] = _per_benchmark_accuracy(data["runs"])

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#6d597a", "#457b9d"]
    markers = ["s", "^"]

    for idx, (os_alias, os_release_str) in enumerate(os_models):
        os_date = datetime.strptime(os_release_str, "%Y-%m-%d")
        os_accs = all_accs.get(os_alias, {})

        benches = [b for b in _BENCH_ORDER if b in os_accs]
        months_behind = []
        matched_models = []

        for bench in benches:
            os_acc = os_accs[bench]

            # Find closest closed-source model by accuracy on this benchmark
            best_match = None
            best_diff = float("inf")
            for cs_alias, cs_release_str in closed_models:
                cs_accs = all_accs.get(cs_alias, {})
                if bench not in cs_accs:
                    continue
                diff = abs(cs_accs[bench] - os_acc)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (cs_alias, cs_release_str, cs_accs[bench])

            if best_match:
                cs_alias, cs_release_str, cs_acc = best_match
                cs_date = datetime.strptime(cs_release_str, "%Y-%m-%d")
                gap_months = (os_date - cs_date).days / 30.44
                months_behind.append(gap_months)
                matched_models.append(cs_alias)
            else:
                months_behind.append(0)
                matched_models.append("none")

        labels = [_BENCH_LABELS.get(b, b) for b in benches]

        ax.plot(
            range(len(benches)),
            months_behind,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=9,
            label=os_alias,
        )

        # Annotate each point with the matched model
        for i, (mb, mm) in enumerate(zip(months_behind, matched_models)):
            short_name = (
                mm.replace("GPT-5.1 Codex Max", "5.1CM")
                .replace("GPT-5.2 Codex", "5.2C")
                .replace("GPT-5.3 Codex", "5.3C")
                .replace("Claude 3 Opus", "C3O")
                .replace("Gemini 2.5 Pro", "G2.5")
                .replace("Opus 4", "Op4")
            )
            ax.annotate(
                short_name,
                (i, mb),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color=colors[idx % len(colors)],
                alpha=0.7,
            )

    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Months behind closest closed-source model")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="grey", linewidth=0.5)

    ax.annotate(
        "Harder tasks \u2192",
        xy=(0.95, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="grey",
        style="italic",
    )

    plt.tight_layout()
    plt.show()

    # Print table
    print(f"\n{'Benchmark':<20}", end="")
    for os_alias, _ in os_models:
        print(f" {os_alias + ' (months)':>20} {'matched':>15}", end="")
    print()
    print("-" * 90)
    for i, bench in enumerate(benches):
        print(f"{bench:<20}", end="")
        for idx, (os_alias, os_release_str) in enumerate(os_models):
            os_accs = all_accs.get(os_alias, {})
            os_date = datetime.strptime(os_release_str, "%Y-%m-%d")
            os_acc = os_accs.get(bench, 0)

            best_match = None
            best_diff = float("inf")
            for cs_alias, cs_release_str in closed_models:
                cs_accs = all_accs.get(cs_alias, {})
                if bench not in cs_accs:
                    continue
                diff = abs(cs_accs[bench] - os_acc)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (cs_alias, cs_release_str)

            if best_match:
                cs_date = datetime.strptime(best_match[1], "%Y-%m-%d")
                gap = (os_date - cs_date).days / 30.44
                print(f" {gap:>18.1f}mo {best_match[0]:>15}", end="")
            else:
                print(f" {'—':>20} {'—':>15}", end="")
        print()

    return fig


def plot_scurve_overlay(
    campaign_data,
    models,
    title="S-Curve Comparison",
    weight_col="invsqrt_task_weight",
    regularization=None,
):
    """Overlay IRT S-curves for selected models on a single axis.

    Useful for comparing open-source models against their frontier
    equivalents. Shows where the curves overlap (short-horizon tasks)
    and where they diverge (long-horizon tasks).

    Args:
        campaign_data: dict of alias -> {runs, runs_human, ...}.
        models: list of model aliases to overlay (e.g. ["GLM-5", "o3", "Opus 4.6"]).
        title: plot title.
        weight_col: weighting column for IRT fitting.
        regularization: IRT regularization parameter. Defaults to DEFAULT_REGULARIZATION.
    """
    if regularization is None:
        regularization = DEFAULT_REGULARIZATION

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use consistent colors per model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    x_range = np.linspace(-2, 14, 300)  # log2 minutes: ~15s to ~16000h

    for alias, color in zip(models, colors):
        data = campaign_data.get(alias)
        if data is None:
            print(f"  Skipping {alias}: not in campaign_data")
            continue

        # Prefer human-derived times if available
        runs = data.get("runs_human", data.get("runs"))
        if runs is None or len(runs) == 0:
            continue

        log2_times = runs["log2_human_minutes"].values
        scores = runs["score_binarized"].values
        weights = runs[weight_col].values if weight_col in runs.columns else None

        p50_log2, coef, intercept = fit_p50(
            log2_times, scores, weights=weights, regularization=regularization
        )

        # Plot the fitted curve
        probs = expit(coef * x_range + intercept) * 100
        p50_min = 2**p50_log2 if np.isfinite(p50_log2) else 0
        label = f"{alias} (P50={_fmt_time(p50_min)})"
        ax.plot(x_range, probs, color=color, linewidth=2.5, label=label)

        # Mark P50 with a dot
        if np.isfinite(p50_log2):
            ax.plot(p50_log2, 50, "o", color=color, markersize=8, zorder=5)

    # Format x-axis with human-readable time labels
    tick_positions = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tick_labels = [
        "30s",
        "1m",
        "2m",
        "4m",
        "8m",
        "16m",
        "30m",
        "1h",
        "2h",
        "4h",
        "8h",
        "16h",
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(x_range[0], x_range[-1])

    ax.axhline(50, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Human time (log scale)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def plot_gap_by_difficulty(
    campaign_data,
    os_models,
    frontier_models,
    title="Open-Source vs Frontier: Gap by Benchmark Difficulty",
):
    """Plot accuracy gap between open-source and frontier models by benchmark.

    Benchmarks on x-axis ordered by difficulty. Y-axis shows accuracy gap
    (frontier - open-source). Visualises the widening-gap-by-difficulty
    pattern where OS models are near-parity on easy tasks but fall behind
    on hard ones.

    Args:
        campaign_data: dict of alias -> {runs, ...}.
        os_models: list of open-source model aliases.
        frontier_models: list of frontier model aliases (gap computed as
            max frontier accuracy - OS accuracy per benchmark).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute per-benchmark accuracy for all models
    frontier_accs = {}
    for alias in frontier_models:
        data = campaign_data.get(alias)
        if data is None:
            continue
        accs = _per_benchmark_accuracy(data["runs"])
        for bench, acc in accs.items():
            frontier_accs[bench] = max(frontier_accs.get(bench, 0), acc)

    # Plot gap for each OS model
    colors = ["#6d597a", "#457b9d"]  # plum, slate
    markers = ["s", "^"]
    for idx, alias in enumerate(os_models):
        data = campaign_data.get(alias)
        if data is None:
            continue
        os_accs = _per_benchmark_accuracy(data["runs"])

        benches = [b for b in _BENCH_ORDER if b in os_accs and b in frontier_accs]
        gaps = [(frontier_accs[b] - os_accs[b]) * 100 for b in benches]
        labels = [_BENCH_LABELS.get(b, b) for b in benches]

        ax.plot(
            range(len(benches)),
            gaps,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=8,
            label=alias,
        )

    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy gap (frontier - open-source, pp)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="grey", linewidth=0.5)

    # Annotate difficulty direction
    ax.annotate(
        "Harder tasks →",
        xy=(0.95, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="grey",
        style="italic",
    )

    plt.tight_layout()
    plt.show()

    # Print table
    print(f"\n{'Benchmark':<20} {'Frontier':>10}", end="")
    for alias in os_models:
        print(f" {alias:>15}", end="")
    print(f" {'Gap':>10}")
    print("-" * 70)
    for b in _BENCH_ORDER:
        if b not in frontier_accs:
            continue
        print(f"{b:<20} {frontier_accs[b]*100:>9.1f}%", end="")
        for alias in os_models:
            data = campaign_data.get(alias)
            if data is None:
                print(f" {'—':>15}", end="")
                continue
            os_accs = _per_benchmark_accuracy(data["runs"])
            acc = os_accs.get(b)
            if acc is not None:
                print(f" {acc*100:>14.1f}%", end="")
            else:
                print(f" {'—':>15}", end="")
        print()

    return fig


def plot_per_benchmark_comparison(
    campaign_data,
    models,
    title="Per-Benchmark Accuracy Comparison",
):
    """Grouped bar chart comparing models across benchmarks.

    Shows all selected models side-by-side for each benchmark, ordered
    by benchmark difficulty. More intuitive than a table for seeing
    patterns across the difficulty spectrum.

    Args:
        campaign_data: dict of alias -> {runs, ...}.
        models: list of model aliases to compare.
    """
    # Compute accuracies
    all_accs = {}
    for alias in models:
        data = campaign_data.get(alias)
        if data is None:
            continue
        all_accs[alias] = _per_benchmark_accuracy(data["runs"])

    benches = [b for b in _BENCH_ORDER if any(b in accs for accs in all_accs.values())]
    n_models = len(models)
    n_benches = len(benches)

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_models))

    for i, alias in enumerate(models):
        accs = all_accs.get(alias, {})
        vals = [accs.get(b, 0) * 100 for b in benches]
        positions = (
            np.arange(n_benches) + i * bar_width - (n_models - 1) * bar_width / 2
        )
        ax.bar(
            positions,
            vals,
            bar_width * 0.9,
            label=alias,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(range(n_benches))
    ax.set_xticklabels([_BENCH_LABELS.get(b, b) for b in benches], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=min(n_models, 4))
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.show()

    return fig


def _fmt_time(minutes):
    """Format minutes into human-readable string."""
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes/60:.1f}h"


# =============================================================================
# Open-Source Analysis Figures
# =============================================================================


def plot_os_trendline_buffer(
    campaign_data,
    closed_sota,
    open_models,
    release_dates,
    title=None,
    weight_col="invsqrt_task_weight",
    regularization=None,
):
    """Primary OS analysis figure: where do open-source models sit on the
    closed-source capability timeline?

    Left panel: P50 trendline with OS models projected onto it. Horizontal
    arrows show the adaptation buffer (months between OS model's release and
    when closed-source frontier reached equivalent capability).

    Right panel: Per-benchmark buffer showing how the gap varies with task
    difficulty. Uses accuracy-matching to find the closest closed-source
    equivalent per benchmark, then computes the date lag.

    Inspired by UK AISI Frontier AI Trends Report Figures 24-25.
    """
    from datetime import datetime, timedelta

    if regularization is None:
        regularization = DEFAULT_REGULARIZATION

    # -- Fit P50 for all models --
    p50s = {}  # alias -> (release_datetime, p50_minutes, p50_log2)
    for alias in closed_sota + open_models:
        data = campaign_data.get(alias)
        if data is None:
            continue
        runs = data.get("runs_human", data.get("runs"))
        if runs is None or len(runs) == 0:
            continue

        log2_times = runs["log2_human_minutes"].values
        scores = runs["score_binarized"].values
        weights = runs[weight_col].values if weight_col in runs.columns else None

        p50_log2, _, _ = fit_p50(
            log2_times, scores, weights=weights, regularization=regularization
        )
        p50_min = 2**p50_log2 if np.isfinite(p50_log2) else 0

        rd = release_dates.get(alias)
        if rd:
            dt = datetime.strptime(rd, "%Y-%m-%d")
            p50s[alias] = (dt, p50_min, p50_log2)

    if not p50s:
        print("No models with both P50 and release date.")
        return None

    # -- Fit exponential trendline through closed SOTA --
    sota_dates = []
    sota_log2_p50s = []
    for alias in closed_sota:
        if alias not in p50s:
            continue
        dt, p50_min, p50_log2 = p50s[alias]
        if np.isfinite(p50_log2) and p50_min > 0:
            sota_dates.append(dt)
            sota_log2_p50s.append(p50_log2)

    if len(sota_dates) < 2:
        print("Need at least 2 SOTA models for trendline.")
        return None

    # Convert dates to days since epoch for regression
    epoch = datetime(2019, 1, 1)
    sota_days = np.array([(d - epoch).days for d in sota_dates]).reshape(-1, 1)
    sota_y = np.array(sota_log2_p50s)

    trend_model = LinearRegression()
    trend_model.fit(sota_days, sota_y)
    slope = trend_model.coef_[0]  # log2(minutes) per day
    intercept = trend_model.intercept_

    # -- Create figure --
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ========== LEFT PANEL: Trendline buffer ==========

    # Plot trendline
    date_range = pd.date_range("2024-01-01", "2026-06-01", freq="MS")
    trend_days = np.array([(d.to_pydatetime() - epoch).days for d in date_range])
    trend_p50 = 2 ** (slope * trend_days + intercept)

    ax1.plot(
        date_range,
        trend_p50,
        color="#cccccc",
        linewidth=2.5,
        zorder=1,
        label="Closed-source trend",
    )

    # Plot closed-source SOTA models
    for alias in closed_sota:
        if alias not in p50s:
            continue
        dt, p50_min, _ = p50s[alias]
        ax1.plot(
            dt,
            p50_min,
            "o",
            color="#ff5b5b",
            markersize=9,
            zorder=4,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        # Label
        ax1.annotate(
            alias,
            (dt, p50_min),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7.5,
            color="#888",
        )

    # Plot open-source models with buffer arrows
    os_colors = {"GLM-5": "#2a9d8f", "DeepSeek V3.1": "#e76f51"}
    os_markers = {"GLM-5": "s", "DeepSeek V3.1": "D"}

    for alias in open_models:
        if alias not in p50s:
            continue
        dt, p50_min, p50_log2 = p50s[alias]
        color = os_colors.get(alias, "#457b9d")
        marker = os_markers.get(alias, "s")

        # Plot the OS model
        ax1.plot(
            dt,
            p50_min,
            marker,
            color=color,
            markersize=11,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=2,
        )

        # Find where this P50 intersects the trendline (effective date)
        if np.isfinite(p50_log2) and slope > 0:
            effective_days = (p50_log2 - intercept) / slope
            effective_date = epoch + timedelta(days=effective_days)
            buffer_months = (dt - effective_date).days / 30.44

            # Draw horizontal arrow from OS model to trendline intersection
            ax1.annotate(
                "",
                xy=(effective_date, p50_min),
                xytext=(dt, p50_min),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    linewidth=2,
                    shrinkA=8,
                    shrinkB=0,
                    connectionstyle="arc3,rad=0",
                ),
                zorder=3,
            )

            # Plot a small marker on the trendline at intersection
            ax1.plot(
                effective_date,
                p50_min,
                "o",
                color=color,
                markersize=5,
                zorder=4,
                alpha=0.6,
            )

            # Annotate with months
            mid_date = effective_date + (dt - effective_date) / 2
            ax1.annotate(
                f"{buffer_months:.1f} mo",
                (mid_date, p50_min),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                    linewidth=0.5,
                ),
            )

            # Label the OS model
            ax1.annotate(
                alias,
                (dt, p50_min),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    # Format left panel
    ax1.set_yscale("log")
    tick_vals = [1, 2, 5, 10, 30, 60, 120, 240]
    tick_labels = ["1m", "2m", "5m", "10m", "30m", "1h", "2h", "4h"]
    ax1.set_yticks(tick_vals)
    ax1.set_yticklabels(tick_labels)
    ax1.set_ylim(0.8, 400)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.set_xlim(datetime(2024, 1, 1), datetime(2026, 5, 1))

    ax1.set_ylabel("P50 time horizon", fontsize=11)
    ax1.set_xlabel("Model release date", fontsize=11)
    ax1.set_title(
        "Where open-source models sit on the\nclosed-source capability timeline",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(alpha=0.15, which="both")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff5b5b",
            markersize=9,
            label="Closed-source (SOTA)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#2a9d8f",
            markersize=9,
            label="GLM-5 (open-weight)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#e76f51",
            markersize=9,
            label="DeepSeek V3.1 (open-weight)",
        ),
        Line2D([0], [0], color="#cccccc", linewidth=2.5, label="Closed-source trend"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8.5, framealpha=0.9)

    # ========== RIGHT PANEL: Dumbbell chart (OS vs frontier accuracy) ==========
    # Shows where in the difficulty spectrum the gap manifests.
    # Each benchmark gets a horizontal line from OS accuracy to frontier accuracy.
    # The line length IS the gap. No noisy month-matching needed.

    # Compute per-benchmark accuracy for all models
    all_accs = {}
    for alias in closed_sota + open_models:
        data = campaign_data.get(alias)
        if data is None:
            continue
        all_accs[alias] = _per_benchmark_accuracy(data["runs"])

    # Frontier accuracy = max across the two best closed models per benchmark
    frontier_aliases = closed_sota[-2:]  # Last two = most recent SOTA
    frontier_accs = {}
    for bench in _BENCH_ORDER:
        vals = [all_accs.get(a, {}).get(bench) for a in frontier_aliases]
        vals = [v for v in vals if v is not None]
        if vals:
            frontier_accs[bench] = max(vals)

    benches = [b for b in _BENCH_ORDER if b in frontier_accs]
    labels = [_BENCH_LABELS.get(b, b).replace("\n", " ") for b in benches]
    y_positions = np.arange(len(benches))

    # Plot frontier accuracy as grey dots
    frontier_vals = [frontier_accs[b] * 100 for b in benches]
    ax2.scatter(
        frontier_vals,
        y_positions,
        color="#aaaaaa",
        s=80,
        zorder=4,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        label="Frontier",
    )

    # Plot each OS model and draw connecting lines
    for os_alias in open_models:
        os_accs = all_accs.get(os_alias, {})
        color = os_colors.get(os_alias, "#457b9d")
        marker = os_markers.get(os_alias, "s")

        os_vals = [os_accs.get(b, 0) * 100 for b in benches]

        for i, (os_v, fr_v) in enumerate(zip(os_vals, frontier_vals)):
            y_offset = 0.12 if os_alias == "DeepSeek V3.1" else -0.12
            y = i + y_offset

            # Draw connecting line (the gap)
            ax2.plot(
                [os_v, fr_v], [y, i], color=color, linewidth=1.5, alpha=0.5, zorder=2
            )

            # OS model dot
            ax2.scatter(
                [os_v],
                [y],
                color=color,
                s=70,
                zorder=5,
                marker=marker,
                edgecolors="white",
                linewidths=1,
            )

            # Annotate gap in pp
            gap_pp = fr_v - os_v
            if gap_pp > 3:  # Only label meaningful gaps
                mid_x = (os_v + fr_v) / 2
                ax2.text(
                    mid_x,
                    y + 0.05,
                    f"{gap_pp:.0f}pp",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    alpha=0.8,
                )

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Accuracy (%)", fontsize=10)
    ax2.set_xlim(-5, 105)
    ax2.set_title(
        "Gap widens with task difficulty", fontsize=12, fontweight="bold", pad=10
    )
    ax2.grid(alpha=0.15, axis="x")
    ax2.invert_yaxis()  # Easiest benchmarks at top

    # Add difficulty annotation
    ax2.annotate(
        "Harder \u2193",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color="grey",
        style="italic",
    )

    # Legend for right panel
    from matplotlib.lines import Line2D as L2D

    rp_legend = [
        L2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#aaaaaa",
            markersize=8,
            label="Frontier (best closed)",
        ),
        L2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#2a9d8f",
            markersize=8,
            label="GLM-5",
        ),
        L2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#e76f51",
            markersize=8,
            label="DeepSeek V3.1",
        ),
    ]
    ax2.legend(handles=rp_legend, loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.show()

    # -- Print summary --
    print(
        f"\n{'Model':<20} {'P50':>8} {'Released':>12} {'Effective date':>15} {'Buffer':>10}"
    )
    print("-" * 70)
    for alias in open_models:
        if alias not in p50s:
            continue
        dt, p50_min, p50_log2 = p50s[alias]
        if np.isfinite(p50_log2) and slope > 0:
            effective_days = (p50_log2 - intercept) / slope
            effective_date = epoch + timedelta(days=effective_days)
            buffer_months = (dt - effective_date).days / 30.44
            print(
                f"{alias:<20} {_fmt_time(p50_min):>8} {dt.strftime('%Y-%m'):>12} "
                f"{effective_date.strftime('%Y-%m'):>15} {buffer_months:>7.1f} mo"
            )
        else:
            print(
                f"{alias:<20} {_fmt_time(p50_min):>8} {dt.strftime('%Y-%m'):>12} {'—':>15} {'—':>10}"
            )

    return fig


# =============================================================================
# Simulation
# =============================================================================


def simulate_scores(log2_times, true_p50_min, true_coef, rng=None):
    """Generate stochastic binary scores from a known logistic model.

    Args:
        log2_times: array of log₂(minutes) task difficulties
        true_p50_min: true P50 in minutes
        true_coef: true discrimination coefficient (negative = harder tasks less likely)
        rng: numpy random Generator

    Returns:
        (scores, probabilities) tuple. scores is 0/1 array.
    """
    if rng is None:
        rng = np.random.default_rng()

    log2_t = np.asarray(log2_times, dtype=float)
    intercept = -true_coef * np.log2(true_p50_min)
    log_odds = true_coef * log2_t + intercept
    probs = 1.0 / (1.0 + np.exp(-log_odds))
    scores = (rng.random(len(log2_t)) < probs).astype(float)
    return scores, probs


def expected_success_rate(log2_times, p50_min, coef):
    """Average logistic probability across a set of tasks."""
    log2_t = np.asarray(log2_times, dtype=float)
    intercept = -coef * np.log2(p50_min)
    log_odds = coef * log2_t + intercept
    probs = 1.0 / (1.0 + np.exp(-log_odds))
    return float(np.mean(probs))


# =============================================================================
# Perturbation
# =============================================================================


def apply_perturbation(
    runs_df, b=0.0, c=0.0, sigma=0.0, delta_benchmarks=None, x_bar=None, rng=None
):
    """Apply the linear perturbation model to log2(human_minutes).

    log₂(t') = log₂(t) + b + c·(log₂(t) - x̄) + δ_benchmark + ε

    Args:
        runs_df: DataFrame with task_id, log2_human_minutes, task_source columns
        b: constant bias (doublings)
        c: difficulty-dependent compression/expansion
        sigma: i.i.d. noise std dev (doublings)
        delta_benchmarks: dict of benchmark_name -> offset (doublings)
        x_bar: mean log2 difficulty (computed from data if None)
        rng: numpy random Generator

    Returns:
        Modified copy of runs_df with perturbed log2_human_minutes.
    """
    df = runs_df.copy()

    task_df = df.drop_duplicates("task_id")[
        ["task_id", "log2_human_minutes", "task_source"]
    ]
    task_ids = task_df["task_id"].values
    log2_t = task_df["log2_human_minutes"].values.copy()

    if x_bar is None:
        x_bar = log2_t.mean()

    perturbed = log2_t + b + c * (log2_t - x_bar)

    if delta_benchmarks:
        sources = task_df["task_source"].values
        for bench, delta in delta_benchmarks.items():
            perturbed[sources == bench] += delta

    if sigma > 0:
        if rng is None:
            rng = np.random.default_rng()
        perturbed = perturbed + rng.normal(0, sigma, len(perturbed))

    perturbation_map = dict(zip(task_ids, perturbed))
    df["log2_human_minutes"] = df["task_id"].map(perturbation_map)
    return df


def apply_nonlinear_perturbation(runs_df, func, sigma=0.0, rng=None):
    """Apply a general perturbation function to log2(human_minutes).

    Args:
        runs_df: DataFrame with task_id, log2_human_minutes columns
        func: callable(log2_t_array) -> perturbed_log2_t_array
        sigma: additional i.i.d. noise (applied after func)
        rng: numpy random Generator

    Returns:
        Modified copy of runs_df with perturbed log2_human_minutes.
    """
    df = runs_df.copy()

    task_df = df.drop_duplicates("task_id")[["task_id", "log2_human_minutes"]]
    task_ids = task_df["task_id"].values
    log2_t = task_df["log2_human_minutes"].values.copy()

    perturbed = func(log2_t)

    if sigma > 0:
        if rng is None:
            rng = np.random.default_rng()
        perturbed = perturbed + rng.normal(0, sigma, len(perturbed))

    perturbation_map = dict(zip(task_ids, perturbed))
    df["log2_human_minutes"] = df["task_id"].map(perturbation_map)
    return df


def p50_recovery_sim(
    log2_times,
    true_p50_min,
    true_coef,
    n_iter=2000,
    rng=None,
    regularization=DEFAULT_REGULARIZATION,
):
    """Monte Carlo P50 recovery simulation.

    Generates n_iter synthetic score vectors from the known logistic model,
    fits P50 each time, and reports the distribution.

    Returns dict with: median_h, q25_h, q75_h, iqr_h, rel_iqr,
                       ci80_lo_h, ci80_hi_h, all_p50s_min
    """
    if rng is None:
        rng = np.random.default_rng(42)

    log2_t = np.asarray(log2_times, dtype=float)
    p50s = []

    for _ in range(n_iter):
        scores, _ = simulate_scores(log2_t, true_p50_min, true_coef, rng)
        if scores.sum() == 0 or scores.sum() == len(scores):
            p50s.append(np.nan)
            continue
        p50_log2, _, _ = fit_p50(log2_t, scores, regularization=regularization)
        p50s.append(2**p50_log2 / 60.0 if np.isfinite(p50_log2) else np.nan)

    p50s = np.array(p50s)
    valid = p50s[np.isfinite(p50s)]

    if len(valid) == 0:
        return {
            "median_h": np.nan,
            "q25_h": np.nan,
            "q75_h": np.nan,
            "iqr_h": np.nan,
            "rel_iqr": np.nan,
            "ci80_lo_h": np.nan,
            "ci80_hi_h": np.nan,
            "all_p50s_min": p50s,
        }

    med = np.median(valid)
    q25, q75 = np.percentile(valid, [25, 75])
    ci10, ci90 = np.percentile(valid, [10, 90])

    return {
        "median_h": med,
        "q25_h": q25,
        "q75_h": q75,
        "iqr_h": q75 - q25,
        "rel_iqr": (q75 - q25) / med if med > 0 else np.nan,
        "ci80_lo_h": ci10,
        "ci80_hi_h": ci90,
        "all_p50s_min": p50s,
    }
