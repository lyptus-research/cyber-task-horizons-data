"""Token efficiency and cost analysis for CTH model evaluations.

Analyzes the relationship between token consumption and task outcomes across
the campaign. Three core analyses:

1. Token distributions: how many tokens do successful vs failed runs consume?
   Answers whether models solve tasks efficiently or brute-force to solutions.

2. Cost per success: using MODEL_PRICING, compute the expected dollar cost to
   achieve one success on each task. Reports in both dollars (for threat model
   framing) and tokens (for longitudinal stability).

3. Cost-constrained success rates: what fraction of "successes" remain viable
   under a cost ceiling? Connects to the Irregular article's framing without
   baking dollars into the IRT methodology.

All functions expect the standard runs DataFrame produced by
results.load_campaign_runs (columns: task_id, task_family, score_binarized,
total_tokens, human_minutes, log2_human_minutes, agent, alias).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import format_time_axis
from .lyptus_style import COLORS


# ---------------------------------------------------------------------------
# Pricing data (inlined from eval-pipeline/analysis/cost_estimation.py)
# ---------------------------------------------------------------------------
# The eval-pipeline uses relative imports that prevent direct import from
# the notebooks venv. Inlined here with the subset needed for cost analysis.
# Source of truth remains cost_estimation.py - update both when prices change.


@dataclass
class _Pricing:
    """Per-million-token pricing for a model. Simplified from cost_estimation.ModelPricing."""

    input_per_mtok: float
    output_per_mtok: float


# Pricing as of Jan 2026. Ignores caching (conservative: overestimates cost).
_MODEL_PRICING: dict[str, _Pricing] = {
    "Claude Opus 4.5": _Pricing(5.00, 25.00),
    "Claude Opus 4": _Pricing(15.00, 75.00),
    "Claude Sonnet 4.5": _Pricing(3.00, 15.00),
    "Claude Haiku 4.5": _Pricing(1.00, 5.00),
    "GPT-5.1 Codex": _Pricing(1.25, 10.00),
    "o3": _Pricing(2.00, 8.00),
    "o1": _Pricing(2.00, 8.00),  # approximate
    "Gemini 3 Pro": _Pricing(4.00, 18.00),
    "Gemini 2.5 Pro": _Pricing(2.50, 15.00),
    "Gemini 2.5 Flash": _Pricing(0.30, 2.50),
    # Together AI pricing for open-source models
    "Together AI (large)": _Pricing(0.90, 0.90),  # GLM-5, DeepSeek
}

# Maps campaign aliases (as used in eval_sets.py / notebook loading) to
# _MODEL_PRICING keys. Update when adding new models.
_ALIAS_TO_PRICING: dict[str, str] = {
    "Opus 4.6": "Claude Opus 4.5",  # same pricing tier
    "Sonnet 4.6": "Claude Sonnet 4.5",
    "Haiku 4.5": "Claude Haiku 4.5",
    "Claude 3 Opus": "Claude Opus 4",
    "Opus 4": "Claude Opus 4",
    "GPT-5.3 Codex": "GPT-5.1 Codex",
    "GPT-5.2 Codex": "GPT-5.1 Codex",
    "GPT-5.1 Codex Max": "GPT-5.1 Codex",
    "o3": "o3",
    "o1": "o1",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "Gemini 3.1 Pro": "Gemini 3 Pro",
    "GLM-5": "Together AI (large)",
    "DeepSeek V3.1": "Together AI (large)",
}


def _get_pricing(alias: str) -> _Pricing | None:
    """Look up pricing for a campaign alias. Returns None if unmapped."""
    key = _ALIAS_TO_PRICING.get(alias)
    if key is None:
        return None
    return _MODEL_PRICING.get(key)


def _tokens_to_cost(total_tokens: float, pricing: _Pricing) -> float:
    """Estimate dollar cost from total token count.

    Assumes a 90/10 input/output split (empirically observed across benchmarks)
    and no caching. Conservative estimate - actual costs are typically lower
    due to cache hits and early termination on failures.
    """
    input_tokens = total_tokens * 0.90
    output_tokens = total_tokens * 0.10
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_mtok
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_mtok
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# 1. Token distributions: successful vs failed runs
# ---------------------------------------------------------------------------


def token_distributions(
    campaign_data: dict[str, dict],
) -> pd.DataFrame:
    """Compute token usage statistics split by outcome for each model.

    Returns DataFrame with columns:
        alias, outcome, count, median_tokens, mean_tokens, p25_tokens, p75_tokens
    """
    rows = []
    for alias, data in campaign_data.items():
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue

        for outcome, label in [(1, "success"), (0, "failure")]:
            subset = runs[runs["score_binarized"] == outcome]
            tokens = subset["total_tokens"]
            if len(tokens) == 0:
                continue
            rows.append(
                {
                    "alias": alias,
                    "outcome": label,
                    "count": len(tokens),
                    "median_tokens": int(tokens.median()),
                    "mean_tokens": int(tokens.mean()),
                    "p25_tokens": int(tokens.quantile(0.25)),
                    "p75_tokens": int(tokens.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def plot_token_distributions(
    campaign_data: dict[str, dict],
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Box plot of token usage for successful vs failed runs, per model.

    Shows the median and IQR for each group. Models are sorted by their
    median success token usage (most efficient on the left).
    """
    records = []
    for alias, data in campaign_data.items():
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue
        for _, row in runs.iterrows():
            records.append(
                {
                    "alias": alias,
                    "outcome": "success" if row["score_binarized"] == 1 else "failure",
                    "tokens_k": row["total_tokens"] / 1000,
                }
            )

    if not records:
        print("No token data available.")
        return None

    df = pd.DataFrame(records)

    # Sort models by median tokens on successful runs
    success_medians = (
        df[df["outcome"] == "success"]
        .groupby("alias")["tokens_k"]
        .median()
        .sort_values()
    )
    model_order = list(success_medians.index)
    # Append models with no successes at the end
    for alias in df["alias"].unique():
        if alias not in model_order:
            model_order.append(alias)

    fig, ax = plt.subplots(figsize=figsize)
    colors = {"success": COLORS["teal"], "failure": COLORS["coral"]}
    width = 0.35
    positions_success = np.arange(len(model_order))
    positions_failure = positions_success + width

    for outcome, positions in [
        ("success", positions_success),
        ("failure", positions_failure),
    ]:
        bp_data = []
        bp_positions = []
        for i, alias in enumerate(model_order):
            subset = df[(df["alias"] == alias) & (df["outcome"] == outcome)]
            if len(subset) > 0:
                bp_data.append(subset["tokens_k"].values)
                bp_positions.append(positions[i])

        if bp_data:
            bp = ax.boxplot(
                bp_data,
                positions=bp_positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=False,  # outliers clutter the plot
                medianprops={"color": "black", "linewidth": 1.5},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[outcome])
                patch.set_alpha(0.7)

    ax.set_xticks(positions_success + width / 2)
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tokens (thousands)")
    ax.set_title("Token usage by outcome (success vs failure)")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=colors["success"], alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, fc=colors["failure"], alpha=0.7),
        ],
        labels=["Success", "Failure"],
        loc="upper right",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Cost per success
# ---------------------------------------------------------------------------


def cost_per_success(
    campaign_data: dict[str, dict],
) -> pd.DataFrame:
    """Compute expected cost per success for each model x benchmark.

    Expected cost per success = (mean cost per run) / success_rate.
    This is the amount an actor can expect to spend before achieving one success.

    Also reports the median cost of successful runs (what a success actually
    costs when it happens - typically cheaper because successes finish early).

    Returns DataFrame with columns:
        alias, benchmark, n_runs, success_rate, mean_cost_per_run,
        median_cost_success, expected_cost_per_success, mean_tokens_per_run,
        median_tokens_success
    """
    rows = []
    for alias, data in campaign_data.items():
        pricing = _get_pricing(alias)
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue

        for benchmark, group in runs.groupby("task_family"):
            n = len(group)
            successes = group[group["score_binarized"] == 1]
            n_success = len(successes)
            success_rate = n_success / n if n > 0 else 0.0

            mean_tokens = group["total_tokens"].mean()
            median_tokens_success = (
                successes["total_tokens"].median() if n_success > 0 else np.nan
            )

            mean_cost = _tokens_to_cost(mean_tokens, pricing) if pricing else np.nan
            median_cost_success = (
                _tokens_to_cost(median_tokens_success, pricing)
                if pricing and n_success > 0
                else np.nan
            )

            expected_cps = mean_cost / success_rate if success_rate > 0 else np.inf
            expected_cps_tokens = (
                mean_tokens / success_rate if success_rate > 0 else np.inf
            )

            rows.append(
                {
                    "alias": alias,
                    "benchmark": benchmark,
                    "n_runs": n,
                    "success_rate": success_rate,
                    "mean_tokens_per_run": int(mean_tokens),
                    "median_tokens_success": (
                        int(median_tokens_success)
                        if np.isfinite(median_tokens_success)
                        else None
                    ),
                    "mean_cost_per_run": round(mean_cost, 2)
                    if np.isfinite(mean_cost)
                    else None,
                    "median_cost_success": (
                        round(median_cost_success, 2)
                        if np.isfinite(median_cost_success)
                        else None
                    ),
                    "expected_cost_per_success": (
                        round(expected_cps, 2) if np.isfinite(expected_cps) else None
                    ),
                    "expected_tokens_per_success": (
                        int(expected_cps_tokens)
                        if np.isfinite(expected_cps_tokens)
                        else None
                    ),
                }
            )

    return pd.DataFrame(rows)


def cost_per_success_summary(campaign_data: dict[str, dict]) -> pd.DataFrame:
    """Aggregate cost-per-success across benchmarks for each model.

    Returns one row per model with overall statistics.
    """
    rows = []
    for alias, data in campaign_data.items():
        pricing = _get_pricing(alias)
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue

        n = len(runs)
        successes = runs[runs["score_binarized"] == 1]
        n_success = len(successes)
        success_rate = n_success / n if n > 0 else 0.0

        failures = runs[runs["score_binarized"] == 0]

        mean_tokens = runs["total_tokens"].mean()
        median_tokens_success = (
            successes["total_tokens"].median() if n_success > 0 else np.nan
        )
        median_tokens_failure = (
            failures["total_tokens"].median() if len(failures) > 0 else np.nan
        )

        # Efficiency ratio: median success tokens / median failure tokens.
        # < 1 means successes are cheaper than failures (model solves efficiently).
        # ~ 1 means successes burn about the same tokens as failures (brute-forcing).
        efficiency_ratio = (
            median_tokens_success / median_tokens_failure
            if np.isfinite(median_tokens_success)
            and np.isfinite(median_tokens_failure)
            and median_tokens_failure > 0
            else np.nan
        )

        mean_cost = _tokens_to_cost(mean_tokens, pricing) if pricing else np.nan
        expected_cps = mean_cost / success_rate if success_rate > 0 else np.inf

        rows.append(
            {
                "alias": alias,
                "n_runs": n,
                "n_success": n_success,
                "success_rate": round(success_rate, 3),
                "mean_tokens_per_run": int(mean_tokens),
                "median_tokens_success": (
                    int(median_tokens_success)
                    if np.isfinite(median_tokens_success)
                    else None
                ),
                "median_tokens_failure": (
                    int(median_tokens_failure)
                    if np.isfinite(median_tokens_failure)
                    else None
                ),
                "token_efficiency_ratio": (
                    round(efficiency_ratio, 3)
                    if np.isfinite(efficiency_ratio)
                    else None
                ),
                "mean_cost_per_run": round(mean_cost, 2)
                if np.isfinite(mean_cost)
                else None,
                "expected_cost_per_success": (
                    round(expected_cps, 2) if np.isfinite(expected_cps) else None
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("expected_cost_per_success")


# ---------------------------------------------------------------------------
# 3. Cost-constrained success rates
# ---------------------------------------------------------------------------


def cost_constrained_success(
    runs: pd.DataFrame,
    cost_ceilings_usd: list[float] | None = None,
    alias: str | None = None,
) -> pd.DataFrame:
    """Compute success rate under cost ceilings for a single model.

    A run "succeeds" only if it both solves the task AND costs less than the
    ceiling. This is the "economically viable success rate."

    Args:
        runs: standard runs DataFrame for one model.
        cost_ceilings_usd: list of dollar thresholds. Defaults to a log range.
        alias: model alias for pricing lookup. If None, uses runs["alias"].iloc[0].

    Returns:
        DataFrame with columns: cost_ceiling_usd, n_viable, success_rate,
        delta_from_unconstrained.
    """
    if cost_ceilings_usd is None:
        cost_ceilings_usd = [1, 5, 10, 25, 50, 100, 250, 500]

    if alias is None:
        alias = runs["alias"].iloc[0]
    pricing = _get_pricing(alias)
    if pricing is None:
        print(f"No pricing for {alias}")
        return pd.DataFrame()

    # Compute per-run cost
    runs = runs.copy()
    runs["cost_usd"] = runs["total_tokens"].apply(lambda t: _tokens_to_cost(t, pricing))

    unconstrained_rate = runs["score_binarized"].mean()

    rows = []
    for ceiling in cost_ceilings_usd:
        viable = runs[(runs["score_binarized"] == 1) & (runs["cost_usd"] <= ceiling)]
        rate = len(viable) / len(runs) if len(runs) > 0 else 0.0
        rows.append(
            {
                "cost_ceiling_usd": ceiling,
                "n_viable": len(viable),
                "success_rate": round(rate, 4),
                "delta_from_unconstrained": round(rate - unconstrained_rate, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Token efficiency vs task difficulty
# ---------------------------------------------------------------------------


def token_efficiency_by_difficulty(
    campaign_data: dict[str, dict],
    n_bins: int = 8,
) -> pd.DataFrame:
    """Compute median tokens for successes vs failures across difficulty bins.

    Groups tasks by log2(human_minutes) into equal-width bins and computes
    token statistics for each. This reveals whether harder tasks require
    proportionally more tokens to solve, or whether failures always exhaust
    the budget regardless of difficulty.

    Returns DataFrame with columns:
        alias, bin_center_log2, bin_label, outcome, median_tokens, count
    """
    rows = []
    for alias, data in campaign_data.items():
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue

        log2_min = runs["log2_human_minutes"].min()
        log2_max = runs["log2_human_minutes"].max()
        bin_edges = np.linspace(log2_min, log2_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for i in range(n_bins):
            mask = (runs["log2_human_minutes"] >= bin_edges[i]) & (
                runs["log2_human_minutes"] < bin_edges[i + 1]
            )
            if i == n_bins - 1:  # include right edge in last bin
                mask = mask | (runs["log2_human_minutes"] == bin_edges[i + 1])

            bin_runs = runs[mask]
            minutes = 2 ** bin_centers[i]
            if minutes < 1:
                label = f"{minutes * 60:.0f}s"
            elif minutes < 60:
                label = f"{minutes:.0f}m"
            else:
                label = f"{minutes / 60:.1f}h"

            for outcome, outcome_label in [(1, "success"), (0, "failure")]:
                subset = bin_runs[bin_runs["score_binarized"] == outcome]
                if len(subset) == 0:
                    continue
                rows.append(
                    {
                        "alias": alias,
                        "bin_center_log2": bin_centers[i],
                        "bin_label": label,
                        "outcome": outcome_label,
                        "median_tokens": int(subset["total_tokens"].median()),
                        "count": len(subset),
                    }
                )

    return pd.DataFrame(rows)


def plot_token_efficiency_by_difficulty(
    campaign_data: dict[str, dict],
    models: list[str] | None = None,
    n_bins: int = 8,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot median token usage vs task difficulty, split by outcome.

    One panel per selected model. Shows whether harder tasks consume
    proportionally more tokens, and whether the gap between success and
    failure token usage varies with difficulty.

    Args:
        campaign_data: dict of alias -> {runs, ...}.
        models: list of aliases to plot. Defaults to all with token data.
        n_bins: number of difficulty bins.
    """
    df = token_efficiency_by_difficulty(campaign_data, n_bins=n_bins)
    if df.empty:
        print("No token data available.")
        return None

    if models is None:
        models = list(df["alias"].unique())

    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), squeeze=False
    )

    for idx, alias in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        model_df = df[df["alias"] == alias]

        for outcome, color, marker in [
            ("success", COLORS["teal"], "o"),
            ("failure", COLORS["coral"], "x"),
        ]:
            subset = model_df[model_df["outcome"] == outcome]
            if len(subset) == 0:
                continue
            ax.plot(
                subset["bin_center_log2"],
                subset["median_tokens"] / 1000,
                color=color,
                marker=marker,
                linewidth=1.5,
                label=outcome,
                markersize=6,
            )

        ax.set_title(alias, fontsize=10)
        ax.set_ylabel("Median tokens (K)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        format_time_axis(ax, which="x")

    # Hide unused axes
    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Token efficiency vs task difficulty",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Aggregate summary table
# ---------------------------------------------------------------------------


def _fmt_tokens(n: int | None) -> str:
    """Format token count as human-readable string."""
    if n is None:
        return "-"
    if n < 1_000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n / 1_000:.0f}K"
    return f"{n / 1_000_000:.1f}M"


def print_efficiency_summary(campaign_data: dict[str, dict]) -> None:
    """Print a compact summary table of token efficiency across the campaign.

    Shows: model, success rate, median tokens (success), median tokens (fail),
    efficiency ratio (success/fail median), and expected cost per success.
    """
    rows = []
    for alias, data in campaign_data.items():
        pricing = _get_pricing(alias)
        runs = data["runs"]
        if "total_tokens" not in runs.columns or runs["total_tokens"].sum() == 0:
            continue

        successes = runs[runs["score_binarized"] == 1]
        failures = runs[runs["score_binarized"] == 0]
        n = len(runs)
        n_success = len(successes)
        success_rate = n_success / n if n > 0 else 0.0

        med_success = int(successes["total_tokens"].median()) if n_success > 0 else None
        med_failure = (
            int(failures["total_tokens"].median()) if len(failures) > 0 else None
        )

        efficiency_ratio = (
            round(med_success / med_failure, 3)
            if med_success is not None and med_failure is not None and med_failure > 0
            else None
        )

        mean_cost = (
            _tokens_to_cost(runs["total_tokens"].mean(), pricing) if pricing else None
        )
        ecps = (
            round(mean_cost / success_rate, 2)
            if mean_cost and success_rate > 0
            else None
        )

        rows.append(
            {
                "Model": alias,
                "SR": f"{success_rate:.1%}",
                "Med tok (S)": _fmt_tokens(med_success),
                "Med tok (F)": _fmt_tokens(med_failure),
                "S/F ratio": f"{efficiency_ratio:.4f}" if efficiency_ratio else "-",
                "E[$/success]": f"${ecps:.2f}" if ecps else "-",
            }
        )

    if not rows:
        print("No token data available.")
        return

    df = pd.DataFrame(rows)
    # Sort by success rate descending
    df = df.sort_values("SR", ascending=False)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Cost per success heatmap (publication figure)
# ---------------------------------------------------------------------------

# Benchmark display order: short-horizon to long-horizon
_BENCHMARK_ORDER = [
    "cybashbench",
    "nl2bash",
    "intercode_ctf",
    "nyuctf",
    "cybench",
    "cvebench",
    "cybergym",
]

_BENCHMARK_LABELS = {
    "cybashbench": "CyBashBench",
    "nl2bash": "NL2Bash",
    "intercode_ctf": "InterCode-CTF",
    "nyuctf": "NYUCTF",
    "cybench": "CyBench",
    "cvebench": "CVEBench",
    "cybergym": "CyberGym",
}


def plot_cost_per_success_heatmap(
    campaign_data: dict[str, dict],
    figsize: tuple[float, float] = (12, 6),
    vmax: float | None = None,
) -> plt.Figure:
    """Heatmap of expected cost per success by model x benchmark.

    Each cell shows the dollar cost an attacker would expect to spend
    before achieving one success on that benchmark. Missing data shown as grey.

    Models are ordered by overall expected cost per success (cheapest at top).
    Benchmarks are ordered from short-horizon to long-horizon.
    """
    cps = cost_per_success(campaign_data)
    if cps.empty:
        print("No cost data available.")
        return None

    # Build pivot table
    pivot = cps.pivot_table(
        index="alias",
        columns="benchmark",
        values="expected_cost_per_success",
        aggfunc="first",
    )

    # Order columns by benchmark difficulty (short to long horizon)
    col_order = [b for b in _BENCHMARK_ORDER if b in pivot.columns]
    pivot = pivot[col_order]
    pivot.columns = [_BENCHMARK_LABELS.get(b, b) for b in col_order]

    # Order rows by overall expected cost per success
    summary = cost_per_success_summary(campaign_data)
    row_order = list(summary["alias"])
    pivot = pivot.reindex([a for a in row_order if a in pivot.index])

    fig, ax = plt.subplots(figsize=figsize)

    # Use log scale for color since costs span orders of magnitude
    from matplotlib.colors import LogNorm

    masked = np.ma.masked_invalid(pivot.values.astype(float))
    if vmax is None:
        vmax = (
            float(np.nanmax(pivot.values))
            if not np.all(np.isnan(pivot.values))
            else 100
        )
    vmin = (
        max(0.01, float(np.nanmin(pivot.values[pivot.values > 0])))
        if np.any(pivot.values > 0)
        else 0.01
    )

    im = ax.imshow(
        masked,
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect="auto",
    )

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="#999")
            elif val < 0.01:
                ax.text(
                    j, i, "<$0.01", ha="center", va="center", fontsize=8, color="black"
                )
            elif val < 1:
                ax.text(
                    j,
                    i,
                    f"${val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            elif val < 100:
                ax.text(
                    j,
                    i,
                    f"${val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            else:
                ax.text(
                    j,
                    i,
                    f"${val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    fig.colorbar(im, ax=ax, label="Expected cost per success (USD)", shrink=0.8)
    ax.set_title(
        "Expected cost per success by model and benchmark", fontsize=13, pad=15
    )

    plt.tight_layout()
    return fig
