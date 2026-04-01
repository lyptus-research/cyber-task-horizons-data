"""Calibration pair building, OLS regression, and Tobit MLE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize
from scipy.stats import linregress, norm as sp_norm, t as t_dist

from .corrections import PLACEHOLDER_ESTIMATE_BENCHMARKS, TIMING_CORRECTIONS
from .outliers import OutlierRegistry


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegressionResult:
    """Unified container for OLS and Tobit fits."""

    slope: float
    intercept: float
    se_slope: float
    r_squared: float | None  # None for Tobit
    sigma: float | None  # residual SD (Tobit); OLS sets this too
    n_uncensored: int
    n_censored: int = 0
    method: str = "ols"

    @property
    def n_total(self) -> int:
        return self.n_uncensored + self.n_censored

    @property
    def c_effective(self) -> float:
        """Difficulty-dependent bias: slope - 1."""
        return self.slope - 1.0

    def slope_test(self, h0: float = 1.0) -> tuple[float, float]:
        """Test H0: slope = *h0*. Returns (t_stat, p_value)."""
        if self.se_slope <= 0 or np.isnan(self.se_slope):
            return np.nan, np.nan
        t_stat = (self.slope - h0) / self.se_slope
        p = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(self.n_total - 2, 1)))
        return t_stat, p

    def summary_dict(self) -> dict:
        """Key metrics as a flat dict (for comparison tables)."""
        _, p = self.slope_test()
        return {
            "N": self.n_total,
            "N_unc": self.n_uncensored,
            "N_cens": self.n_censored,
            "slope": self.slope,
            "SE": self.se_slope,
            "c": self.c_effective,
            "R²": self.r_squared,
            "σ": self.sigma,
            "p(slope=1)": p,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------


# Use the canonical timing correction from corrections.py.
from .corrections import corrected_elapsed as _corrected_server_seconds


def build_calibration_pairs(
    completions: list[dict],
    estimate_lookup: dict[str, float],
    task_bench: dict[str, str],
    expert_lookup: dict[str, str],
    registry: OutlierRegistry | None = None,
    timing_corrections: dict[str, float] | None = None,
    placeholder_benchmarks: set[str] | None = None,
) -> pd.DataFrame:
    """Build calibration pairs from completions and an estimate source.

    *estimate_lookup* maps ``task_id -> estimate_minutes`` and can come
    from original AI-assisted estimates, expert estimates, or model estimates.

    Returns DataFrame with one row per (completion, estimate) pair.
    """
    if placeholder_benchmarks is None:
        placeholder_benchmarks = PLACEHOLDER_ESTIMATE_BENCHMARKS

    rows: list[dict] = []
    for c in completions:
        tid = c["task_id"]
        if registry and registry.is_excluded(tid):
            continue
        est = estimate_lookup.get(tid)
        if not est or est <= 0:
            continue
        bench = task_bench.get(tid, "?")
        if bench in placeholder_benchmarks:
            continue
        server_s = _corrected_server_seconds(c, timing_corrections)
        if server_s <= 0:
            continue
        rows.append(
            {
                "task_id": tid,
                "benchmark": bench,
                "expert": expert_lookup.get(c.get("user_id", ""), "?"),
                "session_id": c.get("session_id", ""),
                "actual_min": server_s / 60,
                "estimate_min": est,
                "log2_actual": np.log2(server_s / 60),
                "log2_estimate": np.log2(est),
            }
        )
    return pd.DataFrame(rows)


def build_all_estimate_means(
    estimations: list[dict],
) -> dict[str, float]:
    """Geometric mean of expert estimates per task, in minutes.

    Unlike build_expert_estimate_lookup, this includes ALL estimates
    regardless of whether the task has a completion or who estimated it.
    Used for concordance analysis (expert vs model estimate comparison).
    """
    from collections import defaultdict

    secs_by_task: dict[str, list[float]] = defaultdict(list)
    for e in estimations:
        s = e.get("estimated_seconds")
        if s and s > 0:
            secs_by_task[e["task_id"]].append(s)

    return {
        tid: float(np.exp(np.mean(np.log(secs))) / 60)
        for tid, secs in secs_by_task.items()
    }


def build_expert_estimate_lookup(
    completions: list[dict],
    estimations: list[dict],
    expert_lookup: dict[str, str],
) -> dict[str, dict]:
    """Build task_id -> {est_min, estimators} for expert-estimate track.

    Only pairs completions with estimates from **different** experts.
    Multiple estimates combined via geometric mean.  Returns a dict
    mapping task_id to ``{"est_min": float, "estimators": str}``.
    """
    # Index completions by task_id -> set of completer user_ids
    completer_ids: dict[str, set[str]] = {}
    for c in completions:
        completer_ids.setdefault(c["task_id"], set()).add(c["user_id"])

    # For each completed task, gather estimates from OTHER experts
    result: dict[str, dict] = {}
    for tid, comp_uids in completer_ids.items():
        matching = [
            e
            for e in estimations
            if e["task_id"] == tid
            and e["user_id"] not in comp_uids
            and e.get("estimated_seconds")
            and e["estimated_seconds"] > 0
        ]
        if not matching:
            continue
        est_seconds = [e["estimated_seconds"] for e in matching]
        geo_mean = np.exp(np.mean(np.log(est_seconds)))
        estimators = ", ".join(
            sorted({expert_lookup.get(e["user_id"], "?") for e in matching})
        )
        result[tid] = {"est_min": geo_mean / 60, "estimators": estimators}
    return result


def aggregate_to_task_level(
    cal_df: pd.DataFrame,
    estimate_col: str = "log2_estimate",
) -> pd.DataFrame:
    """Aggregate per-completion pairs to per-task level.

    Geometric mean (arithmetic in log2) for actuals; first estimate value.
    """
    if cal_df.empty:
        return pd.DataFrame()
    return (
        cal_df.groupby("task_id")
        .agg(
            log2_actual=("log2_actual", "mean"),
            log2_estimate=(estimate_col, "first"),
            n_completions=("log2_actual", "count"),
            benchmark=("benchmark", "first"),
            experts=("expert", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Regression fitting
# ---------------------------------------------------------------------------


def fit_ols(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """OLS linear regression with slope=1 test."""
    result = linregress(x, y)
    residuals = y - (result.intercept + result.slope * x)
    return RegressionResult(
        slope=result.slope,
        intercept=result.intercept,
        se_slope=result.stderr,
        r_squared=result.rvalue**2,
        sigma=float(np.std(residuals, ddof=2)),
        n_uncensored=len(x),
        n_censored=0,
        method="ols",
    )


def tobit_regression(
    x: np.ndarray,
    y: np.ndarray,
    censored: np.ndarray,
) -> RegressionResult:
    """Tobit Type I MLE for right-censored calibration data.

    ``censored[i]=True`` means actual >= y[i] (lower bound only).
    """
    unc = ~censored

    # Initialise from OLS on uncensored data
    if unc.sum() >= 3:
        ols_init = linregress(x[unc], y[unc])
        resid = y[unc] - (ols_init.intercept + ols_init.slope * x[unc])
        x0 = [ols_init.intercept, ols_init.slope, np.log(max(np.std(resid), 0.1))]
    else:
        x0 = [float(np.mean(y)), 0.0, np.log(max(float(np.std(y)), 0.1))]

    def negloglik(params):
        a, b, log_s = params
        s = np.exp(log_s)
        mu = a + b * x
        z = (y - mu) / s
        ll = np.where(
            ~censored,
            -0.5 * z**2 - log_s - 0.5 * np.log(2 * np.pi),
            np.log(np.maximum(1 - sp_norm.cdf(z), 1e-15)),
        )
        return -ll.sum()

    res = sp_minimize(negloglik, x0, method="L-BFGS-B")
    a, b, log_s = res.x
    s = np.exp(log_s)

    try:
        H_inv = res.hess_inv @ np.eye(3)  # type: ignore[union-attr]
        se_slope = float(np.sqrt(max(H_inv[1, 1], 0)))
    except Exception:
        se_slope = np.nan

    return RegressionResult(
        slope=b,
        intercept=a,
        se_slope=se_slope,
        r_squared=None,
        sigma=s,
        n_uncensored=int(unc.sum()),
        n_censored=int(censored.sum()),
        method="tobit",
    )


def fit_calibration_track(
    passes_df: pd.DataFrame,
    fails_df: pd.DataFrame | None = None,
    min_pairs: int = 5,
) -> dict[str, RegressionResult | None]:
    """Fit OLS (and optionally Tobit) for a calibration track.

    *passes_df* and *fails_df* should have ``log2_actual`` and
    ``log2_estimate`` columns (task-level aggregated).

    Returns ``{"ols": RegressionResult | None, "tobit": RegressionResult | None}``.
    """
    result: dict[str, RegressionResult | None] = {"ols": None, "tobit": None}

    if passes_df.empty or len(passes_df) < min_pairs:
        return result

    x = passes_df["log2_estimate"].values
    y = passes_df["log2_actual"].values
    result["ols"] = fit_ols(x, y)

    # Tobit: combine passes (uncensored) + fails (censored)
    if fails_df is not None and not fails_df.empty:
        x_all = np.concatenate([x, fails_df["log2_estimate"].values])
        y_all = np.concatenate([y, fails_df["log2_actual"].values])
        cens = np.concatenate(
            [np.zeros(len(x), dtype=bool), np.ones(len(fails_df), dtype=bool)]
        )
        result["tobit"] = tobit_regression(x_all, y_all, cens)

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def comparison_table(
    results: dict[str, RegressionResult | None],
    label: str = "",
) -> str:
    """Format OLS vs Tobit comparison as a printable table."""
    ols = results.get("ols")
    tobit = results.get("tobit")

    if ols is None:
        return f"  {label}: insufficient data for regression"

    lines = [f"=== {label} ===" if label else ""]
    _, p_ols = ols.slope_test()

    if tobit is None:
        # OLS only
        lines.append(f"  N = {ols.n_total}")
        lines.append(
            f"  log2(actual) = {ols.intercept:.2f} + "
            f"{ols.slope:.3f} * log2(estimate)"
        )
        lines.append(f"  Slope = {ols.slope:.3f} +/- {ols.se_slope:.3f}")
        lines.append(f"  R² = {ols.r_squared:.3f}")
        lines.append(f"  c = {ols.c_effective:+.3f}")
        lines.append(f"  p(slope=1) = {p_ols:.3f}{'  *' if p_ols < 0.05 else ''}")
    else:
        # Side-by-side
        _, p_tob = tobit.slope_test()
        w = 16
        lines.append(f"  {'':30s} {'OLS':>{w}s} {'Tobit':>{w}s}")
        lines.append(f"  {'-' * (30 + 2 * w + 2)}")
        lines.append(f"  {'N':30s} {ols.n_total:>{w}d} {tobit.n_total:>{w}d}")
        lines.append(f"  {'Slope':30s} {ols.slope:>{w}.3f} {tobit.slope:>{w}.3f}")
        lines.append(
            f"  {'SE(slope)':30s} {ols.se_slope:>{w}.3f} " f"{tobit.se_slope:>{w}.3f}"
        )
        lines.append(
            f"  {'c = slope - 1':30s} {ols.c_effective:>+{w}.3f} "
            f"{tobit.c_effective:>+{w}.3f}"
        )
        if ols.r_squared is not None:
            lines.append(f"  {'R²':30s} {ols.r_squared:>{w}.3f} {'—':>{w}s}")
        lines.append(
            f"  {'σ (residual)':30s} {ols.sigma:>{w}.2f} " f"{tobit.sigma:>{w}.2f}"
        )
        lines.append(f"  {'p(slope=1)':30s} {p_ols:>{w}.3f} {p_tob:>{w}.3f}")

    return "\n".join(lines)


def pairs_table(
    df: pd.DataFrame,
    estimate_label: str = "Estimate",
) -> str:
    """Format calibration pairs as a printable table."""
    if df.empty:
        return "  (no pairs)"

    lines = [
        f"{'Task':<42s} {'Actual':>8s} {estimate_label:>9s} "
        f"{'Ratio':>7s} {'Bench':<15s} {'Expert(s)'}",
        "-" * 110,
    ]
    for _, row in df.sort_values("log2_actual").iterrows():
        actual = 2 ** row["log2_actual"]
        est = 2 ** row["log2_estimate"]
        ratio = actual / est
        experts = row.get("experts", "?")
        lines.append(
            f"{row['task_id']:<42s} {actual:>7.1f}m {est:>8.1f}m "
            f"{ratio:>6.1f}x {row['benchmark']:<15s} {experts}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slope poolability (TOST equivalence test)
# ---------------------------------------------------------------------------


@dataclass
class SlopeComparisonResult:
    """Result of comparing two calibration track slopes."""

    slope_a: float
    se_a: float
    n_a: int
    slope_b: float
    se_b: float
    n_b: int
    slope_diff: float
    se_diff: float
    p_nhst: float
    tost_p: float
    equiv_margin: float
    poolable: bool | None  # True=equivalent, False=different, None=inconclusive


def compare_slopes(
    slope_a: float, se_a: float, n_a: int,
    slope_b: float, se_b: float, n_b: int,
    equiv_margin: float = 0.2,
) -> SlopeComparisonResult:
    """Compare two regression slopes via NHST and TOST equivalence test.

    Uses Welch's t-test approximation for unequal variances.

    Args:
        slope_a, se_a, n_a: slope, SE, and N for group A.
        slope_b, se_b, n_b: slope, SE, and N for group B.
        equiv_margin: pre-registered equivalence margin (Δ) on slope difference.

    Returns:
        SlopeComparisonResult with NHST p-value, TOST p-value, and poolability verdict.
    """
    slope_diff = slope_a - slope_b
    se_diff = np.sqrt(se_a**2 + se_b**2)
    df_diff = n_a + n_b - 4

    # Classical NHST: H0: slopes equal
    t_diff = slope_diff / se_diff
    p_diff = 2 * (1 - t_dist.cdf(abs(t_diff), df=df_diff))

    # TOST: H0: |slope_diff| >= Δ
    t_upper = (slope_diff - equiv_margin) / se_diff
    t_lower = (slope_diff + equiv_margin) / se_diff
    p_upper = t_dist.cdf(t_upper, df=df_diff)
    p_lower = 1 - t_dist.cdf(t_lower, df=df_diff)
    tost_p = max(p_upper, p_lower)

    if tost_p < 0.05:
        poolable = True
    elif p_diff < 0.05:
        poolable = False
    else:
        poolable = None

    return SlopeComparisonResult(
        slope_a=slope_a, se_a=se_a, n_a=n_a,
        slope_b=slope_b, se_b=se_b, n_b=n_b,
        slope_diff=slope_diff, se_diff=se_diff,
        p_nhst=p_diff, tost_p=tost_p,
        equiv_margin=equiv_margin, poolable=poolable,
    )
