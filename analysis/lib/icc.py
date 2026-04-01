"""ICC(1,1) computation for inter-rater reliability.

Implements Shrout & Fleiss (1979) ICC(1,1) with F-based confidence
intervals for k=2 raters. Used for both estimation and completion
ICC validation gates.
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


# Minimum number of paired tasks for a meaningful ICC estimate.
# Below this, CI width is too large to be informative.
MIN_ICC_TASKS = 10


def compute_icc(
    rows_df: pd.DataFrame,
    min_tasks: int = MIN_ICC_TASKS,
) -> tuple[float | None, float | None, float | None, int, float | None]:
    """Compute ICC(1,1) from paired rater data.

    Args:
        rows_df: DataFrame with columns: task_id, expert, log2_min.
            Each row is one rater's measurement of one task.
        min_tasks: minimum number of tasks with k>=2 raters.

    Returns:
        (icc, ci_lo, ci_hi, n_tasks, sigma_within).
        All None except n_tasks if insufficient data.
    """
    if rows_df.empty:
        return None, None, None, 0, None

    # Truncate to exactly 2 raters per task. If a task has 3+ raters,
    # take the first two (alphabetically by expert ID, matching _extract_pairs).
    # The ICC(1,1) formula assumes balanced k=2 and would be wrong for k>2.
    truncated = (
        rows_df.sort_values(["task_id", "expert"])
        .groupby("task_id")
        .head(2)
    )
    # Extract paired values directly (avoids sparse pivot with NaN columns)
    pairs = []
    for tid, group in truncated.groupby("task_id"):
        vals_list = group["log2_min"].values
        if len(vals_list) >= 2:
            pairs.append(vals_list[:2])
    if len(pairs) < min_tasks:
        return None, None, None, len(pairs), None

    n_tasks = len(pairs)
    k = 2
    vals = np.array(pairs)  # (n_tasks, 2) — dense, no NaNs
    grand_mean = vals.mean()
    task_means = vals.mean(axis=1)
    SS_between = k * np.sum((task_means - grand_mean) ** 2)
    SS_within = np.sum((vals - task_means[:, None]) ** 2)
    MS_between = SS_between / (n_tasks - 1)
    MS_within = SS_within / (n_tasks * (k - 1))

    icc_val = (MS_between - MS_within) / (MS_between + (k - 1) * MS_within)

    F = MS_between / MS_within
    F_lo = F / f_dist.ppf(0.975, n_tasks - 1, n_tasks * (k - 1))
    F_hi = F / f_dist.ppf(0.025, n_tasks - 1, n_tasks * (k - 1))
    ci_lo = (F_lo - 1) / (F_lo + k - 1)
    ci_hi = (F_hi - 1) / (F_hi + k - 1)
    sigma_w = np.sqrt(MS_within)

    return float(icc_val), float(ci_lo), float(ci_hi), n_tasks, float(sigma_w)
