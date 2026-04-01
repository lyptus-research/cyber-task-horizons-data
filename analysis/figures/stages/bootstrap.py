"""Stage 2: Run hierarchical bootstrap sampling (the slow stage).

Produces bootstrap regression samples for confidence intervals on P50/P80
horizons and trendlines. This is the only expensive stage in the pipeline.

Uses assemble_runs() to join model_runs with a named difficulty column
from task_difficulties, then passes the assembled DataFrame to METR's
compute_bootstrap_regressions.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

# Import METR's bootstrap directly (avoids trendline.py's heavy plot imports)
# Need both paths: src/ for 'from horizon.*' (via symlink), repo root for
# METR's internal 'from src.*' imports.
# Use monorepo-level METR checkout (CTH-level copy is outdated)

from horizon.wrangle.bootstrap import compute_bootstrap_regressions  # noqa: E402
from lib.constants import DEFAULT_REGULARIZATION  # noqa: E402
from lib.data import assemble_runs  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run bootstrap sampling")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--difficulty-col",
        required=True,
        help="Column name from task_difficulties to use as difficulty axis",
    )
    parser.add_argument("--token-budget", default="null")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)
    token_budget = None if args.token_budget == "null" else int(args.token_budget)

    # Assemble: join difficulty source, compute weights
    runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)

    # Apply token budget: runs over budget scored as failures
    if token_budget is not None and "total_tokens" in runs_df.columns:
        over = runs_df["total_tokens"] > token_budget
        runs_df.loc[over, "score_binarized"] = 0

    # Use alias as agent identifier so bootstrap columns match RELEASE_DATES keys
    if "alias" in runs_df.columns:
        runs_df["agent"] = runs_df["alias"]

    bootstrap_df = compute_bootstrap_regressions(
        data=runs_df,
        categories=["task_family", "task_id"],
        n_bootstrap=args.n_bootstrap,
        regularization=DEFAULT_REGULARIZATION,
        weights_col="invsqrt_task_weight",
        success_percents=[50, 80],
        score_col="score_binarized",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_df.to_parquet(str(out))
    print(f"Saved: {out} ({len(bootstrap_df)} rows, {args.n_bootstrap} iterations)")


if __name__ == "__main__":
    main()
