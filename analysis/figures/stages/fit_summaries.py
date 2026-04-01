"""Stage: Fit IRT logistic curves for all models using METR's code.

Central trunk stage. All downstream analysis (trendline, sensitivity,
figures) reads from the model_summaries parquet this produces. Nobody
fits their own P50s.

Uses assemble_runs() to join model_runs with a named difficulty column,
then calls METR's agent_regression() for each model.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

# Use monorepo-level METR checkout (CTH-level copy is outdated)

from horizon.wrangle.logistic import agent_regression  # noqa: E402
from lib.constants import DEFAULT_REGULARIZATION  # noqa: E402
from lib.data import assemble_runs  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402
from figures.stages._sota import compute_sota_set  # noqa: E402


_WRANGLE_PARAMS = {
    "weighting": "invsqrt_task_weight",
    "regularization": DEFAULT_REGULARIZATION,
    "success_percents": [50, 80],
    "confidence_level": 0.95,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit IRT models via METR's agent_regression"
    )
    parser.add_argument("--model-runs", default=None, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", default=None, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--difficulty-col",
        default=None,
        help="Column from task_difficulties to use as difficulty axis",
    )
    parser.add_argument(
        "--assembled-runs",
        default=None,
        help="Pre-assembled parquet (alternative to model-runs + task-difficulties)",
    )
    parser.add_argument(
        "--bootstrap", default=None, help="Bootstrap parquet (optional, adds CI bounds)"
    )
    parser.add_argument("--token-budget", default="null")
    parser.add_argument(
        "--output", required=True, help="Output model_summaries parquet"
    )
    args = parser.parse_args()

    token_budget = None if args.token_budget == "null" else int(args.token_budget)
    release_dates = load_release_dates()

    # Load runs: either assemble from components or use pre-assembled
    if args.assembled_runs:
        runs_df = pd.read_parquet(args.assembled_runs)
    elif args.model_runs and args.task_difficulties and args.difficulty_col:
        model_runs = pd.read_parquet(args.model_runs)
        task_diff = pd.read_parquet(args.task_difficulties)
        runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)
    else:
        raise ValueError(
            "Provide either --assembled-runs or all of --model-runs, --task-difficulties, --difficulty-col"
        )

    # Apply token budget: runs over budget scored as failures
    if token_budget is not None and "total_tokens" in runs_df.columns:
        over = runs_df["total_tokens"] > token_budget
        runs_df.loc[over, "score_binarized"] = 0

    # Use alias as agent identifier (matches release_dates keys)
    if "alias" in runs_df.columns:
        runs_df["agent"] = runs_df["alias"]

    # Load bootstrap if provided
    bootstrap_results = None
    if args.bootstrap:
        bootstrap_results = pd.read_parquet(args.bootstrap)

    # Fit IRT for each model using METR's agent_regression
    results = []
    for agent_name, agent_df in runs_df.groupby("agent"):
        weights = agent_df["invsqrt_task_weight"].values
        regression = agent_regression(
            x=agent_df["human_minutes"].values,
            y=agent_df["score_binarized"].values,
            weights=weights,
            agent_name=agent_name,
            regularization=_WRANGLE_PARAMS["regularization"],
            success_percents=_WRANGLE_PARAMS["success_percents"],
            confidence_level=_WRANGLE_PARAMS["confidence_level"],
            bootstrap_results=bootstrap_results,
            include_empirical_rates=False,
        )
        row = regression.to_dict()
        row["agent"] = agent_name
        row["n_tasks"] = agent_df["task_id"].nunique()
        row["n_runs"] = len(agent_df)
        results.append(row)

    summaries = pd.DataFrame(results)

    # Add release dates
    summaries["release_date"] = pd.to_datetime(summaries["agent"].map(release_dates))

    # Compute SOTA from data using METR's get_sota_agents
    sota_set = compute_sota_set(summaries, release_dates)
    summaries["is_sota"] = summaries["agent"].isin(sota_set)

    # Sort by release date
    summaries = summaries.sort_values("release_date").reset_index(drop=True)

    # Write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(str(out))

    # Print summary
    print(f"Fitted {len(summaries)} models:")
    for _, row in summaries.iterrows():
        sota_marker = " [SOTA]" if row["is_sota"] else ""
        p50 = row.get("p50", 0)
        print(
            f"  {row['agent']:25s}  P50={p50:8.1f}min ({p50/60:.1f}h)  "
            f"release={row['release_date'].strftime('%Y-%m-%d')}{sota_marker}"
        )

    print(f"\nSOTA models: {sorted(sota_set)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
