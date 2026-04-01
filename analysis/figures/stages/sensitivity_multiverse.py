"""Stage: Source-sensitivity multiverse analysis.

For each source variant, sweeps METR's standard uncertainty dimensions
(weighting × regularization × bootstrap over tasks/models) and produces
records in the same format as METR's multiverse_boxplot wrangle stage.

This integrates our source sensitivity into METR's multiverse framework:
each source variant becomes an additional row in the multiverse boxplot,
with a full distribution rather than a point estimate.

Output: JSON dict of {record_type: [{coef, intercept}, ...]} compatible
with METR's plot/multiverse_boxplot.py plotting code.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

# Use monorepo-level METR checkout (CTH-level copy is outdated)
)

from horizon.wrangle.multiverse_boxplot import (  # noqa: E402
    process_in_parallel,
    process_agent_summaries,
    bootstrap_models,
    reg_to_dict,
)
from horizon.wrangle.logistic import run_logistic_regressions  # noqa: E402
from horizon.wrangle.bootstrap import bootstrap_sample  # noqa: E402
from horizon.plot.logistic_alternative_fits import fit_trendline  # noqa: E402

from figures.stages._common import base_parser, load_params  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402


# Display names for the boxplot y-axis labels.
# Order matches the paper table: quality hierarchy first, then leave-one-out.
VARIANT_ORDER = [
    "headline",
    "completions_only",
    "actuals_only",
    "no_cybench_fb",
    "cybench_adjusted",
    # leave-one-out (short → long horizon)
    "no_cybashbench",
    "no_nl2bash",
    "no_intercode_ctf",
    "no_nyuctf",
    "no_cybench",
    "no_cvebench",
    "no_cybergym",
]

VARIANT_LABELS = {
    "headline": "Headline (best-available)",
    "completions_only": "Study completions only",
    "actuals_only": "Actuals only (completions + first-blood ×2.4)",
    "no_cybench_fb": "No first-blood times",
    "cybench_adjusted": "First-blood ×2.4 adjusted",
    "model_est_human_taskset": "Model estimates (human task set)",
    "model_est_full": "Model estimates (full task set)",
    "no_cybashbench": "No CyBashBench",
    "no_nl2bash": "No NL2Bash",
    "no_intercode_ctf": "No InterCode-CTF",
    "no_nyuctf": "No NYUCTF",
    "no_cybench": "No CyBench",
    "no_cvebench": "No CVEBench",
    "no_cybergym": "No CyberGym",
}


def _write_release_dates_yaml(release_dates: dict[str, str], path: Path) -> None:
    """Write release dates as YAML for METR's run_logistic_regressions."""
    import yaml

    # METR expects {date: {agent: date_obj}} format
    dates = {}
    for alias, date_str in release_dates.items():
        dates[alias] = pd.Timestamp(date_str).date()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"date": dates}, f)


def _fit_trendline_with_frontier_p50(
    df_runs: pd.DataFrame,
    release_dates: Path,
    fig_params: dict[str, Any],
    wrangle_params: dict[str, Any],
    frontier_models: list[str] | None = None,
    bootstrap_over_models: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Like METR's fit_trendline_to_runs but also returns frontier model P50s."""
    agent_summaries = run_logistic_regressions(
        df_runs,
        release_dates,
        wrangle_params,
        include_empirical_rates=False,
        ensure_weights_sum_to_1=False,
    )
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)
    if bootstrap_over_models:
        assert rng is not None
        agent_summaries = bootstrap_models(agent_summaries, rng)
    reg, _ = fit_trendline(
        agent_summaries["p50"],
        pd.to_datetime(agent_summaries["release_date"]),
        log_scale=True,
    )
    result = reg_to_dict(reg)

    if frontier_models:
        for fm in frontier_models:
            match = agent_summaries[agent_summaries["agent"] == fm]
            key = f"frontier_p50_{fm}"
            if len(match) > 0 and pd.notna(match.iloc[0]["p50"]):
                result[key] = float(match.iloc[0]["p50"])
            else:
                result[key] = np.nan
        # Keep backwards-compatible "frontier_p50" as the first model
        result["frontier_p50"] = result.get(
            f"frontier_p50_{frontier_models[0]}", np.nan
        )

    return result


def _get_variant_records(
    runs_df: pd.DataFrame,
    release_dates_file: Path,
    fig_params: dict[str, Any],
    gen: np.random.Generator,
    n_samples: int,
    frontier_models: list[str] | None = None,
) -> list[dict[str, float]]:
    """Run full multiverse sweep for a single source variant.

    Combines bootstrap (tasks + models) with weighting/regularization sweeps.
    """
    categories = ["task_family", "task_id"]

    def process_sample(
        idx: int,
        rng: np.random.Generator,
        df_runs: pd.DataFrame,
        fig_params: dict[str, Any],
        release_dates: Path,
        categories: list[str],
        frontier_models: list[str] | None,
    ) -> dict[str, float]:
        sampled_runs = bootstrap_sample(df_runs, categories, rng)
        wrangle_params = {
            "weighting": rng.choice(fig_params["weightings"]),
            "regularization": rng.choice(fig_params["regularizations"]),
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        try:
            return _fit_trendline_with_frontier_p50(
                sampled_runs,
                release_dates,
                fig_params,
                wrangle_params,
                frontier_models=frontier_models,
                bootstrap_over_models=True,
                rng=rng,
            )
        except (ValueError, np.linalg.LinAlgError):
            return {"coef": np.nan, "intercept": np.nan, "frontier_p50": np.nan}

    return process_in_parallel(
        process_sample,
        n_samples,
        gen,
        df_runs=runs_df,
        fig_params=fig_params,
        release_dates=release_dates_file,
        categories=categories,
        frontier_models=frontier_models,
    )


def main():
    parser = base_parser("Source-sensitivity multiverse analysis")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--sensitivity-dir", required=True, help="Directory with variant parquets"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Bootstrap samples per variant"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frontier-model",
        default=None,
        help="Comma-separated model aliases to track P50 for (e.g. 'GPT-5.3 Codex,Opus 4.6')",
    )
    parser.add_argument(
        "--model-date-cutoff",
        default=None,
        help="ISO date (e.g. 2024-01-01). Only include models released on or after this date.",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    release_dates = load_release_dates()

    # Filter to models released on or after the cutoff date
    if args.model_date_cutoff:
        cutoff = pd.Timestamp(args.model_date_cutoff)
        before = len(release_dates)
        release_dates = {
            alias: date_str
            for alias, date_str in release_dates.items()
            if pd.Timestamp(date_str) >= cutoff
        }
        print(
            f"Model date cutoff {args.model_date_cutoff}: {before} -> {len(release_dates)} models"
        )

    # METR's code expects a YAML file for release dates.
    # Use a unique filename to avoid race conditions when running multiple instances.
    cutoff_suffix = f"_{args.model_date_cutoff}" if args.model_date_cutoff else ""
    rd_yaml = (
        _NOTEBOOKS_DIR
        / "figures"
        / "data"
        / "sensitivity"
        / f"_release_dates{cutoff_suffix}.yaml"
    )
    _write_release_dates_yaml(release_dates, rd_yaml)

    # Build fig_params in METR's expected format
    fig_params = {
        "weightings": params.get("multiverse", {}).get(
            "weightings", ["equal_task_weight", "invsqrt_task_weight"]
        ),
        "regularizations": params.get("multiverse", {}).get(
            "regularizations", [0.2, 0.1, 0.05, 0.02, 0.01]
        ),
        "include_agents": list(release_dates.keys()),
        "n_bootstrap": args.n_samples,
    }

    gen = np.random.default_rng(args.seed)
    record_dict: dict[str, list[dict[str, float]]] = {}

    # Headline: assemble from model_runs + task_difficulties
    from lib.data import assemble_runs as _assemble

    mr = pd.read_parquet(
        _NOTEBOOKS_DIR / args.model_runs
        if not Path(args.model_runs).is_absolute()
        else args.model_runs
    )
    td = pd.read_parquet(
        _NOTEBOOKS_DIR / args.task_difficulties
        if not Path(args.task_difficulties).is_absolute()
        else args.task_difficulties
    )
    headline_df = _assemble(mr, td, "best_available_minutes")
    # METR's code groups by 'agent'. Replace with alias.
    if "alias" in headline_df.columns:
        headline_df = headline_df.drop(columns=["agent"]).rename(
            columns={"alias": "agent"}
        )
    # Filter to models with release dates (respects --model-date-cutoff)
    valid_agents = set(release_dates.keys())
    headline_df = headline_df[headline_df["agent"].isin(valid_agents)]

    frontier_models = (
        [m.strip() for m in args.frontier_model.split(",")]
        if args.frontier_model
        else None
    )

    print(
        f"Headline: {headline_df['task_id'].nunique()} tasks, {headline_df['agent'].nunique()} models"
    )
    label = VARIANT_LABELS.get("headline", "Headline")
    records = _get_variant_records(
        headline_df,
        rd_yaml,
        fig_params,
        gen,
        args.n_samples,
        frontier_models=frontier_models,
    )
    valid = [r for r in records if not np.isnan(r.get("coef", np.nan))]
    record_dict[label] = valid
    print(f"  -> {len(valid)} records")

    # Variants
    sens_dir = Path(args.sensitivity_dir)
    if not sens_dir.is_absolute():
        sens_dir = _NOTEBOOKS_DIR / sens_dir

    for parquet in sorted(sens_dir.glob("runs_human_*.parquet")):
        variant_name = parquet.stem.replace("runs_human_", "")
        label = VARIANT_LABELS.get(variant_name)
        if label is None:
            if variant_name.startswith("no_expert_"):
                # e.g. "no_expert_expert_03" -> "No expert: Expert 03"
                anon_id = variant_name[len("no_expert_"):]
                label = f"No expert: {anon_id.replace('_', ' ').title()}"
            else:
                label = variant_name

        variant_df = pd.read_parquet(parquet)
        if "alias" in variant_df.columns:
            variant_df = variant_df.drop(columns=["agent"]).rename(
                columns={"alias": "agent"}
            )
        variant_df = variant_df[variant_df["agent"].isin(valid_agents)]

        n_tasks = variant_df["task_id"].nunique()
        n_models = variant_df["agent"].nunique()
        print(f"\n{variant_name}: {n_tasks} tasks, {n_models} models")

        if n_tasks < 10:
            print("  -> skipping (too few tasks)")
            continue

        records = _get_variant_records(
            variant_df,
            rd_yaml,
            fig_params,
            gen,
            args.n_samples,
            frontier_models=frontier_models,
        )
        # Filter out degenerate bootstrap samples
        valid = [r for r in records if not np.isnan(r.get("coef", np.nan))]
        record_dict[label] = valid
        n_dropped = len(records) - len(valid)
        drop_msg = f" ({n_dropped} degenerate samples dropped)" if n_dropped else ""
        print(f"  -> {len(valid)} records{drop_msg}")

    # Save in METR's format
    out = Path(args.output)
    if not out.is_absolute():
        out = _NOTEBOOKS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record_dict, f)

    print(f"\nSaved {len(record_dict)} record types to {out}")
    for label, records in record_dict.items():
        coefs = [r["coef"] for r in records]
        dt_days = [np.log(2) / c if c > 0 else np.inf for c in coefs]
        dt_finite = [d for d in dt_days if np.isfinite(d)]
        if dt_finite:
            median_dt = np.median(dt_finite)
            print(
                f"  {label}: median DT = {median_dt:.0f} days ({median_dt/30.44:.1f} months)"
            )


if __name__ == "__main__":
    main()
