"""Stage: Combined multiverse — source sensitivity + analytical robustness.

Produces a single records JSON combining:
1. Source sensitivity variants (from sensitivity_multiverse.py output)
   with leave-one-benchmark-out pooled into a single distribution
2. METR's standard analytical dimensions run on headline data:
   bootstrap over tasks, bootstrap over models, weighting/regularisation,
   and baseline noise

Output format matches METR's multiverse_boxplot: {label: [{coef, intercept}, ...]}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

# Use monorepo-level METR checkout (CTH-level copy is outdated)
)

from horizon.wrangle.multiverse_boxplot import (  # noqa: E402
    process_agent_summaries,
    process_in_parallel,
    bootstrap_models,
    reg_to_dict,
)
from horizon.wrangle.logistic import run_logistic_regressions  # noqa: E402
from horizon.wrangle.bootstrap import bootstrap_sample  # noqa: E402
from horizon.plot.logistic_alternative_fits import fit_trendline  # noqa: E402

from figures.stages._common import base_parser, load_params  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402


# Source variant keys that are leave-one-benchmark-out
_LEAVE_ONE_OUT_KEYS = {
    "No CyBashBench",
    "No NL2Bash",
    "No InterCode-CTF",
    "No NYUCTF",
    "No CyBench",
    "No CVEBench",
    "No CyberGym",
}

# Ordered labels for the combined plot
COMBINED_ORDER = [
    "Headline (best-available)",
    "Study completions only",
    "Actuals only (completions + first-blood ×2.4)",
    "No first-blood times",
    "First-blood ×2.4 adjusted",
    "Leave-one-benchmark-out",
    "Leave-one-expert-out",
    "Bootstrap (tasks)",
    "Bootstrap (models)",
    "Weighting / regularisation",
    "Overall",
]


def _write_release_dates_yaml(release_dates: dict[str, str], path: Path) -> None:
    import yaml

    dates = {alias: pd.Timestamp(d).date() for alias, d in release_dates.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"date": dates}, f)


def _extract_frontier_p50s(
    agent_summaries: pd.DataFrame, frontier_models: list[str] | None
) -> dict[str, float]:
    """Extract frontier model P50s from agent_summaries. Returns dict of key->value pairs."""
    result = {}
    if not frontier_models:
        return result
    for fm in frontier_models:
        match = agent_summaries[agent_summaries["agent"] == fm]
        key = f"frontier_p50_{fm}"
        if len(match) > 0 and pd.notna(match.iloc[0]["p50"]):
            result[key] = float(match.iloc[0]["p50"])
        else:
            result[key] = np.nan
    # Backwards-compatible key
    result["frontier_p50"] = result.get(f"frontier_p50_{frontier_models[0]}", np.nan)
    return result


def main():
    parser = base_parser(
        "Combined multiverse: source sensitivity + analytical robustness"
    )
    parser.add_argument(
        "--source-records",
        required=True,
        help="multiverse_records.json from sensitivity_multiverse stage",
    )
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--frontier-model",
        default=None,
        help="Comma-separated model aliases to track P50 for",
    )
    parser.add_argument(
        "--model-date-cutoff",
        default=None,
        help="ISO date (e.g. 2024-01-01). Only include models released on or after this date.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = load_params(args.params)
    release_dates = load_release_dates()

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

    gen = np.random.default_rng(args.seed)

    # --- Load source sensitivity records ---
    with open(args.source_records) as f:
        source_records = json.load(f)

    combined: dict[str, list[dict[str, float]]] = {}

    # Copy source treatments, skipping individual LOO entries (pooled below)
    for label, records in source_records.items():
        if label in _LEAVE_ONE_OUT_KEYS:
            continue
        if label.startswith("No expert:"):
            continue
        combined[label] = records

    # Pool leave-one-benchmark-out into single distribution
    loo_records = []
    for label, records in source_records.items():
        if label in _LEAVE_ONE_OUT_KEYS:
            loo_records.extend(records)
    if loo_records:
        combined["Leave-one-benchmark-out"] = loo_records

    # Pool leave-one-expert-out into single distribution
    expert_records = []
    for label, records in source_records.items():
        if label.startswith("No expert:"):
            expert_records.extend(records)
    if expert_records:
        combined["Leave-one-expert-out"] = expert_records

    # --- Run METR's analytical dimensions on headline ---
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

    # METR expects 'agent' column keyed by alias
    if "alias" in headline_df.columns:
        headline_df = headline_df.drop(columns=["agent"]).rename(
            columns={"alias": "agent"}
        )
    # Filter to models with release dates (respects --model-date-cutoff)
    valid_agents = set(release_dates.keys())
    headline_df = headline_df[headline_df["agent"].isin(valid_agents)]

    rd_yaml = (
        _NOTEBOOKS_DIR / "figures" / "data" / "sensitivity" / "_release_dates.yaml"
    )
    _write_release_dates_yaml(release_dates, rd_yaml)

    fig_params = {
        "weightings": params.get("multiverse", {}).get(
            "weightings", ["equal_task_weight", "invsqrt_task_weight"]
        ),
        "regularizations": params.get("multiverse", {}).get(
            "regularizations", [0.2, 0.1, 0.02, 0.01, 0.001, 0.00001]
        ),
        "include_agents": list(release_dates.keys()),
        "n_bootstrap": args.n_samples,
    }

    # Fit headline summaries for bootstrap-over-models
    wrangle_params = {
        "weighting": "invsqrt_task_weight",
        "regularization": 0.00001,
        "exclude": None,
        "success_percents": [50],
        "confidence_level": 0.95,
    }
    agent_summaries = run_logistic_regressions(
        headline_df,
        rd_yaml,
        wrangle_params,
        include_empirical_rates=False,
        ensure_weights_sum_to_1=False,
    )
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)

    frontier_models = (
        [m.strip() for m in args.frontier_model.split(",")]
        if args.frontier_model
        else None
    )

    # --- Analytical dimensions with frontier P50 extraction ---

    def _fit_with_p50(
        df_runs,
        release_dates,
        fig_params,
        wrangle_params,
        bootstrap_over_models=False,
        rng=None,
    ):
        """Fit trendline and extract frontier P50s."""
        summaries = run_logistic_regressions(
            df_runs,
            release_dates,
            wrangle_params,
            include_empirical_rates=False,
            ensure_weights_sum_to_1=False,
        )
        summaries = process_agent_summaries(summaries, fig_params)
        if bootstrap_over_models:
            summaries = bootstrap_models(summaries, rng)
        reg, _ = fit_trendline(
            summaries["p50"],
            pd.to_datetime(summaries["release_date"]),
            log_scale=True,
        )
        result = reg_to_dict(reg)
        result.update(_extract_frontier_p50s(summaries, frontier_models))
        return result

    # Bootstrap (tasks)
    print("Running bootstrap (tasks)...")

    def _boot_tasks(idx, rng, df_runs, fig_params, release_dates, categories):
        sampled = bootstrap_sample(df_runs, categories, rng)
        wp = {
            "weighting": rng.choice(fig_params["weightings"]),
            "regularization": rng.choice(fig_params["regularizations"]),
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        try:
            return _fit_with_p50(
                sampled,
                release_dates,
                fig_params,
                wp,
                bootstrap_over_models=True,
                rng=rng,
            )
        except (ValueError, np.linalg.LinAlgError):
            return {"coef": np.nan, "intercept": np.nan, "frontier_p50": np.nan}

    combined["Bootstrap (tasks)"] = process_in_parallel(
        _boot_tasks,
        fig_params["n_bootstrap"],
        gen,
        df_runs=headline_df,
        fig_params=fig_params,
        release_dates=rd_yaml,
        categories=["task_family", "task_id"],
    )
    print(f"  -> {len(combined['Bootstrap (tasks)'])} records")

    # Bootstrap (models)
    print("Running bootstrap (models)...")

    def _boot_models(idx, rng, agent_summaries):
        df = bootstrap_models(agent_summaries, rng)
        reg, _ = fit_trendline(
            df["p50"],
            pd.to_datetime(df["release_date"]),
            log_scale=True,
        )
        result = reg_to_dict(reg)
        result.update(_extract_frontier_p50s(df, frontier_models))
        return result

    combined["Bootstrap (models)"] = process_in_parallel(
        _boot_models,
        args.n_samples,
        gen,
        agent_summaries=agent_summaries,
    )
    print(f"  -> {len(combined['Bootstrap (models)'])} records")

    # Weighting / regularisation
    print("Running weighting/regularisation sweep...")
    import itertools

    combinations = list(
        itertools.product(
            fig_params["weightings"],
            fig_params["regularizations"],
        )
    )

    def _weight_reg(idx, rng, df_runs, release_dates, fig_params, combinations):
        w, r = combinations[idx]
        wp = {
            "weighting": w,
            "regularization": r,
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        return _fit_with_p50(df_runs, release_dates, fig_params, wp)

    combined["Weighting / regularisation"] = process_in_parallel(
        _weight_reg,
        len(combinations),
        gen,
        df_runs=headline_df,
        release_dates=rd_yaml,
        fig_params=fig_params,
        combinations=combinations,
    )
    print(f"  -> {len(combined['Weighting / regularisation'])} records")

    # Overall
    print("Running overall uncertainty...")
    overall = process_in_parallel(
        _boot_tasks,
        args.n_samples,
        gen,
        df_runs=headline_df,
        fig_params=fig_params,
        release_dates=rd_yaml,
        categories=["task_family", "task_id"],
    )
    combined["Overall"] = [r for r in overall if not np.isnan(r.get("coef", np.nan))]
    print(f"  -> {len(combined['Overall'])} records")

    # Drop "Overall" — redundant with headline and overlaps legend
    combined.pop("Overall", None)

    # Reorder to match COMBINED_ORDER
    ordered: dict[str, list[dict[str, float]]] = {}
    for label in COMBINED_ORDER:
        if label in combined:
            ordered[label] = combined[label]
    # Add any we missed
    for label, records in combined.items():
        if label not in ordered:
            ordered[label] = records

    # Save
    out = Path(args.output)
    if not out.is_absolute():
        out = _NOTEBOOKS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ordered, f)

    print(f"\nSaved {len(ordered)} record types to {out}")
    for label, records in ordered.items():
        coefs = [
            r["coef"] for r in records if r.get("coef") and not np.isnan(r["coef"])
        ]
        if coefs:
            dts = [np.log(2) / c / 30.44 for c in coefs if c > 0]
            if dts:
                print(
                    f"  {label}: median DT = {np.median(dts):.1f} months ({len(records)} records)"
                )


if __name__ == "__main__":
    main()
