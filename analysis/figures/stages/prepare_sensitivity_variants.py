"""Stage 1b: Prepare source-sensitivity variant DataFrames.

Source variants pick a different difficulty column or apply a filter to
task_difficulties.parquet, then assemble runs via assemble_runs().

Participant variants (leave-one-expert-out) go deeper: for each expert,
they filter the raw human snapshot to remove that expert's sessions,
recompute best_available_times and task_difficulties from scratch, and
assemble runs. Requires --human-snapshot. Uses anonymized labels from
data/keep/anonymization_mapping.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.data import (  # noqa: E402
    assemble_runs,
    build_best_available_times,
    build_task_difficulties,
    load_cybench_first_blood,
)


# The 2.4x adjustment factor for CyBench first-blood times.
CYBENCH_FB_ADJUSTMENT = 2.4


def _build_variant_difficulty(
    task_diff: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """Return a task_difficulties DataFrame with a 'variant_minutes' column.

    Each variant defines how to compute the difficulty value from the
    explicit per-source columns. The returned DataFrame has at minimum
    task_id, task_family, and variant_minutes.
    """
    td = task_diff.copy()

    if variant == "completions_only":
        # Only tasks with expert completions
        td["variant_minutes"] = td["completion_minutes"]

    elif variant == "actuals_only":
        # Completions where available, else first-blood x2.4
        td["variant_minutes"] = td["completion_minutes"]
        mask = td["variant_minutes"].isna() & td["firstblood_minutes"].notna()
        td.loc[mask, "variant_minutes"] = (
            td.loc[mask, "firstblood_minutes"] * CYBENCH_FB_ADJUSTMENT
        )

    elif variant == "no_cybench_fb":
        # Best-available but replace first-blood winners with their
        # next-best source (estimate or censored)
        td["variant_minutes"] = td["best_available_minutes"].copy()
        fb_winner = td["best_available_source"] == "first_blood"
        # For tasks where first-blood won, fall back to estimate
        td.loc[fb_winner, "variant_minutes"] = td.loc[fb_winner, "estimate_minutes"]
        # If no estimate either, fall back to censored
        still_null = fb_winner & td["variant_minutes"].isna()
        td.loc[still_null, "variant_minutes"] = td.loc[
            still_null, "censored_lower_minutes"
        ]

    elif variant == "cybench_adjusted":
        # Best-available but first-blood values scaled by 2.4x
        td["variant_minutes"] = td["best_available_minutes"].copy()
        fb_winner = td["best_available_source"] == "first_blood"
        td.loc[fb_winner, "variant_minutes"] = (
            td.loc[fb_winner, "firstblood_minutes"] * CYBENCH_FB_ADJUSTMENT
        )

    elif variant.startswith("no_"):
        # Leave-one-benchmark-out: exclude tasks from that benchmark
        benchmark = variant[3:]
        td["variant_minutes"] = td["best_available_minutes"].copy()
        excluded = td["task_family"] == benchmark
        # CyberGym tasks may use arvo: prefix
        if benchmark == "cybergym":
            excluded = excluded | td["task_id"].str.startswith("arvo:")
        td.loc[excluded, "variant_minutes"] = np.nan

    elif variant == "model_est_human_taskset":
        # Model estimates, but only for tasks in the headline set
        has_human = td["best_available_minutes"].notna()
        td["variant_minutes"] = np.where(
            has_human, td["model_estimate_minutes"], np.nan
        )

    elif variant == "model_est_full":
        # Model estimates for all tasks
        td["variant_minutes"] = td["model_estimate_minutes"]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return td


# =========================================================================
# Participant sensitivity: leave-one-expert-out
# =========================================================================

EXPERT_VARIANT_PREFIX = "no_expert_"

# Anonymization mapping (UUID -> expert_XX) for deterministic, PII-free labels
_ANON_MAPPING_PATH = (
    _NOTEBOOKS_DIR.parent / "data" / "keep" / "anonymization_mapping.json"
)


def _load_anon_mapping() -> dict[str, str]:
    """Load UUID -> expert_XX mapping. Falls back to sorted index if missing."""
    if _ANON_MAPPING_PATH.exists():
        with open(_ANON_MAPPING_PATH) as f:
            return json.load(f).get("user_mapping", {})
    return {}


def _load_cybench_first_blood() -> dict[str, float]:
    """Load CyBench first-blood competition times (expert-independent).

    Delegates to lib.data.load_cybench_first_blood() - the single source
    of truth for first-blood loading.
    """
    return load_cybench_first_blood()


def _build_task_bench(snapshot: dict) -> dict[str, str]:
    """Build task_id -> benchmark lookup from snapshot sessions."""
    task_bench: dict[str, str] = {}
    for key in ("completions", "estimations", "passes", "fails", "censored"):
        for session in snapshot.get(key, []):
            tid = session.get("task_id", "")
            bench = session.get("benchmark", "")
            if not bench:
                bench = (
                    tid.split("_")[0]
                    if "_" in tid
                    else tid.split("/")[0]
                    if "/" in tid
                    else ""
                )
            if tid and bench:
                task_bench[tid] = bench
    return task_bench


def _build_participant_variants(
    model_runs: pd.DataFrame,
    snapshot: dict,
    output_dir: Path,
) -> dict[str, dict]:
    """Build leave-one-expert-out sensitivity variants.

    For each expert who contributed sessions, removes all their data
    (completions, estimations, fails, censored), recomputes
    best_available_times and task_difficulties from scratch, and
    assembles runs. This tests whether any single expert's contributions
    drive the headline results.

    Uses anonymized labels (expert_01, expert_02, ...) from the project's
    anonymization mapping — no real names appear in filenames or outputs.

    Returns summary dict for each generated variant.
    """
    from lib.corrections import KNOWN_OUTLIERS, TIMING_CORRECTIONS
    from lib.outliers import OutlierRegistry

    registry = OutlierRegistry(KNOWN_OUTLIERS)
    fb_minutes = _load_cybench_first_blood()
    task_bench = _build_task_bench(snapshot)
    task_fam = (
        model_runs.drop_duplicates("task_id")
        .set_index("task_id")["task_family"]
        .to_dict()
    )
    anon_mapping = _load_anon_mapping()

    # Only generate variants for experts with actual session data
    active_uids = set()
    for key in ("passes", "fails", "censored", "estimations"):
        for s in snapshot.get(key, []):
            uid = s.get("user_id")
            if uid:
                active_uids.add(uid)

    summary: dict[str, dict] = {}
    for uid in sorted(active_uids):
        # Use anonymized label (expert_01, etc.) for filenames and outputs
        anon_label = anon_mapping.get(uid)
        if anon_label is None:
            # Fallback: assign by sorted position among active experts
            idx = sorted(active_uids).index(uid) + 1
            anon_label = f"expert_{idx:02d}"
        variant_name = f"{EXPERT_VARIANT_PREFIX}{anon_label}"

        # Filter out this expert's sessions
        filtered: dict[str, list] = {}
        for key in ("passes", "fails", "censored", "estimations"):
            filtered[key] = [
                s for s in snapshot.get(key, []) if s.get("user_id") != uid
            ]

        # Rebuild best_available_times without this expert
        bat = build_best_available_times(
            completions=filtered["passes"],
            censored=filtered["fails"] + filtered["censored"],
            first_blood_minutes=fb_minutes,
            estimations=filtered["estimations"],
            timing_corrections=TIMING_CORRECTIONS,
            task_bench=task_bench,
            excluded_task_ids=registry.excluded_task_ids,
        )

        # Rebuild task_difficulties
        td = build_task_difficulties(
            best_available_times=bat,
            snapshot=filtered,
            first_blood_minutes=fb_minutes,
            timing_corrections=TIMING_CORRECTIONS,
            task_families=task_fam,
        )

        # Assemble runs
        combined = assemble_runs(model_runs, td, "best_available_minutes")

        out_path = output_dir / f"runs_human_{variant_name}.parquet"
        if combined.empty:
            print(f"  {variant_name}: 0 tasks — skipping")
            continue

        combined.to_parquet(out_path)
        n_tasks = combined["task_id"].nunique()
        n_models = combined["alias"].nunique()
        print(
            f"  {variant_name}: "
            f"{len(combined)} rows, {n_tasks} tasks, {n_models} models"
        )

        summary[variant_name] = {
            "n_rows": len(combined),
            "n_tasks": int(n_tasks),
            "n_models": int(n_models),
        }

    return summary


# All supported variants
ALL_VARIANTS = [
    "completions_only",
    "actuals_only",
    "no_cybench_fb",
    "cybench_adjusted",
    "no_cybergym",
    "no_cybench",
    "no_cvebench",
    "no_nyuctf",
    "no_intercode_ctf",
    "no_nl2bash",
    "no_cybashbench",
    "model_est_human_taskset",
    "model_est_full",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare source-sensitivity variant DataFrames"
    )
    parser.add_argument(
        "--model-runs",
        required=True,
        help="model_runs.parquet (evaluation results only)",
    )
    parser.add_argument(
        "--task-difficulties",
        required=True,
        help="task_difficulties.parquet (difficulty sources)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=ALL_VARIANTS,
        help="Which variants to build (default: all)",
    )
    parser.add_argument(
        "--human-snapshot",
        default=None,
        help="human_snapshot.json — enables participant (leave-one-expert-out) variants",
    )
    args = parser.parse_args()

    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        td_variant = _build_variant_difficulty(task_diff, variant)
        combined = assemble_runs(model_runs, td_variant, "variant_minutes")

        out_path = out / f"runs_human_{variant}.parquet"
        if combined.empty:
            print(f"  {variant}: 0 tasks with difficulty data — skipping")
            continue

        combined.to_parquet(out_path)
        n_tasks = combined["task_id"].nunique()
        n_models = combined["alias"].nunique()
        print(
            f"  {variant}: {len(combined)} rows, {n_tasks} tasks, {n_models} models -> {out_path.name}"
        )

    # Participant variants (leave-one-expert-out)
    if args.human_snapshot:
        with open(args.human_snapshot) as f:
            snapshot = json.load(f)
        print("\nGenerating participant sensitivity variants:")
        expert_summary = _build_participant_variants(model_runs, snapshot, out)
    else:
        expert_summary = {}

    # Summary JSON
    summary = {}
    for variant in args.variants:
        p = out / f"runs_human_{variant}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            summary[variant] = {
                "n_rows": len(df),
                "n_tasks": int(df["task_id"].nunique()),
                "n_models": int(df["alias"].nunique()),
            }
    summary.update(expert_summary)
    with open(out / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
