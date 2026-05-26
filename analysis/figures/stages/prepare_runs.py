"""Stage 1: Prepare pipeline data from .eval cache + human snapshot.

Loads model evaluation results from cached .eval files, builds
best_available_times from the human snapshot, and produces the two
canonical pipeline tables:

  - model_runs.parquet: evaluation results only (no difficulty column)
  - task_difficulties.parquet: one row per task, explicit columns per
    difficulty source (completion, estimate, first-blood, model estimate)

Also produces best_available_times.json. Legacy models (GPT-2, GPT-3, GPT-3.5) are
merged from the June 2025 study.
"""

import json
import sys
from pathlib import Path

import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.corrections import KNOWN_OUTLIERS, TIMING_CORRECTIONS  # noqa: E402
from lib.data import (  # noqa: E402
    build_best_available_times,
    build_task_difficulties,
    load_cybench_first_blood,
    load_model_time_estimates,
)
from lib.eval_sets import (  # noqa: E402
    CLAUDE_3_OPUS_EVAL_SETS,
    DEEPSEEK_V31_EVAL_SETS,
    GEMINI_25_PRO_EVAL_SETS,
    GLM_5_EVAL_SETS,
    GPT_4O_EVAL_SETS,
    GPT_51_CM_EVAL_SETS,
    GPT_52_EVAL_SETS,
    GPT_53_EVAL_SETS,
    GPT_55_EVAL_SETS,
    O3_EVAL_SETS,
    OPUS_4_EVAL_SETS,
    OPUS_46_EVAL_SETS,
    SONNET_46_EVAL_SETS,
)
from lib.corrections import EXCLUDED_SESSIONS  # noqa: E402
from lib.outliers import OutlierRegistry  # noqa: E402
from lib.results import load_campaign_runs, load_legacy_runs  # noqa: E402

# Import LEGACY_MODELS without triggering the heavy METR imports in trendline.py
LEGACY_MODELS = [
    {"alias_filter": "GPT 2", "agent": "openai/gpt2-xl", "alias": "GPT-2"},
    {"alias_filter": "GPT 3", "agent": "openai/davinci-002", "alias": "GPT-3"},
    {"alias_filter": "GPT 3.5", "agent": "openai/gpt-3.5-turbo", "alias": "GPT-3.5"},
]

CAMPAIGNS = [
    {
        "eval_sets": SONNET_46_EVAL_SETS,
        "agent": "anthropic/claude-sonnet-4-6",
        "alias": "Sonnet 4.6",
    },
    {
        "eval_sets": OPUS_46_EVAL_SETS,
        "agent": "anthropic/claude-opus-4-6",
        "alias": "Opus 4.6",
    },
    {
        "eval_sets": GPT_53_EVAL_SETS,
        "agent": "openai/gpt-5.3-codex",
        "alias": "GPT-5.3 Codex",
    },
    {
        "eval_sets": GPT_55_EVAL_SETS,
        "agent": "openai/gpt-5.5-2026-04-23",
        "alias": "GPT-5.5",
    },
    {"eval_sets": O3_EVAL_SETS, "agent": "openai/o3-2025-04-16", "alias": "o3"},
    {
        "eval_sets": GPT_51_CM_EVAL_SETS,
        "agent": "openai/gpt-5.1-codex-max",
        "alias": "GPT-5.1 Codex Max",
    },
    {"eval_sets": GLM_5_EVAL_SETS, "agent": "together/zai-org/GLM-5", "alias": "GLM-5"},
    {
        "eval_sets": CLAUDE_3_OPUS_EVAL_SETS,
        "agent": "anthropic/claude-3-opus-20240229",
        "alias": "Claude 3 Opus",
    },
    {
        "eval_sets": OPUS_4_EVAL_SETS,
        "agent": "anthropic/claude-opus-4-20250514",
        "alias": "Opus 4",
    },
    {
        "eval_sets": GEMINI_25_PRO_EVAL_SETS,
        "agent": "google/gemini-2.5-pro",
        "alias": "Gemini 2.5 Pro",
    },
    # Haiku 4.5 and o1: excluded — incomplete campaigns, never finished
    # {"eval_sets": HAIKU_45_EVAL_SETS, "agent": "anthropic/claude-haiku-4-5-20251001", "alias": "Haiku 4.5"},
    # {"eval_sets": O1_EVAL_SETS, "agent": "openai/o1-2024-12-17", "alias": "o1"},
    {
        "eval_sets": GPT_52_EVAL_SETS,
        "agent": "openai/gpt-5.2-codex",
        "alias": "GPT-5.2 Codex",
    },
    {
        "eval_sets": DEEPSEEK_V31_EVAL_SETS,
        "agent": "together/deepseek-ai/DeepSeek-V3.1",
        "alias": "DeepSeek V3.1",
    },
    {
        "eval_sets": GPT_4O_EVAL_SETS,
        "agent": "openai/gpt-4o-2024-08-06",
        "alias": "GPT-4o",
    },
]


def _build_task_bench(snapshot: dict) -> dict[str, str]:
    """Build task_id -> benchmark lookup from all snapshot sources."""
    task_bench = {}
    for key in ("completions", "estimations", "passes", "fails", "censored"):
        for session in snapshot.get(key, []):
            tid = session.get("task_id", "")
            bench = (
                tid.split("_")[0]
                if "_" in tid
                else tid.split("/")[0]
                if "/" in tid
                else ""
            )
            if "benchmark" in session:
                bench = session["benchmark"]
            if tid and bench:
                task_bench[tid] = bench
    return task_bench


def _load_cybench_first_blood() -> dict[str, float]:
    """Load CyBench first-blood competition times.

    Delegates to lib.data.load_cybench_first_blood() - the single source
    of truth for first-blood loading.
    """
    return load_cybench_first_blood()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare runs DataFrames")
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.human_snapshot) as f:
        snapshot = json.load(f)

    registry = OutlierRegistry(KNOWN_OUTLIERS)
    task_bench = _build_task_bench(snapshot)
    fb_minutes = _load_cybench_first_blood()

    # Assert excluded sessions haven't leaked into censored/fails data.
    # EXCLUDED_SESSIONS filtering is applied in api.py to completions and
    # estimations, but censored observations are constructed separately.
    excluded_sids = set(EXCLUDED_SESSIONS.keys())
    for key in ("fails", "censored"):
        for session in snapshot.get(key, []):
            sid = session.get("session_id")
            if sid and sid in excluded_sids:
                raise ValueError(
                    f"Excluded session {sid} leaked into snapshot['{key}']: "
                    f"{EXCLUDED_SESSIONS[sid]}"
                )

    best_available_times = build_best_available_times(
        completions=snapshot["passes"],
        censored=snapshot["fails"] + snapshot["censored"],
        first_blood_minutes=fb_minutes,
        estimations=snapshot["estimations"],
        timing_corrections=TIMING_CORRECTIONS,
        task_bench=task_bench,
        excluded_task_ids=registry.excluded_task_ids,
    )

    # Load all model campaigns
    all_runs = []

    for cfg in CAMPAIGNS:
        runs = load_campaign_runs(
            cfg["eval_sets"], agent=cfg["agent"], alias=cfg["alias"]
        )
        if runs.empty:
            print(f"  {cfg['alias']}: no data, skipping")
            continue
        all_runs.append(runs)

    # Load legacy models (GPT-2, GPT-3, GPT-3.5)
    # Filter to tasks that appear in at least one current campaign.
    # Legacy data predates the current exclusion lists (config.py), so tasks
    # excluded for infrastructure reasons (e.g. walking_to_the_sea_side,
    # intercode-ctf tasks requiring external downloads) would otherwise leak
    # into model_runs.parquet and inflate the headline task count.
    current_task_ids = set()
    for df in all_runs:
        current_task_ids.update(df["task_id"].astype(str))

    for cfg in LEGACY_MODELS:
        runs = load_legacy_runs(
            alias_filter=cfg["alias_filter"],
            agent=cfg["agent"],
            alias=cfg["alias"],
        )
        if runs.empty:
            continue
        before = len(runs)
        runs = runs[runs["task_id"].astype(str).isin(current_task_ids)]
        dropped = before - len(runs)
        if dropped:
            print(f"  {cfg['alias']}: dropped {dropped} legacy-only tasks")
        if not runs.empty:
            all_runs.append(runs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bat_serializable = {
        tid: {"minutes": m, "source": s} for tid, (m, s) in best_available_times.items()
    }
    with open(out / "best_available_times.json", "w") as f:
        json.dump(bat_serializable, f, indent=2)

    # --- New pipeline tables: explicit difficulty sources ---

    # model_runs.parquet: evaluation results only, no difficulty column
    runs_concat = pd.concat(all_runs, ignore_index=True)
    model_runs_cols = [
        "task_id",
        "task_family",
        "agent",
        "alias",
        "score_binarized",
        "total_tokens",
    ]
    # Keep only columns that exist
    model_runs_cols = [c for c in model_runs_cols if c in runs_concat.columns]
    runs_concat[model_runs_cols].to_parquet(out / "model_runs.parquet")

    # task_difficulties.parquet: one row per task, one column per source
    model_estimates = load_model_time_estimates()

    # Build task_family lookup from the runs data
    task_fam = (
        runs_concat.drop_duplicates("task_id")
        .set_index("task_id")["task_family"]
        .to_dict()
    )

    task_diff = build_task_difficulties(
        best_available_times=best_available_times,
        snapshot=snapshot,
        first_blood_minutes=fb_minutes,
        timing_corrections=TIMING_CORRECTIONS,
        model_estimates=model_estimates,
        task_families=task_fam,
    )
    task_diff.to_parquet(out / "task_difficulties.parquet", index=False)

    print(f"\nSaved to {out}:")
    print(f"  model_runs.parquet: {len(runs_concat)} rows")
    print(f"  task_difficulties.parquet: {len(task_diff)} tasks")
    print(f"  best_available_times.json: {len(best_available_times)} tasks")


if __name__ == "__main__":
    main()
