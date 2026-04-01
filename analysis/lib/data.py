"""Data loading helpers for CTH analysis notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Project root: notebooks/ is at cyber-task-horizons/notebooks/
from analysis.config import REPO_ROOT as PROJECT_ROOT, JUNE_2025_DIR

_NOTEBOOKS_DIR = Path(__file__).parent.parent


def load_june_study(project_root=None):
    """Load published June 2025 study data.

    Returns dict with:
        runs: DataFrame (all_runs.jsonl with log2_human_minutes column)
        pub_fits: DataFrame (logistic_fits.csv)
        release_dates: dict (agent full_name -> datetime)
        alias_map: dict (agent full_name -> short alias)
        x_bar: float (mean log2_human_minutes across unique tasks)
        tasks: DataFrame (deduplicated task-level data)
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    metr_dir = root / "published" / "plots" / "metr_data"

    runs = pd.read_json(metr_dir / "all_runs.jsonl", lines=True)
    pub_fits = pd.read_csv(metr_dir / "logistic_fits.csv")

    with open(metr_dir / "release_dates.yaml") as f:
        release_dates = {
            k: pd.to_datetime(v) for k, v in yaml.safe_load(f)["date"].items()
        }

    alias_map = runs.drop_duplicates("agent").set_index("agent")["alias"].to_dict()
    runs["log2_human_minutes"] = np.log2(runs["human_minutes"])

    tasks = runs.drop_duplicates("task_id")
    x_bar = tasks["log2_human_minutes"].mean()

    return {
        "runs": runs,
        "pub_fits": pub_fits,
        "release_dates": release_dates,
        "alias_map": alias_map,
        "x_bar": x_bar,
        "tasks": tasks,
    }


def load_model_time_estimates(project_root=None):
    """Load model-estimated human completion times for all tasks.

    Delegates to ``estimates.load_model_time_estimates()`` — see that
    module for full documentation of sources and load order.
    """
    from .estimates import load_model_time_estimates as _load

    root = Path(project_root) if project_root else PROJECT_ROOT
    return _load(root)


def load_cybench_first_blood(project_root=None) -> dict[str, float]:
    """Load CyBench first-blood competition times.

    Returns dict mapping task_id -> minutes. The source file uses
    estimated_time_seconds (in seconds, despite the misleading field name).

    Single source of truth for first-blood loading - used by prepare_runs
    and prepare_sensitivity_variants.
    """
    import json

    root = Path(project_root) if project_root else PROJECT_ROOT
    fb_path = root / "data" / "keep" / "cybench" / "cybench_human_runs.jsonl"
    fb_minutes: dict[str, float] = {}
    if fb_path.exists():
        with open(fb_path) as f:
            for line in f:
                r = json.loads(line)
                tid = r.get("task_id", "")
                secs = r.get("estimated_time_seconds")
                if tid and secs and secs > 0:
                    fb_minutes[tid] = secs / 60
    return fb_minutes


def build_best_available_times(
    completions: list[dict],
    censored: list[dict],
    first_blood_minutes: dict[str, float],
    estimations: list[dict],
    timing_corrections: dict[str, float] | None = None,
    task_bench: dict[str, str] | None = None,
    excluded_task_ids: set[str] | None = None,
    censored_min_minutes: float = 30.0,
) -> dict[str, tuple[float, str]]:
    """Build task_id -> (minutes, source) using the best available human time.

    Every source in this hierarchy is human-derived. Model estimates are
    excluded — they belong to the IRT pipeline's ``human_minutes`` labels,
    not to this lookup.

    Priority (highest first):

    1. **Successful completions** — geometric mean of server_elapsed across
       sessions. Direct point estimates from the study population.
    2. **Censored completions (when informative)** — expert worked for X
       minutes and could not finish, so actual time >= X. Used only when the
       censored lower bound exceeds the next-best source (first-blood or
       expert estimate). A censored observation from the study population is
       more directly relevant than a competition first-blood time from a
       structurally different population.
    3. **First-blood times** — CyBench competition data. Structurally
       different from individual expert completions (slope 0.89 vs 0.53,
       p=0.013; mean ratio 2.4x).
    4. **Expert estimates** — geometric mean of estimated_seconds across
       raters. Indirect measurement, calibration-corrected.

    *task_bench* maps ``task_id -> benchmark_name``. When provided,
    censored completions on short-horizon benchmarks (cybashbench,
    nl2bash) are skipped — fails on sub-minute tasks almost always
    reflect environment/tooling overhead, not task difficulty.

    *censored_min_minutes* sets a floor below which censored
    observations are discarded (default 30 min). Short censored
    sessions are uninformative lower bounds and add noise.

    *excluded_task_ids* is a set of task IDs to exclude from the output
    (e.g. from ``OutlierRegistry.excluded_task_ids``). These tasks are
    dropped after all sources are merged, so they cannot leak into any
    downstream analysis that uses ``best_available_times``.

    Returns dict mapping task_id to ``(minutes, source_label)`` where
    *source_label* is one of ``"completion"``, ``"censored"``,
    ``"first_blood"``, ``"expert_estimate"``.
    """
    from collections import defaultdict

    corrections = timing_corrections or {}
    bench_lookup = task_bench or {}
    # Benchmarks where fails are environmental noise, not difficulty signal
    _SKIP_CENSORED_BENCHMARKS = {"cybashbench", "nl2bash"}
    result: dict[str, tuple[float, str]] = {}

    # --- 1. Successful completions (geometric mean across sessions) ---
    from .corrections import corrected_elapsed

    comp_times: dict[str, list[float]] = defaultdict(list)
    for c in completions:
        s = corrected_elapsed(c, corrections)
        if s > 0:
            comp_times[c["task_id"]].append(s / 60)

    for tid, times in comp_times.items():
        geo_mean = float(np.exp(np.mean(np.log(times))))
        result[tid] = (geo_mean, "completion")

    # --- 2. Censored completions (lower bounds, used when informative) ---
    # Collect censored times per task (geometric mean if multiple).
    # Only used for tasks without successful completions, and only when
    # the lower bound exceeds the next-best source.
    # Skip short-horizon benchmarks where fails are environmental noise.
    censored_times: dict[str, list[float]] = defaultdict(list)
    for c in censored:
        tid = c["task_id"]
        if bench_lookup.get(tid) in _SKIP_CENSORED_BENCHMARKS:
            continue
        s = corrected_elapsed(c, corrections)
        if s > 0:
            mins = s / 60
            if mins < censored_min_minutes:
                continue
            censored_times[tid].append(mins)

    # Build lower-priority sources first so we can compare
    # --- 3. First-blood times ---
    fallback: dict[str, tuple[float, str]] = {}
    for tid, fb_min in first_blood_minutes.items():
        if fb_min > 0:
            fallback[tid] = (fb_min, "first_blood")

    # --- 4. Expert estimates (geometric mean across raters) ---
    est_times: dict[str, list[float]] = defaultdict(list)
    for e in estimations:
        est_s = e.get("estimated_seconds")
        if est_s and est_s > 0:
            est_times[e["task_id"]].append(est_s / 60)

    for tid, times in est_times.items():
        if tid not in fallback:
            geo_mean = float(np.exp(np.mean(np.log(times))))
            fallback[tid] = (geo_mean, "expert_estimate")

    # Now slot censored observations where they're informative
    for tid, times in censored_times.items():
        if tid in result:
            continue  # already have a successful completion
        censored_min = float(np.exp(np.mean(np.log(times))))
        next_best = fallback.get(tid)
        if next_best is not None and censored_min > next_best[0]:
            # Lower bound exceeds next-best source — use it
            result[tid] = (censored_min, "censored")
        # Otherwise fall through to the next-best source

    # Fill remaining tasks from fallback
    for tid, entry in fallback.items():
        if tid not in result:
            result[tid] = entry

    # Remove known outlier tasks — these have bad timing data that would
    # distort the difficulty axis for all downstream IRT analysis.
    if excluded_task_ids:
        for tid in excluded_task_ids:
            if tid in result:
                result.pop(tid)

    return result


def build_task_bench_lookup(
    phases: dict[str, list[dict]] | None = None,
    completions: list[dict] | None = None,
    estimations: list[dict] | None = None,
    cybench_task_ids: set[str] | None = None,
) -> dict[str, str]:
    """Map task_id -> benchmark name.

    Uses assignment phase data as primary source, then falls back to
    heuristic prefix matching for tasks not in any phase.
    """
    task_bench: dict[str, str] = {}

    if phases:
        for ph_tasks in phases.values():
            for t in ph_tasks:
                task_bench[t["task_id"]] = t["benchmark"]

    all_sessions = (completions or []) + (estimations or [])
    for c in all_sessions:
        tid = c["task_id"]
        if tid in task_bench:
            continue
        if tid.startswith("arvo:") or tid.startswith("oss-fuzz:"):
            task_bench[tid] = "cybergym"
        elif tid.startswith("CVE-"):
            task_bench[tid] = "cvebench"
        elif tid.startswith("cybashbench"):
            task_bench[tid] = "cybashbench"
        elif tid.startswith("intercode-ctf"):
            task_bench[tid] = "intercode-ctf"
        elif tid.startswith("nl2bash"):
            task_bench[tid] = "nl2bash"
        elif "q-" in tid or "f-" in tid:
            task_bench[tid] = "nyuctf"
        elif cybench_task_ids and tid in cybench_task_ids:
            task_bench[tid] = "cybench"
        else:
            task_bench[tid] = "unknown"

    return task_bench


# =========================================================================
# New pipeline: explicit difficulty sources + clean assembly
# =========================================================================


def build_task_difficulties(
    best_available_times: dict[str, tuple[float, str]],
    snapshot: dict,
    first_blood_minutes: dict[str, float],
    timing_corrections: dict[str, float] | None = None,
    model_estimates: dict[str, float] | None = None,
    task_families: dict[str, str] | None = None,
    censored_min_minutes: float = 30.0,
) -> pd.DataFrame:
    """Build a wide-form table with one row per task, one column per source.

    Every difficulty source gets its own column. No ambiguous ``human_minutes``.
    Downstream stages pick the column they need explicitly.

    Columns:
        task_id, task_family,
        completion_minutes, n_completions,
        estimate_minutes, n_estimates,
        firstblood_minutes,
        model_estimate_minutes,
        censored_lower_minutes,
        best_available_minutes, best_available_source
    """
    from collections import defaultdict
    from .corrections import corrected_elapsed

    corrections = timing_corrections or {}
    families = task_families or {}

    # --- Completions (geometric mean per task) ---
    comp_times: dict[str, list[float]] = defaultdict(list)
    for c in snapshot.get("passes", []):
        s = corrected_elapsed(c, corrections)
        if s > 0:
            comp_times[c["task_id"]].append(s / 60)

    # --- Failed/censored completions (lower bounds) ---
    censored_times: dict[str, list[float]] = defaultdict(list)
    for c in snapshot.get("fails", []) + snapshot.get("censored", []):
        s = corrected_elapsed(c, corrections)
        if s > 0:
            mins = s / 60
            if mins < censored_min_minutes:
                continue
            censored_times[c["task_id"]].append(mins)

    # --- Expert estimates (geometric mean per task) ---
    est_times: dict[str, list[float]] = defaultdict(list)
    for e in snapshot.get("estimations", []):
        est_s = e.get("estimated_seconds")
        if est_s and est_s > 0:
            est_times[e["task_id"]].append(est_s / 60)

    # Collect all task IDs from all sources
    all_tids = set()
    all_tids.update(comp_times.keys())
    all_tids.update(censored_times.keys())
    all_tids.update(est_times.keys())
    all_tids.update(first_blood_minutes.keys())
    all_tids.update(best_available_times.keys())
    if model_estimates:
        all_tids.update(model_estimates.keys())

    rows = []
    for tid in sorted(all_tids, key=str):
        comp_vals = comp_times.get(tid, [])
        est_vals = est_times.get(tid, [])
        cens_vals = censored_times.get(tid, [])
        bat = best_available_times.get(tid)

        row = {
            "task_id": tid,
            "task_family": families.get(tid, ""),
            "completion_minutes": (
                float(np.exp(np.mean(np.log(comp_vals)))) if comp_vals else None
            ),
            "n_completions": len(comp_vals),
            "estimate_minutes": (
                float(np.exp(np.mean(np.log(est_vals)))) if est_vals else None
            ),
            "n_estimates": len(est_vals),
            "firstblood_minutes": first_blood_minutes.get(tid),
            "model_estimate_minutes": (
                model_estimates.get(tid) if model_estimates else None
            ),
            "censored_lower_minutes": (
                float(np.exp(np.mean(np.log(cens_vals)))) if cens_vals else None
            ),
            "best_available_minutes": bat[0] if bat else None,
            "best_available_source": bat[1] if bat else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["task_id"] = df["task_id"].astype(str)
    return df


def headline_task_set(
    task_difficulties: pd.DataFrame,
    model_runs: pd.DataFrame | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Return (headline_task_ids, task_family_lookup) from task_difficulties.

    Headline tasks are those with a best_available_minutes value AND
    (if model_runs is provided) that also appear in model evaluation data.
    This replaces the old pattern of reading runs_human.parquet just to
    get the task ID set and family lookup.
    """
    headline = task_difficulties.dropna(subset=["best_available_minutes"])
    if model_runs is not None:
        eval_tasks = set(model_runs["task_id"].astype(str))
        headline = headline[headline["task_id"].astype(str).isin(eval_tasks)]
    task_ids = set(headline["task_id"].astype(str))
    families = headline.set_index("task_id")["task_family"].to_dict()
    return task_ids, families


def assemble_runs(
    model_runs: pd.DataFrame,
    task_difficulties: pd.DataFrame,
    difficulty_col: str = "best_available_minutes",
) -> pd.DataFrame:
    """Join model runs with a specific difficulty source for IRT analysis.

    Picks the named difficulty column from task_difficulties, joins to
    model_runs on task_id, filters to tasks where that column is populated,
    renames it to ``human_minutes`` for METR compatibility, computes
    ``log2_human_minutes`` and per-model weights.

    This is the ONLY place where ``human_minutes`` gets set. Every stage
    that needs a difficulty axis calls this function and specifies which
    source to use. No ambiguity.
    """
    from .corrections import compute_weights

    cols = ["task_id", difficulty_col]
    if "task_family" in task_difficulties.columns:
        cols.append("task_family")

    merged = model_runs.merge(
        task_difficulties[cols].dropna(subset=[difficulty_col]),
        on="task_id",
        suffixes=("", "_diff"),
        validate="many_to_one",
    )
    # Use task_family from difficulties if model_runs doesn't have it
    if "task_family_diff" in merged.columns:
        merged["task_family"] = merged["task_family"].fillna(merged["task_family_diff"])
        merged.drop(columns=["task_family_diff"], inplace=True)

    merged = merged.rename(columns={difficulty_col: "human_minutes"})
    merged["log2_human_minutes"] = np.log2(merged["human_minutes"])

    # Compute weights per model (must be done after filtering)
    dfs = []
    for alias, group in merged.groupby("alias"):
        dfs.append(compute_weights(group))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)

    return merged
