"""Model time estimate loading — stdlib-only, importable from scripts and notebooks.

This module exists separately from data.py because data.py imports numpy/pandas
at module level, making it unusable from lightweight CLI scripts.
"""

import json
from pathlib import Path

BENCHMARKS = [
    "cybashbench",
    "nl2bash",
    "intercode-ctf",
    "nyuctf",
    "cybench",
    "cvebench",
    "cybergym",
]

# Default project root: notebooks/ is at cyber-task-horizons/notebooks/
from analysis.config import REPO_ROOT

_DEFAULT_ROOT = REPO_ROOT


def load_model_time_estimates(project_root: Path | None = None) -> dict[str, float]:
    """Load model-estimated human completion times for all tasks.

    Returns dict mapping task_id -> estimated human_minutes, combining
    three sources per benchmark (later sources override earlier):

      1. **Bundled task labels** from ``data/keep/{bench}/{bench}_human_runs.jsonl``
         (``estimated_time_seconds`` field). AI-assisted time labels from the
         June 2025 study baked into benchmark data files.
      2. **Processed task metadata** from
         ``data/processed/{bench}/{bench}_tasks.jsonl`` (``human_minutes``
         field). Same provenance as (1) but with possible ID format
         differences (e.g. nl2bash integer vs string IDs).
      3. **LLM-generated estimates** from
         ``data/keep/{bench}/{bench}_model_estimates.jsonl``
         (``estimated_seconds`` field). Produced by ``hte estimate``.
         These replace placeholder values (e.g. CVEBench's flat 60m).

    This function provides the model-estimated difficulty axis used for
    task sampling, calibration regression (x-variable), and bucket
    coverage tracking. It does NOT return expert-derived times — those
    come from the production API via ``filter_to_experts()``.
    """
    root = project_root or _DEFAULT_ROOT
    estimates: dict[str, float] = {}

    for bench in BENCHMARKS:
        keep_path = root / "data" / "tasks" / bench
        proc_path = root / "data" / "tasks" / bench

        # Pattern 1: data/keep/{bench}/{bench}_human_runs.jsonl
        human_runs = keep_path / f"{bench}_human_runs.jsonl"
        if human_runs.exists():
            with open(human_runs) as f:
                for line in f:
                    row = json.loads(line)
                    task_id = row.get("task_id")
                    seconds = row.get("estimated_time_seconds")
                    if task_id and seconds:
                        estimates[task_id] = seconds / 60.0

        # Pattern 2: data/processed/{bench}/{bench}_tasks.jsonl
        tasks_file = proc_path / f"{bench}_tasks.jsonl"
        if not tasks_file.exists():
            tasks_file = proc_path / f"{bench.replace('-', '_')}_tasks.jsonl"

        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    row = json.loads(line)
                    task_id = row.get("task_id")
                    minutes = row.get("human_minutes")
                    if task_id and minutes:
                        estimates[task_id] = minutes

        # Pattern 3: data/keep/{bench}/{bench}_model_estimates.jsonl
        model_est = keep_path / f"{bench}_model_estimates.jsonl"
        if model_est.exists():
            with open(model_est) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    task_id = row.get("task_id")
                    seconds = row.get("estimated_seconds")
                    if task_id and seconds and seconds > 0:
                        estimates[task_id] = seconds / 60.0

    return estimates
