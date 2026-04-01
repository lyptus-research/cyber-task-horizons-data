# Human Study Data Dictionary

This directory contains data from the human expert study described in
Section 5 of the paper. The CSVs are the recommended entry point. The
raw `human_snapshot.json` is provided for anyone who needs the original
API snapshot format.

## completions.csv

One row per expert task attempt. Includes successful completions,
failed attempts, and censored sessions (where the expert gave up).

| Column | Type | Description |
|--------|------|-------------|
| `expert_id` | string | Anonymized expert label (expert_01 through expert_10) |
| `task_id` | string | Benchmark task identifier |
| `benchmark` | string | Parent benchmark (cybench, cvebench, cybashbench, nl2bash, intercode-ctf, nyuctf, cybergym) |
| `passed` | boolean | Whether the expert solved the task |
| `censored` | boolean | Right-censored observation: expert gave up before solving. The elapsed time is a lower bound on the true completion time. |
| `elapsed_minutes` | float | Wall-clock time from task start to submission, in minutes. Timing corrections applied where needed (see `analysis/lib/corrections.py`). |
| `submitted_at` | datetime | ISO 8601 timestamp of submission |
| `skip_reason` | string | Only for censored rows: `too_difficult` or `time_constraint` |

**Notes:**
- Answer text has been redacted for anonymity.
- `elapsed_minutes` excludes Docker image download time (timer starts
  after container setup).
- Timing corrections adjust for sessions where the server timer was
  unreliable (e.g., session left open overnight). See
  `analysis/lib/corrections.py` for the correction registry.

## estimations.csv

One row per expert time estimate. Experts reviewed each task and its
reference solution, then estimated how long a cold-start practitioner
would take to solve it.

| Column | Type | Description |
|--------|------|-------------|
| `expert_id` | string | Anonymized expert label |
| `task_id` | string | Benchmark task identifier |
| `benchmark` | string | Parent benchmark |
| `estimated_minutes` | float | Expert's point estimate of task difficulty, in minutes |
| `confidence` | string | Self-reported confidence: `high` (~2x), `medium` (2-3x), `low` (5x+) |
| `notes` | string | Optional reasoning or caveats from the expert |
| `review_minutes` | float | Time the expert spent reviewing the task and solution |
| `submitted_at` | datetime | ISO 8601 timestamp of submission |

**Notes:**
- Estimates are solution-visible: experts see the reference solution
  before estimating.
- `estimated_minutes` is the expert's judgment of how long a skilled
  practitioner would take from scratch, including discovery time and
  dead ends, not just execution of the known solution.
- Confidence levels indicate self-reported uncertainty about the
  estimate, not task difficulty.

## expert_survey.csv

Post-study survey responses from all participants. Contains
self-reported years of experience (hobbyist, professional, offensive
security) and free-text feedback on task representativeness and
environment quality. Email addresses and timestamps have been removed.

## human_snapshot.json

Raw API snapshot with the original nested structure. Contains the same
data as the CSVs above, plus `sample_set_tasks` (the task assignment
list). Provided for compatibility with the DVC pipeline, which consumes
this format directly.

## Quick Start

```python
import pandas as pd

completions = pd.read_csv("data/human/completions.csv")
estimations = pd.read_csv("data/human/estimations.csv")

# Successful completions only
passes = completions[completions["passed"] & ~completions["censored"]]

# Per-task best available human time (completion if available, else estimate)
comp_times = passes.groupby("task_id")["elapsed_minutes"].median()
est_times = estimations.groupby("task_id")["estimated_minutes"].median()
human_minutes = comp_times.combine_first(est_times)
```

## Source Hierarchy

When multiple timing sources exist for a task, the analysis pipeline
uses this priority order:

1. **Expert completions** (actual solve times)
2. **CTF first-blood times** (from `data/tasks/cybench/`)
3. **Expert estimates** (solution-visible)

This hierarchy is implemented in `analysis/lib/data.py`. The CSVs here
provide the raw expert data; first-blood times are in the task metadata.
