# Human Expert Evaluation Logs

90 anonymized [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evaluation
logs from completed expert sessions. Each file records one expert attempting
one task, with terminal transcripts of the full session.

## What are .eval files?

Inspect AI is the UK AI Safety Institute's open-source framework for AI
evaluation. It records each evaluation run in an `.eval` file, a
gzip-compressed ZIP archive containing JSON files with the full execution
trace. See the
[Inspect AI docs](https://inspect.ai-safety-institute.org.uk/log-viewer.html)
for the format specification.

In this study, both model and human expert sessions use Inspect AI. Model
evaluations record the agent's tool calls and reasoning. Human evaluations
record the expert's terminal session via the `human_cli` solver, which uses
the Linux `script` command to capture raw terminal I/O.

The `.eval` format is the same for both, but the contents differ:

| Field | Model evals | Human evals |
|-------|------------|-------------|
| `messages` | Multi-turn agent conversation | Single message (task prompt) |
| `store['HumanAgentState:logs']` | Not present | Raw terminal recordings |
| `store['HumanAgentState:answer']` | Not present | Expert's submitted answer |
| `scores` | Scorer results | May be empty (DB is canonical) |
| `model_usage` | Token counts | Empty (`none/none` model) |

## File naming

Files are named `{session_label}_{timestamp}.eval.gz`, where the session label
is an anonymized identifier (e.g., `session_023`) that matches the session IDs
in `completions.csv`. Some sessions have `_partial` suffixes, indicating the
CLI was interrupted before clean shutdown.

## Directory structure

Files are organized by benchmark and task:

```
eval_logs/
  cybench/
    back_to_the_past/
      session_004_20260129_171037.eval.gz
      session_332_20260302_165544.eval.gz
  cvebench/
    CVE-2024-2771/
      session_462_20260312_151645.eval.gz
  cybergym/
    arvo:34299/
      session_023_20260113_225539.eval.gz
  ...
```

Multiple files per task indicate different experts attempting the same task.

## How to open an .eval file

### Quick look at structure

```bash
# Decompress and list contents
gunzip -c session_023_20260113_225539.eval.gz > eval.zip
unzip -l eval.zip
```

Typical contents:
```
header.json                    # Evaluation metadata, timestamps
samples/{task}_epoch_1.json    # Full execution trace + terminal logs
summaries.json                 # Aggregated results
_journal/start.json            # Execution start state
```

### Extract and explore in Python

```python
import gzip, zipfile, io, json

path = "data/human/eval_logs/cybergym/arvo:34299/session_023_20260113_225539.eval.gz"

with gzip.open(path, "rb") as gz:
    zip_data = gz.read()

with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
    # Read the sample file (contains terminal logs)
    sample_name = [n for n in zf.namelist() if n.startswith("samples/")][0]
    sample = json.loads(zf.read(sample_name))

# Task metadata
print(f"Task: {sample['id']}")
print(f"Duration: {sample.get('total_time', 0) / 60:.1f} minutes")

# Expert's submitted answer
store = sample.get("store", {})
print(f"Answer: {store.get('HumanAgentState:answer', '')}")

# Terminal logs
logs = store.get("HumanAgentState:logs", {})
print(f"Log files: {list(logs.keys())}")
```

### Terminal log files

The `HumanAgentState:logs` dict contains raw terminal recordings serialized
as text. Keys follow the naming pattern `{user}_{date}_{time}.{type}`:

| Extension | Content |
|-----------|---------|
| `.input` | Raw keystrokes (what the expert typed) |
| `.output` | Terminal output (commands + results together) |
| `.timing` | Timing data for each I/O event (for `scriptreplay`) |

The `.output` file is the most useful. It shows commands and their results
together, though it contains ANSI escape sequences and control characters
that need cleaning.

### Cleaning terminal output

```python
import re

def clean_terminal(raw):
    """Remove ANSI escapes and control chars from terminal output."""
    cleaned = re.sub(r"\x1b\[[0-9;?]*[@-~]", "", raw)
    cleaned = re.sub(r"\x1b\][^\x07]*\x07", "", cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned

# Get the .output file
output_keys = [k for k in logs if ".output" in k]
if output_keys:
    raw = logs[output_keys[0]]
    print(clean_terminal(raw)[:5000])
```

### Extracting expert commands

```python
def extract_commands(raw_input):
    """Extract commands from raw keystroke recording."""
    cleaned = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw_input)
    cleaned = re.sub(r"\x1b\[20[01]~", "", cleaned)  # bracketed paste
    while "\x7f" in cleaned:
        cleaned = re.sub(r".\x7f", "", cleaned)  # backspace
    return [line.strip() for line in cleaned.split("\r") if line.strip()]

input_keys = [k for k in logs if ".input" in k]
if input_keys:
    commands = extract_commands(logs[input_keys[0]])
    for cmd in commands[:20]:
        print(f"  $ {cmd}")
```

## Key fields in sample JSON

| Field | Description |
|-------|-------------|
| `id` | Task ID (matches `task_id` in completions.csv) |
| `input` | Task prompt shown to the expert |
| `messages[0]` | Same task prompt as a message object |
| `store['HumanAgentState:logs']` | Dict of terminal recording files |
| `store['HumanAgentState:answer']` | What the expert submitted |
| `store['HumanAgentState:accumulated_time']` | Seconds the task clock was running |
| `store['HumanAgentState:scorings']` | List of intermediate scoring attempts |
| `store['HumanAgentState:notes']` | Expert's in-session notes (if any) |
| `total_time` | Total wall-clock seconds |
| `sandbox.config` | Path to Docker Compose config used |

## Anonymization

These files have been anonymized:

- Session UUIDs in filenames replaced with opaque labels (`session_NNN`)
- Local machine paths (`/Users/<name>/`) replaced with `/Users/expert/`
- Infrastructure IPs replaced with `REDACTED_HOST`
- Temp directory paths normalized

Benchmark task content (WordPress HTML, CTF challenge data, etc.) is
preserved as-is. This is public benchmark material, not expert PII.

## Coverage

90 of 174 completed sessions have .eval files. The 84 missing are almost
entirely CyBashBench (sub-minute terminal command tasks where the CLI
does not produce eval logs). All substantive multi-minute completions across
the other 6 benchmarks have eval files.

## Cross-referencing with completions.csv

The session label in the filename corresponds to entries in `completions.csv`.
However, `completions.csv` uses anonymized session IDs from the snapshot
(which are internal to the JSON), not the filename labels. To link them,
match on `(expert_id, task_id)` rather than session label:

```python
import pandas as pd

completions = pd.read_csv("data/human/completions.csv")
# Find the completion row for a specific eval file
task_id = "arvo:34299"  # from the directory name
task_completions = completions[completions["task_id"] == task_id]
print(task_completions[["expert_id", "passed", "elapsed_minutes"]])
```
