# Offensive Cyber Task Horizons: Data and Analysis

Reproducibility artifact for [Offensive Cyber Task Horizons: Measuring the
Rate of Growth in AI Offensive Cybersecurity
Capability](https://lyptusresearch.org/2026/03/23/offensive-cyber-time-horizons.html)
by Jack Payne, Jeremy Miller, and Sean Peters (Lyptus Research, 2026).

This repository contains all data and analysis code needed to reproduce the
paper's figures, tables, and statistics from raw evaluation logs.

The dataset is also available on
[HuggingFace](https://huggingface.co/datasets/lyptus-research/cyber-task-horizons).

## Quick Start

```bash
git clone https://github.com/lyptus-research/cyber-task-horizons-data.git
cd cyber-task-horizons-data
uv sync                         # Install Python dependencies
cd analysis/figures
uv run dvc repro                # Reproduce all figures
```

The pipeline runs from pre-built intermediates, so the raw `.eval` files
are not needed to reproduce figures. To download them (for custom analysis
of model trajectories or token usage):

```bash
git lfs pull                    # Downloads ~18 GB of .eval files
```

Figures are written to `analysis/figures/out/`. JSON data files used by
the paper's interactive charts are written alongside the PNGs. These are
the source for the `site.data.*` template variables in the paper.

### macOS Note

Several stages require Cairo for SVG rendering. On macOS with Homebrew:

```bash
brew install cairo
```

## What Can You Do With This Data?

**Reproduce the paper.** `dvc repro` regenerates every figure, table, and
statistic from raw data. Change a hyperparameter in `params.yaml` and re-run
to see how it affects the results.

**Fit your own IRT curves.** The pre-built `model_runs.parquet` and
`task_difficulties.parquet` contain everything needed for custom IRT analysis.
Swap in different difficulty sources, weighting schemes, or regularisation
strengths.

**Analyze model behaviour on cyber tasks.** The raw `.eval` files in
`data/eval_logs/` contain complete agent trajectories (tool calls, reasoning,
outputs, scores). You can study how different models approach the same task,
where they get stuck, and what strategies succeed.

**Study how experts solve security tasks.** The 90 human `.eval` files in
`data/human/eval_logs/` contain full terminal transcripts of expert sessions.
Cross-reference with `completions.csv` for timing and scores.

**Explore token budget scaling.** Per-run token counts are in the `.eval`
files, and the 10M-token re-run data shows how success rates scale with
compute. The paper finds frontier models productively use far more tokens
than typical evaluation budgets allow.

**Compare human and model difficulty estimates.** `estimations.csv` has
expert time estimates. `data/tasks/<benchmark>/*_model_estimates.jsonl`
has frontier-model estimates for the same tasks. The paper's cross-source
analysis compares where these agree and diverge.

**Add new models.** The IRT pipeline is model-agnostic. If you run a new model
on the same benchmarks using Inspect AI, you can add the `.eval` files and
re-run the pipeline to see where it falls on the trendline.

## What This Study Measures

The study applies METR's time-horizon methodology to offensive cybersecurity.
Tasks are annotated with human expert completion times. Models are evaluated
on each task. 2-parameter IRT logistic curves are fitted to the
success-vs-difficulty data, and the time horizon (the task duration at which
a model succeeds at a given rate) is read off each curve. Plotting time
horizons against model release date gives a doubling time.

Seven benchmarks span terminal commands through multi-hour exploit development:

| Benchmark | Tasks (evaluated) | Tasks (in JSONL) | Difficulty Range | Type |
|-----------|-------------------|-------------------|-----------------|------|
| CyBashBench | 200 | 200 | 1s - 30s | Command generation |
| NL2Bash | 136 | 136 | 4s - 4min | Command generation |
| InterCode-CTF | 99 | 99 | 10s - 10min | Beginner CTF |
| NYUCTF | 50 | 50 | 2min - 6h | University CTF |
| CyBench | 40 | 40 | 2min - 25h | Professional CTF |
| CVEBench | 40 | 40 | 15min - 8h | Real CVE reproduction |
| CyberGym | 122 | 322 | 30min - 8h | Memory-safety PoC generation |

CyberGym's JSONL includes 322 tasks (the full benchmark). 122 were selected
for model evaluation based on construct validity and difficulty coverage.

Models evaluated span 2019 through early 2026, including GPT-4, Claude 3.5
Sonnet, o1, o3, Gemini 2.5 Pro, Claude Opus 4/4.6, GPT-5.x Codex, and
open-source models (GLM-5, DeepSeek V3.1). All evaluations use a fixed 2M
token budget with Inspect AI's ReAct agent scaffold.

## Repository Structure

```
data/
    eval_logs/          Raw Inspect AI .eval files (Git LFS, ~18 GB)
    human/              Anonymized expert completion and estimation data
    tasks/              Per-benchmark task definitions and timing metadata
        <benchmark>/    Task JSONL, human runs, model estimates per benchmark
        cvebench/
            solutions/  CVEBench solution write-ups (CC-BY-4.0)
    models/             Model release dates, aliases, and provider configs
    methodology/        Evaluation configs and agent scaffold source
        research_agent.py  Agent scaffold: on-continue prompts, tool config
        README.md       System prompts, prompt softening, agent parameters
    june_2025/          Legacy June 2025 study data (METR format)

analysis/
    config.py           Central path configuration
    lib/                Analysis library
        results.py      Eval-set ID registry and .eval file loading
        eval_sets.py    Campaign definitions (which eval-sets per model)
        data.py         Human timing data loading and merging
        irt.py          IRT curve fitting (wraps METR's horizon package)
        trendline.py    Doubling time computation and trendline fitting
        corrections.py  Outlier exclusions, timing corrections, session exclusions
        estimates.py    Expert and model time estimate loading
        constants.py    Benchmark metadata and model release dates
    figures/            DVC pipeline
        dvc.yaml        Pipeline definition (20+ stages)
        params.yaml     All hyperparameters (bootstrap, IRT, sensitivity)
        data/           Pre-built intermediate artifacts (parquets, pickles)
        stages/         Python stage scripts (one per pipeline stage)
        out/            Generated figures and interactive chart JSONs
    tests/              Analysis code tests

references.bib          BibTeX bibliography for all paper citations
```

## Data Description

Both model and human evaluation data are stored as
[Inspect AI](https://inspect.ai-safety-institute.org.uk/) `.eval` files
(gzip-compressed ZIP archives containing JSON execution traces). See the
[Inspect AI documentation](https://inspect.ai-safety-institute.org.uk/log-viewer.html)
for the format specification. `data/human/eval_logs/README.md` covers the
human terminal transcripts specifically.

### Model Evaluation Logs

`data/eval_logs/` contains raw `.eval` files for all model campaigns (~18 GB
via Git LFS). Each subdirectory is named by its eval-set ID (e.g.,
`eval-set-abc123def456`) and contains one or more `.eval` files. Each file
records one model attempting one task, with the full agent trajectory
(system prompt, tool calls, outputs, token usage, and score).

The mapping from eval-set IDs to models and benchmarks is in
`analysis/lib/eval_sets.py`.

### Human Study Data

`data/human/` contains all expert data. Start with `DATA_DICTIONARY.md` for
column definitions and code examples.

| File | Description |
|------|-------------|
| `completions.csv` | 174 expert task attempts with timing, scores, benchmark |
| `estimations.csv` | 310 solution-visible time estimates with confidence |
| `expert_survey.csv` | Post-study survey (experience levels, qualitative feedback) |
| `eval_logs/` | 90 terminal transcripts as anonymized `.eval` files |
| `DATA_DICTIONARY.md` | Column definitions and quick-start code |
| `human_snapshot.json` | Raw API snapshot (same data, nested JSON format) |

Expert identifiers are anonymized (expert_01 through expert_10). Answer text
is redacted in the CSVs. Terminal transcripts in the `.eval` files show the
expert's full working process.

### Task Metadata

`data/tasks/<benchmark>/` contains per-benchmark task definitions (full JSONL
with descriptions, flags, and metadata), frontier-model time estimates
(`*_model_estimates.jsonl`), and human completion data
(`*_human_runs.jsonl`, available for CyBench, CVEBench, InterCode-CTF,
NL2Bash, and NYUCTF). CyBashBench and CyberGym do not have separate
human_runs files. `data/tasks/cvebench/solutions/` contains original
CVEBench solution write-ups.

`data/tasks/task_metadata.csv` is a review artifact from the task selection
process. It covers all benchmarks except CyBashBench. CyberGym has 1507
entries (the full upstream set, not the 122 evaluated).

### Evaluation Methodology

`data/methodology/` contains evaluation configurations extracted from the
pipeline source code:

- **research_agent.py**: The agent scaffold source code, including on-continue
  prompts and the empty-cascade termination logic described in the paper.
- **README.md**: Evaluation parameters, extracted research system prompts
  (applied to GPT-5.x Codex models only), and prompt softening text
  replacements.

### Interactive Chart Data

Running `dvc repro` generates both PNG figures and JSON data files in
`analysis/figures/out/charts/`. The JSON files contain the structured data
behind each figure and are the source for the `site.data.*` template
variables and interactive Plotly charts on the paper's web version.

## Analysis Pipeline

The pipeline uses [DVC](https://dvc.org/) for reproducible figure generation
and depends on [METR's eval-analysis-public](https://github.com/METR/eval-analysis-public)
(pinned to commit `52cb829`) for IRT logistic regression and trendline
computation.

### Frozen vs unfrozen stages

Two early stages are frozen because they depend on infrastructure not
included in this repository:

- **snapshot_human_data** (stage 0): Originally pulled live data from the
  study API. The pre-built snapshot is shipped in `analysis/figures/data/`.
- **prepare_runs** (stage 1): Loads raw .eval files and builds pipeline
  tables. Pre-built parquets are shipped. Can be unfrozen if you want to
  rebuild from raw .eval files.

All downstream stages (bootstrap, IRT fitting, sensitivity analysis, figure
generation, paper statistics) are unfrozen and will run during `dvc repro`.

### Running tests

```bash
cd analysis
uv run pytest tests/
```

### Generated figures

`dvc repro` produces the following in `analysis/figures/out/`:

**Headline results**

| Figure | Description |
|--------|-------------|
| `trendline_p50_runs_human_2M_combined.png` | P50 time horizon trendline (linear + log scale) |
| `irt_grid_runs_human.png` | Per-model IRT logistic fits with human-derived difficulty |
| `source_coverage.png` | Human timing data sources across difficulty spectrum |

**Sensitivity analysis**

| Figure | Description |
|--------|-------------|
| `sensitivity_dual.png` | Doubling time sensitivity to source treatments (2019+ and 2024+) |
| `sensitivity_p50.png` | Frontier P50 sensitivity to treatments |
| `sensitivity_multiverse_boxplot.png` | Full multiverse analysis distributions |
| `completions_only_comparison.png` | IRT fits under completions-only vs headline |
| `regularisation_comparison.png` | IRT fit at strong vs minimal regularisation |
| `bootstrap_models_example.png` | Headline vs egregious bootstrap sample |
| `trendline_alternatives.png` | Exponential, linear, hyperbolic, logistic trendline fits |

**Token budget analysis**

| Figure | Description |
|--------|-------------|
| `token_budget_sensitivity.png` | P50 vs token budget + 1M-to-2M gain per model |
| `token_budget_extended_10m.png` | Extended to 10M tokens (GPT-5.3 Codex re-runs) |
| `token_subset_analysis.png` | Accuracy and cost per success by task difficulty subset |

**Open-source models**

| Figure | Description |
|--------|-------------|
| `os_main_trendline.png` | Open-source models projected onto closed-source trendline |

**Human study validation**

| Figure | Description |
|--------|-------------|
| `icc_agreement.png` | Rater agreement scatter plots (estimations and completions) |
| `icc_gauge.png` | ICC point estimates with confidence intervals |
| `cross_source_grid.png` | Cross-source comparisons (estimates vs completions vs first-blood) |
| `expert_effort.png` | Estimation session duration by task difficulty |

Additional variants (model-estimated difficulty, P80, 1M budget, etc.)
are also produced. See `analysis/figures/dvc.yaml` for the complete list.

## Citation

```bibtex
@article{payne2026cybertaskhorizons,
    title={Offensive Cyber Task Horizons: Measuring the Rate of Growth
           in AI Offensive Cybersecurity Capability},
    author={Payne, Jack and Miller, Jeremy and Peters, Sean},
    year={2026},
    url={https://lyptusresearch.org/2026/03/23/offensive-cyber-time-horizons.html}
}
```

## License

- **Data** (everything under `data/`): CC-BY-4.0
- **Code** (everything under `analysis/`): MIT
