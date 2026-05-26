"""Token budget sensitivity by task subset.

Two panels:
  Left: Accuracy vs token budget for different task subsets
  Right: Cost per success vs token budget for different task subsets

Defaults to GPT-5.3 Codex as the reference model. GPT-5.5 can be rendered with
the local 50M retry eval logs overlaid above the 2M study budget.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import sys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    load_params,
    save_chart_json,
    save_png,
)
from lib.lyptus_style import apply_style, COLORS  # noqa: E402

apply_style()

PRICE_PER_TOKEN = 4.5 / 1e6  # GPT-5.1 Codex tier
DEFAULT_MODEL = "GPT-5.3 Codex"
GPT55_MODEL = "GPT-5.5"

GPT55_50M_EVAL_SETS = [
    "eval-set-d8ms2q1en0ctj1k5",  # cybergym pilot
    "eval-set-cxv11jk9kwbqsyp1",  # cvebench pilot
    "eval-set-anmfra3w3os3wzo5",  # nyuctf pilot
    "eval-set-uoax7forfc0qc0ri",  # nyuctf residuals
    "eval-set-fmrn9ettelvc108q",  # cybergym residual batch 1 / retries
    "eval-set-rd31u48n0r2wrfpm",  # cybergym residual batch 2
    "eval-set-1limh4blc9gwto7k",  # cybergym residual batch 3
    "eval-set-lsqvg4vhwlyalqqj",  # cybergym residual batch 4
    "eval-set-morryc5nsrgn20f0",  # cybergym residual batch 5
]

BUDGETS = [
    50_000,
    100_000,
    200_000,
    500_000,
    750_000,
    1_000_000,
    1_250_000,
    1_500_000,
    1_750_000,
    2_000_000,
    3_000_000,
    4_000_000,
    5_000_000,
    6_000_000,
    7_000_000,
    8_000_000,
    9_000_000,
    10_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
    50_000_000,
]

SUBSETS = [
    {"key": "all", "label": "All tasks", "filter": lambda df: df},
    {
        "key": "cybergym",
        "label": "CyberGym",
        "filter": lambda df: df[df["task_family"] == "cybergym"],
    },
    {
        "key": "hard_30m",
        "label": ">30m tasks",
        "filter": lambda df: df[df["human_minutes"] > 30],
    },
    {
        "key": "hard_2h",
        "label": ">2h tasks",
        "filter": lambda df: df[df["human_minutes"] > 120],
    },
]


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute accuracy and cost at each budget for each subset."""
    from lib.data import assemble_runs

    model = args.model
    mr = pd.read_parquet(
        Path(args.model_runs)
        if Path(args.model_runs).is_absolute()
        else _NOTEBOOKS_DIR / args.model_runs
    )
    td = pd.read_parquet(
        Path(args.task_difficulties)
        if Path(args.task_difficulties).is_absolute()
        else _NOTEBOOKS_DIR / args.task_difficulties
    )
    runs_df = assemble_runs(mr, td, args.difficulty_col)
    model_runs = runs_df[runs_df["alias"] == model].copy()

    if len(model_runs) == 0:
        return {
            "chart_type": "tokenSubsetAnalysis",
            "version": 1,
            "data": {"model": model, "subsets": [], "study_budget_m": 2.0},
            "options": {"title": f"{model} token budget analysis by task subset"},
        }

    # Load extended re-run data if available. GPT-5.3 uses the historical 10M
    # pickle; GPT-5.5 uses local 50M Inspect eval logs from the residual push.
    extended = {}
    if args.ten_m_cache:
        import pickle

        cache_path = (
            Path(args.ten_m_cache)
            if Path(args.ten_m_cache).is_absolute()
            else _NOTEBOOKS_DIR / args.ten_m_cache
        )
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                ten_m_data = pickle.load(f)
            for s in ten_m_data:
                extended[s["task_id"]] = {
                    "score": s["score"],
                    "tokens": s["total_tokens"],
                }
            print(f"Loaded {len(extended)} 10M re-run samples")

    if args.gpt55_50m:
        extended.update(_load_gpt55_50m_results(args.gpt55_50m_cache))
        print(f"Loaded {len(extended)} GPT-5.5 50M re-run samples")

    # Compute accuracy and cost-per-success at each budget for each subset
    chart_subsets = []

    for subset_def in SUBSETS:
        subset_runs = subset_def["filter"](model_runs)
        n_tasks = subset_runs["task_id"].nunique()

        budget_points = []
        for b in BUDGETS:
            # For budgets <= 2M, use the standard approach
            if b <= 2_000_000:
                successes = int(
                    (
                        (subset_runs["score_binarized"] == 1)
                        & (subset_runs["total_tokens"] <= b)
                    ).sum()
                )
                tokens_used = subset_runs["total_tokens"].clip(upper=b)
            else:
                # For extended budgets, incorporate 10M re-run data
                successes = 0
                tokens_used_list = []
                for _, row in subset_runs.iterrows():
                    tok = min(row["total_tokens"], b)
                    tokens_used_list.append(tok)
                    if row["score_binarized"] == 1 and row["total_tokens"] <= b:
                        successes += 1
                    elif row["score_binarized"] == 0 and row["task_id"] in extended:
                        rerun = extended[row["task_id"]]
                        if rerun["score"] > 0 and rerun["tokens"] <= b:
                            successes += 1
                            extra = min(rerun["tokens"], b) - row["total_tokens"]
                            if extra > 0:
                                tokens_used_list[-1] += extra
                tokens_used = pd.Series(tokens_used_list)

            total = len(subset_runs)
            accuracy = successes / total * 100 if total > 0 else 0
            total_cost = float((tokens_used * PRICE_PER_TOKEN).sum())
            cps = total_cost / successes if successes > 0 else float("inf")

            budget_points.append(
                {
                    "budget": b,
                    "budget_m": b / 1e6,
                    "budget_label": f"{b // 1000}K"
                    if b < 1_000_000
                    else f"{b / 1_000_000:.0f}M",
                    "accuracy": round(accuracy, 1),
                    "cost_per_success": round(cps, 2) if np.isfinite(cps) else None,
                    "successes": successes,
                    "total": total,
                }
            )

        chart_subsets.append(
            {
                "key": subset_def["key"],
                "label": f"{subset_def['label']} (n={n_tasks})",
                "n_tasks": n_tasks,
                "points": budget_points,
            }
        )

    return {
        "chart_type": "tokenSubsetAnalysis",
        "version": 1,
        "data": {
            "model": model,
            "subsets": chart_subsets,
            "study_budget_m": 2.0,
            "extended_budget_m": 50.0 if args.gpt55_50m else 10.0 if extended else None,
        },
        "options": {"title": f"{model} token budget analysis by task subset"},
    }


def _load_gpt55_50m_results(cache_path: str) -> dict[str, dict]:
    """Load latest GPT-5.5 50M retry result per task from canonical eval cache."""
    json_path = Path(cache_path)
    if not json_path.is_absolute():
        json_path = _NOTEBOOKS_DIR / json_path
    if json_path.exists():
        return json.loads(json_path.read_text())

    from inspect_ai.log import read_eval_log_samples

    cache = _NOTEBOOKS_DIR.parent / "data" / ".eval-cache"
    results = {}

    for eval_set_id in GPT55_50M_EVAL_SETS:
        eval_dir = cache / eval_set_id
        if not eval_dir.exists():
            continue

        files = sorted(f for f in eval_dir.glob("*.eval") if f.name[:4].isdigit())
        if not files:
            files = sorted(eval_dir.glob("*.eval"))

        for eval_file in files:
            for sample in read_eval_log_samples(
                str(eval_file), all_samples_required=False
            ):
                task_id = sample.id
                if task_id.startswith("CVE-") and task_id.endswith("-one_day"):
                    task_id = task_id[: -len("-one_day")]

                score = _sample_score(sample)
                tokens = _sample_total_tokens(sample)
                if tokens is None:
                    continue

                results[task_id] = {
                    "score": score,
                    "tokens": tokens,
                    "source": eval_set_id,
                }

    return results


def _sample_score(sample) -> float:
    if not sample.scores:
        return 0.0
    score = next(iter(sample.scores.values())).value
    if isinstance(score, str):
        return 1.0 if score == "C" else 0.0
    return float(score or 0.0)


def _sample_total_tokens(sample) -> int | None:
    if not sample.model_usage:
        return None
    usage = next(iter(sample.model_usage.values()))
    return int(usage.total_tokens)


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render token budget sensitivity plots from chart JSON data."""
    chart_subsets = chart_data["data"]["subsets"]
    model = chart_data["data"]["model"]
    study_budget_m = chart_data["data"].get("study_budget_m", 2.0)
    extended_budget_m = chart_data["data"].get("extended_budget_m")

    if not chart_subsets:
        print(f"No data for {model}, skipping render")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    subset_colors = [
        COLORS["teal_dark"],
        COLORS["coral"],
        COLORS["teal_light"],
        COLORS["gold"],
    ]
    subset_styles = ["-", "-", "-", "-"]

    for i, subset in enumerate(chart_subsets):
        budgets_m = [p["budget_m"] for p in subset["points"]]
        accuracies = [p["accuracy"] for p in subset["points"]]
        costs = [p["cost_per_success"] for p in subset["points"]]

        ax1.plot(
            budgets_m,
            accuracies,
            marker="o",
            color=subset_colors[i],
            linewidth=2,
            markersize=5,
            label=subset["label"],
            linestyle=subset_styles[i],
        )

        valid_costs = [(b, c) for b, c in zip(budgets_m, costs) if c is not None]
        if valid_costs:
            ax2.plot(
                [x[0] for x in valid_costs],
                [x[1] for x in valid_costs],
                marker="o",
                color=subset_colors[i],
                linewidth=2,
                markersize=5,
                label=subset["label"],
                linestyle=subset_styles[i],
            )

    # Study budget line
    for ax in [ax1, ax2]:
        ax.axvline(
            study_budget_m, color=COLORS["coral"], linestyle=":", alpha=0.5, linewidth=1
        )
        ax.text(
            study_budget_m,
            ax.get_ylim()[0],
            "Study budget",
            color=COLORS["coral"],
            fontsize=8,
            alpha=0.7,
            ha="right",
            va="bottom",
            rotation=90,
        )
        if extended_budget_m and extended_budget_m > study_budget_m:
            ax.axvline(
                extended_budget_m,
                color=COLORS["teal_dark"],
                linestyle=":",
                alpha=0.5,
                linewidth=1,
            )
            ax.text(
                extended_budget_m,
                ax.get_ylim()[0],
                "Extended budget",
                color=COLORS["teal_dark"],
                fontsize=8,
                alpha=0.7,
                ha="right",
                va="bottom",
                rotation=90,
            )

    ax1.set_xlabel("Token budget")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"{model} accuracy by task subset")
    ax1.set_xscale("log")
    xticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    xlabels = ["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M"]
    if extended_budget_m and extended_budget_m > 10.0:
        xticks.extend([20.0, 50.0])
        xlabels.extend(["20M", "50M"])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    ax2.set_xlabel("Token budget")
    ax2.set_ylabel("Cost per success ($)")
    ax2.set_title(f"{model} cost per success by task subset")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 50)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot token budget sensitivity by task subset")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--task-difficulties", required=True)
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ten-m-cache", default=None, help="Path to 10M re-run cache")
    parser.add_argument(
        "--gpt55-50m",
        action="store_true",
        help="Overlay the local GPT-5.5 50M retry eval logs above the 2M budget",
    )
    parser.add_argument(
        "--gpt55-50m-cache",
        default="../data/keep/gpt55_50m_reruns.json",
        help="JSON cache of GPT-5.5 50M retry results",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
