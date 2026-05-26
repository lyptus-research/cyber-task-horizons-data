"""Emit the 3 GPT-5.5 post chart JSONs for the website.

Reuses compute() from the canonical pipeline stages so the interactive Plotly
charts render from the same underlying numbers as the static PNGs:

  gpt55_irt_2m.json            <- plot_os_trendline.compute() + GPT-5.5 @ 2M overlay
  gpt55_irt_grid.json          <- plot_gpt55_multi_budget_irt.compute()
  token_budget_extended_50m.json <- plot_token_budget.compute()'s extended block

Run from notebooks/:
  uv run python -m figures.stages.emit_gpt55_chart_jsons
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import load_params, save_chart_json  # noqa: E402
from figures.stages import plot_gpt55_multi_budget_irt as irt_grid  # noqa: E402
from figures.stages import plot_os_trendline as os_trendline  # noqa: E402
from figures.stages import plot_token_budget as token_budget  # noqa: E402


# GPT-5.5 @ 2M point used by the static trendline (sketch_os_trendline_metr.py).
GPT55_2M = {
    "p50_min": 328.0,
    "ci_lo": 207.0,
    "ci_hi": 627.0,
    "p50_label": "5.5h",
}

UNRELIABLE_MIN = 12 * 60  # 12h principled cutoff
Y_CAP_MIN = UNRELIABLE_MIN * 1.5625


def emit_trendline(out_dir: Path, params: dict) -> Path:
    args = SimpleNamespace(
        summaries=str(_NOTEBOOKS_DIR / "figures/data/model_summaries_human_2M.parquet"),
        bootstrap=str(_NOTEBOOKS_DIR / "figures/data/bootstrap/runs_human_2M.parquet"),
        x_lim_start=None,
        x_lim_end=None,
    )
    chart_data = os_trendline.compute(args, params)
    chart_data["chart_type"] = "gpt55Trendline"
    chart_data["data"]["gpt55_2m"] = GPT55_2M
    chart_data["options"] = {
        "title": "Time horizon of cybersecurity tasks<br>different LLMs can complete 50% of the time",
        "unreliable_min": UNRELIABLE_MIN,
        "y_cap_min": Y_CAP_MIN,
    }
    return save_chart_json(chart_data, str(out_dir / "gpt55_irt_2m.png"))


def emit_irt_grid(out_dir: Path, params: dict) -> tuple[Path, Path]:
    args = SimpleNamespace(
        model_runs=str(_NOTEBOOKS_DIR / "figures/data/model_runs.parquet"),
        task_difficulties=str(_NOTEBOOKS_DIR / "figures/data/task_difficulties.parquet"),
        difficulty_col="best_available_minutes",
    )
    panels = irt_grid.compute(args, params)
    options = {
        "title": "Per-bin success rate vs human-time difficulty",
        "unreliable_log2": float(irt_grid.UNRELIABLE_LOG2),
        "tick_log2": list(irt_grid._TICK_LOG2),
        "tick_labels": list(irt_grid._TICK_LABELS),
    }
    by_budget = {
        "chart_type": "gpt55IrtGrid",
        "version": 1,
        "data": panels,
        "options": options,
    }
    by_model = {
        "chart_type": "gpt55IrtGridByModel",
        "version": 1,
        "data": panels,
        "options": options,
    }
    p1 = save_chart_json(by_budget, str(out_dir / "gpt55_irt_grid.png"))
    p2 = save_chart_json(by_model, str(out_dir / "gpt55_irt_grid_by_model.png"))
    return p1, p2


def emit_token_budget(out_dir: Path, params: dict) -> Path:
    args = SimpleNamespace(
        model_runs=str(_NOTEBOOKS_DIR / "figures/data/model_runs.parquet"),
        task_difficulties=str(_NOTEBOOKS_DIR / "figures/data/task_difficulties.parquet"),
        difficulty_col="best_available_minutes",
        summaries=str(_NOTEBOOKS_DIR / "figures/data/model_summaries_human_2M.parquet"),
        ten_m_samples=str(_NOTEBOOKS_DIR.parent / "data/keep/10m_samples.pkl"),
        os_cache=str(_NOTEBOOKS_DIR.parent / "data/keep/os_data_cache.pkl"),
        gpt55_50m_cache=str(_NOTEBOOKS_DIR / "figures/data/gpt55_50m_reruns.json"),
        output=str(out_dir / "_unused.png"),
        output_extended=str(out_dir / "token_budget_extended_50m.png"),
    )
    bundle = token_budget.compute(args, params)
    extended = bundle["extended"]
    extended["chart_type"] = "gpt55TokenBudget50m"
    extended.setdefault("options", {})["title"] = (
        "P50 time horizon vs token budget — GPT-5.5 extends past previous frontier"
    )
    return save_chart_json(extended, str(out_dir / "token_budget_extended_50m.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="figures/params.yaml")
    parser.add_argument(
        "--out-dir",
        default=str(_NOTEBOOKS_DIR / "figures/out"),
        help="Directory where the .png anchor lives; JSON lands in <out_dir>/charts/.",
    )
    args = parser.parse_args()
    params = load_params(args.params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_trendline(out_dir, params)
    emit_irt_grid(out_dir, params)
    emit_token_budget(out_dir, params)


if __name__ == "__main__":
    main()
