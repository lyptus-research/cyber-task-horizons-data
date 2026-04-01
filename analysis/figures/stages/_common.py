"""Shared helpers for DVC figure stages."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

# notebooks/ must be importable for lib.*
_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))
_REPO_ROOT = str(_NOTEBOOKS_DIR.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.lyptus_style import apply_style as _apply_style  # noqa: E402

# Apply once on import
_apply_style()


def load_params(path: str = "figures/params.yaml") -> dict:
    p = _NOTEBOOKS_DIR / path
    with open(p) as f:
        return yaml.safe_load(f)


def load_runs(path: str) -> pd.DataFrame:
    return pd.read_parquet(
        Path(path) if Path(path).is_absolute() else _NOTEBOOKS_DIR / path
    )


def load_json(path: str) -> dict:
    p = Path(path) if Path(path).is_absolute() else _NOTEBOOKS_DIR / path
    with open(p) as f:
        return json.load(f)


def rebuild_campaign_data(runs_df: pd.DataFrame) -> dict[str, dict]:
    """Reconstruct {alias -> {runs: df}} from concatenated runs DataFrame."""
    return {alias: {"runs": group.copy()} for alias, group in runs_df.groupby("alias")}


def save_figure(
    fig: plt.Figure,
    output: str,
    params: dict,
    chart_data: dict | None = None,
) -> None:
    """Save matplotlib figure as PNG. Optionally emit interactive chart JSON.

    When chart_data is provided, writes a JSON file to figures/out/charts/<stem>.json
    alongside the PNG. The chart-loader on the website fetches this JSON and renders
    an interactive Plotly chart via lyptus-charts.js, replacing the static PNG.

    chart_data envelope: {"chart_type": str, "version": 1, "data": ..., "options": ...}
    """
    dpi = params.get("output", {}).get("dpi", 300)
    fmt = params.get("output", {}).get("format", "png")
    out = _resolve(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, format=fmt, bbox_inches="tight")
    print(f"Saved: {out}")

    if chart_data is not None:
        save_chart_json(chart_data, output)


def build_campaign_data(
    model_runs: pd.DataFrame,
    assembled_runs: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Build campaign_data dict compatible with lib/irt.py plot functions.

    The lib/irt.py OS analysis functions expect {alias: {runs, runs_human}}.
    This bridges from the pipeline's parquet-based data to that format.

    Args:
        model_runs: Raw evaluation results (task_id, task_family, alias, score_binarized).
        assembled_runs: Output of assemble_runs() with human_minutes, log2_human_minutes,
                       invsqrt_task_weight. If provided, added as runs_human per model.

    Returns:
        {alias: {"runs": df, "runs_human": df}} dict.
    """
    campaign_data = {}
    for alias, group in model_runs.groupby("alias"):
        entry = {"runs": group.copy()}
        if assembled_runs is not None:
            human_subset = assembled_runs[assembled_runs["alias"] == alias]
            if len(human_subset) > 0:
                entry["runs_human"] = human_subset.copy()
        campaign_data[alias] = entry
    return campaign_data


def _resolve(output: str) -> Path:
    """Resolve output path relative to _NOTEBOOKS_DIR if not absolute."""
    return Path(output) if Path(output).is_absolute() else _NOTEBOOKS_DIR / output


def _sanitize_for_json(obj):
    """Deep-sanitize a Python object for JSON serialization.

    Handles numpy types, NaN/Inf, and nested structures. This is the
    canonical sanitizer used by both save_figure() and save_chart_json().
    """
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if v != v or v == float("inf") or v == float("-inf"):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_chart_json(chart_data: dict, png_output: str) -> Path:
    """Serialize chart_data to figures/out/charts/<stem>.json.

    This is the source-of-truth JSON that both matplotlib PNGs and
    interactive Plotly charts render from.
    """
    out = _resolve(png_output)
    chart_dir = out.parent / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / (out.stem + ".json")
    with open(chart_path, "w") as f:
        json.dump(_sanitize_for_json(chart_data), f, default=str)
    print(f"Saved chart JSON: {chart_path}")
    return chart_path


def load_chart_json(png_output: str) -> dict:
    """Load chart JSON from the canonical location for a given PNG output path."""
    out = _resolve(png_output)
    chart_path = out.parent / "charts" / (out.stem + ".json")
    with open(chart_path) as f:
        return json.load(f)


def save_png(fig: plt.Figure, output: str, params: dict) -> None:
    """Save matplotlib figure as PNG only. No JSON side-effect."""
    dpi = params.get("output", {}).get("dpi", 300)
    fmt = params.get("output", {}).get("format", "png")
    out = _resolve(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, format=fmt, bbox_inches="tight")
    print(f"Saved: {out}")


def coerce_floats(values: list) -> list[float]:
    """Convert JSON list to float list, treating None as NaN.

    JSON round-trips replace NaN with null. matplotlib needs NaN (not None)
    for gaps in line plots. Call this in render_png functions before plotting.
    """
    return [float("nan") if v is None else float(v) for v in values]


def coerce_float(v) -> float:
    """Single value: None -> NaN, else float."""
    return float("nan") if v is None else float(v)


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--params", default="figures/params.yaml")
    p.add_argument("--output", required=True)
    return p
