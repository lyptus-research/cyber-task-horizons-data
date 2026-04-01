"""Stage: Collect source-sensitivity results and produce comparison table.

This stage runs AFTER fit_summaries + fit_trendline have been run for each
variant. It simply reads the trendline JSON outputs and model_summaries
parquets, compiles a comparison table, and writes the combined results.

The actual IRT fitting and trendline computation use the shared trunk
(fit_summaries.py → fit_trendline.py) which calls METR's agent_regression()
and get_sota_agents(). This stage does no fitting of its own.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import base_parser, load_params


# Models whose P50 we highlight in the comparison
FRONTIER_MODELS = ["Opus 4.6", "GPT-5.3 Codex"]


def _load_variant_result(
    trendline_json: Path,
    summaries_parquet: Path,
) -> dict | None:
    """Load a single variant's trendline params + per-model P50s."""
    if not trendline_json.exists() or not summaries_parquet.exists():
        return None

    with open(trendline_json) as f:
        trendline = json.load(f)

    summaries = pd.read_parquet(summaries_parquet)

    # Extract per-model P50 horizons
    p50_by_model = {}
    for _, row in summaries.iterrows():
        alias = row["agent"]  # fit_summaries uses alias as agent
        p50 = row.get("p50")
        if p50 is not None and not np.isnan(p50) and p50 > 0:
            p50_by_model[alias] = round(float(p50), 2)

    result = {
        "doubling_time_months": trendline.get("doubling_time_months"),
        "doubling_time_days": trendline.get("doubling_time_days"),
        "r_squared": trendline.get("r_squared"),
        "slope": trendline.get("slope"),
        "intercept": trendline.get("intercept"),
        "n_sota_models": trendline.get("n_sota_models"),
        "sota_models": trendline.get("sota_models"),
        "non_sota_models": trendline.get("non_sota_models"),
        "n_tasks": int(summaries.iloc[0].get("n_tasks", 0)) if len(summaries) > 0 else 0,
        "bootstrap_ci": trendline.get("bootstrap_ci"),
        "p50_by_model": p50_by_model,
    }

    # Top-level frontier P50s
    for model in FRONTIER_MODELS:
        key = f"p50_{model.lower().replace(' ', '_').replace('.', '_')}"
        result[key] = p50_by_model.get(model)

    return result


def main():
    parser = base_parser("Collect source-sensitivity results")
    parser.add_argument("--sensitivity-dir", required=True,
                        help="Directory containing variant summaries and trendline JSONs")
    parser.add_argument("--headline-trendline", required=True,
                        help="Headline trendline_params JSON")
    parser.add_argument("--headline-summaries", required=True,
                        help="Headline model_summaries parquet")
    args = parser.parse_args()

    sens_dir = Path(args.sensitivity_dir)
    if not sens_dir.is_absolute():
        sens_dir = _NOTEBOOKS_DIR / sens_dir

    # Load headline
    headline_tl = Path(args.headline_trendline)
    headline_sm = Path(args.headline_summaries)
    if not headline_tl.is_absolute():
        headline_tl = _NOTEBOOKS_DIR / headline_tl
    if not headline_sm.is_absolute():
        headline_sm = _NOTEBOOKS_DIR / headline_sm

    results = {"headline": _load_variant_result(headline_tl, headline_sm)}

    # Load each variant
    for tl_json in sorted(sens_dir.glob("trendline_*.json")):
        variant_name = tl_json.stem.replace("trendline_", "")
        sm_parquet = sens_dir / f"summaries_{variant_name}.parquet"
        result = _load_variant_result(tl_json, sm_parquet)
        results[variant_name] = result

    # Print comparison table
    headline_dt = (results.get("headline") or {}).get("doubling_time_months")
    print(f"\n{'Variant':<25} {'DT (mo)':<10} {'DT (d)':<10} {'Δ%':<10} {'R²':<8} {'SOTA':<6}", end="")
    for m in FRONTIER_MODELS:
        print(f"  {m} P50", end="")
    print()
    print("-" * 110)

    for name, metrics in results.items():
        if metrics is None:
            print(f"{name:<25} {'no data'}")
            continue

        dt_mo = metrics["doubling_time_months"]
        dt_d = metrics["doubling_time_days"]
        if dt_mo is None:
            print(f"{name:<25} {'no trendline'}")
            continue

        delta = f"{(dt_mo - headline_dt) / headline_dt * 100:+.1f}%" if headline_dt and name != "headline" else "—"
        r2 = metrics["r_squared"] or 0
        n_sota = metrics["n_sota_models"] or 0
        line = f"{name:<25} {dt_mo:<10.1f} {dt_d:<10.0f} {delta:<10} {r2:<8.3f} {n_sota:<6}"

        for m in FRONTIER_MODELS:
            key = f"p50_{m.lower().replace(' ', '_').replace('.', '_')}"
            p50 = metrics.get(key)
            if p50:
                line += f"  {p50:.0f}m"
            else:
                line += f"  —"
        print(line)

    # Save combined results
    out = Path(args.output)
    if not out.is_absolute():
        out = _NOTEBOOKS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
