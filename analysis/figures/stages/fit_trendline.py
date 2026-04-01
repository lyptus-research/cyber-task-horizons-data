"""Stage: Fit exponential trendline on SOTA models.

Reads model_summaries (from fit_summaries) and bootstrap results.
Fits log-space linear regression on SOTA-only models.
Writes trendline_params.json with the headline numbers.

Uses METR's compute_bootstrap_confidence_region for CI.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

# Use monorepo-level METR checkout (CTH-level copy is outdated)

from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region  # noqa: E402
from figures.stages._common_data import load_release_dates  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fit trendline on SOTA models")
    parser.add_argument("--summaries", required=True, help="model_summaries parquet")
    parser.add_argument("--bootstrap", required=True, help="Bootstrap parquet")
    parser.add_argument("--success-percent", type=int, default=50)
    parser.add_argument("--output", required=True, help="Output trendline_params.json")
    args = parser.parse_args()

    summaries = pd.read_parquet(args.summaries)
    bootstrap_df = pd.read_parquet(args.bootstrap)
    release_dates = load_release_dates()
    pct = args.success_percent
    p_col = f"p{pct}"

    # Filter to SOTA models
    sota = summaries[summaries["is_sota"]].copy()
    sota = sota.dropna(subset=["release_date", p_col])
    sota = sota[sota[p_col] > 0]

    if len(sota) < 2:
        raise ValueError(
            f"Only {len(sota)} SOTA models — need at least 2 for trendline"
        )

    # Fit log-space linear regression
    X = mdates.date2num(sota["release_date"].values).reshape(-1, 1)
    y = np.log(sota[p_col].values)
    reg = LinearRegression().fit(X, y)

    dt_days = np.log(2) / reg.coef_[0]
    r2 = reg.score(X, y)

    # Bootstrap CI on doubling time
    rd_dates = {k: pd.Timestamp(v).date() for k, v in release_dates.items()}
    dt_ci = None
    try:
        earliest = pd.to_datetime(sota["release_date"]).min().strftime("%Y-%m-%d")
        latest = pd.to_datetime(sota["release_date"]).max().strftime("%Y-%m-%d")

        boot = bootstrap_df
        if pct != 50:
            p50_cols = [c for c in boot.columns if c.endswith("_p50")]
            boot = boot.drop(columns=p50_cols)
            rename_map = {
                c: c.replace(f"_p{pct}", "_p50")
                for c in boot.columns
                if c.endswith(f"_p{pct}")
            }
            boot = boot.rename(columns=rename_map)

        dt_stats, _, _, _ = compute_bootstrap_confidence_region(
            agent_summaries=sota,
            bootstrap_results=boot,
            release_dates={"date": rd_dates},
            after_date=earliest,
            sota_before_date=latest,
            trendline_end_date="2027-01-01",
            confidence_level=0.95,
            filter_sota=False,  # already filtered to SOTA
        )
        dt_ci = {
            "median_days": dt_stats.median,
            "ci_lower_days": dt_stats.ci_lower,
            "ci_upper_days": dt_stats.ci_upper,
        }
    except Exception as e:
        print(f"Bootstrap CI failed: {e}")

    # Write results
    result = {
        "success_percent": pct,
        "doubling_time_days": round(dt_days, 1),
        "doubling_time_months": round(dt_days / 30.44, 1),
        "r_squared": round(r2, 4),
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "n_sota_models": len(sota),
        "sota_models": sorted(sota["agent"].tolist()),
        "all_models": sorted(summaries["agent"].tolist()),
        "non_sota_models": sorted(summaries[~summaries["is_sota"]]["agent"].tolist()),
    }
    if dt_ci:
        result["bootstrap_ci"] = dt_ci

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Doubling time: {dt_days:.0f} days ({dt_days/30.44:.1f} months)")
    print(f"R² = {r2:.4f}")
    print(f"SOTA models ({len(sota)}): {sorted(sota['agent'].tolist())}")
    print(f"Non-SOTA: {sorted(summaries[~summaries['is_sota']]['agent'].tolist())}")
    if dt_ci:
        print(
            f"Bootstrap CI: [{dt_ci['ci_lower_days']:.0f}, {dt_ci['ci_upper_days']:.0f}] days"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
