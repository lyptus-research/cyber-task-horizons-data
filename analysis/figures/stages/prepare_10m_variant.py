"""Stage: Prepare 10M-augmented model_runs for GPT-5.3 Codex.

Takes the canonical model_runs.parquet and flips GPT-5.3 Codex failures
to successes where the 10M re-run passed. All other models unchanged.
This lets the standard bootstrap + fit_summaries pipeline produce
properly bootstrapped CIs for the 10M P50.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import base_parser  # noqa: E402


def main():
    parser = base_parser("Prepare 10M-augmented model_runs for GPT-5.3 Codex")
    parser.add_argument("--model-runs", required=True)
    parser.add_argument("--ten-m-cache", required=True, help="Path to .10m_samples.pkl")
    parser.add_argument("--model-alias", default="GPT-5.3 Codex")
    parser.add_argument("--budget-cap", type=int, default=10_000_000)
    args = parser.parse_args()

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else _NOTEBOOKS_DIR / p

    mr = pd.read_parquet(resolve(args.model_runs))

    with open(resolve(args.ten_m_cache), "rb") as f:
        ten_m_data = pickle.load(f)
    task_10m = {
        s["task_id"]: {"score": s["score"], "tokens": s["total_tokens"]}
        for s in ten_m_data
    }

    # Flip failures to successes where 10M re-run passed within budget
    mask = mr["alias"] == args.model_alias
    flipped = 0
    for idx in mr[mask].index:
        row = mr.loc[idx]
        if row["score_binarized"] == 0 and row["task_id"] in task_10m:
            r10 = task_10m[row["task_id"]]
            if r10["score"] > 0 and r10["tokens"] <= args.budget_cap:
                mr.loc[idx, "score_binarized"] = 1
                flipped += 1

    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mr.to_parquet(out)

    total = mr[mask]["score_binarized"].sum()
    print(
        f"{args.model_alias}: flipped {flipped} tasks at {args.budget_cap/1e6:.0f}M budget"
    )
    print(f"  Successes: {total - flipped} (2M) + {flipped} (10M re-run) = {total}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
