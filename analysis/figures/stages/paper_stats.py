"""Generate paper_stats.json with key numbers for the website.

Reads pipeline trunk outputs and emits a flat JSON of statistics that
the Jekyll post can reference via {{ site.data.paper_stats.<key> }}.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _NOTEBOOKS_DIR / p


def _fmt_minutes_short(minutes: float) -> str:
    """Format minutes as a compact string (e.g. '3.2h', '45m')."""
    if minutes < 1:
        return f"{round(minutes * 60)}s"
    if minutes < 60:
        m = int(minutes) if minutes == int(minutes) else round(minutes, 1)
        return f"{m}m"
    h = minutes / 60
    return f"{h:.1f}h"


def _natural_time_display(months: float) -> str:
    """Convert months to natural-language time expression for projections."""
    if months < 1.5:
        return "roughly a month"
    if months < 10:
        return f"roughly {round(months)} months"
    if months < 15:
        return "roughly a year"
    if months < 21:
        return "under two years"
    years = months / 12
    half_years = round(years * 2) / 2
    if half_years == int(half_years):
        return f"roughly {int(half_years)} years"
    return f"roughly {int(half_years)} and a half years"


from analysis.config import MODELS_DIR as _MODELS_DIR

_ALIAS_MAP = {
    "Claude Haiku 4.5": "Haiku 4.5",
    "Claude Opus 4": "Opus 4",
    "Claude Sonnet 4.6": "Sonnet 4.6",
    "Claude Opus 4.6": "Opus 4.6",
    "Gemini 2.5 Pro (June 2025)": "Gemini 2.5 Pro",
}

_PROVIDER_DISPLAY = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
}

# Models where the config doesn't capture the blog-post thinking label
_THINKING_OVERRIDES = {
    "Gemini 2.5 Pro": "Adaptive (high)",
    "DeepSeek V3.1": "Default (on)",
    "GLM-5": "Default (on)",
}

_LEGACY_MODELS = [
    {
        "alias": "GPT-2",
        "release": "2019-11",
        "release_full": "2019-11-05",
        "provider": "OpenAI",
        "thinking": "\u2014",
    },
    {
        "alias": "GPT-3",
        "release": "2020-07",
        "release_full": "2020-07-11",
        "provider": "OpenAI",
        "thinking": "\u2014",
    },
    {
        "alias": "GPT-3.5",
        "release": "2022-03",
        "release_full": "2022-03-15",
        "provider": "OpenAI",
        "thinking": "\u2014",
    },
]


def _thinking_label(model: dict) -> str:
    """Format the thinking config for display."""
    thinking = model.get("thinking")
    if not thinking:
        return "\u2014"
    effort = thinking.get("reasoning_effort")
    tokens = thinking.get("reasoning_tokens")
    if effort:
        return f"Adaptive ({effort})"
    if tokens:
        return f"Fixed-budget ({tokens // 1000}K)"
    return "Default (on)"


def _build_models_table(active_aliases: list[str]) -> list[dict]:
    """Build models table from model config JSONs, filtered to active models."""
    rows = list(_LEGACY_MODELS)
    active_set = set(active_aliases)

    if _MODELS_DIR.exists():
        for json_file in sorted(_MODELS_DIR.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            provider_raw = data.get("metadata", {}).get("company", "")
            for model in data.get("models", []):
                alias = model.get("alias", "")
                campaign_alias = _ALIAS_MAP.get(alias, alias)
                if campaign_alias not in active_set:
                    continue
                release_full = model.get("release_date", "")
                release = release_full[:7]  # YYYY-MM for display
                provider = _PROVIDER_DISPLAY.get(provider_raw, "")
                # Together-hosted models: derive display provider from alias
                if not provider:
                    name_lower = (alias + model.get("model_name", "")).lower()
                    if "deepseek" in name_lower:
                        provider = "Together/DeepSeek"
                    elif "glm" in name_lower or "zhipu" in name_lower:
                        provider = "Together/Zhipu"
                    else:
                        provider = model.get("provider", provider_raw).title()
                thinking = _THINKING_OVERRIDES.get(
                    campaign_alias, _thinking_label(model)
                )
                rows.append(
                    {
                        "alias": campaign_alias,
                        "release": release,
                        "release_full": release_full,
                        "provider": provider,
                        "thinking": thinking,
                    }
                )

    # Sort by release date, filter to only active models
    rows = [r for r in rows if r["alias"] in active_set]
    rows.sort(key=lambda r: r["release_full"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate paper stats JSON")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument("--summaries", required=True)
    parser.add_argument(
        "--summaries-10m",
        default=None,
        help="model_summaries with 10M-augmented GPT-5.3",
    )
    parser.add_argument("--trendline", required=True)
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument(
        "--token-subset-chart",
        default=None,
        help="token_subset_analysis.json chart data",
    )
    parser.add_argument(
        "--ten-m-cache",
        default=None,
        help="Path to 10m_samples.pkl for re-run counts",
    )
    parser.add_argument(
        "--summaries-model-est",
        default=None,
        help="model_summaries_runs_2M.parquet (model-estimated difficulty)",
    )
    parser.add_argument(
        "--regularisation-chart",
        default=None,
        help="regularisation_comparison.json from pipeline data",
    )
    parser.add_argument(
        "--trendline-alt-chart",
        default=None,
        help="trendline_alternatives.json chart data",
    )
    parser.add_argument(
        "--os-trendline-chart",
        default=None,
        help="os_main_trendline.json chart data (for adaptation buffers)",
    )
    parser.add_argument(
        "--token-budget-chart",
        default=None,
        help="token_budget_sensitivity.json chart data",
    )
    parser.add_argument(
        "--token-budget-extended-chart",
        default=None,
        help="token_budget_extended_10m.json chart data",
    )
    parser.add_argument(
        "--gpt55-50m-overlay",
        default=None,
        help="JSON of {task_id: {score, tokens}} from the 50M re-run pass",
    )
    parser.add_argument("--params", default="figures/params.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(_resolve(args.params)) as f:
        params = yaml.safe_load(f)

    task_diff = pd.read_parquet(_resolve(args.task_difficulties))
    model_runs = pd.read_parquet(_resolve(args.model_runs))
    summaries = pd.read_parquet(_resolve(args.summaries))
    with open(_resolve(args.trendline)) as f:
        trendline = json.load(f)
    with open(_resolve(args.human_snapshot)) as f:
        snapshot = json.load(f)

    # Headline set: tasks with best-available human times AND model evaluations
    eval_tasks = set(model_runs["task_id"].astype(str))
    headline = task_diff.dropna(subset=["best_available_minutes"])
    headline = headline[headline["task_id"].astype(str).isin(eval_tasks)]
    headline_task_ids = set(headline["task_id"].astype(str))

    # Task counts
    headline_tasks = len(headline)
    eval_set_tasks = model_runs["task_id"].nunique()
    n_benchmarks = headline["task_family"].nunique()

    # Model counts
    n_models = len(summaries)
    n_sota = int(summaries["is_sota"].sum())
    n_non_sota = n_models - n_sota

    # Model lists
    sota_models = sorted(summaries.loc[summaries["is_sota"], "agent"].tolist())
    non_sota_models = sorted(summaries.loc[~summaries["is_sota"], "agent"].tolist())
    all_models = sorted(summaries["agent"].tolist())

    # Trendline (2019+)
    doubling_time_days = trendline["doubling_time_days"]
    doubling_time_months = trendline["doubling_time_months"]
    r_squared = trendline["r_squared"]
    bootstrap = trendline.get("bootstrap_ci", {})
    ci_lower_days = round(bootstrap.get("ci_lower_days", 0), 0)
    ci_upper_days = round(bootstrap.get("ci_upper_days", 0), 0)
    ci_lower_months = round(ci_lower_days / 30.44, 1)
    ci_upper_months = round(ci_upper_days / 30.44, 1)

    # Trendline (2024+) — uses METR's fit_trendline
    # Both paths needed: src/ for `import horizon.*`, parent for `import src.*`

    sota_summaries = summaries[summaries["is_sota"]].copy()
    sota_summaries = sota_summaries[
        sota_summaries["p50"].notna() & (sota_summaries["p50"] > 0)
    ]
    cutoff_2024 = pd.Timestamp("2024-01-01")
    recent_sota = sota_summaries[
        pd.to_datetime(sota_summaries["release_date"]) >= cutoff_2024
    ]
    if len(recent_sota) >= 2:
        reg_2024, r2_2024 = _metr_fit(
            recent_sota["p50"],
            pd.to_datetime(recent_sota["release_date"]),
            log_scale=True,
        )
        dt_2024_days = (
            1.0 / reg_2024.coef_[0] * np.log(2)
            if reg_2024.coef_[0] > 0
            else float("inf")
        )
        doubling_time_2024_months = round(dt_2024_days / 30.44, 1)
        r_squared_2024 = round(r2_2024, 2)
    else:
        doubling_time_2024_months = None
        r_squared_2024 = None

    # Frontier model P50s (top 2 by p50)
    top2 = summaries.nlargest(2, "p50")
    frontier_models = []
    for _, row in top2.iterrows():
        ci_lo = float(row["p50q0.025"]) / 60
        ci_hi = float(row["p50q0.975"]) / 60
        frontier_models.append(
            {
                "name": row["agent"],
                "p50_minutes": round(row["p50"], 1),
                "p50_hours": round(row["p50"] / 60, 1),
                "p50_display": _fmt_minutes_short(row["p50"]),
                "ci_lo_hours": round(ci_lo, 1),
                "ci_hi_hours": round(ci_hi, 1),
                "ci_display": f"[{ci_lo:.1f}h, {ci_hi:.0f}h]",
            }
        )

    # 10M-augmented GPT-5.3 Codex stats (if available)
    token_budget_10m = {}
    if args.summaries_10m:
        s10m = pd.read_parquet(_resolve(args.summaries_10m))
        gpt53_10m = s10m[s10m["agent"] == "GPT-5.3 Codex"]
        if len(gpt53_10m):
            row10 = gpt53_10m.iloc[0]
            token_budget_10m = {
                "p50_minutes": round(float(row10["p50"]), 1),
                "p50_hours": round(float(row10["p50"]) / 60, 1),
                "p50_display": _fmt_minutes_short(row10["p50"]),
                "ci_lo_hours": round(float(row10["p50q0.025"]) / 60, 1),
                "ci_hi_hours": round(float(row10["p50q0.975"]) / 60, 1),
                "ci_display": f"[{row10['p50q0.025']/60:.1f}h, {row10['p50q0.975']/60:.1f}h]",
            }

    # Per-model P50 lookup (for prose references to specific models)
    model_p50 = {}
    for _, row in summaries.iterrows():
        if pd.notna(row["p50"]) and row["p50"] > 0:
            model_p50[row["agent"]] = _fmt_minutes_short(row["p50"])

    # Earliest and latest SOTA models
    sota_df = summaries[summaries["is_sota"]].sort_values("release_date")
    earliest_sota = sota_df.iloc[0]["agent"]
    latest_sota = sota_df.iloc[-1]["agent"]
    earliest_sota_year = int(pd.Timestamp(sota_df.iloc[0]["release_date"]).year)
    latest_sota_year = int(pd.Timestamp(sota_df.iloc[-1]["release_date"]).year)
    latest_sota_month = pd.Timestamp(sota_df.iloc[-1]["release_date"]).strftime("%B %Y")

    # Bootstrap-over-models probability stats
    # Probability a specific model is absent from a bootstrap sample of n_sota
    prob_absent_one = round((1 - 1 / n_sota) ** n_sota, 2) if n_sota > 0 else 0
    # Probability at least one of the top-2 frontier models is absent
    prob_either_absent = round(1 - (1 - prob_absent_one) ** 2, 2)

    # Frontier models' ratio to trendline prediction
    from matplotlib.dates import date2num

    slope = trendline["slope"]
    intercept = trendline["intercept"]
    frontier_above_trend = []
    for _, row in top2.iterrows():
        rd_num = date2num(pd.Timestamp(row["release_date"]))
        predicted_p50 = np.exp(slope * rd_num + intercept)
        actual_p50 = row["p50"]
        ratio = round(actual_p50 / predicted_p50, 1) if predicted_p50 > 0 else None
        frontier_above_trend.append(
            {
                "name": row["agent"],
                "actual_p50": round(actual_p50, 1),
                "predicted_p50": round(predicted_p50, 1),
                "ratio": ratio,
            }
        )
    # Average ratio across top-2
    ratios = [f["ratio"] for f in frontier_above_trend if f["ratio"] is not None]
    frontier_above_trend_avg = round(np.mean(ratios), 1) if ratios else None

    # --- Adaptation buffer: read from OS trendline chart (uses 2024+ fit) ---
    adaptation_buffers = {}
    if args.os_trendline_chart:
        os_path = _resolve(args.os_trendline_chart)
        if os_path.exists():
            with open(os_path) as f:
                os_data = json.load(f)
            for m in os_data.get("data", {}).get("os_models", []):
                name = m["name"]
                buf = m.get("buffer_months")
                if buf is not None:
                    adaptation_buffers[name] = buf

    buffer_values = sorted(adaptation_buffers.values()) if adaptation_buffers else []
    adaptation_buffer_lower_months = buffer_values[0] if buffer_values else None
    adaptation_buffer_upper_months = buffer_values[-1] if buffer_values else None

    # --- Human study stats ---
    # Canonical timing field: server_elapsed_seconds (study_progress_guide.md)
    # Estimation sessions capped at 45 min (EST_SESSION_CAP_SECONDS in lib/budget.py)
    EST_CAP_SECONDS = 45 * 60

    # headline_task_ids already computed above from task_diff

    # Difficulty range (from best_available_minutes in task_difficulties)
    task_times = headline["best_available_minutes"]
    difficulty_min_seconds = round(task_times.min() * 60)
    _max_hours = round(task_times.max() / 60, 1)
    difficulty_max_hours = (
        int(_max_hours) if _max_hours == int(_max_hours) else _max_hours
    )

    # Unique expert participants
    all_user_ids = set()
    for key in ("passes", "fails", "censored", "estimations"):
        for s in snapshot.get(key, []):
            uid = s.get("user_id")
            if uid:
                all_user_ids.add(uid)
    n_participants = len(all_user_ids)

    # Completion hours (passes + fails + censored)
    # Uses corrected_elapsed() to apply timing corrections consistently.
    from lib.corrections import corrected_elapsed, TIMING_CORRECTIONS

    comp_hours = 0.0
    for key in ("passes", "fails", "censored"):
        comp_hours += sum(
            corrected_elapsed(s, TIMING_CORRECTIONS) / 3600
            for s in snapshot.get(key, [])
            if corrected_elapsed(s, TIMING_CORRECTIONS) > 0
        )

    # Estimation hours (server_elapsed with 45-min cap)
    est_times = [
        corrected_elapsed(s, TIMING_CORRECTIONS)
        for s in snapshot.get("estimations", [])
        if corrected_elapsed(s, TIMING_CORRECTIONS) > 0
    ]
    est_capped = [min(t, EST_CAP_SECONDS) for t in est_times]
    est_hours = sum(est_capped) / 3600
    n_est_missing = len(snapshot.get("estimations", [])) - len(est_times)
    if est_capped and n_est_missing > 0:
        est_hours += n_est_missing * (sum(est_capped) / len(est_capped)) / 3600

    total_expert_hours = comp_hours + est_hours

    # Total unique tasks with any human data (may exceed headline set)
    all_study_tasks = set()
    for key in ("passes", "fails", "censored"):
        for s in snapshot.get(key, []):
            all_study_tasks.add(s["task_id"])
    for s in snapshot.get("estimations", []):
        all_study_tasks.add(s["task_id"])
    total_study_tasks = len(all_study_tasks)

    # Per-source task counts from task_difficulties (headline set)
    n_estimation_tasks = int(headline["estimate_minutes"].notna().sum())
    # Total tasks with first-blood data (not just where first-blood won the hierarchy)
    n_firstblood_tasks = int(headline["firstblood_minutes"].notna().sum())
    # Tasks where first-blood is the winning source (subset of above)
    n_firstblood_winner_tasks = int(
        (headline["best_available_source"] == "first_blood").sum()
    )

    # Completion counts (from snapshot, filtered to headline set)
    from collections import Counter

    all_completions = snapshot.get("completions", [])
    headline_completions = [
        s for s in all_completions if s["task_id"] in headline_task_ids
    ]
    n_completion_sessions = len(headline_completions)
    completion_task_counts = Counter(s["task_id"] for s in headline_completions)
    n_completion_tasks = len(completion_task_counts)
    n_completion_paired_tasks = sum(
        1 for c in completion_task_counts.values() if c >= 2
    )

    # Completion-equivalent hours: total estimated task difficulty in hours
    has_estimate = headline["estimate_minutes"].notna()
    completion_equivalent_hours = round(
        headline.loc[has_estimate, "estimate_minutes"].sum() / 60
    )
    estimation_efficiency_ratio = (
        round(completion_equivalent_hours / est_hours, 1) if est_hours > 0 else 0
    )

    # --- Projection timelines ---
    proj_params = params.get("projections", {})
    frontier_p50_hours = frontier_models[0]["p50_hours"]
    projections = {}
    targets = {
        "fullday": proj_params.get("fullday_hours", 8),
        "multiday": proj_params.get("multiday_hours", 40),
        "redteam": proj_params.get("redteam_hours", 160),
    }
    for label, dt in [
        ("2019", doubling_time_months),
        ("2024", doubling_time_2024_months),
    ]:
        if dt is None or dt <= 0:
            continue
        for name, target_h in targets.items():
            months = dt * math.log2(target_h / frontier_p50_hours)
            projections[f"projection_{label}_{name}_months"] = round(months, 1)
            projections[f"projection_{label}_{name}_display"] = _natural_time_display(
                months
            )

    # --- Token budget accuracy stats from chart JSON ---
    token_accuracy = {}
    if args.token_subset_chart:
        chart_path = _resolve(args.token_subset_chart)
        if chart_path.exists():
            with open(chart_path) as f:
                chart = json.load(f)
            for subset in chart.get("data", {}).get("subsets", []):
                key = subset["key"]  # "all", "cybergym", "hard_30m", "hard_2h"
                points = {p["budget"]: p["accuracy"] for p in subset["points"]}
                for budget, label in [
                    (50_000, "50k"),
                    (5_000_000, "5m"),
                    (10_000_000, "10m"),
                ]:
                    if budget in points:
                        token_accuracy[f"token_{key}_{label}_pct"] = points[budget]

    # --- Model-estimated P50s (for Appendix D comparison) ---
    model_p50_model_est = {}
    dt_model_est_months = None
    if args.summaries_model_est:
        s_me = pd.read_parquet(_resolve(args.summaries_model_est))
        for _, row in s_me.iterrows():
            if pd.notna(row["p50"]) and row["p50"] > 0:
                model_p50_model_est[row["agent"]] = _fmt_minutes_short(row["p50"])
        # Compute model-estimated 2019+ doubling time
        sota_me = s_me[s_me["is_sota"]].copy()
        sota_me = sota_me[sota_me["p50"].notna() & (sota_me["p50"] > 0)]
        if len(sota_me) >= 2:
            reg_me, r2_me = _metr_fit(
                sota_me["p50"],
                pd.to_datetime(sota_me["release_date"]),
                log_scale=True,
            )
            dt_me_days = (
                1.0 / reg_me.coef_[0] * np.log(2)
                if reg_me.coef_[0] > 0
                else float("inf")
            )
            dt_model_est_months = round(dt_me_days / 30.44, 1)

    # --- Strong regularisation doubling time (from regularisation_comparison.json) ---
    dt_strong_reg_months = None
    if args.regularisation_chart:
        reg_path = _resolve(args.regularisation_chart)
        if reg_path.exists():
            with open(reg_path) as f:
                reg_data = json.load(f)
            dt_by_lambda = reg_data.get("dt_by_lambda", {})
            if "lambda_0.2" in dt_by_lambda:
                dt_strong_reg_months = dt_by_lambda["lambda_0.2"]

    # --- 10M re-run counts (from pickle cache) ---
    ten_m_stats = {}
    if args.ten_m_cache:
        import pickle

        cache_path = _resolve(args.ten_m_cache)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                ten_m_data = pickle.load(f)
            n_rerun_total = len(ten_m_data)
            n_rerun_passed = sum(1 for s in ten_m_data if s["score"] > 0)
            n_rerun_over_2m = sum(
                1
                for s in ten_m_data
                if s["score"] > 0 and s["total_tokens"] > 2_000_000
            )
            n_rerun_under_2m = n_rerun_passed - n_rerun_over_2m
            ten_m_stats = {
                "ten_m_rerun_total": n_rerun_total,
                "ten_m_rerun_passed": n_rerun_passed,
                "ten_m_rerun_over_2m": n_rerun_over_2m,
                "ten_m_rerun_under_2m": n_rerun_under_2m,
            }

    # --- Token budget 1M-to-2M gain ranges (from chart JSON) ---
    token_gain_stats = {}
    if args.token_budget_chart:
        tb_path = _resolve(args.token_budget_chart)
        if tb_path.exists():
            with open(tb_path) as f:
                tb_data = json.load(f)
            rising_ratios = []
            non_rising_ratios = []
            for m in tb_data.get("data", {}).get("models", []):
                r = m.get("ratio", 1.0)
                if m.get("rising"):
                    rising_ratios.append(r)
                else:
                    non_rising_ratios.append(r)
            if rising_ratios:
                token_gain_stats["token_gain_rising_min"] = round(min(rising_ratios), 1)
                token_gain_stats["token_gain_rising_max"] = round(max(rising_ratios), 1)
            if non_rising_ratios:
                max_non = max(non_rising_ratios)
                token_gain_stats["token_gain_non_rising_max_pct"] = round(
                    (max_non - 1) * 100
                )

    # --- Extended budget ratios (from extended chart JSON) ---
    ext_budget_stats = {}
    if args.token_budget_extended_chart:
        ext_path = _resolve(args.token_budget_extended_chart)
        if ext_path.exists():
            with open(ext_path) as f:
                ext_data = json.load(f)
            for dr in ext_data.get("data", {}).get("doubling_rows", []):
                if dr.get("is_extended"):
                    label = dr["budget_label"]
                    key = label.replace("\u2192", "_to_").replace("M", "m")
                    ext_budget_stats[f"ext_ratio_{key}"] = dr["ratio"]
            p50pm = (
                ext_data.get("data", {})
                .get("gpt53_extended", {})
                .get("p50_per_million", {})
            )
            if p50pm:
                ext_budget_stats["ext_p50_per_million"] = p50pm

    # --- GPT-5.5 per-benchmark pass rate (2M single-attempt + 50M retry overlay) ---
    # Used by the GPT-5.5 saturation note (May 2026). Subset = every task evaluated
    # for GPT-5.5, not the headline human-labelled set, because the saturation claim
    # is about model behaviour on the entire suite.
    _BENCH_DISPLAY = {
        "cybench": "CyBench",
        "intercode_ctf": "InterCode-CTF",
        "cybashbench": "CyBashBench",
        "nl2bash": "NL2Bash",
        "cvebench": "CVEBench",
        "nyuctf": "NYUCTF",
        "cybergym": "CyberGym",
    }
    def _pct_display(n: int, total: int) -> str:
        """100% when integer, else one decimal (94.9%)."""
        pct = 100 * n / total
        return f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"

    def _signed_pp(delta: float) -> str:
        """Format pp delta with leading sign (+5.1pp, -3.0pp, 0pp)."""
        if abs(delta) < 0.05:
            return "0pp"
        sign = "+" if delta > 0 else "-"
        return f"{sign}{abs(delta):.1f}pp"

    gpt55_per_benchmark = []
    gpt55_overall = {}
    g55 = model_runs[model_runs["alias"] == "GPT-5.5"].copy()
    if len(g55):
        overlay = {}
        if args.gpt55_50m_overlay:
            overlay_path = _resolve(args.gpt55_50m_overlay)
            if overlay_path.exists():
                with open(overlay_path) as f:
                    overlay = json.load(f)
        # 50M overlay: any rerun that passed (no token cap beyond what was budgeted).
        g55["score_50m"] = g55["task_id"].astype(str).map(
            lambda t: overlay.get(t, {}).get("score")
        )
        # 10M overlay: only count rerun pass if it used <= 10M tokens.
        # Mirrors plot_gpt55_multi_budget_irt._build_gpt55_at_budget(..., 10_000_000).
        def _score_10m(task_id: str) -> float | None:
            rr = overlay.get(task_id)
            if rr is None:
                return None
            if rr.get("tokens", 0) <= 10_000_000:
                return rr.get("score")
            return None
        g55["score_10m"] = g55["task_id"].astype(str).map(_score_10m)
        # Retry overlays never downgrade a 2M pass
        g55["score_combined"] = g55[["score_binarized", "score_50m"]].max(axis=1)
        g55["score_combined_10m"] = g55[["score_binarized", "score_10m"]].max(axis=1)

        # Comparator per-bench rates — restricted to the GPT-5.5 task subset
        # so deltas are apples-to-apples.
        comparators = ["Opus 4.6", "GPT-5.3 Codex"]
        comp_runs = {
            c: model_runs[
                (model_runs["alias"] == c)
                & (model_runs["task_id"].isin(g55["task_id"]))
            ]
            for c in comparators
        }

        for fam, disp in _BENCH_DISPLAY.items():
            sub = g55[g55["task_family"] == fam]
            if not len(sub):
                continue
            total = int(len(sub))
            pass1 = int(sub["score_binarized"].sum())
            comb = int(sub["score_combined"].sum())
            comb10 = int(sub["score_combined_10m"].sum())
            row = {
                "key": fam,
                "name": disp,
                "pass1_n": pass1,
                "pass1_total": total,
                "pass1_pct": round(100 * pass1 / total, 1),
                "pass1_display": f"{_pct_display(pass1, total)} ({pass1}/{total})",
                "pass1_pct_short": _pct_display(pass1, total),
                "retry10m_n": comb10,
                "retry10m_total": total,
                "retry10m_pct": round(100 * comb10 / total, 1),
                "retry10m_display": f"{_pct_display(comb10, total)} ({comb10}/{total})",
                "retry10m_pct_short": _pct_display(comb10, total),
                "retry50m_n": comb,
                "retry50m_total": total,
                "retry50m_pct": round(100 * comb / total, 1),
                "retry50m_display": f"{_pct_display(comb, total)} ({comb}/{total})",
                "retry50m_pct_short": _pct_display(comb, total),
                "delta_50m_vs_2m_pp": round(100 * (comb - pass1) / total, 1),
                "delta_50m_vs_2m_display": _signed_pp(100 * (comb - pass1) / total),
            }
            # Comparator rates (same task subset)
            best_prior_pct = None
            for c in comparators:
                csub = comp_runs[c][comp_runs[c]["task_family"] == fam]
                if not len(csub):
                    row[f"{c.replace(' ', '_').replace('.', '_').replace('-', '_').lower()}_pct"] = None
                    continue
                cn = int(csub["score_binarized"].sum())
                ct = int(len(csub))
                cpct = round(100 * cn / ct, 1)
                slug = c.replace(" ", "_").replace(".", "_").replace("-", "_").lower()
                row[f"{slug}_pct"] = cpct
                row[f"{slug}_pct_short"] = _pct_display(cn, ct)
                row[f"{slug}_display"] = f"{_pct_display(cn, ct)} ({cn}/{ct})"
                if best_prior_pct is None or cpct > best_prior_pct:
                    best_prior_pct = cpct
            if best_prior_pct is not None:
                row["delta_vs_best_prior_pp"] = round(row["pass1_pct"] - best_prior_pct, 1)
                row["delta_vs_best_prior_display"] = _signed_pp(row["pass1_pct"] - best_prior_pct)
            gpt55_per_benchmark.append(row)
        # Sort descending by 2M pass rate, ties broken by name
        gpt55_per_benchmark.sort(key=lambda r: (-r["pass1_pct"], r["name"]))
        total_all = int(len(g55))
        pass1_all = int(g55["score_binarized"].sum())
        comb_all = int(g55["score_combined"].sum())
        comb10_all = int(g55["score_combined_10m"].sum())
        retry_gain_pp = round(100 * (comb_all - pass1_all) / total_all, 1)
        # Stochastic-vs-budget split (mirrors March note's 5.3 Codex 10M reporting).
        # Stochastic = rerun passed using <=2M tokens (pass@2 with no budget benefit).
        # Budget = rerun passed using >2M tokens (genuinely budget-constrained).
        n_flipped = 0
        n_stochastic = 0
        n_budget = 0
        for _, row in g55.iterrows():
            if row["score_binarized"] == 1:
                continue  # already passed at 2M
            rr = overlay.get(str(row["task_id"]))
            if rr is None or rr.get("score", 0) < 0.7:
                continue
            n_flipped += 1
            if rr.get("tokens", 0) <= 2_000_000:
                n_stochastic += 1
            else:
                n_budget += 1
        gpt55_overall = {
            "pass1_n": pass1_all,
            "pass1_total": total_all,
            "pass1_pct": round(100 * pass1_all / total_all, 1),
            "pass1_display": f"{_pct_display(pass1_all, total_all)} ({pass1_all}/{total_all})",
            "pass1_pct_short": _pct_display(pass1_all, total_all),
            "retry10m_n": comb10_all,
            "retry10m_total": total_all,
            "retry10m_pct": round(100 * comb10_all / total_all, 1),
            "retry10m_display": f"{_pct_display(comb10_all, total_all)} ({comb10_all}/{total_all})",
            "retry10m_pct_short": _pct_display(comb10_all, total_all),
            "retry50m_n": comb_all,
            "retry50m_total": total_all,
            "retry50m_pct": round(100 * comb_all / total_all, 1),
            "retry50m_display": f"{_pct_display(comb_all, total_all)} ({comb_all}/{total_all})",
            "retry50m_pct_short": _pct_display(comb_all, total_all),
            "retry50m_overlay_tasks": len(overlay),
            "retry_gain_pp": retry_gain_pp,
            "rerun_total": len(overlay),
            "rerun_passed": n_flipped,
            "rerun_under_2m": n_stochastic,
            "rerun_over_2m": n_budget,
        }
        # Comparator overall rates on the GPT-5.5 task subset (apples-to-apples with
        # the per-benchmark comparator cells in the gpt55_per_benchmark table).
        for c in comparators:
            csub = comp_runs[c]
            if not len(csub):
                continue
            cn = int(csub["score_binarized"].sum())
            ct = int(len(csub))
            slug = c.replace(" ", "_").replace(".", "_").replace("-", "_").lower()
            gpt55_overall[f"{slug}_pct"] = round(100 * cn / ct, 1)
            gpt55_overall[f"{slug}_pct_short"] = _pct_display(cn, ct)
            gpt55_overall[f"{slug}_display"] = f"{_pct_display(cn, ct)} ({cn}/{ct})"

        # Largest 50M gain benchmark — for prose hooks ("CyberGym shows the largest gain")
        gains = [
            (r["retry50m_n"] - r["pass1_n"], r) for r in gpt55_per_benchmark
        ]
        gains.sort(key=lambda x: -x[0])
        if gains and gains[0][0] > 0:
            top_n, top_row = gains[0]
            gain_pp = round(100 * top_n / top_row["pass1_total"], 0)
            remaining = top_row["retry50m_total"] - top_row["retry50m_n"]
            gpt55_overall["largest_gain_benchmark"] = top_row["name"]
            gpt55_overall["largest_gain_pp"] = int(gain_pp)
            gpt55_overall["largest_gain_remaining_failures"] = remaining

    # --- R-squared for alternative trendline fits (from chart JSON) ---
    alt_fit_stats = {}
    if args.trendline_alt_chart:
        ta_path = _resolve(args.trendline_alt_chart)
        if ta_path.exists():
            with open(ta_path) as f:
                ta_data = json.load(f)
            for panel in ta_data.get("data", {}).get("panels", []):
                r2_val = panel.get("r_squared")
                if r2_val is None:
                    continue
                fit_type = panel.get("fit_type", "")
                zoom = panel.get("zoom", "")
                if "Full" in zoom:
                    alt_fit_stats[f"r2_{fit_type}_full"] = round(r2_val, 2)

    stats = {
        # Task counts
        "headline_tasks": headline_tasks,
        "eval_set_tasks": eval_set_tasks,
        "n_benchmarks": n_benchmarks,
        # Model counts
        "n_models": n_models,
        "n_sota": n_sota,
        "n_non_sota": n_non_sota,
        # Model lists
        "sota_models": sota_models,
        "non_sota_models": non_sota_models,
        "all_models": all_models,
        # Trendline (2019+)
        "doubling_time_days": doubling_time_days,
        "doubling_time_months": doubling_time_months,
        "r_squared": round(r_squared, 2),
        "ci_lower_days": int(ci_lower_days),
        "ci_upper_days": int(ci_upper_days),
        "ci_lower_months": ci_lower_months,
        "ci_upper_months": ci_upper_months,
        # Trendline (2024+)
        "doubling_time_2024_months": doubling_time_2024_months,
        "r_squared_2024": r_squared_2024,
        # Frontier
        "frontier_models": frontier_models,
        "token_budget_10m": token_budget_10m,
        # Token budget accuracy by subset (from chart JSON)
        **token_accuracy,
        "frontier_above_trend": frontier_above_trend,
        "frontier_above_trend_avg": frontier_above_trend_avg,
        # Bootstrap-over-models probabilities
        "prob_model_absent": prob_absent_one,
        "prob_either_frontier_absent": prob_either_absent,
        # Human study
        "n_participants": n_participants,
        "completion_hours": round(comp_hours),
        "estimation_hours": round(est_hours),
        "total_expert_hours": round(total_expert_hours),
        "total_study_tasks": total_study_tasks,
        "n_completion_sessions": n_completion_sessions,
        "n_completion_tasks": n_completion_tasks,
        "n_completion_paired_tasks": n_completion_paired_tasks,
        "n_estimation_tasks": n_estimation_tasks,
        "n_firstblood_tasks": n_firstblood_tasks,
        "n_firstblood_winner_tasks": n_firstblood_winner_tasks,
        "difficulty_min_seconds": difficulty_min_seconds,
        "difficulty_max_hours": difficulty_max_hours,
        "completion_equivalent_hours": completion_equivalent_hours,
        "estimation_efficiency_ratio": estimation_efficiency_ratio,
        "completion_task_pct": round(100 * n_completion_tasks / headline_tasks),
        "completion_hour_pct": round(
            100 * comp_hours / (comp_hours + completion_equivalent_hours)
        ),
        # SOTA span
        "earliest_sota": earliest_sota,
        "earliest_sota_year": earliest_sota_year,
        "latest_sota": latest_sota,
        "latest_sota_year": latest_sota_year,
        "latest_sota_month": latest_sota_month,
        # Adaptation buffer (open-source lag)
        "adaptation_buffer_lower_months": adaptation_buffer_lower_months,
        "adaptation_buffer_upper_months": adaptation_buffer_upper_months,
        "adaptation_buffers": adaptation_buffers,
        # Models table for the blog post
        "models_table": _build_models_table(all_models),
        # Per-model P50 lookup
        "model_p50": model_p50,
        # Model-estimated difficulty P50s (Appendix D)
        "model_p50_model_est": model_p50_model_est,
        "dt_model_est_months": dt_model_est_months,
        # Strong regularisation doubling time
        "dt_strong_reg_months": dt_strong_reg_months,
        # 10M re-run counts
        **ten_m_stats,
        # Token budget 1M-to-2M gain ranges
        **token_gain_stats,
        # Extended budget ratios (2M-5M, 5M-10M)
        **ext_budget_stats,
        # R-squared for alternative trendline fits (full range)
        **alt_fit_stats,
        # GPT-5.5 per-benchmark pass rate (saturation note)
        "gpt55_per_benchmark": gpt55_per_benchmark,
        "gpt55_overall": gpt55_overall,
        # Engagement duration estimates (expert-sourced, from params)
        "pentest_webapp_days": proj_params.get("pentest_webapp_days", "5–8"),
        "pentest_infra_days": proj_params.get("pentest_infra_days", "5–15"),
        "pentest_redteam_days": proj_params.get("pentest_redteam_days", "20+"),
        # Projection timelines (computed from DT + frontier P50)
        **projections,
    }

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Wrote {out_path} ({n_models} models, {headline_tasks} headline tasks, DT={doubling_time_days}d)"
    )


if __name__ == "__main__":
    main()
