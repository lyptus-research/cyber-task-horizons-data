"""Stage: Token budget sensitivity figures.

Two paper figures:

1. token_budget_sensitivity.png - Two panels:
   Left: P50 time horizon vs token budget (all models, rising vs plateau coloring).
   Right: Paired dot plot showing 1M-to-2M P50 gain per model.

2. token_budget_extended_10m.png (optional, requires --10m-samples and --os-cache):
   Same left panel but with GPT-5.3 Codex extended to 10M tokens.
   Right panel: paired dots including extended budget rows for GPT-5.3.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import math
import pickle
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    coerce_floats,
    load_params,
    save_chart_json,
    save_png,
)
from figures.stages._common_data import load_release_dates  # noqa: E402
from lib.data import assemble_runs  # noqa: E402
from lib.irt import fit_p50, p50_vs_token_budget  # noqa: E402
from lib.lyptus_style import COLORS  # noqa: E402

# -- Constants ---------------------------------------------------------------

_EXCLUDE = {"Haiku 4.5", "o1", "GPT-2", "GPT-3", "GPT-3.5"}

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
]

STILL_RISING = {
    "GPT-5.5",
    "Opus 4.6",
    "GPT-5.3 Codex",
    "Sonnet 4.6",
    "GPT-5.2 Codex",
    "GPT-5.1 Codex Max",
    "GLM-5",
}

PLATEAU_COLORS = {
    "o3": "#6db3a8",
    "Opus 4": "#8dc8bf",
    "Gemini 2.5 Pro": "#a8d8d0",
    "DeepSeek V3.1": "#c0c0c0",
    "Claude 3 Opus": "#c8c8c8",
}


# =============================================================================
# Compute: all data loading and statistical computation
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute P50 at each budget for all models, return chart_data dict.

    Returns a dict with keys for both the main figure and optionally the
    extended 10M figure. The dict is fully JSON-serializable.
    """
    # Load data
    model_runs = pd.read_parquet(args.model_runs)
    task_diff = pd.read_parquet(args.task_difficulties)
    runs_df = assemble_runs(model_runs, task_diff, args.difficulty_col)
    campaign_data = {
        alias: {"runs": group.copy()} for alias, group in runs_df.groupby("alias")
    }
    print(f"Loaded {len(campaign_data)} models from pipeline")

    models = [alias for alias in campaign_data if alias not in _EXCLUDE]

    # Precompute P50 at each budget
    all_results = {}
    ratios = {}
    for alias in models:
        data_m = campaign_data.get(alias)
        if data_m is None:
            continue
        all_results[alias] = p50_vs_token_budget(data_m["runs"], budgets=BUDGETS)
        r = all_results[alias]
        p1m = r.loc[r["budget"] == 1_000_000, "p50_minutes"]
        p2m = r.loc[r["budget"] == 2_000_000, "p50_minutes"]
        if len(p1m) > 0 and len(p2m) > 0 and p1m.iloc[0] > 0:
            ratios[alias] = p2m.iloc[0] / p1m.iloc[0]
        else:
            ratios[alias] = 1.0

    # Identify SOTA models from summaries
    summaries_df = pd.read_parquet(args.summaries)
    non_frontier = set(summaries_df[~summaries_df["is_sota"]]["agent"].tolist())
    sota_models = [a for a in models if a in all_results and a not in non_frontier]

    release_dates = load_release_dates()
    dt_per_budget = _compute_dt_per_budget(all_results, sota_models, release_dates)

    print("\nDoubling time by budget:")
    for d in dt_per_budget:
        print(f"  {d['budget_label']:>6s}: {d['dt_months']:.1f} months")

    # Build chart_data for main figure
    chart_models = []
    for alias in models:
        if alias not in all_results:
            continue
        r = all_results[alias]
        p1m = float(r.loc[r["budget"] == 1_000_000, "p50_minutes"].iloc[0])
        p2m = float(r.loc[r["budget"] == 2_000_000, "p50_minutes"].iloc[0])
        chart_models.append(
            {
                "alias": alias,
                "rising": alias in STILL_RISING,
                "budgets_m": [b / 1e6 for b in r["budget"]],
                "p50_minutes": [round(float(v), 2) for v in r["p50_minutes"]],
                "p50_1m": round(p1m, 2),
                "p50_2m": round(p2m, 2),
                "ratio": round(ratios.get(alias, 1.0), 2),
            }
        )

    chart_data = {
        "chart_type": "tokenBudget",
        "version": 1,
        "data": {"models": chart_models, "dt_per_budget": dt_per_budget},
        "options": {"title": "P50 time horizon vs token budget"},
    }

    # Extended 10M figure data (optional)
    extended_data = None
    if args.ten_m_samples and args.os_cache:
        ten_m_path = Path(args.ten_m_samples)
        os_cache_path = Path(args.os_cache)

        if ten_m_path.exists() and os_cache_path.exists():
            with open(ten_m_path, "rb") as f:
                ten_m_data = pickle.load(f)
            with open(os_cache_path, "rb") as f:
                os_cache = pickle.load(f)

            extended_data = _compute_extended(
                models,
                all_results,
                ratios,
                campaign_data,
                os_cache,
                ten_m_data,
                args.gpt55_50m_cache,
            )
        else:
            missing = []
            if not ten_m_path.exists():
                missing.append(str(ten_m_path))
            if not os_cache_path.exists():
                missing.append(str(os_cache_path))
            print(f"Skipping extended data - missing: {', '.join(missing)}")

    # Print summary table
    sorted_models = sorted(
        [(a, ratios.get(a, 1.0)) for a in models if a in all_results],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n{'Model':25s}  {'1M':>8}  {'2M':>8}  {'Ratio':>6}")
    print("-" * 55)
    for alias, ratio in sorted_models:
        r = all_results[alias]
        p1m = r.loc[r["budget"] == 1_000_000, "p50_minutes"].iloc[0]
        p2m = r.loc[r["budget"] == 2_000_000, "p50_minutes"].iloc[0]
        print(f"{alias:25s}  {p1m:7.1f}m  {p2m:7.1f}m  {ratio:5.2f}x")

    return {
        "main": chart_data,
        "extended": extended_data,
    }


def _compute_dt_per_budget(
    all_results: dict[str, pd.DataFrame],
    sota_models: list[str],
    release_dates: dict[str, str],
) -> list[dict]:
    """Compute doubling time at each budget level using METR's fit_trendline."""
    from horizon.plot.logistic import fit_trendline

    dt_per_budget = []
    for b in BUDGETS:
        horizons = []
        dates = []
        for alias in sota_models:
            r = all_results[alias]
            row = r.loc[r["budget"] == b]
            if len(row) == 0:
                continue
            p50_val = float(row["p50_minutes"].iloc[0])
            rd = release_dates.get(alias)
            if p50_val > 0 and rd:
                horizons.append(p50_val)
                dates.append(pd.Timestamp(rd))

        if len(horizons) < 3:
            continue

        reg, _r2 = fit_trendline(
            agent_horizons=pd.Series(horizons),
            release_dates=pd.Series(dates),
            log_scale=True,
        )
        dt_days = np.log(2) / reg.coef_[0] if reg.coef_[0] > 0 else float("inf")
        b_m = b / 1e6
        label = f"{int(b / 1000)}K" if b < 1_000_000 else f"{b / 1_000_000:.0f}M"
        dt_per_budget.append(
            {
                "budget_m": b_m,
                "budget_label": label,
                "dt_months": round(dt_days / 30.44, 1),
            }
        )

    return dt_per_budget


def _compute_extended(
    models: list[str],
    all_results: dict[str, pd.DataFrame],
    ratios: dict[str, float],
    campaign_data: dict[str, dict],
    os_cache: dict,
    ten_m_data: list[dict],
    gpt55_50m_cache: str,
) -> dict | None:
    """Compute extended 10M token budget chart data for GPT-5.3."""
    # Build 10M task lookup
    task_10m = {}
    for s in ten_m_data:
        task_10m[s["task_id"]] = {"score": s["score"], "tokens": s["total_tokens"]}

    # GPT-5.3 runs from os_cache
    gpt53_hd = os_cache["campaign_data"]["GPT-5.3 Codex"]
    runs_ext = gpt53_hd.get("runs_human", gpt53_hd.get("runs"))

    if runs_ext is None or len(runs_ext) == 0:
        print("No GPT-5.3 runs in os_cache - skipping extended data.")
        return None

    log2_vals = runs_ext["log2_human_minutes"].values
    scores_base = runs_ext["score_binarized"].values
    weights = (
        runs_ext["invsqrt_task_weight"].values
        if "invsqrt_task_weight" in runs_ext.columns
        else None
    )

    # Compute P50 at extended budgets for left panel extension (every 1M step)
    ext_budgets_p50 = [b * 1_000_000 for b in range(3, 11)]
    ext_p50 = []
    for b in ext_budgets_p50:
        sc = scores_base.copy()
        for i, (_, row) in enumerate(runs_ext.iterrows()):
            tid = row["task_id"]
            if sc[i] == 0 and tid in task_10m:
                r10 = task_10m[tid]
                if r10["score"] > 0 and r10["tokens"] <= b:
                    sc[i] = 1
        p50_log2, _, _ = fit_p50(log2_vals, sc, weights=weights)
        ext_p50.append(float(2**p50_log2) if np.isfinite(p50_log2) else 0.0)

    # Compute P50 at key budget levels for right panel (every 1M from 1M to 10M)
    p50_key_budgets = [b * 1_000_000 for b in range(1, 11)]
    p50_at = {}
    for b in p50_key_budgets:
        sc = scores_base.copy()
        tokens = (
            runs_ext["total_tokens"].values
            if "total_tokens" in runs_ext.columns
            else np.zeros(len(runs_ext))
        )
        if b <= 2_000_000:
            sc[tokens > b] = 0
        else:
            for i, (_, row) in enumerate(runs_ext.iterrows()):
                if sc[i] == 0 and row["task_id"] in task_10m:
                    r10 = task_10m[row["task_id"]]
                    if r10["score"] > 0 and r10["tokens"] <= b:
                        sc[i] = 1
        p50_log2, _, _ = fit_p50(log2_vals, sc, weights=weights)
        p50_at[b] = float(2**p50_log2) if np.isfinite(p50_log2) else 0.0

    # Build chart_models for JSON
    chart_models = []
    for alias in models:
        if alias not in all_results:
            continue
        r = all_results[alias]
        p1m = float(r.loc[r["budget"] == 1_000_000, "p50_minutes"].iloc[0])
        p2m = float(r.loc[r["budget"] == 2_000_000, "p50_minutes"].iloc[0])
        chart_models.append(
            {
                "alias": alias,
                "rising": alias in STILL_RISING,
                "budgets_m": [b / 1e6 for b in r["budget"]],
                "p50_minutes": [round(float(v), 2) for v in r["p50_minutes"]],
                "p50_1m": round(p1m, 2),
                "p50_2m": round(p2m, 2),
                "ratio": round(ratios.get(alias, 1.0), 2),
            }
        )

    # Build doubling rows - two extended budget buckets for right panel
    doubling_rows = []
    for budget_label, b_lo, b_hi in [
        ("5M\u219210M", 5_000_000, 10_000_000),
        ("2M\u21925M", 2_000_000, 5_000_000),
    ]:
        p_lo = float(p50_at[b_lo])
        p_hi = float(p50_at[b_hi])
        doubling_rows.append(
            {
                "alias": "GPT 5.3",
                "budget_label": budget_label,
                "p50_lo": round(p_lo, 2),
                "p50_hi": round(p_hi, 2),
                "ratio": round(p_hi / p_lo, 2) if p_lo > 0 else 1.0,
                "is_extended": True,
            }
        )

    gpt55_ext = _compute_gpt55_50m_extension(
        campaign_data.get("GPT-5.5", {}).get("runs"),
        all_results,
        gpt55_50m_cache,
    )
    if gpt55_ext is not None:
        doubling_rows.insert(
            0,
            {
                "alias": "GPT-5.5",
                "budget_label": "50M retry overlay",
                "p50_lo": gpt55_ext["last_2m_p50"],
                "p50_hi": gpt55_ext["p50_50m"],
                "ratio": gpt55_ext["ratio_2m_50m"],
                "is_extended": True,
            },
        )

    for cm in sorted(chart_models, key=lambda x: x["ratio"], reverse=True):
        if cm["alias"] in {"GPT-5.5"}:
            continue
        doubling_rows.append(
            {
                "alias": cm["alias"],
                "budget_label": "1M\u21922M",
                "p50_lo": cm["p50_1m"],
                "p50_hi": cm["p50_2m"],
                "ratio": cm["ratio"],
                "is_extended": False,
            }
        )

    ext_bm = [b / 1e6 for b in ext_budgets_p50]
    last_2m_p50 = float(all_results["GPT-5.3 Codex"]["p50_minutes"].iloc[-1])

    # Per-million P50 lookup for stats/analysis
    p50_per_million = {
        f"{b // 1_000_000}M": round(float(p50_at[b]), 2) for b in p50_key_budgets
    }

    return {
        "chart_type": "tokenBudgetExtended",
        "version": 1,
        "data": {
            "models": chart_models,
            "gpt53_extended": {
                "budgets_m": ext_bm,
                "p50_minutes": [round(v, 2) for v in ext_p50],
                "last_2m_p50": round(last_2m_p50, 2),
                "label_10m": f"GPT-5.3 at 10M ({ext_p50[-1] / 60:.1f}h)",
                "p50_per_million": p50_per_million,
            },
            "gpt55_extended": gpt55_ext,
            "doubling_rows": doubling_rows,
        },
        "options": {"title": "P50 time horizon vs token budget"},
    }


def _compute_gpt55_50m_extension(
    runs_ext: pd.DataFrame | None,
    all_results: dict[str, pd.DataFrame],
    cache_path: str,
) -> dict | None:
    if runs_ext is None or len(runs_ext) == 0 or "GPT-5.5" not in all_results:
        return None

    path = Path(cache_path)
    if not path.is_absolute():
        path = _NOTEBOOKS_DIR / path
    if not path.exists():
        print(f"No GPT-5.5 50M cache at {path} - skipping 50M extension.")
        return None

    reruns = json.loads(path.read_text())
    log2_vals = runs_ext["log2_human_minutes"].values
    scores_base = runs_ext["score_binarized"].values
    weights = (
        runs_ext["invsqrt_task_weight"].values
        if "invsqrt_task_weight" in runs_ext.columns
        else None
    )
    budgets = [
        2_250_000, 2_500_000, 2_750_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000,
    ]
    p50_at = {}

    for b in budgets:
        sc = scores_base.copy()
        for i, (_, row) in enumerate(runs_ext.iterrows()):
            if sc[i] == 0 and row["task_id"] in reruns:
                rr = reruns[row["task_id"]]
                if rr["score"] >= 0.7 and rr["tokens"] <= b:
                    sc[i] = 1
        p50_log2, _, _ = fit_p50(log2_vals, sc, weights=weights)
        p50_at[b] = float(2**p50_log2) if np.isfinite(p50_log2) else 0.0

    last_2m = float(all_results["GPT-5.5"]["p50_minutes"].iloc[-1])
    p50_50m = p50_at[50_000_000]
    return {
        "budgets_m": [b / 1e6 for b in budgets],
        "p50_minutes": [round(p50_at[b], 2) for b in budgets],
        "last_2m_p50": round(last_2m, 2),
        "p50_50m": round(p50_50m, 2),
        "ratio_2m_50m": round(p50_50m / last_2m, 2) if last_2m > 0 else 1.0,
    }


# =============================================================================
# Render: matplotlib figures from chart_data dict (no DataFrames)
# =============================================================================


def _get_color(alias: str) -> str:
    if alias == "GPT-5.5":
        return COLORS["coral"]
    if alias in STILL_RISING:
        if alias in {"Opus 4.6", "GPT-5.3 Codex"}:
            return COLORS["teal_dark"]
        return COLORS["teal_light"]
    return PLATEAU_COLORS.get(alias, "#c0c0c0")


def _model_line_params(alias: str) -> dict:
    if alias == "GPT-5.5":
        return {"linewidth": 3.5, "markersize": 7, "zorder": 7}
    if alias in STILL_RISING:
        lw = 3 if alias in {"Opus 4.6", "GPT-5.3 Codex"} else 2
        ms = 6 if alias in {"Opus 4.6", "GPT-5.3 Codex"} else 4
        return {"linewidth": lw, "markersize": ms, "zorder": 5}
    return {"linewidth": 1.5, "markersize": 3, "zorder": 2}


def _place_endpoint_labels(ax, eps: list[tuple[str, float]]) -> None:
    """Add non-overlapping model labels at the 2M endpoint."""
    eps_sorted = sorted(eps, key=lambda x: x[1], reverse=True)
    placed = []
    min_gap = 0.22

    for alias, p50 in eps_sorted:
        log_p50 = math.log2(max(p50, 0.1))
        offset = 0
        for _, pl in placed:
            if abs(log_p50 + offset - pl) < min_gap:
                offset -= min_gap * 1.3
        placed.append((alias, log_p50 + offset))

        color = COLORS["teal"] if alias in STILL_RISING else "#999"
        fs = 8.5 if alias in {"GPT-5.5", "Opus 4.6", "GPT-5.3 Codex"} else 7.5
        fw = (
            "bold"
            if alias in {"GPT-5.5", "Opus 4.6", "GPT-5.3 Codex"}
            else "normal"
        )
        display_y = 2 ** (log_p50 + offset)
        ax.annotate(
            alias,
            (2.0, display_y),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=fs,
            fontweight=fw,
            color=color,
            va="center",
            zorder=6,
        )


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render the main token_budget_sensitivity figure from chart_data."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    main_data = chart_data["main"]
    models_data = main_data["data"]["models"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ---- Left panel: P50 vs token budget lines ----

    # Plateaued models (grey background)
    for m in models_data:
        alias = m["alias"]
        if alias in STILL_RISING:
            continue
        bm = coerce_floats(m["budgets_m"])
        p50 = coerce_floats(m["p50_minutes"])
        ax1.plot(
            bm, p50, marker=".", color="#d0d0d0", linewidth=1.5, markersize=3, zorder=2
        )

    # Rising models (foreground)
    for m in models_data:
        alias = m["alias"]
        if alias not in STILL_RISING:
            continue
        bm = coerce_floats(m["budgets_m"])
        p50 = coerce_floats(m["p50_minutes"])
        c = _get_color(alias)
        p = _model_line_params(alias)
        ax1.plot(bm, p50, marker="o", color=c, **p)

    # Endpoint labels
    eps = [(m["alias"], m["p50_2m"]) for m in models_data]
    _place_endpoint_labels(ax1, eps)

    # Highlight the 1M-2M region
    ax1.axvspan(1.0, 2.0, color=COLORS["teal"], alpha=0.07, zorder=0)

    ax1.set_xlabel("Token budget")
    ax1.set_ylabel("P50 time horizon")
    ax1.set_title("P50 time horizon vs token budget")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    ax1.set_xticklabels(["50K", "100K", "200K", "500K", "1M", "2M"])
    ax1.set_yticks([1, 2, 5, 10, 30, 60, 120, 240, 720])
    ax1.set_yticklabels(["1m", "2m", "5m", "10m", "30m", "1h", "2h", "4h", "12h"])
    ax1.grid(alpha=0.2)

    # ---- Right panel: Paired dot plot (1M vs 2M) ----

    sorted_models = [
        m
        for m in sorted(models_data, key=lambda m: m["ratio"], reverse=True)
        if m["ratio"] >= 1.15 or m["alias"] == "GPT-5.5"
    ]

    for i, m in enumerate(sorted_models):
        alias = m["alias"]
        p1m = m["p50_1m"]
        p2m = m["p50_2m"]
        ratio = m["ratio"]
        color = (
            COLORS["coral"]
            if alias == "GPT-5.5"
            else COLORS["teal"]
            if ratio >= 1.15
            else "#c0c0c0"
        )
        y = len(sorted_models) - i - 1

        ax2.plot([p1m, p2m], [y, y], color=color, linewidth=2.5, zorder=3)
        ax2.plot(p1m, y, "o", color=color, markersize=8, zorder=4)
        ax2.plot(p2m, y, "s", color=color, markersize=8, zorder=4)

        ratio_str = f"{ratio:.1f}x" if ratio >= 1.05 else "flat"
        label = f"{alias}  {ratio_str}"
        ax2.text(
            max(p1m, p2m) * 1.15,
            y,
            label,
            va="center",
            fontsize=8.5,
            color=color,
            fontweight="bold" if ratio > 1.4 or alias == "GPT-5.5" else "normal",
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("P50 time horizon")
    ax2.set_title("P50 gain from 1M to 2M tokens")
    ax2.set_xticks([1, 5, 10, 30, 60, 120, 240, 720, 1440])
    ax2.set_xticklabels(["1m", "5m", "10m", "30m", "1h", "2h", "4h", "12h", "1d"])
    ax2.set_yticks([])
    ax2.grid(alpha=0.2, axis="x")

    dot_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#777",
            markeredgecolor="#777",
            markersize=8,
            label="P50 at 1M tokens",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#777",
            markeredgecolor="#777",
            markersize=8,
            label="P50 at 2M tokens",
        ),
    ]
    ax2.legend(
        handles=dot_legend,
        fontsize=8,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#ddd",
    )

    plt.tight_layout()
    save_png(fig, output, params)


def render_extended_png(chart_data: dict, output: str, params: dict) -> None:
    """Render the extended 10M token budget figure from chart_data."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ext_data = chart_data["extended"]
    if ext_data is None:
        print("No extended data - skipping extended PNG.")
        return

    models_data = ext_data["data"]["models"]
    gpt53_ext = ext_data["data"]["gpt53_extended"]
    gpt55_ext = ext_data["data"].get("gpt55_extended")
    doubling_rows = ext_data["data"]["doubling_rows"]
    y_cap_minutes = 1_440.0
    x_cap_minutes = 2_880.0

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ---- Left panel: all models to 2M, GPT-5.3 to 10M ----

    for m in models_data:
        alias = m["alias"]
        bm = coerce_floats(m["budgets_m"])
        p50 = coerce_floats(m["p50_minutes"])
        c = _get_color(alias)
        p = _model_line_params(alias)
        marker = "o" if alias in STILL_RISING else "."

        if alias == "GPT-5.3 Codex":
            ax_l.plot(bm, p50, marker="o", color=c, **p)
            # Solid extension from 2M to 10M
            last_2m_p50 = gpt53_ext["last_2m_p50"]
            ext_bm = coerce_floats(gpt53_ext["budgets_m"])
            ext_p50 = coerce_floats(gpt53_ext["p50_minutes"])
            ax_l.plot(
                [2.0] + ext_bm,
                [last_2m_p50] + ext_p50,
                marker="s",
                color=COLORS["teal_dark"],
                linewidth=2.5,
                markersize=6,
                linestyle="--",
                zorder=5,
            )
            if ext_p50:
                ax_l.annotate(
                    f"GPT-5.3 at 10M\n({ext_p50[-1] / 60:.1f}h)",
                    (10.0, ext_p50[-1]),
                    textcoords="offset points",
                    xytext=(6, 0),
                    fontsize=8.5,
                    fontweight="bold",
                    color=COLORS["teal_dark"],
                    va="center",
                    zorder=6,
                )
        elif alias == "GPT-5.5":
            ax_l.plot(bm, p50, marker="o", color=c, **p)
            if gpt55_ext is not None:
                ext_bm = coerce_floats(gpt55_ext["budgets_m"])
                ext_p50 = coerce_floats(gpt55_ext["p50_minutes"])
                ext_p50_display = [min(v, y_cap_minutes) for v in ext_p50]
                ax_l.plot(
                    [2.0] + ext_bm,
                    [gpt55_ext["last_2m_p50"]] + ext_p50_display,
                    marker="s",
                    color=COLORS["coral"],
                    linewidth=3,
                    markersize=6,
                    linestyle="-",
                    zorder=7,
                )

                # Highlight 2M / 10M / 50M budget milestones explicitly.
                # 2M is annotated by the endpoint label - skip here to avoid
                # collision with adjacent Opus 4.6 / GPT-5.3 Codex labels.
                milestone_bm = {2.0: "2M", 10.0: "10M", 50.0: "50M"}
                ext_lookup = dict(zip(ext_bm, ext_p50))
                ext_lookup[2.0] = gpt55_ext["last_2m_p50"]
                for mb, label in milestone_bm.items():
                    if mb not in ext_lookup:
                        continue
                    p50_at = ext_lookup[mb]
                    p50_display = min(p50_at, y_cap_minutes)
                    # Larger ring marker on every milestone
                    ax_l.plot(
                        mb,
                        p50_display,
                        marker="o",
                        markersize=14,
                        markerfacecolor="none",
                        markeredgecolor=COLORS["coral"],
                        markeredgewidth=2,
                        zorder=9,
                    )
                    if mb == 2.0:
                        continue  # endpoint label already shows GPT-5.5 at 2M
                    if p50_at >= y_cap_minutes:
                        value_text = f"@ {label}: ≥1d ({p50_at / 60:.0f}h IRT)"
                    elif p50_at >= 60:
                        value_text = f"@ {label}: {p50_at / 60:.1f}h"
                    else:
                        value_text = f"@ {label}: {p50_at:.0f}m"
                    y_off = {10.0: -22, 50.0: 14}[mb]
                    x_off = {10.0: 0, 50.0: -6}[mb]
                    ha = {10.0: "center", 50.0: "right"}[mb]
                    ax_l.annotate(
                        value_text,
                        (mb, p50_display),
                        textcoords="offset points",
                        xytext=(x_off, y_off),
                        fontsize=8.5,
                        fontweight="bold",
                        color=COLORS["coral"],
                        ha=ha,
                        zorder=10,
                    )
        else:
            ax_l.plot(bm, p50, marker=marker, color=c, **p)

    # Endpoint labels
    eps = [(m["alias"], m["p50_2m"]) for m in models_data]
    _place_endpoint_labels(ax_l, eps)

    ax_l.axvspan(1.0, 2.0, color=COLORS["teal"], alpha=0.07, zorder=0)
    if gpt55_ext is not None:
        ax_l.axvspan(2.0, 50.0, color=COLORS["coral"], alpha=0.04, zorder=0)
    ax_l.set_xlabel("Token budget")
    ax_l.set_ylabel("P50 time horizon")
    ax_l.set_title("P50 time horizon vs token budget")
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    ax_l.set_xticklabels(["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M", "20M", "50M"])
    ax_l.set_yticks([1, 2, 5, 10, 30, 60, 120, 240, 360, 720, 1440, 2880])
    ax_l.set_yticklabels(["1m", "2m", "5m", "10m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "2d"])
    ax_l.set_ylim(1, y_cap_minutes * 1.2)
    ax_l.grid(alpha=0.2)

    # ---- Right panel: paired dots with GPT-5.3 extended rows ----

    visible_rows = [
        dr
        for dr in doubling_rows
        if dr["is_extended"]
        or dr["alias"]
        in {
            "GPT-5.3 Codex",
            "Opus 4.6",
            "Sonnet 4.6",
            "GPT-5.2 Codex",
            "GPT-5.1 Codex Max",
            "GLM-5",
        }
    ]

    for i, dr in enumerate(visible_rows):
        y = len(visible_rows) - i - 1
        p_lo = dr["p50_lo"]
        p_hi = dr["p50_hi"]
        p_hi_display = min(p_hi, x_cap_minutes)
        ratio = dr["ratio"]
        is_extended = dr["is_extended"]

        if is_extended and dr["alias"] == "GPT-5.5":
            color = COLORS["coral"]
        elif is_extended:
            color = COLORS["teal_dark"]
        elif dr["alias"] == "GPT-5.5":
            color = COLORS["coral"]
        elif ratio >= 1.15:
            color = COLORS["teal"]
        else:
            color = "#c0c0c0"

        ax_r.plot([p_lo, p_hi_display], [y, y], color=color, linewidth=2.5, zorder=3)
        ax_r.plot(p_lo, y, "o", color=color, markersize=8, zorder=4)
        ax_r.plot(p_hi_display, y, "s", color=color, markersize=8, zorder=4)

        ratio_str = f"{ratio:.1f}x" if ratio >= 1.05 else "flat"
        offscale = p_hi > x_cap_minutes
        label_extra = " (off-scale)" if offscale else ""
        is_gpt55_sat = dr["alias"] == "GPT-5.5" and is_extended
        label_text = (
            f"{dr['alias']}  saturated{label_extra}"
            if is_gpt55_sat
            else f"{dr['alias']}  {ratio_str}{label_extra}"
        )
        if is_gpt55_sat:
            # Place to the LEFT of the line endpoint with right-anchored text
            # so the saturated label always fits inside the chart.
            label_x = p_lo / 1.15
            label_ha = "right"
        else:
            label_x = min(max(p_lo, p_hi_display) * 1.15, x_cap_minutes * 0.82)
            label_ha = "left"
        ax_r.text(
            label_x,
            y,
            label_text,
            va="center",
            ha=label_ha,
            fontsize=8.5,
            color=color,
            fontweight=(
                "bold" if ratio > 1.4 or is_extended or dr["alias"] == "GPT-5.5" else "normal"
            ),
        )

        # Budget range label on prominent lines
        if is_extended or dr["alias"] in {"GPT-5.3 Codex", "Opus 4.6"}:
            budget_label = dr.get("budget_label", "")
            if budget_label:
                mid_x = (
                    (p_lo * p_hi_display) ** 0.5
                    if p_lo > 0 and p_hi_display > 0
                    else p_hi_display
                )
                ax_r.text(
                    mid_x,
                    y + 0.3,
                    budget_label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    alpha=0.8,
                )

    # Separator between extended and 1M-2M rows
    n_ext = sum(1 for dr in visible_rows if dr["is_extended"])
    sep_y = len(visible_rows) - n_ext - 0.5
    ax_r.axhline(sep_y, color="#ddd", linewidth=1, linestyle="-")

    ax_r.set_xscale("log")
    ax_r.set_xlabel("P50 time horizon")
    ax_r.set_title("P50 gain by budget slice")
    ax_r.set_xticks([1, 5, 10, 30, 60, 120, 240, 360, 720, 1440, 2880])
    ax_r.set_xticklabels(["1m", "5m", "10m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "2d"])
    ax_r.set_xlim(0.8, x_cap_minutes)
    ax_r.set_yticks([])
    ax_r.grid(alpha=0.2, axis="x")

    dot_leg = [
        Line2D(
            [0],
            [0],
            color=COLORS["teal"],
            linewidth=2.5,
            label="1M\u21922M (all models)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["teal_dark"],
            linewidth=2.5,
            label="GPT-5.3 extended budget",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["coral"],
            linewidth=2.5,
            label="GPT-5.5 50M retry overlay",
        ),
    ]
    ax_r.legend(
        handles=dot_leg,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Token budget sensitivity figures")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet for SOTA identification",
    )
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument(
        "--10m-samples",
        dest="ten_m_samples",
        default=None,
        help="Pickle cache of 10M token re-run samples",
    )
    parser.add_argument(
        "--os-cache",
        default=None,
        help="Pickle cache with campaign_data for GPT-5.3 extended runs",
    )
    parser.add_argument(
        "--output-extended",
        default=None,
        help="Output path for the extended 10M figure",
    )
    parser.add_argument(
        "--gpt55-50m-cache",
        default="../data/keep/gpt55_50m_reruns.json",
        help="JSON cache of GPT-5.5 50M retry results",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)

    # Save main chart JSON and render main PNG
    save_chart_json(chart_data["main"], args.output)
    render_png(chart_data, args.output, params)

    # Extended figure (optional)
    if args.output_extended and chart_data["extended"] is not None:
        save_chart_json(chart_data["extended"], args.output_extended)
        render_extended_png(chart_data, args.output_extended, params)
    elif args.output_extended:
        print("Skipping extended figure - no extended data computed")


if __name__ == "__main__":
    main()
