"""Stage: Side-by-side IRT S-curve showing regularisation sensitivity.

Shows a frontier model's S-curve at strong vs weak regularisation,
using the same style as the main IRT grid (lib/irt.py:plot_scurve_grid).

Stats JSON (--stats-output) reports per-model P50 at each regularisation
level. For the target model's two plotted panels, the P50 values are
captured directly from plot_scurve's return value (same fit_p50 call
that drives the rendered curve). For all other models, fit_p50 is called
once per model per lambda, which is the same code path plot_scurve uses
internally.

Architecture: compute() builds chart_data dict, save_chart_json() writes it,
render_png() reads from the dict to produce matplotlib. The chart JSON is
the single source of truth for both the PNG and the interactive Plotly chart.
"""

import json
import sys
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
from lib.irt import fit_p50, compute_scurve_data, _TICK_LOG2, _TICK_LABELS  # noqa: E402

# Regularisation levels to sweep for stats output
_REGULARISATIONS = [0.2, 0.1, 0.02, 0.01, 0.001, 0.00001]
_LAMBDA_STRONG = 0.2
_LAMBDA_HEADLINE = 0.00001


def _reg_key(reg: float) -> str:
    return f"lambda_{reg:.10f}".rstrip("0").rstrip(".")


# =============================================================================
# Compute: all statistical work, returns JSON-serializable dict
# =============================================================================


def compute(args, params) -> dict:
    """Load data, compute IRT curves at two regularisation levels, return chart_data."""
    from lib.data import assemble_runs as _assemble

    mr = pd.read_parquet(args.model_runs)
    td = pd.read_parquet(args.task_difficulties)
    runs_df = _assemble(mr, td, args.difficulty_col)
    model_runs = runs_df[runs_df["alias"] == args.model]
    n_tasks = model_runs["task_id"].nunique()

    # Compute shared bin edges
    all_log2 = model_runs["log2_human_minutes"].values
    global_lo = float(np.floor(all_log2.min()) - 0.5)
    global_hi = float(np.ceil(all_log2.max()) + 1.5)
    global_bin_edges = np.arange(global_lo, global_hi, 1.0)

    # Compute panel data for both regularisation levels
    chart_panels = [
        ("\u03bb = 0.2  (strong regularisation)", _LAMBDA_STRONG),
        ("\u03bb = 0.00001  (headline)", _LAMBDA_HEADLINE),
    ]
    chart_models = []
    for label, reg in chart_panels:
        panel_data = compute_scurve_data(
            model_runs, global_bin_edges, regularization=reg
        )
        p50_min = 2 ** panel_data["p50_log2"]
        chart_models.append(
            {
                "alias": f"{args.model} \u2014 {label} (P50 = {p50_min:.0f}m)",
                "provider": "comparison",
                "release_date": "",
                "release_year": "",
                **panel_data,
            }
        )

    chart_data = {
        "chart_type": "irtScurve",
        "version": 1,
        "data": {
            "models": chart_models,
            "global_bin_edges": [round(float(e), 2) for e in global_bin_edges],
            "target_model": args.model,
            "n_tasks": n_tasks,
        },
        "options": {
            "title": f"{args.model} \u2014 regularisation sensitivity ({n_tasks} tasks)",
            "defaultModels": [m["alias"] for m in chart_models],
            "ncols": 2,
            "noControls": True,
        },
    }

    # --- Stats JSON (side output, not part of chart_data) ---
    if args.stats_output:
        _write_stats(args, runs_df, model_runs, chart_panels, chart_models)

    return chart_data


def _write_stats(args, runs_df, model_runs, chart_panels, chart_models):
    """Write per-model regularisation sensitivity stats to JSON."""
    # Capture P50 values from the two plotted panels
    plot_results = {}
    for (label, reg), cm in zip(chart_panels, chart_models):
        p50_min = 2 ** cm["p50_log2"]
        plot_results[reg] = {"p50_min": p50_min}

    per_model = {}
    for alias, group in runs_df.groupby("alias"):
        log2_t = group["log2_human_minutes"].values
        scores = group["score_binarized"].values
        w = (
            group["invsqrt_task_weight"].values
            if "invsqrt_task_weight" in group.columns
            else None
        )

        p50s = {}
        for reg in _REGULARISATIONS:
            # For target model at plotted lambdas, use captured values
            if alias == args.model and reg in plot_results:
                p50_min = plot_results[reg]["p50_min"]
            else:
                p50_log2, _, _ = fit_p50(log2_t, scores, weights=w, regularization=reg)
                p50_min = 2**p50_log2
            p50s[_reg_key(reg)] = round(float(p50_min), 1) if p50_min > 0 else 0.0

        p50_strong = p50s.get(_reg_key(_LAMBDA_STRONG), 0)
        p50_headline = p50s.get(_reg_key(_LAMBDA_HEADLINE), 0)
        ratio = round(p50_strong / p50_headline, 1) if p50_headline > 0 else None
        n_pass = int(group["score_binarized"].sum())
        n_total = len(group.drop_duplicates("task_id"))

        per_model[alias] = {
            "p50_strong": p50_strong,
            "p50_headline": p50_headline,
            "ratio": ratio,
            "n_pass": n_pass,
            "n_tasks": n_total,
            "pass_rate_pct": round(100 * n_pass / len(group)) if len(group) > 0 else 0,
            **p50s,  # per-lambda P50s for dt_by_lambda computation
        }

    target = per_model.get(args.model, {})

    # Models with minimal sensitivity (ratio close to 1.0)
    stable_models = [
        m
        for m, d in per_model.items()
        if d.get("ratio") and 0.8 <= d["ratio"] <= 1.2 and d["p50_headline"] > 1
    ]

    gpt2 = per_model.get("GPT-2", {})

    # DT at each regularisation (trendline fit on SOTA models)
    from lib.trendline import RELEASE_DATES
    from sklearn.linear_model import LinearRegression
    from matplotlib.dates import date2num

    sum_df = pd.read_parquet(args.summaries)
    non_frontier = set(sum_df[~sum_df["is_sota"]]["agent"].tolist())
    sota_models_list = [m for m in per_model if m not in non_frontier]

    dt_by_lambda = {}
    for reg in _REGULARISATIONS:
        p50_vals = []
        dates = []
        for m in sota_models_list:
            p50_val = per_model[m].get(_reg_key(reg), 0)
            rd = RELEASE_DATES.get(m)
            if p50_val > 0 and rd:
                p50_vals.append(p50_val)
                dates.append(date2num(pd.Timestamp(rd)))
        if len(p50_vals) >= 2:
            X = np.array(dates).reshape(-1, 1)
            y = np.log(np.array(p50_vals))
            lr = LinearRegression().fit(X, y)
            dt_days = np.log(2) / lr.coef_[0]
            dt_by_lambda[_reg_key(reg)] = round(dt_days / 30.44, 1)

    stats = {
        "target_model": args.model,
        "target": target,
        "gpt2": gpt2,
        "n_stable_models": len(stable_models),
        "stable_range_first": stable_models[0] if stable_models else None,
        "stable_range_last": stable_models[-1] if stable_models else None,
        "dt_by_lambda": dt_by_lambda,
        "per_model": per_model,
    }

    out_path = (
        Path(args.stats_output)
        if Path(args.stats_output).is_absolute()
        else _NOTEBOOKS_DIR / args.stats_output
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats: {out_path}")


# =============================================================================
# Render: matplotlib figure from chart_data dict (no DataFrames, no numpy)
# =============================================================================


def render_png(chart_data: dict, output: str, params: dict) -> None:
    """Render regularisation sensitivity S-curves from chart JSON data."""
    models = chart_data["data"]["models"]
    global_bin_edges = chart_data["data"]["global_bin_edges"]
    n_tasks = chart_data["data"].get("n_tasks", "")
    target_model = chart_data["data"].get("target_model", "")

    try:
        from lib.lyptus_style import FONT_SANS
    except (ImportError, KeyError):
        FONT_SANS = "Helvetica Neue"

    global_lo = global_bin_edges[0]
    global_hi = global_bin_edges[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, model in zip([ax1, ax2], models):
        bin_centers = model["bin_centers"]
        empirical_rates = model["empirical_rates"]
        bin_counts = model["bin_counts"]
        standard_errors = model.get("standard_errors", [None] * len(bin_centers))
        curve_x = model["curve_x"]
        curve_y = model["curve_y"]
        p50_label = model.get("p50_label", "")
        min_n = model.get("min_n", 5)
        coef = model.get("coef", 0)
        p50_log2 = model.get("p50_log2", 0)
        alias = model["alias"]

        # Extract the panel label (after the em dash)
        # alias format: "Model — label (P50 = Xm)"
        if "\u2014" in alias:
            label = alias.split("\u2014", 1)[1].strip()
            # Remove trailing P50 annotation for axis title
            if "(P50" in label:
                label = label[: label.index("(P50")].strip()
        else:
            label = alias

        # Draw empirical rate bars
        for i, (center, rate, count) in enumerate(
            zip(bin_centers, empirical_rates, bin_counts)
        ):
            if rate is None:
                continue
            alpha = 0.3 if count <= min_n else 0.55
            ax.bar(
                center,
                rate * 100,
                width=0.8,
                color="#2a9d8f",
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5,
            )
            se = standard_errors[i] if i < len(standard_errors) else None
            if se is not None and count > min_n:
                ax.errorbar(
                    center,
                    rate * 100,
                    yerr=se * 100,
                    fmt="none",
                    ecolor="#264653",
                    capsize=3,
                    capthick=1,
                    alpha=0.5,
                )

        # Draw S-curve
        ax.plot(curve_x, curve_y, color="#e76f51", linewidth=2.5, zorder=4)

        # P50 vertical line
        if coef and float(coef) != float("-inf") and p50_log2 is not None:
            ax.axvline(
                p50_log2, color="#e76f51", linestyle="--", alpha=0.6, linewidth=1
            )

        # Title with P50 label
        title_text = f"\u03bb = {label}"
        if p50_label:
            title_text = f"{label}\nP50 = {p50_label}"
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(axis="both", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared formatting
    visible_ticks = [
        (pos, label)
        for pos, label in zip(_TICK_LOG2, _TICK_LABELS)
        if global_lo <= pos <= global_hi
    ]
    if visible_ticks:
        tick_positions, tick_labels_list = zip(*visible_ticks)
        for ax in [ax1, ax2]:
            ax.set_xlim(global_lo, global_hi)
            ax.set_xticks(list(tick_positions))
            ax.set_xticklabels(
                list(tick_labels_list), fontsize=10, fontfamily=FONT_SANS
            )

    ax1.set_ylabel("Success rate (%)", fontsize=11, fontfamily=FONT_SANS)
    ax2.set_ylabel("")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Human time (log\u2082 scale)", fontsize=10, fontfamily=FONT_SANS)

    fig.suptitle(
        f"{target_model} \u2014 regularisation sensitivity ({n_tasks} tasks)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_png(fig, output, params)


# =============================================================================
# Main: compute -> serialize -> render
# =============================================================================


def main():
    parser = base_parser("Plot regularisation sensitivity S-curves")
    parser.add_argument("--model-runs", required=True, help="model_runs.parquet")
    parser.add_argument(
        "--task-difficulties", required=True, help="task_difficulties.parquet"
    )
    parser.add_argument("--difficulty-col", default="best_available_minutes")
    parser.add_argument(
        "--summaries",
        required=True,
        help="model_summaries parquet (for data-driven SOTA)",
    )
    parser.add_argument("--model", default="GPT-5.3 Codex")
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Output JSON with per-model regularisation sensitivity",
    )
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args, params)
    save_chart_json(chart_data, args.output)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
