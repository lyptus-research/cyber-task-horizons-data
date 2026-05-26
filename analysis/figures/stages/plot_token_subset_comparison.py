"""Compare token-budget scaling for two model subset-analysis charts."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from figures.stages._common import (  # noqa: E402
    base_parser,
    load_json,
    load_params,
    save_chart_json,
    save_png,
)
from lib.lyptus_style import COLORS  # noqa: E402


SUBSET_COLORS = {
    "all": COLORS["teal_dark"],
    "cybergym": COLORS["coral"],
    "hard_2h": COLORS["gold"],
}

MODEL_STYLES = {
    "GPT-5.3 Codex": {"linestyle": "--", "alpha": 0.32, "marker": "o"},
    "GPT-5.5": {"linestyle": "-", "alpha": 1.0, "marker": "o"},
}

INCLUDED_SUBSETS = {"all", "cybergym", "hard_2h"}


def compute(args) -> dict:
    charts = [load_json(path) for path in args.charts]
    return {
        "chart_type": "tokenSubsetComparison",
        "version": 1,
        "data": {"models": [chart["data"] for chart in charts]},
        "options": {
            "title": "Token budget scaling by task subset",
            "model_labels": [chart["data"]["model"] for chart in charts],
        },
    }


def _point_series(subset: dict, key: str, max_budget_m: float | None) -> tuple[list, list]:
    points = subset["points"]
    if max_budget_m is not None:
        points = [p for p in points if p["budget_m"] <= max_budget_m]
    return [p["budget_m"] for p in points], [p[key] for p in points]


def render_png(chart_data: dict, output: str, params: dict) -> None:
    models = chart_data["data"]["models"]

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(18, 7))

    max_cost = 0
    for model_data in models:
        model = model_data["model"]
        style = MODEL_STYLES.get(model, {"linestyle": "-", "alpha": 0.9, "marker": "o"})
        max_budget_m = 10.0 if model == "GPT-5.3 Codex" else None

        for subset in model_data["subsets"]:
            if subset["key"] not in INCLUDED_SUBSETS:
                continue
            color = SUBSET_COLORS.get(subset["key"], "#999999")
            label = f"{model} - {subset['label']}"

            x_acc, y_acc = _point_series(subset, "accuracy", max_budget_m)
            ax_acc.plot(
                x_acc,
                y_acc,
                color=color,
                linewidth=2.4,
                markersize=4.8,
                label=label,
                **style,
            )

            x_cost, y_cost_raw = _point_series(subset, "cost_per_success", max_budget_m)
            y_cost = [v for v in y_cost_raw if v is not None]
            if y_cost:
                max_cost = max(max_cost, max(y_cost))
                ax_cost.plot(
                    x_cost,
                    y_cost_raw,
                    color=color,
                    linewidth=2.4,
                    markersize=4.8,
                    label=label,
                    **style,
                )

    for ax in (ax_acc, ax_cost):
        ax.axvline(2.0, color=COLORS["coral"], linestyle=":", alpha=0.55, linewidth=1)
        ax.text(
            2.0,
            0.03,
            "2M",
            transform=ax.get_xaxis_transform(),
            color=COLORS["coral"],
            fontsize=8,
            alpha=0.75,
            ha="right",
            va="bottom",
            rotation=90,
        )
        ax.axvline(50.0, color=COLORS["teal_dark"], linestyle=":", alpha=0.55, linewidth=1)
        ax.text(
            50.0,
            0.03,
            "50M",
            transform=ax.get_xaxis_transform(),
            color=COLORS["teal_dark"],
            fontsize=8,
            alpha=0.75,
            ha="right",
            va="bottom",
            rotation=90,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Token budget")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(
            ["50K", "100K", "200K", "500K", "1M", "2M", "5M", "10M", "20M", "50M"]
        )
        ax.grid(alpha=0.2)

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy by task subset")
    ax_acc.set_ylim(0, 100)

    ax_cost.set_ylabel("Cost per success ($)")
    ax_cost.set_title("Cost per success by task subset")
    ax_cost.set_ylim(0, max(30, max_cost * 1.12))

    subset_handles = [
        plt.Line2D([0], [0], color=color, linewidth=3, label=label)
        for label, color in [
            ("All tasks", SUBSET_COLORS["all"]),
            ("CyberGym", SUBSET_COLORS["cybergym"]),
            (">2h tasks", SUBSET_COLORS["hard_2h"]),
        ]
    ]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.4,
            linestyle=MODEL_STYLES["GPT-5.3 Codex"]["linestyle"],
            label="GPT-5.3 Codex",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.4,
            linestyle=MODEL_STYLES["GPT-5.5"]["linestyle"],
            label="GPT-5.5",
        ),
    ]

    ax_acc.legend(handles=subset_handles + model_handles, fontsize=8.5, loc="upper left")
    ax_cost.legend(handles=subset_handles + model_handles, fontsize=8.5, loc="upper left")

    plt.tight_layout()
    save_chart_json(chart_data, output)
    save_png(fig, output, params)


def main():
    parser = base_parser("Compare token subset analysis charts")
    parser.add_argument("--charts", nargs="+", required=True)
    args = parser.parse_args()
    params = load_params(args.params)

    chart_data = compute(args)
    render_png(chart_data, args.output, params)


if __name__ == "__main__":
    main()
