"""Plotting helpers for CTH analysis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import HTML, display

from matplotlib.ticker import FixedLocator, FixedFormatter

from .calibration import RegressionResult

# ---------------------------------------------------------------------------
# Human-readable time axis
# ---------------------------------------------------------------------------

# Tick positions in minutes, spanning seconds to days
_TIME_TICKS_MIN = [
    (1 / 60, "1s"),
    (5 / 60, "5s"),
    (15 / 60, "15s"),
    (30 / 60, "30s"),
    (1, "1m"),
    (2, "2m"),
    (5, "5m"),
    (10, "10m"),
    (30, "30m"),
    (1 * 60, "1h"),
    (2 * 60, "2h"),
    (4 * 60, "4h"),
    (8 * 60, "8h"),
    (24 * 60, "24h"),
    (3 * 24 * 60, "3d"),
]


def format_time_axis(ax: plt.Axes, which: str = "x") -> None:
    """Replace log₂(minutes) ticks with human-readable time labels.

    Works on any axis that is already in log₂(minutes) space.
    Automatically selects ticks within the current axis limits.
    *which* can be ``"x"``, ``"y"``, or ``"both"``.
    """
    axes = []
    if which in ("x", "both"):
        axes.append(("x", ax.get_xlim))
    if which in ("y", "both"):
        axes.append(("y", ax.get_ylim))

    for axis_name, get_lim in axes:
        lo, hi = get_lim()
        # Filter ticks to those within or close to the data range
        ticks = []
        labels = []
        for minutes, label in _TIME_TICKS_MIN:
            log2_val = np.log2(minutes)
            if lo - 0.5 <= log2_val <= hi + 0.5:
                ticks.append(log2_val)
                labels.append(label)

        if not ticks:
            continue

        if axis_name == "x":
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(labels))
        else:
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter(labels))


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

try:
    from .lyptus_style import COLORS as _C, PALETTE as _P

    BENCH_COLORS: dict[str, str] = {
        "cybashbench": _P[0],  # teal
        "nl2bash": _P[3],  # gold
        "intercode-ctf": _P[2],  # teal_dark
        "nyuctf": _P[1],  # coral
        "cybench": _P[4],  # slate
        "cvebench": _P[5],  # orange
        "cybergym": _P[6],  # plum
    }
except (ImportError, KeyError):
    BENCH_COLORS: dict[str, str] = {
        "cybashbench": "#1f77b4",
        "nl2bash": "#ff7f0e",
        "intercode-ctf": "#2ca02c",
        "nyuctf": "#d62728",
        "cybench": "#9467bd",
        "cvebench": "#8c564b",
        "cybergym": "#e377c2",
    }

try:
    STATUS_COLORS: dict[str, str] = {
        "OK": _C["teal"],
        "WARN": _C["gold"],
        "RED": _C["coral"],
        "GRAY": _C["text_muted"],
    }
except NameError:
    STATUS_COLORS: dict[str, str] = {
        "OK": "#00897b",
        "WARN": "#e9c46a",
        "RED": "#ff5b5b",
        "GRAY": "#888888",
    }


def setup_style() -> None:
    """Apply standard CTH notebook matplotlib style (call once)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.frameon": False,
            "figure.dpi": 120,
            "figure.figsize": (12, 5),
        }
    )


# ---------------------------------------------------------------------------
# Collapsible HTML sections
# ---------------------------------------------------------------------------


def collapsible(
    title: str,
    content: str,
    open_by_default: bool = False,
) -> None:
    """Render *content* in a collapsible HTML ``<details>`` element."""
    open_attr = "open" if open_by_default else ""
    # Escape HTML in content but preserve whitespace
    safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (
        f"<details {open_attr}>"
        f'<summary style="cursor:pointer;font-weight:bold;padding:8px;'
        f'background:#f5f5f5;border-radius:4px;margin:4px 0;">'
        f"{title}</summary>"
        f'<div style="padding:12px;border-left:3px solid #ddd;'
        f'margin-left:8px;"><pre style="margin:0;font-size:12px;">'
        f"{safe}</pre></div></details>"
    )
    display(HTML(html))


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def bench_legend_handles(benchmarks: list[str]) -> list[Line2D]:
    """Create benchmark-coloured legend handles."""
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BENCH_COLORS.get(b, "gray"),
            markersize=6,
            label=b,
        )
        for b in sorted(benchmarks)
    ]


# ---------------------------------------------------------------------------
# Calibration scatter (single Axes)
# ---------------------------------------------------------------------------


def calibration_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    results: dict[str, RegressionResult | None] | None = None,
    censored_df: pd.DataFrame | None = None,
    *,
    x_col: str = "log2_estimate",
    y_col: str = "log2_actual",
    bench_col: str = "benchmark",
    label_col: str = "task_id",
    label_max_len: int = 25,
    title: str = "",
    xlabel: str = "Estimate",
    ylabel: str = "Actual",
    annotate: bool = True,
) -> None:
    """Plot calibration scatter with regression lines on *ax*.

    *results* is ``{"ols": RegressionResult, "tobit": RegressionResult | None}``.
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        return

    # Uncensored points
    for _, row in df.iterrows():
        color = BENCH_COLORS.get(row[bench_col], "gray")
        ax.scatter(
            row[x_col],
            row[y_col],
            c=color,
            s=60,
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
        )
        if annotate:
            ax.annotate(
                str(row[label_col])[:label_max_len],
                (row[x_col], row[y_col]),
                fontsize=5,
                alpha=0.7,
                textcoords="offset points",
                xytext=(4, 4),
            )

    # Censored points (triangles)
    if censored_df is not None and not censored_df.empty:
        for _, row in censored_df.iterrows():
            color = BENCH_COLORS.get(row[bench_col], "gray")
            ax.scatter(
                row[x_col],
                row[y_col],
                c=color,
                s=80,
                zorder=3,
                edgecolors="white",
                linewidth=0.5,
                marker="^",
            )
            if annotate:
                ax.annotate(
                    f"\u2265 {str(row[label_col])[:20]}",
                    (row[x_col], row[y_col]),
                    fontsize=5,
                    alpha=0.7,
                    textcoords="offset points",
                    xytext=(4, 4),
                    color="darkred",
                )

    # Reference and fit lines
    xlim = ax.get_xlim()
    xline = np.linspace(xlim[0] - 1, xlim[1] + 1, 100)
    ax.plot(xline, xline, "k--", alpha=0.3, label="perfect calibration")

    if results:
        ols = results.get("ols")
        tobit = results.get("tobit")
        if ols is not None:
            ax.plot(
                xline,
                ols.slope * xline + ols.intercept,
                "b-",
                alpha=0.5,
                linewidth=1,
                label=f"OLS: slope={ols.slope:.2f}\u00b1{ols.se_slope:.2f}",
            )
        if tobit is not None:
            ax.plot(
                xline,
                tobit.slope * xline + tobit.intercept,
                "r-",
                alpha=0.7,
                linewidth=2,
                label=f"Tobit: slope={tobit.slope:.2f}\u00b1{tobit.se_slope:.2f}",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    format_time_axis(ax, which="both")

    # Build legend
    handles: list[Line2D] = []
    if results:
        ols = results.get("ols")
        tobit = results.get("tobit")
        if ols:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="b",
                    linewidth=1,
                    alpha=0.5,
                    label=f"OLS (N={ols.n_uncensored})",
                )
            )
        if tobit:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="r",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Tobit (N={tobit.n_total})",
                )
            )
    handles.append(
        Line2D(
            [0], [0], color="k", linestyle="--", alpha=0.3, label="perfect calibration"
        )
    )
    all_benchmarks = list(df[bench_col].unique())
    if censored_df is not None and not censored_df.empty:
        all_benchmarks += list(censored_df[bench_col].unique())
    handles += bench_legend_handles(sorted(set(all_benchmarks)))
    ax.legend(handles=handles, fontsize=7, ncol=2)

    # Stat annotation box (color-coded to match fit lines)
    if results:
        ols = results.get("ols")
        tobit = results.get("tobit")
        stat_lines = []
        if ols is not None:
            r2_str = f"  R²={ols.r_squared:.2f}" if ols.r_squared is not None else ""
            stat_lines.append(
                f"OLS:   slope={ols.slope:.2f}\u00b1{ols.se_slope:.2f}"
                f"{r2_str}  \u03c3={ols.sigma:.2f}"
            )
        if tobit is not None:
            n_str = f"N={tobit.n_uncensored}+{tobit.n_censored}cens"
            stat_lines.append(
                f"Tobit: slope={tobit.slope:.2f}\u00b1{tobit.se_slope:.2f}"
                f"  \u03c3={tobit.sigma:.2f}  ({n_str})"
            )
        if stat_lines:
            stat_box(ax, "\n".join(stat_lines), loc="lower right")


def residual_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    result: RegressionResult,
    *,
    x_col: str = "log2_estimate",
    y_col: str = "log2_actual",
    bench_col: str = "benchmark",
    title: str = "Residuals",
    xlabel: str = "Estimate",
    outlier_threshold: float = 3.0,
) -> None:
    """Plot OLS residuals on *ax*."""
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        return

    x = df[x_col].values
    y = df[y_col].values
    residuals = y - (result.intercept + result.slope * x)
    colors = df[bench_col].map(BENCH_COLORS).fillna("gray")

    ax.scatter(
        x, residuals, c=colors, s=60, zorder=3, edgecolors="white", linewidth=0.5
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    if outlier_threshold:
        ax.axhline(outlier_threshold, color="red", linestyle=":", alpha=0.3)
        ax.axhline(-outlier_threshold, color="red", linestyle=":", alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (doublings)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    format_time_axis(ax, which="x")

    for b in sorted(df[bench_col].unique()):
        ax.scatter([], [], c=BENCH_COLORS.get(b, "gray"), label=b, s=30)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Comparison figure (2x2 grid)
# ---------------------------------------------------------------------------


def calibration_comparison_figure(
    model_est: dict,
    expert_est: dict,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Create the main 2x2 calibration comparison figure.

    *model_est* and *expert_est* are dicts with keys:
        df, results, censored_df, title
    (All from ``aggregate_to_task_level`` / ``fit_calibration_track``.)

    Layout::

        [ Completions vs Model Estimates  ] [ Completions vs Expert Estimates ]
        [ Residuals (model est.)          ] [ Residuals (expert est.)         ]
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.25)

    for col, track in enumerate([model_est, expert_est]):
        ax_scatter = fig.add_subplot(gs[0, col])
        calibration_scatter(
            ax_scatter,
            track["df"],
            results=track.get("results"),
            censored_df=track.get("censored_df"),
            title=track.get("title", ""),
        )

        ax_resid = fig.add_subplot(gs[1, col])
        ols = (track.get("results") or {}).get("ols")
        if ols is not None and not track["df"].empty:
            residual_plot(
                ax_resid,
                track["df"],
                ols,
                title=f"Residuals — {track.get('title', '')}",
            )
        else:
            ax_resid.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=ax_resid.transAxes,
                ha="center",
            )
            ax_resid.set_title(f"Residuals — {track.get('title', '')}")

    fig.suptitle(
        "Calibration Regression Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# Stat annotation box
# ---------------------------------------------------------------------------


_LOC_MAP = {
    "upper left": (0.03, 0.97, "left", "top"),
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}


def stat_box(
    ax: plt.Axes,
    text: str,
    *,
    loc: str = "upper right",
    fontsize: int = 8,
    alpha: float = 0.85,
) -> None:
    """Place a semi-transparent stat annotation box on *ax*."""
    x, y, ha, va = _LOC_MAP.get(loc, _LOC_MAP["upper right"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=alpha),
    )
