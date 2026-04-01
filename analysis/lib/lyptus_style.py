"""Lyptus Research plot style.

Visual identity for publication plots. Matches the lyptusresearch.org website
design language (Source Serif 4, linen background, coral accent) combined with
a teal primary color for data elements.

Usage:
    from lib.lyptus_style import apply_style, COLORS, PALETTE

    apply_style()  # call once, affects all subsequent plots

Design tokens derived from lyptus-website/assets/style.scss.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Font detection
# ---------------------------------------------------------------------------
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}

FONT_SERIF = "Source Serif 4" if "Source Serif 4" in _AVAILABLE else "Georgia"
FONT_SANS = "Helvetica Neue" if "Helvetica Neue" in _AVAILABLE else "Helvetica"

# ---------------------------------------------------------------------------
# Color tokens (from lyptus-website/assets/style.scss + hybrid v3)
# ---------------------------------------------------------------------------

COLORS = {
    # Backgrounds
    "bg": "#fffaf0",  # matches website linen background
    "bg_website": "#fffaf0",  # exact website linen
    "bg_dark": "#111111",  # website dark mode
    # Text
    "text": "#222222",  # base text (website $base-color)
    "text_muted": "#888888",  # secondary text (website post-meta)
    "text_dark": "#e0e0e0",  # dark mode text
    # Grid and borders
    "grid": "#e5dfd6",
    "border": "#e0e0e0",
    # Primary data color: teal
    "teal": "#00897b",
    "teal_light": "#2a9d8f",
    "teal_dark": "#264653",
    # Accent: coral (website $accent-color)
    "coral": "#ff5b5b",
    "coral_light": "#ff7b7b",  # for dark backgrounds
    "coral_dark": "#c0392b",
    # Supporting colors
    "gold": "#e9c46a",
    "orange": "#f4a261",
    "slate": "#457b9d",
    "plum": "#6d597a",
}

# Ordered palette for multi-series plots. Teal first (primary data),
# coral second (fitted curves / accents), then supporting colors.
PALETTE = [
    COLORS["teal"],
    COLORS["coral"],
    COLORS["teal_dark"],
    COLORS["gold"],
    COLORS["slate"],
    COLORS["orange"],
    COLORS["plum"],
    COLORS["teal_light"],
]

# ---------------------------------------------------------------------------
# Custom colormaps
# ---------------------------------------------------------------------------

# Cost heatmap: teal (cheap) -> yellow -> coral -> dark red (expensive)
CMAP_COST = LinearSegmentedColormap.from_list(
    "lyptus_cost",
    [
        COLORS["teal"],
        "#7bc8a4",
        "#e8dfa5",
        "#f0a868",
        COLORS["coral"],
        COLORS["coral_dark"],
    ],
    N=256,
)

# Diverging: teal (negative) -> neutral -> coral (positive)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "lyptus_diverging",
    [
        COLORS["teal_dark"],
        COLORS["teal"],
        "#e8dfa5",
        COLORS["coral"],
        COLORS["coral_dark"],
    ],
    N=256,
)

# Sequential teal: light -> dark
CMAP_TEAL = LinearSegmentedColormap.from_list(
    "lyptus_teal",
    ["#b2dfdb", COLORS["teal"], COLORS["teal_dark"]],
    N=256,
)


# ---------------------------------------------------------------------------
# Apply style
# ---------------------------------------------------------------------------


def apply_style(dark: bool = False) -> None:
    """Set matplotlib rcParams to the Lyptus style.

    Call once at the top of a notebook or script. Affects all subsequent plots.

    Args:
        dark: if True, use the dark mode variant (matches website dark toggle).
    """
    if dark:
        bg = COLORS["bg_dark"]
        text = COLORS["text_dark"]
        grid = "#2a2a2a"
        edge = "#2a2a2a"
    else:
        bg = COLORS["bg"]
        text = COLORS["text"]
        grid = COLORS["grid"]
        edge = COLORS["grid"]

    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": bg,
            "figure.dpi": 150,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            # Axes
            "axes.facecolor": bg,
            "axes.edgecolor": edge,
            "axes.labelcolor": text,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.prop_cycle": plt.cycler("color", PALETTE),
            # Grid
            "grid.color": grid,
            "grid.linestyle": "solid",
            "grid.alpha": 0.3,
            # Text and fonts
            "text.color": text,
            "font.family": FONT_SERIF,
            "font.size": 11,
            # Ticks
            "xtick.color": text,
            "ytick.color": text,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 9,
            # Saving
            "savefig.facecolor": bg,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )
