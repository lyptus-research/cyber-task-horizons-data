"""Data loading helpers that avoid heavy METR imports.

The lib/trendline.py module imports METR's eval-analysis-public which requires
cairosvg/cairo C library. These helpers provide the same data (release dates,
legacy model configs) without triggering those imports.
"""

import json
from pathlib import Path

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2]
from analysis.config import MODELS_DIR as _MODELS_DIR

# Alias overrides: config alias -> campaign alias
_ALIAS_MAP = {
    "Claude Haiku 4.5": "Haiku 4.5",
    "Claude Opus 4": "Opus 4",
    "Claude Sonnet 4.6": "Sonnet 4.6",
    "Claude Opus 4.6": "Opus 4.6",
    "Gemini 2.5 Pro (June 2025)": "Gemini 2.5 Pro",
}

# Legacy models not in config JSONs
_LEGACY_DATES = {
    "GPT-2": "2019-11-05",
    "GPT-3": "2020-07-11",
    "GPT-3.5": "2022-03-15",
}


def load_release_dates() -> dict[str, str]:
    """Load release dates from model config JSONs (single source of truth)."""
    dates = dict(_LEGACY_DATES)

    if _MODELS_DIR.exists():
        for json_file in sorted(_MODELS_DIR.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            for model in data.get("models", []):
                alias = model.get("alias", "")
                release = model.get("release_date", "")
                if alias and release:
                    campaign_alias = _ALIAS_MAP.get(alias, alias)
                    dates[campaign_alias] = release

    return dates
