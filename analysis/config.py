"""Central path configuration for the CTH public analysis repo.

All data and analysis paths derive from REPO_ROOT, auto-detected
via __file__ traversal. Import paths from here instead of computing
them locally in each module.
"""
from pathlib import Path

# analysis/config.py -> analysis/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = REPO_ROOT / "data"
EVAL_LOGS_DIR = DATA_DIR / "eval_logs"
TASKS_DIR = DATA_DIR / "tasks"
HUMAN_DIR = DATA_DIR / "human"
MODELS_DIR = DATA_DIR / "models"
JUNE_2025_DIR = DATA_DIR / "june_2025"

# Analysis paths
ANALYSIS_DIR = REPO_ROOT / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DATA_DIR = FIGURES_DIR / "data"
