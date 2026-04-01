"""Shared constants for CTH analysis notebooks.

Single source of truth for values used across multiple notebooks and scripts.
"""

# === IRT Model ===

# METR's eval-analysis-public uses a `regularization` parameter where C = 1/regularization.
# Both time-horizon-1-0 (original paper) and time-horizon-1-1 (current) use 0.00001.
# Verified in reports/time-horizon-1-{0,1}/fig_params/figs.yaml and DVC pipeline.
DEFAULT_REGULARIZATION = 0.00001  # → C = 100,000, matching METR

# Typical frontier model discrimination coefficient (from June study fits).
# Used in ceiling effect simulations where we assume a known ground truth model.
TYPICAL_FRONTIER_COEF = -0.475

# === SOTA Models (June 2025 study) ===
# 7 state-of-the-art models used for trendline fitting in the June study.
# The current pipeline determines SOTA dynamically via _sota.py.
# eval-pipeline/src/.../analysis/plotter.py also has a copy (legacy).
JUNE_SOTA_FULL_NAMES = [
    "openai/gpt2-xl",
    "openai/davinci-002",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-5-sonnet-20241022",
    "openai/o3-2025-04-16",
    "google/gemini-2.5-pro-preview-06-05",
]

# === Difficulty Buckets ===

# Hard task buckets (hours) — used in ceiling effect analysis
HARD_BUCKETS = [
    ("2–4h", 2, 4),
    ("4–8h", 4, 8),
    ("8–16h", 8, 16),
    ("16–32h", 16, 32),
    ("32–64h", 32, 64),
]

# Bucket edges (minutes) — full time axis for expert data analysis.
# Aligned to _TIME_TICKS_MIN positions: doublings below 30m, then clean hours.
# Starts at 1s to capture sub-minute tasks (cybashbench, nl2bash short).
CALIBRATION_BUCKET_EDGES_MIN = [1 / 60, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480]


def _fmt_minutes(minutes: float) -> str:
    """Format a duration in minutes as a compact human-readable string.

    Matches the vocabulary of ``plots._TIME_TICKS_MIN``:
    1s, 5s, 15s, 30s, 1m, 2m, 5m, 10m, 30m, 1h, 2h, 4h, 8h, 24h.
    """
    if minutes < 1:
        s = round(minutes * 60)
        return f"{s}s"
    if minutes < 60:
        m = int(minutes) if minutes == int(minutes) else minutes
        return f"{m}m"
    h = minutes / 60
    h_int = int(h)
    return f"{h_int}h" if h == h_int else f"{h:.1f}h"


def bucket_labels(edges: list[float] | None = None) -> list[str]:
    """Generate human-readable bucket labels from edges.

    Uses the same tick vocabulary as ``plots.format_time_axis``.
    """
    if edges is None:
        edges = CALIBRATION_BUCKET_EDGES_MIN
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{_fmt_minutes(edges[i])}-{_fmt_minutes(edges[i + 1])}")
    labels.append(f">{_fmt_minutes(edges[-1])}")
    return labels


def assign_bucket(minutes: float, edges: list[float] | None = None) -> int:
    """Assign a time value to a difficulty bucket index.

    Returns the bucket index (0-based), or ``len(edges) - 1`` for values
    above the last edge.
    """
    if edges is None:
        edges = CALIBRATION_BUCKET_EDGES_MIN
    for i in range(len(edges) - 1):
        if minutes < edges[i + 1]:
            return i
    return len(edges) - 1


# === Benchmark Difficulty Ranges (June study) ===
# (n_tasks, lo_minutes, hi_minutes)
BENCHMARK_RANGES_JUNE = {
    "cybashbench": (200, 0.01, 0.7),
    "nl2bash": (136, 0.07, 4),
    "intercode_ctf": (99, 0.17, 10),
    "nyuctf": (50, 2, 120),
    "cybench": (40, 2, 1500),
}

# === Expert Study ===


def _load_expert_rates() -> tuple[dict[str, float], dict[str, float]]:
    """Load expert rates from data/keep/experts.json.

    Returns (uuid_rates, name_rates) where:
    - uuid_rates: {uuid: rate} — reliable, use for budget computations
    - name_rates: {display_name: rate} — best-effort by name/full_name, for display code
    """
    import json
    from pathlib import Path

    experts_file = Path("/dev/null")  # experts.json excluded from public repo
    if not experts_file.exists():
        return {}, {}
    data = json.loads(experts_file.read_text())
    uuid_rates = {}
    name_rates = {}
    for e in data["experts"]:
        rate = e["rate_usd_per_hour"]
        if "uuid" in e:
            uuid_rates[e["uuid"]] = rate
        # Index by both name and full_name for best-effort display lookups
        name_rates[e["name"]] = rate
        if "full_name" in e:
            name_rates[e["full_name"]] = rate
    return uuid_rates, name_rates


EXPERT_RATES_BY_UUID, EXPERT_RATES_USD_HR = _load_expert_rates()
DROPPED_EXPERTS: set[str] = set()  # loaded from experts.json status field
TEST_ACCOUNTS = {"Jack Payne", "user test"}
BUDGET_USD = 9_600
