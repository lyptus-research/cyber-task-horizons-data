"""Declarative data corrections for CTH analysis.

Single source of truth for all manual overrides, exclusions, and
categorizations that affect how raw study data is interpreted.

Import from here; do not duplicate these values elsewhere.

Task-level outlier *types* (OutlierEntry, OutlierReason) and the
OutlierRegistry *machinery* live in outliers.py — this module holds
only the data instances.
"""

from .outliers import OutlierEntry, OutlierReason


# UUID -> anonymized session label, loaded from the persisted public-export
# mapping. After the snapshot was anonymized in place, session_ids in
# completions/estimations are labels (session_001, ...) not UUIDs. To keep
# EXCLUDED_SESSIONS and TIMING_CORRECTIONS firing against both forms, we
# auto-mirror every UUID-keyed entry with its label twin at module load.
def _load_session_label_map() -> dict[str, str]:
    import json as _json
    from pathlib import Path as _Path
    path = _Path(__file__).resolve().parents[2] / "data" / "keep" / "anonymization_mapping.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return _json.load(f).get("session_mapping", {})


def _mirror_uuid_keys(d: dict) -> dict:
    """Return a copy of d with each UUID key duplicated under its anonymized label."""
    mapping = _load_session_label_map()
    out = dict(d)
    for uuid, label in mapping.items():
        if uuid in d and label not in out:
            out[label] = d[uuid]
    return out

# ---------------------------------------------------------------------------
# Known outliers (tasks with bad data — exclude from all analysis)
# ---------------------------------------------------------------------------

KNOWN_OUTLIERS = [
    OutlierEntry(
        "intercode-ctf_reverse/task_85",
        OutlierReason.INFRA_ISSUE,
        "infra issue — anomalous completion time (103m actual vs 5m estimate)",
    ),
    OutlierEntry(
        "cybashbench_forensics/contextual_023",
        OutlierReason.INFRA_ISSUE,
        "infra issue — 48m on a sub-minute bash task (est 0.6m)",
    ),
    OutlierEntry(
        "cybashbench_reverse/contextual_022",
        OutlierReason.INFRA_ISSUE,
        "infra issue — 21m on a sub-minute bash task (est 0.2m)",
    ),
    OutlierEntry(
        "glacier-exchange",
        OutlierReason.INFRA_ISSUE,
        "timing artifact — 10.9h server / 26.2h client on a 9m task (session left open >1 day)",
    ),
    OutlierEntry(
        "arvo:35458",
        OutlierReason.INFRA_ISSUE,
        "CyberGym supplementary task — infrastructure failure in eval harness",
    ),
    OutlierEntry(
        "arvo:36485",
        OutlierReason.INFRA_ISSUE,
        "CyberGym supplementary task — infrastructure failure in eval harness",
    ),
    OutlierEntry(
        "arvo:23499",
        OutlierReason.INFRA_ISSUE,
        "CyberGym supplementary task — infrastructure failure in eval harness",
    ),
    OutlierEntry(
        "cybashbench_misc/nl2bash_050",
        OutlierReason.RESIDUAL_OUTLIER,
        "human-derived time 1.5s — sole task below 15s, uninformative for IRT at this time scale",
    ),
]

# ---------------------------------------------------------------------------
# Session-level timing corrections: session_id -> corrected server_elapsed_seconds
# ---------------------------------------------------------------------------

TIMING_CORRECTIONS: dict[str, float] = {
    # rev-rock: idle-without-pause overnight. 48m corroborated by
    # client_active_seconds and two independent docs.
    "session_369": 48 * 60,
}
# Mirror UUID keys to their anonymized labels so corrections fire whether
# the snapshot still has raw UUIDs or has been anonymized in place.
TIMING_CORRECTIONS = _mirror_uuid_keys(TIMING_CORRECTIONS)


def compute_weights(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute equal_task_weight and invsqrt_task_weight for a single-agent DataFrame.

    Uses METR's canonical implementation. Normalizes weights to sum to 1.0.
    The DataFrame must have task_id, task_family, and score_binarized columns.
    """
    import numpy as np

    n_tasks = df["task_id"].nunique()
    family_sizes = df.groupby("task_family")["task_id"].transform("nunique")

    df = df.copy()
    df["equal_task_weight"] = 1.0 / n_tasks
    raw_w = 1.0 / np.sqrt(family_sizes)
    df["invsqrt_task_weight"] = raw_w / raw_w.sum()
    return df


def corrected_elapsed(
    session: dict,
    timing_corrections: dict[str, float] | None = None,
) -> float:
    """Return server_elapsed_seconds with manual corrections applied.

    This is the ONLY function that should read server_elapsed_seconds from
    a session dict. All analysis code must use this instead of reading the
    raw field, to ensure timing corrections are consistently applied.
    """
    server_s = session.get("server_elapsed_seconds", 0) or 0
    sid = session.get("session_id", "")
    corrections = timing_corrections if timing_corrections is not None else TIMING_CORRECTIONS
    if sid in corrections:
        server_s = corrections[sid]
    return server_s

# ---------------------------------------------------------------------------
# Benchmarks whose original estimates are flat placeholders (not real
# estimates). Exclude from calibration until real model estimates exist.
# ---------------------------------------------------------------------------

PLACEHOLDER_ESTIMATE_BENCHMARKS: set[str] = {"cvebench"}

# ---------------------------------------------------------------------------
# Skip/cancel reason categorization
# ---------------------------------------------------------------------------

# Reasons indicating a genuine attempt where the expert worked but could not
# finish. The elapsed time is a valid lower bound — use as right-censored in
# Tobit regression.
CENSORED_SKIP_REASONS: set[str] = {"too_difficult", "time_constraint"}

# Reasons that are uninformative about task difficulty.
# The elapsed time is not meaningful — exclude from analysis.
EXCLUDED_SKIP_REASONS: set[str] = {
    "technical_issue",
    "wrong_expertise",
    "prior_knowledge",
    "admin_reset",
    "auto_timeout",
    "auto_replaced",
    "duplicate_cleanup",
    "cannot_estimate",
    "other",
}

# ---------------------------------------------------------------------------
# Bogus session detection thresholds
# (used by data.py:load_expert_sessions for CSV-based budget analysis)
# ---------------------------------------------------------------------------

BOGUS_SKIP_REASONS: set[str] = {"technical_issue", "wrong_expertise"}
BOGUS_ACTIVE_THRESHOLD_SEC: float = 90_000

# ---------------------------------------------------------------------------
# Session-level exclusions
# Specific sessions to drop from analysis (e.g. prior knowledge, duplicate
# assignments). Keyed by session_id.
# ---------------------------------------------------------------------------

EXCLUDED_SESSIONS: dict[str, str] = {
    # expert_07: 2013q-msc-networking_1 completed in 59s — duplicate assignment,
    # second attempt is not a genuine independent observation.
    "session_077": "expert_07 2013q-msc-networking_1: duplicate assignment, 59s solve",
    # --- Estimate-before-complete contamination ---
    # Expert saw solution during estimation phase, then completed the task.
    # expert_09: estimated task_8 first, completed 3 days later.
    "session_231": "expert_09 intercode-ctf_reverse/task_8: estimated before completing (solution contamination)",
    # expert_04: estimated task_76 first, completed 9 days later.
    "session_498": "expert_04 intercode-ctf_reverse/task_76: estimated before completing (solution contamination)",
    # --- Complete-before-estimate contamination ---
    # expert_08: completed contextual_063 (failed, 3m), then estimated it next day (35s).
    # Completion is clean (no prior knowledge). Estimation is not independent.
    "session_413": "expert_08 cybashbench_reverse/contextual_063: estimation after completion attempt (not independent)",
    # --- Duplicate sessions (same expert, same task, second attempt) ---
    # These are operational issues (re-assignment or retry) — not independent observations.
    "session_027": "expert_06 arvo:34299: duplicate completion (2nd attempt, 115m)",
    "session_461": "expert_08 intercode-ctf_reverse/task_10: duplicate estimation",
    "session_117": "expert_08 intercode-ctf_reverse/task_81: duplicate estimation",
    "session_289": "expert_08 arvo:49903: duplicate estimation",
    "session_440": "expert_09 flag_command: duplicate estimation",
    "session_379": "expert_05 eval_me: duplicate estimation",
    "session_400": "expert_05 it_has_begun: duplicate estimation",
    # --- Domain mismatch: expert_04 CyberGym estimations (low confidence) ---
    # expert_04's anti-skills include heap_buffer_overflow, use_after_free, double_free.
    # These three estimates are implausibly extreme (5-120h solution-visible) and self-rated low confidence.
    "session_430": "expert_04 arvo:1699: domain mismatch — anti-skill, low confidence (est 120h)",
    "session_148": "expert_04 arvo:29243: domain mismatch — anti-skill, low confidence (est 72h)",
    "session_166": "expert_04 arvo:16541: domain mismatch — anti-skill, low confidence (est 48h)",
}
EXCLUDED_SESSIONS = _mirror_uuid_keys(EXCLUDED_SESSIONS)
