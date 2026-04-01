"""SOTA determination for the trendline.

Single source of truth for which models are state-of-the-art.
All downstream stages (trendline, sensitivity, figures) should use this.

A model is SOTA if its P50 horizon is the highest of any model
released on or before the same date. This is computed from data,
not a static exclusion list.
"""

from datetime import datetime

import pandas as pd


def _get_sota_agents(
    agent_summaries: pd.DataFrame,
    release_dates: dict[str, str],
) -> list[str]:
    """Determine which models are on the SOTA frontier.

    Walks through models in release-date order. A model is SOTA if
    its P50 is strictly higher than the previous SOTA high-water mark.
    """
    rows = []
    for _, row in agent_summaries.iterrows():
        agent = row["agent"]
        rd_str = release_dates.get(agent)
        p50 = row.get("p50")
        if rd_str and pd.notna(p50) and p50 > 0:
            rows.append((datetime.strptime(rd_str, "%Y-%m-%d"), p50, agent))

    rows.sort(key=lambda r: (r[0], -r[1]))

    sota = []
    best_p50 = -1.0
    for _rd, p50, agent in rows:
        if p50 > best_p50:
            sota.append(agent)
            best_p50 = p50

    return sota


def compute_sota_set(
    agent_summaries: pd.DataFrame,
    release_dates: dict[str, str],
) -> set[str]:
    """Compute the SOTA frontier.

    Args:
        agent_summaries: DataFrame with 'agent' and 'p50' columns.
        release_dates: {alias: "YYYY-MM-DD"} mapping.

    Returns:
        Set of agent aliases that are SOTA.
    """
    valid = agent_summaries[
        agent_summaries["agent"].isin(release_dates)
        & agent_summaries["p50"].notna()
        & (agent_summaries["p50"] > 0)
    ].copy()

    if valid.empty:
        return set()

    return set(_get_sota_agents(valid, release_dates))


def compute_non_frontier(
    agent_summaries: pd.DataFrame,
    release_dates: dict[str, str],
) -> set[str]:
    """Compute the complement of the SOTA set — models excluded from trendline."""
    sota = compute_sota_set(agent_summaries, release_dates)
    all_agents = set(agent_summaries["agent"].unique())
    return all_agents - sota
