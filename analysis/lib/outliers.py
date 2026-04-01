"""Task-level data quality: known outliers, auto-detection, filtering."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from scipy.stats import linregress


class OutlierReason(Enum):
    """Why a task was flagged."""

    INFRA_ISSUE = "infra_issue"
    TIMING_ANOMALY = "timing_anomaly"
    PLACEHOLDER_ESTIMATE = "placeholder_estimate"
    RESIDUAL_OUTLIER = "residual_outlier"


@dataclass(frozen=True)
class OutlierEntry:
    """A single outlier with rationale."""

    task_id: str
    reason: OutlierReason
    rationale: str
    source: str = "manual"  # "manual" or "auto"


class OutlierRegistry:
    """Registry of known and auto-detected outliers.

    Usage::

        registry = OutlierRegistry(known=KNOWN_OUTLIERS)
        registry.auto_detect_residual(df, "log2_estimate", "log2_actual")

        clean = registry.filter(df, include_outliers=False)
        registry.is_excluded("intercode-ctf_reverse/task_85")
    """

    def __init__(self, known: list[OutlierEntry] | None = None):
        self._entries: dict[str, OutlierEntry] = {}
        for entry in known or []:
            self._entries[entry.task_id] = entry

    def add(self, entry: OutlierEntry) -> None:
        self._entries[entry.task_id] = entry

    def is_excluded(self, task_id: str) -> bool:
        return task_id in self._entries

    def get_entry(self, task_id: str) -> OutlierEntry | None:
        return self._entries.get(task_id)

    def filter(
        self,
        df: pd.DataFrame,
        task_col: str = "task_id",
        include_outliers: bool = False,
    ) -> pd.DataFrame:
        """Return df with outliers included or excluded."""
        if include_outliers:
            return df
        return df[~df[task_col].isin(self._entries)]

    def auto_detect_residual(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        threshold_doublings: float = 3.0,
        task_col: str = "task_id",
    ) -> list[OutlierEntry]:
        """Flag tasks whose OLS residual exceeds *threshold_doublings*.

        Returns list of newly detected entries (also added to registry).
        Does not overwrite existing manual entries.
        """
        if len(df) < 3:
            return []
        result = linregress(df[x_col].values, df[y_col].values)
        predicted = result.intercept + result.slope * df[x_col].values
        residuals = df[y_col].values - predicted

        new_entries: list[OutlierEntry] = []
        for i, (_, row) in enumerate(df.iterrows()):
            tid = row[task_col]
            if tid in self._entries:
                continue
            if abs(residuals[i]) > threshold_doublings:
                entry = OutlierEntry(
                    task_id=tid,
                    reason=OutlierReason.RESIDUAL_OUTLIER,
                    rationale=(
                        f"residual {residuals[i]:+.1f} doublings "
                        f"(threshold: {threshold_doublings})"
                    ),
                    source="auto",
                )
                self._entries[tid] = entry
                new_entries.append(entry)
        return new_entries

    @property
    def excluded_task_ids(self) -> set[str]:
        return set(self._entries.keys())

    @property
    def entries(self) -> list[OutlierEntry]:
        return list(self._entries.values())

    def summary(self) -> str:
        """Human-readable summary for display / collapsible sections."""
        if not self._entries:
            return "No outliers registered."
        lines = [f"Outlier registry ({len(self._entries)} entries):\n"]
        for entry in self._entries.values():
            lines.append(
                f"  {entry.task_id}\n"
                f"    reason: {entry.reason.value}  source: {entry.source}\n"
                f"    {entry.rationale}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Declarative correction data has moved to corrections.py.
# Import KNOWN_OUTLIERS, TIMING_CORRECTIONS, PLACEHOLDER_ESTIMATE_BENCHMARKS
# from .corrections instead.
# ---------------------------------------------------------------------------
