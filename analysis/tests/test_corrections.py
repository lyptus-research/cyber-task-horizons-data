"""Tests for lib.corrections — timing corrections and weight computation.

These are the two most critical utility functions in the pipeline:
corrected_elapsed() ensures all analyses use the same timing values,
compute_weights() ensures consistent IRT weighting.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.corrections import (
    TIMING_CORRECTIONS,
    corrected_elapsed,
    compute_weights,
)


# =============================================================================
# corrected_elapsed
# =============================================================================


class TestCorrectedElapsed:
    def test_uncorrected_session_returns_raw(self):
        session = {"server_elapsed_seconds": 300, "session_id": "abc"}
        assert corrected_elapsed(session) == 300

    def test_corrected_session_returns_override(self):
        # Use the actual rev-rock correction from TIMING_CORRECTIONS
        rev_rock_id = "session_369"
        session = {"server_elapsed_seconds": 99999, "session_id": rev_rock_id}
        assert corrected_elapsed(session) == 48 * 60  # 48 minutes

    def test_missing_server_elapsed_defaults_to_zero(self):
        session = {"session_id": "abc"}
        assert corrected_elapsed(session) == 0

    def test_none_server_elapsed_defaults_to_zero(self):
        session = {"server_elapsed_seconds": None, "session_id": "abc"}
        assert corrected_elapsed(session) == 0

    def test_custom_corrections_override(self):
        custom = {"my-session": 42.0}
        session = {"server_elapsed_seconds": 999, "session_id": "my-session"}
        assert corrected_elapsed(session, timing_corrections=custom) == 42.0

    def test_custom_corrections_dont_use_defaults(self):
        """Custom corrections should replace, not merge with, defaults."""
        rev_rock_id = "session_369"
        session = {"server_elapsed_seconds": 99999, "session_id": rev_rock_id}
        # With empty custom corrections, the default should NOT apply
        assert corrected_elapsed(session, timing_corrections={}) == 99999

    def test_empty_session_id(self):
        session = {"server_elapsed_seconds": 100, "session_id": ""}
        assert corrected_elapsed(session) == 100

    def test_no_session_id_key(self):
        session = {"server_elapsed_seconds": 100}
        assert corrected_elapsed(session) == 100

    def test_empty_dict(self):
        assert corrected_elapsed({}) == 0


# =============================================================================
# compute_weights
# =============================================================================


class TestComputeWeights:
    def _make_df(self, families: dict[str, int]) -> pd.DataFrame:
        """Create a DataFrame with given family sizes.

        families: {family_name: n_tasks}
        """
        rows = []
        for family, n in families.items():
            for i in range(n):
                rows.append({
                    "task_id": f"{family}/task_{i}",
                    "task_family": family,
                    "score_binarized": 1,
                })
        return pd.DataFrame(rows)

    def test_single_family_weights_equal(self):
        """With one family, equal and invsqrt weights should be identical."""
        df = self._make_df({"bench": 10})
        result = compute_weights(df)
        assert np.isclose(result["equal_task_weight"].sum(), 1.0)
        assert np.isclose(result["invsqrt_task_weight"].sum(), 1.0)
        np.testing.assert_allclose(
            result["equal_task_weight"].values,
            result["invsqrt_task_weight"].values,
        )

    def test_two_families_invsqrt_upweights_smaller(self):
        """Smaller family should get higher per-task invsqrt weight."""
        df = self._make_df({"small": 5, "large": 50})
        result = compute_weights(df)

        small_weight = result.loc[result["task_family"] == "small", "invsqrt_task_weight"].iloc[0]
        large_weight = result.loc[result["task_family"] == "large", "invsqrt_task_weight"].iloc[0]
        assert small_weight > large_weight, "Smaller family should have higher per-task weight"

    def test_weights_sum_to_one(self):
        df = self._make_df({"a": 10, "b": 20, "c": 5})
        result = compute_weights(df)
        assert np.isclose(result["equal_task_weight"].sum(), 1.0)
        assert np.isclose(result["invsqrt_task_weight"].sum(), 1.0)

    def test_single_task(self):
        df = self._make_df({"bench": 1})
        result = compute_weights(df)
        assert result["equal_task_weight"].iloc[0] == 1.0
        assert result["invsqrt_task_weight"].iloc[0] == 1.0

    def test_does_not_mutate_input(self):
        df = self._make_df({"bench": 5})
        original_cols = set(df.columns)
        compute_weights(df)
        assert set(df.columns) == original_cols, "Should not mutate input DataFrame"

    def test_output_has_weight_columns(self):
        df = self._make_df({"bench": 5})
        result = compute_weights(df)
        assert "equal_task_weight" in result.columns
        assert "invsqrt_task_weight" in result.columns
