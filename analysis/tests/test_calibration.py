"""Tests for lib.calibration — slope comparison and estimate aggregation."""

import sys
from pathlib import Path

import numpy as np
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.calibration import compare_slopes, build_all_estimate_means


class TestCompareSlopes:
    def test_identical_slopes_are_equivalent(self):
        result = compare_slopes(
            slope_a=0.8, se_a=0.05, n_a=30,
            slope_b=0.8, se_b=0.05, n_b=30,
            equiv_margin=0.2,
        )
        assert result.slope_diff == 0.0
        assert result.p_nhst > 0.05  # can't reject equality
        assert result.poolable is True or result.poolable is None  # TOST may pass

    def test_very_different_slopes_are_not_poolable(self):
        result = compare_slopes(
            slope_a=0.9, se_a=0.05, n_a=50,
            slope_b=0.3, se_b=0.05, n_b=50,
            equiv_margin=0.2,
        )
        assert result.poolable is False
        assert result.p_nhst < 0.05

    def test_result_has_correct_fields(self):
        result = compare_slopes(0.5, 0.1, 20, 0.6, 0.1, 20)
        assert hasattr(result, "slope_diff")
        assert hasattr(result, "tost_p")
        assert hasattr(result, "poolable")
        assert result.slope_diff == pytest.approx(-0.1)

    def test_equiv_margin_affects_tost(self):
        """Wider margin should make equivalence easier to establish."""
        narrow = compare_slopes(0.5, 0.1, 20, 0.6, 0.1, 20, equiv_margin=0.05)
        wide = compare_slopes(0.5, 0.1, 20, 0.6, 0.1, 20, equiv_margin=0.5)
        # Wider margin should give smaller TOST p-value (easier to pass)
        assert wide.tost_p <= narrow.tost_p


class TestBuildAllEstimateMeans:
    def test_single_estimate_returns_minutes(self):
        estimations = [{"task_id": "t1", "estimated_seconds": 120}]
        result = build_all_estimate_means(estimations)
        assert "t1" in result
        assert np.isclose(result["t1"], 2.0)  # 120s = 2min

    def test_geometric_mean_of_multiple_estimates(self):
        estimations = [
            {"task_id": "t1", "estimated_seconds": 60},   # 1 min
            {"task_id": "t1", "estimated_seconds": 240},  # 4 min
        ]
        result = build_all_estimate_means(estimations)
        assert np.isclose(result["t1"], 2.0)  # geo mean of 1,4 = 2

    def test_zero_and_none_estimates_skipped(self):
        estimations = [
            {"task_id": "t1", "estimated_seconds": 0},
            {"task_id": "t2", "estimated_seconds": None},
            {"task_id": "t3", "estimated_seconds": 60},
        ]
        result = build_all_estimate_means(estimations)
        assert "t1" not in result
        assert "t2" not in result
        assert "t3" in result

    def test_empty_input(self):
        assert build_all_estimate_means([]) == {}
