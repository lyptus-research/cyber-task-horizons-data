"""Tests for lib.icc — ICC(1,1) computation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NOTEBOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from lib.icc import compute_icc, MIN_ICC_TASKS


class TestComputeICC:
    def _make_paired_data(self, n_tasks, agreement=0.9, seed=42):
        """Create paired rater data with controlled agreement level."""
        rng = np.random.RandomState(seed)
        true_vals = rng.uniform(0, 10, n_tasks)
        noise = (1 - agreement) * 5
        rater1 = true_vals + rng.normal(0, noise, n_tasks)
        rater2 = true_vals + rng.normal(0, noise, n_tasks)
        rows = []
        for i in range(n_tasks):
            rows.append({"task_id": f"t{i}", "expert": "A", "log2_min": rater1[i]})
            rows.append({"task_id": f"t{i}", "expert": "B", "log2_min": rater2[i]})
        return pd.DataFrame(rows)

    def test_high_agreement_gives_high_icc(self):
        df = self._make_paired_data(30, agreement=0.95)
        icc, ci_lo, ci_hi, n, sigma = compute_icc(df)
        assert icc is not None
        assert icc > 0.7, f"High agreement should give ICC > 0.7, got {icc}"

    def test_low_agreement_gives_low_icc(self):
        df = self._make_paired_data(30, agreement=0.1)
        icc, _, _, _, _ = compute_icc(df)
        assert icc is not None
        assert icc < 0.5, f"Low agreement should give ICC < 0.5, got {icc}"

    def test_confidence_interval_contains_point_estimate(self):
        df = self._make_paired_data(30, agreement=0.8)
        icc, ci_lo, ci_hi, _, _ = compute_icc(df)
        assert ci_lo <= icc <= ci_hi

    def test_insufficient_tasks_returns_nones(self):
        df = self._make_paired_data(3)  # below MIN_ICC_TASKS
        icc, ci_lo, ci_hi, n, sigma = compute_icc(df)
        assert icc is None
        assert n == 3

    def test_empty_dataframe_returns_nones(self):
        df = pd.DataFrame(columns=["task_id", "expert", "log2_min"])
        icc, _, _, n, _ = compute_icc(df)
        assert icc is None
        assert n == 0

    def test_n_tasks_correct(self):
        df = self._make_paired_data(25)
        _, _, _, n, _ = compute_icc(df)
        assert n == 25

    def test_sigma_within_positive(self):
        df = self._make_paired_data(20)
        _, _, _, _, sigma = compute_icc(df)
        assert sigma > 0
