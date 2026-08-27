"""Leakage auditor tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finmlcv.leakage import audit_leakage, purge_preview
from finmlcv.splits import t1_from_horizon


def test_leaked_future_column_is_flagged() -> None:
    n = 200
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n).astype(float)
    leak = np.roll(y, -1)
    leak[-1] = y[-2]
    X = pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            "x_leak": leak,
        }
    )
    t1 = t1_from_horizon(X.index, 5)
    report = audit_leakage(X, y, t1)
    assert "x_leak" in report.future_columns
    assert report.score > 0.0
    printable = str(report)
    assert "x_leak" in printable
    d = report.to_dict()
    assert "future_columns" in d


def test_overlapping_labels_raise_adjacent_fraction() -> None:
    n = 100
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    t1_point = pd.Series(np.arange(n), index=X.index)  # point-in-time labels
    t1_wide = t1_from_horizon(X.index, 20)
    r_point = audit_leakage(X, t1=t1_point)
    r_wide = audit_leakage(X, t1=t1_wide)
    assert r_wide.overlapping_label_fraction > r_point.overlapping_label_fraction
    assert r_wide.median_label_lifetime >= 10


def test_feature_times_after_label_start() -> None:
    n = 30
    idx = pd.RangeIndex(n)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    feature_times = {"f": idx + 3}  # every feature uses 3 bars of future
    report = audit_leakage(
        X, t1=t1_from_horizon(idx, 2), feature_times=feature_times, label_start=idx
    )
    assert "f" in report.future_columns


def test_duplicate_index_is_merge_hazard() -> None:
    idx = pd.Index([0, 1, 1, 2])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    y = np.array([0, 1, 0, 1], dtype=float)
    report = audit_leakage(X, y)
    assert "duplicate_index_on_X" in report.same_timestamp_merge_hazards


def test_purge_preview_counts() -> None:
    n = 40
    X = np.zeros((n, 1))
    t1 = np.minimum(np.arange(n) + 5, n - 1)
    test = np.arange(10, 20)
    prev = purge_preview(X, test, t1, embargo_bars=2)
    assert prev["n_dropped"] > 0
    assert prev["n_train_purged"] < prev["n_train_naive"]
