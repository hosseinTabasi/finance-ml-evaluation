"""Splitter tests: sklearn API, embargo, CPCV coverage, leakage contrast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from finmlcv.experiments.synthetic_leakage import make_overlap_local_dgp
from finmlcv.splits import (
    CombinatorialPurgedCV,
    NaiveKFold,
    PurgedKFold,
    WalkForwardSplit,
    contiguous_segments,
    purge_train_indices,
    t1_from_horizon,
)


def test_sklearn_splitter_api() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.integers(0, 2, size=100)
    t1 = t1_from_horizon(pd.RangeIndex(100), 8)
    for cv in (
        PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01),
        CombinatorialPurgedCV(n_groups=5, n_test_groups=2, t1=t1),
        NaiveKFold(n_splits=4, random_state=0),
        WalkForwardSplit(n_splits=4),
    ):
        n = cv.get_n_splits(X, y)
        splits = list(cv.split(X, y))
        assert len(splits) == n
        for train, test in splits:
            assert train.dtype.kind == "i"
            assert test.dtype.kind == "i"
            assert np.intersect1d(train, test).size == 0
            assert test.size > 0
            assert train.min() >= 0 and test.min() >= 0
            assert train.max() < 100 and test.max() < 100


def test_purge_drops_overlapping_train_rows() -> None:
    n = 20
    # Every label lasts 5 bars: [i, i+4].
    t1 = np.minimum(np.arange(n) + 4, n - 1)
    test = np.arange(8, 12)
    purged = purge_train_indices(n, test, t1, embargo_bars=0)
    # Train row 4 has t1=8, which is the start of the test block -> overlap.
    assert 4 not in purged
    # Train row 7 ends at 11, inside the test block.
    assert 7 not in purged
    # Train row 3 ends at 7, before test start 8: no overlap.
    assert 3 in purged
    # Test rows themselves are never in train.
    assert np.intersect1d(purged, test).size == 0
    # The test *information* interval extends to max t1 of the test rows
    # (t1[11] = 15), so row 12 with [12, 16] overlaps and is purged.
    # Row 16 starts after that information interval.
    assert 12 not in purged
    assert 16 in purged


def test_embargo_drops_post_test_rows() -> None:
    n = 30
    t1 = np.arange(n)  # point labels
    test = np.arange(10, 15)
    no_emb = purge_train_indices(n, test, t1, embargo_bars=0)
    with_emb = purge_train_indices(n, test, t1, embargo_bars=3)
    # Bars 15,16,17 should be dropped only with embargo (test ends at 14).
    for k in (15, 16, 17):
        assert k in no_emb
        assert k not in with_emb
    # Bar 18 is outside the embargo window.
    assert 18 in with_emb


def test_contiguous_segments_nonadjacent() -> None:
    idx = np.array([0, 1, 2, 8, 9, 15])
    segs = contiguous_segments(idx)
    assert segs == [(0, 2), (8, 9), (15, 15)]


def test_cpcv_paths_cover_timeline() -> None:
    n = 60
    X = np.zeros((n, 1))
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    assert cv.get_n_splits(X) == 15  # C(6,2)
    assert cv.n_paths == 5  # C(5,1)
    preds: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    seen_test_union = np.zeros(n, dtype=int)
    for train, test in cv.split(X):
        preds.append(test.astype(float))  # dummy: predict the index
        tests.append(test)
        seen_test_union[test] += 1
        assert np.intersect1d(train, test).size == 0
    # Every row is in some test set (actually in n_paths test sets).
    assert np.all(seen_test_union == cv.n_paths)
    paths = cv.reconstruct_paths(X, preds, tests)
    assert set(paths) == set(range(cv.n_paths))
    for _pid, (pos, val) in paths.items():
        assert np.array_equal(pos, np.arange(n))
        assert val.shape[0] == n


def test_purged_kfold_contiguous_test_blocks() -> None:
    n = 50
    X = np.zeros((n, 1))
    t1 = t1_from_horizon(pd.RangeIndex(n), 6)
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
    for train, test in cv.split(X):
        # Test is a contiguous block.
        assert test.max() - test.min() + 1 == test.size
        # Purged train has no overlapping labels with the test information
        # interval.
        t1_pos = np.minimum(np.arange(n) + 6, n - 1)
        t0, t_end = int(test.min()), int(test.max())
        info_end = int(max(t_end, t1_pos[test].max()))
        for i in train:
            start, end = i, t1_pos[i]
            overlap = start <= info_end and t0 <= end
            assert not overlap, (i, start, end, t0, info_end)


def test_naive_kfold_optimistic_purged_honest() -> None:
    """Overlapping local regimes: shuffled KFold looks skilled, purged does not."""
    X, y, t1, _ = make_overlap_local_dgp(n=600, n_regimes=20, horizon=25, seed=0)
    def _mean_auc(splitter) -> float:
        scores = []
        for tr, te in splitter.split(X, y):
            if np.unique(y[te]).size < 2:
                continue
            clf = KNeighborsClassifier(n_neighbors=11)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            scores.append(roc_auc_score(y[te], p))
        return float(np.mean(scores))

    auc_naive = _mean_auc(NaiveKFold(n_splits=5, shuffle=True, random_state=0))
    auc_purged = _mean_auc(PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.02))
    assert auc_naive > 0.80, auc_naive
    assert auc_purged < 0.70, auc_purged
    assert auc_naive - auc_purged > 0.15
