"""Purged and combinatorial purged cross-validation (AFML ch. 7 and ch. 12).

Reimplemented from the description in López de Prado, *Advances in
Financial Machine Learning*. This is not a copy of mlfinlab or any GPL
source.

The sklearn splitter contract is ``split(X, y=None, groups=None)``
yielding ``(train_idx, test_idx)`` as numpy integer arrays into the
rows of ``X``.

Why random KFold is invalid here
--------------------------------
Triple-barrier and fixed-horizon labels occupy a time interval
``[t, t1]``. Two samples whose intervals overlap share the same
underlying price path. Placing one in train and one in test lets a
flexible model interpolate a relationship that is not available at
decision time. Purging drops train rows whose ``[t, t1]`` overlaps the
test interval. Embargo additionally drops a buffer of bars *after*
each test block to reduce serial-correlation leakage in residuals.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, TimeSeriesSplit

from finmlcv.labeling import get_vertical_barrier


def _n_samples(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


def _sample_index(X: Any, n: int) -> pd.Index:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return pd.Index(X.index)
    return pd.RangeIndex(n)


def _as_t1_positions(
    t1: pd.Series | np.ndarray | Sequence[int] | None,
    index: pd.Index,
) -> np.ndarray:
    """Map label end times to integer positions in ``index``.

    ``t1[i]`` is the *inclusive* end position of the label that starts
    at row i. If ``t1`` is None, labels are treated as point-in-time
    (end = start), so purging reduces to dropping the test rows
    themselves.
    """
    n = len(index)
    if t1 is None:
        return np.arange(n, dtype=int)
    if isinstance(t1, pd.Series):
        aligned = t1.reindex(index)
        if aligned.isna().any():
            # Fall back: interpret values as already-positional if the
            # series is aligned by position rather than by label.
            if len(t1) == n and aligned.isna().all():
                values = np.asarray(t1.to_numpy())
            else:
                missing = int(aligned.isna().sum())
                raise ValueError(
                    f"t1 is missing {missing} values after aligning to X.index"
                )
        else:
            values = aligned.to_numpy()
    else:
        values = np.asarray(t1)
        if values.shape[0] != n:
            raise ValueError("t1 length must equal n_samples")

    if len(values) == 0:
        return np.arange(n, dtype=int)

    # Values may be timestamps (same dtype as index) or integer positions.
    if not np.issubdtype(np.asarray(values).dtype, np.integer):
        pos = index.get_indexer(pd.Index(values))
        if np.any(pos < 0):
            # Last-resort: if values are datetime-like and index is too,
            # searchsorted on a sorted index.
            try:
                pos = np.searchsorted(index.to_numpy(), values, side="left")
                pos = np.clip(pos, 0, n - 1)
                # If the exact label is present, get_indexer failed for a
                # dtype reason; searchsorted-left may land on it.
                exact = index.get_indexer(pd.Index(values))
                pos = np.where(exact >= 0, exact, pos)
            except (TypeError, ValueError) as exc:
                raise ValueError("could not map t1 values onto X.index") from exc
        return pos.astype(int)
    out = np.asarray(values, dtype=int)
    if np.any(out < 0) or np.any(out >= n):
        # Integer timestamps that coincide with a RangeIndex are positions.
        # Clip only if they look like off-by-one end indices equal to n.
        if np.any(out == n):
            out = np.minimum(out, n - 1)
        if np.any(out < 0) or np.any(out >= n):
            raise ValueError("integer t1 positions must satisfy 0 <= t1 < n")
    return out


def contiguous_segments(idx: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] positions of contiguous runs in a sorted index array."""
    if idx.size == 0:
        return []
    order = np.sort(np.unique(idx.astype(int)))
    cuts = np.where(np.diff(order) > 1)[0]
    starts = np.concatenate([[order[0]], order[cuts + 1]])
    ends = np.concatenate([order[cuts], [order[-1]]])
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def embargo_bars_from_spec(
    n_samples: int,
    t1_pos: np.ndarray,
    *,
    embargo_pct: float = 0.0,
    embargo_bars: int | None = None,
    embargo_lifetime_frac: float | None = None,
) -> int:
    """Resolve embargo length in bars.

    Precedence: ``embargo_bars`` > ``embargo_lifetime_frac`` >
    ``embargo_pct``. ``embargo_pct`` is the AFML default: a fraction of
    the *sample size*, not of each label's lifetime.
    """
    if embargo_bars is not None:
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        return int(embargo_bars)
    if embargo_lifetime_frac is not None:
        if embargo_lifetime_frac < 0:
            raise ValueError("embargo_lifetime_frac must be >= 0")
        start = np.arange(n_samples)
        lifetimes = np.maximum(t1_pos - start, 0)
        med = float(np.median(lifetimes)) if lifetimes.size else 0.0
        return int(np.ceil(embargo_lifetime_frac * med))
    if embargo_pct < 0:
        raise ValueError("embargo_pct must be >= 0")
    return int(n_samples * embargo_pct)


def purge_train_indices(
    n_samples: int,
    test_idx: np.ndarray,
    t1_pos: np.ndarray,
    *,
    embargo_bars: int = 0,
) -> np.ndarray:
    """Drop train rows whose label interval overlaps a test block, plus embargo.

    For each contiguous test segment ``[t0, t0_end]`` the *information
    interval* is ``[t0, max(t0_end, max t1 of test rows in the
    segment)]``. A train row i with interval ``[i, t1[i]]`` is purged
    if the two closed intervals overlap. Embargo then drops train rows
    with start in ``(t0_end, t0_end + embargo_bars]``.

    Parameters
    ----------
    n_samples :
        Length of the row axis of X.
    test_idx :
        Positions of test rows (need not be contiguous).
    t1_pos :
        Inclusive label-end positions, shape ``(n_samples,)``.
    embargo_bars :
        Additional bars to drop after each test segment.

    Returns
    -------
    np.ndarray
        Sorted train positions.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    test_idx = np.asarray(test_idx, dtype=int)
    t1_pos = np.asarray(t1_pos, dtype=int)
    if t1_pos.shape[0] != n_samples:
        raise ValueError("t1_pos length must equal n_samples")
    all_idx = np.arange(n_samples, dtype=int)
    if test_idx.size == 0:
        return all_idx
    test_set = np.unique(test_idx)
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_set] = False

    starts = all_idx
    ends = t1_pos
    for seg_start, seg_end in contiguous_segments(test_set):
        in_seg = test_set[(test_set >= seg_start) & (test_set <= seg_end)]
        info_end = int(max(seg_end, int(ends[in_seg].max())))
        # Closed-interval overlap: start <= info_end and seg_start <= end.
        overlap = (starts <= info_end) & (seg_start <= ends)
        train_mask[overlap] = False
        if embargo_bars > 0:
            lo = seg_end + 1
            hi = min(n_samples - 1, seg_end + embargo_bars)
            if lo <= hi:
                train_mask[lo : hi + 1] = False

    return all_idx[train_mask]


def _equal_groups(n_samples: int, n_groups: int) -> np.ndarray:
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_groups > n_samples:
        raise ValueError("n_groups cannot exceed n_samples")
    edges = np.linspace(0, n_samples, n_groups + 1, dtype=int)
    groups = np.empty(n_samples, dtype=int)
    for g in range(n_groups):
        groups[edges[g] : edges[g + 1]] = g
    return groups


class PurgedKFold(BaseCrossValidator):
    """Contiguous K-fold with purging of overlapping labels and embargo.

    Folds are contiguous blocks along the sample order (time). They are
    never shuffled. This is the evaluator recommended in AFML ch. 7 for
    a single held-out test block at a time.

    Parameters
    ----------
    n_splits :
        Number of contiguous folds.
    t1 :
        Label end times, aligned to ``X``. See :func:`purge_train_indices`.
    embargo_pct :
        Fraction of ``n_samples`` embargoed after each test block (AFML).
    embargo_bars :
        If set, overrides ``embargo_pct`` with a fixed bar count.
    embargo_lifetime_frac :
        If set (and ``embargo_bars`` is None), embargo
        ``ceil(frac * median_label_lifetime)`` bars.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | np.ndarray | None = None,
        *,
        embargo_pct: float = 0.0,
        embargo_bars: int | None = None,
        embargo_lifetime_frac: float | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct
        self.embargo_bars = embargo_bars
        self.embargo_lifetime_frac = embargo_lifetime_frac

    def get_n_splits(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        if self.n_splits > n:
            raise ValueError("n_splits cannot exceed n_samples")
        index = _sample_index(X, n)
        t1_pos = _as_t1_positions(self.t1, index)
        e_bars = embargo_bars_from_spec(
            n,
            t1_pos,
            embargo_pct=self.embargo_pct,
            embargo_bars=self.embargo_bars,
            embargo_lifetime_frac=self.embargo_lifetime_frac,
        )
        edges = np.linspace(0, n, self.n_splits + 1, dtype=int)
        for k in range(self.n_splits):
            test_idx = np.arange(edges[k], edges[k + 1], dtype=int)
            train_idx = purge_train_indices(
                n, test_idx, t1_pos, embargo_bars=e_bars
            )
            if test_idx.size == 0:
                continue
            yield train_idx, test_idx


class CombinatorialPurgedCV(BaseCrossValidator):
    """Combinatorial purged CV with backtest path reconstruction (AFML ch. 12).

    The timeline is partitioned into ``n_groups`` contiguous groups.
    Each split takes a combination of ``n_test_groups`` groups as the
    test set and the complement as train, then purges and embargoes.

    A *path* is a complete coverage of the timeline obtained by, for
    each group g, taking the test predictions from exactly one split in
    which g was held out. There are ``C(n_groups-1, n_test_groups-1)``
    such paths. They are not independent, but they give a distribution
    over backtest outcomes rather than a single walk-forward number.

    Parameters
    ----------
    n_groups :
        Number of contiguous timeline groups (N in AFML).
    n_test_groups :
        How many groups are in each test combination (k in AFML).
    t1, embargo_* :
        Forwarded to the purge/embargo step.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        t1: pd.Series | np.ndarray | None = None,
        *,
        embargo_pct: float = 0.0,
        embargo_bars: int | None = None,
        embargo_lifetime_frac: float | None = None,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("require 1 <= n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.embargo_pct = embargo_pct
        self.embargo_bars = embargo_bars
        self.embargo_lifetime_frac = embargo_lifetime_frac

    def get_n_splits(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def _combos(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def path_assignment(self) -> np.ndarray:
        """Map (split, group) -> path id; -1 if the group is not in that test set.

        Shape ``(n_splits, n_groups)``. Path ids run over
        ``0 .. n_paths-1``. For a fixed path p, each group g has exactly
        one split c with ``assignment[c, g] == p``.
        """
        combos = self._combos()
        assign = np.full((len(combos), self.n_groups), -1, dtype=int)
        counters = np.zeros(self.n_groups, dtype=int)
        for c, combo in enumerate(combos):
            for g in combo:
                assign[c, g] = int(counters[g])
                counters[g] += 1
        if not np.all(counters == self.n_paths):
            raise RuntimeError("path assignment counters are inconsistent")
        return assign

    def group_ids(self, X: Any) -> np.ndarray:
        return _equal_groups(_n_samples(X), self.n_groups)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for train_idx, test_idx, _meta in self.split_with_meta(X, y, groups):
            yield train_idx, test_idx

    def split_with_meta(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        n = _n_samples(X)
        index = _sample_index(X, n)
        t1_pos = _as_t1_positions(self.t1, index)
        e_bars = embargo_bars_from_spec(
            n,
            t1_pos,
            embargo_pct=self.embargo_pct,
            embargo_bars=self.embargo_bars,
            embargo_lifetime_frac=self.embargo_lifetime_frac,
        )
        group_ids = self.group_ids(X)
        assign = self.path_assignment()
        for c, combo in enumerate(self._combos()):
            test_mask = np.isin(group_ids, np.asarray(combo))
            test_idx = np.flatnonzero(test_mask)
            train_idx = purge_train_indices(
                n, test_idx, t1_pos, embargo_bars=e_bars
            )
            path_ids = {int(g): int(assign[c, g]) for g in combo}
            meta = {
                "combo": combo,
                "split_id": c,
                "path_ids": path_ids,
                "group_ids": group_ids,
            }
            yield train_idx, test_idx, meta

    def reconstruct_paths(
        self,
        X: Any,
        predictions: Sequence[np.ndarray],
        test_indices: Sequence[np.ndarray],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Stitch per-split test predictions into full-timeline paths.

        Parameters
        ----------
        predictions :
            ``predictions[c]`` holds scores/labels for ``test_indices[c]``,
            in that order.
        test_indices :
            Test positions for each split, in the order of :meth:`split`.

        Returns
        -------
        dict
            ``path_id -> (row_positions, values)`` sorted by position.
            Each path covers every sample exactly once.
        """
        n = _n_samples(X)
        group_ids = self.group_ids(X)
        assign = self.path_assignment()
        combos = self._combos()
        if len(predictions) != len(combos) or len(test_indices) != len(combos):
            raise ValueError("predictions/test_indices must match n_splits")
        paths: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            p: [] for p in range(self.n_paths)
        }
        for c, combo in enumerate(combos):
            pred = np.asarray(predictions[c])
            tidx = np.asarray(test_indices[c], dtype=int)
            if pred.shape[0] != tidx.shape[0]:
                raise ValueError(f"split {c}: prediction length != test length")
            for g in combo:
                p = int(assign[c, g])
                g_mask = group_ids[tidx] == g
                paths[p].append((tidx[g_mask], pred[g_mask]))
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for p, chunks in paths.items():
            pos = np.concatenate([c[0] for c in chunks])
            val = np.concatenate([c[1] for c in chunks])
            order = np.argsort(pos)
            pos, val = pos[order], val[order]
            if pos.shape[0] != n or not np.array_equal(pos, np.arange(n)):
                raise RuntimeError(
                    f"path {p} does not cover the timeline exactly once"
                )
            out[p] = (pos, val)
        return out


class NaiveKFold(BaseCrossValidator):
    """Shuffled KFold. Invalid for overlapping financial labels; baseline only."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._cv = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def get_n_splits(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        return self._cv.get_n_splits(X, y, groups)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._cv.split(X, y, groups)


class WalkForwardSplit(BaseCrossValidator):
    """Thin wrapper around sklearn ``TimeSeriesSplit`` (expanding window)."""

    def __init__(self, n_splits: int = 5, *, max_train_size: int | None = None) -> None:
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self._cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

    def get_n_splits(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        return self._cv.get_n_splits(X, y, groups)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._cv.split(X, y, groups)


def t1_from_horizon(index: pd.Index, horizon: int) -> pd.Series:
    """Convenience: point-start labels that end ``horizon`` bars later."""
    close = pd.Series(np.ones(len(index)), index=index)
    t1 = get_vertical_barrier(close, index, horizon)
    # Rows without a full horizon still need a t1 for the splitter; clip
    # to the last index so the arrays stay aligned.
    full = pd.Series(index[-1], index=index, name="t1")
    full.loc[t1.index] = t1
    return full
