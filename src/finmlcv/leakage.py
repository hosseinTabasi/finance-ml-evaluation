"""Structured leakage audit for financial ML datasets.

Three families of defect are checked:

1. Overlapping labels versus a proposed split (train intervals that
   share a price path with the test interval).
2. Future columns: a feature whose timestamp is strictly after the
   label *start* (look-ahead in the feature matrix itself).
3. Same-timestamp merge hazards: duplicate index values, columns that
   look like they were joined on a raw timestamp rather than an
   as-of / point-in-time merge.

Purged CV addresses (1). It does *not* address (2). A leaked column
equal to ``y[t+1]`` will score well under every splitter, including
purged CV; the auditor is what flags it. This distinction is the
point of the toy experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finmlcv.splits import (
    _as_t1_positions,
    _n_samples,
    _sample_index,
    contiguous_segments,
    purge_train_indices,
)


@dataclass
class LeakageReport:
    """Machine-readable leakage audit.

    ``score`` is in [0, 1] with 0 = no issue found and 1 = severe.
    It is a heuristic for triage, not a statistical test.
    """

    overlapping_label_fraction: float
    n_adjacent_overlaps: int
    n_samples: int
    median_label_lifetime: float
    split_train_overlap_fraction: float | None
    future_columns: list[str]
    same_timestamp_merge_hazards: list[str]
    notes: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "Leakage audit",
            "==============",
            f"n_samples                 : {self.n_samples}",
            f"median label lifetime     : {self.median_label_lifetime:.3f} bars",
            f"adjacent overlap fraction : {self.overlapping_label_fraction:.4f}",
            f"adjacent overlap count    : {self.n_adjacent_overlaps}",
        ]
        if self.split_train_overlap_fraction is not None:
            lines.append(
                f"train/test overlap (pre-purge): "
                f"{self.split_train_overlap_fraction:.4f}"
            )
        lines.append(
            "future columns            : "
            + (", ".join(self.future_columns) if self.future_columns else "(none)")
        )
        lines.append(
            "merge hazards             : "
            + (
                ", ".join(self.same_timestamp_merge_hazards)
                if self.same_timestamp_merge_hazards
                else "(none)"
            )
        )
        lines.append(f"leakage score (heuristic) : {self.score:.3f}")
        if self.notes:
            lines.append("notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _adjacent_overlaps(t1_pos: np.ndarray) -> tuple[int, float]:
    n = t1_pos.size
    if n < 2:
        return 0, 0.0
    starts = np.arange(n)
    # Interval i overlaps i+1 iff t1[i] >= (i+1).
    hits = t1_pos[:-1] >= (starts[:-1] + 1)
    count = int(np.count_nonzero(hits))
    return count, float(count / (n - 1))


def _split_overlap_fraction(
    n: int, test_idx: np.ndarray, t1_pos: np.ndarray
) -> float:
    """Fraction of *non-test* rows whose interval overlaps a test block."""
    test_idx = np.asarray(test_idx, dtype=int)
    if test_idx.size == 0:
        return 0.0
    test_set = np.unique(test_idx)
    train_candidate = np.setdiff1d(np.arange(n), test_set, assume_unique=False)
    if train_candidate.size == 0:
        return 0.0
    overlap_mask = np.zeros(n, dtype=bool)
    starts = np.arange(n)
    ends = t1_pos
    for seg_start, seg_end in contiguous_segments(test_set):
        in_seg = test_set[(test_set >= seg_start) & (test_set <= seg_end)]
        info_end = int(max(seg_end, int(ends[in_seg].max())))
        overlap_mask |= (starts <= info_end) & (seg_start <= ends)
    overlap_mask[test_set] = False
    return float(overlap_mask[train_candidate].mean())


def audit_leakage(
    X: Any,
    y: Any | None = None,
    t1: pd.Series | np.ndarray | None = None,
    *,
    test_idx: np.ndarray | None = None,
    feature_times: Mapping[str, Any] | pd.Series | None = None,
    label_start: pd.Index | np.ndarray | None = None,
    merge_columns: Mapping[str, pd.Index] | None = None,
    future_corr_threshold: float = 0.75,
) -> LeakageReport:
    """Audit a design matrix for label overlap, look-ahead, and merge hazards.

    Parameters
    ----------
    X :
        Feature matrix (DataFrame preferred so columns can be named).
    y :
        Optional labels. If provided and 1-d, columns of ``X`` whose
        absolute correlation with a *forward shift* of y exceeds
        ``future_corr_threshold`` are flagged as future-looking.
    t1 :
        Label end times. If omitted, labels are treated as point-in-time.
    test_idx :
        Optional proposed test positions. If given, the fraction of
        train rows overlapping the test information interval is
        reported (this is the quantity purging removes).
    feature_times :
        Mapping column -> timestamp of the *last* information used to
        compute that column, for the row's label-start time. A column
        is flagged if any row has ``feature_time > label_start``.
    label_start :
        Per-row label start timestamps. Defaults to ``X.index``.
    merge_columns :
        Optional mapping name -> index of a series that was merged
        onto X. Duplicate timestamps are reported as a hazard.
    future_corr_threshold :
        Absolute Pearson correlation with ``y[t+1]`` that triggers a
        flag when y is supplied.

    Returns
    -------
    LeakageReport
    """
    n = _n_samples(X)
    index = _sample_index(X, n)
    t1_pos = _as_t1_positions(t1, index)
    lifetimes = np.maximum(t1_pos - np.arange(n), 0)
    median_life = float(np.median(lifetimes)) if n else 0.0
    n_adj, frac_adj = _adjacent_overlaps(t1_pos)

    split_frac: float | None = None
    if test_idx is not None:
        split_frac = _split_overlap_fraction(n, np.asarray(test_idx), t1_pos)

    notes: list[str] = []
    future_cols: list[str] = []
    hazards: list[str] = []

    if isinstance(X, pd.DataFrame):
        columns = list(map(str, X.columns))
        X_df = X
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        columns = [f"x{i}" for i in range(arr.shape[1])]
        X_df = pd.DataFrame(arr, index=index, columns=columns)

    if not index.is_unique:
        hazards.append("duplicate_index_on_X")
        notes.append(
            "X.index is not unique; an inner merge on timestamps can "
            "duplicate rows and silently look ahead."
        )

    if label_start is None:
        label_start_idx: pd.Index = index
    else:
        label_start_idx = pd.Index(label_start)
        if len(label_start_idx) != n:
            raise ValueError("label_start length must equal n_samples")

    if feature_times is not None:
        if isinstance(feature_times, pd.Series):
            items = list(feature_times.items())
        else:
            items = list(feature_times.items())
        for col, ts in items:
            col_s = str(col)
            if np.isscalar(ts):
                # A single timestamp compared against every label start
                # is only meaningful if it is a column-level as-of time
                # for a snapshot; flag if it is after the earliest label.
                try:
                    if pd.Timestamp(ts) > pd.Timestamp(label_start_idx.min()):
                        future_cols.append(col_s)
                        notes.append(
                            f"column {col_s} has a snapshot time after the "
                            f"first label start"
                        )
                except (ValueError, TypeError):
                    notes.append(f"could not parse feature_times for {col_s}")
            else:
                ts_idx = pd.Index(ts)
                if len(ts_idx) != n:
                    notes.append(
                        f"feature_times[{col_s}] length {len(ts_idx)} != n"
                    )
                    continue
                try:
                    leaked = ts_idx.to_numpy() > label_start_idx.to_numpy()
                except TypeError:
                    leaked = np.array(
                        [a > b for a, b in zip(ts_idx, label_start_idx, strict=True)]
                    )
                if np.any(leaked):
                    future_cols.append(col_s)
                    frac = float(np.mean(leaked))
                    notes.append(
                        f"column {col_s}: feature timestamp > label start "
                        f"in {frac:.1%} of rows"
                    )

    if y is not None:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] == n:
            y_next = np.roll(y_arr, -1)
            y_next[-1] = np.nan
            y_now = y_arr
            for col in columns:
                xcol = pd.to_numeric(X_df[col], errors="coerce").to_numpy(dtype=float)
                mask_next = np.isfinite(xcol) & np.isfinite(y_next)
                mask_now = np.isfinite(xcol) & np.isfinite(y_now)
                if mask_next.sum() >= 10:
                    c_next = float(np.corrcoef(xcol[mask_next], y_next[mask_next])[0, 1])
                else:
                    c_next = 0.0
                if mask_now.sum() >= 10:
                    c_now = float(np.corrcoef(xcol[mask_now], y_now[mask_now])[0, 1])
                else:
                    c_now = 0.0
                if np.isfinite(c_next) and abs(c_next) >= future_corr_threshold:
                    if col not in future_cols:
                        future_cols.append(col)
                    notes.append(
                        f"column {col} correlates with y[t+1] at "
                        f"|r|={abs(c_next):.3f} (contemporaneous |r|="
                        f"{abs(c_now) if np.isfinite(c_now) else float('nan'):.3f})"
                    )
                elif np.isfinite(c_now) and abs(c_now) >= future_corr_threshold:
                    if col not in future_cols:
                        future_cols.append(col)
                    notes.append(
                        f"column {col} is nearly a copy of y "
                        f"(|r|={abs(c_now):.3f}); treat as look-ahead / "
                        f"label leakage in X"
                    )

    if merge_columns:
        for name, idx in merge_columns.items():
            idx = pd.Index(idx)
            if not idx.is_unique:
                hazards.append(f"duplicate_index:{name}")
                notes.append(
                    f"series '{name}' has duplicate timestamps; merging on "
                    f"the raw stamp is not an as-of merge."
                )

    # Heuristic score in [0, 1].
    score = 0.0
    score += min(0.4, 0.4 * frac_adj)
    if split_frac is not None:
        score += min(0.3, 0.3 * split_frac)
    if future_cols:
        score += min(0.4, 0.15 * len(set(future_cols)))
    if hazards:
        score += min(0.2, 0.1 * len(hazards))
    score = float(min(1.0, score))

    if frac_adj > 0.05:
        notes.append(
            "Adjacent labels overlap. Use PurgedKFold / CPCV; random "
            "KFold is not a valid evaluator for these labels."
        )
    if future_cols:
        notes.append(
            "Future-looking columns will inflate *every* CV protocol, "
            "including purged CV. Drop or lag them; do not rely on the "
            "splitter to fix look-ahead in X."
        )

    return LeakageReport(
        overlapping_label_fraction=frac_adj,
        n_adjacent_overlaps=n_adj,
        n_samples=n,
        median_label_lifetime=median_life,
        split_train_overlap_fraction=split_frac,
        future_columns=sorted(set(future_cols)),
        same_timestamp_merge_hazards=hazards,
        notes=notes,
        score=score,
    )


def purge_preview(
    X: Any,
    test_idx: np.ndarray,
    t1: pd.Series | np.ndarray | None = None,
    *,
    embargo_bars: int = 0,
) -> dict[str, int]:
    """How many train rows a purge+embargo step would drop."""
    n = _n_samples(X)
    index = _sample_index(X, n)
    t1_pos = _as_t1_positions(t1, index)
    test_idx = np.asarray(test_idx, dtype=int)
    naive_train = n - np.unique(test_idx).size
    purged = purge_train_indices(n, test_idx, t1_pos, embargo_bars=embargo_bars)
    return {
        "n_samples": n,
        "n_test": int(np.unique(test_idx).size),
        "n_train_naive": int(naive_train),
        "n_train_purged": int(purged.size),
        "n_dropped": int(naive_train - purged.size),
    }
