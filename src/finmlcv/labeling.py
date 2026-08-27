"""Label construction for financial ML experiments.

Implements fixed-horizon returns, triple-barrier labels (López de Prado,
AFML ch. 3), and meta-labeling. All routines are deterministic given the
input series. Timestamps may be a DatetimeIndex or a monotonic integer
index; barrier search is done in observation order, not calendar time.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def _as_close_series(close: ArrayLike) -> pd.Series:
    if isinstance(close, pd.Series):
        if not close.index.is_unique:
            raise ValueError("close series index must be unique")
        return close.sort_index()
    close_arr = np.asarray(close, dtype=float)
    return pd.Series(close_arr, index=pd.RangeIndex(len(close_arr)))


def fixed_horizon_returns(
    close: ArrayLike,
    horizon: int,
    *,
    kind: Literal["log", "simple"] = "log",
) -> pd.Series:
    """Forward returns over a fixed number of bars.

    Parameters
    ----------
    close :
        Price series aligned to a unique, sortable index.
    horizon :
        Number of bars in the holding period. Must be >= 1.
    kind :
        ``"log"`` for log returns, ``"simple"`` for close-to-close
        simple returns.

    Returns
    -------
    pd.Series
        Forward return at t equal to the return from t to t+horizon.
        The last ``horizon`` observations are NaN.

    Notes
    -----
    The label at t uses prices up to t+horizon. Any feature used with
    these labels must be computed from information available at t.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    px = _as_close_series(close)
    if kind == "log":
        log_px = np.log(px.astype(float))
        fwd = log_px.shift(-horizon) - log_px
    elif kind == "simple":
        fwd = px.shift(-horizon) / px - 1.0
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    fwd.name = f"fwd_{kind}_{horizon}"
    return fwd


def get_vertical_barrier(
    close: ArrayLike,
    t_events: pd.Index | np.ndarray | None,
    horizon: int,
) -> pd.Series:
    """Vertical (time) barrier for each event: the index horizon bars later.

    If fewer than ``horizon`` bars remain, the barrier is the last index
    of ``close``. Events that start at the final bar are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    px = _as_close_series(close)
    if t_events is None:
        events = px.index[:-1]
    else:
        events = pd.Index(t_events)
        events = events.intersection(px.index)
    loc = px.index.get_indexer(events)
    if np.any(loc < 0):
        raise KeyError("t_events contains timestamps not in close.index")
    end_loc = np.minimum(loc + horizon, len(px) - 1)
    t1 = pd.Series(px.index[end_loc], index=events, name="t1")
    # A vertical barrier equal to the event time itself is not a horizon.
    t1 = t1.loc[t1.index != t1.values]
    return t1


def triple_barrier_labels(
    close: ArrayLike,
    t1: pd.Series,
    pt: float,
    sl: float,
    *,
    min_ret: float = 0.0,
    side: pd.Series | None = None,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Apply profit-taking, stop-loss, and vertical barriers.

    Parameters
    ----------
    close :
        Price series.
    t1 :
        Vertical barrier (label end time) for each event. Index is the
        event start time; values are end times in ``close.index``.
    pt, sl :
        Upper and lower barrier widths in return units (e.g. 0.02 is 2%).
        A non-positive width disables that horizontal barrier.
    min_ret :
        If the first-touch return is smaller than this in absolute value
        and the vertical barrier is hit, the label is 0.
    side :
        Optional primary-model side (+1 long, -1 short). When provided,
        barriers are one-sided and ``bin`` is the meta-label {0, 1}.
    log_returns :
        If True, barrier hits are evaluated on log returns.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``t1``: first-touch time (horizontal or vertical).
        - ``ret``: return from event start to first-touch, signed by
          ``side`` when meta-labeling.
        - ``bin``: {-1, 0, +1} for primary labels, or {0, 1} for
          meta-labels.
        - ``barrier``: ``"upper"``, ``"lower"``, or ``"vertical"``.

    Notes
    -----
    Search is path-dependent along the observed close path between the
    event time and the vertical barrier, inclusive of the vertical
    barrier bar. Ties (upper and lower on the same bar) are broken by
    treating the closer barrier as hit; if equidistant, the vertical
    case is recorded as ``"vertical"`` with label 0 when both widths
    are equal.
    """
    px = _as_close_series(close)
    if pt < 0 or sl < 0:
        raise ValueError("pt and sl must be non-negative")
    t1 = t1.dropna()
    t1 = t1.loc[t1.index.isin(px.index)]
    if side is not None:
        side = side.reindex(t1.index)
        t1 = t1.loc[side.notna()]
        side = side.loc[t1.index]

    records: list[dict[str, object]] = []
    px_vals = px.to_numpy(dtype=float)
    index = px.index
    loc_map = {ts: i for i, ts in enumerate(index)}

    for event_time, vert_time in t1.items():
        i0 = loc_map[event_time]
        i1 = loc_map[vert_time]
        if i1 < i0:
            raise ValueError(f"t1 {vert_time!r} precedes event {event_time!r}")
        path = px_vals[i0 : i1 + 1]
        p0 = path[0]
        if not np.isfinite(p0) or p0 <= 0:
            continue
        if log_returns:
            rets = np.log(path / p0)
        else:
            rets = path / p0 - 1.0

        event_side = 1.0 if side is None else float(side.loc[event_time])
        if event_side not in (-1.0, 1.0):
            raise ValueError("side must be +1 or -1")

        upper_width = pt if event_side > 0 else sl
        lower_width = sl if event_side > 0 else pt
        # For a short, a profitable move is a negative raw return; we
        # still search the raw path and interpret via event_side below.
        if side is not None:
            upper_width = pt
            lower_width = sl
            if event_side < 0:
                # Flip path returns so that profit-taking is positive.
                rets = -rets

        hit_upper = np.where(rets >= upper_width)[0] if upper_width > 0 else np.array([], dtype=int)
        hit_lower = np.where(rets <= -lower_width)[0] if lower_width > 0 else np.array([], dtype=int)
        # Index 0 is the event bar; a barrier of width 0 would always
        # hit immediately. Widths are required non-negative; a zero
        # width disables that barrier (handled above).
        hit_upper = hit_upper[hit_upper > 0]
        hit_lower = hit_lower[hit_lower > 0]

        first_u = int(hit_upper[0]) if len(hit_upper) else None
        first_l = int(hit_lower[0]) if len(hit_lower) else None

        if first_u is not None and first_l is not None:
            if first_u < first_l:
                touch, barrier, raw_bin = first_u, "upper", 1
            elif first_l < first_u:
                touch, barrier, raw_bin = first_l, "lower", -1
            else:
                touch, barrier, raw_bin = first_u, "vertical", 0
        elif first_u is not None:
            touch, barrier, raw_bin = first_u, "upper", 1
        elif first_l is not None:
            touch, barrier, raw_bin = first_l, "lower", -1
        else:
            touch, barrier, raw_bin = len(rets) - 1, "vertical", 0

        ret_touch = float(rets[touch])
        if barrier == "vertical":
            if abs(ret_touch) < min_ret:
                raw_bin = 0
            else:
                raw_bin = int(np.sign(ret_touch))

        if side is None:
            bin_label: int = int(raw_bin)
            ret_out = float(np.log(path[touch] / p0) if log_returns else path[touch] / p0 - 1.0)
        else:
            # Meta-label: success if the (side-aligned) return is positive
            # enough; the stored return is the side-signed return.
            ret_out = ret_touch
            bin_label = 1 if raw_bin > 0 else 0

        records.append(
            {
                "event": event_time,
                "t1": index[i0 + touch],
                "ret": ret_out,
                "bin": bin_label,
                "barrier": barrier,
            }
        )

    if not records:
        return pd.DataFrame(columns=["t1", "ret", "bin", "barrier"])
    out = pd.DataFrame.from_records(records).set_index("event")
    out.index.name = px.index.name
    return out


def meta_label(
    close: ArrayLike,
    t1: pd.Series,
    side: pd.Series,
    pt: float,
    sl: float,
    *,
    min_ret: float = 0.0,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Secondary labels on a primary betting side (AFML meta-labeling).

    The primary model proposes ``side`` in {+1, -1}. The secondary label
    is 1 if the bet hits the profit-taking barrier (or finishes in the
    money beyond ``min_ret``), and 0 otherwise. Positions that should
    not have been taken are the 0 class; this is the standard setup for
    a subsequent binary classifier that sizes or filters bets.
    """
    side = side.reindex(t1.index).dropna()
    if not set(np.unique(side.to_numpy())).issubset({-1.0, 1.0, -1}):
        raise ValueError("side must contain only +1 and -1")
    return triple_barrier_labels(
        close,
        t1.loc[side.index],
        pt=pt,
        sl=sl,
        min_ret=min_ret,
        side=side.astype(float),
        log_returns=log_returns,
    )


def label_lifetime(t1: pd.Series, start: pd.Index | None = None) -> pd.Series:
    """Number of bars from event start to label end, inclusive of both ends - 1.

    For an integer RangeIndex this is ``t1 - start``. For a DatetimeIndex
    the lifetime is the count of bars between the two timestamps in the
    original index, which the caller should supply via ``close.index``
    if calendar time is not the right unit. Here we return the
    positional difference when ``start`` is the event index.
    """
    events = t1.index if start is None else start
    if len(events) != len(t1):
        raise ValueError("start and t1 must have the same length")
    # Prefer positional lifetime when both live on a common index object
    # the caller can map. Fallback: try numeric subtraction.
    try:
        loc_end = pd.Index(t1.to_numpy())
        loc_start = pd.Index(events)
        # If both are numeric, subtraction is in index units.
        if np.issubdtype(loc_end.dtype, np.number) and np.issubdtype(
            loc_start.dtype, np.number
        ):
            return pd.Series(
                np.asarray(loc_end, dtype=float) - np.asarray(loc_start, dtype=float),
                index=t1.index,
                name="lifetime",
            )
    except (TypeError, ValueError):
        pass
    return pd.Series(t1.to_numpy(), index=t1.index, name="t1")
