"""Triple-barrier, fixed-horizon, and meta-label tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finmlcv.labeling import (
    fixed_horizon_returns,
    get_vertical_barrier,
    meta_label,
    triple_barrier_labels,
)


def test_fixed_horizon_log_and_simple() -> None:
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    log_r = fixed_horizon_returns(close, 1, kind="log")
    smp_r = fixed_horizon_returns(close, 1, kind="simple")
    np.testing.assert_allclose(log_r.iloc[0], np.log(1.1))
    np.testing.assert_allclose(smp_r.iloc[0], 0.1)
    assert np.isnan(log_r.iloc[-1])


def test_triple_barrier_hits_upper() -> None:
    # Straight-line rise: upper barrier at +10% should hit before a
    # 10-bar vertical barrier.
    close = pd.Series(np.linspace(100, 130, 16))
    t1 = get_vertical_barrier(close, close.index[:5], horizon=10)
    out = triple_barrier_labels(close, t1, pt=0.10, sl=0.10)
    assert (out["barrier"] == "upper").all()
    assert (out["bin"] == 1).all()
    # First-touch time is strictly after the event.
    for event, row in out.iterrows():
        assert row["t1"] > event


def test_triple_barrier_hits_lower() -> None:
    close = pd.Series(np.linspace(100, 70, 16))
    t1 = get_vertical_barrier(close, close.index[:5], horizon=10)
    out = triple_barrier_labels(close, t1, pt=0.10, sl=0.10)
    assert (out["barrier"] == "lower").all()
    assert (out["bin"] == -1).all()


def test_triple_barrier_hits_vertical() -> None:
    # Oscillation too small to hit 50% barriers; the vertical wins.
    close = pd.Series(100 + 0.5 * np.sin(np.linspace(0, 6, 30)))
    t1 = get_vertical_barrier(close, close.index[:8], horizon=5)
    out = triple_barrier_labels(close, t1, pt=0.50, sl=0.50)
    assert (out["barrier"] == "vertical").all()
    assert set(out["bin"].unique()).issubset({-1, 0, 1})


def test_constructed_path_upper_then_lower_order() -> None:
    # Custom path: rise 20% by bar 3, then crash. Upper should win.
    path = [100, 105, 110, 120, 90, 80]
    close = pd.Series(path, index=pd.RangeIndex(len(path)))
    t1 = pd.Series({0: 5})
    out = triple_barrier_labels(close, t1, pt=0.12, sl=0.12)
    assert out.loc[0, "barrier"] == "upper"
    assert out.loc[0, "bin"] == 1
    assert int(out.loc[0, "t1"]) == 3


def test_constructed_path_lower_first() -> None:
    path = [100, 95, 85, 120]
    close = pd.Series(path)
    t1 = pd.Series({0: 3})
    out = triple_barrier_labels(close, t1, pt=0.12, sl=0.10)
    assert out.loc[0, "barrier"] == "lower"
    assert out.loc[0, "bin"] == -1


def test_meta_label_success_and_failure() -> None:
    close = pd.Series([100, 101, 102, 111, 112])
    t1 = pd.Series({0: 4})
    # Primary long: +10% is reached -> meta-label 1.
    long_side = pd.Series({0: 1.0})
    lab = meta_label(close, t1, long_side, pt=0.08, sl=0.08)
    assert lab.loc[0, "bin"] == 1
    # Primary short on the same path: profit-taking never hits -> 0.
    short_side = pd.Series({0: -1.0})
    lab_s = meta_label(close, t1, short_side, pt=0.08, sl=0.08)
    assert lab_s.loc[0, "bin"] == 0
