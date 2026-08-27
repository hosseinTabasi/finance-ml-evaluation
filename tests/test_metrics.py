"""Sharpe / DSR / classification metric tests."""

from __future__ import annotations

import numpy as np

from finmlcv.metrics import (
    balanced_accuracy,
    deflated_sharpe_ratio,
    expected_max_sr,
    f1_binary,
    information_coefficient,
    matthews_corrcoef_safe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    turnover,
)


def test_sharpe_of_constant_zero() -> None:
    r = np.zeros(50)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_of_nonzero_constant_is_nan() -> None:
    r = np.ones(50) * 0.01
    assert np.isnan(sharpe_ratio(r))


def test_sharpe_positive_on_positive_drift() -> None:
    rng = np.random.default_rng(0)
    r = 0.001 + 0.01 * rng.normal(size=2000)
    assert sharpe_ratio(r) > 0


def test_dsr_decreases_as_n_trials_increases() -> None:
    rng = np.random.default_rng(1)
    r = 0.002 + 0.01 * rng.normal(size=800)
    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)
    dsr_10 = deflated_sharpe_ratio(r, n_trials=10)
    dsr_100 = deflated_sharpe_ratio(r, n_trials=100)
    dsr_1000 = deflated_sharpe_ratio(r, n_trials=1000)
    assert dsr_1 >= dsr_10 >= dsr_100 >= dsr_1000
    assert dsr_1 - dsr_1000 > 0.05


def test_expected_max_sr_zero_for_one_trial() -> None:
    assert expected_max_sr(1, 0.1) == 0.0
    assert expected_max_sr(50, 0.1) > expected_max_sr(5, 0.1)


def test_psr_near_one_for_strong_sharpe() -> None:
    rng = np.random.default_rng(2)
    r = 0.01 + 0.005 * rng.normal(size=400)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert psr > 0.95


def test_max_drawdown_non_positive() -> None:
    r = np.array([0.1, -0.5, 0.2, -0.1])
    dd = max_drawdown(r)
    assert dd <= 0.0
    assert dd >= -1.0


def test_turnover_and_ic() -> None:
    pos = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    assert turnover(pos) == 1.0
    x = np.arange(20.0)
    assert information_coefficient(x, x) > 0.99
    assert information_coefficient(x, -x) < -0.99


def test_classification_helpers_not_accuracy() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    assert 0.0 < f1_binary(y, pred) < 1.0
    assert -1.0 <= matthews_corrcoef_safe(y, pred) <= 1.0
    assert 0.0 < balanced_accuracy(y, pred) < 1.0
