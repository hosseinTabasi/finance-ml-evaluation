"""Leakage-proof evaluation utilities for financial machine learning.

This package implements purged and combinatorial purged cross-validation,
embargo, triple-barrier labeling, and deflated Sharpe diagnostics following
López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), and
Bailey & López de Prado (2014).

Nothing in this package estimates live trading profits. Metrics computed on
synthetic or downloaded samples are research diagnostics.
"""

from __future__ import annotations

__version__ = "0.1.0"

from finmlcv.labeling import (
    fixed_horizon_returns,
    get_vertical_barrier,
    meta_label,
    triple_barrier_labels,
)
from finmlcv.leakage import LeakageReport, audit_leakage
from finmlcv.metrics import (
    balanced_accuracy,
    deflated_sharpe_ratio,
    f1_binary,
    information_coefficient,
    matthews_corrcoef_safe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    turnover,
)
from finmlcv.splits import (
    CombinatorialPurgedCV,
    NaiveKFold,
    PurgedKFold,
    WalkForwardSplit,
    purge_train_indices,
)

__all__ = [
    "__version__",
    "CombinatorialPurgedCV",
    "LeakageReport",
    "NaiveKFold",
    "PurgedKFold",
    "WalkForwardSplit",
    "audit_leakage",
    "balanced_accuracy",
    "deflated_sharpe_ratio",
    "f1_binary",
    "fixed_horizon_returns",
    "get_vertical_barrier",
    "information_coefficient",
    "matthews_corrcoef_safe",
    "max_drawdown",
    "meta_label",
    "probabilistic_sharpe_ratio",
    "purge_train_indices",
    "sharpe_ratio",
    "triple_barrier_labels",
    "turnover",
]
