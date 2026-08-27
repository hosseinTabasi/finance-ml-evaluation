"""Performance and classification diagnostics for research backtests.

Sharpe, probabilistic Sharpe (PSR) and deflated Sharpe (DSR) follow
Bailey & López de Prado (2012, 2014). These are *statistical*
summaries of a return series. They are not live PnL and they do not
correct for all forms of selection bias — only for the multiple-testing
adjustment encoded in ``n_trials``.

Classification helpers expose F1, Matthews correlation (MCC) and
balanced accuracy. Raw accuracy is intentionally not exported as a
headline metric: under class imbalance it is misleading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

EULER_GAMMA = 0.5772156649015329


def _as_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def sharpe_ratio(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio, or a defined edge case.

    If the standard deviation is 0 and the mean excess return is 0,
    the ratio is defined as 0. If the standard deviation is 0 and the
    mean is not, the ratio is NaN (undefined).
    """
    r = _as_1d(returns)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    excess = r - risk_free
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1)) if excess.size > 1 else 0.0
    if sigma == 0.0:
        return 0.0 if mu == 0.0 else float("nan")
    return mu / sigma * np.sqrt(periods_per_year)


def _non_annualized_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1)) if excess.size > 1 else 0.0
    if sigma == 0.0:
        return 0.0 if mu == 0.0 else float("nan")
    return mu / sigma


def _skew_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    """Fisher skew and Pearson kurtosis (normal = 3)."""
    x = returns - np.mean(returns)
    n = x.size
    if n < 3:
        return 0.0, 3.0
    m2 = float(np.mean(x**2))
    if m2 <= 0:
        return 0.0, 3.0
    m3 = float(np.mean(x**3))
    m4 = float(np.mean(x**4))
    skew = m3 / m2**1.5
    kurt = m4 / m2**2
    return skew, kurt


def _sr_variance(sr: float, skew: float, kurt: float, n: int) -> float:
    """Estimated variance of the non-annualised Sharpe (Bailey–LdP 2014)."""
    if n <= 1:
        return float("inf")
    inside = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    inside = max(inside, 1e-12)
    return inside / (n - 1)


def probabilistic_sharpe_ratio(
    returns: Any,
    *,
    sr_benchmark: float = 0.0,
    risk_free: float = 0.0,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """PSR: estimated P(true Sharpe > ``sr_benchmark``).

    ``sr_benchmark`` is in *non-annualised* per-period units, matching
    the observed Sharpe inside the formula. Assumptions: returns are
    strictly stationary; the sampling distribution of the Sharpe ratio
    is approximated by the Bailey–López de Prado expansion that
    corrects for skewness and kurtosis (not a Student-t).

    Returns a value in [0, 1], or NaN if it is undefined.
    """
    r = _as_1d(returns)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return float("nan")
    sr = _non_annualized_sharpe(r, risk_free=risk_free)
    if not np.isfinite(sr):
        return float("nan")
    sk, ku = _skew_kurtosis(r)
    if skew is not None:
        sk = float(skew)
    if kurtosis is not None:
        ku = float(kurtosis)
    numer = (sr - sr_benchmark) * np.sqrt(n - 1)
    denom = np.sqrt(max(1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr**2, 1e-12))
    return float(norm.cdf(numer / denom))


def expected_max_sr(
    n_trials: int,
    sr_std: float,
) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent draws.

    Under a Gaussian approximation of the Sharpe estimator,
    ``E[max SR] ≈ sr_std * ((1-γ) Z^{-1}(1-1/N) + γ Z^{-1}(1-1/(N e)))``
    with γ the Euler–Mascheroni constant (Bailey & López de Prado
    2014). For ``n_trials <= 1`` the result is 0.
    """
    n_trials = int(n_trials)
    if n_trials <= 1:
        return 0.0
    if sr_std < 0:
        raise ValueError("sr_std must be non-negative")
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return float(sr_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: Any,
    *,
    n_trials: int,
    risk_free: float = 0.0,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """DSR: PSR evaluated at the expected-max Sharpe under ``n_trials``.

    Parameters
    ----------
    returns :
        Per-period strategy returns (not prices).
    n_trials :
        Number of independent experiments / configurations considered
        in the selection that produced this return series. This is an
        *assumption*, not something the function can infer. Understated
        ``n_trials`` inflates DSR.
    skew, kurtosis :
        If omitted, estimated from ``returns``. Kurtosis is Pearson
        (normal = 3).

    Returns
    -------
    float
        Estimated probability that the true Sharpe exceeds the
        expected maximum of ``n_trials`` noise Sharpes.

    Notes
    -----
    Independent trials are assumed. Correlated trials make this
    calculation conservative or anti-conservative depending on the
    correlation sign; the original paper discusses the independent
    case. Annualisation is *not* applied inside DSR: the observed
    Sharpe and the haircut live in per-period units.
    """
    r = _as_1d(returns)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return float("nan")
    sr = _non_annualized_sharpe(r, risk_free=risk_free)
    if not np.isfinite(sr):
        return float("nan")
    sk, ku = _skew_kurtosis(r)
    if skew is not None:
        sk = float(skew)
    if kurtosis is not None:
        ku = float(kurtosis)
    sr_std = np.sqrt(_sr_variance(sr, sk, ku, n))
    sr_star = expected_max_sr(n_trials, sr_std)
    return probabilistic_sharpe_ratio(
        r,
        sr_benchmark=sr_star,
        risk_free=risk_free,
        skew=sk,
        kurtosis=ku,
    )


def information_coefficient(y_true: Any, y_pred: Any) -> float:
    """Spearman rank IC between forecasts and realized values."""
    a = _as_1d(y_true)
    b = _as_1d(y_pred)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # Average ties.
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def turnover(positions: Any) -> float:
    """Mean absolute change in position (fraction of book traded per bar)."""
    p = _as_1d(positions)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(p))))


def max_drawdown(returns: Any) -> float:
    """Maximum drawdown of a cumulative return path, in [-1, 0].

    ``returns`` are simple per-period returns. The wealth index starts
    at 1. If log returns are passed by mistake, the number is still a
    drawdown of the exponentiated path only if they are small; the
    caller is responsible for using simple returns.
    """
    r = _as_1d(returns)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / np.maximum(peak, 1e-16) - 1.0
    return float(np.min(dd))


def f1_binary(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = np.isfinite(yt.astype(float)) & np.isfinite(yp.astype(float))
    if mask.sum() == 0:
        return float("nan")
    return float(f1_score(yt[mask], yp[mask], average="binary", zero_division=0))


def matthews_corrcoef_safe(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = np.isfinite(yt.astype(float)) & np.isfinite(yp.astype(float))
    if mask.sum() == 0:
        return float("nan")
    return float(matthews_corrcoef(yt[mask], yp[mask]))


def balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = np.isfinite(yt.astype(float)) & np.isfinite(yp.astype(float))
    if mask.sum() == 0:
        return float("nan")
    return float(balanced_accuracy_score(yt[mask], yp[mask]))


def roc_auc(y_true: Any, y_score: Any) -> float:
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    mask = np.isfinite(yt.astype(float)) & np.isfinite(ys)
    yt, ys = yt[mask], ys[mask]
    if yt.size == 0 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))
