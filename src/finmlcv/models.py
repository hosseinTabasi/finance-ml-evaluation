"""Thin, seeded wrappers around standard classifiers.

The LSTM path is a small CPU network intended for the toy experiment
only. If PyTorch is not installed, :func:`make_model` with
``name="lstm"`` raises a clear error. If XGBoost is not installed,
``name="xgboost"`` falls back to ``sklearn.ensemble.GradientBoostingClassifier``
and records the substitution on the returned estimator as
``fallback_from``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = Literal["logistic", "rf", "xgboost", "gbm", "lstm"]

try:
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depends on optional extra
    xgb = None  # type: ignore[assignment]
    _HAS_XGBOOST = False

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


class SklearnClfAdapter:
    """Uniform ``fit / predict / predict_proba`` around a sklearn estimator."""

    def __init__(self, estimator: Any, *, name: str, seed: int) -> None:
        self.estimator = estimator
        self.name = name
        self.seed = seed
        self.fallback_from: str | None = getattr(estimator, "fallback_from", None)

    def fit(self, X: Any, y: Any) -> SklearnClfAdapter:
        self.estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(X))
        if hasattr(self.estimator, "decision_function"):
            z = np.asarray(self.estimator.decision_function(X), dtype=float)
            p = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - p, p])
        raise AttributeError(f"{self.name} cannot produce probabilities")


if _HAS_TORCH:

    class _TinyLSTM(nn.Module):
        def __init__(self, n_features: int, hidden: int = 8, n_layers: int = 1) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


    class LSTMClassifier:
        """Binary LSTM on CPU. Each row is treated as a short sequence.

        If ``X`` is 2-d of shape ``(n, f)``, it is viewed as a sequence
        of length ``f`` with 1 channel. If 3-d of shape ``(n, t, f)``,
        that layout is used directly. This is a toy model.
        """

        def __init__(
            self,
            *,
            seed: int = 42,
            hidden: int = 8,
            epochs: int = 8,
            lr: float = 1e-2,
            batch_size: int = 64,
            seq_from_features: bool = True,
        ) -> None:
            if not _HAS_TORCH:
                raise ImportError("PyTorch is required for LSTMClassifier")
            self.seed = seed
            self.hidden = hidden
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.seq_from_features = seq_from_features
            self.name = "lstm"
            self.fallback_from: str | None = None
            self._net: Any = None
            self._n_features: int | None = None

        def _reshape(self, X: Any) -> Any:
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim == 2:
                if self.seq_from_features:
                    arr = arr[:, :, None]
                else:
                    arr = arr[:, None, :]
            elif arr.ndim != 3:
                raise ValueError("X must be 2-d or 3-d")
            return torch.from_numpy(arr)

        def fit(self, X: Any, y: Any) -> LSTMClassifier:
            _seed_everything(self.seed)
            xt = self._reshape(X)
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1))
            n_features = int(xt.shape[-1])
            self._n_features = n_features
            self._net = _TinyLSTM(n_features, hidden=self.hidden)
            self._net.train()
            opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
            loss_fn = nn.BCEWithLogitsLoss()
            ds = TensorDataset(xt, yt)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            for _ in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    logits = self._net(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
            return self

        def predict_proba(self, X: Any) -> np.ndarray:
            if self._net is None:
                raise RuntimeError("LSTMClassifier is not fitted")
            self._net.eval()
            xt = self._reshape(X)
            with torch.no_grad():
                logits = self._net(xt).cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            return np.column_stack([1.0 - p, p])

        def predict(self, X: Any) -> np.ndarray:
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)


def make_model(
    name: ModelName | str = "logistic",
    *,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Factory. Unknown names raise ``ValueError``.

    Parameters accepted via ``kwargs`` are forwarded to the underlying
    constructor. ``seed`` is always applied.
    """
    _seed_everything(seed)
    name_n = str(name).lower()
    if name_n == "logistic":
        C = float(kwargs.pop("C", 1.0))
        max_iter = int(kwargs.pop("max_iter", 200))
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        return SklearnClfAdapter(est, name="logistic", seed=seed)
    if name_n == "rf":
        n_estimators = int(kwargs.pop("n_estimators", 100))
        max_depth = kwargs.pop("max_depth", None)
        min_samples_leaf = int(kwargs.pop("min_samples_leaf", 5))
        est = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
        return SklearnClfAdapter(est, name="rf", seed=seed)
    if name_n in ("xgboost", "xgb"):
        if _HAS_XGBOOST:
            n_estimators = int(kwargs.pop("n_estimators", 80))
            max_depth = int(kwargs.pop("max_depth", 3))
            learning_rate = float(kwargs.pop("learning_rate", 0.1))
            est = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
                verbosity=0,
            )
            return SklearnClfAdapter(est, name="xgboost", seed=seed)
        n_estimators = int(kwargs.pop("n_estimators", 80))
        max_depth = int(kwargs.pop("max_depth", 3))
        learning_rate = float(kwargs.pop("learning_rate", 0.1))
        est = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        est.fallback_from = "xgboost"  # type: ignore[attr-defined]
        adapter = SklearnClfAdapter(est, name="gbm", seed=seed)
        adapter.fallback_from = "xgboost"
        return adapter
    if name_n in ("gbm", "gradientboosting"):
        n_estimators = int(kwargs.pop("n_estimators", 80))
        max_depth = int(kwargs.pop("max_depth", 3))
        learning_rate = float(kwargs.pop("learning_rate", 0.1))
        est = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        return SklearnClfAdapter(est, name="gbm", seed=seed)
    if name_n == "lstm":
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is not installed; LSTM is unavailable. "
                "Install torch (CPU) or use name='logistic' / 'rf' / 'gbm'."
            )
        return LSTMClassifier(
            seed=seed,
            hidden=int(kwargs.pop("hidden", 8)),
            epochs=int(kwargs.pop("epochs", 8)),
            lr=float(kwargs.pop("lr", 1e-2)),
            batch_size=int(kwargs.pop("batch_size", 64)),
        )
    raise ValueError(f"unknown model name: {name}")
