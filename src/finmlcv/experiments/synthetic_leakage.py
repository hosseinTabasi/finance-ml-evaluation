"""TOY experiment: how much apparent skill is an artefact of the splitter.

Two data-generating processes are compared, both with *no* genuine
walk-forward edge:

1. ``overlap_local`` — piecewise-constant unobserved regime labels,
   features that are locally similar along the timeline (a scaled time
   coordinate plus tiny noise). A flexible model interpolates in
   feature space. Shuffled KFold places time-neighbours on both sides
   of the split and reports high AUC. Purged K-fold removes those
   neighbours and the AUC collapses toward 1/2.

2. ``future_column`` — an explicit leaked feature ``x_leak = y[t+1]``.
   Every splitter that is handed the leaked column, including purged
   CV, reports high AUC. The leakage auditor flags the column. Dropping
   it restores chance-level scores. This is the case purged CV does
   *not* fix.

Numbers written to disk are from an actual run and are labelled TOY.
They are not trading profits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from finmlcv.leakage import audit_leakage
from finmlcv.models import make_model
from finmlcv.splits import (
    CombinatorialPurgedCV,
    NaiveKFold,
    PurgedKFold,
    WalkForwardSplit,
    t1_from_horizon,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def make_overlap_local_dgp(
    n: int = 800,
    *,
    n_regimes: int = 32,
    horizon: int = 40,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Index]:
    """Piecewise-constant labels with time-local features. True OOS AUC = 1/2.

    The unobserved regime is constant on contiguous blocks and is *not*
    a global function of a stationary feature. The only way a model
    scores well under shuffled KFold is by interpolating from
    time-neighbours that share the same regime.
    """
    rng = np.random.default_rng(seed)
    block = n // n_regimes
    regimes = rng.integers(0, 2, size=n_regimes)
    y = np.repeat(regimes, block)
    if y.size < n:
        y = np.concatenate([y, np.full(n - y.size, regimes[-1])])
    y = y[:n].astype(int)
    # Time is scaled so that k-NN neighbourhoods are chronological.
    # A tiny noise dimension avoids degenerate distances; it is far too
    # small to dominate the time axis.
    t = np.linspace(0.0, 1.0, n) * 20.0
    noise = rng.normal(0.0, 1e-4, size=n)
    X = np.column_stack([t, noise])
    index = pd.RangeIndex(n, name="t")
    t1 = t1_from_horizon(index, horizon)
    return X, y, t1, index


def make_future_column_dgp(
    n: int = 800,
    *,
    horizon: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Index]:
    """White-noise labels plus an explicit y[t+1] leak in X."""
    rng = np.random.default_rng(seed)
    # The label is the sign of a *future* return. Putting that return
    # in X is look-ahead: every splitter will look skilled until the
    # column is dropped. y[t+1] as a feature would *not* predict y[t]
    # under IID labels; the leak that inflates AUC is the quantity
    # that determines the current label.
    future_ret = rng.normal(0.0, 1.0, size=n)
    y = (future_ret > 0.0).astype(int)
    lag1 = np.roll(y.astype(float), 1)
    lag1[0] = 0.0
    noise = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame(
        {"lag1": lag1, "noise": noise, "x_leak": future_ret},
        index=pd.RangeIndex(n, name="t"),
    )
    index = X.index
    t1 = t1_from_horizon(index, horizon)
    return X, y, t1, index


def _auc_from_splits(
    model_factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    splitter: Any,
) -> list[float]:
    scores: list[float] = []
    X = np.asarray(X)
    y = np.asarray(y)
    for train_idx, test_idx in splitter.split(X, y):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        y_te = y[test_idx]
        if np.unique(y_te).size < 2:
            continue
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[test_idx])
            score = proba[:, 1] if proba.ndim == 2 else proba
        else:
            score = model.predict(X[test_idx]).astype(float)
        try:
            scores.append(float(roc_auc_score(y_te, score)))
        except ValueError:
            continue
    return scores


def _knn_factory(seed: int = 42) -> Any:
    # n_neighbors is large enough to interpolate a regime locally and
    # small enough that a purged hole of one fold (~n/5 rows) cannot
    # be filled from the fold edges.
    return KNeighborsClassifier(n_neighbors=11)


def _rf_factory(seed: int = 42) -> Any:
    return make_model("rf", seed=seed, n_estimators=80, min_samples_leaf=3)


def run_protocol_grid(
    X: np.ndarray,
    y: np.ndarray,
    t1: pd.Series,
    *,
    seed: int = 42,
    n_splits: int = 5,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    model: str = "knn",
) -> pd.DataFrame:
    if model == "knn":
        factory = lambda: _knn_factory(seed)  # noqa: E731
    elif model == "rf":
        factory = lambda: _rf_factory(seed)  # noqa: E731
    else:
        factory = lambda: make_model(model, seed=seed)  # noqa: E731

    protocols: dict[str, Any] = {
        "naive_kfold_INVALID": NaiveKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ),
        "timeseries_split": WalkForwardSplit(n_splits=n_splits),
        "purged_kfold": PurgedKFold(
            n_splits=n_splits, t1=t1, embargo_pct=embargo_pct
        ),
        "cpcv": CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            t1=t1,
            embargo_pct=embargo_pct,
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, splitter in protocols.items():
        aucs = _auc_from_splits(factory, X, y, splitter)
        rows.append(
            {
                "protocol": name,
                "n_folds_scored": len(aucs),
                "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "auc_min": float(np.min(aucs)) if aucs else float("nan"),
                "auc_max": float(np.max(aucs)) if aucs else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_synthetic_leakage(
    *,
    seed: int = 42,
    n: int = 800,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run both DGPs, write CSV + PNG, return a JSON-serialisable dict."""
    root = _project_root()
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    if out_dir is not None:
        tables = Path(out_dir) / "tables"
        figures = Path(out_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    X_ov, y_ov, t1_ov, _ = make_overlap_local_dgp(n=n, seed=seed)
    overlap_knn = run_protocol_grid(X_ov, y_ov, t1_ov, seed=seed, model="knn")
    overlap_knn.insert(0, "dgp", "overlap_local")
    overlap_knn.insert(1, "model", "knn")
    overlap_knn.insert(2, "label", "TOY")

    X_fu, y_fu, t1_fu, _ = make_future_column_dgp(n=n, seed=seed)
    future_full = run_protocol_grid(
        X_fu.to_numpy(), y_fu, t1_fu, seed=seed, model="logistic"
    )
    future_full.insert(0, "dgp", "future_column")
    future_full.insert(1, "model", "logistic")
    future_full.insert(2, "label", "TOY")

    X_dropped = X_fu.drop(columns=["x_leak"]).to_numpy()
    future_drop = run_protocol_grid(
        X_dropped, y_fu, t1_fu, seed=seed, model="logistic"
    )
    future_drop.insert(0, "dgp", "future_column_dropped")
    future_drop.insert(1, "model", "logistic")
    future_drop.insert(2, "label", "TOY")

    report = audit_leakage(X_fu, y_fu, t1_fu)
    audit_path = tables / "leakage_audit_toy.csv"
    pd.DataFrame(
        [
            {
                "label": "TOY",
                "score": report.score,
                "future_columns": "|".join(report.future_columns),
                "adjacent_overlap_fraction": report.overlapping_label_fraction,
                "median_label_lifetime": report.median_label_lifetime,
            }
        ]
    ).to_csv(audit_path, index=False)

    out = pd.concat([overlap_knn, future_full, future_drop], ignore_index=True)
    csv_path = tables / "synthetic_leakage.csv"
    out.to_csv(csv_path, index=False)

    fig_path = _render_figure(out, figures / "synthetic_leakage.png")
    return {
        "csv": str(csv_path),
        "figure": str(fig_path),
        "audit": str(audit_path),
        "audit_future_columns": report.future_columns,
        "table": out.to_dict(orient="records"),
    }


def _render_figure(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    # Figure 1 of the README: overlap_local, the protocol contrast.
    sub = table.loc[table["dgp"] == "overlap_local"].copy()
    labels = {
        "naive_kfold_INVALID": "Naive KFold\n(INVALID)",
        "timeseries_split": "TimeSeriesSplit",
        "purged_kfold": "Purged KFold\n+ embargo",
        "cpcv": "CPCV paths",
    }
    order = list(labels.keys())
    sub["protocol"] = pd.Categorical(sub["protocol"], order, ordered=True)
    sub = sub.sort_values("protocol")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(sub))
    ax.bar(
        x,
        sub["auc_mean"].to_numpy(),
        yerr=sub["auc_std"].to_numpy(),
        capsize=4,
        color=["#b23a48", "#6c757d", "#2f6f4e", "#1d4e89"],
        edgecolor="black",
        linewidth=0.6,
        width=0.72,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in sub["protocol"]], fontsize=9)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("Mean ROC AUC across folds (TOY)")
    ax.set_title(
        "TOY: overlapping regimes, time-local features, k-NN\n"
        "Apparent skill under shuffled KFold does not survive purging"
    )
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    result = run_synthetic_leakage()
    print("Wrote", result["csv"])
    print("Wrote", result["figure"])
    print(pd.DataFrame(result["table"]).to_string(index=False))
    print("Auditor future columns:", result["audit_future_columns"])


if __name__ == "__main__":
    main()
