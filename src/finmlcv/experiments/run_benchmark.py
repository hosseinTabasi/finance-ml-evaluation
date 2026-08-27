"""Protocol benchmark: naive KFold vs TimeSeriesSplit vs purged vs CPCV.

Reads ``configs/experiment.yaml``. In ``mode: toy`` a synthetic series
is generated; no vendor data are required. In ``mode: full`` the
script attempts to load cached OHLCV for the configured tickers and
skips any ticker that is missing (download failures are not retried
here; see ``scripts/download_data.py``).

All scores are research diagnostics on a TOY or, if present, a
downloaded daily series. They are not live PnL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score

from finmlcv.labeling import fixed_horizon_returns
from finmlcv.metrics import (
    balanced_accuracy,
    deflated_sharpe_ratio,
    f1_binary,
    information_coefficient,
    matthews_corrcoef_safe,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
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


class ExperimentConfig(BaseModel):
    seed: int = 42
    mode: str = "toy"
    n_samples: int = 1200
    horizon: int = 10
    n_splits: int = 5
    n_groups: int = 6
    n_test_groups: int = 2
    embargo_grid: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.05])
    model: str = "logistic"
    n_trials_dsr: int = 20
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "BTC-USD"])
    periods_per_year: float = 252.0


def load_config(path: Path | None = None) -> ExperimentConfig:
    cfg_path = path or (_project_root() / "configs" / "experiment.yaml")
    if not cfg_path.exists():
        return ExperimentConfig()
    with cfg_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return ExperimentConfig.model_validate(raw)


def make_toy_panel(n: int, horizon: int, seed: int) -> pd.DataFrame:
    """Synthetic daily-like series with a weak, causal AR signal.

    The signal is small on purpose: the point of the benchmark is the
    *spread across protocols*, not a claim of alpha. A modest AR(1)
    coefficient in the *observable* return makes a correctly specified
    logistic model slightly better than chance, so TimeSeriesSplit and
    purged CV should agree to first order, while shuffled KFold can
    still overstate because of overlapping labels.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.01, size=n)
    r = np.zeros(n)
    phi = 0.08
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.RangeIndex(n, name="t")
    return pd.DataFrame({"close": close, "ret": r}, index=idx)


def causal_features(close: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """Lagged returns, realised vol, simple technicals; all shifted.

    Every column is lagged by at least one bar relative to ``close[t]``
    used as the decision time, except ``ret_lag1`` which is the return
    *ending* at t (known at the close of t). Forward labels must start
    at t and look strictly forward of t, which
    :func:`finmlcv.labeling.fixed_horizon_returns` does.
    """
    logp = np.log(close.astype(float))
    feat = pd.DataFrame(index=close.index)
    feat["ret_lag1"] = ret
    feat["ret_lag2"] = ret.shift(1)
    feat["ret_lag5"] = ret.shift(4)
    feat["rv_5"] = ret.rolling(5).std()
    feat["rv_20"] = ret.rolling(20).std()
    feat["mom_10"] = logp.diff(10)
    feat["mom_20"] = logp.diff(20)
    # Simple causal technical: deviation from a trailing mean, known at t.
    trail = close.rolling(20).mean()
    feat["dev_ma20"] = close / trail - 1.0
    # Strictly causal: shift rolling stats that include the current bar
    # is already using information at t. That is allowed for a decision
    # at the close of t. No forward fill of future bars.
    return feat


def align_xy(
    feat: pd.DataFrame, fwd: pd.Series, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Drop NaNs *before* splitting. Returns X, y_bin, fwd_ret, t1.

    ``t1`` is remapped onto the aligned row axis (0..n-1) so that a
    subsequent conversion of X to ``numpy`` does not desynchronise
    label-end times from row positions.
    """
    y_bin = (fwd > 0).astype(int)
    t1 = t1_from_horizon(feat.index, horizon)
    frame = feat.join(y_bin.rename("y"), how="inner").join(
        fwd.rename("fwd"), how="inner"
    )
    frame = frame.join(t1.rename("t1"), how="left")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    remaining = list(frame.index)
    loc = {ts: i for i, ts in enumerate(remaining)}
    n = len(remaining)
    t1_pos = []
    for v in frame["t1"].to_numpy():
        t1_pos.append(int(loc[v]) if v in loc else n - 1)
    t1_a = pd.Series(t1_pos, index=pd.RangeIndex(n), name="t1")
    X = frame[feat.columns].reset_index(drop=True)
    y = frame["y"].astype(int).reset_index(drop=True)
    fwd_a = frame["fwd"].reset_index(drop=True)
    return X, y, fwd_a, t1_a


def _positions_from_proba(proba: np.ndarray) -> np.ndarray:
    """Long/short from P(y=1): position in {-1, +1}."""
    if proba.ndim == 2:
        p = proba[:, 1]
    else:
        p = proba
    return np.where(p >= 0.5, 1.0, -1.0)


def _fold_metrics(
    y_te: np.ndarray,
    proba: np.ndarray,
    fwd_te: np.ndarray,
    *,
    n_trials: int,
    periods_per_year: float,
) -> dict[str, float]:
    pred = (proba[:, 1] >= 0.5).astype(int) if proba.ndim == 2 else (proba >= 0.5).astype(int)
    pos = _positions_from_proba(proba)
    # Simple return of a long/short overlay on the *forward* return.
    # TOY diagnostic only.
    strat = pos * fwd_te
    auc = float("nan")
    if np.unique(y_te).size == 2:
        try:
            auc = float(roc_auc_score(y_te, proba[:, 1] if proba.ndim == 2 else proba))
        except ValueError:
            auc = float("nan")
    return {
        "auc": auc,
        "mcc": matthews_corrcoef_safe(y_te, pred),
        "f1": f1_binary(y_te, pred),
        "balanced_acc": balanced_accuracy(y_te, pred),
        "ic": information_coefficient(fwd_te, proba[:, 1] if proba.ndim == 2 else proba),
        "sharpe": sharpe_ratio(strat, periods_per_year=periods_per_year),
        "dsr": deflated_sharpe_ratio(strat, n_trials=n_trials),
        "max_dd": max_drawdown(strat),
        "turnover": turnover(pos),
        "frac_sharpe_pos": float(sharpe_ratio(strat, periods_per_year=periods_per_year) > 0),
    }


def evaluate_splitter(
    name: str,
    splitter: Any,
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    *,
    seed: int,
    model_name: str,
    n_trials: int,
    periods_per_year: float,
    is_cpcv: bool = False,
) -> dict[str, Any]:
    fold_rows: list[dict[str, float]] = []
    preds: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    for train_idx, test_idx in splitter.split(X, y):
        if train_idx.size < 20 or test_idx.size < 5:
            continue
        model = make_model(model_name, seed=seed)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])
        mets = _fold_metrics(
            y[test_idx],
            proba,
            fwd[test_idx],
            n_trials=n_trials,
            periods_per_year=periods_per_year,
        )
        fold_rows.append(mets)
        preds.append(proba[:, 1] if proba.ndim == 2 else proba)
        tests.append(np.asarray(test_idx, dtype=int))

    if not fold_rows:
        return {
            "protocol": name,
            "n_scores": 0,
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "mcc_mean": float("nan"),
            "sharpe_mean": float("nan"),
            "sharpe_std": float("nan"),
            "dsr_mean": float("nan"),
            "frac_paths_sharpe_gt_0": float("nan"),
            "ic_mean": float("nan"),
            "max_dd_mean": float("nan"),
        }

    # CPCV: reconstruct paths and recompute Sharpe on each full path.
    path_sharpes: list[float] = []
    path_dsrs: list[float] = []
    if is_cpcv and hasattr(splitter, "reconstruct_paths") and preds:
        try:
            paths = splitter.reconstruct_paths(X, preds, tests)
            for _pid, (pos, val) in paths.items():
                pos_sign = np.where(val >= 0.5, 1.0, -1.0)
                strat = pos_sign * fwd[pos]
                path_sharpes.append(sharpe_ratio(strat, periods_per_year=periods_per_year))
                path_dsrs.append(deflated_sharpe_ratio(strat, n_trials=n_trials))
        except Exception:  # noqa: BLE001 - path reconstruction is best-effort
            path_sharpes = []
            path_dsrs = []

    df = pd.DataFrame(fold_rows)
    sharpe_series = path_sharpes if path_sharpes else df["sharpe"].tolist()
    dsr_series = path_dsrs if path_dsrs else df["dsr"].tolist()
    return {
        "protocol": name,
        "n_scores": int(len(sharpe_series)),
        "auc_mean": float(df["auc"].mean()),
        "auc_std": float(df["auc"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mcc_mean": float(df["mcc"].mean()),
        "sharpe_mean": float(np.nanmean(sharpe_series)),
        "sharpe_std": float(np.nanstd(sharpe_series, ddof=1))
        if len(sharpe_series) > 1
        else 0.0,
        "dsr_mean": float(np.nanmean(dsr_series)),
        "frac_paths_sharpe_gt_0": float(np.mean([s > 0 for s in sharpe_series])),
        "ic_mean": float(df["ic"].mean()),
        "max_dd_mean": float(df["max_dd"].mean()),
    }


def load_full_frame(root: Path, tickers: list[str]) -> pd.DataFrame | None:
    raw = root / "data" / "raw"
    frames: list[pd.DataFrame] = []
    for tkr in tickers:
        for suffix in (".parquet", ".csv"):
            safe = tkr.replace("/", "-")
            path = raw / f"{safe}{suffix}"
            if not path.exists():
                continue
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            col = None
            for candidate in ("adj_close", "Adj Close", "close", "Close"):
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                continue
            s = df[col].astype(float).rename(tkr)
            frames.append(s.to_frame())
            break
    if not frames:
        return None
    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra, how="outer")
    return out.sort_index()


def run_on_panel(
    close: pd.Series,
    ret: pd.Series,
    cfg: ExperimentConfig,
    *,
    tag: str,
) -> pd.DataFrame:
    feat = causal_features(close, ret)
    fwd = fixed_horizon_returns(close, cfg.horizon, kind="log")
    X_df, y_s, fwd_s, t1_s = align_xy(feat, fwd, cfg.horizon)
    X = X_df.to_numpy(dtype=float)
    y = y_s.to_numpy(dtype=int)
    fwd_a = fwd_s.to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    embargo_default = cfg.embargo_grid[1] if len(cfg.embargo_grid) > 1 else 0.01

    protocols: list[tuple[str, Any, bool]] = [
        (
            "naive_kfold_INVALID",
            NaiveKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed),
            False,
        ),
        ("timeseries_split", WalkForwardSplit(n_splits=cfg.n_splits), False),
        (
            "purged_kfold",
            PurgedKFold(n_splits=cfg.n_splits, t1=t1_s, embargo_pct=embargo_default),
            False,
        ),
        (
            "cpcv",
            CombinatorialPurgedCV(
                n_groups=cfg.n_groups,
                n_test_groups=cfg.n_test_groups,
                t1=t1_s,
                embargo_pct=embargo_default,
            ),
            True,
        ),
    ]
    for name, splitter, is_cpcv in protocols:
        rec = evaluate_splitter(
            name,
            splitter,
            X,
            y,
            fwd_a,
            seed=cfg.seed,
            model_name=cfg.model,
            n_trials=cfg.n_trials_dsr,
            periods_per_year=cfg.periods_per_year,
            is_cpcv=is_cpcv,
        )
        rec["embargo_pct"] = embargo_default if name in ("purged_kfold", "cpcv") else 0.0
        rec["tag"] = tag
        rec["label"] = "TOY" if tag.startswith("toy") else "DOWNLOADED"
        rec["n_samples_aligned"] = int(len(y))
        rec["model"] = cfg.model
        rows.append(rec)

    for emb in cfg.embargo_grid:
        splitter = PurgedKFold(n_splits=cfg.n_splits, t1=t1_s, embargo_pct=emb)
        rec = evaluate_splitter(
            f"purged_kfold_embargo_{emb:.2%}",
            splitter,
            X,
            y,
            fwd_a,
            seed=cfg.seed,
            model_name=cfg.model,
            n_trials=cfg.n_trials_dsr,
            periods_per_year=cfg.periods_per_year,
            is_cpcv=False,
        )
        rec["embargo_pct"] = emb
        rec["tag"] = tag
        rec["label"] = "TOY" if tag.startswith("toy") else "DOWNLOADED"
        rec["n_samples_aligned"] = int(len(y))
        rec["model"] = cfg.model
        rec["ablation"] = "embargo"
        rows.append(rec)
    return pd.DataFrame(rows)


def run_benchmark(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    root = _project_root()
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    notes: list[str] = []

    toy = make_toy_panel(cfg.n_samples, cfg.horizon, cfg.seed)
    parts.append(
        run_on_panel(toy["close"], toy["ret"], cfg, tag="toy_synthetic")
    )
    notes.append("TOY synthetic panel ran.")

    if cfg.mode.lower() == "full":
        panel = load_full_frame(root, cfg.tickers)
        if panel is None:
            notes.append(
                "FULL mode requested but no cached OHLCV was found under "
                "data/raw; skipped FULL. Run scripts/download_data.py."
            )
        else:
            for col in panel.columns:
                s = panel[col].dropna()
                if s.size < 300:
                    notes.append(f"skipped {col}: too few rows after dropna")
                    continue
                ret = np.log(s).diff()
                try:
                    parts.append(run_on_panel(s, ret, cfg, tag=f"full_{col}"))
                    notes.append(f"FULL ran on {col} ({s.size} rows).")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"FULL failed on {col}: {exc}")
    else:
        notes.append("mode=toy; FULL download path not attempted.")

    table = pd.concat(parts, ignore_index=True)
    out_csv = tables / "benchmark_protocols.csv"
    table.to_csv(out_csv, index=False)
    notes_path = tables / "benchmark_notes.txt"
    notes_path.write_text("\n".join(notes) + "\n")
    return {"csv": str(out_csv), "notes": notes, "table": table}


def main() -> None:
    cfg = load_config()
    result = run_benchmark(cfg)
    print("Wrote", result["csv"])
    for line in result["notes"]:
        print(line)
    print(result["table"].to_string(index=False))


if __name__ == "__main__":
    main()
