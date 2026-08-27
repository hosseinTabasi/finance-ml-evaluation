#!/usr/bin/env python3
"""Render paper figures from the TOY CSV artefacts.

Does not invent numbers: every bar is read from results/tables/*.csv.
If a CSV is missing, the corresponding toy runner is invoked first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"


def _ensure_csvs() -> None:
    if not (TABLES / "synthetic_leakage.csv").exists():
        from finmlcv.experiments.synthetic_leakage import run_synthetic_leakage

        run_synthetic_leakage()
    if not (TABLES / "benchmark_protocols.csv").exists():
        from finmlcv.experiments.run_benchmark import run_benchmark

        run_benchmark()


def fig_synthetic_leakage() -> Path:
    from finmlcv.experiments.synthetic_leakage import _render_figure

    table = pd.read_csv(TABLES / "synthetic_leakage.csv")
    path = FIGS / "synthetic_leakage.png"
    return _render_figure(table, path)


def fig_future_column() -> Path:
    table = pd.read_csv(TABLES / "synthetic_leakage.csv")
    want = table[table["dgp"].isin(["future_column", "future_column_dropped"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    protocols = [
        "naive_kfold_INVALID",
        "timeseries_split",
        "purged_kfold",
        "cpcv",
    ]
    labels = ["Naive KFold", "TimeSeriesSplit", "Purged KFold", "CPCV"]
    x = np.arange(len(protocols))
    w = 0.36
    leak = want[want["dgp"] == "future_column"].set_index("protocol")
    drop = want[want["dgp"] == "future_column_dropped"].set_index("protocol")
    ax.bar(
        x - w / 2,
        [leak.loc[p, "auc_mean"] for p in protocols],
        w,
        label="with x_leak = y[t+1]",
        color="#b23a48",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        [drop.loc[p, "auc_mean"] for p in protocols],
        w,
        label="x_leak dropped (auditor)",
        color="#2f6f4e",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("Mean ROC AUC (TOY)")
    ax.set_title("TOY: an explicit future column inflates every splitter")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIGS / "future_column_leak.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_embargo_sensitivity() -> Path:
    table = pd.read_csv(TABLES / "benchmark_protocols.csv")
    sub = table[
        (table["tag"] == "toy_synthetic")
        & (table.get("ablation", pd.Series(index=table.index, dtype=object)) == "embargo")
        if "ablation" in table.columns
        else (table["tag"] == "toy_synthetic")
        & table["protocol"].astype(str).str.startswith("purged_kfold_embargo")
    ]
    if sub.empty:
        sub = table[
            (table["tag"] == "toy_synthetic")
            & table["protocol"].astype(str).str.contains("embargo")
        ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        sub["embargo_pct"].to_numpy() * 100.0,
        sub["auc_mean"].to_numpy(),
        marker="o",
        color="#1d4e89",
        label="AUC",
    )
    ax.set_xlabel("Embargo (% of sample size)")
    ax.set_ylabel("Mean ROC AUC (TOY)")
    ax.set_title("TOY: embargo sensitivity, purged k-fold, synthetic panel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGS / "embargo_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_protocol_sharpe() -> Path:
    table = pd.read_csv(TABLES / "benchmark_protocols.csv")
    sub = table[
        (table["tag"] == "toy_synthetic")
        & (table["protocol"].isin(
            ["naive_kfold_INVALID", "timeseries_split", "purged_kfold", "cpcv"]
        ))
    ]
    order = ["naive_kfold_INVALID", "timeseries_split", "purged_kfold", "cpcv"]
    sub = sub.set_index("protocol").loc[order]
    labels = ["Naive KFold\n(INVALID)", "TimeSeriesSplit", "Purged KFold", "CPCV"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(order))
    ax.bar(
        x,
        sub["sharpe_mean"].to_numpy(),
        yerr=sub["sharpe_std"].to_numpy(),
        capsize=4,
        color=["#b23a48", "#6c757d", "#2f6f4e", "#1d4e89"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Sharpe of long/short overlay (TOY)")
    ax.set_title("TOY synthetic panel: protocol contrast (not live PnL)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIGS / "protocol_sharpe_toy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    _ensure_csvs()
    paths = [
        fig_synthetic_leakage(),
        fig_future_column(),
        fig_embargo_sensitivity(),
        fig_protocol_sharpe(),
    ]
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
