# Data

This directory is a *cache*. Raw vendor files are gitignored.

## Layout

- `raw/` — CSV/Parquet written by `scripts/download_data.py`.
- Nothing under `raw/` is required for the TOY experiment.

## Sources (FULL, optional)

| stem | source | note |
| --- | --- | --- |
| `french_ff_daily.csv` | Ken French data library, Fama–French daily factors | percent converted to decimals |
| `SPY.csv`, `QQQ.csv`, `BTC-USD.csv` | Yahoo chart API, then Stooq | fail-soft |

## Checksums

After a successful download the script prints a SHA-256 of each file.
Record those values here when you run FULL:

```
# placeholder — fill in after scripts/download_data.py
# french_ff_daily.csv  sha256=...
# SPY.csv              sha256=...
# QQQ.csv              sha256=...
# BTC-USD.csv          sha256=...
```

FI-2010 (limit-order-book) is **not** bundled. It is a separate research
dataset with its own licence; this repository does not ship it.

## Alignment rule

Drop/align NaNs **before** splitting. See
`finmlcv.experiments.run_benchmark.align_xy`.
