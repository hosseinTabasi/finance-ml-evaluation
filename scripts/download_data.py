#!/usr/bin/env python3
"""Download public daily series with a local cache and checksum printout.

Sources (in order of attempt per ticker):
  * Yahoo Finance chart API (unofficial, may fail)
  * Stooq daily CSV
  * Ken French F-F daily factors (always attempted once)

Failures are printed and skipped. Nothing in this script is required
for the TOY experiment. FULL-mode benchmarks read whatever was cached
under data/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = "finmlcv/0.1 research cache (academic; not a scraper farm)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("bb") if False else path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def save_frame(df: pd.DataFrame, stem: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{stem}.csv"
    df.to_csv(path)
    digest = sha256_file(path)
    print(f"  wrote {path}  sha256={digest}  rows={len(df)}")
    return path


def fetch_stooq(ticker: str) -> pd.DataFrame:
    # Stooq uses a suffix; US equities are ticker.us. BTC is btcusd.
    aliases = {
        "BTC-USD": "btcusd",
        "BTCUSD": "btcusd",
        "SPY": "spy.us",
        "QQQ": "qqq.us",
        "IWM": "iwm.us",
        "^GSPC": "^spx",
    }
    code = aliases.get(ticker.upper(), ticker.lower())
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    raw = _get(url)
    df = pd.read_csv(io.BytesIO(raw))
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"stooq returned no Close column for {ticker}")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"close": "close"})
    return df


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    import json
    import time

    period2 = int(time.time())
    period1 = 0
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?period1={period1}&period2={period2}&interval=1d"
        "&events=history"
    )
    raw = _get(url)
    payload = json.loads(raw.decode("utf-8"))
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize(),
    )
    if adj is not None:
        df["adj_close"] = adj
    df = df.dropna(how="all")
    if df.empty:
        raise RuntimeError(f"yahoo returned empty frame for {ticker}")
    return df


def fetch_french_daily() -> pd.DataFrame:
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    )
    blob = _get(url, timeout=60)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin-1")
    # Skip header until the first data row (yyyymmdd).
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip()[:8].isdigit():
            start = i
            break
    body = "\n".join(lines[start:])
    df = pd.read_csv(io.StringIO(body), header=None)
    # The file has a copyright footer; keep rows whose first col is 8 digits.
    df = df[df.iloc[:, 0].astype(str).str.fullmatch(r"\d{8}", na=False)]
    df.columns = ["date", "mkt_rf", "smb", "hml", "rf"][: df.shape[1]]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["SPY", "QQQ", "BTC-USD"],
        help="Yahoo/Stooq tickers to attempt",
    )
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"cache directory: {RAW}")

    try:
        ff = fetch_french_daily()
        save_frame(ff, "french_ff_daily")
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Ken French download failed (skipped): {exc}")

    rc = 0
    for tkr in args.tickers:
        stem = tkr.replace("/", "-")
        ok = False
        for name, fn in (("yahoo", fetch_yahoo), ("stooq", fetch_stooq)):
            try:
                df = fn(tkr)
                save_frame(df, stem)
                print(f"  {tkr}: source={name}")
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {tkr}: {name} failed: {exc}")
        if not ok:
            print(f"  {tkr}: skipped (no source succeeded)")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
