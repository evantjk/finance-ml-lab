"""Fetch current S&P 500 constituents (+ GICS sector) and bulk-download history.

NOTE on survivorship bias: Wikipedia lists *today's* members, so any backtest on
this set only sees firms that survived to today -> optimistic. We flag this in the
reports; a point-in-time membership feed (paid) would be needed to remove it.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    tbl = pd.read_html(io.StringIO(html))[0]
    tbl = tbl.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                              "GICS Sub-Industry": "sub_industry"})
    tbl["ticker"] = tbl["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    cols = ["ticker", "sector"] + (["sub_industry"] if "sub_industry" in tbl else [])
    meta = tbl[cols].dropna(subset=["ticker", "sector"]).drop_duplicates("ticker")
    if "sub_industry" not in meta:
        meta["sub_industry"] = meta["sector"]
    meta.to_csv(DATA / "sp500_meta.csv", index=False)
    print(f"  constituents: {len(meta)}  sectors: {meta['sector'].nunique()}  "
          f"sub-industries: {meta['sub_industry'].nunique()}")
    return meta


def bulk_download(tickers, years=15, chunk=40):
    have = {p.stem for p in DATA.glob("*.csv")}
    todo = [t for t in tickers if t not in have]
    print(f"  cached: {len(tickers) - len(todo)} | to download: {len(todo)}")
    ok = 0
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        try:
            df = yf.download(batch, period=f"{years}y", interval="1d",
                             auto_adjust=True, progress=False, threads=True, group_by="ticker")
        except Exception as e:  # noqa: BLE001
            print(f"   batch {i//chunk} error: {repr(e)[:80]}")
            time.sleep(3); continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub) > 200:
                    sub.index.name = "Date"
                    sub.to_csv(DATA / f"{t}.csv")
                    ok += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"   chunk {i//chunk+1}/{(len(todo)+chunk-1)//chunk}: saved so far {ok}")
        time.sleep(1.5)
    print(f"  downloaded {ok}/{len(todo)} new")


if __name__ == "__main__":
    meta = fetch_constituents()
    bulk_download(meta["ticker"].tolist())
    have = {p.stem for p in DATA.glob("*.csv")}
    usable = [t for t in meta["ticker"] if t in have]
    print(f"\nUsable S&P 500 names with data: {len(usable)}")
