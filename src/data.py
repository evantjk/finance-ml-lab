"""Data acquisition layer.

Primary source: yfinance (split/dividend-adjusted daily OHLCV).
Fallback:       direct Yahoo Finance chart API (used if yfinance fails).

All data is cached to ``data/<TICKER>.csv`` so we download once and the
experiments are fully reproducible offline afterwards.
"""
from __future__ import annotations

import io
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _from_yahoo_api(ticker: str, years: int) -> pd.DataFrame:
    """Fallback: hit Yahoo's chart endpoint directly and parse JSON."""
    rng = f"{years}y"
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={rng}&interval=1d&events=div%2Csplit"
    )
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                ts = res["timestamp"]
                q = res["indicators"]["quote"][0]
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
                df = pd.DataFrame(
                    {
                        "Open": q["open"],
                        "High": q["high"],
                        "Low": q["low"],
                        "Close": q["close"],
                        "Volume": q["volume"],
                    },
                    index=pd.to_datetime(ts, unit="s").normalize(),
                )
                if adj is not None:
                    # rescale OHLC to the adjusted close so splits/divs are handled
                    factor = pd.Series(adj, index=df.index) / df["Close"]
                    for col in ["Open", "High", "Low", "Close"]:
                        df[col] = df[col] * factor
                df.index.name = "Date"
                return df.dropna()
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Yahoo API failed for {ticker}: {last_err}")


def _from_yfinance(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, period=f"{years}y", interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no rows")
    # yfinance may return a MultiIndex column frame for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df.dropna()


def load(ticker: str, years: int = 15, refresh: bool = False) -> pd.DataFrame:
    """Return cached daily OHLCV for ``ticker``, downloading if needed."""
    cache = DATA_DIR / f"{ticker}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col="Date", parse_dates=True)
        return df

    try:
        df = _from_yfinance(ticker, years)
        src = "yfinance"
    except Exception as e:  # noqa: BLE001
        print(f"  [{ticker}] yfinance failed ({e}); trying Yahoo API ...")
        df = _from_yahoo_api(ticker, years)
        src = "yahoo-api"
    df.to_csv(cache)
    print(f"  [{ticker}] {len(df):>5} rows  {df.index[0].date()} -> "
          f"{df.index[-1].date()}  (source: {src})")
    return df


def load_universe(tickers, years: int = 15, refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        out[t] = load(t, years=years, refresh=refresh)
    return out


if __name__ == "__main__":
    uni = load_universe(["AAPL", "MSFT", "JPM", "XOM", "SPY"])
    for t, df in uni.items():
        print(t, df.shape)
