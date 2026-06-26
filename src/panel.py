"""Cross-sectional panel construction.

Instead of asking "what will AAPL do?", we ask the question that actually has
signal: "which names will out-perform the others over the next month?"

For each monthly rebalance date we compute per-stock signals, normalise them
*cross-sectionally* (z-score across the universe that day), and target the
**relative** forward return (return minus the universe mean). A long-top /
short-bottom book on that target is market-neutral by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

TRADING_DAYS = 252
REBAL = 21  # trading days between rebalances (~1 month)


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-stock signals (known at day t)."""
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ret = np.log(close).diff()
    f = pd.DataFrame(index=df.index)
    # cross-sectional momentum (skip most recent month to avoid st-reversal)
    f["mom_12_1"] = np.log(close.shift(REBAL) / close.shift(252))
    f["mom_6_1"] = np.log(close.shift(REBAL) / close.shift(126))
    f["mom_1m"] = np.log(close / close.shift(REBAL))
    f["st_rev"] = -np.log(close / close.shift(5))          # short-term reversal
    f["vol_21"] = ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_63"] = ret.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f["px_vs_ma200"] = close / close.rolling(200).mean() - 1
    f["hi_52w"] = close / close.rolling(252).max() - 1     # distance from 1y high
    f["dollar_vol"] = np.log((close * vol).rolling(21).mean())
    rng = np.log(high / low)
    f["range_21"] = rng.rolling(21).mean()
    # forward 1-month return (the thing we want to rank)
    f["fwd_ret"] = close.shift(-REBAL) / close - 1
    return f


SIGNAL_COLS = ["mom_12_1", "mom_6_1", "mom_1m", "st_rev", "vol_21", "vol_63",
               "px_vs_ma200", "hi_52w", "dollar_vol", "range_21"]


def build_panel(tickers, years: int = 15, min_names: int = 20) -> pd.DataFrame:
    """Return a (date, ticker)-indexed panel sampled at monthly rebalances.

    Features are z-scored within each date; ``fwd_ret`` is left raw and
    ``fwd_rel`` is the cross-sectionally demeaned (market-neutral) target.
    """
    frames = []
    for t in tickers:
        s = _signals(data.load(t, years=years))
        s["ticker"] = t
        frames.append(s)
    panel = pd.concat(frames).set_index("ticker", append=True)
    panel.index.names = ["date", "ticker"]

    # sample monthly on a common calendar (every REBAL-th trading day)
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    rebal_dates = all_dates[::REBAL]
    panel = panel[panel.index.get_level_values("date").isin(rebal_dates)]
    panel = panel.dropna(subset=SIGNAL_COLS + ["fwd_ret"])

    # require a minimum cross-section each date
    counts = panel.groupby(level="date").size()
    good = counts[counts >= min_names].index
    panel = panel[panel.index.get_level_values("date").isin(good)]

    # preserve raw volatility (for risk-based sizing) before z-scoring
    panel["vol_raw"] = panel["vol_21"]

    # attach GICS sector if the S&P 500 metadata is available
    meta_path = data.DATA_DIR / "sp500_meta.csv"
    if meta_path.exists():
        sec = pd.read_csv(meta_path).set_index("ticker")["sector"]
        tick = panel.index.get_level_values("ticker")
        panel["sector"] = sec.reindex(tick).values
        panel["sector"] = panel["sector"].fillna("Unknown")
    else:
        panel["sector"] = "Unknown"

    # cross-sectional z-score of features + demeaned target, per date
    def _xs(g):
        z = (g[SIGNAL_COLS] - g[SIGNAL_COLS].mean()) / g[SIGNAL_COLS].std(ddof=0)
        g = g.copy()
        g[SIGNAL_COLS] = z.clip(-3, 3)             # winsorise outliers
        g["fwd_rel"] = g["fwd_ret"] - g["fwd_ret"].mean()
        return g

    panel = panel.groupby(level="date", group_keys=False).apply(_xs)
    return panel
