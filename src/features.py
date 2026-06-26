"""Feature engineering and target construction.

CRITICAL RULE (no look-ahead): every *feature* column is built only from
information available up to and including day ``t``. Every *target* column
is a function of days strictly after ``t``. The two are then aligned so the
model at day ``t`` predicts the future without ever seeing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# technical indicators (all causal: use only past/current data)
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _parkinson_vol(high: pd.Series, low: pd.Series, n: int = 10) -> pd.Series:
    """Parkinson high-low range volatility estimator (annualised)."""
    hl = np.log(high / low) ** 2
    var = hl.rolling(n).mean() / (4 * np.log(2))
    return np.sqrt(var * TRADING_DAYS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of causal predictive features indexed like ``df``."""
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ret = np.log(close).diff()  # daily log return

    f = pd.DataFrame(index=df.index)

    # --- lagged returns (momentum / reversal signals) ---
    for k in (1, 2, 3, 5, 10):
        f[f"ret_lag{k}"] = ret.shift(k - 1) if k == 1 else ret.shift(1).rolling(k).sum()

    # --- multi-horizon momentum ---
    for k in (21, 63, 126):
        f[f"mom_{k}"] = np.log(close / close.shift(k))

    # --- rolling realised volatility of returns ---
    for k in (5, 10, 21, 63):
        f[f"vol_{k}"] = ret.rolling(k).std() * np.sqrt(TRADING_DAYS)

    # --- price relative to moving averages (trend) ---
    for k in (20, 50, 200):
        f[f"px_vs_ma{k}"] = close / close.rolling(k).mean() - 1

    # --- oscillators ---
    f["rsi_14"] = _rsi(close, 14)
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd"] = macd / close
    f["macd_sig"] = (macd - macd.ewm(span=9).mean()) / close

    # --- range / volume ---
    f["parkinson_10"] = _parkinson_vol(high, low, 10)
    f["hl_range"] = (high - low) / close
    logv = np.log(vol.replace(0, np.nan))
    f["vol_z"] = (logv - logv.rolling(21).mean()) / logv.rolling(21).std()
    f["dollar_vol_chg"] = (close * vol).pct_change(5)

    # --- calendar ---
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month

    f["_ret"] = ret  # keep raw return around for target construction
    return f


# --------------------------------------------------------------------------- #
# targets (functions of the FUTURE)
# --------------------------------------------------------------------------- #
def add_targets(f: pd.DataFrame, horizon: int = 1, vol_h: int = 21) -> pd.DataFrame:
    ret = f["_ret"]
    out = f.copy()

    # future cumulative log-return over `horizon` days
    fwd = ret.shift(-horizon).rolling(horizon).sum() if horizon > 1 else ret.shift(-1)
    out["y_ret"] = fwd                         # regression target
    out["y_dir"] = (fwd > 0).astype("float")   # classification target (up=1)

    # future realised volatility over the next `vol_h` days (annualised)
    fut_vol = ret.shift(-vol_h).rolling(vol_h).std() * np.sqrt(TRADING_DAYS)
    out["y_vol"] = fut_vol
    # current realised vol -> the random-walk baseline for the vol task
    out["rv_now"] = ret.rolling(vol_h).std() * np.sqrt(TRADING_DAYS)
    return out


def make_dataset(df: pd.DataFrame, horizon: int = 1, vol_h: int = 21):
    """Return (X, targets_df) cleaned of NaNs introduced by windows/shifts."""
    f = add_targets(build_features(df), horizon=horizon, vol_h=vol_h)
    feat_cols = [c for c in f.columns if not c.startswith("y_") and c not in ("_ret", "rv_now")]
    targets = ["y_ret", "y_dir", "y_vol", "rv_now", "_ret"]
    data = f[feat_cols + targets].dropna()
    return data[feat_cols], data[targets]
