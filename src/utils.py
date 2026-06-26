"""Metrics and leakage-safe time-series evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def walk_forward_folds(n: int, n_splits: int = 5, embargo: int = 5):
    """Yield (train_idx, test_idx) for an expanding-window walk-forward.

    An ``embargo`` gap of rows is dropped between train and test so that a
    multi-day target horizon cannot leak future info into the training set.
    """
    fold = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_start = train_end + embargo
        test_end = min(fold * (i + 1) + (fold if i == n_splits else 0), n)
        if test_start >= test_end:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def final_holdout(n: int, frac: float = 0.2, embargo: int = 5):
    """Single chronological train/test split with an embargo gap."""
    cut = int(n * (1 - frac))
    return np.arange(0, cut - embargo), np.arange(cut, n)


# --------------------------------------------------------------------------- #
# regression / classification metrics
# --------------------------------------------------------------------------- #
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2_oos(y, p, y_train_mean):
    """Out-of-sample R^2 vs predicting the *training* mean (Campbell-Thompson).

    Honest benchmark: a positive value means the model beats the naive
    'predict the historical average' forecast on unseen data.
    """
    y, p = np.asarray(y), np.asarray(p)
    sse = np.sum((y - p) ** 2)
    sst = np.sum((y - y_train_mean) ** 2)
    return float(1 - sse / sst)


def directional_accuracy(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.sign(y) == np.sign(p)))


def qlike(y_true_var, y_pred_var):
    """QLIKE loss for variance forecasts (robust to noise in the proxy).

    Lower is better. Operates on variances (vol**2).
    """
    yt = np.asarray(y_true_var)
    yp = np.clip(np.asarray(y_pred_var), 1e-8, None)
    return float(np.mean(yt / yp - np.log(yt / yp) - 1))


# --------------------------------------------------------------------------- #
# backtest statistics
# --------------------------------------------------------------------------- #
def perf_stats(strategy_ret: pd.Series, freq: int = 252) -> dict:
    r = strategy_ret.dropna()
    if len(r) == 0:
        return {}
    cum = (1 + r).prod()
    yrs = len(r) / freq
    cagr = cum ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sharpe = (r.mean() / r.std() * np.sqrt(freq)) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd),
        "Vol_ann": float(r.std() * np.sqrt(freq)),
        "HitRate": float((r > 0).mean()),
    }
