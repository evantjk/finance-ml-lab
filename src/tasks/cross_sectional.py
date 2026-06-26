"""Cross-sectional stock-ranking model with walk-forward evaluation.

Metrics:
  * IC      - mean per-date Spearman rank corr(prediction, forward return)
  * IC-IR   - IC mean / IC std * sqrt(periods/yr)  (signal stability)
  * L/S     - long top-quintile, short bottom-quintile, net of costs
We benchmark the ML ranker against single-factor baselines (12-1 momentum,
short-term reversal) — if ML can't beat a one-line factor, it isn't earning its
keep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from ..panel import SIGNAL_COLS

PERIODS_PER_YEAR = 252 / 21  # ~12 monthly rebalances


def _walk_forward_predict(panel: pd.DataFrame, init_years: int = 5,
                          step_months: int = 12) -> pd.Series:
    """Expanding-window OOS predictions; retrain every ``step_months``."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    init = int(init_years * PERIODS_PER_YEAR)
    preds = []
    i = init
    while i < len(dates):
        train_dates = dates[:i]
        test_dates = dates[i:i + step_months]
        tr = panel[panel.index.get_level_values("date").isin(train_dates)]
        te = panel[panel.index.get_level_values("date").isin(test_dates)]
        if len(te) == 0:
            break
        mdl = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300,
            l2_regularization=1.0, random_state=0,
        )
        mdl.fit(tr[SIGNAL_COLS], tr["fwd_rel"])
        preds.append(pd.Series(mdl.predict(te[SIGNAL_COLS]), index=te.index))
        i += step_months
    return pd.concat(preds).rename("pred")


def _ic_series(pred: pd.Series, fwd: pd.Series) -> pd.Series:
    df = pd.concat([pred, fwd.rename("fwd")], axis=1).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) >= 5:
            out[d] = spearmanr(g["pred"], g["fwd"]).statistic
    return pd.Series(out).sort_index()


def _long_short(score: pd.Series, fwd_ret: pd.Series, q: float = 0.2,
                cost_bps: float = 10.0) -> pd.Series:
    """Per-rebalance L/S return: long top-q, short bottom-q, equal weight."""
    df = pd.concat([score.rename("s"), fwd_ret.rename("r")], axis=1).dropna()
    rets, prev_long, prev_short = {}, set(), set()
    for d, g in df.groupby(level="date"):
        n = max(1, int(len(g) * q))
        g = g.sort_values("s")
        shorts = g.head(n); longs = g.tail(n)
        gross = longs["r"].mean() - shorts["r"].mean()
        long_set = set(longs.index.get_level_values("ticker"))
        short_set = set(shorts.index.get_level_values("ticker"))
        turn = (len(long_set ^ prev_long) + len(short_set ^ prev_short)) / (2 * n)
        rets[d] = gross - turn * (cost_bps / 1e4)
        prev_long, prev_short = long_set, short_set
    return pd.Series(rets).sort_index()


def _stats(ls: pd.Series, ic: pd.Series) -> dict:
    ls = ls.dropna()
    ann = (1 + ls).prod() ** (PERIODS_PER_YEAR / len(ls)) - 1 if len(ls) else np.nan
    sharpe = ls.mean() / ls.std() * np.sqrt(PERIODS_PER_YEAR) if ls.std() > 0 else 0.0
    return {
        "IC": float(ic.mean()),
        "IC_IR": float(ic.mean() / ic.std() * np.sqrt(PERIODS_PER_YEAR)) if ic.std() > 0 else 0.0,
        "LS_ann": float(ann),
        "LS_Sharpe": float(sharpe),
        "LS_hit": float((ls > 0).mean()),
    }


def run(panel: pd.DataFrame) -> dict:
    fwd_rel = panel["fwd_rel"]
    fwd_ret = panel["fwd_ret"]

    # ML ranker (walk-forward OOS)
    pred = _walk_forward_predict(panel)
    oos = panel.loc[pred.index]
    rows, curves = {}, {}

    ml_ic = _ic_series(pred, oos["fwd_rel"])
    ml_ls = _long_short(pred, oos["fwd_ret"])
    rows["ML ranker (GBM)"] = _stats(ml_ls, ml_ic)
    curves["ML ranker (GBM)"] = ml_ls

    # factor baselines on the SAME out-of-sample dates
    for name, col, sign in [("Factor: 12-1 momentum", "mom_12_1", 1),
                            ("Factor: short-rev", "st_rev", 1)]:
        s = sign * oos[col]
        ic = _ic_series(s.rename("pred"), oos["fwd_rel"])
        ls = _long_short(s, oos["fwd_ret"])
        rows[name] = _stats(ls, ic)
        curves[name] = ls

    return {
        "metrics": pd.DataFrame(rows).T,
        "curves": curves,
        "ml_ic_series": ml_ic,
        "oos_dates": (pred.index.get_level_values("date").min(),
                      pred.index.get_level_values("date").max()),
        "n_obs": len(pred),
    }
