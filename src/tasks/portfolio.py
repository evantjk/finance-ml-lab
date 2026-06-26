"""Portfolio construction on the cross-sectional ranker.

A raw signal is not a strategy. Here we turn the ML rank forecast into an
actual long/short book and measure what each construction step adds:

  A. Naive       - long top-quintile / short bottom-quintile, equal weight
  B. +Sector-neutral - rank on sector-demeaned scores (remove sector bets)
  C. +Risk-sized     - inverse-vol weights within each leg, dollar-neutral

Sharpe is leverage-invariant, so improvements come from *which names get weight*
(neutralisation + risk sizing), not from scaling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..panel import SIGNAL_COLS
from . import cross_sectional as xs

PPY = xs.PERIODS_PER_YEAR


def _sector_neutral(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    df = pd.concat([scores.rename("s"), sectors.rename("sec")], axis=1)
    return df.groupby([df.index.get_level_values("date"), "sec"])["s"].transform(
        lambda x: x - x.mean()
    )


def _book(score, fwd, vol, sector, q=0.2, cost_bps=10.0,
          neutralize=False, risk_size=False):
    """Return per-rebalance net strategy returns for one construction variant."""
    if neutralize:
        score = _sector_neutral(score, sector)
    df = pd.concat([score.rename("s"), fwd.rename("r"), vol.rename("v")], axis=1).dropna()
    rets, turnover, prev_w = {}, {}, pd.Series(dtype=float)
    for d, g in df.groupby(level="date"):
        n = max(1, int(len(g) * q))
        g = g.sort_values("s")
        shorts, longs = g.head(n), g.tail(n)
        if risk_size:
            wl = (1 / longs["v"].clip(lower=1e-3)); wl /= wl.sum()
            ws = (1 / shorts["v"].clip(lower=1e-3)); ws /= ws.sum()
        else:
            wl = pd.Series(1 / n, index=longs.index)
            ws = pd.Series(1 / n, index=shorts.index)
        w = pd.concat([wl, -ws])                      # +1 long, -1 short (dollar-neutral)
        rets[d] = float((w * pd.concat([longs["r"], shorts["r"]])).sum())
        aligned = w.reindex(prev_w.index.union(w.index)).fillna(0.0)
        prev = prev_w.reindex(aligned.index).fillna(0.0)
        turnover[d] = float((aligned - prev).abs().sum())
        prev_w = w
    r = pd.Series(rets).sort_index()
    c = pd.Series(turnover).sort_index() * (cost_bps / 1e4)
    return (r - c)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann = (1 + r).prod() ** (PPY / len(r)) - 1 if len(r) else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"Ann": float(ann), "Sharpe": float(sharpe), "MaxDD": float(dd),
            "Vol": float(r.std() * np.sqrt(PPY)), "Hit": float((r > 0).mean())}


def run(panel: pd.DataFrame) -> dict:
    pred = xs._walk_forward_predict(panel)
    oos = panel.loc[pred.index]
    fwd, vol, sector = oos["fwd_ret"], oos["vol_raw"], oos["sector"]

    variants = {
        "A. Naive decile L/S": dict(neutralize=False, risk_size=False),
        "B. + Sector-neutral": dict(neutralize=True, risk_size=False),
        "C. + Risk-sized (inv-vol)": dict(neutralize=True, risk_size=True),
    }
    rows, curves = {}, {}
    for name, kw in variants.items():
        r = _book(pred, fwd, vol, sector, **kw)
        rows[name] = _stats(r)
        curves[name] = r

    ic = xs._ic_series(pred, oos["fwd_rel"])
    return {
        "metrics": pd.DataFrame(rows).T,
        "curves": curves,
        "IC": float(ic.mean()),
        "n_names": panel.index.get_level_values("ticker").nunique(),
        "n_obs": len(pred),
        "oos_dates": (pred.index.get_level_values("date").min(),
                      pred.index.get_level_values("date").max()),
    }
