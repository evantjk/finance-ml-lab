"""Deeper volatility study: HAR variants, GARCH-family, forecast combination,
across multiple horizons (5/21/63 days).

The question: can we push past the simple HAR-RV (OOS R^2 ~0.20), and does the
asymmetry / leverage effect (GJR, EGARCH) or combining forecasts help?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .. import utils

TRADING_DAYS = 252


def _rv(ret: pd.Series, h: int) -> pd.Series:
    return ret.rolling(h).std() * np.sqrt(TRADING_DAYS)


def _har_design(ret: pd.Series, leverage: bool) -> pd.DataFrame:
    d = pd.DataFrame({
        "rv_d": ret.abs() * np.sqrt(TRADING_DAYS),
        "rv_w": _rv(ret, 5),
        "rv_m": _rv(ret, 22),
    })
    if leverage:
        neg = ret.clip(upper=0).abs() * np.sqrt(TRADING_DAYS)  # downside-only
        d["rv_d_neg"] = neg
    return d


def _garch(ret: pd.Series, train_end: int, h: int, vol="Garch", o=0):
    try:
        from arch import arch_model
    except Exception:  # noqa: BLE001
        return None
    r = ret.dropna().values * 100.0
    try:
        am = arch_model(r, mean="Constant", vol=vol, p=1, o=o, q=1, dist="normal")
        fit = am.fit(last_obs=train_end, disp="off")
        fc = fit.forecast(horizon=h, start=0, reindex=False)
        return np.sqrt(fc.variance.values.mean(axis=1) / 1e4 * TRADING_DAYS)
    except Exception:  # noqa: BLE001
        return None


def run_one(ticker: str, df: pd.DataFrame, h: int) -> pd.DataFrame:
    ret = np.log(df["Close"]).diff()
    y = (ret.shift(-h).rolling(h).std() * np.sqrt(TRADING_DAYS))
    rv_now = _rv(ret, h)
    base = pd.DataFrame({"y": y, "rv_now": rv_now, "ret": ret}).dropna()
    idx = base.index
    n = len(base)
    tr, te = utils.final_holdout(n, frac=0.30)
    yv = base["y"].values
    ytr_mean = yv[tr].mean()

    def score(name, pred):
        pred = np.asarray(pred, float)
        return {
            "model": name,
            "RMSE": utils.rmse(yv[te], pred[te]),
            "QLIKE": utils.qlike(yv[te] ** 2, pred[te] ** 2),
            "R2_oos": utils.r2_oos(yv[te], pred[te], ytr_mean),
        }

    rows = [score("RW", base["rv_now"].values)]

    # EWMA(0.94)
    ew = np.sqrt(base["ret"].pow(2).ewm(alpha=0.06, adjust=False).mean() * TRADING_DAYS)
    rows.append(score("EWMA", ew.values))

    # HAR variants
    har_preds = {}
    for name, lev in [("HAR", False), ("HAR-lev", True)]:
        X = _har_design(base["ret"], lev).reindex(idx).ffill().fillna(0.0)
        lr = LinearRegression().fit(X.iloc[tr], yv[tr])
        p = lr.predict(X)
        har_preds[name] = p
        rows.append(score(name, p))
    # log-HAR
    Xl = np.log(_har_design(base["ret"], False).reindex(idx).ffill().bfill().clip(lower=1e-4))
    lr = LinearRegression().fit(Xl.iloc[tr], np.log(yv[tr]))
    p_log = np.exp(lr.predict(Xl))
    rows.append(score("log-HAR", p_log))

    # GARCH family
    garch_preds = {}
    for name, kw in [("GARCH", dict(vol="Garch", o=0)),
                     ("GJR-GARCH", dict(vol="Garch", o=1)),
                     ("EGARCH", dict(vol="EGARCH", o=1))]:
        g = _garch(base["ret"], train_end=int(tr[-1]) + 1, h=h, **kw)
        if g is not None and len(g) == n:
            garch_preds[name] = g
            rows.append(score(name, g))

    # forecast combination: average best HAR + best available GARCH
    combo_parts = [har_preds["HAR-lev"]]
    if "GJR-GARCH" in garch_preds:
        combo_parts.append(garch_preds["GJR-GARCH"])
    combo = np.mean(combo_parts, axis=0)
    rows.append(score("Combo(HAR-lev+GJR)", combo))

    # ML on HAR design + extras
    Xml = _har_design(base["ret"], True).reindex(idx).ffill().fillna(0.0)
    Xml["rv_now"] = base["rv_now"].values
    gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03,
                                       max_iter=400, l2_regularization=1.0, random_state=0)
    gb.fit(Xml.iloc[tr], yv[tr])
    rows.append(score("ML GBM", gb.predict(Xml)))

    out = pd.DataFrame(rows).set_index("model")
    out.columns = pd.MultiIndex.from_product([[f"h={h}"], out.columns])
    return out
