"""Task 3 - Volatility forecasting (next-21d annualised realised vol).

Volatility clusters and is highly persistent, so this is where forecasting
genuinely works. We benchmark ML against the classics:
  * Random walk   - tomorrow's vol = today's realised vol
  * EWMA          - RiskMetrics, lambda = 0.94
  * HAR-RV        - Corsi (2009): regress on daily/weekly/monthly RV
  * GARCH(1,1)    - parameters fit on train, conditional variance updates OOS
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .. import utils

TRADING_DAYS = 252


def _ewma_vol(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    var = ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS)


def _har_features(ret: pd.Series) -> pd.DataFrame:
    """Daily/weekly/monthly realised-vol terms (annualised), causal."""
    rv_d = ret.abs() * np.sqrt(TRADING_DAYS)  # 1-day realised-vol proxy
    rv_w = ret.rolling(5).std() * np.sqrt(TRADING_DAYS)
    rv_m = ret.rolling(22).std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})


def _garch_forecast(ret: pd.Series, train_end: int, h: int) -> np.ndarray | None:
    """GARCH(1,1) annualised vol forecast over next ``h`` days at each origin.

    Parameters are estimated once on the training window (no look-ahead); the
    conditional variance then updates with realised returns out-of-sample.
    """
    try:
        from arch import arch_model
    except Exception:  # noqa: BLE001
        return None
    r = ret.dropna().values * 100.0  # arch prefers percent returns
    try:
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        fit = am.fit(last_obs=train_end, disp="off")
        fc = fit.forecast(horizon=h, start=0, reindex=False)
        # mean variance over the horizon, per origin -> annualised vol
        var_path = fc.variance.values  # shape (n_origins, h), in percent^2
        ann = np.sqrt(var_path.mean(axis=1) / 1e4 * TRADING_DAYS)
        return ann
    except Exception:  # noqa: BLE001
        return None


def run(ticker: str, X: pd.DataFrame, T: pd.DataFrame, vol_h: int = 21) -> dict:
    y = T["y_vol"].values
    ret = T["_ret"]
    n = len(X)
    tr, te = utils.final_holdout(n, frac=0.30)

    rows = []
    preds = {}

    def add(name, p):
        p = np.asarray(p, dtype=float)
        rows.append({
            "model": name,
            "RMSE": utils.rmse(y[te], p[te] if len(p) == n else p),
            "QLIKE": utils.qlike(y[te] ** 2, (p[te] if len(p) == n else p) ** 2),
            "R2_oos": utils.r2_oos(y[te], p[te] if len(p) == n else p, y[tr].mean()),
        })
        preds[name] = p[te] if len(p) == n else p

    # --- baselines ---
    add("Baseline: random-walk", T["rv_now"].values)
    add("Baseline: EWMA(0.94)", _ewma_vol(ret).values)

    # HAR-RV (OLS fit on train)
    har = _har_features(ret)
    mask = har.notna().all(axis=1).values
    har_lr = LinearRegression()
    fit_idx = np.intersect1d(tr, np.where(mask)[0])
    har_lr.fit(har.iloc[fit_idx], y[fit_idx])
    har_pred = har_lr.predict(har.ffill().fillna(0.0))
    add("HAR-RV (OLS)", har_pred)

    # GARCH(1,1)
    g = _garch_forecast(ret, train_end=int(tr[-1]) + 1, h=vol_h)
    if g is not None and len(g) == n:
        add("GARCH(1,1)", g)

    # --- ML: gradient boosting on full feature set + HAR terms ---
    Xv = pd.concat([X, har], axis=1).ffill().fillna(0.0)
    gb = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=400, l2_regularization=1.0, random_state=0
    )
    gb.fit(Xv.iloc[tr], y[tr])
    add("ML: GradBoost+HAR", gb.predict(Xv).astype(float))

    res = pd.DataFrame(rows).set_index("model")
    return {
        "ticker": ticker, "metrics": res, "preds": preds,
        "y_test": y[te], "test_index": T.index[te],
    }
