"""Accuracy lever #1 — do VIX + cross-asset features improve vol forecasting?

Compares a HAR/technical baseline vs the same model + market features
(VIX level/change/richness, 10y-yield change, dollar change, HY-IG credit).
Honest test: R2_oos on a chronological hold-out, averaged over several names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, utils                          # noqa: E402
from src.tasks import volatility_advanced as va       # noqa: E402

NAMES = ["SPY", "AAPL", "JPM", "XOM", "MSFT", "JNJ"]
TD = 252


def _mkt(name):
    return pd.read_csv(data.DATA_DIR / f"_mkt_{name}.csv",
                       index_col="Date", parse_dates=True)["Close"]


def market_features(index) -> pd.DataFrame:
    vix = _mkt("VIX").reindex(index).ffill()
    tnx = _mkt("TNX_10y").reindex(index).ffill()
    dxy = _mkt("DXY").reindex(index).ffill()
    hyg = _mkt("HYG_hy").reindex(index).ffill()
    lqd = _mkt("LQD_ig").reindex(index).ffill()
    f = pd.DataFrame(index=index)
    f["vix"] = vix / 100.0                                  # ~annualised vol units
    f["vix_chg5"] = vix.pct_change(5)
    f["tnx_chg5"] = tnx.diff(5)
    f["dxy_chg5"] = dxy.pct_change(5)
    f["credit"] = (np.log(hyg) - np.log(lqd)).diff(5)       # HY-IG risk appetite
    return f


def run_one(ticker, h=21):
    df = data.load(ticker, years=15)
    ret = np.log(df["Close"]).diff()
    y = ret.shift(-h).rolling(h).std() * np.sqrt(TD)
    har = va._har_design(ret, leverage=True)
    base = pd.concat([har, (ret.rolling(h).std() * np.sqrt(TD)).rename("rv_now")], axis=1)
    mkt = market_features(df.index)
    full = pd.concat([base, mkt, y.rename("y")], axis=1).dropna()
    Xb, Xm, yv = full[base.columns], full[base.columns.tolist() + list(mkt.columns)], full["y"].values
    tr, te = utils.final_holdout(len(full), frac=0.30)
    ym = yv[tr].mean()

    def fit_r2(X, model):
        m = model()
        m.fit(X.iloc[tr], yv[tr])
        return utils.r2_oos(yv[te], m.predict(X.iloc[te]), ym)

    gbm = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03,
                  max_iter=400, l2_regularization=1.0, random_state=0)
    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    return {
        "GBM_base": fit_r2(Xb, gbm), "GBM_+mkt": fit_r2(Xm, gbm),
        "Ridge_base": fit_r2(Xb, ridge), "Ridge_+mkt": fit_r2(Xm, ridge),
    }


def main():
    rows = []
    for t in NAMES:
        r = run_one(t); r["ticker"] = t
        rows.append(r)
        print(f"  {t:5}  GBM {r['GBM_base']:.3f}->{r['GBM_+mkt']:.3f}  "
              f"Ridge {r['Ridge_base']:.3f}->{r['Ridge_+mkt']:.3f}")
    df = pd.DataFrame(rows).set_index("ticker")
    df["GBM_lift"] = df["GBM_+mkt"] - df["GBM_base"]
    df["Ridge_lift"] = df["Ridge_+mkt"] - df["Ridge_base"]
    df.loc["AVERAGE"] = df.mean()
    print("\n=== averages ===")
    print(df.loc["AVERAGE"].round(3).to_string())
    df.to_csv(ROOT / "reports" / "_vol_accuracy.csv")
    print("saved reports/_vol_accuracy.csv")
    return df


if __name__ == "__main__":
    main()
