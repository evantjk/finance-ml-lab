"""Accuracy levers #2/#3 — improve the cross-sectional ranker's IC.

  base        : current 10 price/technical signals
  + res-mom   : add beta-adjusted (residual) 12-1 momentum
  ensemble    : average Ridge + GBM + RandomForest on the enhanced set
  conviction  : does IC rise among high-|prediction| names? (meta-labeling premise)

IC = mean per-month Spearman corr(prediction, next-month relative return), OOS,
expanding walk-forward retrained yearly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, panel as panel_mod              # noqa: E402
from src.panel import SIGNAL_COLS, REBAL              # noqa: E402
from src.tasks.cross_sectional import PERIODS_PER_YEAR  # noqa: E402

TD = 252


def residual_momentum(tickers, dates_index) -> pd.Series:
    """Beta-adjusted 12-1 momentum per (date,ticker), causal."""
    spy = np.log(data.load("SPY", years=15)["Close"]).diff()
    out = {}
    for t in tickers:
        r = np.log(data.load(t, years=15)["Close"]).diff()
        df = pd.concat([r.rename("r"), spy.rename("m")], axis=1).dropna()
        cov = df["r"].rolling(63).cov(df["m"])
        var = df["m"].rolling(63).var()
        beta = (cov / var).shift(1)                       # lag to avoid look-ahead
        resid = df["r"] - beta * df["m"]
        resmom = resid.shift(REBAL).rolling(252 - REBAL).sum()   # 12-1 residual mom
        out[t] = resmom
    s = pd.concat(out, names=["ticker"]).swaplevel().sort_index()
    s.index.names = ["date", "ticker"]
    return s.rename("res_mom")


def walk_forward(panel, cols, model_factory, init_years=4, step=12) -> pd.Series:
    dates = panel.index.get_level_values("date").unique().sort_values()
    init = int(init_years * PERIODS_PER_YEAR)
    preds, i = [], init
    while i < len(dates):
        tr = panel[panel.index.get_level_values("date").isin(dates[:i])]
        te = panel[panel.index.get_level_values("date").isin(dates[i:i + step])]
        if len(te) == 0:
            break
        m = model_factory()
        m.fit(tr[cols].fillna(0.0), tr["fwd_rel"])
        preds.append(pd.Series(m.predict(te[cols].fillna(0.0)), index=te.index))
        i += step
    return pd.concat(preds).rename("pred")


def ic(pred, fwd):
    df = pd.concat([pred, fwd.rename("f")], axis=1).dropna()
    vals = [spearmanr(g["pred"], g["f"]).statistic for _, g in df.groupby(level="date") if len(g) >= 5]
    s = pd.Series(vals)
    return s.mean(), s.mean() / s.std() * np.sqrt(PERIODS_PER_YEAR)


def main():
    meta = pd.read_csv(data.DATA_DIR / "sp500_meta.csv")
    have = {p.stem for p in data.DATA_DIR.glob("*.csv")}
    tickers = [t for t in meta["ticker"] if t in have]
    print(f"Universe {len(tickers)}. Building panel ...")
    panel = panel_mod.build_panel(tickers, years=15, min_names=100)

    # add residual momentum, z-scored cross-sectionally per date
    print("Computing residual (beta-adjusted) momentum ...")
    rm = residual_momentum(tickers, panel.index).reindex(panel.index)
    panel["res_mom"] = rm.groupby(level="date").transform(
        lambda x: ((x - x.mean()) / (x.std(ddof=0) + 1e-9)).clip(-3, 3))
    panel = panel.dropna(subset=["res_mom"])

    gbm = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03,
                  max_iter=300, l2_regularization=1.0, random_state=0)
    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    rf = lambda: RandomForestRegressor(n_estimators=200, max_depth=5,
                 min_samples_leaf=50, n_jobs=-1, random_state=0)

    rows = {}
    # 1) baseline
    p = walk_forward(panel, SIGNAL_COLS, gbm)
    oos = panel.loc[p.index]
    m, ir = ic(p, oos["fwd_rel"]); rows["base (10 signals, GBM)"] = (m, ir)

    # 2) + residual momentum
    cols2 = SIGNAL_COLS + ["res_mom"]
    p2 = walk_forward(panel, cols2, gbm)
    m, ir = ic(p2, panel.loc[p2.index]["fwd_rel"]); rows["+ residual momentum"] = (m, ir)

    # 3) ensemble (avg of 3 models on enhanced set)
    pe = sum(walk_forward(panel, cols2, f) for f in (gbm, ridge, rf)) / 3.0
    m, ir = ic(pe, panel.loc[pe.index]["fwd_rel"]); rows["ensemble (GBM+Ridge+RF)"] = (m, ir)

    # 4) conviction buckets on the ensemble (meta-labeling premise)
    oos2 = panel.loc[pe.index].copy(); oos2["pred"] = pe
    conv = {}
    for d, g in oos2.groupby(level="date"):
        g = g.assign(absp=g["pred"].abs())
        hi = g.nlargest(max(5, len(g) // 3), "absp")
        lo = g.nsmallest(max(5, len(g) // 3), "absp")
        if len(hi) >= 5:
            conv.setdefault("hi", []).append(spearmanr(hi["pred"], hi["fwd_rel"]).statistic)
        if len(lo) >= 5:
            conv.setdefault("lo", []).append(spearmanr(lo["pred"], lo["fwd_rel"]).statistic)

    df = pd.DataFrame(rows, index=["IC", "IC_IR"]).T
    print("\n=== Ranker IC (out-of-sample) ===")
    print(df.round(4).to_string())
    print(f"\nConviction (ensemble): IC high-|pred| tercile = {np.mean(conv['hi']):.4f}  |  "
          f"low-|pred| tercile = {np.mean(conv['lo']):.4f}")

    out = ROOT / "reports" / "_rank_accuracy.csv"
    df.to_csv(out)
    pd.Series({"IC_high_conviction": np.mean(conv["hi"]),
               "IC_low_conviction": np.mean(conv["lo"])}).to_csv(
        ROOT / "reports" / "_rank_conviction.csv")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
