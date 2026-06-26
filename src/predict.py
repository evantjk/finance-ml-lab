"""Inference layer — the production-facing models promoted from the research.

Three shippable, honestly-scoped outputs:
  * Ranker            - cross-sectional stock ranking (relative outperformance)
  * forecast_volatility - HAR-lev + GJR-GARCH combination (next-h realised vol)
  * DirectionModel    - isotonic-calibrated P(up) odds (NOT a price prediction)

The ranker and direction model are trained offline (`train_models.py`) and
persisted to ``models/``. Volatility is fit on demand per ticker (fast) and
cached in-process.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression

from . import data, features, panel as panel_mod
from .panel import SIGNAL_COLS
from .tasks import volatility_advanced as va

MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sp500_tickers() -> list[str]:
    meta = data.DATA_DIR / "sp500_meta.csv"
    have = {p.stem for p in data.DATA_DIR.glob("*.csv")}
    if meta.exists():
        m = pd.read_csv(meta)
        return [t for t in m["ticker"] if t in have]
    return sorted(have)


def _meta() -> pd.DataFrame:
    meta = data.DATA_DIR / "sp500_meta.csv"
    if meta.exists():
        m = pd.read_csv(meta).set_index("ticker")
        if "sub_industry" not in m:
            m["sub_industry"] = m["sector"]
        return m[["sector", "sub_industry"]]
    return pd.DataFrame(columns=["sector", "sub_industry"])


def build_current_cross_section(tickers: list[str]) -> pd.DataFrame:
    """Latest available signal row per ticker, z-scored across the universe."""
    rows = {}
    for t in tickers:
        try:
            s = panel_mod._signals(data.load(t, years=15)).drop(columns="fwd_ret")
            s = s.dropna(subset=SIGNAL_COLS)
            if len(s):
                rows[t] = s.iloc[-1]
        except Exception:  # noqa: BLE001
            continue
    df = pd.DataFrame(rows).T
    z = (df[SIGNAL_COLS] - df[SIGNAL_COLS].mean()) / df[SIGNAL_COLS].std(ddof=0)
    df[SIGNAL_COLS] = z.clip(-3, 3)
    meta = _meta().reindex(df.index)
    df["sector"] = meta["sector"].fillna("Unknown").values
    df["sub_industry"] = meta["sub_industry"].fillna("Unknown").values
    return df


# --------------------------------------------------------------------------- #
# 1) cross-sectional ranker
# --------------------------------------------------------------------------- #
class Ranker:
    PATH = MODELS / "ranker.joblib"

    def __init__(self, model=None, trained_on=None):
        self.model = model
        self.trained_on = trained_on

    @classmethod
    def train(cls, tickers=None, years=15):
        tickers = tickers or _sp500_tickers()
        panel = panel_mod.build_panel(tickers, years=years, min_names=100)
        m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300,
            l2_regularization=1.0, random_state=0)
        m.fit(panel[SIGNAL_COLS], panel["fwd_rel"])
        obj = cls(m, str(pd.Timestamp.utcnow().date()))
        joblib.dump({"model": m, "trained_on": obj.trained_on}, cls.PATH)
        return obj

    @classmethod
    def load(cls):
        d = joblib.load(cls.PATH)
        return cls(d["model"], d["trained_on"])

    def rank(self, tickers=None, top_n=10, max_per_sector=3) -> dict:
        tickers = tickers or _sp500_tickers()
        xs = build_current_cross_section(tickers)
        raw = pd.Series(self.model.predict(xs[SIGNAL_COLS]), index=xs.index)
        # neutralise at SUB-INDUSTRY level (finer than sector): removes the
        # memory/semis-style within-sector concentration, keeps stock-specific skill
        neutral = raw.groupby(xs["sub_industry"]).transform(lambda x: x - x.mean())
        out = pd.DataFrame({
            "signal": neutral.round(6),                       # est. relative monthly return
            "percentile": (neutral.rank(pct=True) * 100).round(1),
            "sector": xs["sector"],
            "sub_industry": xs["sub_industry"],
        }).sort_values("signal", ascending=False)

        longs = self._cap(out, top_n, max_per_sector, top=True)
        shorts = self._cap(out, top_n, max_per_sector, top=False)
        return {
            "trained_on": self.trained_on,
            "n_universe": len(out),
            "max_per_sector": max_per_sector,
            "note": "signal = SUB-INDUSTRY-neutral estimate of next-month relative "
                    "return; book is capped at max_per_sector names per GICS sector. "
                    "Equal-weight long the top / short the bottom = market-neutral.",
            "longs": longs.reset_index(names="ticker").to_dict("records"),
            "shorts": shorts.reset_index(names="ticker").to_dict("records"),
        }

    def pairs(self, tickers=None) -> dict:
        """Best market-neutral pair within EACH GICS sector.

        Long the name the model expects to out-perform its sector, short the one
        to under-perform. Same-sector legs => the pair is sector-neutral by
        construction. Covers every sector with >= 2 names.
        """
        tickers = tickers or _sp500_tickers()
        xs = build_current_cross_section(tickers)
        raw = pd.Series(self.model.predict(xs[SIGNAL_COLS]), index=xs.index)
        df = pd.DataFrame({
            "signal": raw,
            "pct": (raw.rank(pct=True) * 100).round(1),
            "sector": xs["sector"],
            "sub_industry": xs["sub_industry"],
        })

        def leg(row, ticker):
            return {
                "ticker": ticker,
                "sub_industry": row["sub_industry"],
                "signal": round(float(row["signal"]), 6),
                "percentile": float(row["pct"]),
            }

        pairs = []
        for sector, g in df.groupby("sector"):
            if sector == "Unknown" or len(g) < 2:
                continue
            g = g.sort_values("signal")
            lo, hi = g.iloc[0], g.iloc[-1]
            pairs.append({
                "sector": sector,
                "n_names": int(len(g)),
                "edge": round(float(hi["signal"] - lo["signal"]), 6),  # expected monthly spread
                "long": leg(hi, g.index[-1]),
                "short": leg(lo, g.index[0]),
            })
        pairs.sort(key=lambda p: p["edge"], reverse=True)
        # conviction: rank pairs by signal-spread magnitude. Our accuracy study
        # showed IC ~2x higher on the model's high-|prediction| calls, so a wider
        # spread => a more trustworthy pair. Tiers are relative across the sectors.
        if pairs:
            edges = np.array([p["edge"] for p in pairs])
            pcts = edges.argsort().argsort() / max(len(edges) - 1, 1) * 100
            for p, sc in zip(pairs, pcts):
                p["conviction"] = round(float(sc))
                p["conviction_tier"] = ("High" if sc >= 66 else
                                        "Medium" if sc >= 33 else "Low")
        return {"trained_on": self.trained_on, "n_sectors": len(pairs), "pairs": pairs}

    @staticmethod
    def _cap(out: pd.DataFrame, top_n: int, max_per_sector: int, top: bool) -> pd.DataFrame:
        """Greedily pick names from the best end, capping count per GICS sector."""
        ordered = out if top else out.iloc[::-1]
        picked, counts = [], {}
        for tkr, row in ordered.iterrows():
            sec = row["sector"]
            if counts.get(sec, 0) >= max_per_sector:
                continue
            picked.append(tkr); counts[sec] = counts.get(sec, 0) + 1
            if len(picked) >= top_n:
                break
        return out.loc[picked]


# --------------------------------------------------------------------------- #
# 2) volatility forecast (fit-on-demand, cached)
# --------------------------------------------------------------------------- #
_VOL_CACHE: dict[str, tuple[float, dict]] = {}
_VOL_TTL = 3600  # seconds


def forecast_volatility(ticker: str, h: int = 21) -> dict:
    key = f"{ticker}:{h}"
    now = time.time()
    if key in _VOL_CACHE and now - _VOL_CACHE[key][0] < _VOL_TTL:
        return _VOL_CACHE[key][1]

    df = data.load(ticker, years=15)
    ret = np.log(df["Close"]).diff()
    y = (ret.shift(-h).rolling(h).std() * np.sqrt(va.TRADING_DAYS))
    base = pd.DataFrame({"y": y, "ret": ret}).dropna()
    n = len(base)

    # HAR with leverage, fit on all rows with a known target
    X = va._har_design(base["ret"], leverage=True).reindex(base.index).ffill().bfill().fillna(0.0)
    from sklearn.linear_model import LinearRegression
    har = LinearRegression().fit(X.values, base["y"].values)
    # latest features = full-history design at the most recent date
    X_full = va._har_design(ret, leverage=True).ffill().bfill().fillna(0.0)
    har_pred = float(har.predict(X_full.iloc[[-1]].values)[0])

    # GJR-GARCH forecast from the latest conditional variance
    g = va._garch(ret, train_end=n, h=h, vol="Garch", o=1)
    garch_pred = float(g[-1]) if g is not None and len(g) else har_pred

    combo = float(np.mean([har_pred, garch_pred]))
    current = float((ret.rolling(h).std() * np.sqrt(va.TRADING_DAYS)).iloc[-1])
    out = {
        "ticker": ticker, "horizon_days": h,
        "forecast_vol_annual": round(combo, 4),
        "components": {"HAR_lev": round(har_pred, 4), "GJR_GARCH": round(garch_pred, 4)},
        "current_realised_vol": round(current, 4),
        "as_of": str(df.index[-1].date()),
    }
    _VOL_CACHE[key] = (now, out)
    return out


# --------------------------------------------------------------------------- #
# 3) calibrated direction odds
# --------------------------------------------------------------------------- #
def _feat_row(ticker: str) -> pd.Series:
    f = features.build_features(data.load(ticker, years=15))
    feat_cols = [c for c in f.columns if c != "_ret"]
    return f[feat_cols].dropna().iloc[-1]


class DirectionModel:
    PATH = MODELS / "direction.joblib"

    def __init__(self, clf=None, iso=None, cols=None):
        self.clf, self.iso, self.cols = clf, iso, cols

    @classmethod
    def train(cls, tickers=None, years=15, max_names=150):
        tickers = (tickers or _sp500_tickers())[:max_names]
        Xs, ys = [], []
        for t in tickers:
            X, T = features.make_dataset(data.load(t, years=years), horizon=1)
            Xs.append(X); ys.append(T["y_dir"].values)
        X = pd.concat(Xs).reset_index(drop=True)
        y = np.concatenate(ys)
        cut = int(len(X) * 0.8)
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=300,
            l2_regularization=1.0, random_state=0).fit(X.iloc[:cut], y[:cut])
        p_cal = clf.predict_proba(X.iloc[cut:])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y[cut:])
        joblib.dump({"clf": clf, "iso": iso, "cols": list(X.columns)}, cls.PATH)
        return cls(clf, iso, list(X.columns))

    @classmethod
    def load(cls):
        d = joblib.load(cls.PATH)
        return cls(d["clf"], d["iso"], d["cols"])

    def odds(self, ticker: str) -> dict:
        row = _feat_row(ticker)[self.cols]
        raw = float(self.clf.predict_proba(row.to_frame().T)[:, 1][0])
        cal = float(self.iso.predict([raw])[0])
        return {
            "ticker": ticker,
            "prob_up_next_day": round(cal, 4),
            "prob_up_raw": round(raw, 4),
            "note": "Calibrated odds, not a price prediction. ~coin-flip accuracy by design.",
        }
