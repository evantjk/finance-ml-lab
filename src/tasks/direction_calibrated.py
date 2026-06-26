"""Probability-calibrated direction model.

The raw direction classifier is barely above a coin flip on *accuracy*, but a
website doesn't need accuracy — it needs **trustworthy odds**. A model can be
useless at classification yet well-calibrated: when it says "58% chance up", up
should happen ~58% of the time. We measure and fix that with isotonic calibration.

Split (chronological): train -> fit model | calib -> fit isotonic map | test -> score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .. import features


def _pool(tickers, dataf):
    Xs, ys, ds = [], [], []
    for t in tickers:
        X, T = features.make_dataset(dataf(t), horizon=1)
        Xs.append(X.assign(_d=X.index))
        ys.append(T["y_dir"].values)
        ds.append(X.index.values)
    X = pd.concat(Xs).reset_index(drop=True)
    dates = pd.to_datetime(np.concatenate(ds))
    y = np.concatenate(ys)
    order = np.argsort(dates.values)
    feat_cols = [c for c in X.columns if c != "_d"]
    return X[feat_cols].iloc[order].reset_index(drop=True), y[order], dates[order]


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def run(tickers, dataf) -> dict:
    X, y, dates = _pool(tickers, dataf)
    n = len(X)
    i_tr, i_ca = int(n * 0.6), int(n * 0.8)

    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0, random_state=0)
    clf.fit(X.iloc[:i_tr], y[:i_tr])

    p_ca = clf.predict_proba(X.iloc[i_tr:i_ca])[:, 1]
    p_te = clf.predict_proba(X.iloc[i_ca:])[:, 1]
    y_ca, y_te = y[i_tr:i_ca], y[i_ca:]

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, y_ca)
    p_te_cal = iso.predict(p_te)

    acc = float(((p_te >= 0.5).astype(int) == y_te).mean())
    metrics = pd.DataFrame({
        "Brier": [_brier(p_te, y_te), _brier(p_te_cal, y_te)],
        "ECE": [_ece(p_te, y_te), _ece(p_te_cal, y_te)],
    }, index=["uncalibrated", "isotonic-calibrated"])

    frac_unc, mean_unc = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_te, p_te_cal, n_bins=10, strategy="quantile")

    return {
        "metrics": metrics, "accuracy": acc, "n": n,
        "base_rate": float(y_te.mean()),
        "rel_uncal": (mean_unc, frac_unc),
        "rel_cal": (mean_cal, frac_cal),
        "test_span": (dates[i_ca].date(), dates[-1].date()),
    }
