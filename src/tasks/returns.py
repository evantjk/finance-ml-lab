"""Task 2 - Return magnitude forecasting (regression).

This is the hardest task and the honest expectation is that ML will *not*
beat the trivial 'predict the historical mean' baseline out-of-sample
(near-zero or negative OOS R^2). We measure exactly that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import utils


def _models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=50,
            n_jobs=-1, random_state=0
        ),
        "GradBoost": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.02, max_iter=400,
            l2_regularization=1.0, random_state=0
        ),
    }


def run(ticker: str, X: pd.DataFrame, T: pd.DataFrame) -> dict:
    y = T["y_ret"].values
    n = len(X)
    tr, te = utils.final_holdout(n, frac=0.30)
    ytr_mean = y[tr].mean()

    rows = []
    preds = {}

    # --- baselines ---
    rows.append(_score("Baseline: predict-zero", y[te], np.zeros(len(te)), ytr_mean))
    rows.append(_score("Baseline: train-mean", y[te], np.full(len(te), ytr_mean), ytr_mean))
    persist = T["_ret"].shift(1).values  # yesterday's return (AR-1 persistence)
    rows.append(_score("Baseline: persistence", y[te], persist[te], ytr_mean))

    # --- ML models ---
    for name, mdl in _models().items():
        mdl.fit(X.iloc[tr], y[tr])
        p = mdl.predict(X.iloc[te])
        preds[name] = p
        rows.append(_score(name, y[te], p, ytr_mean))

    res = pd.DataFrame(rows).set_index("model")
    return {
        "ticker": ticker, "metrics": res, "preds": preds,
        "y_test": y[te], "test_index": T.index[te],
    }


def _score(name, y, p, ytr_mean):
    return {
        "model": name,
        "RMSE": utils.rmse(y, p),
        "MAE": utils.mae(y, p),
        "R2_oos": utils.r2_oos(y, p, ytr_mean),
        "DirAcc": utils.directional_accuracy(y, p),
    }
