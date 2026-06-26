"""Task 1 - Direction prediction (will the next move be up?).

We compare ML classifiers against two naive baselines and, crucially, run a
cost-aware backtest: an edge in accuracy is worthless if it dies after
transaction costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import utils


def _models():
    return {
        "Logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=50,
            n_jobs=-1, random_state=0
        ),
        "GradBoost": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=400,
            l2_regularization=1.0, random_state=0
        ),
    }


def run(ticker: str, X: pd.DataFrame, T: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    y = T["y_dir"].values
    ret_fwd = T["y_ret"].values  # realised next-day log return (for backtest)
    n = len(X)
    tr, te = utils.final_holdout(n, frac=0.30)

    rows = []
    equity = {}

    # --- baselines ---
    base_rate = y[tr].mean()  # P(up) in training
    # B1: always predict the majority class
    maj = 1.0 if base_rate >= 0.5 else 0.0
    pred_maj = np.full(len(te), maj)
    rows.append(_score("Baseline: always-majority", y[te], pred_maj, None))
    # B2: predict yesterday's direction (momentum-1)
    prev_dir = (T["_ret"].shift(1).values > 0).astype(float)
    rows.append(_score("Baseline: yesterday's-sign", y[te], prev_dir[te], None))

    # --- ML models ---
    for name, mdl in _models().items():
        mdl.fit(X.iloc[tr], y[tr])
        proba = mdl.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(float)
        rows.append(_score(name, y[te], pred, proba))
        # cost-aware long/flat backtest on the test window
        equity[name] = _backtest(pred, ret_fwd[te], T.index[te], cost_bps)

    # buy & hold benchmark over the same window
    equity["BuyHold"] = _backtest(np.ones(len(te)), ret_fwd[te], T.index[te], 0.0)

    res = pd.DataFrame(rows).set_index("model")
    bt = {k: utils.perf_stats(v) for k, v in equity.items()}
    return {
        "ticker": ticker, "metrics": res, "backtest": pd.DataFrame(bt).T,
        "equity": equity, "test_index": T.index[te], "base_rate_up": float(base_rate),
    }


def _score(name, y, pred, proba):
    out = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": _balanced(y, pred),
    }
    out["auc"] = roc_auc_score(y, proba) if proba is not None else np.nan
    return out


def _balanced(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum() > 0:
            accs.append((pred[m] == c).mean())
    return float(np.mean(accs)) if accs else np.nan


def _backtest(pred, ret_fwd, index, cost_bps) -> pd.Series:
    """Long when pred==1 else flat. Charge cost on position changes."""
    pos = pd.Series(pred, index=index)
    gross = pos * pd.Series(ret_fwd, index=index)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost = turnover * (cost_bps / 1e4)
    return (gross - cost).rename("ret")
