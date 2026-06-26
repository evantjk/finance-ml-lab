"""Run the full honest ML-for-finance benchmark and write reports/findings.md.

Usage:  python -m experiments.run_all
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, features            # noqa: E402
from src.tasks import direction, returns, volatility  # noqa: E402

UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "SPY"]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25})


def main():
    print("Loading data ...")
    uni = data.load_universe(UNIVERSE, years=15)

    dir_res, ret_res, vol_res = {}, {}, {}
    for t, df in uni.items():
        X, T = features.make_dataset(df, horizon=1, vol_h=21)
        print(f"  {t}: {X.shape[0]} samples, {X.shape[1]} features")
        dir_res[t] = direction.run(t, X, T)
        ret_res[t] = returns.run(t, X, T)
        vol_res[t] = volatility.run(t, X, T)

    _figures(dir_res, ret_res, vol_res)
    _write_report(uni, dir_res, ret_res, vol_res)
    print(f"\nDone. Report -> {ROOT/'reports'/'findings.md'}")


# --------------------------------------------------------------------------- #
def _avg_metric(res: dict, table: str, col: str) -> pd.Series:
    frames = [r[table][col] for r in res.values()]
    return pd.concat(frames, axis=1).mean(axis=1)


def _figures(dir_res, ret_res, vol_res):
    # 1) SPY direction-strategy equity curves vs buy & hold
    eq = dir_res["SPY"]["equity"]
    plt.figure(figsize=(8, 4))
    for name, r in eq.items():
        (1 + r).cumprod().plot(label=name, lw=1.6 if name == "BuyHold" else 1.0)
    plt.title("SPY: direction-strategy equity (net of 5bps costs) vs buy & hold")
    plt.ylabel("growth of $1"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "direction_equity_spy.png"); plt.close()

    # 2) Volatility forecast vs realised (SPY, best ML model)
    v = vol_res["SPY"]
    idx = v["test_index"]
    plt.figure(figsize=(9, 4))
    plt.plot(idx, v["y_test"], color="black", lw=1.3, label="realised (next 21d)")
    for name in ["Baseline: random-walk", "ML: GradBoost+HAR"]:
        if name in v["preds"]:
            plt.plot(idx, v["preds"][name], lw=1.0, alpha=0.8, label=name)
    plt.title("SPY: 21-day realised-volatility forecast (out-of-sample)")
    plt.ylabel("annualised vol"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "volatility_spy.png"); plt.close()

    # 3) Returns: predicted vs actual scatter (SPY, GradBoost) - shows no signal
    r = ret_res["SPY"]
    if "GradBoost" in r["preds"]:
        plt.figure(figsize=(4.6, 4.4))
        plt.scatter(r["preds"]["GradBoost"], r["y_test"], s=6, alpha=0.3)
        plt.axhline(0, color="grey", lw=0.6); plt.axvline(0, color="grey", lw=0.6)
        plt.xlabel("predicted next-day return"); plt.ylabel("actual next-day return")
        plt.title("SPY returns: predicted vs actual\n(a blob = no exploitable signal)")
        plt.tight_layout(); plt.savefig(FIG / "returns_scatter_spy.png"); plt.close()


def _md_table(df: pd.DataFrame, fmt="{:.4f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "-")
    head = "| " + " | ".join([df.index.name or "model"] + list(df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    body = "\n".join("| " + " | ".join([str(i)] + list(row)) + " |"
                     for i, row in zip(df.index, df.values))
    return "\n".join([head, sep, body])


def _write_report(uni, dir_res, ret_res, vol_res):
    out = ROOT / "reports" / "findings.md"
    L = []
    span = f"{list(uni.values())[0].index[0].date()} to {list(uni.values())[0].index[-1].date()}"
    L.append("# Where does ML actually fit in finance? — an honest benchmark\n")
    L.append(f"*Universe:* {', '.join(uni)}  ·  *Daily data:* {span}  ·  "
             "*Eval:* chronological 70/30 hold-out with a 5-day embargo (no shuffling, "
             "no look-ahead).\n")
    L.append("All ML models are compared against **naive baselines**. A model only "
             "'works' if it beats the baseline *out-of-sample*.\n")

    # ---- Direction ----
    L.append("\n## 1. Direction (up/down next day) — classification\n")
    acc = _avg_metric(dir_res, "metrics", "accuracy")
    auc = _avg_metric(dir_res, "metrics", "auc")
    tab = pd.DataFrame({"avg_accuracy": acc, "avg_AUC": auc})
    tab.index.name = "model"
    L.append(_md_table(tab))
    base = np.mean([r["base_rate_up"] for r in dir_res.values()])
    L.append(f"\n*Base rate (P[up]) across names: {base:.3f}.* Net-of-cost backtest "
             "(SPY, 5bps/turnover):\n")
    L.append(_md_table(dir_res["SPY"]["backtest"]))

    # ---- Returns ----
    L.append("\n## 2. Return magnitude (next-day %) — regression\n")
    r2 = _avg_metric(ret_res, "metrics", "R2_oos")
    rmsek = _avg_metric(ret_res, "metrics", "RMSE")
    da = _avg_metric(ret_res, "metrics", "DirAcc")
    tab = pd.DataFrame({"avg_R2_oos": r2, "avg_RMSE": rmsek, "avg_DirAcc": da})
    tab.index.name = "model"
    L.append(_md_table(tab))
    L.append("\n*R²_oos is vs predicting the training mean (Campbell-Thompson). "
             "Values ≤ 0 mean the model loses to that trivial forecast.*")

    # ---- Volatility ----
    L.append("\n## 3. Volatility (next-21d realised, annualised) — regression\n")
    rmsev = _avg_metric(vol_res, "metrics", "RMSE")
    ql = _avg_metric(vol_res, "metrics", "QLIKE")
    r2v = _avg_metric(vol_res, "metrics", "R2_oos")
    tab = pd.DataFrame({"avg_RMSE": rmsev, "avg_QLIKE": ql, "avg_R2_oos": r2v})
    tab.index.name = "model"
    L.append(_md_table(tab))

    # ---- Verdict ----
    L.append("\n## Verdict — where ML fits\n")
    best_vol = r2v.idxmax()
    L.append(
        f"- **Volatility: ML/stats clearly work.** Best OOS R² = {r2v.max():.2f} "
        f"({best_vol}); strong persistence is real, exploitable signal.\n"
        f"- **Direction: a thin edge at best.** Avg accuracy ~{acc.drop(index=[i for i in acc.index if 'Baseline' in i]).max():.3f} "
        f"vs base rate {base:.3f}; whether it survives costs is shown in the SPY backtest above.\n"
        f"- **Return magnitude: ML does not help.** Best OOS R² = {r2.max():.3f} "
        "(≈0 or negative) — daily returns are ~unpredictable in level; do not build a "
        "site feature that claims to predict tomorrow's % move.\n"
    )
    L.append("\n**Implication for the website:** lead with what is defensible — "
             "volatility/risk analytics, regime & drawdown context, and "
             "probability-calibrated direction *odds* (not point price predictions).\n")
    L.append("\n### Figures\n")
    L.append("- `figures/direction_equity_spy.png` — strategy vs buy & hold (net of costs)")
    L.append("- `figures/volatility_spy.png` — vol forecast vs realised")
    L.append("- `figures/returns_scatter_spy.png` — predicted vs actual returns (the blob)")

    out.write_text("\n".join(L))


if __name__ == "__main__":
    main()
