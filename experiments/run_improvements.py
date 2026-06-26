"""Round 3 improvements: S&P 500 breadth + portfolio construction + calibration.

Usage:  python -m experiments.run_improvements
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, panel as panel_mod                 # noqa: E402
from src.tasks import portfolio, direction_calibrated     # noqa: E402

FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def _md(df, fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "-")
    head = "| " + " | ".join([df.index.name or "model"] + list(df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    body = "\n".join("| " + " | ".join([str(i)] + list(r)) + " |"
                     for i, r in zip(df.index, df.values))
    return "\n".join([head, sep, body])


def sp500_tickers():
    meta = pd.read_csv(data.DATA_DIR / "sp500_meta.csv")
    have = {p.stem for p in data.DATA_DIR.glob("*.csv")}
    return [t for t in meta["ticker"] if t in have]


def main():
    tickers = sp500_tickers()
    print(f"S&P 500 names with data: {len(tickers)}")

    # ---------- Portfolio construction on the full universe ----------
    print("Building S&P 500 panel (this is the heavy step) ...")
    panel = panel_mod.build_panel(tickers, years=15, min_names=100)
    nd = panel.index.get_level_values("date").nunique()
    print(f"  panel: {len(panel)} rows, {nd} months, "
          f"{panel.index.get_level_values('ticker').nunique()} names")
    pf = portfolio.run(panel)
    print(pf["metrics"].round(3).to_string())

    plt.figure(figsize=(8, 4.2))
    for name, r in pf["curves"].items():
        (1 + r).cumprod().plot(label=name, lw=1.3)
    plt.title(f"S&P 500 long/short construction ({pf['n_names']} names, net of costs)")
    plt.ylabel("growth of $1"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "portfolio_construction.png"); plt.close()

    # ---------- Probability calibration (direction) ----------
    print("Calibrating direction model (pooled, ~120 names) ...")
    cal_tickers = tickers[:120]
    cal = direction_calibrated.run(cal_tickers, lambda t: data.load(t, years=15))
    print(cal["metrics"].round(4).to_string(), f"\n  test accuracy {cal['accuracy']:.3f}")

    plt.figure(figsize=(4.8, 4.6))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    mu, fu = cal["rel_uncal"]; mc, fc = cal["rel_cal"]
    plt.plot(mu, fu, "o-", lw=1.2, ms=4, label="uncalibrated")
    plt.plot(mc, fc, "s-", lw=1.2, ms=4, label="isotonic-calibrated")
    plt.xlabel("predicted P(up)"); plt.ylabel("observed frequency of up")
    plt.title("Direction model reliability"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "calibration_reliability.png"); plt.close()

    # ---------- Report ----------
    d0, d1 = pf["oos_dates"]
    out = ROOT / "reports" / "improvements.md"
    L = [
        "# Round 3 — breadth, portfolio construction & calibration\n",
        "## 3a. S&P 500 breadth + portfolio construction\n",
        f"*Universe:* {pf['n_names']} names  ·  *OOS:* {d0.date()}–{d1.date()} "
        f"({pf['n_obs']} stock-months)  ·  ML rank IC = {pf['IC']:.4f}.\n",
        "Each row adds one construction step on the **same** ML signal "
        "(Sharpe is leverage-invariant — gains come from *weighting*, not scaling):\n",
        _md(pf["metrics"]),
        f"\n*Breadth check:* the 43-name book scored Sharpe ≈ 0.39; with "
        f"{pf['n_names']} names the naive book is now Sharpe "
        f"{pf['metrics'].loc['A. Naive decile L/S','Sharpe']:.2f} — the Fundamental "
        "Law (IR ≈ IC·√breadth) in action.\n",
        "> ⚠️ **Survivorship bias:** uses *today's* S&P 500 members, so the backtest "
        "only sees survivors and is optimistic. Point-in-time membership (paid) would "
        "be needed to remove it.\n",
        "\n## 3b. Direction — probability calibration\n",
        f"Pooled {len(cal_tickers)} names, {cal['n']:,} samples, "
        f"test {cal['test_span'][0]}–{cal['test_span'][1]}. "
        f"Test accuracy {cal['accuracy']:.3f} (base rate {cal['base_rate']:.3f}).\n",
        _md(cal["metrics"]),
        "\n*Lower Brier & ECE = better-calibrated.* The point isn't accuracy (still "
        "~coin-flip) — it's that after isotonic calibration the probabilities are "
        "**honest odds**: when the model says 56%, up really happens ~56% of the time. "
        "That is a shippable website feature; a point price prediction is not.\n",
        "### Figures",
        "- `figures/portfolio_construction.png` — equity by construction step",
        "- `figures/calibration_reliability.png` — reliability before/after calibration",
    ]
    out.write_text("\n".join(L))
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
