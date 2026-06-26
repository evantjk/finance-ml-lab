"""Build the panel, train the cross-sectional ranker, write the report.

Usage:  python -m experiments.run_cross_sectional
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

from src import panel as panel_mod        # noqa: E402
from src.tasks import cross_sectional      # noqa: E402
from src.universe import UNIVERSE          # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25})


def _md_table(df, fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "-")
    head = "| " + " | ".join(["model"] + list(df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    body = "\n".join("| " + " | ".join([str(i)] + list(r)) + " |"
                     for i, r in zip(df.index, df.values))
    return "\n".join([head, sep, body])


def main():
    print(f"Building panel for {len(UNIVERSE)} names ...")
    panel = panel_mod.build_panel(UNIVERSE, years=15)
    n_dates = panel.index.get_level_values("date").nunique()
    print(f"  panel: {len(panel)} rows, {n_dates} monthly dates, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers")

    print("Walk-forward training the cross-sectional ranker ...")
    res = cross_sectional.run(panel)
    print(res["metrics"].round(4).to_string())

    # cumulative L/S curves
    plt.figure(figsize=(8, 4.2))
    for name, ls in res["curves"].items():
        (1 + ls).cumprod().plot(label=name, lw=1.6 if "ML" in name else 1.0)
    plt.title("Cross-sectional long/short: cumulative return (net of 10bps, monthly)")
    plt.ylabel("growth of $1"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "cross_sectional_ls.png"); plt.close()

    # rolling IC of the ML ranker
    plt.figure(figsize=(8, 3.2))
    ic = res["ml_ic_series"]
    ic.plot(lw=0.6, alpha=0.4, label="monthly IC")
    ic.rolling(12).mean().plot(lw=1.8, label="12m avg IC")
    plt.axhline(0, color="grey", lw=0.6)
    plt.title("ML ranker: information coefficient over time")
    plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "cross_sectional_ic.png"); plt.close()

    # report
    d0, d1 = res["oos_dates"]
    out = ROOT / "reports" / "cross_sectional.md"
    L = [
        "# Cross-sectional stock ranking — does ML pick winners?\n",
        f"*Universe:* {len(UNIVERSE)} large caps  ·  *Rebalance:* monthly  ·  "
        f"*OOS:* {d0.date()} to {d1.date()} ({res['n_obs']} stock-months), "
        "expanding walk-forward, retrained yearly.\n",
        "Target = next-month **relative** return (market-neutral). "
        "IC = mean monthly rank-correlation of forecast vs realised; "
        "L/S = long top-quintile, short bottom-quintile, net of 10bps/turnover.\n",
        _md_table(res["metrics"]),
        "\n**How to read it:** a real cross-sectional signal shows IC ≈ 0.02–0.05 "
        "and a positive L/S Sharpe. If the ML ranker only matches (or loses to) the "
        "one-line momentum factor, the honest takeaway is *use the factor* — the ML "
        "isn't adding exploitable structure on this universe.\n",
        "### Figures",
        "- `figures/cross_sectional_ls.png` — cumulative long/short vs factor baselines",
        "- `figures/cross_sectional_ic.png` — information coefficient over time",
    ]
    out.write_text("\n".join(L))
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
