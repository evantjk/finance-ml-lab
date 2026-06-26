"""Historical backtest of the per-sector PAIR strategy (walk-forward, OOS).

Each rebalance month, within every GICS sector the model longs its top-ranked
name and shorts its bottom-ranked name (using only information available then).
We compound each sector's pair sleeve and rank sectors by total % growth.

Honest scope: cached data starts ~2011-06; truly out-of-sample predictions
begin after the initial training window, so the OOS span is ~2015 -> 2026.
Uses today's S&P 500 members => survivorship bias (optimistic).

Usage:  python -m experiments.backtest_pairs
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, panel as panel_mod          # noqa: E402
from src.tasks import cross_sectional as xs        # noqa: E402

FIG = ROOT / "reports" / "figures"
PPY = xs.PERIODS_PER_YEAR
COST_BPS = 10.0
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": .25})


def sp500_tickers():
    meta = pd.read_csv(data.DATA_DIR / "sp500_meta.csv")
    have = {p.stem for p in data.DATA_DIR.glob("*.csv")}
    return [t for t in meta["ticker"] if t in have]


def sector_pair_returns(oos: pd.DataFrame):
    """Return (date x sector) net pair returns, predicted edges, and picks."""
    rows, edges, picks = {}, {}, {s: {"long": [], "short": []} for s in oos["sector"].unique()}
    prev = {}
    for date, g in oos.groupby(level="date"):
        rows[date], edges[date] = {}, {}
        for sec, gs in g.groupby("sector"):
            if sec == "Unknown" or len(gs) < 2:
                continue
            gs = gs.sort_values("pred")
            lo_t = gs.index.get_level_values("ticker")[0]
            hi_t = gs.index.get_level_values("ticker")[-1]
            gross = float(gs["fwd_ret"].iloc[-1] - gs["fwd_ret"].iloc[0])
            edges[date][sec] = float(gs["pred"].iloc[-1] - gs["pred"].iloc[0])  # predicted, no look-ahead
            changed = (prev.get(sec, (None, None)) != (hi_t, lo_t))
            cost = (2 * COST_BPS / 1e4) if changed else 0.0
            rows[date][sec] = gross - cost
            prev[sec] = (hi_t, lo_t)
            picks[sec]["long"].append(hi_t); picks[sec]["short"].append(lo_t)
    return (pd.DataFrame(rows).T.sort_index(),
            pd.DataFrame(edges).T.sort_index(), picks)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    total = float((1 + r).prod() - 1)
    yrs = len(r) / PPY
    cagr = (1 + total) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"Total%": total * 100, "CAGR%": cagr * 100, "Sharpe": sharpe,
            "MaxDD%": dd * 100, "Hit%": float((r > 0).mean()) * 100, "Months": len(r)}


def main():
    tickers = sp500_tickers()
    print(f"Universe: {len(tickers)} names. Building panel ...")
    panel = panel_mod.build_panel(tickers, years=15, min_names=100)

    print("Walk-forward predictions (expanding, retrain yearly) ...")
    pred = xs._walk_forward_predict(panel, init_years=4, step_months=12)
    oos = panel.loc[pred.index].copy()
    oos["pred"] = pred
    span = (pred.index.get_level_values("date").min(),
            pred.index.get_level_values("date").max())
    print(f"  OOS span: {span[0].date()} -> {span[1].date()}")

    pair_ret, edge_df, picks = sector_pair_returns(oos)
    # per-sector sleeve stats, ranked by total growth
    table = pd.DataFrame({sec: stats(pair_ret[sec]) for sec in pair_ret.columns}).T
    table = table.dropna().sort_values("Total%", ascending=False)

    # equal-weight blend across all sector pairs = the full strategy
    blend = pair_ret.mean(axis=1)
    blend_stats = stats(blend)

    # high-conviction sleeve: each month, hold the top-K sectors by PREDICTED edge
    K = 4
    hi = {}
    for date in pair_ret.index:
        e = edge_df.loc[date].dropna()
        top = e.nlargest(min(K, len(e))).index
        vals = pair_ret.loc[date, top].dropna()
        if len(vals):
            hi[date] = float(vals.mean())
    high = pd.Series(hi).sort_index()
    high_stats = stats(high)

    # write equity curves for the website chart
    import json
    eqdir = ROOT / "web" / "data"; eqdir.mkdir(parents=True, exist_ok=True)
    eq_high = (1 + high.fillna(0)).cumprod()
    eq_all = (1 + blend.reindex(high.index).fillna(0)).cumprod()
    (eqdir / "equity.json").write_text(json.dumps({
        "span": [str(high.index.min().date()), str(high.index.max().date())],
        "cost_bps": COST_BPS, "top_k": K,
        "dates": [d.strftime("%Y-%m") for d in eq_high.index],
        "high": [round(float(v), 4) for v in eq_high.values],
        "all": [round(float(v), 4) for v in eq_all.values],
        "stats_high": {k: round(v, 2) for k, v in high_stats.items()},
        "stats_all": {k: round(v, 2) for k, v in blend_stats.items()},
    }, indent=2))
    print(f"  high-conviction sleeve: Total {high_stats['Total%']:.0f}%  "
          f"CAGR {high_stats['CAGR%']:.1f}%  Sharpe {high_stats['Sharpe']:.2f}  "
          f"MaxDD {high_stats['MaxDD%']:.0f}%")

    # ---- console ----
    print(f"\n=== Per-sector PAIR growth (net {COST_BPS:.0f}bps), "
          f"{span[0].date()}–{span[1].date()} ===")
    print(table.round(1).to_string())
    print(f"\n=== Blended (equal-weight all sectors) ===")
    print({k: round(v, 1) for k, v in blend_stats.items()})

    # ---- figures ----
    plt.figure(figsize=(9, 4.6))
    for sec in table.index[:6]:
        (1 + pair_ret[sec].fillna(0)).cumprod().plot(lw=1.2, label=sec)
    (1 + blend.fillna(0)).cumprod().plot(lw=2.4, color="black", label="ALL (blend)")
    plt.title(f"Sector pair sleeves — cumulative growth of $1 ({span[0].date()}–{span[1].date()}, net)")
    plt.ylabel("growth of $1"); plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / "pairs_backtest.png"); plt.close()

    plt.figure(figsize=(7.5, 4))
    table["Total%"].sort_values().plot(kind="barh", color="#1a7f5a")
    plt.title("Total % growth by sector pair (OOS, net of costs)")
    plt.xlabel("total return %"); plt.tight_layout()
    plt.savefig(FIG / "pairs_growth_bars.png"); plt.close()

    # ---- report ----
    def md(df):
        h = "| sector | " + " | ".join(df.columns) + " |"
        s = "| --- | " + " | ".join(["---"] * len(df.columns)) + " |"
        b = "\n".join("| " + " | ".join([i] + [f"{v:.1f}" for v in r]) + " |"
                      for i, r in zip(df.index, df.values))
        return "\n".join([h, s, b])

    top = table.index[0]
    common_l = Counter(picks[top]["long"]).most_common(3)
    common_s = Counter(picks[top]["short"]).most_common(3)
    out = ROOT / "reports" / "pairs_backtest.md"
    out.write_text(
        f"# Historical pair-strategy growth ({span[0].date()}–{span[1].date()})\n\n"
        f"Walk-forward, expanding window, retrained yearly. Within each GICS sector the "
        f"model longs its top-ranked name / shorts its bottom-ranked name each month "
        f"(market- & sector-neutral). Net of {COST_BPS:.0f}bps per rotation.\n\n"
        "> ⚠️ Cached data starts ~2011-06 and OOS begins after the initial training "
        "window, so this is **not** a 2010 start. Uses today's S&P 500 members "
        "(**survivorship bias** — optimistic). Research demo, not advice.\n\n"
        "## Top sector pairs by total growth\n\n" + md(table.round(1)) +
        f"\n\n**Blended (equal-weight all {len(table)} sectors):** "
        f"Total {blend_stats['Total%']:.1f}% · CAGR {blend_stats['CAGR%']:.1f}% · "
        f"Sharpe {blend_stats['Sharpe']:.2f} · MaxDD {blend_stats['MaxDD%']:.1f}%.\n\n"
        f"Best sleeve = **{top}** ({table.loc[top,'Total%']:.0f}% total). "
        f"Most-picked longs: {', '.join(f'{t}×{c}' for t,c in common_l)}; "
        f"shorts: {', '.join(f'{t}×{c}' for t,c in common_s)}.\n\n"
        "### Figures\n- `figures/pairs_backtest.png` — cumulative growth (top sleeves + blend)\n"
        "- `figures/pairs_growth_bars.png` — total % growth by sector\n"
    )
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
