"""Advanced volatility benchmark across horizons -> reports/volatility_advanced.md."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data                              # noqa: E402
from src.tasks import volatility_advanced as va   # noqa: E402

NAMES = ["SPY", "AAPL", "JPM", "XOM"]
HORIZONS = [5, 21, 63]


def _md(df, fmt="{:.4f}"):
    df = df.copy()
    df.columns = [f"{a}·{b}" for a, b in df.columns]
    for c in df.columns:
        df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "-")
    head = "| " + " | ".join(["model"] + list(df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    body = "\n".join("| " + " | ".join([str(i)] + list(r)) + " |"
                     for i, r in zip(df.index, df.values))
    return "\n".join([head, sep, body])


def main():
    dfs = {t: data.load(t, years=15) for t in NAMES}
    per_h = []
    for h in HORIZONS:
        stack = [va.run_one(t, dfs[t], h) for t in NAMES]
        # average metrics across names for this horizon
        avg = sum(s.values for s in stack) / len(stack)
        per_h.append(pd.DataFrame(avg, index=stack[0].index, columns=stack[0].columns))
        print(f"h={h} done")
    table = pd.concat(per_h, axis=1)
    # keep just R2_oos and QLIKE columns for readability
    keep = [c for c in table.columns if c[1] in ("R2_oos", "QLIKE")]
    table = table[keep]

    out = ROOT / "reports" / "volatility_advanced.md"
    L = [
        "# Volatility, deeper — HAR variants, GARCH family, combination\n",
        f"*Names:* {', '.join(NAMES)} (metrics averaged)  ·  "
        f"*Horizons:* {', '.join(str(h)+'d' for h in HORIZONS)}  ·  "
        "70/30 chronological hold-out.\n",
        "Higher **R2_oos** is better (vs train-mean); lower **QLIKE** is better.\n",
        _md(table),
        "\n**Reading it:** RW/EWMA are the naive anchors; HAR-lev adds the leverage "
        "(downside) effect; GJR/EGARCH are asymmetric GARCH; Combo averages HAR-lev+GJR. "
        "If Combo or HAR-lev beats plain HAR and GBM, the win comes from **asymmetry + "
        "averaging**, not from ML complexity.\n",
    ]
    out.write_text("\n".join(L))
    print(f"Report -> {out}")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
