"""Precompute the website's data snapshot: best pair per sector, each leg
enriched with the volatility forecast and calibrated direction odds.

Writes web/data/snapshot.json. Run after train_models.py.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, predict  # noqa: E402

OUT = ROOT / "web" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def _spark(ticker: str, days: int = 126, points: int = 44) -> list | None:
    """~6 months of closes, downsampled and normalised to 100 at the start."""
    try:
        c = data.load(ticker, years=15)["Close"].tail(days)
        step = max(1, len(c) // points)
        c = c.iloc[::step]
        return [round(float(x), 2) for x in (c / c.iloc[0] * 100)]
    except Exception:  # noqa: BLE001
        return None


def enrich(leg: dict, direction) -> dict:
    t = leg["ticker"]
    try:
        v = predict.forecast_volatility(t, h=21)
        leg["vol_annual"] = v["forecast_vol_annual"]
        leg["vol_current"] = v["current_realised_vol"]
    except Exception:  # noqa: BLE001
        leg["vol_annual"] = leg["vol_current"] = None
    try:
        leg["prob_up"] = direction.odds(t)["prob_up_next_day"] if direction else None
    except Exception:  # noqa: BLE001
        leg["prob_up"] = None
    leg["spark"] = _spark(t)
    return leg


def main():
    t0 = time.time()
    ranker = predict.Ranker.load()
    try:
        direction = predict.DirectionModel.load()
    except Exception:  # noqa: BLE001
        direction = None

    print("Computing sector pairs ...")
    res = ranker.pairs()
    pairs = res["pairs"]
    print(f"  {len(pairs)} sectors. Enriching legs with vol + odds ...")

    for i, p in enumerate(pairs, 1):
        enrich(p["long"], direction)
        enrich(p["short"], direction)
        print(f"  [{i}/{len(pairs)}] {p['sector']}: "
              f"L {p['long']['ticker']} / S {p['short']['ticker']}")

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": res["trained_on"],
        "n_sectors": res["n_sectors"],
        "universe_size": len(predict._sp500_tickers()),
        "pairs": pairs,
    }
    (OUT / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
    print(f"\nWrote {OUT/'snapshot.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
