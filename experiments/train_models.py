"""Train and persist the production models to models/.

Usage:  python -m experiments.train_models
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import Ranker, DirectionModel  # noqa: E402


def main():
    t0 = time.time()
    print("Training cross-sectional ranker on the S&P 500 panel ...")
    r = Ranker.train()
    print(f"  saved {Ranker.PATH.name} (trained_on={r.trained_on})")

    print("Training + calibrating the direction model ...")
    DirectionModel.train()
    print(f"  saved {DirectionModel.PATH.name}")

    print(f"Done in {time.time()-t0:.0f}s. Models in {ROOT/'models'}")


if __name__ == "__main__":
    main()
