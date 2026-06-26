"""FastAPI backend for finance-ml-lab — the honest analytics API.

Serves both the JSON API *and* the static front-end in ``web/`` from one
origin, so the whole thing runs as a single service (e.g. on Render).

Run:  uvicorn api.main:app --reload --port 8000
Site: http://localhost:8000/       (the Fulcrum front-end)
Docs: http://localhost:8000/docs

Endpoints:
  GET /health                 - liveness + which models are loaded
  GET /universe               - tradable tickers (with data)
  GET /rank?top_n=10          - cross-sectional long/short candidates
  GET /volatility/{ticker}    - next-21d realised-vol forecast (HAR-lev + GJR)
  GET /direction/{ticker}     - calibrated P(up) odds (NOT a price prediction)
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT))

from src import predict  # noqa: E402

app = FastAPI(
    title="finance-ml-lab API",
    description="Honest equity analytics: cross-sectional ranking, volatility "
                "forecasting, and calibrated direction odds. No price predictions.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)

_state: dict = {"ranker": None, "direction": None}


@app.on_event("startup")
def _load_models():
    try:
        _state["ranker"] = predict.Ranker.load()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ranker not loaded: {e} — run `python -m experiments.train_models`")
    try:
        _state["direction"] = predict.DirectionModel.load()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] direction model not loaded: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ranker_loaded": _state["ranker"] is not None,
        "direction_loaded": _state["direction"] is not None,
        "universe_size": len(predict._sp500_tickers()),
    }


@app.get("/universe")
def universe():
    t = predict._sp500_tickers()
    return {"count": len(t), "tickers": t}


@app.get("/rank")
def rank(top_n: int = Query(10, ge=1, le=50),
         max_per_sector: int = Query(3, ge=1, le=20)):
    if _state["ranker"] is None:
        raise HTTPException(503, "Ranker not trained. Run experiments.train_models.")
    return _state["ranker"].rank(top_n=top_n, max_per_sector=max_per_sector)


@app.get("/pairs")
def pairs():
    if _state["ranker"] is None:
        raise HTTPException(503, "Ranker not trained. Run experiments.train_models.")
    return _state["ranker"].pairs()


@app.get("/volatility/{ticker}")
def volatility(ticker: str, horizon: int = Query(21, ge=5, le=126)):
    try:
        return predict.forecast_volatility(ticker.upper(), h=horizon)
    except FileNotFoundError:
        raise HTTPException(404, f"No data for {ticker.upper()}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.get("/direction/{ticker}")
def direction(ticker: str):
    if _state["direction"] is None:
        raise HTTPException(503, "Direction model not trained.")
    try:
        return _state["direction"].odds(ticker.upper())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"Could not score {ticker.upper()}: {e}")


# Serve the static front-end at "/". Mounted LAST so the explicit API routes
# above (and FastAPI's /docs, /openapi.json) take precedence; everything else
# — index.html, style.css, app.js, data/*.json — is served from web/.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
