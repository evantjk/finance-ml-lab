# finance-ml-lab

An **honest** machine-learning lab for equities. Every model is benchmarked
against the naive baselines and classical factors it has to beat — and when ML
*doesn't* add exploitable structure, the report says so. No price predictions,
no curve-fit hero numbers.

> ⚠️ Research / educational demo. Uses today's S&P 500 members (survivorship
> bias) and cached daily data. **Not investment advice.**

## What's inside

| Task | Question | Honest verdict |
| --- | --- | --- |
| **Direction** | Can we predict tomorrow's up/down? | ML ≈ coin flip (AUC ~0.50). Shipped as *calibrated probabilities*, not predictions. |
| **Returns** | Can we forecast next-day return? | No. R²_oos ≤ 0; baselines win. |
| **Volatility** | Can we forecast realised vol? | **Yes** — HAR-leverage + GJR-GARCH combo beats ML and naive anchors (R²_oos up to ~0.20). |
| **Cross-sectional ranking** | Can ML rank winners vs losers? | Marginal edge (IC ~0.027) over a 12-1 momentum factor. |
| **Pairs / portfolio** | Walk-forward sector-neutral long/short | Blended sleeves: ~25% CAGR, Sharpe ~0.96 (survivorship-biased). |
| **Deep learning** | Do LSTM / Transformer beat HAR on vol? | LSTM edges HAR (R²_oos 0.15 vs 0.06); Transformer underperforms. |

Full write-ups with figures live in [`reports/`](reports/).

## Methodology

- **Expanding walk-forward**, retrained yearly, strictly out-of-sample.
- Every task ships **naive + classical baselines** (majority class, persistence,
  random-walk, EWMA, HAR, momentum/short-reversal factors).
- Costs modelled (10 bps per rotation); market-/sector-neutral construction.
- Probabilities **isotonic-calibrated**; conviction terciles reported.

## Stack

Python · scikit-learn · statsmodels · `arch` (GARCH) · PyTorch (LSTM/Transformer)
· pandas / numpy · **FastAPI** backend · vanilla-JS + anime.js front-end.

## Layout

```
src/            feature engineering, data layer, per-task models, predict API
experiments/    runnable scripts that produce the reports
reports/        markdown findings + figures
models/         pre-trained joblib artifacts (ranker, direction)
api/            FastAPI service (rank / volatility / direction odds)
web/            static front-end (reads precomputed snapshot, live API optional)
data/           market-data cache (gitignored; auto-downloaded on demand)
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -r requirements-dl.txt       # optional: deep-learning experiments

# one service: serves the front-end AND the API on a single port
uvicorn api.main:app --reload --port 8000
#   site  -> http://localhost:8000/
#   docs  -> http://localhost:8000/docs

# reproduce a report
python experiments/run_all.py
```

## Deploy

The FastAPI app serves both the static front-end (`web/`) and the JSON API from
one origin, so it runs as a **single service**. `render.yaml` is a ready
Render blueprint — point Render at the repo (New → Blueprint) and it builds from
`requirements.txt` and starts `uvicorn api.main:app`.

## API

| Endpoint | Returns |
| --- | --- |
| `GET /health` | liveness + loaded models |
| `GET /universe` | tradable tickers |
| `GET /rank?top_n=10` | cross-sectional long/short candidates |
| `GET /volatility/{ticker}` | next-21d realised-vol forecast (HAR-lev + GJR) |
| `GET /direction/{ticker}` | **calibrated** P(up) — odds, not a prediction |

## License

[MIT](LICENSE) © Evan Teong. Research and educational use only — not investment advice.
