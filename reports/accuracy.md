# Can we improve prediction accuracy? — four levers, honestly tested

We measured each lever out-of-sample on the S&P 500 panel / multi-name vol set.
Two of four levers were **wash-or-negative** — which is itself the finding: the
existing price-based models already extract most of the freely-available signal.

## Lever #1 — VIX + cross-asset features → volatility accuracy
Added VIX (level/change/richness), 10y-yield change, dollar change, HY–IG credit
to the vol model. Avg OOS R² over 6 names:

| model | baseline | + market | lift |
|---|---|---|---|
| Ridge (fair test) | 0.122 | 0.110 | **−0.01 (wash)** |
| GBM | −0.01 | −0.19 | −0.18 (overfits) |

**Verdict: no reliable gain.** A stock's own trailing realized vol already encodes
the systematic/VIX component, so VIX adds little marginal information and behaves
like noise in a flexible model. (Helped AAPL/XOM/MSFT slightly, hurt JPM/JNJ.)

## Lever #2 — residual (beta-adjusted) momentum → ranker IC
| signal set | IC | IC-IR |
|---|---|---|
| base (10 signals) | 0.0256 | 0.60 |
| + residual momentum | 0.0250 | 0.58 |

**Verdict: no lift.** The existing momentum/trend features already span what
beta-adjusted momentum captures; it's redundant, not additive.

## Lever #3 — ensembling + conviction filtering → ranker IC
| approach | IC |
|---|---|
| single GBM | 0.0256 |
| ensemble (GBM + Ridge + RF) | **0.0290** (+13%) |
| **high-conviction tercile** (top |pred|) | **0.0441** |
| low-conviction tercile | 0.0019 |

**Verdict: the real win.** Ensembling helps modestly. But **conviction filtering**
is the standout — the model's accuracy nearly **doubles** on its most confident
calls (IC 0.044 vs 0.029 overall) and collapses to noise on its least confident.
The model *knows when it knows*. Trading only high-conviction names is the single
most effective accuracy lever we found — and it's free.

## Lever #4 — intraday & options-implied vol (the biggest *potential* jumps)
Empirically probed the free data tier:
- **Intraday:** 5-min bars limited to **60 days**, hourly to **2 years** — far too
  short to build 15y realized-variance history. Realized-vol-from-intraday (the
  R² 0.20→0.45+ jump) is **not backtestable** without a paid feed.
- **Options:** chains are **current-snapshot only**, with implied vol but **no IV
  history** — so options-implied features can't be backtested either.

**Verdict: genuinely high-value, but gated by paid data.** Scope only, not built.

---

## Bottom line
- The ceiling on *free daily data* is close to what we already have. More inputs
  (VIX, residual momentum) **didn't help** — the price signal is largely tapped.
- **What does work: use the predictions more intelligently** — ensemble, and above
  all **filter to high-conviction calls** (≈2× the IC). Accuracy isn't only about
  the model; it's about *which predictions you act on*.
- The big remaining accuracy jumps (intraday vol, options-implied vol, fundamentals
  & earnings-revision signals) all require **data we don't have for free** — that,
  not model cleverness, is the real frontier.
