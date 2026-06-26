# Where does ML actually fit in finance? — an honest benchmark

*Universe:* AAPL, MSFT, JPM, XOM, SPY  ·  *Daily data:* 2011-06-24 to 2026-06-24  ·  *Eval:* chronological 70/30 hold-out with a 5-day embargo (no shuffling, no look-ahead).

All ML models are compared against **naive baselines**. A model only 'works' if it beats the baseline *out-of-sample*.


## 1. Direction (up/down next day) — classification

| model | avg_accuracy | avg_AUC |
| --- | --- | --- |
| Baseline: always-majority | 0.5206 | - |
| Baseline: yesterday's-sign | 0.4987 | - |
| Logistic | 0.5094 | 0.5007 |
| RandomForest | 0.5189 | 0.4920 |
| GradBoost | 0.5094 | 0.4916 |

*Base rate (P[up]) across names: 0.525.* Net-of-cost backtest (SPY, 5bps/turnover):

| model | CAGR | Sharpe | MaxDD | Vol_ann | HitRate |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.1645 | 0.9895 | -0.1925 | 0.1682 | 0.4972 |
| RandomForest | 0.1337 | 0.8054 | -0.2345 | 0.1748 | 0.5253 |
| GradBoost | 0.0830 | 0.5834 | -0.1894 | 0.1579 | 0.3996 |
| BuyHold | 0.1441 | 0.8557 | -0.2345 | 0.1752 | 0.5441 |

## 2. Return magnitude (next-day %) — regression

| model | avg_R2_oos | avg_RMSE | avg_DirAcc |
| --- | --- | --- | --- |
| Baseline: predict-zero | -0.0013 | 0.0156 | 0.0017 |
| Baseline: train-mean | 0.0000 | 0.0156 | 0.5356 |
| Baseline: persistence | -1.0377 | 0.0223 | 0.4972 |
| Ridge | -0.0285 | 0.0158 | 0.5032 |
| RandomForest | -0.0078 | 0.0157 | 0.5229 |
| GradBoost | -0.0536 | 0.0160 | 0.5205 |

*R²_oos is vs predicting the training mean (Campbell-Thompson). Values ≤ 0 mean the model loses to that trivial forecast.*

## 3. Volatility (next-21d realised, annualised) — regression

| model | avg_RMSE | avg_QLIKE | avg_R2_oos |
| --- | --- | --- | --- |
| Baseline: random-walk | 0.0962 | 0.3135 | -0.0911 |
| Baseline: EWMA(0.94) | 0.0899 | 0.2616 | 0.0450 |
| HAR-RV (OLS) | 0.0827 | 0.2414 | 0.1954 |
| GARCH(1,1) | 0.0841 | 0.2229 | 0.1697 |
| ML: GradBoost+HAR | 0.0868 | 0.2699 | 0.1008 |

## Verdict — where ML fits

- **Volatility: ML/stats clearly work.** Best OOS R² = 0.20 (HAR-RV (OLS)); strong persistence is real, exploitable signal.
- **Direction: a thin edge at best.** Avg accuracy ~0.519 vs base rate 0.525; whether it survives costs is shown in the SPY backtest above.
- **Return magnitude: ML does not help.** Best OOS R² = 0.000 (≈0 or negative) — daily returns are ~unpredictable in level; do not build a site feature that claims to predict tomorrow's % move.


**Implication for the website:** lead with what is defensible — volatility/risk analytics, regime & drawdown context, and probability-calibrated direction *odds* (not point price predictions).


### Figures

- `figures/direction_equity_spy.png` — strategy vs buy & hold (net of costs)
- `figures/volatility_spy.png` — vol forecast vs realised
- `figures/returns_scatter_spy.png` — predicted vs actual returns (the blob)