# Round 3 — breadth, portfolio construction & calibration

## 3a. S&P 500 breadth + portfolio construction

*Universe:* 500 names  ·  *OOS:* 2017-07-28–2026-05-07 (51582 stock-months)  ·  ML rank IC = 0.0327.

Each row adds one construction step on the **same** ML signal (Sharpe is leverage-invariant — gains come from *weighting*, not scaling):

| model | Ann | Sharpe | MaxDD | Vol | Hit |
| --- | --- | --- | --- | --- | --- |
| A. Naive decile L/S | 0.1295 | 0.8087 | -0.2086 | 0.1682 | 0.5660 |
| B. + Sector-neutral | 0.0773 | 0.7426 | -0.1502 | 0.1083 | 0.5472 |
| C. + Risk-sized (inv-vol) | 0.0556 | 0.6412 | -0.1345 | 0.0908 | 0.5283 |

*Breadth check:* the 43-name book scored Sharpe ≈ 0.39; with 500 names the naive book is now Sharpe 0.81 — the Fundamental Law (IR ≈ IC·√breadth) in action.

> ⚠️ **Survivorship bias:** uses *today's* S&P 500 members, so the backtest only sees survivors and is optimistic. Point-in-time membership (paid) would be needed to remove it.


## 3b. Direction — probability calibration

Pooled 120 names, 408,150 samples, test 2023-09-06–2026-05-22. Test accuracy 0.517 (base rate 0.524).

| model | Brier | ECE |
| --- | --- | --- |
| uncalibrated | 0.2501 | 0.0168 |
| isotonic-calibrated | 0.2497 | 0.0094 |

*Lower Brier & ECE = better-calibrated.* The point isn't accuracy (still ~coin-flip) — it's that after isotonic calibration the probabilities are **honest odds**: when the model says 56%, up really happens ~56% of the time. That is a shippable website feature; a point price prediction is not.

### Figures
- `figures/portfolio_construction.png` — equity by construction step
- `figures/calibration_reliability.png` — reliability before/after calibration