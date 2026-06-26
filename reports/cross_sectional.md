# Cross-sectional stock ranking — does ML pick winners?

*Universe:* 43 large caps  ·  *Rebalance:* monthly  ·  *OOS:* 2017-07-28 to 2026-05-07 (4558 stock-months), expanding walk-forward, retrained yearly.

Target = next-month **relative** return (market-neutral). IC = mean monthly rank-correlation of forecast vs realised; L/S = long top-quintile, short bottom-quintile, net of 10bps/turnover.

| model | IC | IC_IR | LS_ann | LS_Sharpe | LS_hit |
| --- | --- | --- | --- | --- | --- |
| ML ranker (GBM) | 0.0272 | 0.4044 | 0.0629 | 0.3904 | 0.5283 |
| Factor: 12-1 momentum | 0.0134 | 0.1609 | -0.0012 | 0.1074 | 0.5566 |
| Factor: short-rev | -0.0016 | -0.0250 | -0.0290 | -0.0486 | 0.5189 |

**How to read it:** a real cross-sectional signal shows IC ≈ 0.02–0.05 and a positive L/S Sharpe. If the ML ranker only matches (or loses to) the one-line momentum factor, the honest takeaway is *use the factor* — the ML isn't adding exploitable structure on this universe.

### Figures
- `figures/cross_sectional_ls.png` — cumulative long/short vs factor baselines
- `figures/cross_sectional_ic.png` — information coefficient over time