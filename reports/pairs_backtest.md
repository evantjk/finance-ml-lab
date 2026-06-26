# Historical pair-strategy growth (2016-07-28–2026-05-07)

Walk-forward, expanding window, retrained yearly. Within each GICS sector the model longs its top-ranked name / shorts its bottom-ranked name each month (market- & sector-neutral). Net of 10bps per rotation.

> ⚠️ Cached data starts ~2011-06 and OOS begins after the initial training window, so this is **not** a 2010 start. Uses today's S&P 500 members (**survivorship bias** — optimistic). Research demo, not advice.

## Top sector pairs by total growth

| sector | Total% | CAGR% | Sharpe | MaxDD% | Hit% | Months |
| --- | --- | --- | --- | --- | --- | --- |
| Health Care | 2983.8 | 41.7 | 0.8 | -76.8 | 54.2 | 118.0 |
| Energy | 2081.6 | 36.8 | 0.9 | -35.2 | 53.4 | 118.0 |
| Consumer Discretionary | 1312.0 | 30.9 | 0.7 | -95.4 | 59.3 | 118.0 |
| Financials | 367.9 | 17.0 | 0.5 | -75.8 | 54.2 | 118.0 |
| Communication Services | 159.9 | 10.2 | 0.4 | -62.6 | 54.2 | 118.0 |
| Industrials | 156.6 | 10.1 | 0.4 | -64.4 | 51.7 | 118.0 |
| Information Technology | 53.1 | 4.4 | 0.4 | -89.1 | 54.2 | 118.0 |
| Utilities | 12.1 | 1.2 | 0.3 | -88.1 | 56.8 | 118.0 |
| Consumer Staples | -56.4 | -8.1 | -0.1 | -68.8 | 46.6 | 118.0 |
| Materials | -61.1 | -9.2 | -0.0 | -84.0 | 49.2 | 118.0 |
| Real Estate | -71.0 | -11.8 | -0.1 | -78.5 | 44.9 | 118.0 |

**Blended (equal-weight all 11 sectors):** Total 781.6% · CAGR 24.8% · Sharpe 0.96 · MaxDD -35.7%.

Best sleeve = **Health Care** (2984% total). Most-picked longs: MRNA×49, DXCM×13, ALGN×11; shorts: JNJ×12, ABBV×8, PFE×7.

### Figures
- `figures/pairs_backtest.png` — cumulative growth (top sleeves + blend)
- `figures/pairs_growth_bars.png` — total % growth by sector
