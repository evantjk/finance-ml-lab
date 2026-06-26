# Volatility, deeper — HAR variants, GARCH family, combination

*Names:* SPY, AAPL, JPM, XOM (metrics averaged)  ·  *Horizons:* 5d, 21d, 63d  ·  70/30 chronological hold-out.

Higher **R2_oos** is better (vs train-mean); lower **QLIKE** is better.

| model | h=5·QLIKE | h=5·R2_oos | h=21·QLIKE | h=21·R2_oos | h=63·QLIKE | h=63·R2_oos |
| --- | --- | --- | --- | --- | --- | --- |
| RW | 1.5434 | -0.2998 | 0.3060 | -0.0695 | 0.2611 | -0.2876 |
| EWMA | 0.5070 | 0.0799 | 0.2567 | 0.0544 | 0.2529 | -0.3061 |
| HAR | 0.5519 | 0.1389 | 0.2448 | 0.1763 | 0.1897 | 0.1481 |
| HAR-lev | 0.5484 | 0.1532 | 0.2445 | 0.1780 | 0.1902 | 0.1456 |
| log-HAR | 0.6651 | 0.1262 | 0.2634 | 0.2102 | 0.2106 | 0.1458 |
| GARCH | 0.4897 | 0.1088 | 0.2212 | 0.1600 | 0.1690 | 0.1508 |
| GJR-GARCH | 0.4832 | 0.1362 | 0.2175 | 0.1638 | 0.1678 | 0.1336 |
| Combo(HAR-lev+GJR) | 0.4956 | 0.1739 | 0.2216 | 0.2012 | 0.1717 | 0.1710 |
| ML GBM | 0.5596 | 0.0756 | 0.2582 | 0.0675 | 0.2822 | -0.2641 |

**Reading it:** RW/EWMA are the naive anchors; HAR-lev adds the leverage (downside) effect; GJR/EGARCH are asymmetric GARCH; Combo averages HAR-lev+GJR. If Combo or HAR-lev beats plain HAR and GBM, the win comes from **asymmetry + averaging**, not from ML complexity.
