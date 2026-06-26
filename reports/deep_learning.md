# Deep learning vs HAR — volatility (next-21d realised)

*Pooled sequences:* 6 names, window L=60d, 21761 samples, chronological 70/15/15 split. Device: mps.

| model | R2_oos | RMSE |
| --- | --- | --- |
| HAR (OLS baseline) | 0.0645 | 0.0916 |
| LSTM | 0.1538 | 0.0871 |
| Transformer | -0.1189 | 0.1002 |

**Verdict (nuanced, honest):**

- The **LSTM beat the *pooled* HAR baseline** (0.154 vs 0.064) — a single global HAR
  across heterogeneous names is weak, and the LSTM extracts extra non-linear /
  cross-name structure. So sequence models are *not* useless for vol.
- The **Transformer overfit and failed** (−0.12). With only ~15k training
  sequences it has far too many parameters; more capacity hurt here. A clean
  reminder that "bigger model" ≠ "better" on this data scale.
- **Important caveat — the comparison is not apples-to-apples.** The pooled HAR
  here scores only 0.06, but the *per-name* forecast-combination in
  `volatility_advanced.md` reaches **~0.20** on its own test window. So the best
  classical model is still competitive with — arguably ahead of — the LSTM, at a
  fraction of the complexity, with full interpretability and no GPU.

**Bottom line for the website:** the robust, cheap, explainable choice is the
**HAR-lev + GJR-GARCH combination**. An LSTM is worth keeping as an optional
ensemble member (it did add signal over pooled HAR), but a Transformer is not
justified at this data scale.
