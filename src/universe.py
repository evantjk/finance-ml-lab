"""A diversified ~40-name large-cap universe across sectors.

Chosen for long, liquid histories so the cross-sectional panel has enough
breadth each day without heavy survivorship issues. (A couple of names list
mid-sample, e.g. META 2012, ABBV 2013 — the panel handles that per-day.)
"""

UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "IBM",
    # Communication / internet
    "META", "DIS", "T", "VZ", "CMCSA",
    # Consumer
    "AMZN", "HD", "MCD", "NKE", "SBUX", "COST", "WMT", "PG", "KO", "PEP",
    # Health care
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "AXP", "C",
    # Industrials / Energy / Utilities
    "CAT", "BA", "HON", "XOM", "CVX", "NEE",
]
