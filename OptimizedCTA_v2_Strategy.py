"""Backwards-compatible Freqtrade loader file.

Install this repository as a package, then point Freqtrade at this file.
Original broken research snippets have been moved to a tested signal engine.
"""
from bnb_btc_strategy.freqtrade_strategy import BaseBTCCointegrationStrategy


class OptimizedCTAStrategy(BaseBTCCointegrationStrategy):
    """BTC/BNB 15m causal signal / Freqtrade V3 integration."""
