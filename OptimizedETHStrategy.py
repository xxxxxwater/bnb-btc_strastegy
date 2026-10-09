"""Legacy loader alias. ETH trading is NOT implemented.

An existing ETH strategy fragment had no class/function, invalid parameters,
and was never executable. This compatibility class inherits the BTC/BNB
fail-closed allowlist: any ETH or other pair produces no new entries.
"""
from bnb_btc_strategy.freqtrade_strategy import BaseBTCCointegrationStrategy


class OptimizedETHStrategy(BaseBTCCointegrationStrategy):
    """Compatibility only; supports entry on BTC/BNB pairs, never ETH."""
