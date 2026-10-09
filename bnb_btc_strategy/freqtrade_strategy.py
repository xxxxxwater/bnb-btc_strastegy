"""Freqtrade V3 interface; imports are isolated from the standalone library.

Trading disabled by default in config.example.json. Only BTC/BNB can enter.
Other pairs and unavailable / stale benchmark data fail closed for entries.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from freqtrade.strategy import IStrategy, merge_informative_pair

from .core import SignalConfig, calculate_features

LOG = logging.getLogger(__name__)

PAIRS = ("BTC/USDT:USDT", "BNB/USDT:USDT")


class BaseBTCCointegrationStrategy(IStrategy):
    """Research implementation, not validated profitable execution software."""

    INTERFACE_VERSION = 3
    timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 0.04, "120": 0.02, "360": 0.0}
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    position_adjustment_enable = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def signal_config(self) -> SignalConfig:
        return SignalConfig(require_benchmark=True)

    def informative_pairs(self) -> list[tuple[str, str]]:
        return [(pair, self.timeframe) for pair in PAIRS]

    def leverage(
        self, pair: str, current_time: Any, current_rate: float,
        proposed_leverage: float, max_leverage: float,
        entry_tag: str | None, side: str, **kwargs: Any
    ) -> float:
        return 1.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair")
        counterpart = (
            PAIRS[1] if pair == PAIRS[0] else
            PAIRS[0] if pair == PAIRS[1] else None
        )
        benchmark = None
        if counterpart and self.dp:
            try:
                reference = self.dp.get_pair_dataframe(
                    pair=counterpart, timeframe=self.timeframe
                )
                if (reference is not None and not reference.empty
                        and {"date", "close"}.issubset(reference.columns)):
                    informative = reference[["date", "close"]].copy()
                    informative = informative.rename(columns={"close": "benchmark_close"})
                    informative["benchmark_timestamp"] = informative["date"]
                    # Freqtrade shifts same-timeframe informative candles to
                    # prevent trading on a reference candle not yet closed.
                    dataframe = merge_informative_pair(
                        dataframe, informative, self.timeframe, self.timeframe,
                        ffill=True
                    )
                    field = "benchmark_close_" + self.timeframe
                    if field in dataframe:
                        timestamp_field = "benchmark_timestamp_" + self.timeframe
                        if timestamp_field in dataframe:
                            traded_at = pd.to_datetime(dataframe["date"], utc=True)
                            benchmark_at = pd.to_datetime(
                                dataframe[timestamp_field], utc=True
                            )
                            age = traded_at - benchmark_at
                            # Do not ffill a partner feed indefinitely.
                            fresh = age.ge(pd.Timedelta(0)) & age.le(
                                pd.Timedelta(minutes=15)
                            )
                            benchmark = dataframe[field].where(fresh)
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                LOG.warning("Benchmark unavailable for %s (%s); entry disabled", pair, exc)
        output = calculate_features(
            dataframe, benchmark_close=benchmark, config=self.signal_config
        )
        return output

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair")
        enabled = pair in PAIRS
        dataframe["enter_long"] = (dataframe["enter_long"] if enabled else 0)
        dataframe["enter_short"] = (dataframe["enter_short"] if enabled else 0)
        dataframe["enter_tag"] = "causal_wavelet_rsi_spread"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit signals remain operative if the partner feed is lost.
        return dataframe
