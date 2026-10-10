"""Freqtrade V3 interface; imports are isolated from the standalone library.

Trading disabled by default in config.example.json. Only BTC/BNB can enter.
Other pairs and unavailable / stale benchmark data fail closed for entries.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from freqtrade.strategy import IStrategy

from .core import SignalConfig, calculate_features
from .market_data import align_previous_benchmark

LOG = logging.getLogger(__name__)

PAIRS = ("BTC/USDT:USDT", "BNB/USDT:USDT")


class BaseBTCCointegrationStrategy(IStrategy):
    """Research implementation, not validated profitable execution software."""

    INTERFACE_VERSION = 3
    timeframe = "15m"
    startup_candle_count = 200
    # Never activate an unvalidated model by default.
    mlsd_enabled = os.environ.get("BNB_BTC_MLSD_ENABLED", "0") == "1"
    mlsd_model_path = os.environ.get("BNB_BTC_MLSD_MODEL_PATH", "")
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
                # No same-candle informative merge. Use ONLY benchmark at t-15m
                # and exact timestamp matches, like the offline training CLI.
                benchmark = align_previous_benchmark(dataframe, reference)
            except Exception as exc:
                LOG.warning(
                    "Counterpart candle alignment failed for %s: %s; entries fail closed",
                    pair, exc,
                )
                output = calculate_features(
            dataframe, benchmark_close=benchmark, config=self.signal_config
        )
        if self.mlsd_enabled:
            # Fail closed for *entries* on missing model, peer, schema drift,
            # upstream MLSD import errors or data integrity violations.
            # Exits and broker stoploss are independent of the model.
            output["mlsd_prob_short"] = float("nan")
            output["mlsd_prob_neutral"] = float("nan")
            output["mlsd_prob_long"] = float("nan")
            output["mlsd_gate"] = False
            try:
                if benchmark is None:
                    raise ValueError("benchmark feed unavailable")
                from .mlsd_layer import infer_probabilities, read_model

                pair_key = ("BNB_BTC_MLSD_BTC_MODEL_PATH" if pair == PAIRS[0]
                            else "BNB_BTC_MLSD_BNB_MODEL_PATH")
                model_path = os.environ.get(pair_key) or self.mlsd_model_path
                model = read_model(model_path)
                if model["asset_pair"] != pair:
                    raise ValueError("model artifact asset_pair does not match traded pair")
                probs = infer_probabilities(output, benchmark, model)
                for field in probs.columns:
                    output[field] = probs[field]
                limit = model["threshold"]
                output["enter_long"] = (
                    (output["enter_long"] == 1)
                    & probs["mlsd_prob_long"].ge(limit)
                ).astype(int)
                output["enter_short"] = (
                    (output["enter_short"] == 1)
                    & probs["mlsd_prob_short"].ge(limit)
                ).astype(int)
                output["mlsd_gate"] = probs.notna().all(axis=1)
            except Exception as exc:
                # No silent fallback to the technical-only model on MLSD failure.
                LOG.error("MLSD fail-closed for %s: %s", pair, exc)
                output["enter_long"] = 0
                output["enter_short"] = 0
        return output

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair")
        enabled = pair in PAIRS
        dataframe["enter_long"] = (dataframe["enter_long"] if enabled else 0)
        dataframe["enter_short"] = (dataframe["enter_short"] if enabled else 0)
        dataframe["enter_tag"] = (
            "causal_wavelet_rsi_spread_mlsd"
            if self.mlsd_enabled else "causal_wavelet_rsi_spread"
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit signals remain operative if the partner feed is lost.
        return dataframe
