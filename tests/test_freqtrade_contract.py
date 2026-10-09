"""Adapter contract tests with mocked Freqtrade, no exchange connectivity."""
import importlib
import sys
import types

import numpy as np
import pandas as pd


def test_missing_and_stale_partner_fail_closed(monkeypatch):
    fake = types.ModuleType("freqtrade")
    strategy = types.ModuleType("freqtrade.strategy")

    class IStrategy:
        pass

    def merge(dataframe, informative, timeframe, informative_timeframe, ffill=True):
        copied = informative.rename(columns={
            "benchmark_close": "benchmark_close_15m",
            "benchmark_timestamp": "benchmark_timestamp_15m",
        })
        return dataframe.merge(copied, on="date", how="left").ffill()

    strategy.IStrategy = IStrategy
    strategy.merge_informative_pair = merge
    monkeypatch.setitem(sys.modules, "freqtrade", fake)
    monkeypatch.setitem(sys.modules, "freqtrade.strategy", strategy)
    # Ensure fake Freqtrade is used if a previous test cached the adapter.
    sys.modules.pop("bnb_btc_strategy.freqtrade_strategy", None)
    module = importlib.import_module("bnb_btc_strategy.freqtrade_strategy")

    dates = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    prices = 30000 * np.exp(0.0001 * np.arange(300))
    candles = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": 100,
    })

    class DP:
        def get_pair_dataframe(self, pair, timeframe):
            assert pair == "BNB/USDT:USDT"
            assert timeframe == "15m"
            partner = candles.iloc[:-10][["date", "close"]].copy()
            partner["close"] *= 0.01
            return partner

    subject = module.BaseBTCCointegrationStrategy()
    subject.dp = DP()
    output = subject.populate_indicators(
        candles, {"pair": "BTC/USDT:USDT"}
    )
    assert not output["signal_ready"].iloc[-1]
    assert output["enter_long"].iloc[-1] == 0
    assert output["enter_short"].iloc[-1] == 0
    assert output["exit_long"].iloc[-1] in (0, 1)

    other = subject.populate_indicators(candles, {"pair": "ETH/USDT:USDT"})
    other = subject.populate_entry_trend(other, {"pair": "ETH/USDT:USDT"})
    assert other["enter_long"].sum() == 0
    assert other["enter_short"].sum() == 0
