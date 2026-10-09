"""Freqtrade MLSD failures do not allow fallback or disable exits."""
import importlib
import sys
import types

import numpy as np
import pandas as pd


def test_missing_artifact_suppresses_entries_but_not_exits(monkeypatch):
    fake = types.ModuleType("freqtrade")
    strategy = types.ModuleType("freqtrade.strategy")

    class IStrategy:
        pass

    def merge(dataframe, informative, timeframe, informative_timeframe, ffill=True):
        peer = informative.rename(columns={
            "benchmark_close": "benchmark_close_15m",
            "benchmark_timestamp": "benchmark_timestamp_15m",
        })
        return dataframe.merge(peer, on="date", how="left").ffill()

    strategy.IStrategy = IStrategy
    strategy.merge_informative_pair = merge
    monkeypatch.setitem(sys.modules, "freqtrade", fake)
    monkeypatch.setitem(sys.modules, "freqtrade.strategy", strategy)
    sys.modules.pop("bnb_btc_strategy.freqtrade_strategy", None)
    module = importlib.import_module("bnb_btc_strategy.freqtrade_strategy")
    dates = pd.date_range("2024-01-01", periods=240, freq="15min", tz="UTC")
    close = 100 * np.exp(np.arange(240) / 1000)
    candles = pd.DataFrame({
        "date": dates, "open": close, "close": close,
        "high": close * 1.01, "low": close * 0.99, "volume": 10.0
    })

    class DP:
        def get_pair_dataframe(self, pair, timeframe):
            return candles[["date", "close"]].copy()

    instance = module.BaseBTCCointegrationStrategy()
    instance.dp = DP()
    instance.mlsd_enabled = True
    instance.mlsd_model_path = "/absolutely/nonexistent/untrusted.json"
    output = instance.populate_indicators(
        candles, {"pair": "BTC/USDT:USDT"}
    )
    assert output["enter_long"].sum() == 0
    assert output["enter_short"].sum() == 0
    assert output["mlsd_gate"].eq(False).all()
    assert output["exit_long"].notna().all()
    assert output["exit_short"].notna().all()
