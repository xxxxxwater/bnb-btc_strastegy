"""Freqtrade 2026.x loads backtest OHLCV as datetime64[ms, UTC] and pandas 3
parses exchange candles as us: the timestamp contract must not depend on unit."""
import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from bnb_btc_strategy.market_data import align_previous_benchmark


def feeds(n=8):
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    asset = pd.DataFrame({"date": dates, "close": np.arange(n) + 100.0})
    peer = pd.DataFrame({"date": dates, "close": np.arange(n) + 1000.0})
    return asset, peer


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_non_nanosecond_dates_align_like_nanoseconds(unit):
    asset, peer = feeds()
    reference = align_previous_benchmark(asset, peer)
    asset["date"] = asset["date"].dt.as_unit(unit)
    peer["date"] = peer["date"].dt.as_unit(unit)
    pd.testing.assert_series_equal(align_previous_benchmark(asset, peer), reference)
    indexed = align_previous_benchmark(asset.set_index("date"), peer.set_index("date"))
    np.testing.assert_array_equal(indexed.to_numpy(), reference.to_numpy())


def test_non_nanosecond_offgrid_dates_still_rejected():
    asset, peer = feeds()
    asset["date"] = (asset["date"] + pd.Timedelta(seconds=1)).dt.as_unit("ms")
    with pytest.raises(ValueError, match="15-minute UTC"):
        align_previous_benchmark(asset, peer)


def test_strategy_signals_identical_for_freqtrade_millisecond_dates(monkeypatch):
    freqtrade = types.ModuleType("freqtrade")
    module = types.ModuleType("freqtrade.strategy")

    class IStrategy:
        pass

    module.IStrategy = IStrategy
    monkeypatch.setitem(sys.modules, "freqtrade", freqtrade)
    monkeypatch.setitem(sys.modules, "freqtrade.strategy", module)
    sys.modules.pop("bnb_btc_strategy.freqtrade_strategy", None)
    strategy_module = importlib.import_module("bnb_btc_strategy.freqtrade_strategy")
    dates = pd.date_range("2024-01-01", periods=240, freq="15min", tz="UTC")
    t = np.arange(len(dates), dtype=float)
    close = 40000 * np.exp(0.0003 * t + 0.013 * np.sin(t / 8))
    peer_close = 300 * np.exp(0.00025 * t + 0.011 * np.sin(t / 8))

    def run(unit):
        stamp = dates.as_unit(unit)
        candles = pd.DataFrame({
            "date": stamp, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 80.0,
        })
        partner = pd.DataFrame({"date": stamp, "close": peer_close, "volume": 50.0})

        class DP:
            def get_pair_dataframe(self, pair, timeframe):
                return partner

        s = strategy_module.BaseBTCCointegrationStrategy()
        s.mlsd_enabled = False
        s.dp = DP()
        return s.populate_indicators(candles, {"pair": "BTC/USDT:USDT"})

    reference = run("ns")
    backtest = run("ms")
    assert reference["signal_ready"].any()
    cols = ["spread_z", "signal_ready", "enter_long", "enter_short",
            "exit_long", "exit_short"]
    pd.testing.assert_frame_equal(backtest[cols], reference[cols])
