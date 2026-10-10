"""Strategy-level check: exchange informative source uses prior candle only."""
import importlib
import sys
import types

import numpy as np
import pandas as pd


def test_strategy_matches_closed_peer_and_keeps_exit_when_peer_lost(monkeypatch):
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
    close = np.linspace(40000.0, 43000.0, len(dates))
    candles = pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.ones(len(dates)) * 80,
    })
    partner = pd.DataFrame({"date": dates, "close": close / 100})

    class DP:
        def get_pair_dataframe(self, pair, timeframe):
            assert pair == "BNB/USDT:USDT"
            return partner

    s = strategy_module.BaseBTCCointegrationStrategy()
    s.mlsd_enabled = False
    s.dp = DP()
    baseline = s.populate_indicators(candles, {"pair": "BTC/USDT:USDT"})

    # Modify only the partner's last candle: the last signal must not change.
    partner.loc[239, "close"] = 999999.0
    same = s.populate_indicators(candles, {"pair": "BTC/USDT:USDT"})
    assert baseline["spread_z"].iloc[-1] == same["spread_z"].iloc[-1]
    assert baseline["enter_long"].iloc[-1] == same["enter_long"].iloc[-1]
    assert baseline["enter_short"].iloc[-1] == same["enter_short"].iloc[-1]

    # Missing previous partner candle disables entries without disabling exits.
    partner.drop(index=238, inplace=True)
    gone = s.populate_indicators(candles, {"pair": "BTC/USDT:USDT"})
    assert not gone["signal_ready"].iloc[-1]
    assert gone["enter_long"].iloc[-1] == 0
    assert gone["enter_short"].iloc[-1] == 0
    assert gone["exit_long"].iloc[-1] in (0, 1)
    assert gone["exit_short"].iloc[-1] in (0, 1)
