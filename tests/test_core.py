import numpy as np
import pandas as pd
import pytest

from bnb_btc_strategy.core import SignalConfig, calculate_features


def candles(n=320):
    t = np.arange(n, dtype=float)
    prices = 100 * np.exp(0.0003 * t + 0.013 * np.sin(t / 8))
    frame = pd.DataFrame({
        "open": prices, "high": prices * 1.01,
        "low": prices * 0.99, "close": prices,
        "volume": np.full(n, 100.0)
    })
    partner = pd.Series(80 * np.exp(0.00025 * t + 0.011 * np.sin(t / 8)))
    return frame, partner


def test_output_is_finite_when_ready_and_mutually_exclusive():
    data, benchmark = candles()
    out = calculate_features(data, benchmark)
    assert len(out) == len(data)
    assert not (out["enter_long"] & out["enter_short"]).any()
    assert np.isfinite(out["signal_score"].iloc[150:]).all()
    assert out.loc[out["signal_ready"], "signal_score"].abs().le(1).all()
    assert out["signal_score"].iloc[:47].isna().all()  # EMA(48) ready at index 47
    assert np.isfinite(out["signal_score"].iloc[47:]).all()


def test_past_does_not_change_when_future_prices_are_edited():
    df, benchmark = candles()
    original = calculate_features(df, benchmark)
    modified = df.copy()
    modified.loc[280:, "close"] *= 2
    modified.loc[280:, "high"] *= 2
    modified.loc[280:, "low"] *= 2
    modified.loc[280:, "open"] *= 2
    later = calculate_features(modified, benchmark)
    pd.testing.assert_series_equal(
        original["signal_score"].iloc[:280],
        later["signal_score"].iloc[:280]
    )
    pd.testing.assert_series_equal(
        original["enter_long"].iloc[:280], later["enter_long"].iloc[:280]
    )


def test_missing_benchmark_fail_closed():
    data, _ = candles()
    result = calculate_features(data)
    assert not result["signal_ready"].any()
    assert result["enter_long"].sum() == 0
    assert result["enter_short"].sum() == 0


def test_bad_volume_and_price_prevent_entry():
    data, benchmark = candles()
    data.loc[200, "volume"] = 0
    data.loc[201, "close"] = -1
    out = calculate_features(data, benchmark)
    assert not out.loc[200, "signal_ready"]
    assert not out.loc[201, "signal_ready"]


def test_short_history_never_enters():
    df, benchmark = candles(50)
    assert calculate_features(df, benchmark)["enter_long"].sum() == 0


def test_bad_parameters():
    with pytest.raises(ValueError):
        SignalConfig(fast_ema=100, slow_ema=30)
    with pytest.raises(ValueError):
        SignalConfig(score_threshold=1.5)


def test_no_mutation_of_input():
    df, benchmark = candles()
    cols = list(df.columns)
    calculate_features(df, benchmark)
    assert list(df.columns) == cols
