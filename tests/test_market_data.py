"""Training and runtime share the same strict peer-as-of contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bnb_btc_strategy.market_data import align_previous_benchmark


def feeds(n=8):
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    asset = pd.DataFrame({"date": dates, "close": np.arange(n) + 100.0})
    peer = pd.DataFrame({"date": dates, "close": np.arange(n) + 1000.0})
    return asset, peer


def test_prior_bar_only_and_index_preserved():
    asset, peer = feeds()
    aligned = align_previous_benchmark(asset, peer)
    assert aligned.index.equals(asset.index)
    assert np.isnan(aligned.iloc[0])
    assert aligned.iloc[1] == 1000.0
    assert aligned.iloc[-1] == 1006.0
    # A contemporaneous benchmark candle must not change current signal.
    latest = peer.copy()
    latest.loc[7, "close"] = 9e12
    again = align_previous_benchmark(asset, latest)
    pd.testing.assert_series_equal(aligned, again)


def test_exact_timestamps_never_forward_fill_missing_or_future_peers():
    asset, peer = feeds()
    missing = peer.drop(index=3)
    values = align_previous_benchmark(asset, missing)
    assert np.isnan(values.iloc[4])  # partner[3] absent
    assert values.iloc[5] == 1004.0
    # If last partner candle is older than 15 minutes, entry stays blocked.
    old = peer.iloc[:4]
    last = align_previous_benchmark(asset, old)
    assert np.isnan(last.iloc[-1])
    assert last.iloc[4] == 1003.0


@pytest.mark.parametrize("bad", ["duplicate", "out_of_order", "off_grid", "nat"])
def test_bad_peer_date_contract_fails_closed(bad):
    asset, peer = feeds()
    if bad == "duplicate":
        peer.loc[3, "date"] = peer.loc[2, "date"]
    elif bad == "out_of_order":
        peer.loc[3, "date"] = peer.loc[5, "date"]
    elif bad == "off_grid":
        peer.loc[3, "date"] += pd.Timedelta(seconds=1)
    else:
        peer.loc[3, "date"] = pd.NaT
    with pytest.raises(ValueError):
        align_previous_benchmark(asset, peer)


def test_datetime_index_csv_and_feed_dataframe_parity():
    asset, peer = feeds()
    csv_asset = asset.set_index("date")
    csv_peer = peer.set_index("date")
    offline = align_previous_benchmark(csv_asset, csv_peer)
    online = align_previous_benchmark(asset, peer)
    np.testing.assert_allclose(offline.to_numpy()[1:],
                               online.to_numpy()[1:])
    assert offline.index.equals(csv_asset.index)


def test_invalid_peer_price_is_not_forward_filled():
    asset, peer = feeds()
    peer.loc[2, "close"] = -1.0
    result = align_previous_benchmark(asset, peer)
    assert np.isnan(result.iloc[3])
    assert result.iloc[4] == 1003.0


def test_missing_asset_interval_fails_closed_instead_of_rolling_past_gap():
    asset, peer = feeds()
    asset = asset.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="irregular 15-minute"):
        align_previous_benchmark(asset, peer)
