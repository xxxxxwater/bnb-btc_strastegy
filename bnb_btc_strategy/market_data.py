"""Single conservative closed-candle benchmark alignment contract.

Freqtrade candle date is normally the candle OPEN timestamp. At an asset
row with open timestamp t, the benchmark is only allowed to use its
previous candle (open t-15m), not the potentially still-forming row at t.

No ffill, bfill or nearest merge: a missing prior peer bar fails closed.
When the peer frame carries volume, zero-volume rows (Freqtrade's synthetic
gap fill) count as missing too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR = pd.Timedelta(minutes=15)


def _timestamps(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if "date" in frame.columns:
        dates = pd.DatetimeIndex(pd.to_datetime(
            frame["date"], errors="raise", utc=True
        ))
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    else:
        raise ValueError(f"{name} must have a date column or DatetimeIndex")
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"{name} has missing, duplicate or unsorted timestamps")
    if (dates.asi8 % BAR.value != 0).any():
        raise ValueError(f"{name} timestamps must align to 15-minute UTC boundaries")
    return dates


def align_previous_benchmark(
    asset: pd.DataFrame, peer: pd.DataFrame
) -> pd.Series:
    """Align exactly peer[t-15m] to asset[t], indexed as asset.

    Consecutive closed asset candles use previously closed peer candles,
    even when a future peer candle is supplied. Missing periods produce
    NaN instead of using stale candles. The first available peer row cannot
    be used until the following asset bar.
    """
    asset_times = _timestamps(asset, "asset")
    peer_times = _timestamps(peer, "benchmark")
    if len(asset_times) > 1 and not (asset_times[1:] - asset_times[:-1] == BAR).all():
        raise ValueError("asset has missing or irregular 15-minute candles")
    if "close" not in peer.columns:
        raise ValueError("benchmark must contain close")
    if "close" not in asset.columns:
        raise ValueError("asset must contain close")
    values = pd.to_numeric(peer["close"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(values) & (values > 0)
    if "volume" in peer.columns:
        # Freqtrade's ohlcv_fill_up_missing_data turns a missing candle into a
        # zero-volume row carrying the previous close: a stale forward fill.
        volume = pd.to_numeric(peer["volume"], errors="coerce").to_numpy(dtype=float)
        usable &= np.isfinite(volume) & (volume > 0)
    values = np.where(usable, values, np.nan)
    # The shift is on the time axis, NOT the peer row position. A gap cannot
    # accidentally be filled by the next available row.
    available_at = peer_times + BAR
    shifted = pd.Series(values, index=available_at)
    previous = shifted.reindex(asset_times).to_numpy(dtype=float)
    return pd.Series(previous, index=asset.index, name="benchmark_close")
