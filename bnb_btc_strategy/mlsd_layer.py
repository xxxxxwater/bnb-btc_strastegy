"""Actual MLSD 0.2.1 integration: causal OHLCV -> typed Series -> features.

No model is fitted here: BasicSeries is a stateless 21-column-per-series
summary. Learned scalers/classifiers are fitted only in mlsd_train.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "bnb-btc-mlsd/v2"
MLSD_SOURCE_SHA = "85e356c1fb6703fef2b03ac0070c5e0659e8c671"
FEATURE_SERIES = ("asset_returns", "benchmark_returns", "relative_returns",
                  "log_volume_delta", "candle_range")
BAR = pd.Timedelta(minutes=15)


def _validate_and_extract(
    candles: pd.DataFrame, benchmark_close: pd.Series
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], np.ndarray]:
    """Strict 15m candle and alignment contract; no positional peer mis-joins."""
    required = {"open", "high", "low", "close", "volume"}
    if not isinstance(candles, pd.DataFrame) or not required.issubset(candles):
        raise ValueError("asset requires OHLCV columns")
    if not isinstance(benchmark_close, pd.Series):
        raise TypeError("benchmark_close must be a pandas Series")
    if not benchmark_close.index.equals(candles.index):
        raise ValueError("benchmark_close index must exactly match asset rows")
    if "date" in candles:
        timestamps = pd.DatetimeIndex(pd.to_datetime(candles["date"], utc=True))
    elif isinstance(candles.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(pd.to_datetime(candles.index, utc=True))
    else:
        raise ValueError("asset needs a UTC-compatible date column or DatetimeIndex")
    if (timestamps.hasnans or not timestamps.is_unique
            or not timestamps.is_monotonic_increasing):
        raise ValueError("asset timestamps must be strictly increasing and unique")
    if len(timestamps) > 1 and not (timestamps[1:] - timestamps[:-1] == BAR).all():
        raise ValueError("asset candles must be contiguous 15m intervals")

    cols = {x: pd.to_numeric(candles[x], errors="coerce").to_numpy(dtype=float)
            for x in required}
    peer = pd.to_numeric(benchmark_close, errors="coerce").to_numpy(dtype=float)
    positive = (cols["close"] > 0) & (cols["open"] > 0) & (peer > 0)
    positive &= (cols["volume"] > 0) & (cols["low"] > 0)
    positive &= cols["high"] >= np.maximum(cols["open"], cols["close"])
    positive &= cols["low"] <= np.minimum(cols["open"], cols["close"])
    valid = positive & np.isfinite(peer)
    for name in required:
        valid &= np.isfinite(cols[name])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_asset = np.log(np.where(cols["close"] > 0, cols["close"], np.nan))
        log_peer = np.log(np.where(peer > 0, peer, np.nan))
        log_volume = np.log1p(np.where(cols["volume"] > 0, cols["volume"], np.nan))
        r_asset = np.r_[np.nan, np.diff(log_asset)]
        r_peer = np.r_[np.nan, np.diff(log_peer)]
        delta_volume = np.r_[np.nan, np.diff(log_volume)]
        spread = (cols["high"] - cols["low"]) / cols["close"]
    arrays = {
        "asset_returns": r_asset,
        "benchmark_returns": r_peer,
        "relative_returns": r_asset - r_peer,
        "log_volume_delta": delta_volume,
        "candle_range": spread,
    }
    return timestamps, arrays, valid


def make_structured_frame(
    candles: pd.DataFrame,
    benchmark_close: pd.Series,
    window: int = 32,
):
    """Return (MLSD.SDataFrame, original integer positions).

    Every sample at row t contains ONLY [t-window+1, ..., t]; missing
    peers or invalid OHLCV in any window drop the entire sample.
    """
    if not isinstance(window, int) or not 8 <= window <= 256:
        raise ValueError("window must be an integer in [8, 256]")
    from MLSD import SData, SDataFrame  # optional dependency, real upstream API

    times, arrays, valid = _validate_and_extract(candles, benchmark_close)
    rows = {name: [] for name in FEATURE_SERIES}
    positions = []
    for t in range(window, len(candles)):
        begin = t - window + 1
        if not valid[begin:t + 1].all():
            continue
        if not all(np.isfinite(values[begin:t + 1]).all()
                   for values in arrays.values()):
            continue
        for name in FEATURE_SERIES:
            rows[name].append(arrays[name][begin:t + 1].copy())
        positions.append(t)

    if not positions:
        raise ValueError("no complete, valid MLSD windows")
    index = times.take(positions)
    frame = SDataFrame([
        SData(rows[name], index=index, column=name, dtype="Series")
        for name in FEATURE_SERIES
    ])
    return frame, np.asarray(positions, dtype=int)


def extract_mlsd_features(
    candles: pd.DataFrame,
    benchmark_close: pd.Series,
    window: int = 32,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature-only research helper; MLSD BasicSeries does not learn from rows."""
    from MLSD.validation import require_numeric_features

    frame, positions = make_structured_frame(candles, benchmark_close, window)
    output = frame.fit_transform()
    require_numeric_features(output, allow_missing=False)
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("MLSD produced nonfinite features")
    return output, positions


def dataset_fingerprint(candles: pd.DataFrame, benchmark_close: pd.Series) -> str:
    """Hash the *entire* ordered training feature source including timestamps."""
    from .market_data import _timestamps

    times, _, _ = _validate_and_extract(candles, benchmark_close)
    # Require one canonical UTC grid. Date changes must alter the fingerprint.
    if not _timestamps(candles, "asset").equals(times):
        raise ValueError("dataset timestamps changed during validation")
    digest = hashlib.sha256()
    digest.update(times.asi8.astype("<i8").tobytes())
    for name in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(candles[name], errors="coerce").to_numpy(dtype="<f8")
        digest.update(values.tobytes())
    peer = pd.to_numeric(benchmark_close, errors="coerce").to_numpy(dtype="<f8")
    digest.update(peer.tobytes())
    return digest.hexdigest()


def _no_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate model JSON key: {key}")
        result[key] = value
    return result


def read_model(path: str | Path) -> dict:
    """Load checked JSON ONLY (never unpickle an untrusted model)."""
    file = Path(path)
    if not file.is_file() or file.stat().st_size > 2_000_000:
        raise ValueError("model JSON file absent or too large")
    with file.open(encoding="utf-8") as handle:
        model = json.load(handle, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(model, dict) or model.get("schema") != SCHEMA:
        raise ValueError("MLSD model schema mismatch")
    if model.get("benchmark_lag_bars") != 1:
        raise ValueError("model benchmark lag contract mismatch")
    if model.get("mlsd_source_sha") != MLSD_SOURCE_SHA:
        raise ValueError("MLSD upstream revision mismatch")
    if not isinstance(model.get("columns"), list) or not model["columns"]:
        raise ValueError("missing model feature columns")
    if len(model["columns"]) != 105 or not all(isinstance(c, str) for c in model["columns"]):
        raise ValueError("unexpected MLSD feature schema")
    if len(model["columns"]) != len(set(model["columns"])):
        raise ValueError("duplicate model features")
    if model.get("asset_pair") not in ("BTC/USDT:USDT", "BNB/USDT:USDT"):
        raise ValueError("model must be bound to BTC or BNB asset pair")
    if model.get("classes") != [-1, 0, 1]:
        raise ValueError("expected three model classes [-1, 0, 1]")
    n = len(model["columns"])
    coef = np.asarray(model.get("coef"), dtype=float)
    intercept = np.asarray(model.get("intercept"), dtype=float)
    center = np.asarray(model.get("center"), dtype=float)
    scale = np.asarray(model.get("scale"), dtype=float)
    if (coef.shape != (3, n) or intercept.shape != (3,)
            or center.shape != (n,) or scale.shape != (n,)):
        raise ValueError("model matrix dimensions mismatch")
    if not all(np.isfinite(a).all() for a in (coef, intercept, center, scale)):
        raise ValueError("model contains nonfinite parameters")
    if (scale <= 0).any() or type(model.get("window")) is not int or model["window"] not in range(8, 257):
        raise ValueError("invalid scaler or window")
    if type(model.get("horizon")) is not int or model["horizon"] not in range(1, 33):
        raise ValueError("invalid training horizon")
    if not 0.5 < model.get("threshold", 0) < 1:
        raise ValueError("model confidence threshold must be >0.5 and <1")
    watermark = pd.Timestamp(model.get("label_matured_until"))
    trained = pd.Timestamp(model.get("trained_through"))
    if pd.isna(trained) or trained.tzinfo is None or trained > watermark:
        raise ValueError("training cutoff or label maturity is invalid")
    fingerprint = model.get("dataset_sha256")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("model dataset fingerprint missing or malformed")
    if pd.isna(watermark) or watermark.tzinfo is None:
        raise ValueError("model needs a timezone-aware training maturity cutoff")
    return model


def infer_probabilities(
    candles: pd.DataFrame, benchmark_close: pd.Series, model: dict,
) -> pd.DataFrame:
    """Return 3 class probabilities aligned to input rows, or NaN if ineligible.

    Predictions are intentionally suppressed at or before the timestamp when
    the model's last training label matured (anti in-sample/backtest leakage).
    """
    if (model.get("schema") != SCHEMA or model.get("mlsd_source_sha") != MLSD_SOURCE_SHA
            or model.get("benchmark_lag_bars") != 1):
        raise ValueError("untrusted model schema/upstream")
    features, positions = extract_mlsd_features(
        candles, benchmark_close, window=model["window"]
    )
    if list(features.columns) != model["columns"]:
        raise ValueError("MLSD feature ordering mismatch")
    mat = features.to_numpy(dtype=float)
    center = np.asarray(model["center"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    if scale.shape != (mat.shape[1],) or coef.shape != (3, mat.shape[1],):
        raise ValueError("MLSD coefficient dimensions mismatch")
    if (not np.isfinite(mat).all() or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
        raise ValueError("invalid MLSD feature matrix or model coefficients")
    logits = ((mat - center) / scale) @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    valid = features.index > pd.Timestamp(model["label_matured_until"])
    columns = ["mlsd_prob_short", "mlsd_prob_neutral", "mlsd_prob_long"]
    output = pd.DataFrame(np.nan, index=candles.index, columns=columns)
    selected = positions[valid]
    if len(selected):
        output.iloc[selected, :] = probs[np.asarray(valid)]
    return output
