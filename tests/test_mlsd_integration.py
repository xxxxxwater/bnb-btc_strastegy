"""MLSD 0.2.1 real upstream integration / causal training tests."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("MLSD")
from bnb_btc_strategy.mlsd_layer import (
    MLSD_SOURCE_SHA, extract_mlsd_features, infer_probabilities, make_structured_frame,
    read_model,
)
from bnb_btc_strategy.mlsd_train import train_offline


def sample(n=450):
    rng = np.random.default_rng(2026)
    t = np.arange(n)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rets = rng.normal(0, 0.002, n)
    price = 100 * np.exp(np.cumsum(rets))
    peer = 85 * np.exp(np.cumsum(0.55 * rets + rng.normal(0, 0.0018, n)))
    dataframe = pd.DataFrame({
        "date": dates, "open": price, "close": price,
        "high": price * (1.002 + 0.001 * np.sin(t / 7)**2),
        "low": price * (0.998 - 0.001 * np.cos(t / 5)**2),
        "volume": rng.uniform(2, 50, n),
    })
    # Index deliberately RangeIndex, matching the Freqtrade adapter.
    return dataframe, pd.Series(peer, index=dataframe.index)


def test_real_mlsd_transforms_5_series_to_105_columns():
    bars, peer = sample(120)
    features, positions = extract_mlsd_features(bars, peer, window=32)
    assert len(features) == len(positions) == 88
    assert positions[0] == 32
    assert features.shape[1] == 105
    assert "asset_returns__value_mean" in features.columns
    assert "benchmark_returns__diff1_std" in features.columns
    assert features.index.is_monotonic_increasing
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_mlsd_windows_never_read_future():
    bars, peer = sample(95)
    base, base_positions = extract_mlsd_features(bars, peer)
    mutated = bars.copy()
    mutated.loc[75:, "close"] *= 3
    mutated.loc[75:, "open"] *= 3
    mutated.loc[75:, "high"] *= 3
    mutated.loc[75:, "low"] *= 3
    later, later_positions = extract_mlsd_features(mutated, peer)
    assert np.array_equal(base_positions, later_positions)
    pd.testing.assert_frame_equal(base.iloc[:43], later.iloc[:43])


def test_corrupt_peer_or_irregular_dates_fail_closed():
    bars, peer = sample(120)
    broken = peer.copy()
    broken.loc[60] = np.nan
    features, positions = extract_mlsd_features(bars, broken)
    assert 60 not in positions
    assert 93 in positions  # peer[60] also invalidates return at 61; window starts 62
    with pytest.raises(ValueError, match="index must exactly"):
        extract_mlsd_features(bars, broken.reset_index(drop=True).iloc[::-1])
    bars.loc[80, "date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="contiguous 15m"):
        make_structured_frame(bars, peer)


def test_purged_training_safe_json_and_inference(tmp_path):
    bars, peer = sample()
    model, report = train_offline(bars, peer, window=32, horizon=4, n_splits=2)
    assert model["mlsd_source_sha"] == MLSD_SOURCE_SHA
    assert model["classes"] == [-1, 0, 1]
    assert len(model["columns"]) == 105
    assert report["holdout"]["n_test"] > 0
    assert all(np.isfinite(f["balanced_accuracy"]) for f in report["folds"])
    assert pd.Timestamp(model["label_matured_until"]) < pd.Timestamp(
        report["holdout"]["from"]
    )
    destination = tmp_path / "trained.json"
    destination.write_text(json.dumps(model), encoding="utf-8")
    checked = read_model(destination)
    pred = infer_probabilities(bars, peer, checked)
    assert (pred.loc[:100].isna()).all().all()  # historical fit windows blocked
    valid = pred.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.sum(axis=1).to_numpy(), 1.0, atol=1e-10)


def test_invalid_model_schema_and_matrix_are_rejected(tmp_path):
    bars, peer = sample()
    model, _ = train_offline(bars, peer, n_splits=2)
    model["coef"] = [[1, 2]]
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions"):
        read_model(path)
    model["mlsd_source_sha"] = "untrusted"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        read_model(path)


def test_artifact_rejects_duplicate_json_keys_and_wrong_lag(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate model JSON key"):
        read_model(path)
    path.write_text(json.dumps({"schema": "bnb-btc-mlsd/v2",
                                "benchmark_lag_bars": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="lag contract"):
        read_model(path)


def test_full_data_fingerprint_changes_with_ohlcv_peer_and_time():
    from bnb_btc_strategy.mlsd_layer import dataset_fingerprint

    bars, peer = sample(80)
    reference = dataset_fingerprint(bars, peer)
    changed_high = bars.copy()
    changed_high.loc[20, "high"] *= 1.01
    assert dataset_fingerprint(changed_high, peer) != reference

    changed_peer = peer.copy()
    changed_peer.loc[20] *= 1.01
    assert dataset_fingerprint(bars, changed_peer) != reference

    shifted = bars.copy()
    shifted["date"] += pd.Timedelta(minutes=15)
    assert dataset_fingerprint(shifted, peer) != reference


def test_training_skips_malformed_future_target_bars():
    bars, peer = sample(450)
    reference, _ = train_offline(bars, peer, n_splits=2)
    broken = bars.copy()
    # This OHLCV corruption contaminates several future targets in addition
    # to the trailing feature windows. A valid earlier feature must not use
    # this malformed later candle as its label.
    broken.loc[120, "high"] = broken.loc[120, "close"] * 0.99
    model, report = train_offline(broken, peer, n_splits=2)
    assert model["dataset_sha256"] != reference["dataset_sha256"]
    assert model["training_rows"] < reference["training_rows"]
    assert report["holdout"]["n_test"] > 0


def test_offgrid_candle_series_rejected_without_peer_side_effects():
    bars, peer = sample(100)
    bars["date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="15m UTC grid"):
        extract_mlsd_features(bars, peer)


def test_large_finite_weights_do_not_produce_infinite_predictions():
    bars, peer = sample()
    model, _ = train_offline(bars, peer, n_splits=2)
    model["coef"] = [[1e308] * 105] * 3
    model["center"] = [0.0] * 105
    model["scale"] = [1.0] * 105
    with pytest.raises(ValueError, match="logits overflowed"):
        infer_probabilities(bars, peer, model)


@pytest.mark.parametrize("key", ["fee_threshold", "threshold"])
def test_training_rejects_nonfinite_tuning_values(key):
    bars, peer = sample(100)
    with pytest.raises(ValueError, match=key):
        train_offline(bars, peer, **{key: float("nan")})
