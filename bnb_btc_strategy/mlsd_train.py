"""Research-only MLSD training, purged OOS validation and JSON artifact export.

No trades, no performance claims, no serialization of executable pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _features_on_split(source, train_ids, eval_ids):
    """Fit MLSD transformers strictly on training rows, then transform eval."""
    from MLSD.validation import require_numeric_features

    fitted = source.take(train_ids)
    training = fitted.fit_transform()
    heldout = fitted.transform(source.take(eval_ids))
    require_numeric_features(training, allow_missing=False)
    require_numeric_features(heldout, allow_missing=False)
    if list(training.columns) != list(heldout.columns):
        raise ValueError("Feature schema drift")
    return training, heldout


def _classifier(x_train, y_train, x_eval):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if set(np.unique(y_train)) != {-1, 0, 1}:
        raise ValueError("each training window must include all three labels")
    scaler = StandardScaler()
    a = scaler.fit_transform(x_train)
    b = scaler.transform(x_eval)
    clf = LogisticRegression(max_iter=600, class_weight="balanced")
    clf.fit(a, y_train)
    if list(clf.classes_) != [-1, 0, 1]:
        raise ValueError("unexpected label classes")
    return clf.predict(b), scaler, clf


def train_offline(
    candles: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    horizon: int = 4,
    window: int = 32,
    fee_threshold: float = 0.001,
    threshold: float = 0.60,
    n_splits: int = 3,
    asset_pair: str = "BTC/USDT:USDT",
) -> tuple[dict, dict]:
    """Fit on past ONLY; return JSON-safe artifact and non-PnL OOS metrics."""
    from sklearn.metrics import balanced_accuracy_score
    from MLSD.model_selection import PurgedEventTimeSeriesSplit

    from .mlsd_layer import BAR, MLSD_SOURCE_SHA, SCHEMA, make_structured_frame

    if asset_pair not in ("BTC/USDT:USDT", "BNB/USDT:USDT"):
        raise ValueError("asset_pair must be BTC/USDT:USDT or BNB/USDT:USDT")
    if not isinstance(horizon, int) or not 1 <= horizon <= 32:
        raise ValueError("horizon must be in [1, 32]")
    if not (0 < fee_threshold < 0.1):
        raise ValueError("fee_threshold must lie in (0, 0.1)")
    if not (0.5 < threshold < 1):
        raise ValueError("threshold must lie in (0.5, 1)")
    if not isinstance(n_splits, int) or not 2 <= n_splits <= 5:
        raise ValueError("n_splits must be 2 through 5")

    raw, positions = make_structured_frame(candles, benchmark_close, window)
    prices = pd.to_numeric(candles["close"], errors="coerce").to_numpy(dtype=float)
    times = (pd.DatetimeIndex(pd.to_datetime(candles["date"], utc=True))
             if "date" in candles else pd.DatetimeIndex(candles.index))
    keep = []
    labels = []
    for j, t in enumerate(positions):
        if t + horizon >= len(prices):
            continue
        forward = prices[t + horizon] / prices[t]
        if not np.isfinite(forward) or forward <= 0:
            continue
        change = float(np.log(forward))
        labels.append(1 if change > fee_threshold else
                      -1 if change < -fee_threshold else 0)
        keep.append(j)
    if len(keep) < 160:
        raise ValueError("need >=160 fully matured, valid labelled MLSD samples")
    raw = raw.take(keep)
    y = np.asarray(labels, dtype=int)
    event_ends = pd.Series(raw.index + horizon * BAR, index=raw.index)
    splitter = PurgedEventTimeSeriesSplit(
        event_ends, n_splits=n_splits, gap=0
    )
    folds = []
    for train_ids, test_ids in splitter.split(raw):
        x_train, x_test = _features_on_split(raw, train_ids, test_ids)
        predicted, _, _ = _classifier(x_train, y[train_ids], x_test)
        folds.append({
            "train_rows": int(len(train_ids)),
            "test_rows": int(len(test_ids)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_ids], predicted)),
            "first_test_time": raw.index[test_ids[0]].isoformat(),
        })

    # Final model: train before the held-out trailing 20%, purge label overlap.
    test_begin = int(len(raw) * 0.8)
    first_test = raw.index[test_begin]
    train_ids = np.flatnonzero((event_ends < first_test).to_numpy())
    test_ids = np.arange(test_begin, len(raw))
    x_train, x_test = _features_on_split(raw, train_ids, test_ids)
    predicted, scaler, classifier = _classifier(x_train, y[train_ids], x_test)
    last_label_maturity = event_ends.iloc[train_ids[-1]]
    if not last_label_maturity < first_test:
        raise AssertionError("label leakage into final holdout")

    artifact = {
        "schema": SCHEMA,
        "asset_pair": asset_pair,
        "mlsd_source_sha": MLSD_SOURCE_SHA,
        "window": window,
        "horizon": horizon,
        "fee_threshold": fee_threshold,
        "threshold": threshold,
        "classes": [-1, 0, 1],
        "columns": list(x_train.columns),
        "center": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "label_matured_until": last_label_maturity.isoformat(),
        "trained_through": raw.index[train_ids[-1]].isoformat(),
        "training_rows": int(len(train_ids)),
        "label_counts": {str(k): int((y[train_ids] == k).sum()) for k in (-1, 0, 1)},
        "dataset_sha256": hashlib.sha256(
            np.ascontiguousarray(prices).tobytes()
            + np.ascontiguousarray(pd.to_numeric(benchmark_close).to_numpy()).tobytes()
        ).hexdigest(),
    }
    report = {
        "research_only": True,
        "label_definition": "forward log return threshold, net fees NOT simulated",
        "purging": "label end strictly earlier than first test candle",
        "folds": folds,
        "holdout": {
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
            "from": first_test.isoformat(),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_ids], predicted)),
        },
    }
    return artifact, report


def main(argv: list[str] | None = None) -> int:
    from .cli import load_csv
    from .market_data import align_previous_benchmark
    from .mlsd_layer import read_model

    parser = argparse.ArgumentParser(description="Train MLSD research model without trading")
    parser.add_argument("--asset-pair", required=True,
                        choices=["BTC/USDT:USDT", "BNB/USDT:USDT"])
    parser.add_argument("--asset", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--fee-threshold", type=float, default=0.001)
    args = parser.parse_args(argv)
    asset = load_csv(args.asset)
    peer = load_csv(args.benchmark)
    # Only *prior* benchmark candle may be used by the feature extractor.
    aligned = align_previous_benchmark(asset, peer)
    artifact, report = train_offline(
        asset, aligned, horizon=args.horizon, window=args.window,
        fee_threshold=args.fee_threshold, asset_pair=args.asset_pair
    )
    # Atomic writes: no partially readable artifact.
    destination = Path(args.model_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(destination)
    read_model(destination)
    target = Path(args.report_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"model": str(destination), "report": str(target),
                      "holdout": report["holdout"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
