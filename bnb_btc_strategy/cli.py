"""Offline CSV signal export. No exchange orders or backtest performance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import calculate_features
from .market_data import align_previous_benchmark


def load_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must include date,open,high,low,close,volume")
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="raise")
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("CSV dates must be unique and valid")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("CSV must be sorted chronologically")
    gaps = data["date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("CSV must contain contiguous 15-minute candles")
    return data.set_index("date")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline causal BTC/BNB signal research")
    parser.add_argument("--asset", required=True, help="BTC or BNB OHLCV CSV")
    parser.add_argument("--benchmark", required=True, help="Other asset OHLCV CSV")
    parser.add_argument("--output", default="signals.csv")
    args = parser.parse_args(argv)

    asset = load_csv(args.asset)
    benchmark = load_csv(args.benchmark)
    # Only the previous completed benchmark candle may be used.
    aligned = align_previous_benchmark(asset, benchmark)
    result = calculate_features(asset, benchmark_close=aligned)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    last = result.iloc[-1]
    print(json.dumps({"output": args.output, "rows": len(result),
                      "ready": bool(last["signal_ready"]),
                      "enter_long": int(last["enter_long"]),
                      "enter_short": int(last["enter_short"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
