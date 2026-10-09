import pandas as pd
import pytest

from bnb_btc_strategy.cli import load_csv, main


def test_load_csv_rejects_unsorted_and_duplicates(tmp_path):
    file = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-02", "2024-01-01"],
                  "open": [1, 1], "high": [2, 2], "low": [0.5, 0.5],
                  "close": [1, 1], "volume": [1, 1]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="chronologically"):
        load_csv(str(file))


def test_cli_emits_signal_file(tmp_path):
    dates = pd.date_range("2024-01-01", periods=250, freq="15min", tz="UTC")
    for symbol, price in [("btc", 200.0), ("bnb", 30.0)]:
        pd.DataFrame({"date": dates, "open": price, "high": price + 1,
                      "low": price - 1, "close": price,
                      "volume": 20}).to_csv(tmp_path / (symbol + ".csv"), index=False)
    out = tmp_path / "signals.csv"
    assert main(["--asset", str(tmp_path / "btc.csv"),
                 "--benchmark", str(tmp_path / "bnb.csv"),
                 "--output", str(out)]) == 0
    assert out.is_file()
    assert len(pd.read_csv(out)) == 250


def test_load_csv_rejects_gaps(tmp_path):
    file = tmp_path / "gaps.csv"
    pd.DataFrame({"date": ["2024-01-01T00:00:00Z",
                            "2024-01-01T00:30:00Z"],
                  "open": [10, 10], "high": [11, 11], "low": [9, 9],
                  "close": [10, 10], "volume": [1, 1]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="contiguous 15-minute"):
        load_csv(str(file))
