"""Training CLI must never install an artifact that the runtime rejects."""
import json

import numpy as np
import pandas as pd
import pytest

from bnb_btc_strategy import mlsd_train


def _csvs(tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    for symbol, price in [("btc", 200.0), ("bnb", 30.0)]:
        pd.DataFrame({"date": dates, "open": price, "high": price + 1,
                      "low": price - 1, "close": price,
                      "volume": 20}).to_csv(tmp_path / (symbol + ".csv"), index=False)
    return tmp_path / "btc.csv", tmp_path / "bnb.csv"


def test_rejected_artifact_does_not_replace_existing_model(tmp_path, monkeypatch):
    asset, peer = _csvs(tmp_path)
    model = tmp_path / "model.json"
    report = tmp_path / "report.json"
    model.write_text('{"previous": "trusted"}', encoding="utf-8")
    report.write_text('{"previous": "report"}', encoding="utf-8")
    monkeypatch.setattr(mlsd_train, "train_offline",
                        lambda *a, **k: ({"schema": "wrong"}, {"holdout": {}}))
    with pytest.raises(ValueError, match="schema mismatch"):
        mlsd_train.main(["--asset-pair", "BTC/USDT:USDT", "--asset", str(asset),
                         "--benchmark", str(peer), "--model-output", str(model),
                         "--report-output", str(report)])
    assert json.loads(model.read_text(encoding="utf-8")) == {"previous": "trusted"}
    assert json.loads(report.read_text(encoding="utf-8")) == {"previous": "report"}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bnb.csv", "btc.csv", "model.json", "report.json"]


def test_cli_installs_validated_model_and_report(tmp_path):
    pytest.importorskip("MLSD")
    from bnb_btc_strategy.mlsd_layer import read_model

    rng = np.random.default_rng(7)
    n = 450
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rets = rng.normal(0, 0.002, n)
    for symbol, scale, noise in [("btc", 100.0, 0.0), ("bnb", 85.0, 0.0018)]:
        price = scale * np.exp(np.cumsum(rets + rng.normal(0, noise, n)))
        pd.DataFrame({"date": dates, "open": price, "close": price,
                      "high": price * 1.003, "low": price * 0.997,
                      "volume": rng.uniform(2, 50, n)}).to_csv(
            tmp_path / (symbol + ".csv"), index=False)
    model = tmp_path / "out" / "model.json"
    report = tmp_path / "out" / "report.json"
    assert mlsd_train.main([
        "--asset-pair", "BTC/USDT:USDT", "--asset", str(tmp_path / "btc.csv"),
        "--benchmark", str(tmp_path / "bnb.csv"), "--model-output", str(model),
        "--report-output", str(report)]) == 0
    assert read_model(model)["asset_pair"] == "BTC/USDT:USDT"
    assert json.loads(report.read_text(encoding="utf-8"))["research_only"] is True
    assert sorted(p.name for p in model.parent.iterdir()) == ["model.json", "report.json"]
