# MLSD integration: v0.4.0

Upstream repo: https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data
Pinned source SHA: **85e356c1fb6703fef2b03ac0070c5e0659e8c671** (upstream v0.2.1).
Dependency is installed from its source archive; no MLSD fork, copied third-party source or private unpublished model is bundled.

## Pipeline and contracts

1. Candles: strictly chronological **closed 15-minute** OHLCV. Peer close must be aligned on the exact row index. The offline CLI shifts peer candles by one 15m bar. The Freqtrade adapter rejects peer data older than 15 minutes and will not enter without a usable peer feed.
2. Typed windows: five 32-candle series are assembled for each valid sample using **only candles at or before t**: asset returns, benchmark returns, relative returns, log-volume change and candle high-low range.
3. Upstream MLSD SData/SDataFrame with BasicSeries outputs 105 named numerical features (21 per stream); invalid or incomplete windows are discarded. No centered wavelets, future normalizers, fitted global statistics or label features are included.
4. Labels: three classes [-1,0,1] from the **future h-bar log return** beyond a small class-defining threshold. The threshold is **not** a full fee or slippage model.
5. Training: fit each fold's MLSD transform, StandardScaler and logistic classifier only on its past training rows. Upstream PurgedEventTimeSeriesSplit excludes training labels whose event end overlaps the validation start. The final trailing 20% holdout is not refitted.
6. Export: plain JSON model with ordered columns, coefficient/scale arrays, classes, per-asset binding, maturity watermark, and SHA256 dataset fingerprint; no pickle / executable deserialization.
7. Freqtrade: MLSD disabled by default. Explicit mode gates the existing technical long/short signals using per-asset model confidence. Bad/missing artifacts, model trained too late for an old backtest candle, benchmark outage or schema drift are fail-closed for **new entries**. Technical exits and broker stops remain active.

## Training commands

~~~bash
pip install -e '.[test,mlsd]'
python -m pytest
bnb-btc-mlsd-train --asset-pair BTC/USDT:USDT \
  --asset BTC_15m.csv --benchmark BNB_15m.csv \
  --model-output models/btc.json --report-output models/btc-report.json
bnb-btc-mlsd-train --asset-pair BNB/USDT:USDT \
  --asset BNB_15m.csv --benchmark BTC_15m.csv \
  --model-output models/bnb.json --report-output models/bnb-report.json
~~~

Keep both files private and validate provenance. Loading trusted JSON does not amount to cryptographic signature verification.

## Enabling Freqtrade gate (research dry-run only)

~~~bash
export BNB_BTC_MLSD_ENABLED=1
export BNB_BTC_MLSD_BTC_MODEL_PATH=/trusted/models/btc.json
export BNB_BTC_MLSD_BNB_MODEL_PATH=/trusted/models/bnb.json
# config.json remains dry_run=true and initial_state=stopped
~~~

Pair-to-model binding is enforced at runtime. If only one model path is configured, the other pair fails closed rather than borrowing the wrong model. Model path variables must be provided to the process environment; GHCR contains no trained weights.

## Validation evidence and missing safeguards

The Python CI runs synthetic tests for the upstream MLSD implementation contract, feature schema, temporal invariance, missing counterpart data, JSON checks, event purging, and model-mode entry failures. These do **not** verify funded execution, Freqtrade exchange backtests, calibration, latency, walk-forward trading returns or reliability of strategy risk stops on an exchange. Real historical data, fees, funding, slippage, kill switches, order reconciliation and external monitoring must be independently verified before any funded deployment. Existing V4 research performance numbers are historical claims, not reproducible results of this model.
