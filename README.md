# BTC/BNB Multi-Scale Signal Research

**Causal wavelet-inspired features · RSI · BTC/BNB spread filter · Freqtrade V3**

[CI](https://github.com/xxxxxwater/bnb-btc_strastegy/actions) · [Releases](https://github.com/xxxxxwater/bnb-btc_strastegy/releases) · [GHCR Packages](https://github.com/users/xxxxxwater/packages?repo_name=bnb-btc_strastegy) · [Architecture](docs/ARCHITECTURE.md) · [Migration](docs/MIGRATION.md)

> Research-only, not financial advice. This repository is **not** the private V4 strategy, and no historical return metrics are established for this rebuilt engine.


## v0.4.1 — Time-alignment and model-provenance hardening

This release corrects a **material** data-integrity risk in v0.4.0: the Freqtrade same-timeframe informative merge could use a different benchmark candle from the offline trainer. Training, CSV export, and execution now share one strict timestamp adapter: at asset candle opening time **t**, they use **only** the benchmark candle opening at **t−15 minutes**.

- **No future candle / stale forward fill:** a missing, off-grid, duplicate or old peer candle cannot silently become the newest price. Asset time-grid gaps disable entries.
- **Auditable MLSD model schema v2:** model JSON requires exactly 105 ordered BasicSeries features, an explicit one-bar benchmark lag, pair identity, label maturity and a complete timestamp/OHLCV/peer dataset fingerprint.
- **Security:** duplicate JSON object keys are rejected, coefficients and scaler must be finite, and model exports never deserialize pickle.
- **Observability:** MLSD eligibility is calculated **per candle**, rather than displaying a misleading always-true gate.
- **Build correctness:** Python 3.10–3.12 tests, Ruff correctness lint, and a Docker build-time import smoke test of the **real Freqtrade V3 adapter**.

**Compatibility:** Retrain model files created with v0.4.0; the v0.4.1 schema intentionally rejects old artifacts rather than silently assuming they were trained under the new timestamp contract. Both releases remain available. Nothing in this release establishes economic profitability or authorizes real-money trading.

## v0.4.0 — MLSD structured-data feature engineering (new)

The [Machine Learning for Structured Data](https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data) **v0.2.1** implementation is now integrated as a pinned, real upstream runtime dependency. The deterministic v0.3.0 wavelet-inspired + RSI + BTC/BNB spread baseline is retained, and MLSD adds a strictly opt-in machine-learning confidence gate. This is not the unpublished private V4 model.

| Layer | v0.4.0 code |
| --- | --- |
| Raw inputs | Unique, chronological, contiguous **closed** BTC/BNB 15m OHLCV bars and exact peer index alignment |
| Typed MLSD | Five 32-bar Series: asset/peer log returns, relative returns, volume changes and candle ranges |
| Statistics | MLSD SDataFrame / BasicSeries -> **105** prefixed scalar features (5 x 21) |
| Train/evaluation | Offline 3-class logistic model, PurgedEventTimeSeriesSplit and independent trailing 20% holdout |
| Model file | Auditable JSON, no Python pickle, SHA-pinned MLSD dependency, schema + feature ordering + trained-label maturity |
| Execution gate | Optional, per-pair local artifact, model + original technical signal required; **fail closed** on missing or mismatched inputs |

### Offline training / validation

~~~bash
python -m pip install -e '.[test,mlsd]'
python -m pytest

bnb-btc-mlsd-train \
  --asset-pair BTC/USDT:USDT --asset BTC_15m.csv --benchmark BNB_15m.csv \
  --window 32 --horizon 4 \
  --model-output models/btc.json --report-output models/btc-report.json

bnb-btc-mlsd-train \
  --asset-pair BNB/USDT:USDT --asset BNB_15m.csv --benchmark BTC_15m.csv \
  --window 32 --horizon 4 \
  --model-output models/bnb.json --report-output models/bnb-report.json
~~~

Both CSVs contain date,open,high,low,close,volume at exactly 15m UTC-aligned intervals. No exchange connection or trading is performed by the training CLI. Review classification OOS metrics separately from economic performance; there are **no verified profitability or Sharpe results** for this implementation.

### Optional Freqtrade gate (dry-run only)

~~~bash
export BNB_BTC_MLSD_ENABLED=1
export BNB_BTC_MLSD_BTC_MODEL_PATH=/trusted/models/btc.json
export BNB_BTC_MLSD_BNB_MODEL_PATH=/trusted/models/bnb.json
# config.json must retain dry_run=true, initial_state=stopped
~~~

If MLSD is disabled, the original v0.3.0 technical signal still runs. If enabled but a model, peer feed or training schema is missing/mismatched, **all new entries for that pair are blocked**, while exit indicators remain available. A trained model cannot be used for pre-training in-sample bars, because the inference gate enforces a cutoff later than its latest training label maturity.

See [MLSD integration guide](docs/MLSD_INTEGRATION.md). The default released container does **not** contain training models or exchange credentials, and downloading it does not enable trading.

---

## What's actually implemented in v0.3.0?

The original open-source file referenced undefined 'lstm_pred' and 'arima_pred' variables and an undefined 'dataframe', so it could not run. The old ETH strategy was only an invalid indented fragment; the configured Freqtrade class was missing. The v0.3.0 overhaul implements a **different, functional, causal, non-forecast baseline**, not a reproduction of an unreleased learned model.

| Layer | Original description | v0.3.0 shipped code |
| --- | --- | --- |
| Wavelets | Multi-Scale Morlet CWT, 1h/6h bands | One-sided Morlet-inspired filters at 2/4/8/16/32 15m bars (not centered CWT) |
| LSTM / ARIMA | Prediction and dynamic fusion | **Not implemented**; no dummy forecasts |
| RSI | RSI 14 | Wilder RSI and directional gates |
| Cointegration | BTC/BNB Johansen discussed; old code actually BTC/ETH | BTC/BNB rolling relative log-spread **proxy** online; optional offline Johansen diagnostic |
| Execution | Broken demo | Freqtrade V3 long/short adapter, 1x leverage, stops, pair allowlist |
| Safety | Hardcoded API auth and permissive defaults | Dry-run, initial_state=stopped, disabled API/Telegram, secrets removed |
| Validation | Unsupported performance statements | Unit tests, causal invariance and CLI checks; no PnL claims |

## Strategy flow

1. Load **closed** BTC and BNB 15-minute OHLCV candles (no order-book or depth feed).
2. Build logarithmic returns and **causal** wavelet-inspired components for sub-hour, 1–6h and >6h activity. Add 12/48 EMA and RSI(14).
3. Fuse bounded technical scores, filter by RSI, trend and a rolling BTC/BNB log-spread z-score.
4. Require valid positive volume, warmup and counterpart candles; **fail closed for new entries** if data is unavailable.
5. Output enter_long, enter_short, exit_long, exit_short. Execution is delegated to Freqtrade; no trained model predictions are generated.

**Online spread z-score is not a Johansen cointegration test.** The optional offline diagnostic is explicitly separate.

## Quick start: offline research

Python 3.10 or newer:

~~~bash
python -m pip install -e '.[test]'
python -m pytest
bnb-btc-signals --asset BTC_15m.csv --benchmark BNB_15m.csv --output signals.csv
~~~

Both CSVs require: date, open, high, low, close, volume; ascending unique timestamps. The CLI shifts the benchmark one completed candle and writes features/signals only. It does **not** calculate Sharpe, drawdown, trading profit or execution results.

## Freqtrade integration

Install compatible Freqtrade separately (or pull GHCR image once the publisher confirms it). Install this package with pip. Example commands:

~~~bash
cp config.example.json config.json
freqtrade list-strategies --strategy-path . --recursive-strategy-search
freqtrade download-data --exchange binance --trading-mode futures --timeframe 15m --pairs BTC/USDT:USDT BNB/USDT:USDT
freqtrade backtesting --config config.json --strategy OptimizedCTAStrategy --strategy-path . --timerange 20240101-20240701
~~~

Defaults in config.example.json are dry_run=true, initial_state=stopped, stake=100 USDT, futures isolated and API/Telegram disabled. **Do not run funded orders with this experimental strategy.** The ETH compatibility file explicitly cannot place ETH entries. The public Dockerfile bundles Freqtrade and the strategy but no credentials.

## Original research direction (retained for historical context)

The previous README described **Multi-Scale Wavelet Decomposition and LSTM-ARIMA-RSI Fusion Model for Cryptocurrency Trading Signal Generation**. It proposed Morlet wavelets, decomposition of 2–7 days of price/volume/order-book depth, ARIMA modeling of low-frequency components, LSTM modeling of high-frequency components, RSI threshold fusion and BTC–BNB Johansen cointegration. It referred to an empirical study of 2023 BTC/USDT Binance data. That proposal is **not the current shipped architecture**; no trained LSTM weights, order-flow dataset, backtest trade list or original private V4 source are in this repository.

Historical statements retained from earlier README — **not independently verified and not v0.3.0 results**:

- V4 CAGR 41.88% and "0.28 Drawdown" (original wording ambiguous).
- V2 annual return approximately 19% and drawdown 2.78%.
- Sharpe 3.72 vs standalone LSTM 2.94, maximum drawdown 1.9%, win rate 58.7%, signal latency under 300 ms, 0.05% fees.
- Another V2 claim of Sharpe 3.52 and drawdown under 2%.

The original paper PDF remains in the repo as a research reference. The old README's illustrations are preserved:

![Historical signal processing](https://github.com/user-attachments/assets/0b2694b2-09a7-4951-979e-bcf343a5afc3)

<img width="574" alt="Original research illustration" src="https://github.com/user-attachments/assets/82a368ae-d1e5-4615-8c76-2c7d2499c0e5" />

![Original research parameters](https://github.com/user-attachments/assets/1db09980-3d97-48d7-856a-a05117168259)

## Files and security

- bnb_btc_strategy/core.py: causal feature and signal engine.
- bnb_btc_strategy/research.py: optional offline Johansen diagnostic.
- bnb_btc_strategy/cli.py: CSV exporter.
- bnb_btc_strategy/freqtrade_strategy.py: Freqtrade adapter.
- OptimizedCTA_v2_Strategy.py: working loader compatibility name.
- OptimizedETHStrategy.py: intentionally no ETH trading.
- config.example.json: safer test-only configuration.
- tests/: regression checks.
- docs/ARCHITECTURE.md, docs/MIGRATION.md, docs/RELEASE.md and SECURITY.md: documentation.

**Rotate/revoke any historical published JWT/WS tokens or API keys.** Cleaning the latest config does not delete Git history. Read [SECURITY.md](SECURITY.md) and [release policy](docs/RELEASE.md).
