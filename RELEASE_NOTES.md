# v0.4.1 — MLSD candle alignment and model trust hardening

**Bug fixed:** The previous Freqtrade same-timeframe informative merge did not enforce the offline trainer's one-completed-benchmark-bar delay. Live/backtest and offline inference now use one exact timestamp as-of adapter: peer[t−15m] only. Missing or stale peer candles are rejected rather than forward-filled.

**Included:**
- Shared alignment library with UTC 15m grid, unique/ordered timestamps and contiguous asset bar enforcement.
- Tests for future-peer mutation invariance, missing peer gaps, malformed timestamps and offline/runtime parity.
- Versioned **MLSD model schema v2**: pair binding, explicit benchmark lag, exact 105-feature schema, maturity watermark, stricter scaler validation and full OHLCV/peer/timestamp SHA256 fingerprint.
- JSON duplicate-key defense and validation of nonfinite inference inputs.
- MLSD gate status now reflects per-candle probability eligibility.
- CI adds Ruff correctness checks. GHCR Docker build actually imports MLSD **and the installed Freqtrade adapter**.
- Updated README, security guidance and architecture docs.

**Breaking safety change:** v0.4.0 (schema v1) JSON model files must be **retrained using v0.4.1**. No trained model or exchange credential is included in published images. All trading remains disabled by default in supplied configuration.

No real market PnL or latency benchmark was performed, and no production strategy/server was changed.
