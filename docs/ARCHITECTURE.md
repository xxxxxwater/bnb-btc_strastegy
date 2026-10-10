# Architecture, assumptions and gaps

Input contract: fully closed 15-minute BTC and BNB OHLCV bars, synchronized UTC timestamps, strictly positive prices and volume. The current technical baseline does **not** consume true order-book depth, ARIMA forecasts or LSTM weights.

Closed-candle log returns -> lag-only Morlet-inspired sub-hour/1–6h/>6h convolutions -> bounded score mixed with EMA(12/48) -> RSI(14) -> trailing BTC/BNB spread deviation gate -> Freqtrade long/short entry/exit.

The online spread gate estimates rolling log-price hedge coefficients and uses the coefficient lagged one bar; the standardized residual is a causal *proxy*, not evidence of cointegration. A Johansen trace diagnostic (statsmodels extra) is available for offline research on matched complete history, with its own limitations.

Fail-closed rules for new entries: missing counterpart, invalid OHLCV, insufficient history, non-computable spread and unsupported pair. Exit signals do not require available counterpart data. Freqtrade sample uses isolated 1x and -5% stoploss, illustrative only; execution-level liquidation prevention, slippage limits, funding, reconciliation, owned-order handling, broker kill switch, failure recovery and real exchange tests are **not implemented**.

Benchmarks, walk-forward validation, transaction fees, funding and overfit controls are required before making any performance claims. Any timeframe other than 15m requires new calibration of wavelet periods.


## v0.4.0 MLSD Layer

A separate adapter builds five strictly trailing 32-bar Series in MLSD containers and extracts 105 deterministic BasicSeries statistics. Purged label-aware offline training creates pair-bound JSON models, while the Freqtrade gate restricts entries to eligible post-training bars and fails closed on missing, stale, incompatible or mismatched per-pair model inputs. For model schema and evaluation safety rules, read [MLSD_INTEGRATION.md](MLSD_INTEGRATION.md).


## v0.4.1 strict as-of contract

The shared market_data.align_previous_benchmark is now the **single source of truth** for online/offline pairing. For a candle with timestamp t, use only peer[t-15m] with exact UTC grid matching, not same-row informative merges or old ffilled values. The derived MLSD JSON schema v2 requires a one-bar lag marker, whole-source OHLCV/timestamp fingerprint, 105-feature schema, pair-specific model identity and valid training-label maturity. Incorrect or unavailable features fail closed for new entries; model eligibility is visible per candle.
