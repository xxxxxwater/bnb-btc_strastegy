# v0.4.0 — MLSD Feature Engineering Layer

- Integrated actual upstream Machine_Learning_For_Structured_Data v0.2.1 (source commit 85e356c1fb6703fef2b03ac0070c5e0659e8c671).
- Added 105 causal rolling Series statistics from MLSD SData / SDataFrame / BasicSeries (asset, peer, relative, volume, range).
- Added offline three-class training and mature-label PurgedEventTimeSeriesSplit validation, plus a trailing untouched time-based holdout.
- Added JSON-only classifier export with exact feature schema, MLSD source pin, asset-pair binding, training cutoff and dataset fingerprint.
- Added opt-in Freqtrade confidence gating with separate BTC/BNB model paths; no model or invalid feed/asset/schema => no entries.
- Added regression coverage for lookahead invariance, missing peer data, contract failure, and train/inference alignment.
- Expanded Python CI and GHCR image to install MLSD from the pinned upstream source.
- Preserved previous v0.3.0 algorithm and historical research descriptions; no live production deployment.

Research only: No proven Sharpe, PnL or realistic fee/funding/slippage evaluation. Train and validate with actual historical BTC/BNB data before using capital. Previously published repository secrets must still be rotated.
