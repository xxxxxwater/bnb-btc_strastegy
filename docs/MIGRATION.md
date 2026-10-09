# Migration from broken legacy snippets

1. Install this repository as a package. The old OptimizedCTA_v2_Strategy.py had no loadable IStrategy and now delegates to bnb_btc_strategy.freqtrade_strategy, exporting OptimizedCTAStrategy.
2. The original OptimizedETHStrategy.py was a syntactically invalid class-body fragment containing fractional IntParameter defaults. It is now a safe compatibility class: BTC/BNB allowed; ETH **not enabled**.
3. The original config copy.json exposed fixed authentication tokens, a network API listener and an inappropriate six-pair configuration. Use config.example.json with private local credentials; never reuse the old values. Credentials remain visible in Git history: **revoke them**.
4. LSTM and ARIMA inference were never executable; new signal scores do not replicate private research/V4 forecasts.
5. The old code checked BTC/ETH while README described BTC/BNB. New execution gates depend on BTC/BNB rolling spread proxy only; offline Johansen is separate.

Backtest afresh with aligned 15m BTC and BNB candles; preserve dry-run and stopped settings, validate exchange data scheduling and require external order/risk reconciliation before funded deployment.
