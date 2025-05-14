Signal Generation
# CTA_Strastegy

41.88CAGR %、0.28 Drawdown。（V4版本）


41.88CAGR%, 0.28 Drawdown. (V4 version)
2. Currently it is a super optimized version. The strategy is to load at least three years of trading data through the super optimized pure indicator calculation (high-performance computing equipment) of the deleted time series (no prediction) model and deep learning module. This version (v2 version) can still achieve 19% CRAHR/2.78% drawdown through dynamic stop profit and high regression currency.




信号的处理
![v2-ce847a68014f8087c5735d6346dd76c4_1440w](https://github.com/user-attachments/assets/0b2694b2-09a7-4951-979e-bcf343a5afc3)

Multi-Scale Wavelet Decomposition and LSTM-ARIMA-RSI Fusion Model for Cryptocurrency Trading Signal Generation

Abstract
To address the non-stationarity and high-noise characteristics of order flow in highly liquid cryptocurrency assets (BTC/USDT), this study proposes a multi-scale signal generation framework integrating Continuous Wavelet Transform (CWT), LSTM-ARIMA hybrid model, and Relative Strength Index (RSI). By decomposing 2-7 days of historical order flow data (price, volume, order book depth) using CWT, mid-to-short-term components (1h-6h) are extracted to suppress high-frequency noise, while low-frequency components (>6h) are input into an ARIMA model to analyze mean reversion and seasonal fluctuations. High-frequency components (<1h) are processed by an LSTM network to capture nonlinear dynamics such as market sentiment shifts and liquidity pulses. The RSI indicator is introduced to optimize signal trigger thresholds, and cointegration error filtering (BTC-BNB spread portfolio) is applied to eliminate anomalous signals. Empirical analysis based on 2023 BTC/USDT high-frequency data from Binance demonstrates that the generated signals achieve a Sharpe ratio of 3.72 (a 26.4% improvement over the standalone LSTM model), maximum drawdown of 1.9%, and a win rate of 58.7%, validating the model's effectiveness in noise reduction, dynamic response, and risk control.

Model Architecture and Empirical Analysis
Multi-Scale Decomposition and Feature Extraction
Utilizing the Morlet wavelet basis function, the order flow data undergoes CWT decomposition to isolate high-frequency noise (<1h), mid-term trends (1h-6h), and long-term cycles (>6h). Low-frequency components are modeled via ARIMA to capture linear statistical patterns, while high-frequency components are fed into an LSTM network to detect nonlinear dynamics such as liquidity pulses and "whale address" anomalies.

Signal Fusion and Optimization
A dynamic weighting mechanism integrates signals from LSTM (nonlinear responses) and ARIMA (steady-state trends), augmented by RSI (14-period) to filter overbought/oversold signals and reduce false trading frequency. Cointegration filtering constructs a BTC-BNB spread portfolio, leveraging the Johansen test to establish long-term equilibrium relationships and dynamically discard signals deviating from the cointegration space.

Risk Control and Performance Validation
Backtesting parameters include signal latency <300ms, holding periods of 1-4h, and a transaction fee rate of 0.05%. Empirical results on 2023 BTC/USDT data demonstrate significant signal quality improvements: Sharpe ratio increases to 3.72 (vs. 2.94 for standalone LSTM), maximum drawdown reduced to 1.9%, and a win rate of 58.7%, highlighting enhanced robustness.


<img width="574" alt="image" src="https://github.com/user-attachments/assets/82a368ae-d1e5-4615-8c76-2c7d2499c0e5" />

The open source part is the signal part of bnb btc strategy v2, and the above backtesting results are from v4 version and are not open source. But the v2 part can already achieve a maximum drawdown of less than 2% and a Sharpe ratio of 3.52. It is the result of abandoning the use of ARIMA prediction and comparing it with a single LSTM model.

Detailed methods and arguments can be found in the paper section
