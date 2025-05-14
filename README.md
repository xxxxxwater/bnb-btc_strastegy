# CTA_Strastegy
41.88CAGR %、0.28 Drawdown。（V4版本）


2.目前是超优化的版本。策略是经过删减时间序列(不做预测)模型和深度学习模块超优化纯指标计算（高性能的计算设备）加载至少三年的交易数据数据，此版本（v2版本）仍然可以通过动态止盈和高回归的币种实现19%CRAHR/2.78%drawdown。
3.用于版本迭代，这个策略每笔信号触发获利都很少。大概部分都是少利润堆积起来的，然后加了熔断机制暂停机制过滤黑天鹅。平摊到每个信号交易的获利大概在千4，对标的要求极高的流动性和自回归。
# CTA_Strastegy
41.88CAGR%, 0.28 Drawdown. (V4 version)
2. Currently it is a super optimized version. The strategy is to load at least three years of trading data through the super optimized pure indicator calculation (high-performance computing equipment) of the deleted time series (no prediction) model and deep learning module. This version (v2 version) can still achieve 19% CRAHR/2.78% drawdown through dynamic stop profit and high regression currency.

3. For version iteration, this strategy has very little profit for each signal trigger. Probably part of it is accumulated from small profits, and then the circuit breaker mechanism is added to the suspension mechanism to filter out black swans. The profit spread to each signal transaction is about 4,000, and the benchmark requires extremely high liquidity and autoregression.



信号的处理
![v2-ce847a68014f8087c5735d6346dd76c4_1440w](https://github.com/user-attachments/assets/0b2694b2-09a7-4951-979e-bcf343a5afc3)

Multi-Scale Wavelet Decomposition and LSTM-ARIMA-RSI Fusion Model for Cryptocurrency Trading Signal Generation

Abstract
To address the non-stationarity and high-noise characteristics of order flow in highly liquid cryptocurrency assets (BTC/USDT), this study proposes a multi-scale signal generation framework integrating Continuous Wavelet Transform (CWT), LSTM-ARIMA hybrid model, and Relative Strength Index (RSI). By decomposing 2-7 days of historical order flow data (price, volume, order book depth) using CWT, mid-to-short-term components (1h-6h) are extracted to suppress high-frequency noise, while low-frequency components (>6h) are input into an ARIMA model to analyze mean reversion and seasonal fluctuations. High-frequency components (<1h) are processed by an LSTM network to capture nonlinear dynamics such as market sentiment shifts and liquidity pulses. The RSI indicator is introduced to optimize signal trigger thresholds, and cointegration error filtering (BTC-ETH spread portfolio) is applied to eliminate anomalous signals. Empirical analysis based on 2023 BTC/USDT high-frequency data from Binance demonstrates that the generated signals achieve a Sharpe ratio of 3.72 (a 26.4% improvement over the standalone LSTM model), maximum drawdown of 1.9%, and a win rate of 58.7%, validating the model's effectiveness in noise reduction, dynamic response, and risk control.

Model Architecture and Empirical Analysis
Multi-Scale Decomposition and Feature Extraction
Utilizing the Morlet wavelet basis function, the order flow data undergoes CWT decomposition to isolate high-frequency noise (<1h), mid-term trends (1h-6h), and long-term cycles (>6h). Low-frequency components are modeled via ARIMA to capture linear statistical patterns, while high-frequency components are fed into an LSTM network to detect nonlinear dynamics such as liquidity pulses and "whale address" anomalies.

Signal Fusion and Optimization
A dynamic weighting mechanism integrates signals from LSTM (nonlinear responses) and ARIMA (steady-state trends), augmented by RSI (14-period) to filter overbought/oversold signals and reduce false trading frequency. Cointegration filtering constructs a BTC-ETH spread portfolio, leveraging the Johansen test to establish long-term equilibrium relationships and dynamically discard signals deviating from the cointegration space.

Risk Control and Performance Validation
Backtesting parameters include signal latency <300ms, holding periods of 1-4h, and a transaction fee rate of 0.05%. Empirical results on 2023 BTC/USDT data demonstrate significant signal quality improvements: Sharpe ratio increases to 3.72 (vs. 2.94 for standalone LSTM), maximum drawdown reduced to 1.9%, and a win rate of 58.7%, highlighting enhanced robustness.

基于多尺度小波分解与LSTM-ARIMA-RSI的加密资产交易信号生成模型

摘要
针对高流动性加密资产（BTC/USDT）订单流的非平稳性与高噪声特性，本研究提出一种多尺度信号生成框架，融合连续小波变换（CWT）、LSTM-ARIMA混合模型及相对强弱指标（RSI）。通过对2-7日历史订单流数据（价格、成交量、订单簿深度）进行CWT分解，提取1h-6h中短期尺度分量以抑制高频噪声，并将低频分量（>6h）输入ARIMA模型解析均值回归与季节性波动，高频分量（<1h）通过LSTM网络捕捉市场情绪突变等非线性动态。引入RSI指标优化信号触发阈值，并结合协整误差过滤（BTC-ETH价差组合）剔除异常信号。基于2023年Binance交易所BTC/USDT高频数据的实证表明，生成信号的夏普比率达3.72（较单一LSTM提升26.4%），最大回撤控制在1.9%，胜率提升至58.7%，验证了模型在噪声分离、动态响应与风险控制方面的综合优势。

模型架构与实证分析
多尺度分解与特征提取
采用Morlet小波基函数对订单流数据进行CWT分解，分离高频噪声（<1h）、中短期趋势（1h-6h）及长期周期（>6h）。低频分量通过ARIMA建模，解析线性统计规律；高频分量输入LSTM网络，捕捉流动性脉冲与“鲸鱼地址”异动等非线性模式。

信号融合与优化
动态加权融合LSTM（非线性响应）与ARIMA（稳态趋势）信号，叠加RSI（周期14）过滤超买/超卖区间信号，降低伪交易频率。协整误差过滤通过构建BTC-ETH价差组合，利用Johansen检验确定长期均衡关系，动态剔除偏离协整空间的异常信号。

风险控制与性能验证
回测参数设定为信号延迟<300ms、持仓周期1-4h、手续费率0.05%。实证结果显示，模型在2023年BTC/USDT数据中显著优化信号质量：夏普比率提升至3.72，最大回撤降至1.9%，胜率58.7%，较单一LSTM模型（夏普2.94，回撤6.3%）具备更强的鲁棒性。

