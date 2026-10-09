"""Causal signal engine based on closed OHLCV candles.

No learned forecasts, PnL claims or exchange credentials are embedded here.
Every output at t depends only on rows <= t. Signals are research hypotheses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    rsi_period: int = 14
    fast_ema: int = 12
    slow_ema: int = 48
    min_history: int = 100
    spread_window: int = 64
    spread_z_limit: float = 2.0
    score_threshold: float = 0.16
    require_benchmark: bool = True

    def __post_init__(self) -> None:
        if min(self.rsi_period, self.fast_ema, self.slow_ema,
               self.min_history, self.spread_window) < 2:
            raise ValueError("periods and history must be >= 2")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be shorter than slow_ema")
        if self.min_history < max(self.slow_ema, self.spread_window):
            raise ValueError("min_history must cover slow EMA and spread window")
        if not (0.0 < self.score_threshold < 1.0):
            raise ValueError("score_threshold must be in (0, 1)")
        if not (0.0 < self.spread_z_limit <= 10.0):
            raise ValueError("spread_z_limit must be in (0, 10]")


def _causal_wavelet(returns: np.ndarray, period: int) -> np.ndarray:
    """One-sided, zero-mean Morlet-like filter (not a centered CWT).

    Normalizing each filter makes the feature scale stable across periods.
    The first value at time t uses the current *closed* candle and its past.
    """
    lag = np.arange(0, period * 4, dtype=float)
    kernel = np.exp(-0.5 * (lag / period) ** 2) * np.cos(2 * np.pi * lag / period)
    kernel -= np.mean(kernel)
    kernel /= max(np.sum(np.abs(kernel)), 1e-12)
    return np.convolve(returns, kernel, mode="full")[: len(returns)]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False,
                                   min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False,
                                     min_periods=period).mean()
    relative = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative)
    result = result.mask((loss == 0) & (gain > 0), 100)
    result = result.mask((gain == 0) & (loss > 0), 0)
    return result.mask((gain == 0) & (loss == 0), 50)


def calculate_features(
    frame: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Return a fresh DataFrame with scores, entry/exit flags and diagnostics.

    Benchmark is BTC for BNB signals, or BNB for BTC signals. The spread
    stability gate is explicitly a rolling proxy, NOT proof of cointegration.
    Callers must align benchmark on fully closed candles, not future values.
    """
    cfg = config or SignalConfig()
    missing = {"close", "high", "low", "volume"} - set(frame.columns)
    if missing:
        raise ValueError("Missing OHLCV columns: " + ", ".join(sorted(missing)))
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    valid = (close > 0) & np.isfinite(close) & (volume > 0) & np.isfinite(volume)
    valid &= (high >= low) & (high >= close) & (low <= close)
    safe_close = close.where(close > 0)
    log_returns = np.log(safe_close).diff().replace([np.inf, -np.inf], np.nan)
    array = log_returns.fillna(0).to_numpy(dtype=float)
    out["rsi"] = _rsi(safe_close, cfg.rsi_period)
    ema_fast = safe_close.ewm(span=cfg.fast_ema, adjust=False,
                              min_periods=cfg.fast_ema).mean()
    ema_slow = safe_close.ewm(span=cfg.slow_ema, adjust=False,
                              min_periods=cfg.slow_ema).mean()
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow

    # For 15-minute candles: periods 2, 4, 8, 16 and 32 approximate
    # sub-hour, 1-6 hour and >6 hour price movement components.
    out["wavelet_high"] = _causal_wavelet(array, 2)
    out["wavelet_mid"] = (
        _causal_wavelet(array, 4) + _causal_wavelet(array, 8)
        + _causal_wavelet(array, 16)
    ) / 3
    out["wavelet_low"] = _causal_wavelet(array, 32)
    wavelet = (0.25 * out["wavelet_high"] + 0.5 * out["wavelet_mid"]
               + 0.25 * out["wavelet_low"])
    trend = (ema_fast / ema_slow - 1).replace([np.inf, -np.inf], np.nan)
    out["signal_score"] = (
        0.65 * np.tanh(120 * wavelet)
        + 0.35 * np.tanh(25 * trend)
    ).clip(-1, 1)

    benchmark_valid = pd.Series(False, index=out.index)
    out["spread_z"] = np.nan
    if benchmark_close is not None:
        benchmark = pd.to_numeric(
            pd.Series(benchmark_close).reindex(out.index), errors="coerce"
        )
        benchmark_valid = (benchmark > 0) & np.isfinite(benchmark)
        log_close = np.log(safe_close)
        log_benchmark = np.log(benchmark.where(benchmark > 0))
        rolling_cov = log_close.rolling(cfg.spread_window).cov(log_benchmark)
        rolling_var = log_benchmark.rolling(cfg.spread_window).var()
        hedge = (rolling_cov / rolling_var.where(rolling_var > 1e-12)).clip(-10, 10)
        # Use the PRIOR hedge estimate, avoiding lookahead during rolling fit.
        residual = log_close - hedge.shift(1) * log_benchmark
        center = residual.rolling(cfg.spread_window).mean()
        dispersion = residual.rolling(cfg.spread_window).std()
        out["spread_z"] = (residual - center) / dispersion.where(dispersion > 1e-12)

    if cfg.require_benchmark:
        spread_ok = benchmark_valid & out["spread_z"].abs().le(cfg.spread_z_limit)
    else:
        spread_ok = pd.Series(True, index=out.index)

    enough = pd.Series(np.arange(len(out)) >= cfg.min_history, index=out.index)
    ready = valid & enough & spread_ok & out["rsi"].notna() & out["signal_score"].notna()
    out["signal_ready"] = ready.astype(bool)
    out["enter_long"] = (
        ready & out["signal_score"].ge(cfg.score_threshold)
        & (out["ema_fast"] > out["ema_slow"])
        & out["rsi"].between(45, 70)
    ).astype(int)
    out["enter_short"] = (
        ready & out["signal_score"].le(-cfg.score_threshold)
        & (out["ema_fast"] < out["ema_slow"])
        & out["rsi"].between(30, 55)
    ).astype(int)
    # Exit signals operate independently of readiness; invalid candles are
    # never accepted as entries. Risk stop-loss is handled by the executor.
    out["exit_long"] = (
        out["signal_score"].lt(0) | out["rsi"].gt(78)
    ).fillna(False).astype(int)
    out["exit_short"] = (
        out["signal_score"].gt(0) | out["rsi"].lt(22)
    ).fillna(False).astype(int)
    return out
