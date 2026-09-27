"""Quantitative strategy module for Sentinel-Trader.

Provides 100% deterministic technical indicator calculation and entry signal evaluation:
- RSI(14) with Wilder's exponential smoothing.
- EMA(200) trend filter.
- Entry signal: Close > EMA(200) AND RSI(14) crosses upward through 35.
"""

import asyncio
from typing import Any, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("sentinel.strategy")


class Candidate(BaseModel):
    """Candidate trading opportunity identified during universe screening."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    close: float
    ema: float
    rsi: float
    prev_rsi: float
    rsi_bounce: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0


class IndicatorResult(tuple):
    """Container returning RSI and EMA, supporting tuple unpacking, attributes, and column access."""

    def __new__(cls, rsi: pd.Series, ema: pd.Series, df: Optional[pd.DataFrame] = None):
        return super().__new__(cls, (rsi, ema))

    def __init__(self, rsi: pd.Series, ema: pd.Series, df: Optional[pd.DataFrame] = None):
        self.rsi = rsi
        self.ema = ema
        self.df = df

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item.lower() in ("rsi", "rsi14", "rsi_14"):
                return self.rsi
            if item.lower() in ("ema", "ema200", "ema_200"):
                return self.ema
            if self.df is not None and item in self.df.columns:
                return self.df[item]
            raise KeyError(f"Key '{item}' not found in IndicatorResult or underlying DataFrame")
        return super().__getitem__(item)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's exponential smoothing.

    Args:
        series: Price series (typically close price).
        period: Smoothing lookback period (default: 14).

    Returns:
        pd.Series: Computed RSI values oscillating between 0.0 and 100.0.
    """
    if len(series) < 2:
        return pd.Series(index=series.index, data=np.nan, dtype="float64")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's Smoothing: ewm with alpha = 1 / period (span = 2 * period - 1)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Edge cases:
    # If loss == 0 and gain > 0 -> RSI = 100.0
    # If gain == 0 and loss == 0 -> RSI = 50.0
    # If gain == 0 and loss > 0 -> RSI = 0.0
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where((avg_gain != 0) | (avg_loss != 0), 50.0)

    return rsi


def calculate_ema(series: pd.Series, span: int = 200, min_periods: int = 1) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        series: Price series (typically close price).
        span: Smoothing window span (default: 200).
        min_periods: Minimum observations in window required (default: 1).

    Returns:
        pd.Series: Computed EMA values.
    """
    if series.empty:
        return pd.Series(index=series.index, data=[], dtype="float64")

    return series.ewm(span=span, adjust=False, min_periods=min_periods).mean()


def calculate_indicators(df: pd.DataFrame) -> IndicatorResult:
    """Calculate RSI(14) and EMA(200) for a price dataframe.

    Adds 'rsi' and 'ema' columns to the dataframe in-place and returns an IndicatorResult
    which can be unpacked as (rsi, ema) or accessed via attributes/keys.

    Args:
        df: Pandas DataFrame containing at least a 'close' (or 'Close') column.

    Returns:
        IndicatorResult: Unpackable as (rsi, ema), with .rsi and .ema attributes.
    """
    if df.empty:
        empty_series = pd.Series(dtype="float64")
        return IndicatorResult(empty_series, empty_series, df=df)

    # Resolve close column case-insensitively
    close_col = None
    for candidate in ("close", "Close", "CLOSE"):
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        raise KeyError(
            f"DataFrame must contain a 'close' column. Found columns: {list(df.columns)}"
        )

    close_series = pd.to_numeric(df[close_col], errors="coerce")

    # Compute indicators
    rsi_series = calculate_rsi(close_series, period=14)
    ema_series = calculate_ema(close_series, span=200, min_periods=1)

    # Attach to dataframe in-place
    df["rsi"] = rsi_series
    df["ema"] = ema_series
    df["rsi14"] = rsi_series
    df["ema200"] = ema_series

    return IndicatorResult(rsi_series, ema_series, df=df)


def check_entry_signal(df: pd.DataFrame) -> bool:
    """Check if the latest completed candles satisfy the active entry condition.

    Evaluates indicators strictly on finalized/completed bars to prevent mid-candle
    repainting caused by temporary intraday wicks or fluctuating forming candles.
    The active forming candle (df.iloc[-1]) is ignored.
    Crossover logic is based strictly on the last finalized candle (df.iloc[-2])
    and the preceding finalized candle (df.iloc[-3]).

    Condition:
        1. Last Finalized Close (df.iloc[-2]) > Finalized EMA(200) (df.iloc[-2]) (trend confirmation filter)
        2. RSI(14) crosses upward through 35:
           - Preceding finalized candle (df.iloc[-3]) RSI <= 35.0
           - Last finalized candle (df.iloc[-2]) RSI > 35.0

    Args:
        df: Pandas DataFrame with OHLCV data. If 'rsi' or 'ema' columns are not present,
            they will be calculated automatically.

    Returns:
        bool: True if entry conditions are strictly met on completed candles; False otherwise.
    """
    if df is None or len(df) < 3:
        return False

    # Ensure indicators exist
    if "rsi" not in df.columns or "ema" not in df.columns:
        calculate_indicators(df)

    # Resolve close column
    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is None:
        return False

    # Closed-candle evaluation:
    # Ignore/drop active forming candle at df.iloc[-1] to eliminate mid-candle repainting.
    # Evaluate strictly on last finalized bar (df.iloc[-2]) and prior bar (df.iloc[-3]).
    prev_rsi = df["rsi"].iloc[-3]
    curr_rsi = df["rsi"].iloc[-2]
    curr_close = df[close_col].iloc[-2]
    curr_ema = df["ema"].iloc[-2]

    # Guard against NaN / incomplete indicator warmups
    if (
        pd.isna(prev_rsi)
        or pd.isna(curr_rsi)
        or pd.isna(curr_close)
        or pd.isna(curr_ema)
    ):
        return False

    # Evaluation strictly on finalized bar
    rsi_crossed_up = (float(prev_rsi) <= 35.0) and (float(curr_rsi) > 35.0)
    above_trend = float(curr_close) > float(curr_ema)

    signal = bool(rsi_crossed_up and above_trend)

    if signal:
        logger.info(
            "[ENTRY SIGNAL TRIGGERED] Closed Candle Close=$%.4f > EMA(200)=$%.4f | RSI(14) crossed up: %.2f -> %.2f (Threshold 35)",
            curr_close,
            curr_ema,
            prev_rsi,
            curr_rsi,
        )

    return signal


async def scan_universe_for_signals(
    exchange: Any,
    universe: List[str],
    timeframe: str = "15m",
) -> List[Candidate]:
    """Scan screened universe pairs concurrently using asyncio.gather.

    Checks entry conditions strictly on completed candle (iloc[-2]):
    - Close > EMA(200) (trend filter)
    - RSI(14) upward crossover through 35 (iloc[-3] <= 35.0 and iloc[-2] > 35.0)

    Returns:
        List[Candidate]: Matching candidate pairs sorted descending by RSI bounce strength or 24h volume.
    """
    if not universe:
        return []

    async def _evaluate_symbol(symbol: str) -> Optional[Candidate]:
        try:
            if hasattr(exchange, "fetch_ohlcv_dataframe"):
                df = await exchange.fetch_ohlcv_dataframe(symbol, timeframe=timeframe, limit=250)
            elif hasattr(exchange, "fetch_ohlcv"):
                raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                return None

            if df is None or len(df) < 3:
                return None

            close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
            if close_col is None:
                return None

            if "rsi" not in df.columns or "ema" not in df.columns:
                calculate_indicators(df)

            prev_rsi = df["rsi"].iloc[-3]
            curr_rsi = df["rsi"].iloc[-2]
            curr_close = df[close_col].iloc[-2]
            curr_ema = df["ema"].iloc[-2]

            if (
                pd.isna(prev_rsi)
                or pd.isna(curr_rsi)
                or pd.isna(curr_close)
                or pd.isna(curr_ema)
            ):
                return None

            rsi_crossed_up = (float(prev_rsi) <= 35.0) and (float(curr_rsi) > 35.0)
            above_trend = float(curr_close) > float(curr_ema)

            if rsi_crossed_up and above_trend:
                vol = (
                    float(df["volume"].iloc[-2])
                    if "volume" in df.columns and not pd.isna(df["volume"].iloc[-2])
                    else 0.0
                )
                bounce = float(curr_rsi) - float(prev_rsi)
                candidate = Candidate(
                    symbol=symbol,
                    close=float(curr_close),
                    ema=float(curr_ema),
                    rsi=float(curr_rsi),
                    prev_rsi=float(prev_rsi),
                    rsi_bounce=round(bounce, 4),
                    volume=vol,
                    quote_volume=vol * float(curr_close),
                )
                logger.info(
                    "[CANDIDATE FOUND] %s: Close=$%.4f > EMA=$%.4f | RSI %.2f -> %.2f (bounce: +%.2f) | Vol: %.2f",
                    symbol,
                    curr_close,
                    curr_ema,
                    prev_rsi,
                    curr_rsi,
                    bounce,
                    vol,
                )
                return candidate
            return None
        except Exception as e:
            logger.debug("Error scanning signals for candidate %s: %s", symbol, e)
            return None

    tasks = [_evaluate_symbol(sym) for sym in universe]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: List[Candidate] = [
        res for res in results
        if res is not None and not isinstance(res, Exception)
    ]

    # Sort descending by RSI bounce strength (and secondary by volume)
    candidates.sort(key=lambda c: (c.rsi_bounce, c.volume), reverse=True)
    return candidates
