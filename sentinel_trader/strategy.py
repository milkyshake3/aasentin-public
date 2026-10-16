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

from sentinel_trader.config import Config

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
    atr: float = 0.0
    ema20: float = 0.0


class CandidateList(list):
    """Container for Candidate list that preserves list semantics while exposing diagnostic records."""

    def __init__(
        self,
        candidates: List[Candidate],
        diagnostics: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
    ):
        super().__init__(candidates)
        self.diagnostics: List[Dict[str, Any]] = diagnostics or []
        self.summary: str = summary


def format_price(val: float) -> str:
    """Format price cleanly with commas and appropriate decimal precision for diagnostic scan output."""
    if val >= 100:
        return f"${round(val):,}"
    elif val >= 1:
        return f"${val:,.2f}"
    else:
        return f"${val:,.4f}"


class IndicatorResult(tuple):
    """Container returning RSI and EMA, supporting tuple unpacking, attributes, and column access."""

    def __new__(
        cls,
        rsi: pd.Series,
        ema: pd.Series,
        df: Optional[pd.DataFrame] = None,
        atr: Optional[pd.Series] = None,
        ema20: Optional[pd.Series] = None,
    ):
        return super().__new__(cls, (rsi, ema))

    def __init__(
        self,
        rsi: pd.Series,
        ema: pd.Series,
        df: Optional[pd.DataFrame] = None,
        atr: Optional[pd.Series] = None,
        ema20: Optional[pd.Series] = None,
    ):
        self.rsi = rsi
        self.ema = ema
        self.df = df
        self.atr = atr if atr is not None else (df["atr14"] if df is not None and "atr14" in df.columns else None)
        self.atr14 = self.atr
        self.ema20 = ema20 if ema20 is not None else (df["ema20"] if df is not None and "ema20" in df.columns else None)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item.lower() in ("rsi", "rsi14", "rsi_14"):
                return self.rsi
            if item.lower() in ("ema", "ema200", "ema_200", "ema100", "ema_100"):
                return self.ema
            if item.lower() in ("atr", "atr14", "atr_14"):
                return self.atr
            if item.lower() in ("ema20", "ema_20"):
                return self.ema20
            if self.df is not None and item in self.df.columns:
                return self.df[item]
            raise KeyError(f"Key '{item}' not found in IndicatorResult or underlying DataFrame")
        return super().__getitem__(item)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's exponential smoothing.

    Computes True Range:
        TR = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
    Smooths via Wilder's EMA (alpha = 1 / period).

    Args:
        df: Pandas DataFrame containing 'high', 'low', 'close' columns (case-insensitive).
        period: Smoothing lookback period (default: 14).

    Returns:
        pd.Series: Computed ATR values.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # Resolve close, high, low columns case-insensitively
    close_col = next((c for c in ("close", "Close", "CLOSE") if c in df.columns), None)
    if close_col is None:
        raise KeyError(f"DataFrame must contain a 'close' column. Found columns: {list(df.columns)}")

    high_col = next((c for c in ("high", "High", "HIGH") if c in df.columns), None)
    low_col = next((c for c in ("low", "Low", "LOW") if c in df.columns), None)

    close = pd.to_numeric(df[close_col], errors="coerce")
    high = pd.to_numeric(df[high_col], errors="coerce") if high_col else close
    low = pd.to_numeric(df[low_col], errors="coerce") if low_col else close

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # pandas max(axis=1) skips NaN by default so index 0 uses tr1 (high - low)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing: alpha = 1 / period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    atr.name = f"atr{period}"
    return atr


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


def calculate_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    ema_period: int = 100,
    atr_period: int = 14,
    ema_fast_period: int = 20,
) -> IndicatorResult:
    """Calculate RSI(14), EMA trend filter, ATR(14), and EMA(20) for a price dataframe.

    Adds 'rsi', 'ema', 'rsi14', 'ema{period}', 'ema200', 'atr', 'atr14', and 'ema20'
    columns to the dataframe in-place and returns an IndicatorResult which can be unpacked
    as (rsi, ema) or accessed via attributes (.atr, .ema20) and dictionary keys.

    Args:
        df: Pandas DataFrame containing at least a 'close' (or 'Close') column.
        rsi_period: Lookback period for Wilder's RSI (default: 14).
        ema_period: Lookback period for EMA trend filter (default: 100).
        atr_period: Lookback period for Wilder's ATR (default: 14).
        ema_fast_period: Lookback period for short-term EMA fair-value target (default: 20).

    Returns:
        IndicatorResult: Unpackable as (rsi, ema), with .rsi, .ema, .atr, and .ema20 attributes.
    """
    if df.empty:
        empty_series = pd.Series(dtype="float64")
        return IndicatorResult(empty_series, empty_series, df=df, atr=empty_series, ema20=empty_series)

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
    rsi_series = calculate_rsi(close_series, period=rsi_period)
    ema_series = calculate_ema(close_series, span=ema_period, min_periods=1)
    atr_series = calculate_atr(df, period=atr_period)
    ema20_series = calculate_ema(close_series, span=ema_fast_period, min_periods=1)

    # Attach to dataframe in-place
    df["rsi"] = rsi_series
    df["ema"] = ema_series
    df["rsi14"] = rsi_series
    df[f"ema{ema_period}"] = ema_series
    if ema_period != 200:
        df["ema200"] = calculate_ema(close_series, span=200, min_periods=1)
    else:
        df["ema200"] = ema_series

    # Attach quantitative indicators: atr, atr14, ema20
    df["atr"] = atr_series
    df["atr14"] = atr_series
    df["ema20"] = ema20_series

    return IndicatorResult(rsi_series, ema_series, df=df, atr=atr_series, ema20=ema20_series)


def check_entry_signal(
    df: pd.DataFrame,
    rsi_threshold: float = 35.0,
    ema_period: int = 200,
) -> bool:
    """Check if the latest completed candles satisfy the active entry condition.

    Evaluates indicators strictly on finalized/completed bars to prevent mid-candle
    repainting caused by temporary intraday wicks or fluctuating forming candles.
    The active forming candle (df.iloc[-1]) is ignored.
    Crossover logic is based strictly on the last finalized candle (df.iloc[-2])
    and the preceding finalized candle (df.iloc[-3]).

    Condition:
        1. Last Finalized Close (df.iloc[-2]) > Finalized EMA (df.iloc[-2]) (trend confirmation filter)
        2. RSI(14) crosses upward through rsi_threshold:
           - Preceding finalized candle (df.iloc[-3]) RSI <= rsi_threshold
           - Last finalized candle (df.iloc[-2]) RSI > rsi_threshold

    Args:
        df: Pandas DataFrame with OHLCV data. If 'rsi' or 'ema' columns are not present,
            they will be calculated automatically.
        rsi_threshold: RSI threshold for oversold crossover (default: 35.0).
        ema_period: EMA lookback period for trend filter (default: 200).

    Returns:
        bool: True if entry conditions are strictly met on completed candles; False otherwise.
    """
    if df is None or len(df) < 3:
        return False

    # Ensure indicators exist
    if "rsi" not in df.columns or "ema" not in df.columns:
        calculate_indicators(df, ema_period=ema_period)

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
    rsi_crossed_up = (float(prev_rsi) <= rsi_threshold) and (float(curr_rsi) > rsi_threshold)
    above_trend = float(curr_close) > float(curr_ema)

    signal = bool(rsi_crossed_up and above_trend)

    if signal:
        logger.info(
            "[ENTRY SIGNAL TRIGGERED] Closed Candle Close=$%.4f > EMA(%d)=$%.4f | RSI(14) crossed up: %.2f -> %.2f (Threshold %g)",
            curr_close,
            ema_period,
            curr_ema,
            prev_rsi,
            curr_rsi,
            rsi_threshold,
        )

    return signal


async def scan_universe_for_signals(
    exchange: Any,
    universe: List[str],
    timeframe: str = "15m",
    rsi_threshold: Optional[float] = None,
    ema_period: Optional[int] = None,
) -> CandidateList:
    """Scan screened universe pairs concurrently using asyncio.gather with diagnostic logging.

    Checks entry conditions strictly on completed candle (iloc[-2]):
    - Close > EMA (trend filter)
    - RSI(14) upward crossover through rsi_threshold (iloc[-3] <= threshold and iloc[-2] > threshold)

    Logs compact diagnostic summaries for evaluated candidates:
    - [SCAN DIAGNOSTICS] {N} pairs evaluated:
      • BTC/USDT: Price $64,120 < EMA $64,800 [REJECT: Below EMA100]
      • ETH/USDT: Price $2,640 > EMA $2,580 | RSI 52.4 [REJECT: RSI not crossing 38]
      • SOL/USDT: Price $152 > EMA $148 | RSI 36.1 (was 32.4) [PASS: Signal Found]
    - If no pairs trigger: [SCAN RESULT] 0/{N} passed entry criteria. Maintaining 100% Cash.

    Returns:
        CandidateList: Matching candidate pairs sorted descending by RSI bounce strength.
    """
    if not universe:
        return CandidateList([], diagnostics=[], summary="[SCAN DIAGNOSTICS] 0 pairs evaluated:\n[SCAN RESULT] 0/0 passed entry criteria. Maintaining 100% Cash.")

    # Resolve sensitivity calibration parameters
    cfg = getattr(exchange, "config", None)
    is_real_config = isinstance(cfg, Config)

    if rsi_threshold is not None:
        effective_rsi_threshold = float(rsi_threshold)
    elif is_real_config and hasattr(cfg, "rsi_oversold_threshold"):
        effective_rsi_threshold = float(cfg.rsi_oversold_threshold)
    else:
        effective_rsi_threshold = 35.0

    if ema_period is not None:
        effective_ema_period = int(ema_period)
    elif is_real_config and hasattr(cfg, "ema_period"):
        effective_ema_period = int(cfg.ema_period)
    else:
        effective_ema_period = 200

    async def _evaluate_symbol(symbol: str) -> Tuple[Optional[Candidate], Dict[str, Any]]:
        try:
            if hasattr(exchange, "fetch_ohlcv_dataframe"):
                df = await exchange.fetch_ohlcv_dataframe(symbol, timeframe=timeframe, limit=250)
            elif hasattr(exchange, "fetch_ohlcv"):
                raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                line = f"• {symbol}: [REJECT: No OHLCV data interface]"
                return None, {"symbol": symbol, "passed": False, "line": line}

            if df is None or len(df) < 3:
                line = f"• {symbol}: [REJECT: Insufficient candle history]"
                return None, {"symbol": symbol, "passed": False, "line": line}

            close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
            if close_col is None:
                line = f"• {symbol}: [REJECT: Missing close column]"
                return None, {"symbol": symbol, "passed": False, "line": line}

            if "rsi" not in df.columns or "ema" not in df.columns:
                calculate_indicators(df, ema_period=effective_ema_period)

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
                line = f"• {symbol}: [REJECT: Incomplete indicator warmup]"
                return None, {"symbol": symbol, "passed": False, "line": line}

            curr_close_f = float(curr_close)
            curr_ema_f = float(curr_ema)
            curr_rsi_f = float(curr_rsi)
            prev_rsi_f = float(prev_rsi)

            price_str = format_price(curr_close_f)
            ema_str = format_price(curr_ema_f)
            thresh_str = f"{effective_rsi_threshold:g}"

            above_trend = curr_close_f > curr_ema_f
            rsi_crossed_up = (prev_rsi_f <= effective_rsi_threshold) and (curr_rsi_f > effective_rsi_threshold)

            if not above_trend:
                line = f"• {symbol}: Price {price_str} < EMA {ema_str} [REJECT: Below EMA{effective_ema_period}]"
                return None, {
                    "symbol": symbol,
                    "passed": False,
                    "close": curr_close_f,
                    "ema": curr_ema_f,
                    "rsi": curr_rsi_f,
                    "prev_rsi": prev_rsi_f,
                    "line": line,
                }
            elif not rsi_crossed_up:
                line = f"• {symbol}: Price {price_str} > EMA {ema_str} | RSI {curr_rsi_f:.1f} [REJECT: RSI not crossing {thresh_str}]"
                return None, {
                    "symbol": symbol,
                    "passed": False,
                    "close": curr_close_f,
                    "ema": curr_ema_f,
                    "rsi": curr_rsi_f,
                    "prev_rsi": prev_rsi_f,
                    "line": line,
                }
            else:
                line = f"• {symbol}: Price {price_str} > EMA {ema_str} | RSI {curr_rsi_f:.1f} (was {prev_rsi_f:.1f}) [PASS: Signal Found]"
                vol = (
                    float(df["volume"].iloc[-2])
                    if "volume" in df.columns and not pd.isna(df["volume"].iloc[-2])
                    else 0.0
                )
                curr_atr = (
                    float(df["atr14"].iloc[-2])
                    if "atr14" in df.columns and not pd.isna(df["atr14"].iloc[-2])
                    else 0.0
                )
                curr_ema20 = (
                    float(df["ema20"].iloc[-2])
                    if "ema20" in df.columns and not pd.isna(df["ema20"].iloc[-2])
                    else 0.0
                )
                bounce = curr_rsi_f - prev_rsi_f
                candidate = Candidate(
                    symbol=symbol,
                    close=curr_close_f,
                    ema=curr_ema_f,
                    rsi=curr_rsi_f,
                    prev_rsi=prev_rsi_f,
                    rsi_bounce=round(bounce, 4),
                    volume=vol,
                    quote_volume=vol * curr_close_f,
                    atr=curr_atr,
                    ema20=curr_ema20,
                )
                logger.info(
                    "[CANDIDATE FOUND] %s: Close=$%.4f > EMA=$%.4f | RSI %.2f -> %.2f (bounce: +%.2f) | Vol: %.2f",
                    symbol,
                    curr_close_f,
                    curr_ema_f,
                    prev_rsi_f,
                    curr_rsi_f,
                    bounce,
                    vol,
                )
                return candidate, {
                    "symbol": symbol,
                    "passed": True,
                    "close": curr_close_f,
                    "ema": curr_ema_f,
                    "rsi": curr_rsi_f,
                    "prev_rsi": prev_rsi_f,
                    "line": line,
                }

        except Exception as e:
            logger.debug("Error scanning signals for candidate %s: %s", symbol, e)
            line = f"• {symbol}: [REJECT: Scan error: {e}]"
            return None, {"symbol": symbol, "passed": False, "line": line}

    tasks = [_evaluate_symbol(sym) for sym in universe]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: List[Candidate] = []
    diagnostics: List[Dict[str, Any]] = []

    for res in results:
        if isinstance(res, Exception):
            logger.debug("Evaluation task exception: %s", res)
            continue
        if res is not None:
            cand, diag = res
            if diag:
                diagnostics.append(diag)
            if cand:
                candidates.append(cand)

    # Sort descending by RSI bounce strength (and secondary by volume)
    candidates.sort(key=lambda c: (c.rsi_bounce, c.volume), reverse=True)

    # Diagnostic Scan Logging:
    total_evaluated = len(diagnostics)
    logger.info("[SCAN DIAGNOSTICS] %d pairs evaluated:", total_evaluated)
    for diag in diagnostics:
        logger.info("%s", diag["line"])

    if not candidates:
        logger.info(f"[SCAN RESULT] 0/{total_evaluated} passed entry criteria. Maintaining 100% Cash.")
    else:
        logger.info(f"[SCAN RESULT] {len(candidates)}/{total_evaluated} passed entry criteria.")

    summary_lines = [f"[SCAN DIAGNOSTICS] {total_evaluated} pairs evaluated:"]
    summary_lines.extend(d["line"] for d in diagnostics)
    if not candidates:
        summary_lines.append(f"[SCAN RESULT] 0/{total_evaluated} passed entry criteria. Maintaining 100% Cash.")
    else:
        summary_lines.append(f"[SCAN RESULT] {len(candidates)}/{total_evaluated} passed entry criteria.")
    summary_str = "\n".join(summary_lines)

    return CandidateList(candidates, diagnostics=diagnostics, summary=summary_str)
