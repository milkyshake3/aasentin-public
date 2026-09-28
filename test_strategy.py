"""Unit and integration test suite for Sentinel-Trader Strategy & FSM Lifecycle.

Verifies:
1. RSI(14) and EMA(200) calculation accuracy on synthetic trending and mean-reverting price arrays.
2. Signal determinism: Close > EMA(200) AND RSI(14) upward crossover through 35.
3. Edge cases: Trend filter rejection, already-crossed rejection, downward-cross rejection, NaN handling.
4. IndicatorResult tuple unpacking, attribute, and column access semantics.
5. ExchangeClient OHLCV dataframe fetching and idempotent market buy execution.
6. Full Finite State Machine (FSM) transitions:
   CASH_SCANNING -> ENTERING -> IN_POSITION -> COOLDOWN -> CASH_SCANNING.
7. AI Veto enforcement blocking technical entry upon active high-threat news.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from sentinel_trader.ai_sentinel import NewsAnalysis
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.main import BotState, SentinelTrader
from sentinel_trader.news_watcher import NewsArticle
from sentinel_trader.strategy import (
    Candidate,
    IndicatorResult,
    calculate_atr,
    calculate_ema,
    calculate_indicators,
    calculate_rsi,
    check_entry_signal,
)


# =====================================================================
# 1. Synthetic Data Generators for Signal Determinism
# =====================================================================

def make_trending_dataframe(n: int = 250, start_price: float = 100.0, trend: str = "up") -> pd.DataFrame:
    """Generate deterministic synthetic trending price series."""
    prices = [start_price]
    step = 0.5 if trend == "up" else -0.5
    for i in range(1, n):
        prices.append(max(1.0, prices[-1] + step))

    df = pd.DataFrame({
        "timestamp": [1600000000000 + (i * 900000) for i in range(n)],
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [max(0.1, p - 0.5) for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })
    return df


def make_mean_reverting_dataframe(n: int = 250, mean_price: float = 100.0, amplitude: float = 10.0) -> pd.DataFrame:
    """Generate deterministic synthetic mean-reverting sinusoidal price series."""
    prices = []
    for i in range(n):
        # Sine wave oscillating around mean_price
        p = mean_price + amplitude * np.sin(2 * np.pi * i / 20)
        prices.append(round(p, 4))

    df = pd.DataFrame({
        "timestamp": [1600000000000 + (i * 900000) for i in range(n)],
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })
    return df


# =====================================================================
# 2. Indicator Calculation Tests
# =====================================================================

def test_rsi_calculation_on_synthetic_uptrend():
    """Verify RSI reaches high overbought values (> 70) on persistent upward price trend."""
    df = make_trending_dataframe(n=250, start_price=100.0, trend="up")
    rsi = calculate_rsi(df["close"], period=14)

    assert len(rsi) == 250
    # First 14 values warm up
    assert pd.isna(rsi.iloc[0])
    # Persistent uptrend with zero downward steps must reach 100 or near 100
    latest_rsi = rsi.iloc[-1]
    assert latest_rsi > 80.0
    assert (rsi.dropna() <= 100.0).all()
    assert (rsi.dropna() >= 0.0).all()


def test_rsi_calculation_on_synthetic_downtrend():
    """Verify RSI reaches low oversold values (< 30) on persistent downward price trend."""
    df = make_trending_dataframe(n=250, start_price=200.0, trend="down")
    rsi = calculate_rsi(df["close"], period=14)

    assert len(rsi) == 250
    latest_rsi = rsi.iloc[-1]
    assert latest_rsi < 20.0
    assert (rsi.dropna() <= 100.0).all()
    assert (rsi.dropna() >= 0.0).all()


def test_rsi_calculation_on_mean_reverting_series():
    """Verify RSI oscillates deterministically between bounds on mean-reverting series."""
    df = make_mean_reverting_dataframe(n=250, mean_price=100.0, amplitude=15.0)
    rsi = calculate_rsi(df["close"], period=14)

    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 200
    # Mean of oscillating RSI should center near 50
    assert 40.0 < valid_rsi.mean() < 60.0
    # Oscillation must touch both lower and upper regions
    assert valid_rsi.min() < 35.0
    assert valid_rsi.max() > 65.0


def test_ema_calculation_on_trending_and_mean_reverting():
    """Verify EMA(200) properly lags price in trends and stays stable in mean-reversion."""
    # Uptrend: Close must be above EMA(200)
    up_df = make_trending_dataframe(n=250, start_price=100.0, trend="up")
    ema_up = calculate_ema(up_df["close"], span=200)
    assert ema_up.iloc[-1] < up_df["close"].iloc[-1]

    # Downtrend: Close must be below EMA(200)
    down_df = make_trending_dataframe(n=250, start_price=200.0, trend="down")
    ema_down = calculate_ema(down_df["close"], span=200)
    assert ema_down.iloc[-1] > down_df["close"].iloc[-1]

    # Mean-reverting: EMA should hover near the 100.0 center
    mr_df = make_mean_reverting_dataframe(n=250, mean_price=100.0, amplitude=10.0)
    ema_mr = calculate_ema(mr_df["close"], span=200)
    assert 95.0 < ema_mr.iloc[-1] < 105.0


def test_indicator_result_container_semantics():
    """Verify IndicatorResult supports tuple unpacking, attributes, and dict-like indexing."""
    df = make_trending_dataframe(n=100, start_price=100.0, trend="up")

    # 1. Unpacking as 2-tuple (rsi, ema)
    rsi, ema = calculate_indicators(df)
    assert isinstance(rsi, pd.Series)
    assert isinstance(ema, pd.Series)
    assert len(rsi) == 100

    # 2. DataFrame mutation check
    assert "rsi" in df.columns
    assert "ema" in df.columns
    assert "rsi14" in df.columns
    assert "ema200" in df.columns
    assert "atr" in df.columns
    assert "atr14" in df.columns
    assert "ema20" in df.columns

    # 3. Object attribute and item access
    res = calculate_indicators(df)
    assert isinstance(res, IndicatorResult)
    assert res.rsi is not None
    assert res.ema is not None
    assert res.atr is not None
    assert res.atr14 is not None
    assert res.ema20 is not None
    pd.testing.assert_series_equal(res["rsi"], rsi)
    pd.testing.assert_series_equal(res["ema"], ema)
    pd.testing.assert_series_equal(res["atr"], res.atr)
    pd.testing.assert_series_equal(res["atr14"], res.atr)
    pd.testing.assert_series_equal(res["ema20"], res.ema20)
    pd.testing.assert_series_equal(res["close"], df["close"])


def test_atr_calculation_constant_tr():
    """Verify ATR(14) converges exactly to constant True Range."""
    n = 50
    # High-Low is constantly 2.0, close is constantly in the middle
    df = pd.DataFrame({
        "high": [102.0] * n,
        "low": [100.0] * n,
        "close": [101.0] * n,
    })
    atr = calculate_atr(df, period=14)
    assert len(atr) == n
    # Since TR is 2.0 at every single candle, Wilder's EMA must be exactly 2.0
    assert np.allclose(atr.values, 2.0, atol=1e-8)


def test_atr_calculation_accuracy_against_control_series():
    """Verify ATR(14) numerical accuracy against a strictly calculated control series."""
    # 20 candles of realistic price data with varying highs, lows, and gaps
    highs = [
        105.0, 107.0, 106.5, 108.0, 110.0,
        109.5, 111.0, 112.5, 111.0, 113.0,
        115.0, 114.0, 116.0, 117.5, 119.0,
        118.0, 120.0, 121.5, 120.0, 122.0,
    ]
    lows = [
        101.0, 103.0, 102.0, 104.0, 106.0,
        105.0, 107.0, 108.0, 107.5, 109.0,
        111.0, 110.0, 112.0, 113.5, 115.0,
        114.0, 116.0, 117.0, 116.5, 118.0,
    ]
    closes = [
        103.0, 106.0, 104.0, 107.5, 108.0,
        107.0, 110.0, 110.5, 108.5, 112.0,
        113.0, 111.5, 115.0, 116.0, 117.0,
        115.5, 119.0, 119.5, 118.0, 121.0,
    ]

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    # Compute control series step-by-step using Wilder's exact formula
    tr_control = []
    for i in range(len(highs)):
        h = highs[i]
        l = lows[i]
        if i == 0:
            tr = h - l
        else:
            prev_c = closes[i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_control.append(tr)

    period = 14
    alpha = 1.0 / period
    atr_control = []
    curr_atr = tr_control[0]
    atr_control.append(curr_atr)
    for i in range(1, len(tr_control)):
        curr_atr = (1.0 - alpha) * curr_atr + alpha * tr_control[i]
        atr_control.append(curr_atr)

    calculated_atr = calculate_atr(df, period=14)

    assert len(calculated_atr) == len(atr_control)
    # Numerical accuracy check across all 20 candles
    assert np.allclose(calculated_atr.values, atr_control, atol=1e-6)


# =====================================================================
# 3. Entry Signal Determinism & Edge Case Tests
# =====================================================================

def test_check_entry_signal_true_crossover():
    """Verify signal triggers when Close > EMA(200) AND RSI crosses upward through 35 on finalized bar."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Force preceding finalized candle: oversold (RSI <= 35)
    df.loc[df.index[-3], "rsi"] = 32.5

    # Force last finalized candle: crossed upward through 35 (RSI > 35) and above EMA
    df.loc[df.index[-2], "rsi"] = 36.8
    df.loc[df.index[-2], "close"] = 153.5
    df.loc[df.index[-2], "ema"] = 150.0

    assert check_entry_signal(df) is True


def test_check_entry_signal_rejected_by_trend_filter():
    """Verify signal is rejected if RSI crosses 35 but Close <= EMA(200) on finalized candle."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Crosses 35 upward
    df.loc[df.index[-3], "rsi"] = 33.0
    df.loc[df.index[-2], "rsi"] = 37.0

    # BUT Close is below EMA(200) (bearish regime) on finalized candle
    df.loc[df.index[-2], "close"] = 145.0
    df.loc[df.index[-2], "ema"] = 150.0

    assert check_entry_signal(df) is False


def test_check_entry_signal_rejected_if_already_above_35():
    """Verify signal is rejected if RSI was already > 35 on preceding finalized candle."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Prior finalized candle was already 38.0 (> 35)
    df.loc[df.index[-3], "rsi"] = 38.0
    df.loc[df.index[-2], "rsi"] = 42.0
    df.loc[df.index[-2], "close"] = 155.0
    df.loc[df.index[-2], "ema"] = 150.0

    assert check_entry_signal(df) is False


def test_check_entry_signal_rejected_if_downward_cross():
    """Verify signal is rejected if RSI crossed downward through 35 on finalized candle."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Downward cross: 38 -> 33
    df.loc[df.index[-3], "rsi"] = 38.0
    df.loc[df.index[-2], "rsi"] = 33.0
    df.loc[df.index[-2], "close"] = 155.0
    df.loc[df.index[-2], "ema"] = 150.0

    assert check_entry_signal(df) is False


def test_check_entry_signal_edge_cases_empty_and_nans():
    """Verify robust handling of empty, short, or NaN dataframes."""
    assert check_entry_signal(None) is False
    assert check_entry_signal(pd.DataFrame()) is False
    assert check_entry_signal(pd.DataFrame({"close": [150.0]})) is False
    assert check_entry_signal(pd.DataFrame({"close": [150.0, 151.0]})) is False

    df_nans = pd.DataFrame({"close": [np.nan, np.nan, np.nan, np.nan]})
    assert check_entry_signal(df_nans) is False


def test_check_entry_signal_ignores_fluctuating_forming_candle():
    """Verify check_entry_signal ignores fluctuating values in df.iloc[-1] and triggers strictly on df.iloc[-2]."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Finalized crossover on closed candles: df.iloc[-3] oversold, df.iloc[-2] crossed up above EMA
    df.loc[df.index[-3], "rsi"] = 32.0
    df.loc[df.index[-2], "rsi"] = 36.5
    df.loc[df.index[-2], "close"] = 155.0
    df.loc[df.index[-2], "ema"] = 150.0

    # 1. Forming candle (df.iloc[-1]) dumps into extreme oversold mid-candle:
    # Must STILL trigger because the finalized bar closed bullishly.
    df.loc[df.index[-1], "rsi"] = 18.0
    df.loc[df.index[-1], "close"] = 130.0
    df.loc[df.index[-1], "ema"] = 150.0
    assert check_entry_signal(df) is True

    # 2. Forming candle (df.iloc[-1]) wicks up into overbought:
    # Must STILL trigger based strictly on df.iloc[-2].
    df.loc[df.index[-1], "rsi"] = 85.0
    df.loc[df.index[-1], "close"] = 180.0
    assert check_entry_signal(df) is True


def test_check_entry_signal_anti_repainting_mid_candle_false_breakout():
    """Verify temporary intraday wicks in forming bar do NOT trigger false breakout signals."""
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    calculate_indicators(df)

    # Finalized candle (df.iloc[-2]) did NOT cross: stayed oversold at 32.0
    df.loc[df.index[-3], "rsi"] = 31.0
    df.loc[df.index[-2], "rsi"] = 32.0
    df.loc[df.index[-2], "close"] = 148.0
    df.loc[df.index[-2], "ema"] = 150.0

    # Active forming candle (df.iloc[-1]) experiences a temporary mid-candle pump
    # that pushes RSI above 35 and Close above EMA
    df.loc[df.index[-1], "rsi"] = 42.0
    df.loc[df.index[-1], "close"] = 160.0
    df.loc[df.index[-1], "ema"] = 150.0

    # Must NOT trigger because df.iloc[-2] is the last finalized bar and did not cross
    assert check_entry_signal(df) is False


# =====================================================================
# 4. Exchange Client OHLCV and Market Buy Idempotency Tests
# =====================================================================

@pytest.mark.asyncio
async def test_exchange_client_fetch_ohlcv_dataframe():
    """Verify ExchangeClient generates valid OHLCV dataframe in mock mode."""
    config = Config(mock_exchange=True, timeframe="15m")
    client = ExchangeClient(config)
    await client.initialize()

    df = await client.fetch_ohlcv_dataframe("SOL/USDT", timeframe="15m", limit=150)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["low"]).all()
    assert (df["volume"] > 0).all()


@pytest.mark.asyncio
async def test_exchange_client_market_buy_idempotent_mock():
    """Verify execute_market_buy_idempotent allocates quote asset and updates base balance."""
    config = Config(mock_exchange=True)
    client = ExchangeClient(config)
    await client.initialize()

    client.set_mock_balance("USDT", 1000.0)
    client.set_mock_balance("SOL", 0.0)
    client.set_mock_price("SOL/USDT", 100.0)

    # Buy with $250.0 USDT
    receipt = await client.execute_market_buy_idempotent("SOL/USDT", cost_usdt=250.0)

    assert receipt["side"] == "buy"
    assert receipt["status"] == "closed"
    assert receipt["price"] == 100.0
    assert receipt["amount"] == 2.5
    assert receipt["cost"] == 250.0

    # Ledger check
    assert client._mock_balances["USDT"] == 750.0
    assert client._mock_balances["SOL"] == 2.5


# =====================================================================
# 5. Full FSM Lifecycle & State Transition Tests
# =====================================================================

@pytest.mark.asyncio
async def test_fsm_cash_scanning_to_entering_transition():
    """Verify CASH_SCANNING transitions to ENTERING upon valid signal with clear AI sentiment."""
    config = Config(
        mock_exchange=True,
        llm_provider="mock",
        enable_rss_news=False,
        enable_simulated_news=False,
        poll_interval_seconds=0.01,
        allocation_pct=0.25,
    )
    bot = SentinelTrader(config, scenario="default")
    await bot.initialize()

    # Create synthetic OHLCV that has an entry signal on finalized bar
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0, amplitude=5.0)
    df.loc[df.index[-3], "close"] = 148.0
    df.loc[df.index[-2], "close"] = 152.0
    calculate_indicators(df)
    df.loc[df.index[-3], "rsi"] = 34.0
    df.loc[df.index[-2], "rsi"] = 39.5
    df.loc[df.index[-2], "ema"] = 150.0

    bot.exchange.set_mock_ohlcv(df)
    bot.state = BotState.CASH_SCANNING

    # Run one scanning step
    await bot._handle_cash_scanning()

    # Must have transitioned to ENTERING
    assert bot.state == BotState.ENTERING


@pytest.mark.asyncio
async def test_fsm_ai_veto_blocks_entering_transition():
    """Verify CASH_SCANNING stays in CASH_SCANNING if news has HIGH/CRITICAL threat within 2h."""
    config = Config(
        mock_exchange=True,
        llm_provider="mock",
        enable_rss_news=False,
        enable_simulated_news=False,
        poll_interval_seconds=0.01,
    )
    bot = SentinelTrader(config, scenario="default")
    await bot.initialize()

    # Inject critical threat news into history (30 minutes ago)
    threat_article = NewsArticle(
        id="threat_hash_1",
        headline="SOLANA CORE EXPLOIT: Critical zero-day vulnerability in runtime",
        source="EmergencyWire",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    threat_analysis = NewsAnalysis(
        action="PANIC_SELL",
        threat_level="CRITICAL",
        confidence=0.95,
        reasoning="Critical zero-day exploit detected",
    )
    bot._news_history.append((threat_article, threat_analysis, datetime.now(timezone.utc) - timedelta(minutes=30)))

    # Set up valid technical signal on finalized candle
    df = make_mean_reverting_dataframe(n=220, mean_price=150.0)
    calculate_indicators(df)
    df.loc[df.index[-3], "rsi"] = 34.0
    df.loc[df.index[-2], "rsi"] = 39.5
    df.loc[df.index[-2], "close"] = 155.0
    df.loc[df.index[-2], "ema"] = 150.0
    bot.exchange.set_mock_ohlcv(df)

    bot.state = BotState.CASH_SCANNING
    await bot._handle_cash_scanning()

    # VETO must block entry: State remains CASH_SCANNING
    assert bot.state == BotState.CASH_SCANNING


@pytest.mark.asyncio
async def test_fsm_entering_to_in_position_execution():
    """Verify ENTERING calculates 25% sizing, buys token, sends Discord alert, and enters IN_POSITION."""
    config = Config(
        mock_exchange=True,
        allocation_pct=0.25,
        take_profit_pct=0.05,
        stop_loss_pct=0.025,
        poll_interval_seconds=0.01,
    )
    bot = SentinelTrader(config)
    await bot.initialize()

    # Setup mock balance: 1000 USDT free
    bot.exchange.set_mock_balance("USDT", 1000.0)
    bot.exchange.set_mock_balance("SOL", 0.0)
    bot.exchange.set_mock_price("SOL/USDT", 150.0)

    # Mock notifier method
    bot.notifier.notify_position_entered = MagicMock(return_value=None)

    bot.state = BotState.ENTERING
    await bot._handle_entering()

    # Verify state transition to IN_POSITION
    assert bot.state == BotState.IN_POSITION

    # 25% of 1000 = 250 USDT spent
    assert bot.exchange._mock_balances["USDT"] < 1000.0
    assert bot.exchange._mock_balances["SOL"] > 0.0

    # Risk engine entry price set
    assert bot.risk_engine.entry_price == 150.0
    assert bot.risk_engine.is_liquidated is False

    # Discord alert sent
    bot.notifier.notify_position_entered.assert_called_once()
    kwargs = bot.notifier.notify_position_entered.call_args.kwargs
    assert kwargs["symbol"] == "SOL/USDT"
    assert kwargs["entry_price"] == 150.0
    assert kwargs["tp_pct"] == 0.05
    assert kwargs["sl_pct"] == 0.025


@pytest.mark.asyncio
async def test_fsm_in_position_liquidation_to_cooldown():
    """Verify IN_POSITION liquidates on Take-Profit and transitions to COOLDOWN."""
    config = Config(
        mock_exchange=True,
        take_profit_pct=0.05,  # +5%
        stop_loss_pct=0.025,  # -2.5%
        poll_interval_seconds=0.01,
        cooldown_minutes=15,
    )
    bot = SentinelTrader(config, scenario="default")
    await bot.initialize()

    bot.risk_engine.set_entry_price(100.0)
    bot.exchange.set_mock_balance("SOL", 5.0)
    bot.exchange.set_mock_price("SOL/USDT", 106.0)  # +6% breaches +5% TP
    bot.state = BotState.IN_POSITION

    await bot._handle_in_position()

    # Liquidation must be recorded
    assert bot.risk_engine.is_liquidated is True
    assert bot.risk_engine.last_liquidation is not None
    assert bot.risk_engine.last_liquidation.trigger.value == "TAKE_PROFIT"

    # State must transition to COOLDOWN
    assert bot.state == BotState.COOLDOWN


@pytest.mark.asyncio
async def test_fsm_complete_lifecycle_cycle():
    """Verify end-to-end cycle: CASH_SCANNING -> ENTERING -> IN_POSITION -> COOLDOWN -> CASH_SCANNING."""
    config = Config(
        mock_exchange=True,
        llm_provider="mock",
        enable_rss_news=False,
        enable_simulated_news=False,
        poll_interval_seconds=0.01,
        cooldown_minutes=0,  # 0 minutes for instant test cooldown
        allocation_pct=0.25,
        take_profit_pct=0.05,
        stop_loss_pct=0.025,
    )
    bot = SentinelTrader(config, scenario="default", max_cycles=1)
    await bot.initialize()

    # Step 1: Set up entry signal on finalized candle
    df = make_mean_reverting_dataframe(n=220, mean_price=100.0)
    calculate_indicators(df)
    df.loc[df.index[-3], "rsi"] = 33.0
    df.loc[df.index[-2], "rsi"] = 39.5
    df.loc[df.index[-2], "close"] = 105.0
    df.loc[df.index[-2], "ema"] = 100.0
    bot.exchange.set_mock_ohlcv(df)
    bot.exchange.set_mock_balance("USDT", 1000.0)
    bot.exchange.set_mock_price("SOL/USDT", 105.0)

    # Transition 1: CASH_SCANNING -> ENTERING
    assert bot.state == BotState.CASH_SCANNING
    await bot._handle_cash_scanning()
    assert bot.state == BotState.ENTERING

    # Transition 2: ENTERING -> IN_POSITION
    await bot._handle_entering()
    assert bot.state == BotState.IN_POSITION
    assert bot.risk_engine.entry_price == 105.0

    # Step 2: Trigger Take-Profit (+6%)
    bot.exchange.set_mock_price("SOL/USDT", 112.0)

    # Transition 3: IN_POSITION -> COOLDOWN
    await bot._handle_in_position()
    assert bot.state == BotState.COOLDOWN
    assert bot.risk_engine.is_liquidated is True

    # Transition 4: COOLDOWN -> CASH_SCANNING
    await bot._handle_cooldown()
    assert bot.state == BotState.CASH_SCANNING
    assert bot._cycle_count == 1
