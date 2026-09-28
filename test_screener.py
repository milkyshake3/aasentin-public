"""Comprehensive Unit and Integration Tests for Dynamic Universe Screener and Multi-Slot Portfolio Manager.

Verifies:
1. Stablecoin blacklist filtering (assert USDC/USDT is removed despite #1 volume).
2. Sorting by quoteVolume (24h USDT turnover) vs baseVolume (token count).
3. 24-hour in-memory cache expiration logic and force_refresh bypass.
4. Leveraged token exclusion (UP/, DOWN/, BULL/, BEAR/) and inactive/non-spot market exclusion.
5. Mock exchange fallback to default liquid list ["BTC/USDT", "ETH/USDT", ...].
6. Multi-candidate signal evaluation (scan_universe_for_signals) using asyncio.gather.
7. Multi-slot portfolio management:
   - MAX_ACTIVE_POSITIONS = 3 boundary enforcement.
   - SLOT_ALLOCATION_PCT = 0.25 sizing with $15 minimum trade size.
   - Independent TP (+5.0%) and SL (-2.5%) price basis tracking per slot.
   - Asset-specific AI threat isolation.
   - Global Macro Breaker concurrent liquidation across all open slots.
8. Multi-position state persistence and crash-recovery rehydration.
"""

import asyncio
from datetime import datetime, timezone
import time
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import pytest

from sentinel_trader.ai_sentinel import NewsAnalysis
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.main import BotState, SentinelTrader
from sentinel_trader.risk_engine import (
    MAX_ACTIVE_POSITIONS,
    SLOT_ALLOCATION_PCT,
    LiquidationRecord,
    PositionState,
    RiskEngine,
    TriggerReason,
)
from sentinel_trader.state_store import clear_state, load_state, save_state
from sentinel_trader.strategy import Candidate, calculate_indicators, scan_universe_for_signals
from sentinel_trader.universe_screener import (
    DEFAULT_MOCK_UNIVERSE,
    STABLECOIN_BLACKLIST,
    UniverseScreener,
    get_dynamic_universe,
)


# =====================================================================
# 1. Universe Screener Tests
# =====================================================================

@pytest.mark.asyncio
async def test_screener_stablecoin_blacklist_filtering():
    """Verify stablecoins and fiat pairs are excluded even when possessing highest volume."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    # USDC/USDT has top volume ($500M), FDUSD/USDT has $300M, BTC/USDT has $150M
    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "USDC/USDT": {"symbol": "USDC/USDT", "quoteVolume": 500000000.0, "spot": True, "active": True},
            "FDUSD/USDT": {"symbol": "FDUSD/USDT", "quoteVolume": 300000000.0, "spot": True, "active": True},
            "EUR/USDT": {"symbol": "EUR/USDT", "quoteVolume": 200000000.0, "spot": True, "active": True},
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 150000000.0, "spot": True, "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 100000000.0, "spot": True, "active": True},
            "SOL/USDT": {"symbol": "SOL/USDT", "quoteVolume": 80000000.0, "spot": True, "active": True},
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400)
    universe = await screener.get_dynamic_universe(mock_exchange, top_n=10)

    # Blacklisted assets must be filtered out completely
    assert "USDC/USDT" not in universe
    assert "FDUSD/USDT" not in universe
    assert "EUR/USDT" not in universe

    # Legitimate spot crypto assets must be ranked by quoteVolume
    assert universe[0] == "BTC/USDT"
    assert universe[1] == "ETH/USDT"
    assert universe[2] == "SOL/USDT"
    assert len(universe) == 3


@pytest.mark.asyncio
async def test_screener_sorting_quote_vs_base_volume():
    """Verify universe is strictly sorted by quoteVolume (USDT turnover), not baseVolume (token count)."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            # PEPE: 1,000,000,000 base tokens, but only $10,000 USDT quote turnover
            "PEPE/USDT": {
                "symbol": "PEPE/USDT",
                "baseVolume": 1000000000.0,
                "quoteVolume": 10000.0,
                "spot": True,
                "active": True,
            },
            # BTC: 10 base tokens, but $650,000 USDT quote turnover
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "baseVolume": 10.0,
                "quoteVolume": 650000.0,
                "spot": True,
                "active": True,
            },
            # SOL: 2,000 base tokens, $300,000 USDT quote turnover
            "SOL/USDT": {
                "symbol": "SOL/USDT",
                "baseVolume": 2000.0,
                "quoteVolume": 300000.0,
                "spot": True,
                "active": True,
            },
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400)
    universe = await screener.get_dynamic_universe(mock_exchange, top_n=3)

    # Must be ordered by quoteVolume descending: BTC ($650k) > SOL ($300k) > PEPE ($10k)
    assert universe == ["BTC/USDT", "SOL/USDT", "PEPE/USDT"]


@pytest.mark.asyncio
async def test_screener_24h_cache_expiration():
    """Verify 24-hour cache prevents duplicate API calls until expiration or forced refresh."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}
    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 1000000.0, "spot": True, "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 500000.0, "spot": True, "active": True},
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400)

    # 1. First fetch: Queries exchange
    u1 = await screener.get_dynamic_universe(mock_exchange, top_n=5)
    assert u1 == ["BTC/USDT", "ETH/USDT"]
    assert mock_exchange.fetch_tickers.call_count == 1

    # 2. Second fetch immediately: Cache hit, fetch_tickers NOT called
    u2 = await screener.get_dynamic_universe(mock_exchange, top_n=5)
    assert u2 == ["BTC/USDT", "ETH/USDT"]
    assert mock_exchange.fetch_tickers.call_count == 1

    # 3. Simulate cache expiration (24h + 1s elapsed)
    screener._cache_timestamp = time.time() - 86401
    assert screener.is_cache_valid() is False

    # 4. Third fetch: Cache expired, re-queries exchange
    u3 = await screener.get_dynamic_universe(mock_exchange, top_n=5)
    assert u3 == ["BTC/USDT", "ETH/USDT"]
    assert mock_exchange.fetch_tickers.call_count == 2

    # 5. Force refresh bypasses valid cache
    u4 = await screener.get_dynamic_universe(mock_exchange, top_n=5, force_refresh=True)
    assert u4 == ["BTC/USDT", "ETH/USDT"]
    assert mock_exchange.fetch_tickers.call_count == 3


@pytest.mark.asyncio
async def test_screener_leveraged_tokens_and_inactive_markets_filtered():
    """Verify leveraged tokens and non-spot / inactive markets are excluded."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}
    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "BTCUP/USDT": {"symbol": "BTCUP/USDT", "quoteVolume": 9000000.0, "spot": True, "active": True},
            "BTCDOWN/USDT": {"symbol": "BTCDOWN/USDT", "quoteVolume": 8000000.0, "spot": True, "active": True},
            "ETHBULL/USDT": {"symbol": "ETHBULL/USDT", "quoteVolume": 7000000.0, "spot": True, "active": True},
            "ETHBEAR/USDT": {"symbol": "ETHBEAR/USDT", "quoteVolume": 6000000.0, "spot": True, "active": True},
            "DELISTED/USDT": {"symbol": "DELISTED/USDT", "quoteVolume": 5000000.0, "spot": True, "active": False},
            "FUTURES/USDT": {"symbol": "FUTURES/USDT", "quoteVolume": 4000000.0, "spot": False, "active": True},
            "BTC/EUR": {"symbol": "BTC/EUR", "quoteVolume": 3000000.0, "spot": True, "active": True},
            "AVAX/USDT": {"symbol": "AVAX/USDT", "quoteVolume": 2000000.0, "spot": True, "active": True},
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400)
    universe = await screener.get_dynamic_universe(mock_exchange, top_n=10)

    # Only AVAX/USDT is a valid active spot pair settling in USDT
    assert universe == ["AVAX/USDT"]


@pytest.mark.asyncio
async def test_screener_mock_exchange_support():
    """Verify mock exchange returns default liquid list."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = True

    universe = await get_dynamic_universe(mock_exchange, top_n=6)
    assert universe == DEFAULT_MOCK_UNIVERSE
    assert universe == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]


# =====================================================================
# 2. Multi-Candidate Signal Evaluation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_scan_universe_for_signals():
    """Verify scan_universe_for_signals fetches candles concurrently and ranks by RSI bounce."""
    exchange = AsyncMock()

    # Create synthetic DataFrames for 3 symbols
    def make_signal_df(rsi_prev: float, rsi_curr: float, close: float, ema: float) -> pd.DataFrame:
        n = 220
        df = pd.DataFrame({
            "timestamp": [1600000000000 + (i * 900000) for i in range(n)],
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        calculate_indicators(df)
        df.loc[df.index[-3], "rsi"] = rsi_prev
        df.loc[df.index[-2], "rsi"] = rsi_curr
        df.loc[df.index[-2], "close"] = close
        df.loc[df.index[-2], "ema"] = ema
        return df

    # Pair 1 (Strong bounce: 30 -> 40, bounce = +10)
    df_sol = make_signal_df(30.0, 40.0, close=155.0, ema=150.0)
    # Pair 2 (Weak bounce: 34 -> 36, bounce = +2)
    df_eth = make_signal_df(34.0, 36.0, close=2650.0, ema=2600.0)
    # Pair 3 (No signal: below EMA)
    df_btc = make_signal_df(30.0, 40.0, close=59000.0, ema=60000.0)

    async def mock_fetch_ohlcv(symbol, timeframe="15m", limit=250):
        if symbol == "SOL/USDT":
            return df_sol
        elif symbol == "ETH/USDT":
            return df_eth
        else:
            return df_btc

    exchange.fetch_ohlcv_dataframe = AsyncMock(side_effect=mock_fetch_ohlcv)

    candidates = await scan_universe_for_signals(exchange, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    assert len(candidates) == 2
    # Ranked by bounce strength: SOL (bounce 10.0) > ETH (bounce 2.0)
    assert candidates[0].symbol == "SOL/USDT"
    assert candidates[0].rsi_bounce == 10.0
    assert candidates[1].symbol == "ETH/USDT"
    assert candidates[1].rsi_bounce == 2.0


# =====================================================================
# 3. Multi-Slot Portfolio Manager Tests
# =====================================================================

def test_risk_engine_multi_slot_boundaries():
    """Verify MAX_ACTIVE_POSITIONS = 3 and SLOT_ALLOCATION_PCT = 0.25 boundaries."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    assert risk.max_active_positions == 3
    assert risk.slot_allocation_pct == 0.25
    assert risk.can_open_position is True

    # Open Slot 1
    p1 = risk.open_position("SOL/USDT", entry_price=150.0, amount=10.0)
    assert len(risk.active_positions) == 1
    assert risk.can_open_position is True

    # Open Slot 2
    p2 = risk.open_position("ETH/USDT", entry_price=2600.0, amount=0.5)
    assert len(risk.active_positions) == 2
    assert risk.can_open_position is True

    # Open Slot 3 (Portfolio Full)
    p3 = risk.open_position("BTC/USDT", entry_price=65000.0, amount=0.1)
    assert len(risk.active_positions) == 3
    assert risk.can_open_position is False

    # Attempt to open Slot 4 must raise ValueError
    with pytest.raises(ValueError, match="MAX_ACTIVE_POSITIONS.*reached"):
        risk.open_position("AVAX/USDT", entry_price=30.0, amount=50.0)


def test_risk_engine_slot_allocation_sizing():
    """Verify position sizing allocates 25% of free USDT with $15 minimum trade size."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    # 1000 USDT free -> 25% = 250 USDT
    assert risk.calculate_slot_size(1000.0) == 250.0

    # 50 USDT free -> 25% = 12.50 USDT -> rounded up to 15.0 minimum
    assert risk.calculate_slot_size(50.0) == 15.0

    # 10 USDT free -> below $15 absolute minimum -> 0.0 (abort)
    assert risk.calculate_slot_size(10.0) == 0.0


def test_risk_engine_independent_slot_tp_sl():
    """Verify each slot independently monitors its entry basis against TP (+5%) and SL (-2.5%)."""
    config = Config(mock_exchange=True, take_profit_pct=0.05, stop_loss_pct=0.025)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0)
    risk.open_position("ETH/USDT", entry_price=2000.0, amount=1.0)

    # SOL pumps +6% (breaches +5% TP)
    sol_trigger = risk.evaluate_mathematical_limits(106.0, symbol="SOL/USDT")
    assert sol_trigger == TriggerReason.TAKE_PROFIT

    # ETH remains at entry (no breach)
    eth_trigger = risk.evaluate_mathematical_limits(2000.0, symbol="ETH/USDT")
    assert eth_trigger is None

    # ETH dumps -3% (breaches -2.5% SL)
    eth_dump_trigger = risk.evaluate_mathematical_limits(1940.0, symbol="ETH/USDT")
    assert eth_dump_trigger == TriggerReason.STOP_LOSS


@pytest.mark.asyncio
async def test_risk_engine_global_macro_breaker_all_slots():
    """Verify Global Macro Breaker liquidates ALL active slots simultaneously on systemic news."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    # Setup 3 active slots
    risk.open_position("SOL/USDT", entry_price=150.0, amount=10.0)
    risk.open_position("ETH/USDT", entry_price=2600.0, amount=1.0)
    risk.open_position("BTC/USDT", entry_price=65000.0, amount=0.1)

    assert len(risk.active_positions) == 3

    # Systemic macro news headline
    systemic_headline = "BREAKING: Binance halt and blanket crypto ban enforced globally"
    systemic_analysis = NewsAnalysis(
        action="PANIC_SELL",
        threat_level="CRITICAL",
        confidence=0.98,
        reasoning="Blanket crypto ban and exchange halt represents catastrophic systemic threat",
    )

    # Evaluate tick with systemic threat
    record = await risk.evaluate_tick(
        current_price=150.0,
        news_analysis=systemic_analysis,
        headline=systemic_headline,
    )

    assert record is not None
    assert record.trigger == TriggerReason.GLOBAL_MACRO_BREAKER

    # All 3 positions must have been liquidated
    assert len(risk.active_positions) == 0
    assert risk.is_liquidated is True
    assert len(risk.liquidation_history) == 3


# =====================================================================
# 4. Multi-Position State Store Tests
# =====================================================================

def test_state_store_multi_position_schema(tmp_path):
    """Verify state.json serializes and deserializes multiple active slots and screener cache."""
    state_file = str(tmp_path / "multi_state.json")

    positions = {
        "SOL/USDT": {"entry_price": 150.0, "amount": 10.0, "order_id": "ord_sol_1"},
        "ETH/USDT": {"entry_price": 2600.0, "amount": 0.5, "order_id": "ord_eth_2"},
    }
    screener_cache = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    cache_ts = 1727412000.0

    saved = save_state(
        state=BotState.IN_POSITION,
        active_positions=positions,
        screener_cache=screener_cache,
        screener_timestamp=cache_ts,
        file_path=state_file,
    )

    assert "active_positions" in saved
    assert len(saved["active_positions"]) == 2
    assert saved["active_positions"]["SOL/USDT"]["entry_price"] == 150.0
    assert saved["screener_cache"] == screener_cache
    assert saved["screener_timestamp"] == cache_ts

    # Load state from file
    loaded = load_state(state_file)
    assert loaded is not None
    assert loaded["active_positions"]["ETH/USDT"]["amount"] == 0.5
    assert loaded["screener_cache"] == screener_cache


@pytest.mark.asyncio
async def test_bot_multi_slot_rehydration(tmp_path):
    """Verify bot rehydrates multiple active slots directly into monitoring on restart."""
    state_file = str(tmp_path / "rehydrate_state.json")

    positions = {
        "SOL/USDT": {"entry_price": 140.0, "amount": 5.0, "order_id": "sol_recov_1"},
        "ETH/USDT": {"entry_price": 2500.0, "amount": 1.0, "order_id": "eth_recov_2"},
    }
    save_state(
        state=BotState.IN_POSITION,
        active_positions=positions,
        screener_cache=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        screener_timestamp=time.time(),
        file_path=state_file,
    )

    config = Config(mock_exchange=True, poll_interval_seconds=0.01)
    bot = SentinelTrader(config, scenario="default", state_file=state_file)

    # Free balances exist on exchange
    bot.exchange.set_mock_balance("SOL", 5.0)
    bot.exchange.set_mock_balance("ETH", 1.0)
    bot.exchange.set_mock_balance("USDT", 2000.0)

    await bot.initialize()

    # Verify both slots were rehydrated into active monitoring
    assert bot.state == BotState.IN_POSITION
    assert len(bot.risk_engine.active_positions) == 2
    assert "SOL/USDT" in bot.risk_engine.active_positions
    assert "ETH/USDT" in bot.risk_engine.active_positions
    assert bot.risk_engine.active_positions["SOL/USDT"].entry_price == 140.0
    assert bot.risk_engine.active_positions["ETH/USDT"].entry_price == 2500.0


# =====================================================================
# 9. Expanded Stablecoin Blacklist & USD Base Regex Filter Tests
# =====================================================================

def test_screener_expanded_stablecoin_blacklist_constants():
    """Verify that all required stablecoin tickers are present in STABLECOIN_BLACKLIST."""
    required_stables = [
        "USD1/USDT",
        "USDE/USDT",
        "PYUSD/USDT",
        "BFUSD/USDT",
        "FDUSD/USDT",
        "USDD/USDT",
    ]
    for ticker in required_stables:
        assert ticker in STABLECOIN_BLACKLIST, f"Missing {ticker} in STABLECOIN_BLACKLIST"


@pytest.mark.asyncio
async def test_screener_usd_base_regex_filtering():
    """Verify safety regex filter excludes any pair containing 'USD' in base asset unless whitelisted."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            # Stablecoins / USD variants in base asset
            "USD1/USDT": {"symbol": "USD1/USDT", "quoteVolume": 900000000.0, "spot": True, "active": True},
            "USDE/USDT": {"symbol": "USDE/USDT", "quoteVolume": 800000000.0, "spot": True, "active": True},
            "PYUSD/USDT": {"symbol": "PYUSD/USDT", "quoteVolume": 700000000.0, "spot": True, "active": True},
            "BFUSD/USDT": {"symbol": "BFUSD/USDT", "quoteVolume": 600000000.0, "spot": True, "active": True},
            "FDUSD/USDT": {"symbol": "FDUSD/USDT", "quoteVolume": 500000000.0, "spot": True, "active": True},
            "USDD/USDT": {"symbol": "USDD/USDT", "quoteVolume": 400000000.0, "spot": True, "active": True},
            "NEWUSD/USDT": {"symbol": "NEWUSD/USDT", "quoteVolume": 350000000.0, "spot": True, "active": True},
            "USDXYZ/USDT": {"symbol": "USDXYZ/USDT", "quoteVolume": 300000000.0, "spot": True, "active": True},
            # Legitimate assets (quote is USDT, base does not contain USD)
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 200000000.0, "spot": True, "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 100000000.0, "spot": True, "active": True},
            "SOL/USDT": {"symbol": "SOL/USDT", "quoteVolume": 50000000.0, "spot": True, "active": True},
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400)
    universe = await screener.get_dynamic_universe(mock_exchange, top_n=10)

    # All USD-base symbols must be excluded
    excluded = [
        "USD1/USDT", "USDE/USDT", "PYUSD/USDT", "BFUSD/USDT",
        "FDUSD/USDT", "USDD/USDT", "NEWUSD/USDT", "USDXYZ/USDT",
    ]
    for sym in excluded:
        assert sym not in universe, f"{sym} should have been excluded by blacklist/regex filter"

    # Only legitimate crypto pairs should remain
    assert universe == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


@pytest.mark.asyncio
async def test_screener_usd_base_regex_whitelist():
    """Verify that explicitly whitelisted symbols containing 'USD' in base asset are allowed."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "WHITELISTEDUSD/USDT": {"symbol": "WHITELISTEDUSD/USDT", "quoteVolume": 500000000.0, "spot": True, "active": True},
            "NONWHITELISTEDUSD/USDT": {"symbol": "NONWHITELISTEDUSD/USDT", "quoteVolume": 400000000.0, "spot": True, "active": True},
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 300000000.0, "spot": True, "active": True},
        }
    )

    screener = UniverseScreener(cache_ttl_seconds=86400, whitelisted_symbols={"WHITELISTEDUSD/USDT"})
    universe = await screener.get_dynamic_universe(mock_exchange, top_n=10)

    assert "WHITELISTEDUSD/USDT" in universe
    assert "NONWHITELISTEDUSD/USDT" not in universe
    assert "BTC/USDT" in universe
    assert universe == ["WHITELISTEDUSD/USDT", "BTC/USDT"]


# =====================================================================
# 10. Institutional Quant Mechanics Tests
# =====================================================================

def test_chandelier_stop_ratchet_up_never_down():
    """Verify Chandelier Trailing Stop only ratchets UP, never downward on price dips."""
    # Use higher take_profit_pct so chandelier trailing stop dynamics can be tested without fixed TP early-exit
    config = Config(mock_exchange=True, atr_sl_multiplier=1.5, take_profit_pct=0.20)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    # Entry at $100 with ATR = 2.0
    # Initial stop = 100 - (2.0 * 1.5) = 97.0
    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    assert pos.highest_price == 100.0
    assert pos.current_stop_price == 97.0
    assert pos.atr_at_entry == 2.0

    # Tick 1: Price rises to $105.0
    # highest_price -> 105.0
    # trail_target = 105.0 - (2.0 * 2.0) = 101.0
    # current_stop_price = max(97.0, 101.0) -> 101.0 (ratchets UP)
    trigger1 = risk.evaluate_mathematical_limits(105.0, symbol="SOL/USDT")
    assert trigger1 is None
    assert pos.highest_price == 105.0
    assert pos.current_stop_price == 101.0

    # Tick 2: Price pulls back to $103.0
    # highest_price stays 105.0
    # trail_target = 105.0 - 4.0 = 101.0
    # current_stop_price must STAY 101.0 (CANNOT DECREASE)
    trigger2 = risk.evaluate_mathematical_limits(103.0, symbol="SOL/USDT")
    assert trigger2 is None
    assert pos.highest_price == 105.0
    assert pos.current_stop_price == 101.0

    # Tick 3: Price pushes higher to $110.0
    # highest_price -> 110.0
    # trail_target = 110.0 - 4.0 = 106.0
    # current_stop_price = max(101.0, 106.0) -> 106.0 (ratchets UP again)
    trigger3 = risk.evaluate_mathematical_limits(110.0, symbol="SOL/USDT")
    assert trigger3 is None
    assert pos.highest_price == 110.0
    assert pos.current_stop_price == 106.0

    # Tick 4: Price dips to $107.0 (above stop $106.0) -> No breach, stop stays $106.0
    trigger4 = risk.evaluate_mathematical_limits(107.0, symbol="SOL/USDT")
    assert trigger4 is None
    assert pos.current_stop_price == 106.0

    # Tick 5: Price dumps to $105.5 (<= current_stop_price $106.0) -> Triggers CHANDELIER_STOP
    trigger5 = risk.evaluate_mathematical_limits(105.5, symbol="SOL/USDT")
    assert trigger5 == TriggerReason.CHANDELIER_STOP


def test_volatility_parity_sizing_high_vs_low_atr():
    """Verify Volatility Parity sizing: high ATR generates strictly smaller sizing than low ATR."""
    config = Config(
        mock_exchange=True,
        portfolio_risk_per_trade_pct=0.01,  # 1% equity risk
        atr_sl_multiplier=1.5,
    )
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    free_usdt = 10000.0
    total_equity = 10000.0
    entry_price = 100.0

    # High volatility asset: ATR = 5.0
    # stop_dist = 5.0 * 1.5 = 7.5
    # dollar_risk = 10000 * 0.01 = 100.0
    # target_qty = 100.0 / 7.5 = 13.3333
    # notional = 13.3333 * 100 = 1333.33
    cost_high_atr = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=5.0,
        entry_price=entry_price,
        total_equity=total_equity,
    )

    # Low volatility asset: ATR = 1.0
    # stop_dist = 1.0 * 1.5 = 1.5
    # dollar_risk = 100.0
    # target_qty = 100.0 / 1.5 = 66.6667
    # notional = 66.6667 * 100 = 6666.67 -> capped at 33% of free USDT = 3300.0
    cost_low_atr = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=1.0,
        entry_price=entry_price,
        total_equity=total_equity,
    )

    assert cost_high_atr == pytest.approx(1333.33, abs=0.5)
    assert cost_low_atr == pytest.approx(3300.0, abs=0.5)
    assert cost_high_atr < cost_low_atr

    # Verify ExchangeClient implementation matches
    ec_cost_high = exchange.calculate_volatility_parity_sizing(
        total_equity=total_equity,
        free_usdt=free_usdt,
        entry_price=entry_price,
        atr=5.0,
    )
    assert ec_cost_high == pytest.approx(cost_high_atr, abs=1e-4)


def test_alpha_decay_time_stop_triggers_at_candle_16():
    """Verify Alpha Decay time stop triggers liquidation when position is held for 16 candles."""
    config = Config(mock_exchange=True, max_holding_candles=16)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    assert pos.candles_held == 0

    # Simulate candle increments up to 15 candles
    for c in range(1, 16):
        risk.increment_all_candles()
        assert pos.candles_held == c
        # Price hovers around entry (no stop breach)
        trigger = risk.evaluate_mathematical_limits(100.2, symbol="SOL/USDT")
        assert trigger is None, f"Premature trigger at candle {c}"

    # Increment to candle 16 (MAX_HOLDING_CANDLES threshold reached)
    risk.increment_all_candles()
    assert pos.candles_held == 16

    trigger16 = risk.evaluate_mathematical_limits(100.2, symbol="SOL/USDT")
    assert trigger16 == TriggerReason.ALPHA_DECAY_TIME_STOP


def test_statistical_fair_value_target_mean_reversion():
    """Verify Statistical Fair-Value Target triggers when Price >= EMA(20) AND RSI >= 58.0 AND PnL > 0."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)

    # Case 1: In profit (Price $103 > $100), above EMA20 ($102), RSI 60.0 (>= 58.0) -> Trigger
    trig1 = risk.evaluate_mathematical_limits(
        current_price=103.0,
        symbol="SOL/USDT",
        ema20=102.0,
        rsi=60.0,
    )
    assert trig1 == TriggerReason.FAIR_VALUE_REVERSION

    # Case 2: Above EMA20 and in profit, but RSI is only 52.0 (< 58.0) -> No trigger
    trig2 = risk.evaluate_mathematical_limits(
        current_price=103.0,
        symbol="SOL/USDT",
        ema20=102.0,
        rsi=52.0,
    )
    assert trig2 is None

    # Case 3: Fresh position (entry 100, stop 97). Price 98.5 is above stop ($97.0) and above EMA20 ($97.0),
    # RSI >= 58.0, but unrealized PnL is negative ($98.5 < $100.0) -> No trigger
    risk.open_position("ETH/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    trig3 = risk.evaluate_mathematical_limits(
        current_price=98.5,
        symbol="ETH/USDT",
        ema20=97.0,
        rsi=62.0,
    )
    assert trig3 is None


def test_state_store_quant_telemetry_serialization(tmp_path):
    """Verify state.json serializes and deserializes quant fields across restarts."""
    state_file = str(tmp_path / "quant_state.json")

    positions = {
        "SOL/USDT": {
            "entry_price": 150.0,
            "amount": 10.0,
            "order_id": "ord_sol_1",
            "highest_price": 162.5,
            "atr_at_entry": 3.4,
            "current_stop_price": 155.7,
            "candles_held": 8,
        },
    }

    save_state(
        state=BotState.IN_POSITION,
        active_positions=positions,
        file_path=state_file,
    )

    loaded = load_state(state_file)
    assert loaded is not None
    sol = loaded["active_positions"]["SOL/USDT"]
    assert sol["highest_price"] == 162.5
    assert sol["atr_at_entry"] == 3.4
    assert sol["current_stop_price"] == 155.7
    assert sol["candles_held"] == 8


@pytest.mark.asyncio
async def test_quant_liquidation_records_in_trade_history(tmp_path):
    """Verify quantitative liquidation triggers are recorded in trade_history.json."""
    from sentinel_trader.analytics import load_trades

    trade_file = str(tmp_path / "trade_history_test.json")
    config = Config(mock_exchange=True, max_holding_candles=16)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange, trade_history_file=trade_file)

    # 1. Chandelier Stop liquidation
    risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    exchange.set_mock_balance("SOL", 5.0)
    exchange.set_mock_price("SOL/USDT", 95.0)
    rec1 = await risk.execute_liquidation(TriggerReason.CHANDELIER_STOP, notes="Chandelier test", symbol="SOL/USDT")
    assert rec1.trigger == TriggerReason.CHANDELIER_STOP

    # 2. Fair-Value Reversion liquidation
    risk.open_position("ETH/USDT", entry_price=2500.0, amount=1.0, atr=50.0)
    exchange.set_mock_balance("ETH", 1.0)
    exchange.set_mock_price("ETH/USDT", 2580.0)
    rec2 = await risk.execute_liquidation(TriggerReason.FAIR_VALUE_REVERSION, notes="Fair value test", symbol="ETH/USDT")
    assert rec2.trigger == TriggerReason.FAIR_VALUE_REVERSION

    # 3. Alpha Decay Time Stop liquidation
    risk.open_position("BTC/USDT", entry_price=60000.0, amount=0.1, atr=1000.0)
    exchange.set_mock_balance("BTC", 0.1)
    exchange.set_mock_price("BTC/USDT", 60100.0)
    rec3 = await risk.execute_liquidation(TriggerReason.ALPHA_DECAY_TIME_STOP, notes="Alpha decay test", symbol="BTC/USDT")
    assert rec3.trigger == TriggerReason.ALPHA_DECAY_TIME_STOP

    # Verify trade_history records
    trades = load_trades(trade_file)
    assert len(trades) == 3
    reasons = [t["trigger_reason"] for t in trades]
    assert "CHANDELIER_STOP" in reasons
    assert "FAIR_VALUE_REVERSION" in reasons
    assert "ALPHA_DECAY_TIME_STOP" in reasons


# =====================================================================
# 9. Reliability Scoring System & 24-Hour Penalty Box Tests
# =====================================================================

def test_reliability_score_adjustments():
    """Verify reward and penalty score adjustments on trade close matching specification."""
    from sentinel_trader.analytics import calculate_score_adjustment

    # 1. Fast Win (PnL > 0 and candles_held <= 6): S_new = min(S + 0.5, 3.0)
    assert calculate_score_adjustment(1.0, realized_pnl_usd=50.0, realized_pnl_pct=0.03, trigger_reason="TAKE_PROFIT", candles_held=4) == 1.5
    # Capped at 3.0
    assert calculate_score_adjustment(2.8, realized_pnl_usd=50.0, realized_pnl_pct=0.03, trigger_reason="TAKE_PROFIT", candles_held=2) == 3.0

    # 2. Slow Win / Drag (PnL > 0 and candles_held > 10): S_new = S + 0.1
    assert calculate_score_adjustment(1.0, realized_pnl_usd=10.0, realized_pnl_pct=0.01, trigger_reason="TAKE_PROFIT", candles_held=12) == 1.1

    # 3. Time Decay Scratch (ALPHA_DECAY_TIME_STOP or PnL <= 0): S_new = S - 0.3
    # Case A: ALPHA_DECAY_TIME_STOP trigger
    assert calculate_score_adjustment(1.0, realized_pnl_usd=1.0, realized_pnl_pct=0.001, trigger_reason="ALPHA_DECAY_TIME_STOP", candles_held=16) == 0.7
    # Case B: PnL <= 0 (scratch, flat exit)
    assert calculate_score_adjustment(1.0, realized_pnl_usd=0.0, realized_pnl_pct=0.0, trigger_reason="MANUAL", candles_held=5) == 0.7
    # Case C: Mild loss (e.g. -0.5% > -1.5%)
    assert calculate_score_adjustment(1.0, realized_pnl_usd=-5.0, realized_pnl_pct=-0.005, trigger_reason="STOP_LOSS", candles_held=8) == 0.7

    # 4. Hard Stop Loss (CHANDELIER_STOP or PnL < -1.5%): S_new = S - 0.7
    # Case A: CHANDELIER_STOP trigger
    assert calculate_score_adjustment(1.0, realized_pnl_usd=-10.0, realized_pnl_pct=-0.01, trigger_reason="CHANDELIER_STOP", candles_held=3) == 0.3
    # Case B: PnL < -1.5% (e.g. -2.0% loss)
    assert calculate_score_adjustment(1.0, realized_pnl_usd=-20.0, realized_pnl_pct=-0.02, trigger_reason="STOP_LOSS", candles_held=5) == 0.3


def test_reliability_score_decay_until_penalty_box():
    """Verify score decay on successive losses until an asset hits S <= 0.0 and enters 24h penalty box."""
    from sentinel_trader.analytics import update_asset_score

    scores: dict = {}
    t0 = 1700000000.0

    # 1. Initial State (starts at 1.0)
    # Loss 1: Hard Stop Loss (-0.7) -> Score drops to 0.3
    res1 = update_asset_score(
        symbol="SOL/USDT",
        realized_pnl_usd=-25.0,
        realized_pnl_pct=-0.02,  # -2.0% < -1.5%
        trigger_reason="STOP_LOSS",
        candles_held=4,
        current_time=t0,
        asset_scores_dict=scores,
    )
    assert res1["score"] == 0.3
    assert res1["trades_count"] == 1
    assert res1["penalized_until"] is None

    # Loss 2: Time Decay Scratch (-0.3) -> Score drops to 0.0 (S <= 0.0)
    # Triggers 24-hour penalty box!
    res2 = update_asset_score(
        symbol="SOL/USDT",
        realized_pnl_usd=-2.0,
        realized_pnl_pct=-0.002,
        trigger_reason="ALPHA_DECAY_TIME_STOP",
        candles_held=16,
        current_time=t0 + 100,
        asset_scores_dict=scores,
    )
    assert res2["score"] == 0.0
    assert res2["trades_count"] == 2
    assert res2["penalized_until"] is not None
    assert res2["penalized_until"] == (t0 + 100 + 86400.0)


@pytest.mark.asyncio
async def test_penalized_asset_excluded_from_dynamic_universe_even_rank_1(caplog):
    """Verify penalized asset is excluded from get_dynamic_universe even if it ranks #1 by volume."""
    import logging
    caplog.set_level(logging.WARNING)

    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    # SOL/USDT ranks #1 with $1,000,000,000 quote volume!
    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "SOL/USDT": {"symbol": "SOL/USDT", "quoteVolume": 1000000000.0, "spot": True, "active": True},
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 500000000.0, "spot": True, "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 250000000.0, "spot": True, "active": True},
            "NEAR/USDT": {"symbol": "NEAR/USDT", "quoteVolume": 100000000.0, "spot": True, "active": True},
        }
    )

    t_now = 1700000000.0
    # SOL/USDT is currently penalized for 24 hours
    scores = {
        "SOL/USDT": {
            "score": -0.4,
            "penalized_until": t_now + 86400.0,
            "trades_count": 3,
        }
    }

    screener = UniverseScreener(cache_ttl_seconds=86400, asset_scores=scores)
    universe = await screener.get_dynamic_universe(
        mock_exchange,
        top_n=3,
        current_time=t_now,
    )

    # 1. SOL/USDT must NOT be present in universe despite #1 volume
    assert "SOL/USDT" not in universe

    # 2. Remaining pairs must be ranked properly: BTC #1, ETH #2, NEAR #3
    assert universe == ["BTC/USDT", "ETH/USDT", "NEAR/USDT"]

    # 3. Discarded candidate must be logged with specified format:
    # [PENALTY BOX] {symbol} excluded (banned until {iso_time}, Score: {score:.1f})
    expected_iso = datetime.fromtimestamp(t_now + 86400.0, tz=timezone.utc).isoformat()
    expected_log_snippet = f"[PENALTY BOX] SOL/USDT excluded (banned until {expected_iso}, Score: -0.4)"
    assert expected_log_snippet in caplog.text


@pytest.mark.asyncio
async def test_penalized_asset_reinstated_after_24_hours():
    """Verify asset reinstatement after 24 hours: score reset to 0.5 and re-included in universe."""
    mock_exchange = AsyncMock()
    mock_exchange.is_mock = False
    mock_exchange.markets = {}

    mock_exchange.fetch_tickers = AsyncMock(
        return_value={
            "SOL/USDT": {"symbol": "SOL/USDT", "quoteVolume": 1000000000.0, "spot": True, "active": True},
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 500000000.0, "spot": True, "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 250000000.0, "spot": True, "active": True},
        }
    )

    t0 = 1700000000.0
    pen_until = t0 + 86400.0

    scores = {
        "SOL/USDT": {
            "score": -0.4,
            "penalized_until": pen_until,
            "trades_count": 3,
        }
    }

    screener = UniverseScreener(cache_ttl_seconds=86400, asset_scores=scores)

    # At t0 + 12 hours: Still penalized
    t_mid = t0 + 43200.0
    u_mid = await screener.get_dynamic_universe(mock_exchange, top_n=3, current_time=t_mid, force_refresh=True)
    assert "SOL/USDT" not in u_mid
    assert screener.is_asset_penalized("SOL/USDT", current_time=t_mid) is True

    # At t0 + 24 hours + 1 second: Ban expired, asset reinstated!
    t_after = pen_until + 1.0
    u_after = await screener.get_dynamic_universe(mock_exchange, top_n=3, current_time=t_after, force_refresh=True)

    # Reinstated checks:
    # 1. Re-included in dynamic universe at #1 spot based on quoteVolume
    assert "SOL/USDT" in u_after
    assert u_after[0] == "SOL/USDT"

    # 2. is_asset_penalized returns False
    assert screener.is_asset_penalized("SOL/USDT", current_time=t_after) is False

    # 3. Score baseline reset to 0.5 and penalized_until cleared to None
    sol_entry = screener.get_asset_score("SOL/USDT", current_time=t_after)
    assert sol_entry["score"] == 0.5
    assert sol_entry["penalized_until"] is None
    assert sol_entry["trades_count"] == 3


def test_asset_scores_atomic_persistence_and_unban_on_startup(tmp_path):
    """Verify atomic persistence to asset_scores.json and automatic unbanning of expired entries on startup."""
    from sentinel_trader.state_store import clear_asset_scores, load_asset_scores, save_asset_scores

    scores_file = str(tmp_path / "asset_scores.json")
    t0 = 1700000000.0

    initial_scores = {
        "SOL/USDT": {
            "score": -0.4,
            "penalized_until": t0 + 3600.0,  # Expired after 1 hour
            "trades_count": 5,
        },
        "DOGE/USDT": {
            "score": -0.7,
            "penalized_until": t0 + 86400.0,  # Active for 24 hours
            "trades_count": 2,
        },
        "BTC/USDT": {
            "score": 2.0,
            "penalized_until": None,
            "trades_count": 4,
        },
    }

    # 1. Atomically save scores to disk
    save_asset_scores(initial_scores, file_path=scores_file)

    # 2. Simulate startup 2 hours later (t0 + 7200s):
    # SOL/USDT penalty has expired -> should automatically unban and reset score to 0.5
    # DOGE/USDT penalty is still active -> remains penalized
    # BTC/USDT -> unchanged
    t_startup = t0 + 7200.0
    loaded = load_asset_scores(file_path=scores_file, current_time=t_startup)

    # Verify SOL/USDT automatically unbanned
    assert loaded["SOL/USDT"]["penalized_until"] is None
    assert loaded["SOL/USDT"]["score"] == 0.5
    assert loaded["SOL/USDT"]["trades_count"] == 5

    # Verify DOGE/USDT remains penalized
    assert loaded["DOGE/USDT"]["penalized_until"] == t0 + 86400.0
    assert loaded["DOGE/USDT"]["score"] == -0.7

    # Verify BTC/USDT unchanged
    assert loaded["BTC/USDT"]["score"] == 2.0
    assert loaded["BTC/USDT"]["penalized_until"] is None

    # Clear file
    assert clear_asset_scores(scores_file) is True
    assert load_asset_scores(scores_file) == {}


@pytest.mark.asyncio
async def test_risk_engine_record_trade_score_integration(tmp_path):
    """Verify RiskEngine liquidation passes candles_held to record_trade and updates asset scores."""
    from sentinel_trader.state_store import load_asset_scores

    trade_file = str(tmp_path / "trade_history.json")
    scores_file = str(tmp_path / "asset_scores.json")

    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(
        config,
        exchange,
        trade_history_file=trade_file,
        asset_scores_file=scores_file,
    )

    # Open position with 4 candles held
    risk.open_position("SOL/USDT", entry_price=100.0, amount=2.0)
    risk.positions["SOL/USDT"].candles_held = 4
    exchange.set_mock_balance("SOL", 2.0)
    exchange.set_mock_price("SOL/USDT", 110.0)  # +10% gain -> Fast Win (candles_held <= 6)

    # Execute Take Profit liquidation
    rec = await risk.execute_liquidation(TriggerReason.TAKE_PROFIT, notes="Fast win test", symbol="SOL/USDT")
    assert rec.trigger == TriggerReason.TAKE_PROFIT

    # Verify asset score updated to 1.0 + 0.5 = 1.5
    scores = load_asset_scores(scores_file)
    assert "SOL/USDT" in scores
    assert scores["SOL/USDT"]["score"] == 1.5
    assert scores["SOL/USDT"]["trades_count"] == 1
    assert scores["SOL/USDT"]["penalized_until"] is None

    # Next trade on SOL/USDT: Chandelier Stop Loss
    risk.open_position("SOL/USDT", entry_price=110.0, amount=2.0)
    risk.positions["SOL/USDT"].candles_held = 3
    exchange.set_mock_balance("SOL", 2.0)
    exchange.set_mock_price("SOL/USDT", 100.0)

    # Execute Chandelier Stop liquidation -> Hard Stop Loss (-0.7)
    rec2 = await risk.execute_liquidation(TriggerReason.CHANDELIER_STOP, notes="Hard stop test", symbol="SOL/USDT")
    assert rec2.trigger == TriggerReason.CHANDELIER_STOP

    # Verify asset score dropped to 1.5 - 0.7 = 0.8
    scores2 = load_asset_scores(scores_file)
    assert scores2["SOL/USDT"]["score"] == 0.8
    assert scores2["SOL/USDT"]["trades_count"] == 2


def test_trailing_stop_tightening_at_candle_5():
    """Verify Chandelier stop multiplier tightens from 2.0*ATR (Phase A) to 1.25*ATR (Phase B) at candle 5."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    # Entry stop was 100.0 - (2.0 * 1.5) = 97.0
    assert pos.candles_held == 0
    assert pos.exit_phase == "MOMENTUM"

    # Candle 1 to 4: trailing stop at highest_price - 2.0 * atr = 100 - 4.0 = 96.0 (stop stays 97.0)
    for c in range(1, 5):
        risk.increment_all_candles()
        assert pos.candles_held == c
        assert pos.exit_phase == "MOMENTUM"
        risk.evaluate_mathematical_limits(100.0, symbol="SOL/USDT")
        assert pos.current_stop_price == 97.0

    # Candle 5: Momentum Stagnation (Phase B)
    # Trailing distance ratchets inward to highest_price - (atr_at_entry * 1.25)
    # 100.0 - (2.0 * 1.25) = 100.0 - 2.5 = 97.5
    # Stop ratchets up from 97.0 to 97.5
    risk.increment_all_candles()
    assert pos.candles_held == 5
    assert pos.exit_phase == "STAGNATION"
    trigger = risk.evaluate_mathematical_limits(100.0, symbol="SOL/USDT")
    assert trigger is None
    assert pos.current_stop_price == 97.5


def test_time_decay_scratch_triggers_at_candle_11():
    """Verify TIME_DECAY_SCRATCH triggers at candle 11 when PnL is +0.6% (+0.006 >= +0.005 hurdle)."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    pos.candles_held = 11
    assert pos.exit_phase == "SCRATCH"

    # Price = 100.60 -> PnL = +0.6% (+0.006 >= +0.005 hurdle)
    trigger = risk.evaluate_mathematical_limits(100.60, symbol="SOL/USDT")
    assert trigger == TriggerReason.TIME_DECAY_SCRATCH


def test_no_premature_scratch_candles_1_to_10_when_price_above_entry():
    """Verify no premature scratch occurs during Candles 1 to 10 if price is above entry (+0.6%)."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)

    # For candles 1 to 10, with price at $100.60 (+0.6% PnL)
    for c in range(1, 11):
        pos.candles_held = c
        trigger = risk.evaluate_mathematical_limits(100.60, symbol="SOL/USDT")
        assert trigger is None, f"Premature scratch triggered at candle {c}"
        if c <= 4:
            assert pos.exit_phase == "MOMENTUM"
        else:
            assert pos.exit_phase == "STAGNATION"

    # At candle 11, it MUST trigger TIME_DECAY_SCRATCH
    pos.candles_held = 11
    assert pos.exit_phase == "SCRATCH"
    trigger11 = risk.evaluate_mathematical_limits(100.60, symbol="SOL/USDT")
    assert trigger11 == TriggerReason.TIME_DECAY_SCRATCH


def test_phase_c_breakeven_stop_ratchet():
    """Verify Phase C ratchets stop price to entry_price * 1.001."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)
    pos.candles_held = 11
    assert pos.exit_phase == "SCRATCH"

    # Price = 100.30 (+0.3% < +0.5% hurdle, so no scratch)
    trigger = risk.evaluate_mathematical_limits(100.30, symbol="SOL/USDT")
    assert trigger is None
    # Stop should be ratcheted to at least entry_price * 1.001 = 100.10
    assert pos.current_stop_price >= 100.10

    # If price drops to 100.05 (<= 100.10), Chandelier stop triggers at breakeven
    trigger_stop = risk.evaluate_mathematical_limits(100.05, symbol="SOL/USDT")
    assert trigger_stop == TriggerReason.CHANDELIER_STOP


def test_phase_b_fair_value_rsi_hurdle_lowered_to_52():
    """Verify Phase B allows modest bounce exit with RSI >= 52.0 (Phase A requires RSI >= 58.0)."""
    config = Config(mock_exchange=True)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange)

    pos = risk.open_position("SOL/USDT", entry_price=100.0, amount=5.0, atr=2.0)

    # Candle 3 (Phase A): price 102.0 > ema20 101.0, rsi 54.0 (< 58.0) -> No trigger
    pos.candles_held = 3
    trig_a = risk.evaluate_mathematical_limits(102.0, symbol="SOL/USDT", ema20=101.0, rsi=54.0)
    assert trig_a is None

    # Candle 6 (Phase B): price 102.0 > ema20 101.0, rsi 54.0 (>= 52.0) -> Triggers FAIR_VALUE_REVERSION
    pos.candles_held = 6
    trig_b = risk.evaluate_mathematical_limits(102.0, symbol="SOL/USDT", ema20=101.0, rsi=54.0)
    assert trig_b == TriggerReason.FAIR_VALUE_REVERSION




