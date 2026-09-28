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

