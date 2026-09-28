"""Unit and integration tests for Dual-Client Architecture, Diagnostic Scan Logging, and Sensitivity Calibration.

Verifies:
1. Dual-Client Architecture:
   - data_exchange: public, unauthenticated ccxt.binance() pointing to live production endpoints (set_sandbox_mode(False)).
     Exclusively routes fetch_tickers(), fetch_ohlcv(), fetch_ohlcv_dataframe(), get_ticker_price().
   - order_exchange: authenticated client configured via .env (points to Testnet when BINANCE_SANDBOX=True).
     Exclusively routes get_balance(), execute_market_buy_idempotent(), liquidate_to_usdt().
   - Graceful close() closes both clients.
2. Diagnostic Scan Logging:
   - Evaluates candle closes for all candidates.
   - Generates compact diagnostic lines:
     * Price < EMA: [REJECT: Below EMA{ema_period}]
     * Price > EMA but RSI not crossing: [REJECT: RSI not crossing {rsi_threshold}]
     * Price > EMA and RSI crossed: [PASS: Signal Found]
   - When no pairs trigger: [SCAN RESULT] 0/{N} passed entry criteria. Maintaining 100% Cash.
   - CandidateList container exposes .diagnostics and .summary.
3. Sensitivity Calibration:
   - Config defaults: RSI_OVERSOLD_THRESHOLD = 38.0, EMA_PERIOD = 100.
   - Input validation bounds for both parameters.
   - Indicator and entry signal evaluation with custom sensitivity parameters.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.strategy import (
    Candidate,
    CandidateList,
    calculate_indicators,
    calculate_rsi,
    calculate_ema,
    check_entry_signal,
    format_price,
    scan_universe_for_signals,
)


# =====================================================================
# 1. Dual-Client Architecture Tests
# =====================================================================

@pytest.mark.asyncio
async def test_dual_client_initialization_sandbox():
    """Verify dual-client architecture separates data_exchange (live, unauthenticated) and order_exchange (testnet, authenticated)."""
    config = Config(
        mock_exchange=False,
        binance_api_key="mock_testnet_key",
        binance_api_secret="mock_testnet_secret",
        binance_sandbox=True,
    )
    client = ExchangeClient(config)

    with patch("ccxt.async_support.binance") as mock_ccxt_class:
        mock_data_inst = AsyncMock()
        mock_data_inst.set_sandbox_mode = MagicMock()
        mock_order_inst = AsyncMock()
        mock_order_inst.set_sandbox_mode = MagicMock()
        mock_ccxt_class.side_effect = [mock_data_inst, mock_order_inst]

        await client.initialize()

        assert client.data_exchange is mock_data_inst
        assert client.order_exchange is mock_order_inst
        assert client.is_mock is False

        # Verify data_exchange is unauthenticated and sandbox mode is explicitly False
        mock_data_inst.set_sandbox_mode.assert_called_with(False)

        # Verify order_exchange is authenticated (apiKey & secret passed) and sandbox mode is True
        mock_order_inst.set_sandbox_mode.assert_called_with(True)

        # Verify both loaded markets
        mock_data_inst.load_markets.assert_awaited_once()
        mock_order_inst.load_markets.assert_awaited_once()


@pytest.mark.asyncio
async def test_dual_client_market_data_routing():
    """Verify market data calls route strictly to data_exchange."""
    config = Config(
        mock_exchange=False,
        binance_api_key="mock_key",
        binance_api_secret="mock_sec",
        binance_sandbox=True,
    )
    client = ExchangeClient(config)

    mock_data = AsyncMock()
    mock_order = AsyncMock()

    mock_data.fetch_ticker.return_value = {"last": 152.5}
    mock_data.fetch_tickers.return_value = {"SOL/USDT": {"symbol": "SOL/USDT", "last": 152.5}}
    mock_data.fetch_ohlcv.return_value = [
        [1600000000000, 150.0, 153.0, 149.0, 152.5, 1000.0],
    ]
    mock_data.markets = {"SOL/USDT": {"id": "SOLUSDT"}}

    client.data_exchange = mock_data
    client.order_exchange = mock_order
    client._is_initialized = True

    # 1. get_ticker_price routes to data_exchange
    price = await client.get_ticker_price("SOL/USDT")
    assert price == 152.5
    mock_data.fetch_ticker.assert_awaited_once_with("SOL/USDT")
    mock_order.fetch_ticker.assert_not_called()

    # 2. fetch_tickers routes to data_exchange
    tickers = await client.fetch_tickers(["SOL/USDT"])
    assert "SOL/USDT" in tickers
    mock_data.fetch_tickers.assert_awaited_once()
    mock_order.fetch_tickers.assert_not_called()

    # 3. fetch_ohlcv routes to data_exchange
    candles = await client.fetch_ohlcv("SOL/USDT", timeframe="15m", limit=10)
    assert len(candles) == 1
    mock_data.fetch_ohlcv.assert_awaited_once()
    mock_order.fetch_ohlcv.assert_not_called()

    # 4. fetch_ohlcv_dataframe routes to data_exchange
    df = await client.fetch_ohlcv_dataframe("SOL/USDT", timeframe="15m", limit=10)
    assert not df.empty
    assert df["close"].iloc[0] == 152.5

    # 5. markets property prefers data_exchange
    assert client.markets == {"SOL/USDT": {"id": "SOLUSDT"}}


@pytest.mark.asyncio
async def test_dual_client_order_execution_routing():
    """Verify balance queries and order placement route strictly to order_exchange."""
    config = Config(
        mock_exchange=False,
        binance_api_key="mock_key",
        binance_api_secret="mock_sec",
        binance_sandbox=True,
    )
    client = ExchangeClient(config)

    mock_data = AsyncMock()
    mock_order = AsyncMock()

    # Setup mock data for data_exchange
    mock_data.fetch_ticker.return_value = {"last": 150.0}

    # Setup mock order execution for order_exchange
    mock_order.fetch_balance.return_value = {"free": {"USDT": 1000.0, "SOL": 5.0}}
    mock_order.market = MagicMock(return_value={
        "symbol": "SOL/USDT",
        "precision": {"amount": 0.01, "price": 0.01},
        "limits": {
            "amount": {"min": 0.01, "max": 1000.0},
            "cost": {"min": 5.0, "max": 100000.0},
        },
    })
    mock_order.price_to_precision = MagicMock(side_effect=lambda s, p: str(round(p, 2)))
    mock_order.amount_to_precision = MagicMock(side_effect=lambda s, a: str(round(a, 2)))
    mock_order.precisionMode = 4
    mock_order.paddingMode = 0
    mock_order.create_market_buy_order = AsyncMock(return_value={
        "id": "buy_ord_1",
        "clientOrderId": "sentinel_buy_1",
        "status": "closed",
        "filled": 1.0,
        "price": 150.0,
        "cost": 150.0,
    })
    mock_order.fetch_order_book = AsyncMock(return_value={"bids": [[149.5, 10.0]], "asks": [[150.5, 10.0]]})
    mock_order.create_order = AsyncMock(return_value={
        "id": "sell_ord_1",
        "clientOrderId": "sentinel_sell_1",
        "status": "closed",
        "filled": 1.0,
        "price": 149.0,
        "cost": 149.0,
    })

    client.data_exchange = mock_data
    client.order_exchange = mock_order
    client._is_initialized = True

    # 1. get_balance routes strictly to order_exchange
    bal = await client.get_balance("USDT")
    assert bal == 1000.0
    mock_order.fetch_balance.assert_awaited_once()
    mock_data.fetch_balance.assert_not_called()

    # 2. execute_market_buy_idempotent routes buy order to order_exchange
    buy_res = await client.execute_market_buy_idempotent("SOL/USDT", cost_usdt=150.0)
    assert buy_res["status"] == "closed"
    mock_order.create_market_buy_order.assert_awaited_once()
    mock_data.create_market_buy_order.assert_not_called()

    # 3. liquidate_to_usdt routes sell order to order_exchange
    sell_res = await client.liquidate_to_usdt("SOL/USDT", amount=1.0)
    assert sell_res.status == "closed"
    mock_order.create_order.assert_awaited_once()
    mock_data.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_dual_client_graceful_close():
    """Verify close() terminates both data_exchange and order_exchange async sessions."""
    config = Config(mock_exchange=False, binance_api_key="k", binance_api_secret="s")
    client = ExchangeClient(config)

    mock_data = AsyncMock()
    mock_order = AsyncMock()
    client.data_exchange = mock_data
    client.order_exchange = mock_order
    client._is_initialized = True

    await client.close()

    mock_data.close.assert_awaited_once()
    mock_order.close.assert_awaited_once()
    assert client.data_exchange is None
    assert client.order_exchange is None
    assert client._is_initialized is False


# =====================================================================
# 2. Diagnostic Scanner Logging Tests
# =====================================================================

def test_format_price_helper():
    """Verify format_price formats integers and decimals cleanly matching diagnostic specs."""
    assert format_price(64120.0) == "$64,120"
    assert format_price(64800.0) == "$64,800"
    assert format_price(2640.0) == "$2,640"
    assert format_price(2580.0) == "$2,580"
    assert format_price(152.0) == "$152"
    assert format_price(148.0) == "$148"
    assert format_price(15.25) == "$15.25"
    assert format_price(0.0456) == "$0.0456"


@pytest.mark.asyncio
async def test_scan_universe_diagnostic_logging(caplog):
    """Verify scan_universe_for_signals produces the required diagnostic format and summary."""
    caplog.set_level(logging.INFO)

    def make_ohlcv(close: float, ema: float, rsi_prev: float, rsi_curr: float) -> pd.DataFrame:
        n = 220
        df = pd.DataFrame({
            "timestamp": [1600000000000 + (i * 900000) for i in range(n)],
            "open": [close] * n,
            "high": [close + 1.0] * n,
            "low": [close - 1.0] * n,
            "close": [close] * n,
            "volume": [1000.0] * n,
        })
        calculate_indicators(df, ema_period=100)
        df.loc[df.index[-3], "rsi"] = rsi_prev
        df.loc[df.index[-2], "rsi"] = rsi_curr
        df.loc[df.index[-2], "close"] = close
        df.loc[df.index[-2], "ema"] = ema
        return df

    # Pair 1: Below EMA (BTC: Price $64,120 < EMA $64,800)
    df_btc = make_ohlcv(close=64120.0, ema=64800.0, rsi_prev=45.0, rsi_curr=48.0)
    # Pair 2: Above EMA, but RSI not crossing (ETH: Price $2,640 > EMA $2,580 | RSI 52.4)
    df_eth = make_ohlcv(close=2640.0, ema=2580.0, rsi_prev=50.0, rsi_curr=52.4)
    # Pair 3: Above EMA, RSI crossed upward (SOL: Price $152 > EMA $148 | RSI 39.5 was 32.4)
    df_sol = make_ohlcv(close=152.0, ema=148.0, rsi_prev=32.4, rsi_curr=39.5)

    exchange = AsyncMock()
    async def mock_ohlcv(sym, timeframe="15m", limit=250):
        if sym == "BTC/USDT":
            return df_btc
        elif sym == "ETH/USDT":
            return df_eth
        else:
            return df_sol

    exchange.fetch_ohlcv_dataframe = AsyncMock(side_effect=mock_ohlcv)

    candidates = await scan_universe_for_signals(
        exchange,
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        rsi_threshold=38.0,
        ema_period=100,
    )

    # CandidateList type and contents
    assert isinstance(candidates, CandidateList)
    assert len(candidates) == 1
    assert candidates[0].symbol == "SOL/USDT"

    # Verify diagnostics list attached
    assert len(candidates.diagnostics) == 3

    diag_btc = candidates.diagnostics[0]
    assert diag_btc["symbol"] == "BTC/USDT"
    assert diag_btc["passed"] is False
    assert "Price $64,120 < EMA $64,800 [REJECT: Below EMA100]" in diag_btc["line"]

    diag_eth = candidates.diagnostics[1]
    assert diag_eth["symbol"] == "ETH/USDT"
    assert diag_eth["passed"] is False
    assert "Price $2,640 > EMA $2,580 | RSI 52.4 [REJECT: RSI not crossing 38]" in diag_eth["line"]

    diag_sol = candidates.diagnostics[2]
    assert diag_sol["symbol"] == "SOL/USDT"
    assert diag_sol["passed"] is True
    assert "Price $152 > EMA $148 | RSI 39.5 (was 32.4) [PASS: Signal Found]" in diag_sol["line"]

    # Verify log output matches specification
    log_text = caplog.text
    assert "[SCAN DIAGNOSTICS] 3 pairs evaluated:" in log_text
    assert "BTC/USDT: Price $64,120 < EMA $64,800 [REJECT: Below EMA100]" in log_text
    assert "ETH/USDT: Price $2,640 > EMA $2,580 | RSI 52.4 [REJECT: RSI not crossing 38]" in log_text
    assert "SOL/USDT: Price $152 > EMA $148 | RSI 39.5 (was 32.4) [PASS: Signal Found]" in log_text
    assert "[SCAN RESULT] 1/3 passed entry criteria." in log_text


@pytest.mark.asyncio
async def test_scan_universe_zero_passed_logging(caplog):
    """Verify '[SCAN RESULT] 0/N passed entry criteria. Maintaining 100% Cash.' is logged when no pairs trigger."""
    caplog.set_level(logging.INFO)

    def make_rejected_df():
        n = 220
        df = pd.DataFrame({
            "timestamp": [1600000000000 + (i * 900000) for i in range(n)],
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        calculate_indicators(df, ema_period=100)
        df.loc[df.index[-3], "rsi"] = 50.0
        df.loc[df.index[-2], "rsi"] = 52.0
        df.loc[df.index[-2], "close"] = 90.0
        df.loc[df.index[-2], "ema"] = 100.0
        return df

    exchange = AsyncMock()
    exchange.fetch_ohlcv_dataframe = AsyncMock(return_value=make_rejected_df())

    candidates = await scan_universe_for_signals(
        exchange,
        ["BTC/USDT", "ETH/USDT"],
        rsi_threshold=38.0,
        ema_period=100,
    )

    assert len(candidates) == 0
    assert "[SCAN RESULT] 0/2 passed entry criteria. Maintaining 100% Cash." in caplog.text
    assert "[SCAN RESULT] 0/2 passed entry criteria. Maintaining 100% Cash." in candidates.summary


# =====================================================================
# 3. Sensitivity Calibration Tests
# =====================================================================

def test_config_sensitivity_defaults():
    """Verify default values for RSI_OVERSOLD_THRESHOLD and EMA_PERIOD."""
    config = Config(mock_exchange=True)
    assert config.rsi_oversold_threshold == 38.0
    assert config.ema_period == 100


def test_config_sensitivity_overrides_and_validation():
    """Verify custom overrides and field validation constraints."""
    custom = Config(mock_exchange=True, rsi_oversold_threshold=35.0, ema_period=200)
    assert custom.rsi_oversold_threshold == 35.0
    assert custom.ema_period == 200

    # Bounds validation: RSI must be in (0, 100)
    with pytest.raises(ValueError, match="rsi_oversold_threshold"):
        Config(mock_exchange=True, rsi_oversold_threshold=0.0)

    with pytest.raises(ValueError, match="rsi_oversold_threshold"):
        Config(mock_exchange=True, rsi_oversold_threshold=100.0)

    # Bounds validation: EMA_PERIOD must be strictly positive
    with pytest.raises(ValueError, match="ema_period"):
        Config(mock_exchange=True, ema_period=0)


def test_calculate_indicators_ema_period_support():
    """Verify calculate_indicators computes requested EMA period and provides backwards compatibility."""
    prices = [100.0 + i for i in range(150)]
    df = pd.DataFrame({"close": prices})

    res = calculate_indicators(df, ema_period=100)

    assert "ema" in df.columns
    assert "ema100" in df.columns
    assert "ema200" in df.columns
    assert res["ema"] is not None
    assert res["ema100"] is not None
    assert res["ema200"] is not None


def test_check_entry_signal_custom_sensitivity():
    """Verify check_entry_signal respects custom rsi_threshold and ema_period."""
    prices = [100.0] * 150
    df = pd.DataFrame({"close": prices})
    calculate_indicators(df, ema_period=100)

    # RSI crosses 38 upward (36.0 -> 39.0) and Close > EMA
    df.loc[df.index[-3], "rsi"] = 36.0
    df.loc[df.index[-2], "rsi"] = 39.0
    df.loc[df.index[-2], "close"] = 105.0
    df.loc[df.index[-2], "ema"] = 100.0

    # Under 38 threshold: True
    assert check_entry_signal(df, rsi_threshold=38.0, ema_period=100) is True

    # Under higher 40 threshold: False (39.0 is not > 40.0)
    assert check_entry_signal(df, rsi_threshold=40.0, ema_period=100) is False
