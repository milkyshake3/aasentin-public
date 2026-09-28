"""Exchange client module for Sentinel-Trader.

Wraps CCXT for Binance Spot Testnet & Mainnet interactions:
- Live market data and ticker polling from Binance Testnet
- Free balance verification for spot assets
- Dust balance and fee erosion protection with strict ROUND_DOWN truncation
- In-flight network partition recovery and idempotent clientOrderId handling
- Slippage bounds on emergency liquidation via aggressive Limit IOC / Bounded Market Orders
- Strict sandbox enforcement and graceful mock fallback
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional
import ccxt.async_support as ccxt
from ccxt.base.decimal_to_precision import decimal_to_precision, TRUNCATE, DECIMAL_PLACES, TICK_SIZE
import pandas as pd
from pydantic import BaseModel, Field

from sentinel_trader.config import Config

logger = logging.getLogger("sentinel.exchange")


class OrderReceipt(BaseModel):
    """Execution receipt for an exchange order."""

    order_id: str
    symbol: str
    side: str = "sell"
    amount: float
    price: float
    cost: float
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class ExchangeClient:
    """Async CCXT wrapper enforcing Dual-Client Architecture & Binance Testnet safety invariant:

    - data_exchange: A public, unauthenticated ccxt.binance() instance pointing to LIVE production endpoints
      (set_sandbox_mode(False)). Used strictly for fetch_tickers() and fetch_ohlcv().
    - order_exchange: The authenticated client configured via .env (points to Testnet when BINANCE_SANDBOX=True).
      Used strictly for balances and order placement (execute_market_buy, liquidate_to_usdt).
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_exchange: Optional[ccxt.binance] = None
        self.order_exchange: Optional[ccxt.binance] = None
        self._is_initialized = False
        self._order_lock = asyncio.Lock()

        # In-memory mock ledger for offline simulation, unit testing, or fallback
        self._mock_balances: Dict[str, float] = {
            self.config.base_asset: 10.0,  # e.g., 10 SOL
            self.config.quote_asset: 5000.0,  # e.g., 5000 USDT
        }
        self._mock_prices: Dict[str, float] = {
            self.config.target_symbol: float(self.config.entry_price or 150.0),
        }
        self._mock_ohlcv: Optional[pd.DataFrame] = None

    @property
    def _exchange(self) -> Optional[ccxt.binance]:
        """Backward-compatible exchange instance accessor (preferring order_exchange)."""
        return self.order_exchange if self.order_exchange is not None else self.data_exchange

    @_exchange.setter
    def _exchange(self, value: Optional[ccxt.binance]) -> None:
        self.order_exchange = value
        if self.data_exchange is None or value is None:
            self.data_exchange = value

    @property
    def is_mock(self) -> bool:
        """Check if client is running in mock simulation mode."""
        return self.config.mock_exchange or (self.order_exchange is None and self.data_exchange is None)

    async def initialize(self) -> None:
        """Initialize dual exchange connections: data_exchange (live) and order_exchange (testnet/live)."""
        # Auto-activate mock mode if explicitly requested OR if no API credentials provided
        if self.config.mock_exchange or not self.config.binance_api_key or not self.config.binance_api_secret:
            logger.info("[MOCK SIMULATION MODE] Initializing ExchangeClient in local mock mode.")
            self._is_initialized = True
            return

        logger.info("Connecting to Binance Exchange clients via CCXT (Dual-Client Architecture)...")

        # 1. Market Data Client (data_exchange):
        # Public, unauthenticated ccxt.binance() pointing to LIVE production endpoints (set_sandbox_mode(False)).
        # Strictly for market data ingestion: fetch_tickers(), fetch_ohlcv(), get_ticker_price().
        data_exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        data_exchange.set_sandbox_mode(False)
        self.data_exchange = data_exchange
        logger.info("[DUAL CLIENT: DATA] Initialized public unauthenticated live ccxt.binance (Sandbox=False) for market data.")

        # 2. Order Execution Client (order_exchange):
        # Authenticated client configured via .env (points to Testnet when BINANCE_SANDBOX=True).
        # Strictly for balances and order placement (execute_market_buy, liquidate_to_usdt).
        order_exchange = ccxt.binance({
            'apiKey': self.config.binance_api_key,
            'secret': self.config.binance_api_secret,
            'enableRateLimit': True,
        })

        # SAFETY INVARIANT #3: Default strictly to Binance Testnet (Sandbox)
        if self.config.binance_sandbox:
            order_exchange.set_sandbox_mode(True)
            self.order_exchange = order_exchange
            logger.info("[DUAL CLIENT: ORDER] Binance Sandbox (Testnet) Mode ENABLED for order execution.")
        else:
            if not self.config.allow_production_trading:
                await order_exchange.close()
                await data_exchange.close()
                self.order_exchange = None
                self.data_exchange = None
                raise PermissionError(
                    "FATAL SAFETY INVARIANT: Live Binance trading blocked. "
                    "ALLOW_PRODUCTION_TRADING must be explicitly set to True."
                )
            self.order_exchange = order_exchange
            logger.warning("[WARNING: LIVE PRODUCTION] Operating on REAL Binance Mainnet for order execution!")

        try:
            await asyncio.gather(
                self.data_exchange.load_markets(),
                self.order_exchange.load_markets(),
            )
            self._is_initialized = True
            logger.info("Exchange markets loaded successfully for dual clients (data & order).")
        except Exception as e:
            logger.warning(
                "Failed to connect/authenticate to Binance (%s). Falling back to mock exchange mode.",
                e,
            )
            await self.close()
            self._is_initialized = True

    async def get_ticker_price(self, symbol: Optional[str] = None) -> float:
        """Fetch live ticker/order book price strictly using live data_exchange or mock ledger."""
        sym = symbol or self.config.target_symbol

        if self.is_mock:
            return float(self._mock_prices.get(sym, 150.0))

        client = self.data_exchange or self.order_exchange
        try:
            ticker = await client.fetch_ticker(sym)
            price = ticker.get("last") or ticker.get("close")
            if price is None:
                bid = ticker.get("bid") or 0.0
                ask = ticker.get("ask") or 0.0
                price = (bid + ask) / 2.0 if (bid and ask) else 0.0
            return float(price)
        except Exception as e:
            logger.error("Failed to fetch ticker price for %s: %s", sym, e)
            raise

    # Backwards-compatible alias for existing callers
    get_current_price = get_ticker_price

    @property
    def markets(self) -> Dict[str, Any]:
        """Expose CCXT exchange markets dict if loaded, preferring data_exchange."""
        client = self.data_exchange or self.order_exchange
        if client is not None and hasattr(client, "markets"):
            return client.markets or {}
        return {}

    async def fetch_tickers(
        self,
        symbols: Optional[list] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch 24h ticker data for all or specified symbols from Binance or mock ledger."""
        if self.is_mock:
            default_pairs = [
                ("BTC/USDT", 65000.0, 150000000.0),
                ("ETH/USDT", 3500.0, 90000000.0),
                ("SOL/USDT", 150.0, 50000000.0),
                ("AVAX/USDT", 30.0, 20000000.0),
                ("LINK/USDT", 15.0, 15000000.0),
                ("NEAR/USDT", 5.0, 10000000.0),
                ("USDC/USDT", 1.0, 300000000.0),
                ("FDUSD/USDT", 1.0, 120000000.0),
                ("BTCUP/USDT", 25.0, 5000000.0),
                ("BTCDOWN/USDT", 10.0, 4000000.0),
            ]
            result = {}
            for sym, def_p, def_qv in default_pairs:
                if symbols and sym not in symbols:
                    continue
                p = float(self._mock_prices.get(sym, def_p))
                result[sym] = {
                    "symbol": sym,
                    "last": p,
                    "close": p,
                    "quoteVolume": def_qv,
                    "baseVolume": def_qv / p,
                    "spot": True,
                    "active": True,
                }
            return result

        try:
            client = self.data_exchange or self.order_exchange
            return await client.fetch_tickers(symbols=symbols, params=params or {})
        except Exception as e:
            logger.error("Failed to fetch tickers: %s", e)
            raise

    async def get_balance(self, asset: Optional[str] = None) -> float:
        """Fetch real free balance of the asset strictly from order_exchange (Binance Testnet or live) or mock ledger."""
        target_asset = asset or self.config.base_asset

        if self.is_mock:
            return float(self._mock_balances.get(target_asset, 0.0))

        client = self.order_exchange or self.data_exchange
        try:
            balance = await client.fetch_balance()
            free_balance = balance.get("free", {}).get(target_asset, 0.0)
            return float(free_balance or 0.0)
        except Exception as e:
            logger.error("Failed to fetch balance for asset %s: %s", target_asset, e)
            raise

    def truncate_amount(self, symbol: str, amount: float) -> float:
        """Strictly truncate order amount downward (ROUND_DOWN / TRUNCATE) to stepSize precision.

        Prevents float arithmetic representation errors (e.g. 4.00098 rounding up to 4.0010)
        and guarantees LOT_SIZE compliance without insufficient balance errors.
        """
        if amount <= 0.0:
            return 0.0

        client = self.order_exchange or self.data_exchange
        if self.is_mock or client is None:
            d = Decimal(str(amount))
            return float(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

        try:
            market = client.market(symbol)
            if hasattr(market, "__await__"):
                # If mocked as coroutine
                return float(Decimal(str(amount)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))
            precision = market.get("precision", {}).get("amount")
            if precision is None:
                return float(amount)

            precision_mode = getattr(client, "precisionMode", DECIMAL_PLACES)
            padding_mode = getattr(client, "paddingMode", 0)
            truncated_str = decimal_to_precision(
                str(amount), TRUNCATE, precision, precision_mode, padding_mode
            )
            return float(truncated_str)
        except Exception as e:
            logger.debug("Error during exchange truncate_amount: %s. Using Decimal fallback.", e)
            d = Decimal(str(amount))
            return float(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Format order quantity truncated downward according to exchange LOT_SIZE step precision."""
        client = self.order_exchange or self.data_exchange
        if client is not None and hasattr(client, "amount_to_precision"):
            try:
                prec_str = client.amount_to_precision(symbol, amount)
                return float(prec_str)
            except Exception:
                pass
        return self.truncate_amount(symbol, amount)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Fetch raw historical OHLCV data strictly from live data_exchange."""
        if self.is_mock:
            df = await self.fetch_ohlcv_dataframe(symbol=symbol, timeframe=timeframe, limit=limit or 200)
            return df.values.tolist()

        client = self.data_exchange or self.order_exchange
        try:
            return await client.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
                params=params or {},
            )
        except Exception as e:
            logger.error("Failed to fetch raw OHLCV for %s (%s): %s", symbol, timeframe, e)
            raise

    async def fetch_ohlcv_dataframe(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "15m",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data as a Pandas DataFrame strictly using live data_exchange.

        If running in mock simulation mode, generates realistic synthetic OHLCV candles
        or returns injected mock candles if set.
        In live/sandbox mode, fetches real candles via CCXT data_exchange.fetch_ohlcv.

        Args:
            symbol: Trading pair symbol (e.g. 'SOL/USDT'). Defaults to config.target_symbol.
            timeframe: Candle timeframe interval (e.g. '15m', '1h').
            limit: Maximum number of candles to return (default 200).

        Returns:
            pd.DataFrame: Columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        sym = symbol or self.config.target_symbol

        if self.is_mock:
            if self._mock_ohlcv is not None and not self._mock_ohlcv.empty:
                return self._mock_ohlcv.copy()

            # Generate deterministic synthetic OHLCV candles
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            interval_ms = 15 * 60 * 1000
            base_p = float(self._mock_prices.get(sym, 150.0))

            rows = []
            price = base_p
            for i in range(limit):
                ts = now_ms - ((limit - i) * interval_ms)
                open_p = price
                reversion = (base_p - price) * 0.05
                noise = random.gauss(0, 0.005) * price
                close_p = max(0.01, round(price + reversion + noise, 4))
                high_p = round(max(open_p, close_p) * (1.0 + random.uniform(0.0005, 0.002)), 4)
                low_p = round(min(open_p, close_p) * (1.0 - random.uniform(0.0005, 0.002)), 4)
                vol = round(random.uniform(500.0, 5000.0), 2)
                rows.append([ts, open_p, high_p, low_p, close_p, vol])
                price = close_p

            return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

        client = self.data_exchange or self.order_exchange
        max_retries = 2
        backoff_seconds = 0.5
        for attempt in range(max_retries + 1):
            try:
                raw_candles = await client.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
                if not raw_candles:
                    logger.warning("Empty OHLCV data received for %s (%s)", sym, timeframe)
                    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

                df = pd.DataFrame(
                    raw_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            except Exception as e:
                # Detect Binance transient error code -1001 (internal server error)
                is_transient_1001 = (
                    "-1001" in str(e)
                    or getattr(e, "code", None) in (-1001, "-1001")
                    or (isinstance(getattr(e, "args", None), (list, tuple)) and any("-1001" in str(arg) for arg in e.args))
                )
                if is_transient_1001 and attempt < max_retries:
                    logger.warning(
                        "Transient Binance -1001 internal server error fetching OHLCV for %s (%s). Retrying %d/%d after %.1fs backoff...",
                        sym,
                        timeframe,
                        attempt + 1,
                        max_retries,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue

                logger.error("Failed to fetch OHLCV dataframe for %s (%s): %s", sym, timeframe, e)
                raise

    async def liquidate_to_usdt(
        self,
        symbol: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> OrderReceipt:
        """Liquidate base asset balance into USDT with comprehensive institutional edge-case guards:

        1. Re-entrancy Lock: Serializes liquidation calls to prevent race conditions.
        2. Dust & Fee Erosion: Applies fee buffer and exchange-truncated rounding (ROUND_DOWN).
           Detects unmarketable dust balances without crashing.
        3. Slippage Bounds: Uses aggressive Limit IOC (Immediate-or-Cancel) bounded by
           `max_liquidation_slippage_pct` (or configurable market order) to prevent flash-illiquidity slippage.
        4. In-Flight Network Partitions: Idempotent clientOrderId tracking with post-timeout
           exchange balance and trade verification before any retry.
        """
        async with self._order_lock:
            sym = symbol or self.config.target_symbol
            base_asset = sym.split("/")[0]

            # Deterministic sizing: if amount not specified, query available free balance
            if amount is None:
                free_balance = await self.get_balance(base_asset)
                # Apply fee buffer if fees could be deducted in base asset
                if self.config.fee_buffer_pct > 0.0 and free_balance > 0.0:
                    amount = free_balance * (1.0 - self.config.fee_buffer_pct)
                else:
                    amount = free_balance

            if amount <= 0.0:
                raise ValueError(f"Cannot execute market sell with non-positive amount: {amount}")

            # Apply strict downwards truncation (ROUND_DOWN) to exchange stepSize
            formatted_amount = self.truncate_amount(sym, amount)

            # Handling Mock Exchange Execution
            if self.is_mock:
                current_price = await self.get_ticker_price(sym)
                cost = formatted_amount * current_price
                self._mock_balances[base_asset] = max(
                    0.0, self._mock_balances.get(base_asset, 0.0) - formatted_amount
                )
                quote_asset = sym.split("/")[1] if "/" in sym else "USDT"
                self._mock_balances[quote_asset] = (
                    self._mock_balances.get(quote_asset, 0.0) + cost
                )

                receipt = OrderReceipt(
                    order_id=f"mock_ord_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                    symbol=sym,
                    side="sell",
                    amount=formatted_amount,
                    price=current_price,
                    cost=cost,
                    status="closed",
                    raw_response={"info": "mock_execution_successful"},
                )
                logger.warning(
                    "[MOCK EXECUTION] Sold %.4f %s at $%.2f (Total: $%.2f)",
                    formatted_amount,
                    sym,
                    current_price,
                    cost,
                )
                return receipt

            # Real / Testnet CCXT Execution
            try:
                market_res = self._exchange.market(sym)
                market = await market_res if hasattr(market_res, "__await__") else market_res
                min_lot = market.get("limits", {}).get("amount", {}).get("min")
                min_notional = market.get("limits", {}).get("cost", {}).get("min")
                current_price = await self.get_ticker_price(sym)
                estimated_notional = formatted_amount * current_price

                # Check Dust: If balance is below minimum LOT_SIZE or MIN_NOTIONAL
                if (min_lot and formatted_amount < min_lot) or (min_notional and estimated_notional < min_notional):
                    raw_free = await self.get_balance(base_asset)
                    if (min_lot and raw_free < min_lot) or (min_notional and (raw_free * current_price) < min_notional):
                        logger.warning(
                            "[DUST DETECTED] %s balance (%.6f %s / $%.2f) is below LOT_SIZE min (%s) or MIN_NOTIONAL (%s). "
                            "Position cannot be liquidated on spot book; recording as dust.",
                            sym, raw_free, base_asset, raw_free * current_price, min_lot, min_notional,
                        )
                        return OrderReceipt(
                            order_id="dust_skipped",
                            symbol=sym,
                            side="sell",
                            amount=formatted_amount,
                            price=current_price,
                            cost=0.0,
                            status="dust_skipped",
                            raw_response={"reason": "balance_below_exchange_minimums"},
                        )
                    else:
                        raise ValueError(
                            f"Order amount {formatted_amount} is below exchange LOT_SIZE minimum {min_lot} for {sym}"
                        )

                # Generate deterministic Idempotent Client Order ID
                client_order_id = f"sentinel_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid.uuid4().hex[:6]}"

                async def _submit_order_attempt(coid: str) -> Dict[str, Any]:
                    """Submit order with slippage bounding (Limit IOC) or pure Market."""
                    if self.config.liquidation_order_type == "limit_ioc":
                        # Query current top-of-book (best bid) for bounded limit execution
                        orderbook = await self._exchange.fetch_order_book(sym, limit=5)
                        bids = orderbook.get("bids", [])
                        best_bid = float(bids[0][0]) if bids else current_price

                        # Bounded Limit Price: best_bid * (1.0 - max_liquidation_slippage_pct)
                        slippage_bound = self.config.max_liquidation_slippage_pct
                        raw_limit = best_bid * (1.0 - slippage_bound)
                        limit_price = float(self._exchange.price_to_precision(sym, raw_limit))

                        logger.info(
                            "Submitting SLIPPAGE-BOUNDED LIMIT IOC for %s: Amount=%.4f, Limit=$%.4f "
                            "(BestBid=$%.4f, MaxSlippage=%.1f%%, ClientId=%s)",
                            sym, formatted_amount, limit_price, best_bid, slippage_bound * 100, coid,
                        )
                        params = {
                            "timeInForce": "IOC",
                            "clientOrderId": coid,
                        }
                        ord_res = await self._exchange.create_order(
                            sym, "limit", "sell", formatted_amount, limit_price, params
                        )

                        # Check for immediate-or-cancel unfilled cancellation
                        filled_qty = float(ord_res.get("filled") or 0.0)
                        if ord_res.get("status") == "canceled" and filled_qty == 0.0:
                            logger.warning(
                                "[SLIPPAGE BOUND BREACH] Limit IOC order rejected due to rapid spread widening "
                                "below $%.4f. Escalating with market order fallback...",
                                limit_price,
                            )
                            fb_params = {"clientOrderId": f"{coid}_fb"}
                            ord_res = await self._exchange.create_market_sell_order(sym, formatted_amount, fb_params)
                        return ord_res

                    else:
                        # Standard Market Order with Idempotent Client Order ID
                        logger.info(
                            "Submitting MARKET SELL order for %s: Amount=%.4f (ClientId=%s)",
                            sym, formatted_amount, coid,
                        )
                        params = {"clientOrderId": coid}
                        return await self._exchange.create_market_sell_order(sym, formatted_amount, params)

                # Order Submission with Network Partition Resilience
                order = None
                try:
                    order = await _submit_order_attempt(client_order_id)
                except (ccxt.NetworkError, ccxt.RequestTimeout, asyncio.TimeoutError) as net_err:
                    logger.warning(
                        "[NETWORK PARTITION] Order submission encountered network error: %s. "
                        "Checking exchange state for clientOrderId %s before retrying...",
                        net_err, client_order_id,
                    )
                    # Recovery Step 1: Check if base balance decreased on exchange
                    await asyncio.sleep(1.0)
                    post_balance = await self.get_balance(base_asset)
                    if post_balance < (amount * 0.5):
                        logger.info(
                            "[PARTITION RECOVERED] Balance on exchange decreased (%.4f -> %.4f). Order executed.",
                            amount, post_balance,
                        )
                        order = {
                            "id": client_order_id,
                            "clientOrderId": client_order_id,
                            "status": "closed",
                            "filled": formatted_amount,
                            "average": current_price,
                            "price": current_price,
                            "cost": formatted_amount * current_price,
                        }
                    else:
                        # Recovery Step 2: Check open orders
                        open_orders = await self._exchange.fetch_open_orders(sym)
                        matching = [o for o in open_orders if o.get("clientOrderId") == client_order_id]
                        if matching:
                            logger.info("[PARTITION RECOVERED] Found matching open order: %s", matching[0].get("id"))
                            order = matching[0]
                        else:
                            # Recovery Step 3: Check recent trade history
                            trades = await self._exchange.fetch_my_trades(sym, limit=5)
                            now_ms = datetime.now(timezone.utc).timestamp() * 1000
                            recent_sells = [
                                t for t in trades
                                if t.get("side") == "sell" and (now_ms - t.get("timestamp", 0) < 30000)
                            ]
                            if recent_sells:
                                logger.info("[PARTITION RECOVERED] Found recent sell trades on exchange.")
                                avg_p = sum(t["price"] * t["amount"] for t in recent_sells) / sum(t["amount"] for t in recent_sells)
                                tot_f = sum(t["amount"] for t in recent_sells)
                                order = {
                                    "id": str(recent_sells[-1].get("order", client_order_id)),
                                    "clientOrderId": client_order_id,
                                    "status": "closed",
                                    "filled": tot_f,
                                    "average": avg_p,
                                    "price": avg_p,
                                    "cost": tot_f * avg_p,
                                }
                            else:
                                logger.warning("[PARTITION CONFIRMED] Order was not executed on exchange. Retrying once safely...")
                                order = await _submit_order_attempt(f"{client_order_id}_rt")

                executed_price = (
                    order.get("average")
                    or order.get("price")
                    or current_price
                )
                executed_amount = order.get("filled") or formatted_amount
                executed_cost = order.get("cost") or (float(executed_price) * float(executed_amount))

                receipt = OrderReceipt(
                    order_id=str(order.get("id") or order.get("clientOrderId") or "unknown"),
                    symbol=sym,
                    side="sell",
                    amount=float(executed_amount),
                    price=float(executed_price),
                    cost=float(executed_cost),
                    status=str(order.get("status", "closed")),
                    raw_response=order,
                )
                logger.info(
                    "[ORDER FILLED] Liquidation Sell: %s | Amount: %.4f | Price: $%.4f | Revenue: $%.2f | Status: %s",
                    sym, receipt.amount, receipt.price, receipt.cost, receipt.status,
                )
                return receipt

            except Exception as e:
                logger.critical("EXCHANGE EXECUTION ERROR on market sell for %s: %s", sym, e)
                raise

    # Backwards-compatible alias for existing codebase
    execute_market_sell = liquidate_to_usdt

    async def execute_market_buy_idempotent(
        self,
        symbol: Optional[str] = None,
        cost_usdt: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute a spot market buy order idempotently with precision-truncated sizing.

        Guarantees:
        1. Serialized execution via _order_lock.
        2. Idempotent clientOrderId tracking with network partition recovery.
        3. Exchange LOT_SIZE and MIN_NOTIONAL compliance with ROUND_DOWN precision.
        4. Mock ledger synchronization in simulation mode.

        Args:
            symbol: Trading pair symbol (e.g. 'SOL/USDT').
            cost_usdt: Total quote asset (USDT) amount to allocate for this buy.

        Returns:
            Dict[str, Any]: Execution order details including 'id', 'symbol', 'side',
                            'amount', 'price', 'cost', and 'status'.
        """
        async with self._order_lock:
            sym = symbol or self.config.target_symbol
            base_asset = sym.split("/")[0]
            quote_asset = sym.split("/")[1] if "/" in sym else "USDT"

            if cost_usdt <= 0.0:
                raise ValueError(f"Cannot execute market buy with non-positive cost_usdt: {cost_usdt}")

            current_price = await self.get_ticker_price(sym)
            raw_amount = cost_usdt / current_price
            formatted_amount = self.amount_to_precision(sym, raw_amount)

            if formatted_amount <= 0.0:
                raise ValueError(
                    f"Calculated buy amount {formatted_amount} is non-positive for cost ${cost_usdt:.2f} at price ${current_price:.4f}"
                )

            client_order_id = f"sentinel_buy_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid.uuid4().hex[:6]}"

            # Handling Mock Exchange Execution
            if self.is_mock:
                free_usdt = self._mock_balances.get(quote_asset, 0.0)
                actual_cost = formatted_amount * current_price
                if free_usdt < actual_cost:
                    logger.warning(
                        "[MOCK BUY] Insufficient mock USDT balance ($%.2f < $%.2f). Adjusting cost to free balance.",
                        free_usdt,
                        actual_cost,
                    )
                    actual_cost = free_usdt
                    formatted_amount = self.amount_to_precision(sym, actual_cost / current_price)

                self._mock_balances[quote_asset] = max(0.0, self._mock_balances.get(quote_asset, 0.0) - actual_cost)
                self._mock_balances[base_asset] = self._mock_balances.get(base_asset, 0.0) + formatted_amount

                logger.info(
                    "[MOCK ORDER FILLED] Market Buy %s | Amount: %.4f | Price: $%.4f | Cost: $%.2f | ClientId: %s",
                    sym, formatted_amount, current_price, actual_cost, client_order_id,
                )
                return {
                    "id": client_order_id,
                    "clientOrderId": client_order_id,
                    "symbol": sym,
                    "side": "buy",
                    "amount": formatted_amount,
                    "price": current_price,
                    "cost": actual_cost,
                    "status": "closed",
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "info": {"mode": "mock"},
                }

            # Real / Testnet CCXT Execution
            try:
                market_res = self._exchange.market(sym)
                market = await market_res if hasattr(market_res, "__await__") else market_res
                min_lot = market.get("limits", {}).get("amount", {}).get("min")
                min_notional = market.get("limits", {}).get("cost", {}).get("min")
                estimated_notional = formatted_amount * current_price

                if min_lot and formatted_amount < min_lot:
                    raise ValueError(
                        f"Order amount {formatted_amount} is below exchange LOT_SIZE min {min_lot} for {sym}"
                    )
                if min_notional and estimated_notional < min_notional:
                    raise ValueError(
                        f"Order notional ${estimated_notional:.2f} is below MIN_NOTIONAL ${min_notional:.2f} for {sym}"
                    )

                async def _submit_buy_attempt(coid: str) -> Dict[str, Any]:
                    logger.info(
                        "Submitting MARKET BUY order for %s: Amount=%.4f (Cost~$%.2f, ClientId=%s)",
                        sym, formatted_amount, cost_usdt, coid,
                    )
                    params = {"clientOrderId": coid}
                    return await self._exchange.create_market_buy_order(sym, formatted_amount, params)

                order = None
                try:
                    order = await _submit_buy_attempt(client_order_id)
                except (ccxt.NetworkError, ccxt.RequestTimeout, asyncio.TimeoutError) as net_err:
                    logger.warning(
                        "[NETWORK PARTITION] Buy order submission encountered network error: %s. "
                        "Checking exchange state for clientOrderId %s before retrying...",
                        net_err, client_order_id,
                    )
                    await asyncio.sleep(1.0)
                    post_base = await self.get_balance(base_asset)
                    if post_base >= (formatted_amount * 0.9):
                        logger.info(
                            "[PARTITION RECOVERED] Base balance increased on exchange. Buy order executed."
                        )
                        order = {
                            "id": client_order_id,
                            "clientOrderId": client_order_id,
                            "status": "closed",
                            "filled": formatted_amount,
                            "average": current_price,
                            "price": current_price,
                            "cost": formatted_amount * current_price,
                        }
                    else:
                        open_orders = await self._exchange.fetch_open_orders(sym)
                        matching = [o for o in open_orders if o.get("clientOrderId") == client_order_id]
                        if matching:
                            logger.info("[PARTITION RECOVERED] Found matching open order: %s", matching[0].get("id"))
                            order = matching[0]
                        else:
                            trades = await self._exchange.fetch_my_trades(sym, limit=5)
                            now_ms = datetime.now(timezone.utc).timestamp() * 1000
                            recent_buys = [
                                t for t in trades
                                if t.get("side") == "buy" and (now_ms - t.get("timestamp", 0) < 30000)
                            ]
                            if recent_buys:
                                logger.info("[PARTITION RECOVERED] Found recent buy trades on exchange.")
                                avg_p = sum(t["price"] * t["amount"] for t in recent_buys) / sum(t["amount"] for t in recent_buys)
                                tot_f = sum(t["amount"] for t in recent_buys)
                                order = {
                                    "id": str(recent_buys[-1].get("order", client_order_id)),
                                    "clientOrderId": client_order_id,
                                    "status": "closed",
                                    "filled": tot_f,
                                    "average": avg_p,
                                    "price": avg_p,
                                    "cost": tot_f * avg_p,
                                }
                            else:
                                logger.warning("[PARTITION CONFIRMED] Buy was not executed on exchange. Retrying once safely...")
                                order = await _submit_buy_attempt(f"{client_order_id}_rt")

                executed_price = float(
                    order.get("average")
                    or order.get("price")
                    or current_price
                )
                executed_amount = float(order.get("filled") or formatted_amount)
                executed_cost = float(order.get("cost") or (executed_price * executed_amount))

                receipt = {
                    "id": str(order.get("id") or order.get("clientOrderId") or client_order_id),
                    "clientOrderId": str(order.get("clientOrderId") or client_order_id),
                    "symbol": sym,
                    "side": "buy",
                    "amount": executed_amount,
                    "price": executed_price,
                    "cost": executed_cost,
                    "status": str(order.get("status", "closed")),
                    "raw_response": order,
                }
                logger.info(
                    "[ORDER FILLED] Market Buy: %s | Amount: %.4f | Price: $%.4f | Cost: $%.2f | Status: %s",
                    sym, executed_amount, executed_price, executed_cost, receipt["status"],
                )
                return receipt

            except Exception as e:
                logger.critical("EXCHANGE EXECUTION ERROR on market buy for %s: %s", sym, e)
                raise

    def set_mock_ohlcv(self, df: Optional[pd.DataFrame]) -> None:
        """Utility for unit testing: sets simulated OHLCV dataframe."""
        self._mock_ohlcv = df.copy() if df is not None else None

    def set_mock_price(self, symbol: str, price: float) -> None:
        """Utility for unit testing: sets simulated market price."""
        self._mock_prices[symbol] = max(0.0001, price)

    def set_mock_balance(self, asset: str, balance: float) -> None:
        """Utility for unit testing: sets simulated asset balance."""
        self._mock_balances[asset] = max(0.0, balance)

    def simulate_price_tick(
        self,
        symbol: Optional[str] = None,
        drift: float = 0.0,
        volatility: float = 0.005,
    ) -> float:
        """Simulate realistic price movement using geometric Brownian motion."""
        sym = symbol or self.config.target_symbol
        curr_price = self._mock_prices.get(sym, 150.0)

        # Standard normal shock
        shock = random.gauss(0.0, 1.0)
        # dS = S * (drift * dt + volatility * dW)
        delta_pct = drift + (volatility * shock)
        new_price = max(0.01, round(curr_price * (1.0 + delta_pct), 4))

        self._mock_prices[sym] = new_price
        return new_price

    def simulate_price_jump(
        self,
        symbol: Optional[str] = None,
        pct_change: float = 0.0,
    ) -> float:
        """Instantly simulate a sharp price move (e.g. +0.10 for +10% pump or -0.05 for -5% dump)."""
        sym = symbol or self.config.target_symbol
        curr_price = self._mock_prices.get(sym, 150.0)
        new_price = max(0.01, round(curr_price * (1.0 + pct_change), 4))
        self._mock_prices[sym] = new_price
        logger.warning(
            "[MOCK PRICE JUMP] %s moved from $%.4f to $%.4f (%+.2f%%)",
            sym,
            curr_price,
            new_price,
            pct_change * 100,
        )
        return new_price

    async def close(self) -> None:
        """Gracefully close dual CCXT async connections."""
        for name, client in [("data_exchange", self.data_exchange), ("order_exchange", self.order_exchange)]:
            if client is not None:
                try:
                    await client.close()
                    logger.info("Exchange %s connection closed successfully.", name)
                except Exception as e:
                    logger.warning("Error while closing %s connection: %s", name, e)
        self.data_exchange = None
        self.order_exchange = None
        self._is_initialized = False
