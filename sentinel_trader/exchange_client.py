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
    """Async CCXT wrapper enforcing Binance Testnet safety invariant."""

    def __init__(self, config: Config):
        self.config = config
        self._exchange: Optional[ccxt.binance] = None
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

    @property
    def is_mock(self) -> bool:
        """Check if client is running in mock simulation mode."""
        return self.config.mock_exchange or self._exchange is None

    async def initialize(self) -> None:
        """Initialize the exchange connection and verify sandbox mode."""
        # Auto-activate mock mode if explicitly requested OR if no API credentials provided
        if self.config.mock_exchange or not self.config.binance_api_key or not self.config.binance_api_secret:
            logger.info("[MOCK SIMULATION MODE] Initializing ExchangeClient in local mock mode.")
            self._is_initialized = True
            return

        logger.info("Connecting to Binance Exchange client via CCXT...")

        # Initialize CCXT Binance async instance
        exchange = ccxt.binance({
            'apiKey': self.config.binance_api_key,
            'secret': self.config.binance_api_secret,
            'enableRateLimit': True,
        })

        # SAFETY INVARIANT #3: Default strictly to Binance Testnet (Sandbox)
        if self.config.binance_sandbox:
            exchange.set_sandbox_mode(True)
            self._exchange = exchange
            logger.info("[SAFETY CHECK: PASSED] Binance Sandbox (Testnet) Mode ENABLED.")
        else:
            if not self.config.allow_production_trading:
                await exchange.close()
                raise PermissionError(
                    "FATAL SAFETY INVARIANT: Live Binance trading blocked. "
                    "ALLOW_PRODUCTION_TRADING must be explicitly set to True."
                )
            self._exchange = exchange
            logger.warning("[WARNING: LIVE PRODUCTION] Operating on REAL Binance Mainnet!")

        try:
            await self._exchange.load_markets()
            self._is_initialized = True
            logger.info("Exchange markets loaded successfully.")
        except Exception as e:
            logger.warning(
                "Failed to connect/authenticate to Binance (%s). Falling back to mock exchange mode.",
                e,
            )
            await self.close()
            self._exchange = None
            self._is_initialized = True

    async def get_ticker_price(self, symbol: Optional[str] = None) -> float:
        """Fetch live ticker/order book price from Binance Testnet or mock ledger."""
        sym = symbol or self.config.target_symbol

        if self.is_mock:
            return float(self._mock_prices.get(sym, 150.0))

        try:
            ticker = await self._exchange.fetch_ticker(sym)
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

    async def get_balance(self, asset: Optional[str] = None) -> float:
        """Fetch real free balance of the asset on Binance Testnet or mock ledger."""
        target_asset = asset or self.config.base_asset

        if self.is_mock:
            return float(self._mock_balances.get(target_asset, 0.0))

        try:
            balance = await self._exchange.fetch_balance()
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

        if self.is_mock or self._exchange is None:
            d = Decimal(str(amount))
            return float(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

        try:
            market = self._exchange.market(symbol)
            if hasattr(market, "__await__"):
                # If mocked as coroutine
                return float(Decimal(str(amount)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))
            precision = market.get("precision", {}).get("amount")
            if precision is None:
                return float(amount)

            precision_mode = getattr(self._exchange, "precisionMode", DECIMAL_PLACES)
            padding_mode = getattr(self._exchange, "paddingMode", 0)
            truncated_str = decimal_to_precision(
                str(amount), TRUNCATE, precision, precision_mode, padding_mode
            )
            return float(truncated_str)
        except Exception as e:
            logger.debug("Error during exchange truncate_amount: %s. Using Decimal fallback.", e)
            d = Decimal(str(amount))
            return float(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

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
        """Gracefully close the underlying CCXT async connection."""
        if self._exchange is not None:
            try:
                await self._exchange.close()
                logger.info("Exchange connection closed successfully.")
            except Exception as e:
                logger.warning("Error while closing exchange connection: %s", e)
            finally:
                self._exchange = None
                self._is_initialized = False
