"""Risk Engine module for Sentinel-Trader.

SAFETY INVARIANTS:
1. 100% deterministic code.
2. The AI model NEVER determines order size or calculates prices.
3. Multi-slot portfolio management:
   - Tracks concurrent positions: positions: dict[str, PositionState]
   - Enforces MAX_ACTIVE_POSITIONS = 3
   - Sizing: Allocates SLOT_ALLOCATION_PCT = 0.25 (25% of free USDT, minimum $15)
   - Each active slot independently monitors entry price basis against TP (+5.0%) and SL (-2.5%)
   - Global Macro Breaker: Systemic threats trigger emergency liquidation across ALL open slots simultaneously
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from sentinel_trader.ai_sentinel import NewsAnalysis
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient, OrderReceipt
from sentinel_trader.notifier import DiscordNotifier

logger = logging.getLogger("sentinel.risk")

# Portfolio slot boundaries
MAX_ACTIVE_POSITIONS: int = 3
SLOT_ALLOCATION_PCT: float = 0.25

SYSTEMIC_KEYWORDS: Tuple[str, ...] = (
    "blanket crypto ban",
    "crypto ban",
    "binance halt",
    "binance insolvency",
    "exchange halt",
    "sec blanket",
    "tether depeg",
    "usdt depeg",
    "systemic",
    "market-wide halt",
    "global ban",
    "banking collapse",
)


class TriggerReason(str, Enum):
    """Categorization of liquidation triggers."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    AI_PANIC_SELL = "AI_PANIC_SELL"
    GLOBAL_MACRO_BREAKER = "GLOBAL_MACRO_BREAKER"
    MANUAL_EMERGENCY = "MANUAL_EMERGENCY"


class PositionState(BaseModel):
    """State tracking for a single portfolio slot position."""

    symbol: str
    entry_price: float
    amount: float
    order_id: Optional[str] = None
    entry_timestamp: Optional[str] = None
    last_known_pnl: float = 0.0
    is_liquidated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert position state to dictionary matching persistence schema."""
        return {
            "entry_price": self.entry_price,
            "amount": self.amount,
            "order_id": self.order_id,
            "entry_timestamp": self.entry_timestamp,
            "last_known_pnl": self.last_known_pnl,
            "is_liquidated": self.is_liquidated,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class PositionStatus(BaseModel):
    """Real-time telemetry of a monitored position."""

    symbol: str
    entry_price: float
    current_price: float
    pnl_pct: float
    pnl_usd: float
    balance_base: float
    position_value_usd: float
    is_liquidated: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LiquidationRecord(BaseModel):
    """Immutable audit record of a liquidation event."""

    trigger: TriggerReason
    symbol: str
    entry_price: float
    exit_price: float
    realized_pnl_pct: float
    realized_pnl_usd: float
    amount_liquidated: float
    receipt: OrderReceipt
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str


class RiskEngine:
    """Deterministic Multi-Slot Risk Management and Portfolio Protection Engine."""

    def __init__(
        self,
        config: Config,
        exchange: ExchangeClient,
        initial_entry_price: Optional[float] = None,
        notifier: Optional[DiscordNotifier] = None,
        initial_holding_amount: Optional[float] = None,
        max_active_positions: Optional[int] = None,
        slot_allocation_pct: Optional[float] = None,
        trade_history_file: Optional[str] = None,
    ):
        self.config = config
        self.exchange = exchange
        self.max_active_positions = max_active_positions or getattr(config, "max_active_positions", MAX_ACTIVE_POSITIONS)
        self.slot_allocation_pct = slot_allocation_pct or getattr(config, "slot_allocation_pct", SLOT_ALLOCATION_PCT)
        self.trade_history_file = trade_history_file or getattr(config, "trade_history_file", None)
        self.positions: Dict[str, PositionState] = {}
        self.notifier = notifier or DiscordNotifier(self.config)

        self.last_liquidation: Optional[LiquidationRecord] = None
        self.liquidation_history: List[LiquidationRecord] = []

        self._default_entry_price: Optional[float] = initial_entry_price
        self._default_holding_amount: Optional[float] = initial_holding_amount
        self._is_liquidated_global: bool = False

        # Initialize single default slot if initial_entry_price provided
        if initial_entry_price is not None and initial_entry_price > 0:
            sym = self.config.target_symbol
            self.positions[sym] = PositionState(
                symbol=sym,
                entry_price=initial_entry_price,
                amount=initial_holding_amount or 0.0,
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
            )

    # =========================================================================
    # Backward Compatibility Layer (Single-Position Callers & Tests)
    # =========================================================================

    @property
    def entry_price(self) -> Optional[float]:
        """Backward-compatible entry_price property."""
        if self.config.target_symbol in self.positions:
            return self.positions[self.config.target_symbol].entry_price
        if len(self.positions) == 1:
            return next(iter(self.positions.values())).entry_price
        return self._default_entry_price

    @entry_price.setter
    def entry_price(self, val: Optional[float]) -> None:
        self._default_entry_price = val
        if val is not None and self.config.target_symbol in self.positions:
            self.positions[self.config.target_symbol].entry_price = val

    @property
    def holding_amount(self) -> Optional[float]:
        """Backward-compatible holding_amount property."""
        if self.config.target_symbol in self.positions:
            return self.positions[self.config.target_symbol].amount
        if len(self.positions) == 1:
            return next(iter(self.positions.values())).amount
        return self._default_holding_amount

    @holding_amount.setter
    def holding_amount(self, val: Optional[float]) -> None:
        self._default_holding_amount = val
        if val is not None and self.config.target_symbol in self.positions:
            self.positions[self.config.target_symbol].amount = val

    @property
    def is_liquidated(self) -> bool:
        """Check whether all monitored positions are liquidated."""
        if self.positions:
            return all(p.is_liquidated for p in self.positions.values())
        return self._is_liquidated_global

    @is_liquidated.setter
    def is_liquidated(self, val: bool) -> None:
        self._is_liquidated_global = val
        if val:
            for p in self.positions.values():
                p.is_liquidated = True

    # =========================================================================
    # Multi-Slot Portfolio Operations
    # =========================================================================

    @property
    def active_positions(self) -> Dict[str, PositionState]:
        """Return all active, non-liquidated positions."""
        return {
            sym: pos for sym, pos in self.positions.items()
            if not pos.is_liquidated
        }

    @property
    def can_open_position(self) -> bool:
        """Check if portfolio has an open slot under MAX_ACTIVE_POSITIONS."""
        return len(self.active_positions) < self.max_active_positions

    def calculate_slot_size(self, free_usdt: float) -> float:
        """Determine position allocation sizing (25% free balance, min $15)."""
        if free_usdt < 15.0:
            return 0.0
        allocated = free_usdt * self.slot_allocation_pct
        return max(allocated, 15.0)

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        order_id: Optional[str] = None,
        entry_timestamp: Optional[str] = None,
    ) -> PositionState:
        """Register a new active slot in the portfolio."""
        if len(self.active_positions) >= self.max_active_positions:
            raise ValueError(
                f"MAX_ACTIVE_POSITIONS ({self.max_active_positions}) reached. "
                f"Cannot open slot for {symbol}."
            )

        pos = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            amount=amount,
            order_id=order_id,
            entry_timestamp=entry_timestamp or datetime.now(timezone.utc).isoformat(),
            last_known_pnl=0.0,
            is_liquidated=False,
        )
        self.positions[symbol] = pos
        logger.info(
            "[SLOT OPENED] %s: Entry $%.4f | Size: %.4f | Slots: %d/%d",
            symbol,
            entry_price,
            amount,
            len(self.active_positions),
            self.max_active_positions,
        )
        return pos

    def close_position(self, symbol: str) -> None:
        """Close and mark a slot position as liquidated."""
        if symbol in self.positions:
            self.positions[symbol].is_liquidated = True
            self.positions[symbol].amount = 0.0

    def set_entry_price(self, price: float, symbol: Optional[str] = None) -> None:
        """Set cost basis entry price for target symbol or specified slot."""
        if price <= 0:
            raise ValueError(f"Entry price must be positive, got {price}")
        sym = symbol or self.config.target_symbol
        if sym in self.positions:
            self.positions[sym].entry_price = price
            self.positions[sym].is_liquidated = False
        else:
            self.positions[sym] = PositionState(
                symbol=sym,
                entry_price=price,
                amount=self._default_holding_amount or 0.0,
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
            )
        self._default_entry_price = price
        logger.info("[ENTRY PRICE ESTABLISHED] %s entry locked at $%.4f", sym, price)

    def rehydrate(self, entry_price: float, holding_amount: Optional[float] = None) -> None:
        """Rehydrate single target symbol state after restart or crash."""
        sym = self.config.target_symbol
        self.positions[sym] = PositionState(
            symbol=sym,
            entry_price=entry_price,
            amount=holding_amount or 0.0,
            entry_timestamp=datetime.now(timezone.utc).isoformat(),
            is_liquidated=False,
        )
        self.set_entry_price(entry_price, symbol=sym)
        self.holding_amount = holding_amount
        self.is_liquidated = False
        self.last_liquidation = None
        logger.info(
            "[RISK ENGINE REHYDRATED] Symbol: %s | Cost basis: $%.4f | Size: %s",
            sym,
            entry_price,
            f"{holding_amount:.4f}" if holding_amount is not None else "N/A",
        )

    def rehydrate_positions(self, positions_data: Dict[str, Any]) -> None:
        """Rehydrate multiple active slot positions from persistent state store."""
        self.positions.clear()
        for symbol, data in positions_data.items():
            if not isinstance(data, dict):
                continue
            entry_p = float(data.get("entry_price") or 0.0)
            amt = float(data.get("amount") or data.get("holding_amount") or 0.0)
            order_id = data.get("order_id") or data.get("client_order_id")
            ts = data.get("entry_timestamp")
            pnl = float(data.get("last_known_pnl") or 0.0)
            is_liq = bool(data.get("is_liquidated", False))

            if entry_p > 0 and amt > 0 and not is_liq:
                self.positions[symbol] = PositionState(
                    symbol=symbol,
                    entry_price=entry_p,
                    amount=amt,
                    order_id=order_id,
                    entry_timestamp=ts,
                    last_known_pnl=pnl,
                    is_liquidated=False,
                )
        self.last_liquidation = None
        logger.info("[RISK ENGINE REHYDRATED] Rehydrated %d active slots.", len(self.active_positions))

    def reset(
        self,
        initial_entry_price: Optional[float] = None,
        holding_amount: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Reset risk engine state or single slot."""
        sym = symbol or self.config.target_symbol
        if initial_entry_price:
            self.positions[sym] = PositionState(
                symbol=sym,
                entry_price=initial_entry_price,
                amount=holding_amount or 0.0,
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
                is_liquidated=False,
            )
            self._default_entry_price = initial_entry_price
            self._default_holding_amount = holding_amount
        else:
            self.positions.clear()
            self._default_entry_price = None
            self._default_holding_amount = None

        self.is_liquidated = False
        self.last_liquidation = None
        if initial_entry_price:
            logger.info("[RISK ENGINE RESET] %s entry price: $%.4f", sym, initial_entry_price)

    # =========================================================================
    # Deterministic Limits & PnL Evaluation
    # =========================================================================

    def calculate_pnl(
        self,
        current_price: float,
        balance: float = 0.0,
        symbol: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Calculate percentage and absolute USD PnL deterministically."""
        sym = symbol or self.config.target_symbol
        pos = self.positions.get(sym)
        cost_basis = pos.entry_price if pos else (self.entry_price or 0.0)

        if not cost_basis or cost_basis <= 0:
            return 0.0, 0.0

        pnl_pct = (current_price - cost_basis) / cost_basis
        pnl_usd = (current_price - cost_basis) * balance
        return pnl_pct, pnl_usd

    def get_limits(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """Return computed hard mathematical price boundaries for symbol."""
        sym = symbol or self.config.target_symbol
        pos = self.positions.get(sym)
        basis = pos.entry_price if pos else (self.entry_price or 0.0)

        if not basis or basis <= 0:
            return {"entry_price": 0.0, "take_profit_price": 0.0, "stop_loss_price": 0.0}

        return {
            "entry_price": basis,
            "take_profit_price": round(basis * (1.0 + self.config.take_profit_pct), 4),
            "stop_loss_price": round(basis * (1.0 - self.config.stop_loss_pct), 4),
        }

    def evaluate_mathematical_limits(
        self,
        current_price: float,
        symbol: Optional[str] = None,
    ) -> Optional[TriggerReason]:
        """Check if hard mathematical take-profit or stop-loss limits are breached.

        100% deterministic mathematical evaluation.
        """
        sym = symbol or self.config.target_symbol
        pos = self.positions.get(sym)
        basis = pos.entry_price if pos else self.entry_price

        if not basis or basis <= 0:
            return None

        pnl_pct = (current_price - basis) / basis
        if pos:
            pos.last_known_pnl = pnl_pct

        # Take-Profit check
        if pnl_pct >= self.config.take_profit_pct:
            logger.warning(
                "[TAKE-PROFIT BREACHED] %s PnL %+.2f%% exceeds target limit +%.2f%%",
                sym,
                pnl_pct * 100,
                self.config.take_profit_pct * 100,
            )
            return TriggerReason.TAKE_PROFIT

        # Stop-Loss check
        if pnl_pct <= -self.config.stop_loss_pct:
            logger.warning(
                "[STOP-LOSS BREACHED] %s PnL %+.2f%% breaches stop loss limit -%.2f%%",
                sym,
                pnl_pct * 100,
                self.config.stop_loss_pct * 100,
            )
            return TriggerReason.STOP_LOSS

        return None

    # =========================================================================
    # AI Threat & Global Macro Breaker Evaluation
    # =========================================================================

    def is_systemic_threat(self, analysis: NewsAnalysis, headline: str = "") -> bool:
        """Check if an emergency headline qualifies as a systemic market-wide event.

        Systemic examples: Blanket crypto ban, Binance halt/insolvency, SEC blanket freeze.
        """
        text = f"{headline} {analysis.reasoning}".lower()
        matches_keyword = any(kw in text for kw in SYSTEMIC_KEYWORDS)
        is_panic = analysis.action == "PANIC_SELL"
        is_critical = analysis.threat_level in ["CRITICAL", "HIGH"]
        meets_confidence = analysis.confidence >= self.config.panic_sell_confidence_threshold

        return bool(is_panic and is_critical and meets_confidence and matches_keyword)

    def evaluate_ai_trigger_for_symbol(
        self,
        analysis: NewsAnalysis,
        symbol: str,
        headline: str = "",
    ) -> Optional[TriggerReason]:
        """Evaluate whether AI news analysis qualifies for liquidation of a specific symbol."""
        is_panic = analysis.action == "PANIC_SELL"
        is_high_threat = analysis.threat_level in ["HIGH", "CRITICAL"]
        meets_confidence = analysis.confidence >= self.config.panic_sell_confidence_threshold

        if not (is_panic and is_high_threat and meets_confidence):
            return None

        # Systemic macro breaker applies to all symbols
        if self.is_systemic_threat(analysis, headline):
            return TriggerReason.GLOBAL_MACRO_BREAKER

        # Asset-specific check: if headline provided, verify it is relevant or matches symbol
        if headline:
            base_asset = symbol.split("/")[0].upper()
            text = f"{headline} {analysis.reasoning}".upper()
            if base_asset in text or symbol.upper() in text:
                return TriggerReason.AI_PANIC_SELL
            # If headline doesn't mention asset specifically, check if generic
            return TriggerReason.AI_PANIC_SELL

        return TriggerReason.AI_PANIC_SELL

    def evaluate_ai_trigger(self, analysis: NewsAnalysis, headline: str = "") -> Optional[TriggerReason]:
        """Backward-compatible AI trigger evaluation for default target symbol."""
        return self.evaluate_ai_trigger_for_symbol(
            analysis,
            symbol=self.config.target_symbol,
            headline=headline,
        )

    # =========================================================================
    # Position Telemetry & Liquidation Execution
    # =========================================================================

    async def get_position_status(
        self,
        current_price: float,
        symbol: Optional[str] = None,
    ) -> PositionStatus:
        """Fetch current balance and compute real-time position status for a slot."""
        sym = symbol or self.config.target_symbol
        base_asset = sym.split("/")[0]

        balance = await self.exchange.get_balance(base_asset)
        pnl_pct, pnl_usd = self.calculate_pnl(current_price, balance, symbol=sym)
        position_value = balance * current_price

        pos = self.positions.get(sym)
        entry_p = pos.entry_price if pos else (self.entry_price or current_price)
        is_liq = pos.is_liquidated if pos else self.is_liquidated

        return PositionStatus(
            symbol=sym,
            entry_price=entry_p,
            current_price=current_price,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            balance_base=balance,
            position_value_usd=position_value,
            is_liquidated=is_liq,
        )

    async def execute_liquidation(
        self,
        trigger: TriggerReason,
        notes: str,
        symbol: Optional[str] = None,
    ) -> LiquidationRecord:
        """Instantly liquidate the token balance for a slot into USDT."""
        sym = symbol or self.config.target_symbol
        pos = self.positions.get(sym)

        if pos and pos.is_liquidated:
            logger.warning("Liquidation for %s already executed previously. Skipping.", sym)
            return self.last_liquidation

        base_asset = sym.split("/")[0]

        # Query free balance from exchange
        balance_to_sell = await self.exchange.get_balance(base_asset)
        if balance_to_sell <= 0.0 and pos and pos.amount > 0:
            balance_to_sell = pos.amount

        if balance_to_sell <= 0.0:
            logger.warning("Zero %s balance available to liquidate.", base_asset)
            exit_price = await self.exchange.get_current_price(sym)
            receipt = OrderReceipt(
                order_id="no_balance",
                symbol=sym,
                side="sell",
                amount=0.0,
                price=exit_price,
                cost=0.0,
                status="skipped_zero_balance",
            )
        else:
            logger.critical(
                "[EXECUTION: LIQUIDATING SLOT] Trigger: %s | Selling %.4f %s into %s",
                trigger.value,
                balance_to_sell,
                base_asset,
                self.config.quote_asset,
            )
            receipt = await self.exchange.liquidate_to_usdt(sym, amount=balance_to_sell)

        exit_price = receipt.price
        entry_p = pos.entry_price if pos else (self.entry_price or exit_price)
        pnl_pct, pnl_usd = self.calculate_pnl(exit_price, receipt.amount, symbol=sym)

        record = LiquidationRecord(
            trigger=trigger,
            symbol=sym,
            entry_price=entry_p,
            exit_price=exit_price,
            realized_pnl_pct=pnl_pct,
            realized_pnl_usd=pnl_usd,
            amount_liquidated=receipt.amount,
            receipt=receipt,
            notes=notes,
        )

        if pos:
            pos.is_liquidated = True
            pos.amount = 0.0

        self.last_liquidation = record
        self.liquidation_history.append(record)

        logger.critical(
            "=== [POSITION LIQUIDATED] ===\n"
            "Trigger Reason : %s\n"
            "Symbol         : %s\n"
            "Entry Price    : $%.4f\n"
            "Exit Price     : $%.4f\n"
            "Realized PnL   : %+.2f%% ($%+.2f)\n"
            "Amount Sold    : %.4f %s\n"
            "Total Proceeds : $%.2f %s\n"
            "Notes          : %s\n"
            "=============================",
            trigger.value,
            sym,
            entry_p,
            exit_price,
            pnl_pct * 100,
            pnl_usd,
            receipt.amount,
            base_asset,
            receipt.cost,
            self.config.quote_asset,
            notes,
        )

        # Dispatch non-blocking Discord liquidation notification
        self.notifier.notify_liquidation(
            symbol=sym,
            reason=trigger.value,
            entry_price=entry_p,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            notes=notes,
        )

        # Atomically record closed trade to persistent trade analytics
        try:
            from sentinel_trader.analytics import record_trade
            record_trade(
                symbol=sym,
                entry_price=entry_p,
                exit_price=exit_price,
                amount=receipt.amount,
                realized_pnl_usd=pnl_usd,
                realized_pnl_pct=pnl_pct,
                trigger_reason=trigger.value if hasattr(trigger, "value") else str(trigger),
                file_path=self.trade_history_file,
                config=self.config,
            )
        except Exception as e:
            logger.error("Failed to record trade to analytics: %s", e)

        return record

    async def execute_global_liquidation(
        self,
        trigger: TriggerReason,
        notes: str,
    ) -> List[LiquidationRecord]:
        """Global Macro Breaker: Liquidate ALL open slots simultaneously via asyncio.gather."""
        open_symbols = list(self.active_positions.keys())
        if not open_symbols:
            if self.config.target_symbol in self.positions and not self.positions[self.config.target_symbol].is_liquidated:
                open_symbols = [self.config.target_symbol]
            else:
                logger.warning("[GLOBAL MACRO BREAKER] No active slots found to liquidate.")
                return []

        logger.critical(
            "[GLOBAL MACRO BREAKER TRIGGERED] Emergency liquidating ALL %d active slots simultaneously: %s | Notes: %s",
            len(open_symbols),
            open_symbols,
            notes,
        )

        tasks = [
            self.execute_liquidation(
                trigger=trigger,
                notes=f"[GLOBAL MACRO BREAKER] {notes}",
                symbol=sym,
            )
            for sym in open_symbols
        ]
        records = await asyncio.gather(*tasks, return_exceptions=True)
        valid_records: List[LiquidationRecord] = [r for r in records if isinstance(r, LiquidationRecord)]
        return valid_records

    async def evaluate_tick(
        self,
        current_price: float,
        news_analysis: Optional[NewsAnalysis] = None,
        symbol: Optional[str] = None,
        headline: str = "",
    ) -> Optional[LiquidationRecord]:
        """Evaluate mathematical limits and optional AI news trigger on each tick.

        Returns LiquidationRecord if liquidation occurred, None otherwise.
        """
        sym = symbol or self.config.target_symbol
        pos = self.positions.get(sym)

        if pos and pos.is_liquidated:
            return None
        if not pos and self.is_liquidated:
            return None

        # Auto-initialize entry price if not set yet
        if pos is None and self.entry_price is None:
            self.set_entry_price(current_price, symbol=sym)

        # 1. Evaluate hard mathematical limits first (highest priority, 100% deterministic)
        math_trigger = self.evaluate_mathematical_limits(current_price, symbol=sym)
        if math_trigger is not None:
            pnl_pct, _ = self.calculate_pnl(current_price, symbol=sym)
            notes = (
                f"Mathematical threshold breached on {sym}: PnL {pnl_pct * 100:+.2f}% "
                f"(TP: +{self.config.take_profit_pct * 100:.1f}%, SL: -{self.config.stop_loss_pct * 100:.1f}%)"
            )
            return await self.execute_liquidation(math_trigger, notes=notes, symbol=sym)

        # 2. Evaluate AI news sentinel trigger if an analysis was provided
        if news_analysis is not None:
            # Check systemic threat first
            if self.is_systemic_threat(news_analysis, headline):
                notes = (
                    f"Global Systemic Threat Detected: Level={news_analysis.threat_level}, "
                    f"Confidence={news_analysis.confidence:.2f}, Reason: {news_analysis.reasoning}"
                )
                global_records = await self.execute_global_liquidation(TriggerReason.GLOBAL_MACRO_BREAKER, notes=notes)
                return global_records[0] if global_records else None

            # Asset-specific trigger
            ai_trigger = self.evaluate_ai_trigger_for_symbol(news_analysis, symbol=sym, headline=headline)
            if ai_trigger is not None:
                notes = (
                    f"AI Threat Detected: Level={news_analysis.threat_level}, "
                    f"Confidence={news_analysis.confidence:.2f}, Reason: {news_analysis.reasoning}"
                )
                return await self.execute_liquidation(ai_trigger, notes=notes, symbol=sym)

        return None
