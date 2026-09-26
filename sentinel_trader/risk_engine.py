"""Risk Engine module for Sentinel-Trader.

SAFETY INVARIANTS:
1. 100% deterministic code.
2. The AI model NEVER determines order size or calculates prices.
3. Liquidates the token balance into USDT immediately upon breach of:
   - Hard mathematical take-profit or stop-loss limits.
   - High-confidence emergency events identified by AI Sentinel.
"""

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field

from sentinel_trader.ai_sentinel import NewsAnalysis
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient, OrderReceipt

logger = logging.getLogger("sentinel.risk")


class TriggerReason(str, Enum):
    """Categorization of liquidation triggers."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    AI_PANIC_SELL = "AI_PANIC_SELL"
    MANUAL_EMERGENCY = "MANUAL_EMERGENCY"


class PositionStatus(BaseModel):
    """Real-time telemetry of monitored position."""

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
    """100% deterministic risk management engine."""

    def __init__(self, config: Config, exchange: ExchangeClient, initial_entry_price: Optional[float] = None):
        self.config = config
        self.exchange = exchange
        self.entry_price: Optional[float] = initial_entry_price
        self.is_liquidated: bool = False
        self.last_liquidation: Optional[LiquidationRecord] = None

    def set_entry_price(self, price: float) -> None:
        """Set the cost basis entry price for PnL calculation."""
        if price <= 0:
            raise ValueError(f"Entry price must be positive, got {price}")
        self.entry_price = price
        logger.info("[ENTRY PRICE ESTABLISHED] %s entry price locked at $%.4f", self.config.target_symbol, price)

    def calculate_pnl(self, current_price: float, balance: float = 0.0) -> Tuple[float, float]:
        """Calculate mathematical percentage and absolute USD PnL deterministically."""
        if not self.entry_price or self.entry_price <= 0:
            return 0.0, 0.0

        pnl_pct = (current_price - self.entry_price) / self.entry_price
        pnl_usd = (current_price - self.entry_price) * balance
        return pnl_pct, pnl_usd

    def get_limits(self) -> Dict[str, float]:
        """Return computed hard mathematical price boundaries."""
        if not self.entry_price or self.entry_price <= 0:
            return {"entry_price": 0.0, "take_profit_price": 0.0, "stop_loss_price": 0.0}
        return {
            "entry_price": self.entry_price,
            "take_profit_price": round(self.entry_price * (1.0 + self.config.take_profit_pct), 4),
            "stop_loss_price": round(self.entry_price * (1.0 - self.config.stop_loss_pct), 4),
        }

    def reset(self, initial_entry_price: Optional[float] = None) -> None:
        """Reset risk engine state for testing or new position monitoring."""
        self.entry_price = initial_entry_price
        self.is_liquidated = False
        self.last_liquidation = None
        if initial_entry_price:
            logger.info("[RISK ENGINE RESET] New entry price: $%.4f", initial_entry_price)

    def evaluate_mathematical_limits(self, current_price: float) -> Optional[TriggerReason]:
        """Check if hard mathematical take-profit or stop-loss limits are breached.

        100% deterministic mathematical evaluation.
        """
        if not self.entry_price or self.entry_price <= 0:
            return None

        pnl_pct = (current_price - self.entry_price) / self.entry_price

        # Check Take-Profit threshold (e.g. +10%)
        if pnl_pct >= self.config.take_profit_pct:
            logger.warning(
                "[TAKE-PROFIT BREACHED] PnL %+.2f%% exceeds target limit +%.2f%%",
                pnl_pct * 100,
                self.config.take_profit_pct * 100,
            )
            return TriggerReason.TAKE_PROFIT

        # Check Stop-Loss threshold (e.g. -4%)
        if pnl_pct <= -self.config.stop_loss_pct:
            logger.warning(
                "[STOP-LOSS BREACHED] PnL %+.2f%% breaches stop loss limit -%.2f%%",
                pnl_pct * 100,
                self.config.stop_loss_pct * 100,
            )
            return TriggerReason.STOP_LOSS

        return None

    def evaluate_ai_trigger(self, analysis: NewsAnalysis) -> Optional[TriggerReason]:
        """Evaluate whether an AI news analysis qualifies for emergency liquidation.

        SAFETY INVARIANT:
        The AI model NEVER decides execution size or routes orders.
        The Risk Engine strictly evaluates if the classification satisfies deterministic gating rules.
        """
        is_panic = analysis.action == "PANIC_SELL"
        is_high_threat = analysis.threat_level in ["HIGH", "CRITICAL"]
        meets_confidence = analysis.confidence >= self.config.panic_sell_confidence_threshold

        if is_panic and is_high_threat and meets_confidence:
            logger.critical(
                "[AI EMERGENCY TRIGGERED] Classification: %s | Threat: %s | Confidence: %.2f | Reasoning: %s",
                analysis.action,
                analysis.threat_level,
                analysis.confidence,
                analysis.reasoning,
            )
            return TriggerReason.AI_PANIC_SELL

        return None

    async def get_position_status(self, current_price: float) -> PositionStatus:
        """Fetch current balance and compute real-time position status."""
        balance = await self.exchange.get_balance(self.config.base_asset)
        pnl_pct, pnl_usd = self.calculate_pnl(current_price, balance)
        position_value = balance * current_price

        return PositionStatus(
            symbol=self.config.target_symbol,
            entry_price=self.entry_price or current_price,
            current_price=current_price,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            balance_base=balance,
            position_value_usd=position_value,
            is_liquidated=self.is_liquidated,
        )

    async def execute_liquidation(
        self,
        trigger: TriggerReason,
        notes: str,
    ) -> LiquidationRecord:
        """Instantly liquidate the token balance into USDT.

        SAFETY INVARIANTS:
        - Sizing is 100% deterministic (fetches total free base balance).
        - AI model has zero influence over execution size or price.
        """
        if self.is_liquidated:
            logger.warning("Liquidation already executed previously. Skipping.")
            return self.last_liquidation

        sym = self.config.target_symbol
        base_asset = self.config.base_asset

        # Sizing is 100% deterministic: query actual free balance from exchange
        balance_to_sell = await self.exchange.get_balance(base_asset)
        if balance_to_sell <= 0.0:
            logger.warning("Zero %s balance available to liquidate.", base_asset)
            # Fetch price for auditing
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
                "[EXECUTION: LIQUIDATING POSITION] Trigger: %s | Selling %.4f %s into %s",
                trigger.value,
                balance_to_sell,
                base_asset,
                self.config.quote_asset,
            )
            receipt = await self.exchange.liquidate_to_usdt(sym, amount=balance_to_sell)

        exit_price = receipt.price
        entry_price = self.entry_price or exit_price
        pnl_pct, pnl_usd = self.calculate_pnl(exit_price, receipt.amount)

        record = LiquidationRecord(
            trigger=trigger,
            symbol=sym,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl_pct=pnl_pct,
            realized_pnl_usd=pnl_usd,
            amount_liquidated=receipt.amount,
            receipt=receipt,
            notes=notes,
        )

        self.is_liquidated = True
        self.last_liquidation = record

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
            entry_price,
            exit_price,
            pnl_pct * 100,
            pnl_usd,
            receipt.amount,
            base_asset,
            receipt.cost,
            self.config.quote_asset,
            notes,
        )

        return record

    async def evaluate_tick(
        self,
        current_price: float,
        news_analysis: Optional[NewsAnalysis] = None,
    ) -> Optional[LiquidationRecord]:
        """Evaluate mathematical limits and optional AI news trigger on each tick.

        Returns LiquidationRecord if liquidation occurred, None otherwise.
        """
        if self.is_liquidated:
            return None

        # Auto-initialize entry price if not set yet
        if self.entry_price is None:
            self.set_entry_price(current_price)

        # 1. Evaluate hard mathematical limits first (highest priority, 100% deterministic)
        math_trigger = self.evaluate_mathematical_limits(current_price)
        if math_trigger is not None:
            pnl_pct, _ = self.calculate_pnl(current_price)
            notes = (
                f"Mathematical threshold breached: PnL {pnl_pct * 100:+.2f}% "
                f"(TP: +{self.config.take_profit_pct * 100:.1f}%, SL: -{self.config.stop_loss_pct * 100:.1f}%)"
            )
            return await self.execute_liquidation(math_trigger, notes=notes)

        # 2. Evaluate AI news sentinel trigger if an analysis was provided
        if news_analysis is not None:
            ai_trigger = self.evaluate_ai_trigger(news_analysis)
            if ai_trigger is not None:
                notes = (
                    f"AI Threat Detected: Level={news_analysis.threat_level}, "
                    f"Confidence={news_analysis.confidence:.2f}, Reason: {news_analysis.reasoning}"
                )
                return await self.execute_liquidation(ai_trigger, notes=notes)

        return None
