"""Discord-only notification subsystem for Sentinel-Trader.

Connects to Discord incoming webhooks using non-blocking aiohttp.
Dispatches alerts asynchronously in background tasks with a 2.5-second timeout
so network lag never delays price monitoring or order execution.
Silently bypasses alerts when DISCORD_WEBHOOK_URL is unset.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import aiohttp

from sentinel_trader.config import Config

logger = logging.getLogger("sentinel.notifier")

# Standard Discord Embed Colors (Hex Decimal Representation)
COLOR_GREEN = 0x2ECC71      # Take-Profit (Success / Gains)
COLOR_RED = 0xE74C3C        # Stop-Loss (Standard Protection)
COLOR_DARK_RED = 0x992D22   # AI Panic-Sell (Catastrophic Threat)
COLOR_ORANGE = 0xE67E22     # Threat Detected / Time Decay Scratch (Warning / Capital Recycled)
COLOR_YELLOW = 0xF1C40F     # Orange/Yellow indicator
COLOR_BLUE = 0x3498DB       # System Startup (Informational)
COLOR_GRAY = 0x95A5A6       # Hourly Summary (Neutral / Flat / 0 Trades)


class DiscordNotifier:
    """Asynchronous, non-blocking Discord webhook notifier."""

    DEFAULT_TIMEOUT_SECONDS: float = 2.5

    def __init__(
        self,
        config: Optional[Config] = None,
        webhook_url: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.config = config or Config()
        if webhook_url is not None:
            self.webhook_url = webhook_url.strip()
        else:
            self.webhook_url = (self.config.discord_webhook_url or "").strip()
        self.timeout_seconds: float = timeout_seconds

    @property
    def is_enabled(self) -> bool:
        """Return True if a valid Discord webhook URL is configured."""
        return bool(self.webhook_url and self.webhook_url.startswith(("https://", "http://")))

    # ------------------------------------------------------------------
    # Embed Constructors
    # ------------------------------------------------------------------

    def build_liquidation_embed(
        self,
        symbol: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        notes: str,
    ) -> Dict[str, Any]:
        """Construct rich Discord embed for position liquidation event."""
        reason_upper = str(reason).upper()

        if "TAKE_PROFIT" in reason_upper:
            color = COLOR_GREEN
            title = f"💰 [TAKE PROFIT] Position Liquidated: {symbol}"
            description = f"Spot position in **{symbol}** has been liquidated into USDT."
        elif "AI_PANIC" in reason_upper:
            color = COLOR_DARK_RED
            title = f"🚨 [AI PANIC SELL] Emergency Liquidation: {symbol}"
            description = f"Spot position in **{symbol}** has been liquidated into USDT."
        elif "TIME_DECAY_SCRATCH" in reason_upper or "SCRATCH" in reason_upper:
            color = COLOR_ORANGE  # Orange/Yellow indicator (0xE67E22)
            title = f"⏳ [TIME DECAY SCRATCH] Capital Recycled: {symbol}"
            description = f"Capital recycled due to time decay. Spot position in **{symbol}** has been liquidated into USDT."
        else:
            color = COLOR_RED
            title = f"🛑 [STOP LOSS] Position Liquidated: {symbol}"
            description = f"Spot position in **{symbol}** has been liquidated into USDT."

        pnl_sign = "+" if pnl_pct >= 0 else ""
        fields: List[Dict[str, Any]] = [
            {"name": "Trigger Reason", "value": f"`{reason}`", "inline": True},
            {"name": "Realized PnL", "value": f"**{pnl_sign}{pnl_pct * 100:.2f}%**", "inline": True},
            {"name": "Entry Price", "value": f"${entry_price:.4f}", "inline": True},
            {"name": "Exit Price", "value": f"${exit_price:.4f}", "inline": True},
            {"name": "Audit Notes", "value": notes or "N/A", "inline": False},
        ]

        return {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Sentinel-Trader • Autonomous Risk Mitigation Bot"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def build_threat_embed(
        self,
        headline: str,
        threat_level: str,
        confidence: float,
        reasoning: str,
    ) -> Dict[str, Any]:
        """Construct rich Discord embed for high-threat crypto news event."""
        fields: List[Dict[str, Any]] = [
            {"name": "Threat Level", "value": f"**{threat_level.upper()}**", "inline": True},
            {"name": "Action", "value": "`PANIC_SELL`", "inline": True},
            {"name": "Confidence", "value": f"**{confidence:.0%}** ({confidence:.2f})", "inline": True},
            {"name": "AI Reasoning", "value": reasoning or "No reasoning provided.", "inline": False},
        ]

        return {
            "title": "⚠️ [AI THREAT DETECTED] High-Risk Crypto Event",
            "description": f"**Headline:** {headline}",
            "color": COLOR_ORANGE,
            "fields": fields,
            "footer": {"text": "Sentinel-Trader • AI Sentinel Ingestion"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def build_startup_embed(
        self,
        symbol: str,
        tp_pct: float,
        sl_pct: float,
        sandbox_mode: bool,
    ) -> Dict[str, Any]:
        """Construct rich Discord embed for system boot / startup."""
        mode_str = "Binance Testnet (Sandbox)" if sandbox_mode else "LIVE PRODUCTION"
        fields: List[Dict[str, Any]] = [
            {"name": "Monitored Symbol", "value": f"**{symbol}**", "inline": True},
            {"name": "Operating Mode", "value": f"`{mode_str}`", "inline": True},
            {"name": "Hard Take-Profit", "value": f"`+{tp_pct * 100:.1f}%`", "inline": True},
            {"name": "Hard Stop-Loss", "value": f"`-{sl_pct * 100:.1f}%`", "inline": True},
        ]

        return {
            "title": "🛡️ [SYSTEM ONLINE] Sentinel-Trader Booted",
            "description": f"Autonomous risk mitigation bot has initialized and is actively protecting **{symbol}**.",
            "color": COLOR_BLUE,
            "fields": fields,
            "footer": {"text": "Sentinel-Trader • Startup Telemetry"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def build_position_entered_embed(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        rsi_value: float,
        tp_price: float,
        sl_price: float,
        tp_pct: float,
        sl_pct: float,
    ) -> Dict[str, Any]:
        """Construct rich Discord embed for position entry event."""
        fields: List[Dict[str, Any]] = [
            {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
            {"name": "Entry Price", "value": f"**${entry_price:.4f}**", "inline": True},
            {"name": "Quantity", "value": f"`{amount:.4f}`", "inline": True},
            {"name": "Trigger RSI(14)", "value": f"`{rsi_value:.2f}`", "inline": True},
            {"name": "Take-Profit Target", "value": f"`${tp_price:.4f}` (+{tp_pct * 100:.1f}%)", "inline": True},
            {"name": "Stop-Loss Target", "value": f"`${sl_price:.4f}` (-{sl_pct * 100:.1f}%)", "inline": True},
        ]

        return {
            "title": f"🔵 [POSITION ENTERED] Active Spot Trade: {symbol}",
            "description": f"Entry signal confirmed: Close > EMA(200) and RSI(14) crossed upward through 35. Position in **{symbol}** is actively protected.",
            "color": COLOR_BLUE,
            "fields": fields,
            "footer": {"text": "Sentinel-Trader • Active FSM Execution"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def build_hourly_summary_embed(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Construct rich Discord embed for hourly performance summary."""
        net_pnl_usd = float(metrics.get("total_realized_pnl_usd", 0.0))
        net_pnl_pct = float(metrics.get("total_realized_pnl_pct", 0.0))
        total_trades = int(metrics.get("total_trades", 0))

        # Color evaluation: Green if > 0, Red if < 0, Gray/Gold if 0 trades or flat PnL
        if total_trades == 0 or net_pnl_usd == 0.0:
            color = COLOR_GRAY
        elif net_pnl_usd > 0.0:
            color = COLOR_GREEN
        else:
            color = COLOR_RED

        # 1. Total Net PnL: +$XX.XX (+Y.YY%)
        pnl_sign = "+" if net_pnl_usd >= 0 else "-"
        pct_sign = "+" if net_pnl_pct >= 0 else "-"
        value_net = f"{pnl_sign}${abs(net_pnl_usd):.2f} ({pct_sign}{abs(net_pnl_pct):.2f}%)"

        # 2. Made / Lost: +$XX.XX won | -$YY.YY lost
        gross_profit = float(metrics.get("gross_profit_usd", 0.0))
        gross_loss = abs(float(metrics.get("gross_loss_usd", 0.0)))
        value_made_lost = f"+${gross_profit:.2f} won | -${gross_loss:.2f} lost"

        # 3. Past 1-Hour PnL: +$XX.XX
        pnl_1h = float(metrics.get("rolling_1h_pnl_usd", 0.0))
        h1_sign = "+" if pnl_1h >= 0 else "-"
        value_1h = f"{h1_sign}${abs(pnl_1h):.2f}"

        # 4. Win Rate: XX.X% (XW / YL across N trades)
        win_rate = float(metrics.get("win_rate_pct", 0.0))
        wins = int(metrics.get("wins", 0))
        losses = int(metrics.get("losses", 0))
        trades_count = int(metrics.get("total_trades", wins + losses))
        value_win_rate = f"{win_rate:.1f}% ({wins}W / {losses}L across {trades_count} trades)"

        # 5. Free USDT: $XXXX.XX
        free_usdt = float(metrics.get("free_usdt", 0.0))
        value_usdt = f"${free_usdt:,.2f}"

        # 6. Active Slots (X/3): Bulleted list of open coins and unrealized PnL, or 100% Cash
        active_slots = metrics.get("active_positions", {})
        active_count = int(metrics.get("active_slots_count", len(active_slots)))
        max_slots = int(metrics.get("max_slots", 3))
        slot_title = f"Active Slots ({active_count}/{max_slots})"

        if not active_slots or active_count == 0:
            value_slots = "100% Cash"
        else:
            bullets = []
            for sym, pos_data in active_slots.items():
                pnl_num = 0.0
                if hasattr(pos_data, "last_known_pnl"):
                    pnl_num = float(getattr(pos_data, "last_known_pnl", 0.0) or 0.0)
                elif isinstance(pos_data, dict):
                    raw_val = pos_data.get("last_known_pnl")
                    if raw_val is None:
                        raw_val = pos_data.get("unrealized_pnl_pct", pos_data.get("pnl_pct", 0.0))
                    pnl_num = float(raw_val or 0.0)

                # Format percentage: if |pnl_num| <= 1.0 and non-zero, treat as decimal
                pct_val = pnl_num * 100.0 if abs(pnl_num) <= 1.0 and pnl_num != 0.0 else pnl_num
                sign_sym = "+" if pct_val >= 0 else ""

                phase_val = None
                if hasattr(pos_data, "exit_phase") and getattr(pos_data, "exit_phase"):
                    phase_val = getattr(pos_data, "exit_phase")
                elif isinstance(pos_data, dict) and pos_data.get("exit_phase"):
                    phase_val = pos_data["exit_phase"]

                phase_str = f" [{phase_val}]" if phase_val else ""
                bullets.append(f"• **{sym}**{phase_str}: {sign_sym}{pct_val:.2f}%")
            value_slots = "\n".join(bullets) if bullets else "100% Cash"

        fields = [
            {"name": "Total Net PnL", "value": value_net, "inline": True},
            {"name": "Made / Lost", "value": value_made_lost, "inline": True},
            {"name": "Past 1-Hour PnL", "value": value_1h, "inline": True},
            {"name": "Win Rate", "value": value_win_rate, "inline": True},
            {"name": "Free USDT", "value": value_usdt, "inline": True},
            {"name": slot_title, "value": value_slots, "inline": False},
        ]

        return {
            "title": "⏱️ Sentinel-Trader Hourly Performance Report",
            "description": "Hourly account performance and risk telemetry summary.",
            "color": color,
            "fields": fields,
            "footer": {"text": "Sentinel-Trader • Hourly Telemetry"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # HTTP Dispatch Logic
    # ------------------------------------------------------------------

    async def send_payload(self, payload: Dict[str, Any]) -> bool:
        """Send payload to Discord webhook URL with strict timeout protection."""
        if not self.is_enabled:
            logger.debug("Discord webhook URL not configured. Notification bypassed silently.")
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            headers = {"Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload, headers=headers) as resp:
                    if resp.status in (200, 204):
                        logger.info("Discord webhook alert successfully delivered (HTTP %s).", resp.status)
                        return True
                    else:
                        resp_text = await resp.text()
                        logger.warning(
                            "Discord webhook rejected (HTTP %s): %s",
                            resp.status,
                            resp_text[:200],
                        )
                        return False
        except asyncio.TimeoutError:
            logger.warning("Discord webhook timed out after %.1fs (non-blocking).", self.timeout_seconds)
            return False
        except Exception as e:
            logger.warning("Failed to dispatch Discord webhook: %s", e)
            return False

    def _dispatch_background(self, payload: Dict[str, Any]) -> Optional[asyncio.Task]:
        """Dispatch a webhook payload inside a non-blocking background task.

        Guarantees that price loops and order execution are never blocked.
        """
        if not self.is_enabled:
            return None

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self.send_payload(payload),
                name="sentinel_discord_webhook",
            )
            return task
        except RuntimeError:
            # If no active event loop in current thread, log and return None without crashing
            logger.debug("No active event loop found for background webhook dispatch.")
            return None

    # ------------------------------------------------------------------
    # Public Notification Methods (Non-Blocking)
    # ------------------------------------------------------------------

    def notify_liquidation(
        self,
        symbol: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        notes: str,
    ) -> Optional[asyncio.Task]:
        """Asynchronously dispatch liquidation event alert without blocking."""
        embed = self.build_liquidation_embed(
            symbol=symbol,
            reason=reason,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            notes=notes,
        )
        return self._dispatch_background({"embeds": [embed]})

    def notify_threat_detected(
        self,
        headline: str,
        threat_level: str,
        confidence: float,
        reasoning: str,
    ) -> Optional[asyncio.Task]:
        """Asynchronously dispatch AI emergency threat alert without blocking."""
        embed = self.build_threat_embed(
            headline=headline,
            threat_level=threat_level,
            confidence=confidence,
            reasoning=reasoning,
        )
        return self._dispatch_background({"embeds": [embed]})

    def notify_startup(
        self,
        symbol: str,
        tp_pct: float,
        sl_pct: float,
        sandbox_mode: bool,
    ) -> Optional[asyncio.Task]:
        """Asynchronously dispatch bot startup confirmation without blocking."""
        embed = self.build_startup_embed(
            symbol=symbol,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            sandbox_mode=sandbox_mode,
        )
        return self._dispatch_background({"embeds": [embed]})

    def notify_position_entered(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        rsi_value: float,
        tp_price: float,
        sl_price: float,
        tp_pct: float,
        sl_pct: float,
    ) -> Optional[asyncio.Task]:
        """Asynchronously dispatch position entry alert without blocking."""
        embed = self.build_position_entered_embed(
            symbol=symbol,
            entry_price=entry_price,
            amount=amount,
            rsi_value=rsi_value,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )
        return self._dispatch_background({"embeds": [embed]})

    def notify_hourly_summary(self, metrics: Dict[str, Any]) -> Optional[asyncio.Task]:
        """Asynchronously dispatch hourly performance report via fire-and-forget background task."""
        embed = self.build_hourly_summary_embed(metrics)
        return self._dispatch_background({"embeds": [embed]})

    # ------------------------------------------------------------------
    # Awaitable Async Methods (for tests and direct delivery confirmation)
    # ------------------------------------------------------------------

    async def notify_liquidation_async(
        self,
        symbol: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        notes: str,
    ) -> bool:
        """Awaitable liquidation notification (convenient for test verification)."""
        embed = self.build_liquidation_embed(symbol, reason, entry_price, exit_price, pnl_pct, notes)
        return await self.send_payload({"embeds": [embed]})

    async def notify_threat_detected_async(
        self,
        headline: str,
        threat_level: str,
        confidence: float,
        reasoning: str,
    ) -> bool:
        """Awaitable threat notification (convenient for test verification)."""
        embed = self.build_threat_embed(headline, threat_level, confidence, reasoning)
        return await self.send_payload({"embeds": [embed]})

    async def notify_startup_async(
        self,
        symbol: str,
        tp_pct: float,
        sl_pct: float,
        sandbox_mode: bool,
    ) -> bool:
        """Awaitable startup notification (convenient for test verification)."""
        embed = self.build_startup_embed(symbol, tp_pct, sl_pct, sandbox_mode)
        return await self.send_payload({"embeds": [embed]})

    async def notify_position_entered_async(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        rsi_value: float,
        tp_price: float,
        sl_price: float,
        tp_pct: float,
        sl_pct: float,
    ) -> bool:
        """Awaitable position entry notification."""
        embed = self.build_position_entered_embed(
            symbol=symbol,
            entry_price=entry_price,
            amount=amount,
            rsi_value=rsi_value,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )
        return await self.send_payload({"embeds": [embed]})

    async def notify_hourly_summary_async(self, metrics: Dict[str, Any]) -> bool:
        """Awaitable hourly summary notification for test verification and direct dispatch."""
        embed = self.build_hourly_summary_embed(metrics)
        return await self.send_payload({"embeds": [embed]})
