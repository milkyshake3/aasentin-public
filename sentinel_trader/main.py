"""Main orchestrator for Sentinel-Trader.

Implements an autonomous continuous Finite State Machine (FSM) trading bot:
    CASH_SCANNING -> ENTERING -> IN_POSITION -> COOLDOWN -> CASH_SCANNING

States:
1. CASH_SCANNING:
   - Evaluates OHLCV candles (15m timeframe) for EMA(200) trend filter and RSI(14) cross-up through 35.
   - Enforces AI Veto (blocks entry if any HIGH or CRITICAL news threats in the last 2 hours).
2. ENTERING:
   - Sizes position allocating 25% of free USDT (minimum $15).
   - Executes precision-truncated market buy via idempotent order loop.
   - Dispatches Discord Position Entered card with SL/TP targets and trigger RSI.
3. IN_POSITION:
   - Actively protects position with mathematical Take-Profit (+5%), Stop-Loss (-2.5%),
     and AI Panic Sell triggers. Liquidates to USDT upon trigger breach.
4. COOLDOWN:
   - Enforces a 15-minute sleep period post-liquidation to avoid high-volatility chop re-entry,
     then loops back to CASH_SCANNING.
"""

import argparse
import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import os
import re
import signal
import sys
import time
from typing import Any, List, Optional, Tuple

# Ensure project root is always in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Auto-detect and bootstrap local virtual environment if run from global python
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    candidates = [
        os.path.join(root_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(root_dir, ".venv", "bin", "python"),
    ]
    for venv_py in candidates:
        if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_py).lower():
            import subprocess
            cmd = [venv_py, "-m", "sentinel_trader.main"] + sys.argv[1:]
            sys.exit(subprocess.call(cmd, cwd=root_dir))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional rich library support with graceful plain-text fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console(highlight=False, legacy_windows=False)
except ImportError:
    HAS_RICH = False

    class FallbackConsole:
        def print(self, text="", *args, **kwargs):
            if hasattr(text, "plain"):
                text = text.plain
            clean = re.sub(r"\[/?(?:bold|cyan|green|red|yellow|magenta|dim|/)?[^\]]*\]", "", str(text))
            print(clean)

    console = FallbackConsole()

import pandas as pd
from sentinel_trader.ai_sentinel import AISentinel, NewsAnalysis
from sentinel_trader.analytics import (
    DEFAULT_TRADE_HISTORY_FILE,
    DEFAULT_MOCK_TRADE_HISTORY_FILE,
    calculate_metrics,
    record_trade,
    resolve_trade_history_path,
)
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.news_watcher import NewsWatcher, NewsArticle
from sentinel_trader.notifier import DiscordNotifier
from sentinel_trader.risk_engine import RiskEngine, TriggerReason
from sentinel_trader.state_store import DEFAULT_STATE_FILE, clear_state, load_state, save_state
from sentinel_trader.strategy import calculate_indicators, check_entry_signal, scan_universe_for_signals, Candidate
from sentinel_trader.universe_screener import UniverseScreener, get_dynamic_universe


def timeframe_to_seconds(timeframe: str) -> int:
    """Parse timeframe string (e.g. '15m', '1h', '1d', '30s') into integer seconds."""
    if not timeframe:
        return 15 * 60
    tf = timeframe.strip().lower()
    unit = tf[-1]
    try:
        val = int(tf[:-1])
    except ValueError:
        return 15 * 60

    if unit == "m":
        return val * 60
    elif unit == "h":
        return val * 3600
    elif unit == "d":
        return val * 86400
    elif unit == "s":
        return val
    elif unit == "w":
        return val * 604800
    return 15 * 60

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sentinel.orchestrator")


class BotState(str, Enum):
    """Finite State Machine states for active trading lifecycle."""

    CASH_SCANNING = "CASH_SCANNING"
    ENTERING = "ENTERING"
    IN_POSITION = "IN_POSITION"
    COOLDOWN = "COOLDOWN"


class SentinelTrader:
    """Continuous active trading bot managed by a 4-state Finite State Machine."""

    def __init__(
        self,
        config: Optional[Config] = None,
        scenario: Optional[str] = None,
        initial_state: Optional[BotState] = None,
        max_cycles: Optional[int] = None,
        state_file: Optional[str] = None,
        trade_history_file: Optional[str] = None,
    ):
        self.config = config or Config()
        self.scenario = scenario or "default"
        self.initial_state = initial_state
        self.max_cycles = max_cycles
        self.state_file = state_file or DEFAULT_STATE_FILE
        self.trade_history_file = trade_history_file or resolve_trade_history_path(
            getattr(self.config, "trade_history_file", None),
            config=self.config,
        )
        self.exchange = ExchangeClient(self.config)
        self.news_watcher = NewsWatcher(self.config)
        self.ai_sentinel = AISentinel(self.config)
        self.notifier = DiscordNotifier(self.config)
        self.screener = UniverseScreener(cache_ttl_seconds=86400)
        self.risk_engine = RiskEngine(
            self.config,
            self.exchange,
            initial_entry_price=self.config.entry_price,
            notifier=self.notifier,
            trade_history_file=self.trade_history_file,
        )

        self.state: BotState = BotState.CASH_SCANNING
        self._shutdown_event = asyncio.Event()
        self._latest_price: Optional[float] = None
        self._latest_rsi: float = 35.0
        self._pending_symbol: Optional[str] = None
        self._running = False
        self._price_tick_count = 0
        self._news_tick_count = 0
        self._scan_tick_count = 0
        self._cycle_count = 0
        self._entry_timestamp: Optional[str] = None
        self._client_order_id: Optional[str] = None
        self._news_history: List[Tuple[NewsArticle, NewsAnalysis, datetime]] = []

    def transition_to(self, new_state: BotState) -> None:
        """Execute and log an FSM state transition."""
        old_state = self.state
        self.state = new_state
        logger.info("[FSM STATE TRANSITION] %s -> %s", old_state.value, new_state.value)
        console.print(
            f"\n[bold cyan][FSM TRANSITION][/bold cyan] [yellow]{old_state.value}[/yellow] "
            f"➜ [bold green]{new_state.value}[/bold green]\n"
        )
        if old_state == BotState.COOLDOWN and new_state == BotState.CASH_SCANNING:
            self._cycle_count += 1

    def record_news_analysis(self, article: NewsArticle, analysis: NewsAnalysis) -> None:
        """Record analyzed headline for rolling 2-hour AI Veto evaluation."""
        now = datetime.now(timezone.utc)
        self._news_history.append((article, analysis, now))
        cutoff = now - timedelta(hours=24)
        self._news_history = [(art, an, ts) for art, an, ts in self._news_history if ts >= cutoff]

    def has_active_ai_veto(self, lookback_hours: float = 2.0) -> Tuple[bool, Optional[str]]:
        """Verify whether any news within the lookback window has threat_level in HIGH or CRITICAL."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        recent_threats = [
            (art, an) for art, an, ts in self._news_history
            if ts >= cutoff and an.threat_level in ["HIGH", "CRITICAL"]
        ]
        if recent_threats:
            last_art, last_threat = recent_threats[-1]
            reason = f"{last_threat.threat_level} threat from {last_art.source}: '{last_art.headline}'"
            return True, reason
        return False, None

    def print_banner(self) -> None:
        """Display startup information and active risk boundaries."""
        mode_text = (
            "[bold green]BINANCE TESTNET (SANDBOX)[/bold green]"
            if self.config.binance_sandbox
            else "[bold red]LIVE PRODUCTION (CAUTION)[/bold red]"
        )
        scenario_text = (
            f"[bold cyan]{self.scenario.upper()}[/bold cyan]"
            if self.exchange.is_mock
            else "[dim]N/A (LIVE)[/dim]"
        )

        banner_text = Text()
        banner_text.append("[SENTINEL-TRADER] AUTONOMOUS ACTIVE FSM TRADING BOT\n", style="bold cyan")
        banner_text.append(f"Operating Mode       : {mode_text}\n")
        banner_text.append(f"Mock Scenario        : {scenario_text}\n")
        banner_text.append(f"Target Pair          : [bold yellow]{self.config.target_symbol}[/bold yellow]\n")
        banner_text.append(f"Candle Timeframe     : [bold]{self.config.timeframe}[/bold]\n")
        banner_text.append(f"Capital Allocation   : [bold]{self.config.allocation_pct * 100:.0f}%[/bold] of free USDT (min $15)\n")
        banner_text.append(f"Hard Take-Profit     : [bold green]+{self.config.take_profit_pct * 100:.1f}%[/bold green]\n")
        banner_text.append(f"Hard Stop-Loss       : [bold red]-{self.config.stop_loss_pct * 100:.1f}%[/bold red]\n")
        banner_text.append(f"Post-Exit Cooldown   : [bold]{self.config.cooldown_minutes} minutes[/bold]\n")
        banner_text.append(f"AI Provider          : [bold magenta]{self.config.llm_provider.upper()}[/bold magenta] ({self.config.get_effective_llm_model()})\n")
        banner_text.append(f"Panic Sell Threshold : [bold]{self.config.panic_sell_confidence_threshold:.0%}[/bold] confidence\n")
        banner_text.append(f"Initial State        : [bold cyan]{self.state.value}[/bold cyan]\n")

        console.print(Panel(banner_text, title="System Initialized", border_style="cyan"))

    def render_position_table(self, status) -> None:
        """Render live position status telemetry table to console."""
        table = Table(title="Live Position Telemetry", border_style="blue")
        table.add_column("Symbol", justify="center", style="cyan", no_wrap=True)
        table.add_column("Current Price", justify="right", style="bold")
        table.add_column("Entry Price", justify="right")
        table.add_column("PnL (%)", justify="right")
        table.add_column("Holding Amount", justify="right")
        table.add_column("Position Value", justify="right")
        table.add_column("State", justify="center")

        pnl_style = "bold green" if status.pnl_pct >= 0 else "bold red"
        status_text = "[red]LIQUIDATED[/red]" if status.is_liquidated else "[green]PROTECTED[/green]"

        table.add_row(
            status.symbol,
            f"${status.current_price:.4f}",
            f"${status.entry_price:.4f}",
            f"[{pnl_style}]{status.pnl_pct * 100:+.2f}%[/{pnl_style}]",
            f"{status.balance_base:.4f} {self.config.base_asset}",
            f"${status.position_value_usd:.2f}",
            status_text,
        )
        console.print(table)

    async def initialize(self) -> None:
        """Connect to exchange, load markets, and initialize initial state."""
        await self.exchange.initialize()
        self._latest_price = await self.exchange.get_ticker_price()

        # Check crash recovery persistence
        saved_state = load_state(self.state_file)
        recovered = False

        if saved_state and saved_state.get("screener_cache"):
            self.screener.set_cache(saved_state["screener_cache"], saved_state.get("screener_timestamp"))

        if (
            self.initial_state is None
            and self.config.entry_price is None
            and not (self.scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe"])
        ):
            if saved_state and "active_positions" in saved_state and saved_state["active_positions"]:
                active_pos_dict = saved_state["active_positions"]
                for sym, pos_info in active_pos_dict.items():
                    base_sym = sym.split("/")[0]
                    base_balance = await self.exchange.get_balance(base_sym)
                    if base_balance > 0.0:
                        entry_p = float(pos_info.get("entry_price") or self._latest_price or 150.0)
                        holding_amt = float(pos_info.get("amount") or pos_info.get("holding_amount") or base_balance)
                        order_id = pos_info.get("order_id") or pos_info.get("client_order_id")
                        entry_ts = pos_info.get("entry_timestamp")

                        self.risk_engine.open_position(
                            symbol=sym,
                            entry_price=entry_p,
                            amount=holding_amt,
                            order_id=order_id,
                            entry_timestamp=entry_ts,
                        )
                        if sym == self.config.target_symbol:
                            self._entry_timestamp = entry_ts
                            self._client_order_id = order_id
                        recovered = True

                if recovered:
                    self.state = BotState.IN_POSITION
                    logger.critical(
                        "[CRASH RECOVERY ACTIVE] Detected %d active persistent slots in %s. "
                        "Rehydrating RiskEngine. Skipping CASH_SCANNING and resuming monitoring.",
                        len(self.risk_engine.active_positions),
                        self.state_file,
                    )
                    console.print(
                        f"\n[bold red][CRASH RECOVERY][/bold red] Resumed [bold green]IN_POSITION[/bold green] "
                        f"from {self.state_file} with {len(self.risk_engine.active_positions)} active slots "
                        f"| Skipping CASH_SCANNING & duplicate buy orders.\n"
                    )
                else:
                    logger.warning(
                        "[CRASH RECOVERY] Saved state indicated active positions, but free balances are 0.0. "
                        "Positions were likely liquidated offline. Clearing stale state file."
                    )
                    clear_state(self.state_file)
            elif saved_state and saved_state.get("state") in (BotState.IN_POSITION.value, "IN_POSITION"):
                base_balance = await self.exchange.get_balance(self.config.base_asset)
                if base_balance > 0.0:
                    entry_p = float(saved_state.get("entry_price") or self._latest_price or 150.0)
                    holding_amt = float(saved_state.get("holding_amount") or base_balance)
                    self._entry_timestamp = saved_state.get("entry_timestamp")
                    self._client_order_id = saved_state.get("client_order_id")

                    self.risk_engine.rehydrate(entry_price=entry_p, holding_amount=holding_amt)
                    self.state = BotState.IN_POSITION
                    recovered = True
                    logger.critical(
                        "[CRASH RECOVERY ACTIVE] Detected persistent state IN_POSITION. Free balance: %.4f %s. "
                        "Rehydrating RiskEngine (Entry: $%.4f). Skipping CASH_SCANNING and resuming monitoring.",
                        base_balance,
                        self.config.base_asset,
                        entry_p,
                    )
                    console.print(
                        f"\n[bold red][CRASH RECOVERY][/bold red] Resumed [bold green]IN_POSITION[/bold green] "
                        f"from {self.state_file} | Entry: ${entry_p:.4f} | Size: {base_balance:.4f} {self.config.base_asset} "
                        f"| Skipping CASH_SCANNING & duplicate buy orders.\n"
                    )
                else:
                    logger.warning(
                        "[CRASH RECOVERY] Saved state indicated IN_POSITION, but free %s balance is 0.0. "
                        "Position was likely liquidated offline. Clearing stale state file.",
                        self.config.base_asset,
                    )
                    clear_state(self.state_file)

        if not recovered:
            # Determine starting FSM state
            if self.initial_state is not None:
                self.state = self.initial_state
            elif self.config.entry_price is not None:
                self.risk_engine.set_entry_price(self.config.entry_price)
                self.state = BotState.IN_POSITION
            elif self.scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe"]:
                # Single-exit benchmark scenarios begin in IN_POSITION
                self.risk_engine.set_entry_price(self._latest_price)
                self.state = BotState.IN_POSITION
            else:
                self.state = BotState.CASH_SCANNING

        self.print_banner()

        # Seed news deduplication cache on cold boot before entering trading or monitoring loops
        seeded = await self.news_watcher.seed_cache_on_startup()
        if seeded > 0:
            console.print(
                f"[bold cyan][NEWS WARMUP][/bold cyan] Seeded cache with {seeded} existing headlines. "
                f"Only fresh articles published after startup will be evaluated."
            )

        # Dispatch Discord startup notification
        self.notifier.notify_startup(
            symbol=self.config.target_symbol,
            tp_pct=self.config.take_profit_pct,
            sl_pct=self.config.stop_loss_pct,
            sandbox_mode=self.config.binance_sandbox,
        )

    # =====================================================================
    # State 1: CASH_SCANNING
    # =====================================================================

    async def _handle_cash_scanning(self) -> None:
        """State 1: Scan market for entry signals using RSI(14) and EMA(200) with AI Veto."""
        self._scan_tick_count += 1
        try:
            # 1. Fetch dynamic universe using 24-hour volume screener
            universe = await self.screener.get_dynamic_universe(
                self.exchange,
                top_n=self.config.dynamic_universe_size,
            )

            chosen_symbol = self.config.target_symbol
            has_signal = False
            curr_close = self._latest_price or 150.0
            curr_rsi = 35.0
            curr_ema = 150.0
            curr_atr = 0.0

            # Single-pair test override (when mock OHLCV is injected directly)
            if getattr(self.exchange, "_mock_ohlcv", None) is not None:
                df = await self.exchange.fetch_ohlcv_dataframe(
                    self.config.target_symbol,
                    timeframe=self.config.timeframe,
                    limit=200,
                )
                if self.exchange.is_mock and self.scenario == "fsm_cycle" and self._scan_tick_count >= 2:
                    if len(df) >= 200:
                        for k in range(18):
                            df.loc[df.index[-20 + k], "close"] = 150.0 - (k * 1.5)
                        df.loc[df.index[-3], "close"] = 125.0
                        df.loc[df.index[-2], "close"] = 152.0

                has_signal = check_entry_signal(
                    df,
                    rsi_threshold=self.config.rsi_oversold_threshold,
                    ema_period=self.config.ema_period,
                )
                chosen_symbol = self.config.target_symbol
                if not df.empty and "close" in df and len(df) >= 2:
                    curr_close = float(df["close"].iloc[-2])
                    curr_rsi = float(df["rsi"].iloc[-2]) if "rsi" in df and not pd.isna(df["rsi"].iloc[-2]) else 35.0
                    curr_ema = float(df["ema"].iloc[-2]) if "ema" in df and not pd.isna(df["ema"].iloc[-2]) else 150.0
                    curr_atr = float(df["atr14"].iloc[-2]) if "atr14" in df and not pd.isna(df["atr14"].iloc[-2]) else 0.0
            else:
                # Multi-candidate scanning across dynamic universe
                scan_symbols = [self.config.target_symbol] + [s for s in universe if s != self.config.target_symbol]
                candidates = await scan_universe_for_signals(
                    self.exchange,
                    scan_symbols,
                    timeframe=self.config.timeframe,
                    rsi_threshold=self.config.rsi_oversold_threshold,
                    ema_period=self.config.ema_period,
                )
                open_slots = set(self.risk_engine.active_positions.keys())
                available_candidates = [c for c in candidates if c.symbol not in open_slots]

                # Rich console diagnostic display
                if hasattr(candidates, "diagnostics") and candidates.diagnostics:
                    console.print(f"[bold cyan][SCAN DIAGNOSTICS][/bold cyan] {len(candidates.diagnostics)} pairs evaluated:")
                    for diag in candidates.diagnostics:
                        style = "bold green" if diag.get("passed") else "dim"
                        console.print(f"  [{style}]{diag['line']}[/{style}]")
                    if not available_candidates:
                        console.print(
                            f"  [bold yellow][SCAN RESULT][/bold yellow] 0/{len(candidates.diagnostics)} "
                            f"passed entry criteria. Maintaining 100% Cash."
                        )

                if available_candidates:
                    best = available_candidates[0]
                    chosen_symbol = best.symbol
                    curr_close = best.close
                    curr_rsi = best.rsi
                    curr_ema = best.ema
                    curr_atr = getattr(best, "atr", 0.0)
                    has_signal = True
                else:
                    curr_atr = 0.0
                    if self.exchange.is_mock and self.scenario == "fsm_cycle" and self._scan_tick_count >= 2:
                        df = await self.exchange.fetch_ohlcv_dataframe(self.config.target_symbol, timeframe=self.config.timeframe, limit=200)
                        if len(df) >= 200:
                            for k in range(18):
                                df.loc[df.index[-20 + k], "close"] = 150.0 - (k * 1.5)
                            df.loc[df.index[-3], "close"] = 125.0
                            df.loc[df.index[-2], "close"] = 152.0
                        has_signal = check_entry_signal(
                            df,
                            rsi_threshold=self.config.rsi_oversold_threshold,
                            ema_period=self.config.ema_period,
                        )
                        chosen_symbol = self.config.target_symbol
                        if not df.empty and "close" in df and len(df) >= 2:
                            curr_close = float(df["close"].iloc[-2])
                            curr_rsi = float(df["rsi"].iloc[-2]) if "rsi" in df and not pd.isna(df["rsi"].iloc[-2]) else 35.0
                            curr_ema = float(df["ema"].iloc[-2]) if "ema" in df and not pd.isna(df["ema"].iloc[-2]) else 150.0
                            curr_atr = float(df["atr14"].iloc[-2]) if "atr14" in df and not pd.isna(df["atr14"].iloc[-2]) else 0.0

            self._latest_price = curr_close

            # 2. Poll breaking news to maintain rolling 2-hour sentiment cache
            new_articles = await self.news_watcher.poll_new_headlines()
            for article in new_articles:
                analysis = await self.ai_sentinel.analyze(article.headline, chosen_symbol)
                self.record_news_analysis(article, analysis)

                if analysis.action == "PANIC_SELL" or analysis.threat_level in ["CRITICAL", "HIGH"]:
                    self.notifier.notify_threat_detected(
                        headline=article.headline,
                        threat_level=analysis.threat_level,
                        confidence=analysis.confidence,
                        reasoning=analysis.reasoning,
                    )

            free_usdt = await self.exchange.get_balance(self.config.quote_asset)

            console.print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"[bold cyan][CASH_SCANNING][/bold cyan] Screened {len(universe)} pairs | Candidate: {chosen_symbol}: ${curr_close:.4f} | "
                f"RSI(14): {curr_rsi:.1f} | EMA({self.config.ema_period}): ${curr_ema:.2f} | Free USDT: ${free_usdt:.2f} | "
                f"Signal: [{'bold green' if has_signal else 'dim'}]{has_signal}[/]"
            )

            # 3. Evaluate Entry Signal & AI Veto
            if has_signal:
                vetoed, veto_reason = self.has_active_ai_veto(lookback_hours=2.0)
                if vetoed:
                    logger.warning("[AI VETO] Entry for %s blocked: %s. Remaining in CASH_SCANNING.", chosen_symbol, veto_reason)
                    console.print(
                        f"   [yellow][AI VETO TRIGGERED][/yellow] Entry signal for {chosen_symbol} vetoed: {veto_reason}. "
                        f"Holding USDT."
                    )
                else:
                    self._latest_rsi = curr_rsi
                    self._latest_atr = curr_atr
                    self._pending_atr = curr_atr
                    self._pending_symbol = chosen_symbol
                    logger.info("[SIGNAL VALIDATED] Entry criteria met for %s with clear AI sentiment. Transitioning to ENTERING.", chosen_symbol)
                    self.transition_to(BotState.ENTERING)
                    return

        except Exception as e:
            logger.error("Error in CASH_SCANNING handler: %s", e)

        # Synchronize polling with candle boundary if running live; use fast poll_interval for mock/test runs
        if not self.exchange.is_mock:
            now = time.time()
            interval_sec = timeframe_to_seconds(self.config.timeframe)
            seconds_until_close = interval_sec - (now % interval_sec) + 2.0  # 2-second exchange buffer
            logger.info(
                "[CANDLE SYNC] Live mode: sleeping %.1fs until next %s candle boundary closes (buffer: 2s)...",
                seconds_until_close,
                self.config.timeframe,
            )
            sleep_duration = seconds_until_close
        else:
            sleep_duration = self.config.poll_interval_seconds

        # Sleep interruptible by shutdown event
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=sleep_duration,
            )
        except asyncio.TimeoutError:
            pass

    # =====================================================================
    # State 2: ENTERING
    # =====================================================================

    async def _handle_entering(self) -> None:
        """State 2: Query free USDT, allocate 25% (min $15), and execute market buy idempotently."""
        target_sym = getattr(self, "_pending_symbol", None) or self.config.target_symbol
        base_asset = target_sym.split("/")[0]
        logger.info("[ENTERING] Preparing spot market buy for %s...", target_sym)

        try:
            # 1. Query free quote balance and calculate volatility parity slot size
            free_usdt = await self.exchange.get_balance(self.config.quote_asset)
            total_equity = await self.exchange.get_total_equity()
            entry_est_price = await self.exchange.get_ticker_price(target_sym)
            pending_atr = getattr(self, "_pending_atr", None) or getattr(self, "_latest_atr", None)

            cost_usdt = self.risk_engine.calculate_slot_size(
                free_usdt=free_usdt,
                atr=pending_atr,
                entry_price=entry_est_price,
                total_equity=total_equity,
            )

            # Minimum trade size check
            if free_usdt < 15.0 or cost_usdt < 15.0:
                logger.warning(
                    "[INSUFFICIENT BALANCE] Free USDT ($%.2f) below minimum trade size ($15.00). Aborting entry.",
                    free_usdt,
                )
                console.print(
                    f"[red][ENTRY ABORTED][/red] Free USDT (${free_usdt:.2f}) < $15.00 minimum. "
                    f"Returning to CASH_SCANNING."
                )
                self._pending_symbol = None
                self.transition_to(BotState.CASH_SCANNING)
                await asyncio.sleep(self.config.poll_interval_seconds)
                return

            if cost_usdt > free_usdt:
                cost_usdt = free_usdt

            # 2. Dispatch market buy order via idempotent execution loop
            receipt = await self.exchange.execute_market_buy_idempotent(
                target_sym,
                cost_usdt=cost_usdt,
            )

            # 3. Record exact filled entry price and token amount
            entry_price = float(receipt.get("price") or receipt.get("average") or self._latest_price or 150.0)
            token_amount = float(receipt.get("amount") or receipt.get("filled") or 0.0)
            client_order_id = str(receipt.get("client_order_id") or receipt.get("order_id") or "")
            entry_ts = datetime.now(timezone.utc).isoformat()

            self._entry_timestamp = entry_ts
            self._client_order_id = client_order_id
            self._latest_price = entry_price

            # Open slot in RiskEngine with institutional quant mechanics
            pos = self.risk_engine.open_position(
                symbol=target_sym,
                entry_price=entry_price,
                amount=token_amount,
                order_id=client_order_id,
                entry_timestamp=entry_ts,
                atr=pending_atr,
            )

            # Atomically persist active positions state to disk
            save_state(
                state=BotState.IN_POSITION,
                active_positions=self.risk_engine.positions,
                screener_cache=self.screener.cached_universe or [],
                screener_timestamp=self.screener.cache_timestamp,
                entry_price=entry_price,
                holding_amount=token_amount,
                entry_timestamp=entry_ts,
                client_order_id=client_order_id,
                last_known_pnl=0.0,
                highest_price=pos.highest_price,
                atr_at_entry=pos.atr_at_entry,
                current_stop_price=pos.current_stop_price,
                candles_held=pos.candles_held,
                file_path=self.state_file,
                target_symbol=target_sym,
            )

            # 4. Calculate targets
            tp_price = round(entry_price * (1.0 + self.config.take_profit_pct), 4)
            sl_price = pos.current_stop_price
            rsi_val = getattr(self, "_latest_rsi", 35.0)

            # 5. Discord Alert: Send Blue "Position Entered" card
            self.notifier.notify_position_entered(
                symbol=target_sym,
                entry_price=entry_price,
                amount=token_amount,
                rsi_value=rsi_val,
                tp_price=tp_price,
                sl_price=sl_price,
                tp_pct=self.config.take_profit_pct,
                sl_pct=self.config.stop_loss_pct,
            )

            console.print(
                f"[bold green][POSITION ENTERED][/bold green] Bought {token_amount:.4f} {base_asset} "
                f"at ${entry_price:.4f} (Cost: ${receipt.get('cost', cost_usdt):.2f}) | "
                f"TP: ${tp_price:.4f} (+{self.config.take_profit_pct*100:.1f}%) | "
                f"SL: ${sl_price:.4f} (-{self.config.stop_loss_pct*100:.1f}%) | "
                f"Slots: {len(self.risk_engine.active_positions)}/{self.config.max_active_positions}"
            )

            # 6. Transition to IN_POSITION
            self._pending_symbol = None
            self.transition_to(BotState.IN_POSITION)

        except Exception as e:
            logger.error("Error during position entry execution: %s", e)
            self._pending_symbol = None
            self.transition_to(BotState.CASH_SCANNING)
            await asyncio.sleep(self.config.poll_interval_seconds)

    # =====================================================================
    # State 3: IN_POSITION
    # =====================================================================

    async def _handle_in_position(self) -> None:
        """State 3: Multi-slot monitoring (tick price, TP/SL, asset-specific news, and systemic macro breaker)."""
        try:
            active_slots = list(self.risk_engine.active_positions.values())
            if not active_slots and self.config.target_symbol in self.risk_engine.positions:
                active_slots = [self.risk_engine.positions[self.config.target_symbol]]

            # 1. Price Monitoring & Mathematical limits across active slots
            for pos in active_slots:
                sym = pos.symbol
                if self.exchange.is_mock:
                    self._price_tick_count += 1
                    if self.scenario == "take_profit":
                        pump_pct = 0.025 * self._price_tick_count
                        price = round(pos.entry_price * (1.0 + pump_pct), 4)
                        self.exchange.set_mock_price(sym, price)
                    elif self.scenario == "stop_loss":
                        dump_pct = -0.015 * self._price_tick_count
                        price = round(pos.entry_price * (1.0 + dump_pct), 4)
                        self.exchange.set_mock_price(sym, price)
                    elif self.scenario == "market_walk":
                        price = self.exchange.simulate_price_tick(drift=0.0005, volatility=0.004)
                    elif self.scenario == "fsm_cycle":
                        pump_pct = 0.03 * self._price_tick_count
                        price = round(pos.entry_price * (1.0 + pump_pct), 4)
                        self.exchange.set_mock_price(sym, price)
                    else:
                        price = self.exchange.simulate_price_tick(drift=0.0, volatility=0.002)
                else:
                    price = await self.exchange.get_ticker_price(sym)

                self._latest_price = price
                await self.risk_engine.evaluate_tick(price, symbol=sym)

            # 2. News Monitoring & Global Macro Breaker
            if self.risk_engine.active_positions:
                if self.exchange.is_mock:
                    self._news_tick_count += 1
                    if self.scenario == "ai_panic" and self._news_tick_count >= 2:
                        self.news_watcher.inject_emergency_headline()
                    elif self.scenario == "failsafe" and self._news_tick_count >= 2:
                        self.news_watcher.inject_headline(
                            "Network latency warning issued by node operators",
                            source="ValidatorNet",
                        )

                new_articles = await self.news_watcher.poll_new_headlines()
                for article in new_articles:
                    if self._shutdown_event.is_set() or not self.risk_engine.active_positions:
                        break

                    if self.scenario == "failsafe" and self._news_tick_count >= 2:
                        analysis = NewsAnalysis(
                            action="HOLD",
                            threat_level="LOW",
                            confidence=0.0,
                            reasoning="Fail-safe fallback triggered: Simulated LLM timeout after 5.0s",
                        )
                    else:
                        analysis = await self.ai_sentinel.analyze(
                            article.headline,
                            self.config.target_symbol,
                        )

                    self.record_news_analysis(article, analysis)

                    if analysis.action == "PANIC_SELL" or analysis.threat_level in ["CRITICAL", "HIGH"]:
                        self.notifier.notify_threat_detected(
                            headline=article.headline,
                            threat_level=analysis.threat_level,
                            confidence=analysis.confidence,
                            reasoning=analysis.reasoning,
                        )

                    # Global Macro Breaker: Systemic news triggers emergency liquidation across ALL open slots
                    if self.risk_engine.is_systemic_threat(analysis, article.headline):
                        notes = (
                            f"Global Systemic Threat Detected: Level={analysis.threat_level}, "
                            f"Confidence={analysis.confidence:.2f}, Reason: {analysis.reasoning}"
                        )
                        await self.risk_engine.execute_global_liquidation(
                            TriggerReason.GLOBAL_MACRO_BREAKER,
                            notes=notes,
                        )
                        break

                    # Asset-specific news trigger evaluation
                    for sym in list(self.risk_engine.active_positions.keys()):
                        curr_p = await self.exchange.get_ticker_price(sym)
                        await self.risk_engine.evaluate_tick(
                            current_price=curr_p,
                            news_analysis=analysis,
                            symbol=sym,
                            headline=article.headline,
                        )

            # 3. If All Positions Liquidated: Clear state and transition to COOLDOWN
            if self.risk_engine.is_liquidated or (self.risk_engine.last_liquidation is not None and not self.risk_engine.active_positions):
                rec = self.risk_engine.last_liquidation
                trigger_name = rec.trigger.value if rec else "UNKNOWN"
                pnl_pct = rec.realized_pnl_pct * 100 if rec else 0.0
                logger.critical("[POSITIONS LIQUIDATED] Trigger: %s | PnL: %+.2f%%", trigger_name, pnl_pct)

                clear_state(self.state_file)
                self._entry_timestamp = None
                self._client_order_id = None

                # For single-scenario exit tests/benchmarks:
                if self.scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe"]:
                    logger.info("Single-scenario test completed (%s). Shutting down.", self.scenario)
                    self._shutdown_event.set()
                    return

                # Continuous FSM lifecycle: Transition to COOLDOWN
                self.transition_to(BotState.COOLDOWN)
                return

            # Display live telemetry & persist state for active slots
            save_state(
                state=BotState.IN_POSITION,
                active_positions=self.risk_engine.positions,
                screener_cache=self.screener.cached_universe or [],
                screener_timestamp=self.screener.cache_timestamp,
                file_path=self.state_file,
            )

            # Increment candles_held across active slots at 15m candle close boundaries
            now_ts = time.time()
            candle_interval_sec = 900.0
            current_bucket = int(now_ts // candle_interval_sec)
            if hasattr(self, "_last_candle_bucket") and self._last_candle_bucket is not None:
                if current_bucket > self._last_candle_bucket:
                    self.risk_engine.increment_all_candles()
                    self._last_candle_bucket = current_bucket
            else:
                self._last_candle_bucket = current_bucket

            for sym, pos in self.risk_engine.active_positions.items():
                p = await self.exchange.get_ticker_price(sym)
                status = await self.risk_engine.get_position_status(p, symbol=sym)
                tp_price = pos.entry_price * (1.0 + self.config.take_profit_pct)
                sl_price = pos.current_stop_price if pos.current_stop_price > 0 else pos.entry_price * (1.0 - self.config.stop_loss_pct)

                console.print(
                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                    f"[bold green][IN_POSITION][/bold green] {sym}: ${p:.4f} | "
                    f"PnL: [{'bold green' if status.pnl_pct >= 0 else 'bold red'}]{status.pnl_pct * 100:+.2f}%[/] | "
                    f"Limits: [red]Chandelier SL ${sl_price:.2f} (Peak ${pos.highest_price:.2f})[/red] <-> [green]TP ${tp_price:.2f}[/green] | "
                    f"Held: {pos.candles_held} candles"
                )

        except Exception as e:
            logger.error("Error in IN_POSITION loop: %s", e)

        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=self.config.poll_interval_seconds,
            )
        except asyncio.TimeoutError:
            pass

    # =====================================================================
    # State 4: COOLDOWN
    # =====================================================================

    async def _handle_cooldown(self) -> None:
        """State 4: Sleep for COOLDOWN_MINUTES to prevent immediate re-entry into high-volatility chop."""
        cooldown_mins = self.config.cooldown_minutes
        cooldown_secs = 0.1 if self.scenario == "fsm_cycle" else (cooldown_mins * 60.0)

        console.print(
            f"\n[bold yellow][COOLDOWN][/bold yellow] Position closed. Cooling down for {cooldown_mins}m "
            f"to prevent immediate chop re-entry...\n"
        )
        logger.info("[COOLDOWN] Cooling down for %.1f minutes (%.1fs)...", cooldown_mins, cooldown_secs)

        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=cooldown_secs)
        except asyncio.TimeoutError:
            pass

        if not self._shutdown_event.is_set():
            logger.info("[COOLDOWN EXPIRED] Transitioning back to CASH_SCANNING.")
            self.transition_to(BotState.CASH_SCANNING)

    # =====================================================================
    # Main FSM Loop & Lifecycle Control
    # =====================================================================

    async def hourly_summary_loop(self) -> None:
        """Independent asynchronous worker running automated hourly Discord performance reports.

        Aligns to the top of each hour (e.g. :00:00). Runs concurrently alongside the FSM trading loop.
        """
        if not getattr(self.config, "hourly_summary_enabled", True):
            logger.info("[HOURLY SUMMARY] Hourly summary worker disabled via configuration.")
            return

        logger.info("[HOURLY SUMMARY] Background worker started. Aligned to top of the hour.")

        while not self._shutdown_event.is_set():
            now = time.time()
            sleep_seconds = 3600 - (now % 3600)  # Sleep until the next full hour
            if sleep_seconds <= 0:
                sleep_seconds = 3600

            logger.info(
                "[HOURLY SUMMARY] Next hourly performance report scheduled in %.1fs (at :00).",
                sleep_seconds,
            )

            try:
                # Sleep interruptible by shutdown event
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=sleep_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

            if self._shutdown_event.is_set():
                break

            try:
                # 1. Fetch free quote balance
                free_usdt = await self.exchange.get_balance(self.config.quote_asset)

                # 2. Query open active positions and compute latest prices / unrealized PnL
                active_pos: Dict[str, Any] = {}
                for sym, pos in self.risk_engine.active_positions.items():
                    pos_dict = pos.to_dict() if hasattr(pos, "to_dict") else dict(pos)
                    try:
                        curr_p = await self.exchange.get_ticker_price(sym)
                        pnl_pct, pnl_usd = self.risk_engine.calculate_pnl(curr_p, pos.amount, symbol=sym)
                        pos_dict["last_known_pnl"] = pnl_pct
                        pos_dict["pnl_usd"] = pnl_usd
                        pos_dict["current_price"] = curr_p
                    except Exception as e:
                        logger.warning("Could not fetch ticker price for %s in hourly summary: %s", sym, e)
                    active_pos[sym] = pos_dict

                # 3. Calculate metrics from persistent trade history
                metrics = calculate_metrics(
                    free_usdt=free_usdt,
                    active_positions=active_pos,
                    file_path=self.trade_history_file,
                    max_slots=self.config.max_active_positions,
                )

                logger.info(
                    "[HOURLY SUMMARY] Dispatching hourly summary: %d trades, Net PnL: $%.2f, Free USDT: $%.2f",
                    metrics.get("total_trades", 0),
                    metrics.get("total_realized_pnl_usd", 0.0),
                    free_usdt,
                )

                # 4. Dispatch Discord notification
                self.notifier.notify_hourly_summary(metrics)

            except Exception as e:
                logger.error("[HOURLY SUMMARY] Error generating hourly summary: %s", e)

    async def _fsm_loop(self) -> None:
        """Continuous asynchronous FSM trading loop."""
        while not self._shutdown_event.is_set():
            if self.state == BotState.CASH_SCANNING:
                await self._handle_cash_scanning()
            elif self.state == BotState.ENTERING:
                await self._handle_entering()
            elif self.state == BotState.IN_POSITION:
                await self._handle_in_position()
            elif self.state == BotState.COOLDOWN:
                await self._handle_cooldown()

            # If max cycles limit reached (useful for testing or scheduled runs)
            if self.max_cycles is not None and self._cycle_count >= self.max_cycles:
                logger.info("Reached maximum requested FSM cycles (%s). Terminating loop.", self.max_cycles)
                self._shutdown_event.set()
                break

    async def start(self) -> None:
        """Run the bot state machine loop and hourly summary worker concurrently."""
        self._running = True
        try:
            await self.initialize()

            # Run trading loop and hourly summary concurrently via asyncio.gather
            await asyncio.gather(
                self._fsm_loop(),
                self.hourly_summary_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Bot execution cancelled.")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully close connections, flush webhooks, and print final session summary."""
        logger.info("Shutting down Sentinel-Trader...")
        self._shutdown_event.set()
        await self.exchange.close()

        # Flush any pending background webhook dispatches
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        if self.risk_engine.last_liquidation:
            rec = self.risk_engine.last_liquidation
            console.print("\n" + "=" * 60, style="bold red")
            console.print("[LIQUIDATION ALERT] FINAL REPORT: POSITION LIQUIDATED", style="bold red")
            console.print(f"Trigger Reason : [bold]{rec.trigger.value}[/bold]")
            console.print(f"Symbol         : {rec.symbol}")
            console.print(f"Entry Price    : ${rec.entry_price:.4f}")
            console.print(f"Exit Price     : ${rec.exit_price:.4f}")
            console.print(f"Realized PnL   : [bold]{rec.realized_pnl_pct * 100:+.2f}% (${rec.realized_pnl_usd:+.2f})[/bold]")
            console.print(f"Amount Sold    : {rec.amount_liquidated:.4f}")
            console.print(f"Order ID       : {rec.receipt.order_id}")
            console.print(f"Audit Notes    : {rec.notes}")
            console.print("=" * 60 + "\n", style="bold red")
        else:
            console.print("[green]Bot terminated safely.[/green]")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected ('true' or 'false').")


def parse_args() -> argparse.Namespace:
    """Parse command line flags for simulation, symbols, and FSM parameters."""
    parser = argparse.ArgumentParser(
        description="Sentinel-Trader: Autonomous Crypto FSM Trading Bot"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["default", "continuous", "take_profit", "stop_loss", "ai_panic", "failsafe", "market_walk", "fsm_cycle"],
        default=None,
        help="Simulation scenario: take_profit, stop_loss, ai_panic, failsafe, market_walk, fsm_cycle",
    )
    parser.add_argument(
        "--initial-state",
        type=str,
        choices=["CASH_SCANNING", "ENTERING", "IN_POSITION", "COOLDOWN"],
        default=None,
        help="Initial FSM state override (default: CASH_SCANNING)",
    )
    parser.add_argument(
        "--mock",
        nargs="?",
        const=True,
        type=str2bool,
        default=None,
        help="Force in-memory mock exchange (True/False). Pass '--mock False' for live/testnet mode",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Target trading pair (e.g. SOL/USDT)",
    )
    parser.add_argument(
        "--entry-price",
        type=float,
        default=None,
        help="Initial position entry price",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Price monitoring polling interval in seconds",
    )
    parser.add_argument(
        "--max-news-age",
        type=int,
        default=None,
        help="Maximum news age in minutes for freshness cutoff (default: 30)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum FSM cycles before termination (optional)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to atomic persistence state file (default: state.json)",
    )
    parser.add_argument(
        "--trade-history-file",
        type=str,
        default=None,
        help="Path to atomic trade history file (default: trade_history.json)",
    )
    return parser.parse_args()


def run() -> None:
    """CLI entrypoint with signal handling and scenario configuration."""
    args = parse_args()
    config = Config()

    # Apply CLI overrides
    if args.mock is not None:
        config.mock_exchange = args.mock
    if args.symbol:
        config.target_symbol = args.symbol
    if args.entry_price:
        config.entry_price = args.entry_price
    if args.poll_interval:
        config.poll_interval_seconds = args.poll_interval
    if args.max_news_age:
        config.max_news_age_minutes = args.max_news_age

    scenario = args.scenario
    if args.mock is False:
        # Explicit live mode: consume live RSS feeds, disable synthetic news injection
        config.mock_exchange = False
        config.enable_rss_news = True
        config.enable_simulated_news = False
    else:
        # If no Binance credentials exist or scenario requested, default to mock mode
        if not config.binance_api_key or scenario:
            config.mock_exchange = True
            if not scenario:
                scenario = "market_walk"

        # Offline self-containment for test scenarios
        if config.mock_exchange and scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe", "fsm_cycle"]:
            config.enable_rss_news = False
            if not config.llm_api_key or scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe", "fsm_cycle"]:
                config.llm_provider = "mock"

    init_state = BotState(args.initial_state) if args.initial_state else None
    bot = SentinelTrader(
        config,
        scenario=scenario,
        initial_state=init_state,
        max_cycles=args.max_cycles,
        state_file=args.state_file,
        trade_history_file=args.trade_history_file,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        logger.warning("Shutdown signal received (Ctrl+C). Terminating gracefully...")
        bot._shutdown_event.set()

    # Register signals if available
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except (NotImplementedError, AttributeError):
        pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        loop.close()


if __name__ == "__main__":
    run()
