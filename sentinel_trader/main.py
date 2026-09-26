"""Main orchestrator for Sentinel-Trader.

Runs concurrent asynchronous loops for:
1. Real-time price monitoring and mathematical risk thresholds.
2. News headline ingestion, SHA-256 deduplication, and AI threat detection.
3. Instant deterministic liquidation upon safety trigger breach.
"""

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

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

from sentinel_trader.ai_sentinel import AISentinel, NewsAnalysis
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.news_watcher import NewsWatcher
from sentinel_trader.risk_engine import RiskEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sentinel.orchestrator")


class SentinelTrader:
    """Async trading orchestrator managing risk, price, and sentiment loops."""

    def __init__(self, config: Optional[Config] = None, scenario: Optional[str] = None):
        self.config = config or Config()
        self.scenario = scenario or "default"
        self.exchange = ExchangeClient(self.config)
        self.news_watcher = NewsWatcher(self.config)
        self.ai_sentinel = AISentinel(self.config)
        self.risk_engine = RiskEngine(
            self.config,
            self.exchange,
            initial_entry_price=self.config.entry_price,
        )

        self._shutdown_event = asyncio.Event()
        self._latest_price: Optional[float] = None
        self._running = False
        self._price_tick_count = 0
        self._news_tick_count = 0

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
        banner_text.append("[SENTINEL-TRADER] CRYPTO RISK MITIGATION BOT\n", style="bold cyan")
        banner_text.append(f"Operating Mode       : {mode_text}\n")
        banner_text.append(f"Mock Scenario        : {scenario_text}\n")
        banner_text.append(f"Target Pair          : [bold yellow]{self.config.target_symbol}[/bold yellow]\n")
        banner_text.append(f"Base Asset           : [bold]{self.config.base_asset}[/bold]\n")
        banner_text.append(f"Quote Asset          : [bold]{self.config.quote_asset}[/bold]\n")
        banner_text.append(f"Hard Take-Profit     : [bold green]+{self.config.take_profit_pct * 100:.1f}%[/bold green]\n")
        banner_text.append(f"Hard Stop-Loss       : [bold red]-{self.config.stop_loss_pct * 100:.1f}%[/bold red]\n")
        banner_text.append(f"AI Provider          : [bold magenta]{self.config.llm_provider.upper()}[/bold magenta] ({self.config.get_effective_llm_model()})\n")
        banner_text.append(f"Panic Sell Threshold : [bold]{self.config.panic_sell_confidence_threshold:.0%}[/bold] confidence\n")
        banner_text.append(f"Price Poll Interval  : {self.config.poll_interval_seconds}s | News Poll: {self.config.news_poll_interval_seconds}s\n")

        console.print(Panel(banner_text, title="System Initialized", border_style="cyan"))

    def render_status_table(self, status) -> None:
        """Render beautiful telemetry table to console."""
        table = Table(title="Live Position Telemetry", border_style="blue")
        table.add_column("Symbol", justify="center", style="cyan", no_wrap=True)
        table.add_column("Current Price", justify="right", style="bold")
        table.add_column("Entry Price", justify="right")
        table.add_column("PnL (%)", justify="right")
        table.add_column("Holding Amount", justify="right")
        table.add_column("Position Value", justify="right")
        table.add_column("Status", justify="center")

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
        """Connect to exchange, load markets, and initialize position state."""
        await self.exchange.initialize()

        # Fetch current price
        self._latest_price = await self.exchange.get_current_price()

        # Initialize entry price if not set in config
        if self.risk_engine.entry_price is None:
            self.risk_engine.set_entry_price(self._latest_price)

        self.print_banner()

        # Display initial position status
        status = await self.risk_engine.get_position_status(self._latest_price)
        self.render_status_table(status)

    async def price_monitor_loop(self) -> None:
        """Asynchronous loop continuously monitoring price and mathematical limits."""
        logger.info("Starting Price Monitoring Loop (Interval: %ss)...", self.config.poll_interval_seconds)

        while not self._shutdown_event.is_set() and not self.risk_engine.is_liquidated:
            try:
                # If running mock simulation, generate scenario-driven price ticks
                if self.exchange.is_mock:
                    self._price_tick_count += 1
                    if self.scenario == "take_profit":
                        # Gradually pump price +2.5% each tick up to and breaching +10%
                        pump_pct = 0.025 * self._price_tick_count
                        price = round(self.risk_engine.entry_price * (1.0 + pump_pct), 4)
                        self.exchange.set_mock_price(self.config.target_symbol, price)
                    elif self.scenario == "stop_loss":
                        # Gradually dump price -1.5% each tick down to and breaching -4%
                        dump_pct = -0.015 * self._price_tick_count
                        price = round(self.risk_engine.entry_price * (1.0 + dump_pct), 4)
                        self.exchange.set_mock_price(self.config.target_symbol, price)
                    elif self.scenario == "market_walk":
                        # Standard Brownian motion market walk
                        price = self.exchange.simulate_price_tick(drift=0.0005, volatility=0.004)
                    else:
                        # Slight realistic oscillation for ai_panic / failsafe / default
                        price = self.exchange.simulate_price_tick(drift=0.0, volatility=0.002)
                else:
                    price = await self.exchange.get_current_price()

                self._latest_price = price

                # Evaluate mathematical limits (TP / SL)
                liquidation = await self.risk_engine.evaluate_tick(price)
                if liquidation:
                    logger.critical("[SHUTDOWN SIGNAL] Position liquidated via %s", liquidation.trigger.value)
                    self._shutdown_event.set()
                    break

                # Telemetry output
                status = await self.risk_engine.get_position_status(price)
                tp_price = self.risk_engine.entry_price * (1.0 + self.config.take_profit_pct)
                sl_price = self.risk_engine.entry_price * (1.0 - self.config.stop_loss_pct)

                console.print(
                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                    f"[bold]{self.config.target_symbol}[/bold]: [yellow]${price:.4f}[/yellow] | "
                    f"PnL: [bold {'green' if status.pnl_pct >= 0 else 'red'}]{status.pnl_pct * 100:+.2f}%[/] | "
                    f"Limits: [red]SL ${sl_price:.2f}[/red] <-> [green]TP ${tp_price:.2f}[/green]"
                )

            except Exception as e:
                logger.error("Error in price monitor loop: %s", e)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def news_monitor_loop(self) -> None:
        """Asynchronous loop polling news headlines and evaluating AI sentiment."""
        logger.info("Starting News Sentiment Loop (Interval: %ss)...", self.config.news_poll_interval_seconds)

        while not self._shutdown_event.is_set() and not self.risk_engine.is_liquidated:
            try:
                # If running mock scenario, inject scenario-specific headlines
                if self.exchange.is_mock:
                    self._news_tick_count += 1
                    if self.scenario == "ai_panic" and self._news_tick_count == 2:
                        # Emergency exploit headline is ONLY injected when --scenario ai_panic is specifically selected
                        self.news_watcher.inject_emergency_headline()
                    elif self.scenario == "failsafe" and self._news_tick_count == 2:
                        self.news_watcher.inject_headline(
                            "Network latency warning issued by node operators",
                            source="ValidatorNet",
                        )

                new_articles = await self.news_watcher.poll_new_headlines()

                for article in new_articles:
                    if self._shutdown_event.is_set() or self.risk_engine.is_liquidated:
                        break

                    console.print(
                        f"\n[cyan][NEWS] Incoming Headline ({article.source}):[/cyan] "
                        f"[bold]{article.headline}[/bold] (SHA-256: {article.id[:8]}...)"
                    )

                    # In failsafe scenario, simulate LLM failure on headline 2
                    if self.scenario == "failsafe" and self._news_tick_count >= 2:
                        logger.warning("[SIMULATION: FAIL-SAFE TEST] Injecting simulated LLM timeout...")
                        analysis = NewsAnalysis(
                            action="HOLD",
                            threat_level="LOW",
                            confidence=0.0,
                            reasoning="Fail-safe fallback triggered: Simulated LLM timeout after 5.0s",
                        )
                    else:
                        # Evaluate AI Sentinel sentiment
                        analysis = await self.ai_sentinel.analyze(
                            article.headline,
                            self.config.target_symbol,
                        )

                    console.print(
                        f"   [AI] Sentinel Assessment: Action=[bold {'red' if analysis.action == 'PANIC_SELL' else 'green'}]{analysis.action}[/] | "
                        f"Threat=[bold]{analysis.threat_level}[/] | Conf={analysis.confidence:.2f}\n"
                        f"   Reasoning: {analysis.reasoning}"
                    )

                    # Route through deterministic risk engine
                    curr_price = self._latest_price or await self.exchange.get_current_price()
                    liquidation = await self.risk_engine.evaluate_tick(
                        current_price=curr_price,
                        news_analysis=analysis,
                    )

                    if liquidation:
                        logger.critical("[SHUTDOWN SIGNAL] Position liquidated via %s", liquidation.trigger.value)
                        self._shutdown_event.set()
                        break

            except Exception as e:
                logger.error("Error in news monitor loop: %s", e)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.news_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        """Run the bot orchestrator with graceful shutdown handling."""
        self._running = True
        try:
            await self.initialize()

            # Execute concurrent loops
            price_task = asyncio.create_task(self.price_monitor_loop(), name="price_monitor")
            news_task = asyncio.create_task(self.news_monitor_loop(), name="news_monitor")

            # Await shutdown or liquidation event
            await self._shutdown_event.wait()

            # Cancel remaining tasks
            price_task.cancel()
            news_task.cancel()
            await asyncio.gather(price_task, news_task, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Bot execution cancelled.")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully close connections and print summary."""
        logger.info("Shutting down Sentinel-Trader...")
        self._shutdown_event.set()
        await self.exchange.close()

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
            console.print("[green]Bot terminated safely. Spot position remained intact.[/green]")


def parse_args() -> argparse.Namespace:
    """Parse command line flags for local simulation scenarios."""
    parser = argparse.ArgumentParser(
        description="Sentinel-Trader: Automated Crypto Risk-Mitigation Bot"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["default", "take_profit", "stop_loss", "ai_panic", "failsafe", "market_walk"],
        default=None,
        help="Select local simulation scenario (take_profit, stop_loss, ai_panic, failsafe, market_walk)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force in-memory mock exchange and offline simulation mode",
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
    return parser.parse_args()


def run() -> None:
    """CLI entrypoint with signal handling and scenario support."""
    args = parse_args()
    config = Config()

    # Apply CLI overrides
    if args.mock:
        config.mock_exchange = True
    if args.symbol:
        config.target_symbol = args.symbol
    if args.entry_price:
        config.entry_price = args.entry_price
    if args.poll_interval:
        config.poll_interval_seconds = args.poll_interval

    # If no Binance credentials exist or scenario requested, default to mock mode
    scenario = args.scenario
    if not config.binance_api_key or scenario:
        config.mock_exchange = True
        if not scenario:
            scenario = "market_walk"

    # For mock simulation scenarios, ensure offline self-containment
    if config.mock_exchange and scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe"]:
        config.enable_rss_news = False
        if not config.llm_api_key or scenario in ["take_profit", "stop_loss", "ai_panic", "failsafe"]:
            config.llm_provider = "mock"

    bot = SentinelTrader(config, scenario=scenario)

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
        # Windows compatibility
        pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        loop.close()


if __name__ == "__main__":
    run()
