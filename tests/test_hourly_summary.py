"""Standalone verification script for Sentinel-Trader Hourly Performance Report.

Generates realistic mock trade data in `trade_history.json`, calculates comprehensive
analytics, and immediately fires an hourly performance summary card to Discord
to verify embed structure, color schemes, and field formatting.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import os
import sys

# Ensure project root is in sys.path
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
            cmd = [venv_py, os.path.abspath(__file__)] + sys.argv[1:]
            sys.exit(subprocess.call(cmd, cwd=root_dir))

from typing import Any, Dict, Optional
from aiohttp import web

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sentinel_trader.analytics import (
    DEFAULT_TRADE_HISTORY_FILE,
    DEFAULT_MOCK_TRADE_HISTORY_FILE,
    calculate_metrics,
    clear_trades,
    record_trade,
)
from sentinel_trader.config import Config
from sentinel_trader.notifier import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GRAY,
    DiscordNotifier,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


async def main() -> int:
    print_banner("SENTINEL-TRADER: HOURLY SUMMARY CARD VERIFICATION")

    # 1. Parse CLI arguments (optional custom webhook URL)
    cli_webhook: Optional[str] = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        cli_webhook = sys.argv[1]

    config = Config()
    target_webhook = cli_webhook or config.discord_webhook_url or ""
    masked_url = (
        target_webhook[:32] + "..." + target_webhook[-8:]
        if len(target_webhook) > 40
        else (target_webhook or "(None configured - local mock mode)")
    )
    print(f" Target Webhook: {masked_url}")

    # 2. Generate Realistic Mock Trade Data in trade_history.json
    trade_file = os.path.join(root_dir, DEFAULT_MOCK_TRADE_HISTORY_FILE)
    print(f" Initializing test trade history in: {trade_file}")
    clear_trades(trade_file)

    now = datetime.now(timezone.utc)
    ts_25m_ago = (now - timedelta(minutes=25)).isoformat()
    ts_40m_ago = (now - timedelta(minutes=40)).isoformat()
    ts_2h_ago = (now - timedelta(hours=2)).isoformat()

    print(" -> Seeding mock trades:")
    # Winning trade within past 1 hour (+5.0%)
    t1 = record_trade(
        symbol="SOL/USDT",
        entry_price=140.0,
        exit_price=147.0,
        amount=5.0,
        realized_pnl_usd=35.00,
        realized_pnl_pct=0.05,
        trigger_reason="TAKE_PROFIT",
        file_path=trade_file,
        timestamp=ts_25m_ago,
    )
    print(f"    [1] {t1['symbol']} ({t1['trigger_reason']}): Entry ${t1['entry_price']} -> Exit ${t1['exit_price']} | PnL +${t1['realized_pnl_usd']:.2f} (25m ago)")

    # Losing trade older than 1 hour (-2.5%)
    t2 = record_trade(
        symbol="ETH/USDT",
        entry_price=2600.0,
        exit_price=2535.0,
        amount=0.5,
        realized_pnl_usd=-32.50,
        realized_pnl_pct=-0.025,
        trigger_reason="STOP_LOSS",
        file_path=trade_file,
        timestamp=ts_2h_ago,
    )
    print(f"    [2] {t2['symbol']} ({t2['trigger_reason']}): Entry ${t2['entry_price']} -> Exit ${t2['exit_price']} | PnL ${t2['realized_pnl_usd']:.2f} (2h ago)")

    # Winning trade within past 1 hour (+5.0%)
    t3 = record_trade(
        symbol="BTC/USDT",
        entry_price=62000.0,
        exit_price=65100.0,
        amount=0.05,
        realized_pnl_usd=155.00,
        realized_pnl_pct=0.05,
        trigger_reason="TAKE_PROFIT",
        file_path=trade_file,
        timestamp=ts_40m_ago,
    )
    print(f"    [3] {t3['symbol']} ({t3['trigger_reason']}): Entry ${t3['entry_price']} -> Exit ${t3['exit_price']} | PnL +${t3['realized_pnl_usd']:.2f} (40m ago)")

    # 3. Mock Active Portfolio Positions & Free USDT
    mock_free_usdt = 2450.50
    mock_active_positions = {
        "SOL/USDT": {
            "entry_price": 152.0,
            "amount": 2.5,
            "last_known_pnl": 0.0185,  # +1.85% unrealized
        },
        "AVAX/USDT": {
            "entry_price": 28.5,
            "amount": 10.0,
            "last_known_pnl": -0.0075, # -0.75% unrealized
        },
    }

    # 4. Calculate Analytics Metrics
    print("\n Computing analytics metrics...")
    metrics = calculate_metrics(
        free_usdt=mock_free_usdt,
        active_positions=mock_active_positions,
        file_path=trade_file,
        max_slots=3,
    )

    print(" ==========================================================================")
    print(f"  Total Trades        : {metrics['total_trades']}")
    print(f"  Win Rate            : {metrics['win_rate_pct']:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  Total Net PnL       : +${metrics['total_realized_pnl_usd']:.2f} (+{metrics['total_realized_pnl_pct']:.2f}%)")
    print(f"  Made / Lost         : +${metrics['gross_profit_usd']:.2f} won | -${metrics['gross_loss_usd']:.2f} lost")
    print(f"  Past 1-Hour PnL     : +${metrics['rolling_1h_pnl_usd']:.2f}")
    print(f"  Free USDT           : ${metrics['free_usdt']:,.2f}")
    print(f"  Active Slots        : {metrics['active_slots_count']}/{metrics['max_slots']}")
    print(" ==========================================================================")

    # Assert metric correctness
    assert metrics["total_trades"] == 3
    assert metrics["wins"] == 2
    assert metrics["losses"] == 1
    assert metrics["win_rate_pct"] == 66.7
    assert metrics["total_realized_pnl_usd"] == 157.50
    assert metrics["gross_profit_usd"] == 190.00
    assert metrics["gross_loss_usd"] == 32.50
    assert metrics["rolling_1h_pnl_usd"] == 190.00  # Only trades 1 & 3 within 60m
    print(" [PASSED] Mathematical analytics calculations verified.")

    # 5. Build and Validate Discord Embed
    notifier = DiscordNotifier(config, webhook_url=target_webhook)
    embed = notifier.build_hourly_summary_embed(metrics)

    assert embed["title"] == "⏱️ Sentinel-Trader Hourly Performance Report"
    assert embed["color"] == COLOR_GREEN == 0x2ECC71, f"Expected Green (0x2ecc71), got {hex(embed['color'])}"

    field_map = {f["name"]: f["value"] for f in embed["fields"]}
    print("\n -> Formatted Discord Embed Fields:")
    for name, val in field_map.items():
        print(f"    • {name:20}: {val}")

    assert "Total Net PnL" in field_map
    assert "+$157.50" in field_map["Total Net PnL"]
    assert "Made / Lost" in field_map
    assert "+$190.00 won | -$32.50 lost" in field_map["Made / Lost"]
    assert "Past 1-Hour PnL" in field_map
    assert "+$190.00" in field_map["Past 1-Hour PnL"]
    assert "Win Rate" in field_map
    assert "66.7% (2W / 1L across 3 trades)" in field_map["Win Rate"]
    assert "Free USDT" in field_map
    assert "$2,450.50" in field_map["Free USDT"]
    assert "Risk Mode" in field_map
    assert field_map["Risk Mode"] == "1.0x (Normal)"
    assert "Streak" in field_map
    assert "Active Slots (2/3)" in field_map
    assert "SOL/USDT" in field_map["Active Slots (2/3)"]
    assert "AVAX/USDT" in field_map["Active Slots (2/3)"]
    print(" [PASSED] Discord Embed structure and field specifications verified.")

    # 6. Test Flat / 0 Trades and Negative PnL Embed Colors
    empty_metrics = calculate_metrics(free_usdt=1000.0, active_positions={}, trades=[])
    empty_embed = notifier.build_hourly_summary_embed(empty_metrics)
    assert empty_embed["color"] == COLOR_GRAY == 0x95A5A6
    assert any(f["name"] == "Active Slots (0/3)" and f["value"] == "100% Cash" for f in empty_embed["fields"])
    print(" [PASSED] 0-Trades Embed Color verified as Gray (0x95a5a6) with '100% Cash'.")

    loss_metrics = calculate_metrics(
        free_usdt=1000.0,
        active_positions={},
        trades=[{"symbol": "SOL/USDT", "realized_pnl_usd": -50.0, "realized_pnl_pct": -0.05, "entry_price": 100.0, "amount": 10.0}],
    )
    loss_embed = notifier.build_hourly_summary_embed(loss_metrics)
    assert loss_embed["color"] == COLOR_RED == 0xE74C3C
    print(" [PASSED] Loss Embed Color verified as Red (0xe74c3c).")

    # 7. Dispatch to Discord (Live or In-Process Mock Server)
    print_banner("DISCORD WEBHOOK DISPATCH")
    if target_webhook.startswith("https://"):
        print(f" Delivering live Hourly Performance Report to Discord: {masked_url}")
        success = await notifier.notify_hourly_summary_async(metrics)
        assert success, "Failed to deliver hourly report to live Discord webhook"
        print(" [SUCCESS] Live Discord Hourly Performance Report successfully delivered! (HTTP 204/200)")
        print("           Check your Discord channel to view the card.")
    else:
        print(" No live webhook URL found. Launching local mock HTTP server...")
        received_payloads = []

        async def mock_handler(request):
            data = await request.json()
            received_payloads.append(data)
            return web.Response(status=204)

        app = web.Application()
        app.router.add_post("/webhook", mock_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        try:
            local_notifier = DiscordNotifier(config, webhook_url=f"http://127.0.0.1:{port}/webhook")
            sent = await local_notifier.notify_hourly_summary_async(metrics)
            assert sent is True
            assert len(received_payloads) == 1
            payload = received_payloads[0]
            assert payload["embeds"][0]["title"] == "⏱️ Sentinel-Trader Hourly Performance Report"
            print(" [SUCCESS] Mock HTTP Discord server received and validated the hourly summary payload.")
        finally:
            await runner.cleanup()

    print_banner("VERIFICATION COMPLETE: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
