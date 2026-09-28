"""Persistent trade analytics and performance metrics module for Sentinel-Trader.

Provides crash-resilient local persistence of closed trades to `trade_history.json`
using atomic filesystem writes (os.replace). Calculates multi-dimensional performance
metrics including win rates, gross profit/loss, rolling 1-hour PnL, and slot allocations.
"""

from datetime import datetime, timezone, timedelta
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sentinel.analytics")

DEFAULT_TRADE_HISTORY_FILE = "trade_history.json"
DEFAULT_MOCK_TRADE_HISTORY_FILE = "trade_history_mock.json"
MOCK_TRADE_HISTORY_FILE = "trade_history_mock.json"


def resolve_trade_history_path(
    file_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> str:
    """Resolve target trade history file, isolating mock/test runs to trade_history_mock.json.

    If config.mock_exchange is True (or mock environment is active), and the caller did
    not supply an isolated custom file path, defaults to trade_history_mock.json.
    """
    is_mock = False
    if config is not None:
        if isinstance(config, bool):
            is_mock = config
        else:
            is_mock = bool(getattr(config, "mock_exchange", False))
    elif os.environ.get("MOCK_EXCHANGE", "").lower() in ("true", "1", "yes") or os.environ.get("SENTINEL_MOCK_EXCHANGE", "").lower() in ("true", "1", "yes"):
        is_mock = True

    if is_mock:
        if file_path is None or file_path == DEFAULT_TRADE_HISTORY_FILE or os.path.abspath(file_path) == os.path.abspath(DEFAULT_TRADE_HISTORY_FILE):
            return DEFAULT_MOCK_TRADE_HISTORY_FILE

    return file_path or DEFAULT_TRADE_HISTORY_FILE


def record_trade(
    symbol: str,
    entry_price: float,
    exit_price: float,
    amount: float,
    realized_pnl_usd: float,
    realized_pnl_pct: float,
    trigger_reason: str,
    file_path: Optional[str] = None,
    timestamp: Optional[str] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Atomically record a closed trade into trade_history.json (or trade_history_mock.json).

    Args:
        symbol: Trading pair symbol (e.g. 'SOL/USDT').
        entry_price: Execution entry price (cost basis).
        exit_price: Execution exit price.
        amount: Quantity of base asset closed/liquidated.
        realized_pnl_usd: Dollar profit or loss realized.
        realized_pnl_pct: Percentage profit or loss realized (decimal e.g. 0.05 for +5%).
        trigger_reason: Reason for liquidation (e.g. 'TAKE_PROFIT', 'STOP_LOSS', 'AI_PANIC_SELL').
        file_path: Optional path override for trade history JSON file.
        timestamp: Optional ISO-8601 timestamp string override.
        config: Optional configuration object or boolean (checks config.mock_exchange).

    Returns:
        Dict[str, Any]: The newly recorded trade record dictionary.
    """
    target_path = resolve_trade_history_path(file_path, config)
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)

    iso_timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    record: Dict[str, Any] = {
        "symbol": str(symbol),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "amount": float(amount),
        "realized_pnl_usd": float(realized_pnl_usd),
        "realized_pnl_pct": float(realized_pnl_pct),
        "trigger_reason": str(trigger_reason),
        "timestamp": str(iso_timestamp),
    }

    trades = load_trades(target_path)
    trades.append(record)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, encoding="utf-8") as f:
            temp_path = f.name
            json.dump(trades, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # CRITICAL FOR WINDOWS: File handle must be closed before calling os.replace
        os.replace(temp_path, target_path)
        logger.debug("Trade atomically recorded to %s: %s", target_path, record)
    except Exception as e:
        logger.error("Failed to atomically persist trade history to %s: %s", target_path, e)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e

    return record


def load_trades(
    file_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Load historical closed trades from JSON persistence.

    Args:
        file_path: Optional path override for trade history JSON file.
        config: Optional configuration object or boolean (checks config.mock_exchange).

    Returns:
        List[Dict[str, Any]]: List of trade records, or empty list if file doesn't exist or is corrupt.
    """
    target_path = resolve_trade_history_path(file_path, config)
    if not os.path.exists(target_path):
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "trades" in data and isinstance(data["trades"], list):
            return data["trades"]
        else:
            logger.warning("Unrecognized schema in trade history file %s. Expected list.", target_path)
            return []
    except Exception as e:
        logger.warning("Failed to load trade history from %s: %s", target_path, e)
        return []


def clear_trades(
    file_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> bool:
    """Clear/remove the trade history file (useful for test resets).

    Args:
        file_path: Optional path override for trade history JSON file.
        config: Optional configuration object or boolean (checks config.mock_exchange).

    Returns:
        bool: True if cleared successfully or file was already absent.
    """
    target_path = resolve_trade_history_path(file_path, config)
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            return True
        except OSError as e:
            logger.warning("Failed to clear trade history file %s: %s", target_path, e)
            return False
    return True


def parse_trade_timestamp(ts_val: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string or numeric timestamp into timezone-aware UTC datetime."""
    if not ts_val:
        return None
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(ts_val, tz=timezone.utc)
    if isinstance(ts_val, str):
        try:
            # Handle ISO string with trailing 'Z'
            cleaned = ts_val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def calculate_metrics(
    free_usdt: float,
    active_positions: Dict[str, Any],
    file_path: Optional[str] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
    lookback_minutes: int = 60,
    max_slots: int = 3,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Calculate multi-dimensional trading performance metrics.

    Args:
        free_usdt: Current unencumbered quote asset balance (e.g. USDT).
        active_positions: Dictionary of currently open positions {symbol: position_data}.
        file_path: Optional path to trade history file if trades list not explicitly provided.
        trades: Optional pre-loaded list of trade dictionaries.
        lookback_minutes: Window for rolling PnL calculation (default 60 minutes).
        max_slots: Total available portfolio slot capacity (default 3).
        config: Optional configuration object or boolean (checks config.mock_exchange).

    Returns:
        Dict[str, Any]: Computed performance metrics containing:
            - total_trades (int)
            - wins (int)
            - losses (int)
            - ties (int)
            - win_rate_pct (float)
            - total_realized_pnl_usd (float)
            - total_realized_pnl_pct (float)
            - gross_profit_usd (float)
            - gross_loss_usd (float)
            - rolling_1h_pnl_usd (float)
            - free_usdt (float)
            - active_positions (dict)
            - active_slots_count (int)
            - max_slots (int)
    """
    if trades is None:
        trades = load_trades(file_path, config=config)

    total_trades = len(trades)
    wins = 0
    losses = 0
    ties = 0

    total_realized_pnl_usd = 0.0
    gross_profit_usd = 0.0
    gross_loss_usd = 0.0
    total_cost_basis = 0.0
    sum_individual_pnl_pct = 0.0

    now_utc = datetime.now(timezone.utc)
    cutoff_time = now_utc - timedelta(minutes=lookback_minutes)
    rolling_1h_pnl_usd = 0.0

    for t in trades:
        pnl_usd = float(t.get("realized_pnl_usd", 0.0))
        pnl_pct = float(t.get("realized_pnl_pct", 0.0))
        entry_p = float(t.get("entry_price", 0.0))
        amt = float(t.get("amount", 0.0))

        cost = entry_p * amt
        if cost > 0:
            total_cost_basis += cost

        total_realized_pnl_usd += pnl_usd

        # Realized percentage accumulation
        if abs(pnl_pct) <= 1.0 and pnl_pct != 0.0:
            sum_individual_pnl_pct += pnl_pct * 100.0
        else:
            sum_individual_pnl_pct += pnl_pct

        # Win / Loss / Tie Classification
        if pnl_usd > 0.0:
            wins += 1
            gross_profit_usd += pnl_usd
        elif pnl_usd < 0.0:
            losses += 1
            gross_loss_usd += abs(pnl_usd)
        else:
            ties += 1

        # 1-Hour Rolling Window Evaluation
        ts = parse_trade_timestamp(t.get("timestamp"))
        if ts is not None and ts >= cutoff_time:
            rolling_1h_pnl_usd += pnl_usd

    # Win rate computation
    if total_trades > 0:
        win_rate_pct = round((wins / total_trades) * 100.0, 1)
    else:
        win_rate_pct = 0.0

    # Total Realized PnL percentage computation
    if total_cost_basis > 0:
        total_realized_pnl_pct = round((total_realized_pnl_usd / total_cost_basis) * 100.0, 2)
    elif total_trades > 0:
        total_realized_pnl_pct = round(sum_individual_pnl_pct, 2)
    else:
        total_realized_pnl_pct = 0.0

    # Format active slots
    active_count = len(active_positions) if active_positions else 0

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_pct": win_rate_pct,
        "total_realized_pnl_usd": round(total_realized_pnl_usd, 2),
        "total_realized_pnl_pct": total_realized_pnl_pct,
        "gross_profit_usd": round(gross_profit_usd, 2),
        "gross_loss_usd": round(gross_loss_usd, 2),
        "rolling_1h_pnl_usd": round(rolling_1h_pnl_usd, 2),
        "free_usdt": round(float(free_usdt), 2),
        "active_positions": active_positions or {},
        "active_slots_count": active_count,
        "max_slots": max_slots,
    }
