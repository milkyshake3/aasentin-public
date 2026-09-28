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
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("sentinel.analytics")

DEFAULT_TRADE_HISTORY_FILE = "trade_history.json"
DEFAULT_MOCK_TRADE_HISTORY_FILE = "trade_history_mock.json"
MOCK_TRADE_HISTORY_FILE = "trade_history_mock.json"
DEFAULT_ASSET_SCORES_FILE = "asset_scores.json"
DEFAULT_MOCK_ASSET_SCORES_FILE = "asset_scores_mock.json"

# In-memory asset reliability scores cache
asset_scores: Dict[str, Dict[str, Any]] = {}


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


def resolve_asset_scores_path(
    file_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> str:
    """Resolve target asset scores file, isolating mock/test runs to asset_scores_mock.json."""
    is_mock = False
    if config is not None:
        if isinstance(config, bool):
            is_mock = config
        else:
            is_mock = bool(getattr(config, "mock_exchange", False))
    elif os.environ.get("MOCK_EXCHANGE", "").lower() in ("true", "1", "yes") or os.environ.get("SENTINEL_MOCK_EXCHANGE", "").lower() in ("true", "1", "yes"):
        is_mock = True

    if is_mock:
        if file_path is None or file_path == DEFAULT_ASSET_SCORES_FILE or os.path.abspath(file_path) == os.path.abspath(DEFAULT_ASSET_SCORES_FILE):
            return DEFAULT_MOCK_ASSET_SCORES_FILE

    return file_path or DEFAULT_ASSET_SCORES_FILE


def calculate_score_adjustment(
    score: float,
    realized_pnl_usd: float,
    realized_pnl_pct: float,
    trigger_reason: Union[str, Any],
    candles_held: int = 0,
) -> float:
    """Calculate the updated reliability score based on trade outcome.

    Adjustment rules:
    - Hard Stop Loss (CHANDELIER_STOP or PnL < -1.5%): S_new = S - 0.7
    - Time Decay Scratch (ALPHA_DECAY_TIME_STOP or PnL <= 0): S_new = S - 0.3
    - Fast Win (PnL > 0 and candles_held <= 6): S_new = min(S + 0.5, 3.0)
    - Slow Win / Drag (PnL > 0 and candles_held > 10): S_new = min(S + 0.1, 3.0)
    - Moderate Win (PnL > 0 and 6 < candles_held <= 10): S_new = min(S + 0.25, 3.0)
    """
    trigger_str = trigger_reason.value if hasattr(trigger_reason, "value") else str(trigger_reason)

    # Normalize percentage to percent unit (e.g. -0.015 -> -1.5)
    if abs(realized_pnl_pct) <= 1.0 and realized_pnl_pct != 0.0:
        pct_value = realized_pnl_pct * 100.0
    else:
        pct_value = realized_pnl_pct

    # Precedence 1: Hard Stop Loss (CHANDELIER_STOP or PnL < -1.5%)
    if trigger_str == "CHANDELIER_STOP" or pct_value < -1.5:
        new_score = score - 0.7
    # Precedence 2: Time Decay Scratch (ALPHA_DECAY_TIME_STOP or PnL <= 0)
    elif trigger_str == "ALPHA_DECAY_TIME_STOP" or pct_value <= 0.0 or realized_pnl_usd <= 0.0:
        new_score = score - 0.3
    # Precedence 3: Win (PnL > 0)
    elif candles_held <= 6:
        new_score = min(score + 0.5, 3.0)
    elif candles_held > 10:
        new_score = min(score + 0.1, 3.0)
    else:
        new_score = min(score + 0.25, 3.0)

    return round(new_score, 4)


def update_asset_score(
    symbol: str,
    realized_pnl_usd: float,
    realized_pnl_pct: float,
    trigger_reason: Union[str, Any],
    candles_held: int = 0,
    file_path: Optional[str] = None,
    current_time: Optional[float] = None,
    asset_scores_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Update asset reliability score and manage penalty box state.

    Args:
        symbol: Trading pair symbol (e.g. 'SOL/USDT').
        realized_pnl_usd: Profit or loss in USD.
        realized_pnl_pct: Percentage profit or loss realized.
        trigger_reason: Liquidation trigger reason.
        candles_held: Number of 15m candles position was held.
        file_path: Custom persistence file path for asset_scores.json.
        current_time: Optional epoch timestamp override.
        asset_scores_dict: Optional pre-loaded asset_scores dictionary.

    Returns:
        Dict[str, Any]: Updated asset score dict with 'score', 'penalized_until', 'trades_count'.
    """
    now_ts = current_time if current_time is not None else time.time()

    should_save = False
    if asset_scores_dict is not None:
        scores = asset_scores_dict
    else:
        from sentinel_trader.state_store import load_asset_scores
        scores = load_asset_scores(file_path=file_path, current_time=now_ts)
        should_save = True

    if symbol not in scores:
        scores[symbol] = {
            "score": 1.0,
            "penalized_until": None,
            "trades_count": 0,
        }

    entry = scores[symbol]

    # Automatic unban check
    if entry.get("penalized_until") is not None and now_ts >= entry["penalized_until"]:
        entry["penalized_until"] = None
        entry["score"] = 0.5
        logger.info("[PENALTY BOX] %s reinstated with baseline score 0.5", symbol)

    current_s = float(entry.get("score", 1.0))
    new_s = calculate_score_adjustment(
        score=current_s,
        realized_pnl_usd=realized_pnl_usd,
        realized_pnl_pct=realized_pnl_pct,
        trigger_reason=trigger_reason,
        candles_held=candles_held,
    )
    entry["score"] = new_s
    entry["trades_count"] = int(entry.get("trades_count", 0)) + 1

    # Penalty box trigger: S <= 0.0
    if new_s <= 0.0:
        entry["penalized_until"] = now_ts + 86400.0
        iso_time = datetime.fromtimestamp(entry["penalized_until"], tz=timezone.utc).isoformat()
        logger.warning(
            "[PENALTY BOX] %s score fell to %.1f (<= 0.0). Penalized for 24 hours until %s.",
            symbol,
            new_s,
            iso_time,
        )

    # Sync module-level dict
    asset_scores[symbol] = entry

    if should_save:
        from sentinel_trader.state_store import save_asset_scores
        save_asset_scores(scores, file_path=file_path)

    return entry


def get_asset_score(
    symbol: str,
    file_path: Optional[str] = None,
    current_time: Optional[float] = None,
    asset_scores_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Retrieve score info for a symbol, unbanning if expired."""
    now_ts = current_time if current_time is not None else time.time()
    if asset_scores_dict is not None:
        scores = asset_scores_dict
    else:
        from sentinel_trader.state_store import load_asset_scores
        scores = load_asset_scores(file_path=file_path, current_time=now_ts)

    if symbol not in scores:
        return {
            "score": 1.0,
            "penalized_until": None,
            "trades_count": 0,
        }

    entry = scores[symbol]
    if entry.get("penalized_until") is not None and now_ts >= entry["penalized_until"]:
        entry["penalized_until"] = None
        entry["score"] = 0.5
        if asset_scores_dict is None:
            from sentinel_trader.state_store import save_asset_scores
            save_asset_scores(scores, file_path=file_path)

    return entry


def get_all_asset_scores(
    file_path: Optional[str] = None,
    current_time: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve all asset reliability scores."""
    from sentinel_trader.state_store import load_asset_scores
    return load_asset_scores(file_path=file_path, current_time=current_time)


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
    candles_held: int = 0,
    asset_scores_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically record a closed trade into trade_history.json (or trade_history_mock.json)
    and update asset reliability score.

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
        candles_held: Number of 15m candles position was held (default 0).
        asset_scores_file: Optional path override for asset_scores.json.

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
        "candles_held": int(candles_held),
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

    # Update asset reliability score and manage penalty box
    scores_path = asset_scores_file
    if scores_path is None:
        if (
            file_path is not None
            and file_path != DEFAULT_TRADE_HISTORY_FILE
            and file_path != DEFAULT_MOCK_TRADE_HISTORY_FILE
        ):
            scores_path = os.path.join(target_dir, DEFAULT_ASSET_SCORES_FILE)
        else:
            scores_path = resolve_asset_scores_path(None, config)

    try:
        update_asset_score(
            symbol=symbol,
            realized_pnl_usd=float(realized_pnl_usd),
            realized_pnl_pct=float(realized_pnl_pct),
            trigger_reason=trigger_reason,
            candles_held=int(candles_held),
            file_path=scores_path,
        )
    except Exception as e:
        logger.error("Failed to update asset reliability score for %s: %s", symbol, e)

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


def calculate_drawdown_multiplier(trades: List[Dict[str, Any]]) -> float:
    """Calculate dynamic risk multiplier based on trade history drawdown ladder.

    Drawdown ladder rules:
    - 0 or 1 loss: Multiplier = 1.0x (Risk = 1.0% equity)
    - 2 consecutive losses: Multiplier = 0.5x (Risk cuts to 0.5% equity)
    - 3+ consecutive losses: Multiplier = 0.25x (Risk cuts to 0.25% equity floor)
    - 2 consecutive wins: Restores Multiplier step-wise up to 1.0x (e.g. 0.25 -> 0.5 -> 1.0)
    """
    multiplier = 1.0
    cons_losses = 0
    cons_wins = 0

    for t in trades:
        pnl = float(t.get("realized_pnl_usd", t.get("pnl_usd", t.get("pnl", 0.0))))
        if pnl == 0.0 and "realized_pnl_pct" in t:
            pnl = float(t["realized_pnl_pct"])

        if pnl < 0.0:
            cons_losses += 1
            cons_wins = 0
            if cons_losses >= 3:
                multiplier = 0.25
            elif cons_losses == 2:
                multiplier = min(multiplier, 0.5)
            else:
                multiplier = min(multiplier, 1.0)
        elif pnl > 0.0:
            cons_wins += 1
            cons_losses = 0
            if cons_wins >= 2 and cons_wins % 2 == 0:
                if multiplier == 0.25:
                    multiplier = 0.5
                elif multiplier == 0.5:
                    multiplier = 1.0
        else:
            # Flat/scratch trade with exactly 0 PnL
            pass

    return multiplier


def get_current_streak(
    window: int = 10,
    file_path: Optional[str] = None,
    config: Optional[Any] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Calculate rolling streak metrics and dynamic risk multiplier.

    Args:
        window: Lookback window of closed trades for rolling win rate calculation (default 10).
        file_path: Optional path override for trade history JSON file.
        config: Optional configuration object or boolean (to isolate mock runs).
        trades: Optional pre-loaded list of trade dictionaries.

    Returns:
        Dict[str, Any]:
            - consecutive_losses: int (current count of consecutive losing trades)
            - consecutive_wins: int (current count of consecutive winning trades)
            - rolling_win_rate: float (win rate over the last `window` trades, 0.0 to 1.0)
            - risk_multiplier: float (1.0, 0.5, or 0.25 based on drawdown ladder)
    """
    if trades is None:
        trades = load_trades(file_path=file_path, config=config)

    if not trades:
        return {
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "rolling_win_rate": 0.0,
            "risk_multiplier": 1.0,
        }

    # 1. Determine active consecutive loss or win streak from most recent trades backwards
    consecutive_losses = 0
    consecutive_wins = 0

    for t in reversed(trades):
        pnl = float(t.get("realized_pnl_usd", t.get("pnl_usd", t.get("pnl", 0.0))))
        if pnl == 0.0 and "realized_pnl_pct" in t:
            pnl = float(t["realized_pnl_pct"])

        if pnl < 0.0:
            if consecutive_wins > 0:
                break
            consecutive_losses += 1
        elif pnl > 0.0:
            if consecutive_losses > 0:
                break
            consecutive_wins += 1
        else:
            break

    # 2. Determine rolling win rate over recent window
    eff_window = window if window > 0 else 10
    recent_trades = trades[-eff_window:]
    win_count = 0
    for t in recent_trades:
        pnl = float(t.get("realized_pnl_usd", t.get("pnl_usd", t.get("pnl", 0.0))))
        if pnl == 0.0 and "realized_pnl_pct" in t:
            pnl = float(t["realized_pnl_pct"])
        if pnl > 0.0:
            win_count += 1

    rolling_win_rate = round(win_count / len(recent_trades), 4) if recent_trades else 0.0

    # 3. Determine drawdown ladder risk multiplier
    risk_multiplier = calculate_drawdown_multiplier(trades)

    return {
        "consecutive_losses": consecutive_losses,
        "consecutive_wins": consecutive_wins,
        "rolling_win_rate": rolling_win_rate,
        "risk_multiplier": risk_multiplier,
    }


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

    streak_info = get_current_streak(file_path=file_path, config=config, trades=trades)

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
        "consecutive_losses": streak_info["consecutive_losses"],
        "consecutive_wins": streak_info["consecutive_wins"],
        "rolling_win_rate": streak_info["rolling_win_rate"],
        "risk_multiplier": streak_info["risk_multiplier"],
        "streak": streak_info,
    }
