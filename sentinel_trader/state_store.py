"""Atomic state persistence module for Sentinel-Trader crash recovery.

Provides crash-resilient local persistence of multi-slot bot operational state using
temporary file writes and atomic filesystem replacement (os.replace) to eliminate
the risk of file corruption on unexpected crashes or power loss.

Multi-Position Schema:
{
  "active_positions": {
    "SOL/USDT": {"entry_price": 150.0, "amount": 10.0, "order_id": "..."},
    "ETH/USDT": {"entry_price": 2600.0, "amount": 0.5, "order_id": "..."}
  },
  "screener_cache": ["BTC/USDT", "ETH/USDT", ...],
  "screener_timestamp": 1727412000
}
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("sentinel.state_store")

DEFAULT_STATE_FILE = "state.json"


def save_state(
    state: Union[str, Any] = "IN_POSITION",
    entry_price: Optional[float] = None,
    holding_amount: Optional[float] = None,
    entry_timestamp: Optional[Union[str, float]] = None,
    client_order_id: Optional[str] = None,
    last_known_pnl: Optional[float] = None,
    active_positions: Optional[Dict[str, Any]] = None,
    screener_cache: Optional[List[str]] = None,
    screener_timestamp: Optional[Union[float, int]] = None,
    file_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Atomically persist multi-position state data to state.json using os.replace.

    Args:
        state: Bot operational state (BotState enum, string, or dictionary).
        entry_price: Cost basis entry price (for single position or backward compatibility).
        holding_amount: Quantity of base asset currently held.
        entry_timestamp: ISO-8601 or unix timestamp when position was opened.
        client_order_id: Unique exchange client order ID.
        last_known_pnl: Last recorded unrealized PnL percentage.
        active_positions: Dictionary of active slot positions {symbol: {...}}.
        screener_cache: List of screened candidate universe symbols.
        screener_timestamp: Timestamp when screener cache was captured.
        file_path: Optional custom file path (defaults to 'state.json').
        **kwargs: Additional metadata to persist.

    Returns:
        Dict[str, Any]: The saved state dictionary.
    """
    target_path = file_path or DEFAULT_STATE_FILE
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)

    if isinstance(state, dict):
        payload = state.copy()
        payload.update(kwargs)
    else:
        state_str = state.value if hasattr(state, "value") else str(state)
        payload = {"state": state_str, **kwargs}

    # 1. Format active_positions
    positions_dict: Dict[str, Any] = {}
    if active_positions is not None:
        for sym, pos_data in active_positions.items():
            if hasattr(pos_data, "to_dict"):
                positions_dict[sym] = pos_data.to_dict()
            elif isinstance(pos_data, dict):
                positions_dict[sym] = pos_data
            elif hasattr(pos_data, "__dict__"):
                positions_dict[sym] = vars(pos_data)
    elif "active_positions" in payload and isinstance(payload["active_positions"], dict):
        positions_dict = payload["active_positions"]
    elif entry_price is not None or holding_amount is not None:
        target_sym = kwargs.get("target_symbol") or kwargs.get("symbol") or "SOL/USDT"
        positions_dict[target_sym] = {
            "entry_price": float(entry_price) if entry_price is not None else None,
            "amount": float(holding_amount) if holding_amount is not None else 0.0,
            "order_id": str(client_order_id) if client_order_id is not None else None,
            "entry_timestamp": str(entry_timestamp) if entry_timestamp is not None else None,
            "last_known_pnl": float(last_known_pnl) if last_known_pnl is not None else 0.0,
            "highest_price": float(kwargs.get("highest_price") if kwargs.get("highest_price") is not None else (entry_price or 0.0)),
            "atr_at_entry": float(kwargs.get("atr_at_entry") or 0.0),
            "current_stop_price": float(kwargs.get("current_stop_price") or 0.0),
            "candles_held": int(kwargs.get("candles_held") or 0),
        }

    payload["active_positions"] = positions_dict

    # 2. Screener cache and timestamp
    if screener_cache is not None:
        payload["screener_cache"] = list(screener_cache)
    elif "screener_cache" not in payload:
        payload["screener_cache"] = []

    if screener_timestamp is not None:
        payload["screener_timestamp"] = screener_timestamp
    elif "screener_timestamp" not in payload:
        payload["screener_timestamp"] = None

    # 3. Maintain top-level backward compatibility keys for legacy tests & quant state
    if entry_price is not None:
        payload["entry_price"] = float(entry_price)
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        if "entry_price" in first_pos and first_pos["entry_price"] is not None:
            payload["entry_price"] = float(first_pos["entry_price"])

    if holding_amount is not None:
        payload["holding_amount"] = float(holding_amount)
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        amt = first_pos.get("amount") if "amount" in first_pos else first_pos.get("holding_amount")
        if amt is not None:
            payload["holding_amount"] = float(amt)

    if entry_timestamp is not None:
        payload["entry_timestamp"] = str(entry_timestamp)
    if client_order_id is not None:
        payload["client_order_id"] = str(client_order_id)
    if last_known_pnl is not None:
        payload["last_known_pnl"] = float(last_known_pnl)

    # Quant telemetry top-level persistence
    if "highest_price" in kwargs:
        payload["highest_price"] = float(kwargs["highest_price"])
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        if "highest_price" in first_pos:
            payload["highest_price"] = first_pos["highest_price"]

    if "current_stop_price" in kwargs:
        payload["current_stop_price"] = float(kwargs["current_stop_price"])
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        if "current_stop_price" in first_pos:
            payload["current_stop_price"] = first_pos["current_stop_price"]

    if "atr_at_entry" in kwargs:
        payload["atr_at_entry"] = float(kwargs["atr_at_entry"])
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        if "atr_at_entry" in first_pos:
            payload["atr_at_entry"] = first_pos["atr_at_entry"]

    if "candles_held" in kwargs:
        payload["candles_held"] = int(kwargs["candles_held"])
    elif positions_dict:
        first_pos = next(iter(positions_dict.values()))
        if "candles_held" in first_pos:
            payload["candles_held"] = first_pos["candles_held"]

    # Normalize state key if enum
    if hasattr(payload.get("state"), "value"):
        payload["state"] = payload["state"].value

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, encoding="utf-8") as f:
            temp_path = f.name
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # CRITICAL FOR WINDOWS: File handle must be closed before calling os.replace
        os.replace(temp_path, target_path)
        logger.debug("State atomically persisted to %s", target_path)
    except Exception as e:
        logger.error("Failed to atomically persist state to %s: %s", target_path, e)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

    return payload


def load_state(file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load persistent bot state from state.json if present and valid.

    Args:
        file_path: Optional custom file path (defaults to 'state.json').

    Returns:
        Optional[Dict[str, Any]]: Persisted state dictionary, or None if file doesn't exist or is invalid.
    """
    target_path = file_path or DEFAULT_STATE_FILE
    if not os.path.exists(target_path):
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Corrupted state file %s (not a valid JSON object). Ignoring.", target_path)
                return None

            # Reconcile multi-position schema
            if "active_positions" not in data:
                data["active_positions"] = {}
                if data.get("entry_price") is not None or data.get("holding_amount") is not None:
                    target_sym = data.get("target_symbol") or data.get("symbol") or "SOL/USDT"
                    entry_p = data.get("entry_price")
                    data["active_positions"][target_sym] = {
                        "entry_price": entry_p,
                        "amount": data.get("holding_amount"),
                        "order_id": data.get("client_order_id"),
                        "entry_timestamp": data.get("entry_timestamp"),
                        "last_known_pnl": data.get("last_known_pnl"),
                        "highest_price": data.get("highest_price") or entry_p,
                        "atr_at_entry": data.get("atr_at_entry", 0.0),
                        "current_stop_price": data.get("current_stop_price", 0.0),
                        "candles_held": data.get("candles_held", 0),
                    }

            if "screener_cache" not in data:
                data["screener_cache"] = []
            if "screener_timestamp" not in data:
                data["screener_timestamp"] = None

            # Reconcile legacy top-level keys if active_positions present
            if data["active_positions"]:
                first_pos = next(iter(data["active_positions"].values()))
                if data.get("entry_price") is None and "entry_price" in first_pos:
                    data["entry_price"] = first_pos["entry_price"]
                if data.get("holding_amount") is None:
                    data["holding_amount"] = first_pos.get("amount") or first_pos.get("holding_amount")
                if data.get("highest_price") is None and "highest_price" in first_pos:
                    data["highest_price"] = first_pos["highest_price"]
                if data.get("current_stop_price") is None and "current_stop_price" in first_pos:
                    data["current_stop_price"] = first_pos["current_stop_price"]
                if data.get("atr_at_entry") is None and "atr_at_entry" in first_pos:
                    data["atr_at_entry"] = first_pos["atr_at_entry"]
                if data.get("candles_held") is None and "candles_held" in first_pos:
                    data["candles_held"] = first_pos["candles_held"]

            return data
    except Exception as e:
        logger.warning("Error reading state file %s: %s. Ignoring corrupted state.", target_path, e)
        return None


def clear_state(file_path: Optional[str] = None) -> bool:
    """Clear and remove the state persistence file.

    Args:
        file_path: Optional custom file path (defaults to 'state.json').

    Returns:
        bool: True if cleared successfully or did not exist; False on deletion error.
    """
    target_path = file_path or DEFAULT_STATE_FILE
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            logger.info("Cleared persistent state file: %s", target_path)
            return True
        except Exception as e:
            logger.error("Failed to remove state file %s: %s", target_path, e)
            return False
    return True
