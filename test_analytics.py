"""Unit and integration test suite for Streak-Based Adaptive Risk Scaling (Drawdown Ladder).

Verifies:
1. Rolling streak tracker (`analytics.get_current_streak`):
   - Empty trade history baseline.
   - Consecutive losses count and multiplier reduction:
     * 0 or 1 loss -> 1.0x (Normal)
     * 2 consecutive losses -> 0.5x (Defensive)
     * 3+ consecutive losses -> 0.25x (Capital Preservation)
   - Consecutive wins count and step-wise recovery:
     * 1 win after losses -> stays at current multiplier
     * 2 consecutive wins -> steps up multiplier (0.5x -> 1.0x or 0.25x -> 0.5x)
     * 4 consecutive wins -> steps up from 0.25x to 1.0x
   - Rolling win rate across configured lookback window.
2. Configuration boundaries:
   - `BASE_RISK_PER_TRADE_PCT = 0.01`
   - `MIN_RISK_PER_TRADE_PCT = 0.0025`
3. Risk Engine Position Sizing Integration:
   - Volatility parity sizing scales down proportionally (1.0x -> 0.5x -> 0.25x).
   - Log message formatting: `[RISK ADAPTATION] Streak: {losses}L | Multiplier: {mult}x | Sizing Risk: {pct}%`.
4. Hourly & Discord Reporting (`notifier.py`):
   - `Risk Mode`: `1.0x (Normal)` vs `0.5x (Defensive)` vs `0.25x (Capital Preservation)`.
   - `Streak` field formatting.
"""

import json
import logging
import pytest

from sentinel_trader.analytics import (
    calculate_drawdown_multiplier,
    calculate_metrics,
    clear_trades,
    get_current_streak,
    record_trade,
)
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient
from sentinel_trader.notifier import DiscordNotifier
from sentinel_trader.risk_engine import RiskEngine


# =====================================================================
# 1. Rolling Streak Tracker & Multiplier Formula Tests
# =====================================================================

def test_get_current_streak_empty():
    """Verify empty trade history returns default baseline streak metrics."""
    streak = get_current_streak(trades=[])
    assert streak["consecutive_losses"] == 0
    assert streak["consecutive_wins"] == 0
    assert streak["rolling_win_rate"] == 0.0
    assert streak["risk_multiplier"] == 1.0


def test_multiplier_reduction_losing_streaks():
    """Verify multiplier reduction: 0/1 loss -> 1.0x, 2 losses -> 0.5x, 3+ losses -> 0.25x."""
    # 1 Loss: [Loss] -> 1.0x
    t1 = [{"realized_pnl_usd": -25.0, "realized_pnl_pct": -0.02}]
    s1 = get_current_streak(trades=t1)
    assert s1["consecutive_losses"] == 1
    assert s1["consecutive_wins"] == 0
    assert s1["risk_multiplier"] == 1.0

    # 2 Consecutive Losses: [Loss, Loss] -> 0.5x
    t2 = [
        {"realized_pnl_usd": -25.0, "realized_pnl_pct": -0.02},
        {"realized_pnl_usd": -30.0, "realized_pnl_pct": -0.025},
    ]
    s2 = get_current_streak(trades=t2)
    assert s2["consecutive_losses"] == 2
    assert s2["consecutive_wins"] == 0
    assert s2["risk_multiplier"] == 0.5

    # 3 Consecutive Losses: [Loss, Loss, Loss] -> 0.25x
    t3 = [
        {"realized_pnl_usd": -25.0, "realized_pnl_pct": -0.02},
        {"realized_pnl_usd": -30.0, "realized_pnl_pct": -0.025},
        {"realized_pnl_usd": -15.0, "realized_pnl_pct": -0.015},
    ]
    s3 = get_current_streak(trades=t3)
    assert s3["consecutive_losses"] == 3
    assert s3["consecutive_wins"] == 0
    assert s3["risk_multiplier"] == 0.25

    # 4+ Consecutive Losses: [Loss, Loss, Loss, Loss] -> 0.25x (floor)
    t4 = t3 + [{"realized_pnl_usd": -50.0, "realized_pnl_pct": -0.05}]
    s4 = get_current_streak(trades=t4)
    assert s4["consecutive_losses"] == 4
    assert s4["consecutive_wins"] == 0
    assert s4["risk_multiplier"] == 0.25


def test_multiplier_recovery_winning_streaks():
    """Verify multiplier recovery: 2 consecutive wins restores step-wise up to 1.0x."""
    # Sequence: [Loss, Loss, Win] -> 1 win does not restore yet, stays 0.5x
    t_partial = [
        {"realized_pnl_usd": -25.0},
        {"realized_pnl_usd": -30.0},
        {"realized_pnl_usd": 40.0},
    ]
    s_partial = get_current_streak(trades=t_partial)
    assert s_partial["consecutive_losses"] == 0
    assert s_partial["consecutive_wins"] == 1
    assert s_partial["risk_multiplier"] == 0.5

    # Sequence: [Loss, Loss, Win, Win] -> 2 consecutive wins restores to 1.0x
    t_restored = t_partial + [{"realized_pnl_usd": 35.0}]
    s_restored = get_current_streak(trades=t_restored)
    assert s_restored["consecutive_losses"] == 0
    assert s_restored["consecutive_wins"] == 2
    assert s_restored["risk_multiplier"] == 1.0

    # Step-wise recovery from 0.25x floor:
    # [Loss, Loss, Loss] -> 0.25x
    t_deep = [
        {"realized_pnl_usd": -20.0},
        {"realized_pnl_usd": -25.0},
        {"realized_pnl_usd": -30.0},
    ]
    assert calculate_drawdown_multiplier(t_deep) == 0.25

    # [Loss, Loss, Loss, Win] -> still 0.25x
    t_deep_win1 = t_deep + [{"realized_pnl_usd": 50.0}]
    s_deep_win1 = get_current_streak(trades=t_deep_win1)
    assert s_deep_win1["consecutive_wins"] == 1
    assert s_deep_win1["risk_multiplier"] == 0.25

    # [Loss, Loss, Loss, Win, Win] -> 2 wins steps up to 0.5x
    t_deep_win2 = t_deep_win1 + [{"realized_pnl_usd": 45.0}]
    s_deep_win2 = get_current_streak(trades=t_deep_win2)
    assert s_deep_win2["consecutive_wins"] == 2
    assert s_deep_win2["risk_multiplier"] == 0.5

    # [Loss, Loss, Loss, Win, Win, Win] -> 3 wins stays 0.5x
    t_deep_win3 = t_deep_win2 + [{"realized_pnl_usd": 60.0}]
    s_deep_win3 = get_current_streak(trades=t_deep_win3)
    assert s_deep_win3["consecutive_wins"] == 3
    assert s_deep_win3["risk_multiplier"] == 0.5

    # [Loss, Loss, Loss, Win, Win, Win, Win] -> 4 wins steps up to 1.0x
    t_deep_win4 = t_deep_win3 + [{"realized_pnl_usd": 55.0}]
    s_deep_win4 = get_current_streak(trades=t_deep_win4)
    assert s_deep_win4["consecutive_wins"] == 4
    assert s_deep_win4["risk_multiplier"] == 1.0


def test_rolling_win_rate_window():
    """Verify rolling win rate calculation respects the window size."""
    # 12 trades: 6 wins, 6 losses
    trades = [
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": -10.0},
        {"realized_pnl_usd": -10.0},
        {"realized_pnl_usd": -10.0},
        {"realized_pnl_usd": -10.0},
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": 10.0},
        {"realized_pnl_usd": -10.0},
        {"realized_pnl_usd": 10.0},
    ]
    # Total trades: 12. Last 10 trades: 5 losses, 5 wins -> 5 wins / 10 = 50% = 0.5
    streak10 = get_current_streak(trades=trades, window=10)
    assert streak10["rolling_win_rate"] == 0.5

    # Last 4 trades: [10, 10, -10, 10] -> 3 wins out of 4 = 75%
    streak4 = get_current_streak(trades=trades, window=4)
    assert streak4["rolling_win_rate"] == 0.75


def test_streak_persistence_from_file(tmp_path):
    """Verify streak calculation works when reading from disk persistence file."""
    trade_file = str(tmp_path / "trade_history.json")

    # Record 2 losses to disk
    record_trade("SOL/USDT", 100.0, 95.0, 1.0, -5.0, -0.05, "STOP_LOSS", file_path=trade_file)
    record_trade("ETH/USDT", 2000.0, 1950.0, 0.5, -25.0, -0.025, "STOP_LOSS", file_path=trade_file)

    streak = get_current_streak(file_path=trade_file)
    assert streak["consecutive_losses"] == 2
    assert streak["risk_multiplier"] == 0.5

    # Record 2 wins to disk
    record_trade("SOL/USDT", 100.0, 105.0, 1.0, 5.0, 0.05, "TAKE_PROFIT", file_path=trade_file)
    record_trade("BTC/USDT", 60000.0, 63000.0, 0.1, 300.0, 0.05, "TAKE_PROFIT", file_path=trade_file)

    streak_recovered = get_current_streak(file_path=trade_file)
    assert streak_recovered["consecutive_losses"] == 0
    assert streak_recovered["consecutive_wins"] == 2
    assert streak_recovered["risk_multiplier"] == 1.0


# =====================================================================
# 2. Configuration Parameters Tests
# =====================================================================

def test_config_risk_parameters_defaults():
    """Verify BASE_RISK_PER_TRADE_PCT and MIN_RISK_PER_TRADE_PCT configuration parameters."""
    cfg = Config()
    assert cfg.base_risk_per_trade_pct == 0.01
    assert cfg.min_risk_per_trade_pct == 0.0025

    # Custom override
    custom_cfg = Config(base_risk_per_trade_pct=0.02, min_risk_per_trade_pct=0.005)
    assert custom_cfg.base_risk_per_trade_pct == 0.02
    assert custom_cfg.min_risk_per_trade_pct == 0.005


def test_config_risk_parameters_validation():
    """Verify risk percentage bounds validation."""
    with pytest.raises(ValueError):
        Config(base_risk_per_trade_pct=-0.01)

    with pytest.raises(ValueError):
        Config(min_risk_per_trade_pct=0.0)

    with pytest.raises(ValueError):
        Config(base_risk_per_trade_pct=1.5)


# =====================================================================
# 3. Position Sizing Integration & Proportional Scale Down Tests
# =====================================================================

def test_position_sizing_proportional_scale_down(tmp_path):
    """Verify position size scales down proportionally (1.0x -> 0.5x -> 0.25x) during losing streaks."""
    trade_file = str(tmp_path / "trade_history_ladder.json")
    config = Config(
        mock_exchange=True,
        base_risk_per_trade_pct=0.01,
        min_risk_per_trade_pct=0.0025,
        atr_sl_multiplier=1.5,
    )
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange, trade_history_file=trade_file)

    free_usdt = 10000.0
    total_equity = 10000.0
    entry_price = 100.0
    atr = 5.0
    # Stop distance = 5.0 * 1.5 = 7.5

    # 1. Baseline: 0 losses -> 1.0x Multiplier (1.0% equity risk = $100 risk)
    # Target quantity = 100.0 / 7.5 = 13.3333
    # Sizing cost = 13.3333 * 100 = $1333.33
    cost_base = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=atr,
        entry_price=entry_price,
        total_equity=total_equity,
    )
    assert cost_base == pytest.approx(1333.33, abs=0.5)

    # 2. Add 2 losses to history: -> 0.5x Multiplier (0.5% equity risk = $50 risk)
    # Target quantity = 50.0 / 7.5 = 6.6667
    # Sizing cost = 6.6667 * 100 = $666.67 (exactly 50% of baseline)
    record_trade("SOL/USDT", 100.0, 95.0, 1.0, -5.0, -0.05, "STOP_LOSS", file_path=trade_file)
    record_trade("ETH/USDT", 2000.0, 1950.0, 0.5, -25.0, -0.025, "STOP_LOSS", file_path=trade_file)

    cost_05x = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=atr,
        entry_price=entry_price,
        total_equity=total_equity,
    )
    assert cost_05x == pytest.approx(666.67, abs=0.5)
    assert cost_05x == pytest.approx(cost_base * 0.5, abs=0.5)

    # 3. Add 3rd loss: -> 0.25x Multiplier (0.25% equity risk = $25 risk)
    # Target quantity = 25.0 / 7.5 = 3.3333
    # Sizing cost = 3.3333 * 100 = $333.33 (exactly 25% of baseline)
    record_trade("BTC/USDT", 60000.0, 58000.0, 0.05, -100.0, -0.033, "STOP_LOSS", file_path=trade_file)

    cost_025x = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=atr,
        entry_price=entry_price,
        total_equity=total_equity,
    )
    assert cost_025x == pytest.approx(333.33, abs=0.5)
    assert cost_025x == pytest.approx(cost_base * 0.25, abs=0.5)

    # 4. Recovery: 2 consecutive wins steps up to 0.5x
    record_trade("SOL/USDT", 100.0, 105.0, 1.0, 5.0, 0.05, "TAKE_PROFIT", file_path=trade_file)
    record_trade("ETH/USDT", 2000.0, 2100.0, 0.5, 50.0, 0.05, "TAKE_PROFIT", file_path=trade_file)

    cost_recovered_step1 = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=atr,
        entry_price=entry_price,
        total_equity=total_equity,
    )
    assert cost_recovered_step1 == pytest.approx(666.67, abs=0.5)

    # 5. Recovery: another 2 consecutive wins restores to 1.0x
    record_trade("BTC/USDT", 60000.0, 63000.0, 0.05, 150.0, 0.05, "TAKE_PROFIT", file_path=trade_file)
    record_trade("SOL/USDT", 100.0, 106.0, 1.0, 6.0, 0.06, "TAKE_PROFIT", file_path=trade_file)

    cost_fully_recovered = risk.calculate_slot_size(
        free_usdt=free_usdt,
        atr=atr,
        entry_price=entry_price,
        total_equity=total_equity,
    )
    assert cost_fully_recovered == pytest.approx(1333.33, abs=0.5)


def test_risk_adaptation_logging(caplog, tmp_path):
    """Verify log output includes [RISK ADAPTATION] Streak: {losses}L | Multiplier: {mult}x | Sizing Risk: {pct}%."""
    trade_file = str(tmp_path / "trade_history_log.json")
    record_trade("SOL/USDT", 100.0, 95.0, 1.0, -5.0, -0.05, "STOP_LOSS", file_path=trade_file)
    record_trade("ETH/USDT", 2000.0, 1950.0, 0.5, -25.0, -0.025, "STOP_LOSS", file_path=trade_file)

    config = Config(mock_exchange=True, base_risk_per_trade_pct=0.01)
    exchange = ExchangeClient(config)
    risk = RiskEngine(config, exchange, trade_history_file=trade_file)

    with caplog.at_level(logging.INFO):
        risk.calculate_volatility_parity_sizing(
            free_usdt=10000.0,
            atr=5.0,
            entry_price=100.0,
            total_equity=10000.0,
        )

    assert "[RISK ADAPTATION]" in caplog.text
    assert "Streak: 2L" in caplog.text
    assert "Multiplier: 0.5x" in caplog.text
    assert "Sizing Risk: 0.50%" in caplog.text


# =====================================================================
# 4. Hourly & Discord Reporting Embed Tests
# =====================================================================

def test_hourly_summary_risk_mode_embed():
    """Verify Risk Mode and Streak fields in hourly performance summary embed."""
    config = Config(mock_exchange=True)
    notifier = DiscordNotifier(config)

    # 1. Normal Mode: 1.0x
    metrics_normal = {
        "total_trades": 5,
        "wins": 4,
        "losses": 1,
        "win_rate_pct": 80.0,
        "total_realized_pnl_usd": 120.0,
        "total_realized_pnl_pct": 2.5,
        "gross_profit_usd": 150.0,
        "gross_loss_usd": 30.0,
        "rolling_1h_pnl_usd": 50.0,
        "free_usdt": 5000.0,
        "active_positions": {},
        "risk_multiplier": 1.0,
        "consecutive_losses": 0,
        "consecutive_wins": 3,
    }
    embed_normal = notifier.build_hourly_summary_embed(metrics_normal)
    field_map_normal = {f["name"]: f["value"] for f in embed_normal["fields"]}
    assert "Risk Mode" in field_map_normal
    assert field_map_normal["Risk Mode"] == "1.0x (Normal)"
    assert "Streak" in field_map_normal
    assert field_map_normal["Streak"] == "3W"

    # 2. Defensive Mode: 0.5x
    metrics_defensive = dict(metrics_normal)
    metrics_defensive.update({
        "risk_multiplier": 0.5,
        "consecutive_losses": 2,
        "consecutive_wins": 0,
    })
    embed_defensive = notifier.build_hourly_summary_embed(metrics_defensive)
    field_map_defensive = {f["name"]: f["value"] for f in embed_defensive["fields"]}
    assert field_map_defensive["Risk Mode"] == "0.5x (Defensive)"
    assert field_map_defensive["Streak"] == "2L"

    # 3. Capital Preservation Mode: 0.25x
    metrics_preservation = dict(metrics_normal)
    metrics_preservation.update({
        "risk_multiplier": 0.25,
        "consecutive_losses": 4,
        "consecutive_wins": 0,
    })
    embed_preservation = notifier.build_hourly_summary_embed(metrics_preservation)
    field_map_preservation = {f["name"]: f["value"] for f in embed_preservation["fields"]}
    assert field_map_preservation["Risk Mode"] == "0.25x (Capital Preservation)"
    assert field_map_preservation["Streak"] == "4L"
