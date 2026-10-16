"""Configuration module for Sentinel-Trader.

Loads and validates environment variables, exchange credentials,
and risk parameters with strict safety defaults.
"""

from typing import Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Sentinel-Trader system configuration and risk boundaries."""

    model_config = SettingsConfigDict(
        env_file=(".env", "sentinel_trader/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Exchange Configuration
    binance_api_key: str = Field(
        default="",
        description="Binance API Key (Testnet or Live)",
    )
    binance_api_secret: str = Field(
        default="",
        description="Binance API Secret (Testnet or Live)",
    )
    binance_sandbox: bool = Field(
        default=True,
        description="SAFETY INVARIANT: Default strictly to Binance Testnet sandbox mode",
    )
    allow_production_trading: bool = Field(
        default=False,
        description="SAFETY LOCK: Must be explicitly True to operate in production mode",
    )

    # LLM Sentinel Configuration
    llm_provider: Literal["gemini", "openai", "mock"] = Field(
        default="gemini",
        description="LLM provider for news threat analysis ('gemini', 'openai', or 'mock')",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for chosen LLM provider",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model name override (defaults to gemini-2.5-flash or gpt-4o-mini)",
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Alias for Gemini model override if specified as GEMINI_MODEL",
    )

    # Trading & Market Configuration
    target_symbol: str = Field(
        default="SOL/USDT",
        description="Trading pair symbol to monitor and protect",
    )
    timeframe: str = Field(
        default="15m",
        description="Candle timeframe for OHLCV strategy analysis (e.g. '15m')",
    )
    allocation_pct: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Fraction of free quote asset (USDT) to allocate per entry (e.g. 0.25 for 25%)",
    )
    dynamic_universe_size: int = Field(
        default=15,
        ge=1,
        description="Number of top volume assets to monitor in dynamic universe (default 15)",
    )
    max_active_positions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum concurrent active slot positions (default 3)",
    )
    slot_allocation_pct: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Fraction of free quote asset to allocate per slot entry (default 0.25 for 25%)",
    )
    cooldown_minutes: int = Field(
        default=15,
        ge=0,
        description="Cooldown sleep duration in minutes after a position is closed before scanning for re-entry",
    )
    rsi_oversold_threshold: float = Field(
        default=38.0,
        ge=1.0,
        le=99.0,
        description="RSI oversold entry threshold (default 38.0 to catch shallow pullbacks)",
    )
    ema_period: int = Field(
        default=100,
        ge=1,
        description="EMA trend filter lookback period (default 100 for faster responsiveness)",
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Initial position cost basis. If None, captures initial market price at launch",
    )

    # Mathematical Risk Engine Thresholds & Institutional Quant Mechanics
    take_profit_pct: float = Field(
        default=0.05,
        description="Hard mathematical Take-Profit limit (e.g. 0.05 for +5%)",
    )
    stop_loss_pct: float = Field(
        default=0.025,
        description="Hard mathematical Stop-Loss limit (e.g. 0.025 for -2.5%)",
    )
    portfolio_risk_per_trade_pct: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.20,
        description="Fixed Fractional Risk: fraction of total portfolio equity risked per trade (default 0.01 = 1.0%)",
    )
    base_risk_per_trade_pct: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.20,
        description="Base Risk Per Trade: Base fraction of total portfolio equity risked per trade (default 0.01 = 1.0%)",
    )
    min_risk_per_trade_pct: float = Field(
        default=0.0025,
        ge=0.0001,
        le=0.10,
        description="Minimum Risk Per Trade: Floor fraction of total portfolio equity risked per trade (default 0.0025 = 0.25%)",
    )
    atr_sl_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        le=10.0,
        description="ATR multiplier for initial stop-loss distance (default 1.5)",
    )
    max_holding_candles: int = Field(
        default=16,
        ge=1,
        description="Maximum holding candles before Alpha Decay Time Stop liquidation (default 16 = 4 hours on 15m)",
    )
    panic_sell_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum AI confidence score (0.0 - 1.0) required to execute panic sell",
    )
    liquidation_order_type: Literal["limit_ioc", "market"] = Field(
        default="limit_ioc",
        description="Order type for emergency liquidation: 'limit_ioc' (slippage-bounded IOC) or 'market'",
    )
    max_liquidation_slippage_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.20,
        description="Max acceptable slippage bound for emergency liquidation (e.g. 0.02 for 2%)",
    )
    fee_buffer_pct: float = Field(
        default=0.001,
        ge=0.0,
        le=0.05,
        description="Safety fee deduction buffer (e.g. 0.001 for 0.1%) when fees may be charged in base asset",
    )

    # Polling & Loop Cadence
    poll_interval_seconds: float = Field(
        default=3.0,
        description="Market price and PnL polling interval in seconds",
    )
    news_poll_interval_seconds: float = Field(
        default=5.0,
        description="News watcher ingestion polling interval in seconds",
    )
    max_news_age_minutes: int = Field(
        default=30,
        ge=1,
        description="Maximum age of news articles in minutes for freshness validation",
    )

    # Simulation & Testing Toggles
    mock_exchange: bool = Field(
        default=False,
        description="If True, runs simulated exchange in-memory without remote API calls",
    )
    enable_simulated_news: bool = Field(
        default=True,
        description="If True, provides synthetic crypto emergency and normal headlines for testing",
    )
    enable_rss_news: bool = Field(
        default=True,
        description="If True, polls public crypto RSS feeds for live breaking news",
    )

    # Discord Notification Subsystem
    discord_webhook_url: str = Field(
        default="",
        description="Discord incoming webhook URL for real-time trade and threat alerts",
    )
    hourly_summary_enabled: bool = Field(
        default=True,
        description="If True, runs automated hourly Discord performance reports aligned to the top of the hour",
    )

    @field_validator("take_profit_pct")
    @classmethod
    def validate_take_profit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"TAKE_PROFIT_PCT must be positive, got {v}")
        if v > 10.0:
            raise ValueError(f"TAKE_PROFIT_PCT unreasonably high (>1000%): {v}")
        return v

    @field_validator("stop_loss_pct")
    @classmethod
    def validate_stop_loss(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"STOP_LOSS_PCT must be positive, got {v}")
        if v >= 1.0:
            raise ValueError(f"STOP_LOSS_PCT must be less than 1.0 (100%), got {v}")
        return v

    @field_validator("allocation_pct", "slot_allocation_pct")
    @classmethod
    def validate_allocation(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Allocation percentage must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("dynamic_universe_size", "max_active_positions")
    @classmethod
    def validate_positive_counts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Universe size and position counts must be strictly positive, got {v}")
        return v

    @field_validator("cooldown_minutes")
    @classmethod
    def validate_cooldown(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"COOLDOWN_MINUTES must be non-negative, got {v}")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("TIMEFRAME cannot be empty")
        return clean

    @field_validator("poll_interval_seconds", "news_poll_interval_seconds")
    @classmethod
    def validate_positive_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Polling intervals must be strictly positive, got {v}")
        return v

    @field_validator("max_news_age_minutes")
    @classmethod
    def validate_max_news_age(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"MAX_NEWS_AGE_MINUTES must be strictly positive, got {v}")
        return v

    @field_validator("panic_sell_confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"panic_sell_confidence_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("rsi_oversold_threshold")
    @classmethod
    def validate_rsi_threshold(cls, v: float) -> float:
        if not (0.0 < v < 100.0):
            raise ValueError(f"RSI_OVERSOLD_THRESHOLD must be between 0 and 100, got {v}")
        return v

    @field_validator("ema_period")
    @classmethod
    def validate_ema_period(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"EMA_PERIOD must be positive, got {v}")
        return v

    @field_validator("base_risk_per_trade_pct", "min_risk_per_trade_pct")
    @classmethod
    def validate_risk_pct(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"Risk percentage must be between 0.0 and 1.0, got {v}")
        return v

    @model_validator(mode="after")
    def enforce_sandbox_safety(self) -> "Config":
        """Safety Invariant: Prevent accidental live trading without explicit double confirmation."""
        if not self.binance_sandbox and not self.allow_production_trading:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: BINANCE_SANDBOX is False but ALLOW_PRODUCTION_TRADING is not True. "
                "Sentinel-Trader requires explicit environment confirmation to trade on live mainnet!"
            )
        # Keep base_risk_per_trade_pct and portfolio_risk_per_trade_pct synchronized
        if self.portfolio_risk_per_trade_pct != 0.01 and self.base_risk_per_trade_pct == 0.01:
            self.base_risk_per_trade_pct = self.portfolio_risk_per_trade_pct
        elif self.base_risk_per_trade_pct != 0.01 and self.portfolio_risk_per_trade_pct == 0.01:
            self.portfolio_risk_per_trade_pct = self.base_risk_per_trade_pct
        return self

    @property
    def base_asset(self) -> str:
        """Extract the base asset from TARGET_SYMBOL (e.g. 'SOL' from 'SOL/USDT')."""
        parts = self.target_symbol.split("/")
        return parts[0].strip().upper()

    @property
    def quote_asset(self) -> str:
        """Extract the quote asset from TARGET_SYMBOL (e.g. 'USDT' from 'SOL/USDT')."""
        parts = self.target_symbol.split("/")
        return parts[1].strip().upper() if len(parts) > 1 else "USDT"

    def get_effective_llm_model(self) -> str:
        """Resolve LLM model name based on provider if not explicitly overridden."""
        model = self.llm_model or self.gemini_model
        if model:
            if model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
                return "gemini-3.1-flash-lite"
            return model
        if self.llm_provider == "openai":
            return "gpt-4o-mini"
        return "gemini-3.1-flash-lite"
