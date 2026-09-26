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
    entry_price: Optional[float] = Field(
        default=None,
        description="Initial position cost basis. If None, captures initial market price at launch",
    )

    # Mathematical Risk Engine Thresholds
    take_profit_pct: float = Field(
        default=0.10,
        description="Hard mathematical Take-Profit limit (e.g. 0.10 for +10%)",
    )
    stop_loss_pct: float = Field(
        default=0.04,
        description="Hard mathematical Stop-Loss limit (e.g. 0.04 for -4%)",
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

    @field_validator("poll_interval_seconds", "news_poll_interval_seconds")
    @classmethod
    def validate_positive_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Polling intervals must be strictly positive, got {v}")
        return v

    @field_validator("panic_sell_confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"panic_sell_confidence_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @model_validator(mode="after")
    def enforce_sandbox_safety(self) -> "Config":
        """Safety Invariant: Prevent accidental live trading without explicit double confirmation."""
        if not self.binance_sandbox and not self.allow_production_trading:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: BINANCE_SANDBOX is False but ALLOW_PRODUCTION_TRADING is not True. "
                "Sentinel-Trader requires explicit environment confirmation to trade on live mainnet!"
            )
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
