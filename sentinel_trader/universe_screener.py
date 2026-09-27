"""Dynamic 24-Hour Volume Universe Screener for Sentinel-Trader.

Fetches 24-hour ticker volume from Binance, filters out inactive markets, non-USDT pairs,
stablecoins, fiat, and leveraged tokens, and returns the top liquid candidates sorted
descending by 24-hour USDT turnover (quoteVolume).
Includes an in-memory 24-hour cache (TTL 86,400s) and liquid mock exchange fallback.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sentinel.screener")

STABLECOIN_BLACKLIST: Set[str] = {
    "USDC/USDT",
    "FDUSD/USDT",
    "TUSD/USDT",
    "BUSD/USDT",
    "EUR/USDT",
    "DAI/USDT",
    "USDP/USDT",
    "AEUR/USDT",
}

LEVERAGED_PATTERNS: Tuple[str, ...] = ("UP/", "DOWN/", "BULL/", "BEAR/")

DEFAULT_MOCK_UNIVERSE: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "NEAR/USDT",
]


class UniverseScreener:
    """Dynamic universe screener maintaining a 24-hour in-memory cache."""

    def __init__(self, cache_ttl_seconds: int = 86400):
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_universe: Optional[List[str]] = None
        self._cache_timestamp: Optional[float] = None

    @property
    def cached_universe(self) -> Optional[List[str]]:
        """Access cached universe list."""
        return self._cached_universe

    @property
    def cache_timestamp(self) -> Optional[float]:
        """Access timestamp when universe was cached."""
        return self._cache_timestamp

    def is_cache_valid(self) -> bool:
        """Check if cached universe exists and has not expired."""
        if self._cached_universe is None or self._cache_timestamp is None:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl_seconds

    def clear_cache(self) -> None:
        """Invalidate the in-memory cache."""
        self._cached_universe = None
        self._cache_timestamp = None

    def set_cache(self, universe: List[str], timestamp: Optional[float] = None) -> None:
        """Seed or restore in-memory cache from persistence."""
        self._cached_universe = list(universe)
        self._cache_timestamp = timestamp or time.time()

    async def get_dynamic_universe(
        self,
        exchange: Any,
        top_n: int = 15,
        mock_exchange: Optional[bool] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        """Fetch all 24h tickers in a single call and return top_n pairs by quoteVolume.

        Filters:
        - Must be an active spot market (market['spot'] is True and market['active'] is True).
        - Symbol must settle in USDT (symbol.endswith('/USDT')).
        - Exclude Blacklisted Stablecoins & Fiat (STABLECOIN_BLACKLIST).
        - Exclude leveraged tokens (symbols containing UP/, DOWN/, BULL/, BEAR/).
        - Sort remaining pairs descending by ticker['quoteVolume'] (24-hour USDT turnover).
        - Cache result for 24 hours unless force_refresh is True.
        """
        # 1. Mock exchange support
        is_mock = False
        if mock_exchange is True:
            is_mock = True
        elif mock_exchange is None:
            if getattr(exchange, "is_mock", None) is True:
                is_mock = True
            elif hasattr(exchange, "config"):
                cfg = getattr(exchange, "config", None)
                if cfg is not None and getattr(cfg, "mock_exchange", None) is True:
                    is_mock = True

        if is_mock:
            logger.info("[UNIVERSE SCREENER] Mock exchange active. Returning default liquid mock universe.")
            return DEFAULT_MOCK_UNIVERSE[:top_n]

        # 2. In-memory 24-Hour Cache evaluation
        if not force_refresh and self.is_cache_valid():
            logger.debug(
                "[UNIVERSE SCREENER] Cache hit: Returning cached universe (age: %.1fs)",
                time.time() - (self._cache_timestamp or 0),
            )
            return (self._cached_universe or [])[:top_n]

        # 3. Single call to fetch all 24h tickers
        logger.info("[UNIVERSE SCREENER] Fetching all 24h tickers from exchange...")
        raw_tickers: Dict[str, Any] = await exchange.fetch_tickers()
        if not raw_tickers:
            logger.warning("[UNIVERSE SCREENER] Empty tickers response received from exchange.")
            return (self._cached_universe or [])[:top_n]

        # Resolve exchange markets dictionary if available
        markets: Dict[str, Any] = {}
        if hasattr(exchange, "markets") and isinstance(exchange.markets, dict):
            markets = exchange.markets
        elif hasattr(exchange, "_exchange") and hasattr(exchange._exchange, "markets"):
            markets = getattr(exchange._exchange, "markets", {}) or {}

        filtered: List[Tuple[str, float]] = []

        for symbol, ticker in raw_tickers.items():
            if not symbol or not isinstance(symbol, str):
                continue

            # Must settle in USDT
            if not symbol.endswith("/USDT"):
                continue

            # Exclude Blacklisted Stablecoins & Fiat
            if symbol in STABLECOIN_BLACKLIST:
                continue

            # Exclude leveraged tokens (UP/, DOWN/, BULL/, BEAR/)
            if any(p in symbol for p in LEVERAGED_PATTERNS):
                continue

            # Resolve market info for spot & active flags
            market: Optional[Dict[str, Any]] = None
            if markets and isinstance(markets, dict) and symbol in markets and isinstance(markets[symbol], dict):
                market = markets[symbol]
            elif isinstance(ticker.get("market"), dict):
                market = ticker["market"]
            elif hasattr(exchange, "market") and callable(getattr(exchange, "market", None)):
                try:
                    res = exchange.market(symbol)
                    if asyncio.iscoroutine(res):
                        res.close()
                    elif isinstance(res, dict):
                        market = res
                except Exception:
                    market = None

            if market is None and isinstance(ticker, dict):
                market = ticker

            is_spot = False
            is_active = False
            if isinstance(market, dict):
                is_spot = market.get("spot") is True
                is_active = market.get("active") is True

            if not (is_spot and is_active):
                continue

            # Extract quoteVolume (24-hour USDT turnover)
            quote_vol = ticker.get("quoteVolume")
            if quote_vol is None and isinstance(ticker.get("info"), dict):
                quote_vol = ticker["info"].get("quoteVolume")
            if quote_vol is None:
                quote_vol = 0.0

            try:
                qv_float = float(quote_vol)
            except (ValueError, TypeError):
                qv_float = 0.0

            filtered.append((symbol, qv_float))

        # Sort remaining pairs descending by ticker['quoteVolume'] (24-hour USDT turnover)
        filtered.sort(key=lambda item: item[1], reverse=True)
        universe = [symbol for symbol, _ in filtered]

        self._cached_universe = universe
        self._cache_timestamp = time.time()
        logger.info(
            "[UNIVERSE SCREENER] Dynamic universe updated: %d qualified spot pairs (top %d: %s)",
            len(universe),
            min(top_n, len(universe)),
            universe[:top_n],
        )

        return universe[:top_n]


# Default singleton screener instance
_default_screener = UniverseScreener()


async def get_dynamic_universe(
    exchange: Any,
    top_n: int = 15,
    mock_exchange: Optional[bool] = None,
    force_refresh: bool = False,
) -> List[str]:
    """Top-level convenience method matching specification."""
    return await _default_screener.get_dynamic_universe(
        exchange=exchange,
        top_n=top_n,
        mock_exchange=mock_exchange,
        force_refresh=force_refresh,
    )
