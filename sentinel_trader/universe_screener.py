"""Dynamic 24-Hour Volume Universe Screener for Sentinel-Trader.

Fetches 24-hour ticker volume from Binance, filters out inactive markets, non-USDT pairs,
stablecoins, fiat, and leveraged tokens, and returns the top liquid candidates sorted
descending by 24-hour USDT turnover (quoteVolume).
Includes an in-memory 24-hour cache (TTL 86,400s) and liquid mock exchange fallback.
"""

import asyncio
from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sentinel.screener")

# In-memory asset reliability scores dictionary
asset_scores: Dict[str, Dict[str, Any]] = {}

STABLECOIN_BLACKLIST: Set[str] = {
    "USDC/USDT",
    "FDUSD/USDT",
    "TUSD/USDT",
    "BUSD/USDT",
    "EUR/USDT",
    "DAI/USDT",
    "USDP/USDT",
    "AEUR/USDT",
    "USD1/USDT",
    "USDE/USDT",
    "PYUSD/USDT",
    "BFUSD/USDT",
    "USDD/USDT",
}

# Safety Regex: match any base asset containing "USD" (case-insensitive)
USD_BASE_REGEX = re.compile(r"USD", re.IGNORECASE)

# Module-level set of explicitly whitelisted USD base assets or symbols
WHITELISTED_USD_BASE_ASSETS: Set[str] = set()

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
    """Dynamic universe screener maintaining a 24-hour in-memory cache and reliability penalty box."""

    def __init__(
        self,
        cache_ttl_seconds: int = 86400,
        whitelisted_symbols: Optional[Set[str]] = None,
        asset_scores: Optional[Dict[str, Dict[str, Any]]] = None,
        asset_scores_file: Optional[str] = None,
    ):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.whitelisted_symbols: Set[str] = set(whitelisted_symbols or set())
        self._cached_universe: Optional[List[str]] = None
        self._cache_timestamp: Optional[float] = None
        self.asset_scores: Dict[str, Dict[str, Any]] = asset_scores if asset_scores is not None else {}
        self.asset_scores_file: Optional[str] = asset_scores_file

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

    def get_asset_score(
        self,
        symbol: str,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve score entry for an asset, automatically reconciling expired bans."""
        now_ts = current_time if current_time is not None else time.time()
        if symbol not in self.asset_scores:
            from sentinel_trader.state_store import load_asset_scores
            disk_scores = load_asset_scores(file_path=self.asset_scores_file, current_time=now_ts)
            if symbol in disk_scores:
                self.asset_scores[symbol] = disk_scores[symbol]
            else:
                return {"score": 1.0, "penalized_until": None, "trades_count": 0}

        entry = self.asset_scores[symbol]
        if entry.get("penalized_until") is not None and now_ts >= entry["penalized_until"]:
            entry["penalized_until"] = None
            entry["score"] = 0.5
            logger.info("[PENALTY BOX] %s reinstated with baseline score 0.5", symbol)
        return entry

    def is_asset_penalized(
        self,
        symbol: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """Check if an asset is actively excluded under the 24-hour penalty box."""
        now_ts = current_time if current_time is not None else time.time()
        entry = self.get_asset_score(symbol, current_time=now_ts)
        pen_until = entry.get("penalized_until")
        return pen_until is not None and now_ts < pen_until

    async def get_dynamic_universe(
        self,
        exchange: Any,
        top_n: int = 15,
        mock_exchange: Optional[bool] = None,
        force_refresh: bool = False,
        whitelisted_symbols: Optional[Set[str]] = None,
        asset_scores: Optional[Dict[str, Dict[str, Any]]] = None,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """Fetch all 24h tickers in a single call and return top_n pairs by quoteVolume.

        Filters:
        - Must be an active spot market (market['spot'] is True and market['active'] is True).
        - Symbol must settle in USDT (symbol.endswith('/USDT')).
        - Exclude Blacklisted Stablecoins & Fiat (STABLECOIN_BLACKLIST).
        - Safety Regex Filter: Exclude symbols containing 'USD' in base asset unless whitelisted.
        - Exclude leveraged tokens (symbols containing UP/, DOWN/, BULL/, BEAR/).
        - Penalty Box Exclusion Filter: Exclude pairs where time.time() < penalized_until.
        - Sort remaining pairs descending by ticker['quoteVolume'] (24-hour USDT turnover).
        - Cache result for 24 hours unless force_refresh is True.
        """
        now_ts = current_time if current_time is not None else time.time()

        # Resolve active whitelist for USD base asset safety filter
        active_whitelist: Set[str] = set(self.whitelisted_symbols)
        if whitelisted_symbols:
            active_whitelist.update(whitelisted_symbols)
        active_whitelist.update(WHITELISTED_USD_BASE_ASSETS)

        # Resolve active reliability scores
        if asset_scores is not None:
            active_scores = asset_scores
        elif self.asset_scores:
            active_scores = self.asset_scores
        elif self.asset_scores_file:
            from sentinel_trader.state_store import load_asset_scores
            active_scores = load_asset_scores(file_path=self.asset_scores_file, current_time=now_ts)
            self.asset_scores = active_scores
        else:
            active_scores = {}

        # Ensure expired entries in active_scores are automatically reinstated
        for s, entry in list(active_scores.items()):
            if isinstance(entry, dict) and entry.get("penalized_until") is not None:
                if now_ts >= entry["penalized_until"]:
                    entry["penalized_until"] = None
                    entry["score"] = 0.5
                    logger.info("[PENALTY BOX] %s reinstated with baseline score 0.5", s)

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
            filtered_mock = []
            for s in DEFAULT_MOCK_UNIVERSE:
                entry = active_scores.get(s)
                if entry and entry.get("penalized_until") is not None and now_ts < entry["penalized_until"]:
                    iso_time = datetime.fromtimestamp(entry["penalized_until"], tz=timezone.utc).isoformat()
                    score_val = float(entry.get("score", 0.0))
                    logger.warning("[PENALTY BOX] %s excluded (banned until %s, Score: %.1f)", s, iso_time, score_val)
                    continue
                filtered_mock.append(s)
            return filtered_mock[:top_n]

        # 2. In-memory 24-Hour Cache evaluation
        if not force_refresh and self.is_cache_valid():
            logger.debug(
                "[UNIVERSE SCREENER] Cache hit: Returning cached universe (age: %.1fs)",
                time.time() - (self._cache_timestamp or 0),
            )
            filtered_cache = []
            for s in (self._cached_universe or []):
                entry = active_scores.get(s)
                if entry and entry.get("penalized_until") is not None and now_ts < entry["penalized_until"]:
                    iso_time = datetime.fromtimestamp(entry["penalized_until"], tz=timezone.utc).isoformat()
                    score_val = float(entry.get("score", 0.0))
                    logger.warning("[PENALTY BOX] %s excluded (banned until %s, Score: %.1f)", s, iso_time, score_val)
                    continue
                filtered_cache.append(s)
            return filtered_cache[:top_n]

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

            # Safety Regex Filter: Ignore any symbol containing USD in base asset unless whitelisted
            base_asset = symbol.split("/")[0] if "/" in symbol else symbol
            if USD_BASE_REGEX.search(base_asset):
                if symbol not in active_whitelist and base_asset not in active_whitelist:
                    logger.debug(
                        "[UNIVERSE SCREENER] Filtered out %s: base asset '%s' contains 'USD' and is not whitelisted",
                        symbol,
                        base_asset,
                    )
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

            # Penalty Box Exclusion Filter: Exclude any pair where time.time() < penalized_until
            score_entry = active_scores.get(symbol)
            if score_entry and score_entry.get("penalized_until") is not None:
                pen_until = score_entry["penalized_until"]
                if now_ts < pen_until:
                    iso_time = datetime.fromtimestamp(pen_until, tz=timezone.utc).isoformat()
                    score_val = float(score_entry.get("score", 0.0))
                    logger.warning(
                        "[PENALTY BOX] %s excluded (banned until %s, Score: %.1f)",
                        symbol,
                        iso_time,
                        score_val,
                    )
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
        self._cache_timestamp = now_ts
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
    whitelisted_symbols: Optional[Set[str]] = None,
    asset_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    current_time: Optional[float] = None,
) -> List[str]:
    """Top-level convenience method matching specification."""
    return await _default_screener.get_dynamic_universe(
        exchange=exchange,
        top_n=top_n,
        mock_exchange=mock_exchange,
        force_refresh=force_refresh,
        whitelisted_symbols=whitelisted_symbols,
        asset_scores=asset_scores,
        current_time=current_time,
    )


# Expose scoring functions from analytics for convenience
from sentinel_trader.analytics import (
    calculate_score_adjustment,
    get_all_asset_scores,
    get_asset_score,
    update_asset_score,
)

