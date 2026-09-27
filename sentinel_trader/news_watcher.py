"""News watcher module for Sentinel-Trader.

Ingests crypto news headlines from live RSS feeds and simulated sources,
guaranteeing strict SHA-256 deduplication and asset/macro keyword filtering
before forwarding to AI sentiment evaluation.
"""

import asyncio
import calendar
from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
import re
import time
from typing import Deque, List, Optional, Set, Tuple
import unicodedata
import xml.etree.ElementTree as ET
import aiohttp
from pydantic import BaseModel, Field

try:
    import feedparser
except ImportError:
    feedparser = None

from sentinel_trader.config import Config

logger = logging.getLogger("sentinel.news")


class NewsArticle(BaseModel):
    """Normalized crypto news article representation."""

    id: str = Field(description="SHA-256 hash of normalized headline")
    headline: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    url: Optional[str] = None
    summary: Optional[str] = None
    matched_filter: Optional[str] = Field(
        default=None,
        description="Filter pattern that matched: target_asset or macro_risk",
    )


class NewsWatcher:
    """Async news aggregator with SHA-256 deduplication and asset/macro filtering."""

    # Public Crypto RSS Feeds for live breaking news
    RSS_FEEDS = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Decrypt", "https://decrypt.co/feed"),
    ]

    # Benign / Routine simulation headlines (safe for take_profit, stop_loss, market_walk)
    BENIGN_TEST_HEADLINES = [
        ("Solana ecosystem transaction volume steady following DEX integration", "SolanaStatus"),
        ("Routine network maintenance scheduled for Solana devnet validator clusters", "NetworkHealth"),
        ("Market Update: Crypto assets hold steady across major spot pairs", "MarketDesk"),
        ("Developer activity on Solana reaches new quarterly high", "EcosystemNews"),
        ("Institutional custody provider announces expanded staking support", "CustodyWire"),
        ("Cross-chain bridging protocol upgrades security multi-sig parameters", "DeFiStandard"),
        ("Federal Reserve leaves interest rates unchanged in unanimous decision", "MacroNews"),
    ]

    # Catastrophic emergency simulation headlines (strictly for ai_panic testing)
    EMERGENCY_TEST_HEADLINES = [
        ("SOLANA CORE EXPLOIT: Critical zero-day vulnerability detected in Solana validator runtime, funds frozen", "EmergencyAlert"),
        ("BREAKING: SEC files emergency injunction against Solana Foundation, orders immediate asset seizure", "CryptoWire"),
        ("MAJOR HACK: Largest Solana cross-chain bridge drained of $320M, attackers dumping SOL on open market", "DeFiWatch"),
    ]

    # Backward compatibility alias
    SIMULATED_TEST_HEADLINES = BENIGN_TEST_HEADLINES

    # Default sliding cache capacity for SHA-256 deduplication
    DEFAULT_CACHE_SIZE = 1000

    def __init__(self, config: Config, max_cache_size: int = DEFAULT_CACHE_SIZE):
        self.config = config
        self.max_cache_size = max_cache_size
        self._seen_hashes: Set[str] = set()
        self._hash_fifo: Deque[str] = deque(maxlen=max_cache_size)
        self._injected_queue: asyncio.Queue[NewsArticle] = asyncio.Queue()
        self._benign_index = 0
        self._emergency_index = 0
        self._simulated_index = 0
        self.rss_feeds = list(self.RSS_FEEDS)

    @property
    def seen_count(self) -> int:
        """Return the current number of cached SHA-256 hashes."""
        return len(self._seen_hashes)

    @staticmethod
    def normalize_headline(headline: str) -> str:
        """Normalize headline text for strict cross-feed SHA-256 deduplication."""
        if not headline:
            return ""
        # Unicode decomposition for smart quotes, em-dashes, and special characters
        text = unicodedata.normalize("NFKD", str(headline))
        # Standardize typographic variations
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        text = text.replace("—", "-").replace("–", "-")
        # Lowercase and strip leading/trailing whitespace
        text = text.lower().strip()
        # Collapse multiple whitespace characters into single space
        text = re.sub(r"\s+", " ", text)
        # Strip trailing punctuation (periods, colons, dashes) that differ across syndicated wires
        text = re.sub(r"[\s\.\,\;\:\-\!\?]+$", "", text)
        return text

    @classmethod
    def compute_sha256(cls, headline: str, source: str = "") -> str:
        """Compute normalized SHA-256 hash of headline for cross-feed deduplication.

        The source parameter is retained for backwards compatibility, but identical
        or syndicated headlines across multiple feeds produce identical hashes.
        """
        clean_text = cls.normalize_headline(headline)
        return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

    def is_duplicate(self, article_hash: str) -> bool:
        """Check if an article hash has already been processed."""
        return article_hash in self._seen_hashes

    def record_hash(self, article_hash: str) -> None:
        """Record article hash to seen set with bounded FIFO sliding cache (up to max_cache_size)."""
        if article_hash not in self._seen_hashes:
            if len(self._hash_fifo) >= self.max_cache_size:
                oldest = self._hash_fifo.popleft()
                self._seen_hashes.discard(oldest)
            self._seen_hashes.add(article_hash)
            self._hash_fifo.append(article_hash)

    @staticmethod
    def sanitize_headline(headline: str) -> str:
        """Sanitize raw headline input to neutralize adversarial prompt injection attempts."""
        if not headline:
            return ""
        normalized = unicodedata.normalize("NFKD", str(headline))
        # Strip control characters, zero-width spaces, non-printable unicode
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", " ", normalized)

        # Strip common adversarial command injection patterns & instruction overrides
        injection_patterns = [
            r"(?i)\b(?:system|developer|admin)\s*(?:instruction|directive|prompt|override)\s*:\s*",
            r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
            r"(?i)\bdo\s+not\s+sell\b",
            r"(?i)\boutput\s+hold\b",
            r"(?i)\[/?(?:inst|system)\]",
            r"(?i)<\|im_(?:start|end)\|>",
            r"(?i)</?(?:instructions?|system|context|text)>",
        ]
        for pattern in injection_patterns:
            cleaned = re.sub(pattern, " ", cleaned)

        # Collapse multiple whitespaces and truncate to 500 characters
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:500]

    def get_headline_filter_match(self, headline: str, target_symbol: Optional[str] = None) -> Optional[str]:
        """Evaluate if a headline matches the target base asset or broad macro-systemic risks.

        Returns:
            Optional[str]: Description of match (e.g. 'target_asset_SOL' or 'macro_risk_sec_emergency')
                           or None if the headline is irrelevant altcoin noise.
        """
        if not headline:
            return None

        symbol = target_symbol or self.config.target_symbol
        base_asset = symbol.split("/")[0].strip().upper() if "/" in symbol else symbol.strip().upper()

        # 1. Base Asset Synonyms & Nicknames (with strict word boundaries)
        asset_synonyms = {
            "SOL": ["sol", "solana"],
            "BTC": ["btc", "bitcoin"],
            "ETH": ["eth", "ethereum", "ether"],
            "BNB": ["bnb", "binance coin"],
            "XRP": ["xrp", "ripple"],
            "ADA": ["ada", "cardano"],
            "DOGE": ["doge", "dogecoin"],
            "AVAX": ["avax", "avalanche"],
        }
        keywords = asset_synonyms.get(base_asset, [base_asset.lower()])

        # Word boundary match prevents false positives (e.g., 'insolvent', 'consolidation' != 'sol')
        asset_regex = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
        if asset_regex.search(headline):
            return f"target_asset_{base_asset}"

        # 2. Broad Macro-Systemic Risk Patterns
        macro_patterns: List[Tuple[str, str]] = [
            (r"\bcrypto\s+(?:market\s+)?ban\b", "crypto_ban"),
            (r"\bblanket\s+ban\b", "blanket_ban"),
            (r"\bsec\s+(?:files\s+|issues\s+|orders\s+|seeks\s+)?(?:emergency(?:\s+freeze)?|freeze|injunction|halts?|lawsuit|enforcement)\b", "sec_emergency"),
            (r"\bbinance\s+(?:halts?|freeze|pause|suspension|suspended|down|hack|insolvent|insolvency)\b", "binance_halt"),
            (r"\bexchange\s+(?:halts?|freeze|suspension|suspended|insolvency)\b", "exchange_halt"),
            (r"\bemergency\s+freeze\b", "emergency_freeze"),
            (r"\bzero-?day\b", "zero_day_exploit"),
            (r"\bcritical\s+exploit\b", "critical_exploit"),
            (r"\bsystemic\s+crash\b", "systemic_crash"),
            (r"\b(?:stablecoin|tether|usdt|usdc)\s+depeg\b", "stablecoin_depeg"),
            (r"\bliquidity\s+crisis\b", "liquidity_crisis"),
            (r"\bmarket\s+contagion\b", "market_contagion"),
            (r"\b(?:fbi|doj)\s+seizure\b", "doj_fbi_seizure"),
        ]

        for pattern, reason in macro_patterns:
            if re.search(pattern, headline, re.IGNORECASE):
                return f"macro_risk_{reason}"

        return None

    def is_relevant_headline(self, headline: str, target_symbol: Optional[str] = None) -> bool:
        """Boolean check whether headline passes asset or macro-risk filter."""
        return self.get_headline_filter_match(headline, target_symbol) is not None

    def inject_headline(
        self,
        headline: str,
        source: str = "manual_inject",
        timestamp: Optional[datetime] = None,
    ) -> NewsArticle:
        """Inject a headline programmatically with adversarial sanitization."""
        clean_headline = self.sanitize_headline(headline)
        article_hash = self.compute_sha256(clean_headline, source)
        match_reason = self.get_headline_filter_match(clean_headline)
        article = NewsArticle(
            id=article_hash,
            headline=clean_headline,
            source=source,
            timestamp=timestamp or datetime.now(timezone.utc),
            matched_filter=match_reason or "manual_injection",
        )
        self._injected_queue.put_nowait(article)
        logger.info("[NEWS INJECTED] (%s): %s", source, clean_headline)
        return article

    async def fetch_rss_feed(
        self,
        session: aiohttp.ClientSession,
        source_name: str,
        url: str,
    ) -> List[NewsArticle]:
        """Fetch and parse an RSS feed with timeout protection, freshness check, and sanitization."""
        articles: List[NewsArticle] = []
        try:
            headers = {"User-Agent": "SentinelTrader/1.0 (Crypto Risk Mitigation Bot)"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status != 200:
                    logger.debug("RSS feed %s returned HTTP %s", source_name, resp.status)
                    return []

                content = await resp.text()

                # Primary parser: feedparser (handles RSS 2.0, Atom, RDF, and XML)
                if feedparser is not None:
                    parsed = feedparser.parse(content)
                    for entry in parsed.entries[:25]:
                        raw_title = entry.get("title") if hasattr(entry, "get") else getattr(entry, "title", None)
                        if not raw_title:
                            continue
                        clean_title = self.sanitize_headline(raw_title)
                        link = entry.get("link") if hasattr(entry, "get") else getattr(entry, "link", None)
                        raw_desc = (
                            (entry.get("summary") or entry.get("description"))
                            if hasattr(entry, "get")
                            else (getattr(entry, "summary", None) or getattr(entry, "description", None))
                        )
                        clean_desc = None
                        if raw_desc:
                            # Strip raw HTML tags from feed summary
                            clean_desc = re.sub(r"<[^>]+>", " ", str(raw_desc)).strip()[:500]

                        article_hash = self.compute_sha256(clean_title, source_name)

                        # Freshness validation cutoff (MAX_NEWS_AGE_MINUTES)
                        published_struct = None
                        if hasattr(entry, "get"):
                            published_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                        if not published_struct:
                            published_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)

                        article_ts = datetime.now(timezone.utc)
                        if published_struct:
                            try:
                                published_epoch = calendar.timegm(published_struct)
                                age_seconds = time.time() - published_epoch
                                if age_seconds > (self.config.max_news_age_minutes * 60):
                                    # If article is older than MAX_NEWS_AGE_MINUTES, hash into deduplication cache
                                    # so it is never re-evaluated, but DO NOT yield or return to AI Sentinel.
                                    self.record_hash(article_hash)
                                    logger.debug(
                                        "[FILTER EXPIRED] %s: '%s' published %.1fm ago exceeds max_news_age_minutes (%dm)",
                                        source_name,
                                        clean_title,
                                        age_seconds / 60.0,
                                        self.config.max_news_age_minutes,
                                    )
                                    continue
                                article_ts = datetime.fromtimestamp(published_epoch, tz=timezone.utc)
                            except Exception as e:
                                logger.debug("Failed parsing publication date for '%s': %s", clean_title, e)

                        articles.append(
                            NewsArticle(
                                id=article_hash,
                                headline=clean_title,
                                source=source_name,
                                timestamp=article_ts,
                                url=link,
                                summary=clean_desc,
                            )
                        )
                else:
                    # Fallback parser: XML ElementTree
                    root = ET.fromstring(content)
                    items = root.findall(".//item")
                    for item in items[:25]:
                        title_elem = item.find("title")
                        if title_elem is not None and title_elem.text:
                            raw_title = title_elem.text.strip()
                            clean_title = self.sanitize_headline(raw_title)
                            link_elem = item.find("link")
                            link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
                            desc_elem = item.find("description")
                            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else None

                            article_hash = self.compute_sha256(clean_title, source_name)
                            article_ts = datetime.now(timezone.utc)

                            pub_elem = item.find("pubDate")
                            if pub_elem is None:
                                pub_elem = item.find("{http://purl.org/dc/elements/1.1/}date")
                            if pub_elem is not None and pub_elem.text:
                                try:
                                    import email.utils
                                    parsed_tuple = email.utils.parsedate(pub_elem.text.strip())
                                    if parsed_tuple:
                                        pub_epoch = calendar.timegm(parsed_tuple)
                                        age_sec = time.time() - pub_epoch
                                        if age_sec > (self.config.max_news_age_minutes * 60):
                                            self.record_hash(article_hash)
                                            continue
                                        article_ts = datetime.fromtimestamp(pub_epoch, tz=timezone.utc)
                                except Exception:
                                    pass

                            articles.append(
                                NewsArticle(
                                    id=article_hash,
                                    headline=clean_title,
                                    source=source_name,
                                    timestamp=article_ts,
                                    url=link,
                                    summary=desc,
                                )
                            )
        except Exception as e:
            logger.debug("Could not fetch RSS feed %s: %s", source_name, e)

        return articles

    async def seed_cache_on_startup(self) -> int:
        """Seed the deduplication cache on cold boot by polling all active RSS feeds once.

        Hashes every existing headline into the FIFO sliding cache without routing any
        articles to AI Sentinel. Only fresh articles published after startup will be evaluated.

        Returns:
            int: Number of headlines seeded into the cache.
        """
        if not self.config.enable_rss_news:
            logger.info("[NEWS WARMUP] RSS news disabled; seeded cache with 0 existing headlines.")
            return 0

        initial_count = self.seen_count

        try:
            headers = {"User-Agent": "SentinelTrader/1.0 (Crypto Risk Mitigation Bot)"}
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                tasks = [
                    self.fetch_rss_feed(session, source_name, url)
                    for source_name, url in self.rss_feeds
                ]
                feed_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in feed_results:
                    if isinstance(res, list):
                        for article in res:
                            self.record_hash(article.id)
        except Exception as e:
            logger.error("Error during startup cache seeding: %s", e)

        seeded_count = self.seen_count - initial_count
        logger.info(
            "[NEWS WARMUP] Seeded cache with %d existing headlines. Only fresh articles published after startup will be evaluated.",
            seeded_count,
        )
        return seeded_count

    def get_next_simulated_headline(self, category: str = "benign") -> Optional[NewsArticle]:
        """Yield the next simulated headline (strictly benign by default)."""
        if not self.config.enable_simulated_news:
            return None

        if category == "emergency":
            headline, source = self.EMERGENCY_TEST_HEADLINES[
                self._emergency_index % len(self.EMERGENCY_TEST_HEADLINES)
            ]
            self._emergency_index += 1
        else:
            headline, source = self.BENIGN_TEST_HEADLINES[
                self._benign_index % len(self.BENIGN_TEST_HEADLINES)
            ]
            self._benign_index += 1

        article_hash = self.compute_sha256(headline, source)
        match_reason = self.get_headline_filter_match(headline)
        return NewsArticle(
            id=article_hash,
            headline=headline,
            source=source,
            timestamp=datetime.now(timezone.utc),
            matched_filter=match_reason or "simulated",
        )

    def inject_emergency_headline(self) -> NewsArticle:
        """Explicitly inject an emergency zero-day headline (strictly for ai_panic scenario)."""
        headline, source = self.EMERGENCY_TEST_HEADLINES[
            self._emergency_index % len(self.EMERGENCY_TEST_HEADLINES)
        ]
        self._emergency_index += 1
        return self.inject_headline(headline, source=source)

    async def poll_new_headlines(self, filter_keywords: bool = True) -> List[NewsArticle]:
        """Poll all sources and return only fresh, non-duplicate articles that match asset/macro filters.

        Args:
            filter_keywords: If True, only returns headlines mentioning base asset or macro risks.
                            If False, returns all deduplicated headlines (useful for stream verification).
        """
        fresh_articles: List[NewsArticle] = []

        # 1. Drain any manually injected headlines (e.g. from scenarios or unit tests)
        while not self._injected_queue.empty():
            article = self._injected_queue.get_nowait()
            if not self.is_duplicate(article.id):
                age_seconds = (datetime.now(timezone.utc) - article.timestamp).total_seconds()
                if age_seconds > (self.config.max_news_age_minutes * 60):
                    self.record_hash(article.id)
                    logger.debug(
                        "[FILTER EXPIRED] Injected '%s' is %.1fm old (> %dm cutoff)",
                        article.headline,
                        age_seconds / 60.0,
                        self.config.max_news_age_minutes,
                    )
                    continue
                self.record_hash(article.id)
                fresh_articles.append(article)

        # 2. Poll live RSS feeds if enabled
        if self.config.enable_rss_news:
            try:
                headers = {"User-Agent": "SentinelTrader/1.0 (Crypto Risk Mitigation Bot)"}
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    tasks = [
                        self.fetch_rss_feed(session, source_name, url)
                        for source_name, url in self.rss_feeds
                    ]
                    feed_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in feed_results:
                        if isinstance(res, list):
                            for article in res:
                                # Deduplication check: Has this headline already been evaluated?
                                if self.is_duplicate(article.id):
                                    continue

                                # Freshness cutoff check
                                age_seconds = (datetime.now(timezone.utc) - article.timestamp).total_seconds()
                                if age_seconds > (self.config.max_news_age_minutes * 60):
                                    self.record_hash(article.id)
                                    continue

                                # Record hash immediately to avoid re-evaluating across subsequent polls
                                self.record_hash(article.id)

                                # Keyword filtering: Check for target base asset or broad macro risks
                                match_reason = self.get_headline_filter_match(article.headline)
                                article.matched_filter = match_reason

                                if filter_keywords and not match_reason:
                                    logger.debug(
                                        "[FILTER IGNORED] %s: '%s' ignored (not relevant to %s or macro risks)",
                                        article.source,
                                        article.headline,
                                        self.config.base_asset,
                                    )
                                    continue

                                fresh_articles.append(article)
            except Exception as e:
                logger.error("Error during RSS polling: %s", e)

        # 3. If no fresh live news and simulated news enabled (strictly for mock simulation runs)
        if not fresh_articles and self.config.enable_simulated_news and not self.config.enable_rss_news:
            simulated = self.get_next_simulated_headline(category="benign")
            if simulated and not self.is_duplicate(simulated.id):
                self.record_hash(simulated.id)
                fresh_articles.append(simulated)

        return fresh_articles
