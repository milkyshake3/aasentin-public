"""News watcher module for Sentinel-Trader.

Ingests crypto news headlines from live RSS and simulated sources,
guaranteeing SHA-256 deduplication to prevent redundant LLM evaluations.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Deque, List, Optional, Set
import xml.etree.ElementTree as ET
import aiohttp
from pydantic import BaseModel, Field

from sentinel_trader.config import Config

logger = logging.getLogger("sentinel.news")


class NewsArticle(BaseModel):
    """Normalized crypto news article representation."""

    id: str = Field(description="SHA-256 hash of source and headline")
    headline: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    url: Optional[str] = None
    summary: Optional[str] = None


class NewsWatcher:
    """Async news aggregator with SHA-256 deduplication engine."""

    # Public Crypto RSS Feeds for live news
    RSS_FEEDS = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ]

    # Benign / Routine simulation headlines (safe for take_profit, stop_loss, market_walk)
    BENIGN_TEST_HEADLINES = [
        ("Solana ecosystem transaction volume steady following DEX integration", "SolanaStatus"),
        ("Routine network maintenance scheduled for devnet validator clusters", "NetworkHealth"),
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

    def __init__(self, config: Config, max_cache_size: int = 5000):
        self.config = config
        self.max_cache_size = max_cache_size
        self._seen_hashes: Set[str] = set()
        self._hash_fifo: Deque[str] = deque(maxlen=max_cache_size)
        self._injected_queue: asyncio.Queue[NewsArticle] = asyncio.Queue()
        self._benign_index = 0
        self._emergency_index = 0
        self._simulated_index = 0

    @staticmethod
    def compute_sha256(headline: str, source: str = "") -> str:
        """Compute normalized SHA-256 hash of headline and source."""
        # Normalize text: strip whitespace, lowercase, remove extra interior spaces
        clean_text = re.sub(r"\s+", " ", f"{source.strip().lower()}||{headline.strip().lower()}")
        return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

    def is_duplicate(self, article_hash: str) -> bool:
        """Check if an article hash has already been processed."""
        return article_hash in self._seen_hashes

    def record_hash(self, article_hash: str) -> None:
        """Record article hash to seen set with bounded FIFO memory safety."""
        if article_hash not in self._seen_hashes:
            if len(self._seen_hashes) >= self.max_cache_size:
                oldest = self._hash_fifo.popleft()
                self._seen_hashes.discard(oldest)
            self._seen_hashes.add(article_hash)
            self._hash_fifo.append(article_hash)

    @staticmethod
    def sanitize_headline(headline: str) -> str:
        """Sanitize raw headline input to neutralize adversarial prompt injection attempts."""
        if not headline:
            return ""
        import unicodedata
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

    def inject_headline(self, headline: str, source: str = "manual_inject") -> NewsArticle:
        """Inject a headline programmatically with adversarial sanitization."""
        clean_headline = self.sanitize_headline(headline)
        article_hash = self.compute_sha256(clean_headline, source)
        article = NewsArticle(
            id=article_hash,
            headline=clean_headline,
            source=source,
            timestamp=datetime.now(timezone.utc),
        )
        self._injected_queue.put_nowait(article)
        logger.info("[NEWS INJECTED] (%s): %s", source, clean_headline)
        return article

    async def fetch_rss_feed(self, session: aiohttp.ClientSession, source_name: str, url: str) -> List[NewsArticle]:
        """Fetch and parse an RSS feed with timeout protection and adversarial sanitization."""
        articles: List[NewsArticle] = []
        try:
            headers = {"User-Agent": "SentinelTrader/1.0 (Crypto Risk Mitigation Bot)"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=4.0)) as resp:
                if resp.status != 200:
                    logger.debug("RSS feed %s returned HTTP %s", source_name, resp.status)
                    return []

                content = await resp.text()
                # Parse RSS XML
                root = ET.fromstring(content)
                # Find all <item> elements across RSS 2.0 or Atom
                items = root.findall(".//item")
                for item in items[:10]:  # Inspect top 10 latest
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        raw_title = title_elem.text.strip()
                        clean_title = self.sanitize_headline(raw_title)
                        link_elem = item.find("link")
                        link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
                        desc_elem = item.find("description")
                        desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else None

                        article_hash = self.compute_sha256(clean_title, source_name)
                        articles.append(
                            NewsArticle(
                                id=article_hash,
                                headline=clean_title,
                                source=source_name,
                                url=link,
                                summary=desc,
                            )
                        )
        except Exception as e:
            logger.debug("Could not fetch RSS feed %s: %s", source_name, e)

        return articles

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
        return NewsArticle(
            id=article_hash,
            headline=headline,
            source=source,
            timestamp=datetime.now(timezone.utc),
        )

    def inject_emergency_headline(self) -> NewsArticle:
        """Explicitly inject an emergency zero-day headline (strictly for ai_panic scenario)."""
        headline, source = self.EMERGENCY_TEST_HEADLINES[
            self._emergency_index % len(self.EMERGENCY_TEST_HEADLINES)
        ]
        self._emergency_index += 1
        return self.inject_headline(headline, source=source)

    async def poll_new_headlines(self) -> List[NewsArticle]:
        """Poll all sources and return only fresh, non-duplicate articles."""
        fresh_articles: List[NewsArticle] = []

        # 1. Drain any manually injected headlines
        while not self._injected_queue.empty():
            article = self._injected_queue.get_nowait()
            if not self.is_duplicate(article.id):
                self.record_hash(article.id)
                fresh_articles.append(article)

        # 2. Poll live RSS feeds if enabled
        if self.config.enable_rss_news:
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = [
                        self.fetch_rss_feed(session, source_name, url)
                        for source_name, url in self.RSS_FEEDS
                    ]
                    feed_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in feed_results:
                        if isinstance(res, list):
                            for article in res:
                                if not self.is_duplicate(article.id):
                                    self.record_hash(article.id)
                                    fresh_articles.append(article)
            except Exception as e:
                logger.debug("Error during RSS polling: %s", e)

        # 3. If no fresh live news and simulated news enabled, yield one strictly benign headline
        if not fresh_articles and self.config.enable_simulated_news:
            simulated = self.get_next_simulated_headline(category="benign")
            if simulated and not self.is_duplicate(simulated.id):
                self.record_hash(simulated.id)
                fresh_articles.append(simulated)

        return fresh_articles
