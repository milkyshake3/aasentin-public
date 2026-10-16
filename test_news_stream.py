"""Phase 4 Verification Script: Live Real-Time News Ingestion & Filtering.

Verifies:
1. Live RSS Feeds connection and ingestion from CoinTelegraph, CoinDesk, and Decrypt.
2. In-memory sliding cache SHA-256 deduplication and subsequent poll cache hits.
3. Strict asset keyword filtering (SOL/Solana base asset vs macro systemic risks vs ignored altcoins).
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Auto-detect and bootstrap local virtual environment if run from global python
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    candidates = [
        os.path.join(root_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(root_dir, ".venv", "bin", "python"),
    ]
    for venv_py in candidates:
        if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_py).lower():
            import subprocess
            cmd = [venv_py, os.path.abspath(__file__)] + sys.argv[1:]
            sys.exit(subprocess.call(cmd, cwd=root_dir))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sentinel_trader.config import Config
from sentinel_trader.news_watcher import NewsWatcher, NewsArticle


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)


async def main() -> int:
    print_header("PHASE 4: LIVE REAL-TIME NEWS INGESTION VERIFICATION")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(" Target Asset: SOL (SOL/USDT)")
    print(" Feeds: CoinTelegraph, CoinDesk, Decrypt")

    config = Config(
        target_symbol="SOL/USDT",
        enable_rss_news=True,
        enable_simulated_news=False,
        max_news_age_minutes=1440,
    )
    news_watcher = NewsWatcher(config, max_cache_size=1000)

    # ==================================================================
    # [CHECK A] Connect to RSS Feeds & Ingest Live Headlines
    # ==================================================================
    print_header("[CHECK A] LIVE RSS FEEDS INGESTION & PARSING")
    print(" Polling live crypto wires without filter to inspect raw stream...")

    raw_articles = await news_watcher.poll_new_headlines(filter_keywords=False)
    print(f" Ingested {len(raw_articles)} unique headlines across feeds.")

    if len(raw_articles) < 3:
        print(f" [FAIL] Expected at least 3 live articles, got {len(raw_articles)}")
        return 1

    print("\n 3 Most Recent Live Headlines Ingested:")
    for idx, art in enumerate(raw_articles[:3], 1):
        print(f"\n   [{idx}] Source   : {art.source}")
        print(f"       Headline : {art.headline}")
        print(f"       SHA-256  : {art.id}")
        if art.summary:
            clean_sum = art.summary[:100].replace("\n", " ") + "..."
            print(f"       Summary  : {clean_sum}")

    print("\n [OK] Check A Passed: Successfully connected to RSS feeds and ingested live headlines.")

    # ==================================================================
    # [CHECK B] SHA-256 Deduplication & Cache Hit Verification
    # ==================================================================
    print_header("[CHECK B] SHA-256 DEDUPLICATION & CACHE HIT VERIFICATION")

    initial_cache_size = news_watcher.seen_count
    print(f" Current in-memory cache size: {initial_cache_size} hashes (Capacity: 1,000)")

    # Test B.1: Subsequent poll of the same feeds
    print("\n Polling live feeds a second time immediately...")
    second_poll = await news_watcher.poll_new_headlines(filter_keywords=False)
    print(f" Second poll returned: {len(second_poll)} new articles")
    if len(second_poll) != 0:
        print(f" [FAIL] Deduplication failed: {len(second_poll)} duplicate articles were returned!")
        return 1
    print(" [PASS] 100% Cache Hit: Zero duplicates returned on immediate subsequent poll.")

    # Test B.2: Cross-feed syndication deduplication
    print("\n Testing cross-feed syndicated headline deduplication...")
    headline_wire = "SEC files emergency injunction against crypto foundation"
    hash_coindesk = news_watcher.compute_sha256(headline_wire, "CoinDesk")
    hash_decrypt = news_watcher.compute_sha256(headline_wire, "Decrypt")
    hash_cased = news_watcher.compute_sha256(f"  {headline_wire.upper()}  ", "CoinTelegraph")

    assert hash_coindesk == hash_decrypt == hash_cased, "Cross-feed hashes must be identical!"
    print(f"   Normalized SHA-256: {hash_coindesk[:16]}... (Identical across feeds)")
    print(" [PASS] Cross-feed syndicated headlines produce matching hashes.")

    # Test B.3: Sliding FIFO Cache Size Limit (Eviction at 1000 items)
    print("\n Testing sliding cache FIFO bounded capacity (1,000 max)...")
    mini_watcher = NewsWatcher(config, max_cache_size=10)
    for i in range(15):
        h = f"Test Headline Item #{i}"
        mini_watcher.record_hash(mini_watcher.compute_sha256(h))

    assert mini_watcher.seen_count == 10, f"Expected 10 cached hashes, got {mini_watcher.seen_count}"
    # Verify oldest hash (0) was evicted and newest hash (14) is present
    oldest_hash = mini_watcher.compute_sha256("Test Headline Item #0")
    newest_hash = mini_watcher.compute_sha256("Test Headline Item #14")
    assert not mini_watcher.is_duplicate(oldest_hash), "Oldest hash should have been evicted!"
    assert mini_watcher.is_duplicate(newest_hash), "Newest hash should be cached!"
    print(f" [PASS] Sliding cache strictly bounded: {mini_watcher.seen_count}/10 with oldest evicted.")

    print("\n [OK] Check B Passed: SHA-256 deduplication and cache hits verified.")

    # ==================================================================
    # [CHECK C] Asset Keyword Filter Verification
    # ==================================================================
    print_header("[CHECK C] ASSET KEYWORD & MACRO-SYSTEMIC RISK FILTER")

    # Test C.1: Evaluate all ingested live headlines
    passed_live = []
    ignored_live = []
    for art in raw_articles:
        match_type = news_watcher.get_headline_filter_match(art.headline)
        if match_type:
            passed_live.append((art, match_type))
        else:
            ignored_live.append(art)

    print(f" Evaluated {len(raw_articles)} live headlines against asset/macro filter:")
    print(f"   - Passed Filter  : {len(passed_live)} headlines")
    print(f"   - Filtered Out   : {len(ignored_live)} headlines (irrelevant altcoins / noise)")

    if passed_live:
        print("\n Live Headlines that PASSED Filter:")
        for art, match in passed_live[:5]:
            print(f"   [+] [{match}] ({art.source}) {art.headline}")
    else:
        print("\n (No live headlines currently mention SOL or macro emergencies in top wire batches)")

    print("\n Sample Live Headlines that were FILTERED OUT (Altcoin Isolation):")
    for art in ignored_live[:4]:
        print(f"   [-] [IGNORED] ({art.source}) {art.headline}")

    # Test C.2: Rigorous Deterministic Control Test Matrix
    print("\n Running Deterministic Filter Test Matrix...")
    control_matrix = [
        ("Solana ecosystem transaction volume steady following DEX integration", True, "target_asset_SOL"),
        ("SOL drops 4% in hourly market retrace", True, "target_asset_SOL"),
        ("SEC files emergency freeze against unregistered crypto fund", True, "macro_risk_sec_emergency"),
        ("Binance halt on deposits sparks volatility across derivatives", True, "macro_risk_binance_halt"),
        ("Government issues blanket crypto ban in national decree", True, "macro_risk_crypto_ban"),
        ("Critical zero-day vulnerability detected in core validator runtime", True, "macro_risk_zero_day_exploit"),
        ("Major bridge drained in critical exploit of multi-sig parameters", True, "macro_risk_critical_exploit"),
        ("Cardano developers prepare for Hydra node upgrade", False, None),
        ("Dogecoin reaches new weekly high following meme rally", False, None),
        ("Insolvent crypto lender completes asset recovery procedure", False, None),  # Word boundary test
        ("Consolidated quarterly earnings report released by mining group", False, None),  # Word boundary test
    ]

    for headline, should_pass, expected_match in control_matrix:
        match_reason = news_watcher.get_headline_filter_match(headline)
        passes = match_reason is not None
        assert passes == should_pass, f"Filter mismatch on '{headline}': got {passes}, expected {should_pass}"
        if should_pass and expected_match:
            assert match_reason == expected_match, f"Expected {expected_match}, got {match_reason}"
        status_tag = f"[PASS: {match_reason}]" if passes else "[BLOCKED: Altcoin/Noise]"
        print(f"   {status_tag:<32} '{headline[:50]}...'")

    print("\n [OK] Check C Passed: Asset keyword filtering and altcoin isolation verified.")

    # ==================================================================
    # [CHECK D] Live Filtered Stream Run
    # ==================================================================
    print_header("[CHECK D] LIVE FILTERED STREAM RUN")
    fresh_watcher = NewsWatcher(config, max_cache_size=1000)
    filtered_articles = await fresh_watcher.poll_new_headlines(filter_keywords=True)
    print(f" Filtered live stream returned {len(filtered_articles)} actionable headlines for LLM evaluation.")
    for art in filtered_articles:
        assert art.matched_filter is not None
        print(f"   * [{art.matched_filter}] ({art.source}) {art.headline}")

    print("\n [OK] Check D Passed: Live filtered stream yielded only relevant articles.")

    # ==================================================================
    # VERIFICATION SUMMARY
    # ==================================================================
    print_header("PHASE 4 VERIFICATION SUMMARY")
    print(" Check A (Live RSS Feeds Ingestion):         PASSED [OK]")
    print(" Check B (SHA-256 Deduplication Cache Hits): PASSED [OK]")
    print(" Check C (Asset Keyword & Macro Filtering):  PASSED [OK]")
    print(" Check D (Live Filtered Stream Execution):   PASSED [OK]")
    print("=" * 72)
    print("\n [SUCCESS] ALL PHASE 4 VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
