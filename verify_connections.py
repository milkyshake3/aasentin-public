"""Phase 3 Verification Script: Live Binance Testnet & Real LLM Integration.

Executes and verifies:
- Check A: Binance Testnet spot connectivity, account balance retrieval (USDT & SOL),
  and live SOL/USDT market ticker query.
- Check B: Live LLM emergency inference (Google GenAI) with structured JSON output,
  verifying that a catastrophic exploit headline triggers action="PANIC_SELL".
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console encoding for terminal symbols
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure virtualenv execution if invoked directly with global python
_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    cmd = [str(_VENV_PYTHON), "-m", "sentinel_trader.verify_connections"] + sys.argv[1:]
    import subprocess
    sys.exit(subprocess.call(cmd))

# Ensure root is in sys.path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel_trader.ai_sentinel import AISentinel
from sentinel_trader.config import Config
from sentinel_trader.exchange_client import ExchangeClient


async def check_a_binance_testnet(config: Config) -> bool:
    """Check A: Ping Binance Testnet, retrieve USDT/SOL balances, and fetch live ticker."""
    print("\n" + "=" * 70)
    print(" [CHECK A] BINANCE TESTNET SPOT CONNECTION & MARKET DATA")
    print("=" * 70)

    if not config.binance_api_key or not config.binance_api_secret:
        print(" [FAILED] BINANCE_API_KEY or BINANCE_API_SECRET is missing from .env!")
        return False

    if not config.binance_sandbox:
        print(" [FAILED] BINANCE_SANDBOX is not set to True! Refusing live mainnet operation.")
        return False

    print(f" * Target Trading Symbol: {config.target_symbol}")
    print(f" * Sandbox (Testnet) Mode: {config.binance_sandbox} (Strictly enforced)")
    print(f" * API Key: {config.binance_api_key[:8]}...{config.binance_api_key[-4:]}")

    client = ExchangeClient(config)
    try:
        print("\n Connecting to Binance Testnet via CCXT...")
        await client.initialize()

        if client.is_mock:
            print(" [FAILED] Exchange initialized in fallback mock mode instead of live testnet!")
            return False

        print(" [PASSED] Binance Testnet markets loaded successfully.")

        # 1. Query USDT Balance
        usdt_balance = await client.get_balance("USDT")
        print(f" [PASSED] Account Free Balance (USDT): {usdt_balance:,.2f} USDT")

        # 2. Query Base Asset (SOL) Balance
        base_asset = config.base_asset
        sol_balance = await client.get_balance(base_asset)
        print(f" [PASSED] Account Free Balance ({base_asset}): {sol_balance:,.4f} {base_asset}")

        # 3. Query Live Ticker Price
        sol_price = await client.get_ticker_price(config.target_symbol)
        print(f" [PASSED] Live Market Ticker Price ({config.target_symbol}): ${sol_price:,.4f}")

        # 4. Verify sanity
        assert sol_price > 0.0, "Ticker price must be positive"
        assert usdt_balance >= 0.0, "USDT balance must be non-negative"
        assert sol_balance >= 0.0, f"{base_asset} balance must be non-negative"

        print(f"\n [OK] Check A Passed: Binance Testnet is live and fully operational.")
        return True

    except Exception as e:
        print(f"\n [FAILED] Check A Error: {type(e).__name__}: {e}")
        return False
    finally:
        await client.close()


async def check_b_live_llm_inference(config: Config) -> bool:
    """Check B: Pass mock exploit headline to live LLM and verify action='PANIC_SELL'."""
    print("\n" + "=" * 70)
    print(" [CHECK B] LIVE LLM EMERGENCY SENTIMENT INFERENCE")
    print("=" * 70)

    if not config.llm_api_key:
        print(" [FAILED] LLM_API_KEY is missing from .env!")
        return False

    effective_model = config.get_effective_llm_model()
    print(f" * LLM Provider: {config.llm_provider}")
    print(f" * LLM Model: {effective_model}")
    print(f" * API Key: {config.llm_api_key[:6]}...{config.llm_api_key[-4:]}")

    sentinel = AISentinel(config, timeout_seconds=10.0)

    test_headline = "Critical zero-day bridge exploit detected in Solana runtime"
    print(f"\n Ingesting Headline for Threat Analysis:")
    print(f"   \"{test_headline}\"")

    analysis = None
    for attempt in range(1, 3):
        try:
            analysis = await sentinel.analyze(test_headline, config.target_symbol)
            if analysis.action == "PANIC_SELL":
                break
            if attempt < 2:
                print(f" [RETRY] Attempt {attempt} returned {analysis.action} ({analysis.reasoning[:50]}...). Retrying once...")
                await asyncio.sleep(1.0)
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(1.0)
            else:
                print(f"\n [FAILED] Check B Error: {type(e).__name__}: {e}")
                return False

    print("\n Structured AI Sentinel Output:")
    print(f"   - Action:       {analysis.action}")
    print(f"   - Threat Level: {analysis.threat_level}")
    print(f"   - Confidence:   {analysis.confidence:.2f}")
    print(f"   - Reasoning:    {analysis.reasoning}")

    # Assertions
    if analysis.action != "PANIC_SELL":
        print(f"\n [FAILED] Expected action='PANIC_SELL', got '{analysis.action}'!")
        return False

    if analysis.confidence < 0.70:
        print(f"\n [FAILED] Confidence {analysis.confidence:.2f} is below safety threshold 0.70!")
        return False

    print(f"\n [OK] Check B Passed: Live LLM correctly classified emergency panic sell.")
    return True


async def check_c_benign_headline(config: Config) -> bool:
    """Check C: Verify benign headline returns action='HOLD'."""
    print("\n" + "=" * 70)
    print(" [CHECK C] BENIGN HEADLINE CONTROL VERIFICATION")
    print("=" * 70)

    sentinel = AISentinel(config, timeout_seconds=10.0)
    benign_headline = "Solana ecosystem transaction volume steady following scheduled validator update"
    print(f" Ingesting Benign Headline:")
    print(f"   \"{benign_headline}\"")

    try:
        analysis = await sentinel.analyze(benign_headline, config.target_symbol)
        print(f"   - Action:       {analysis.action}")
        print(f"   - Threat Level: {analysis.threat_level}")
        print(f"   - Confidence:   {analysis.confidence:.2f}")

        if analysis.action != "HOLD":
            print(f"\n [FAILED] Expected action='HOLD' for benign news, got '{analysis.action}'!")
            return False

        print(f"\n [OK] Check C Passed: Benign news correctly maintained HOLD.")
        return True
    except Exception as e:
        print(f"\n [FAILED] Check C Error: {type(e).__name__}: {e}")
        return False


async def main():
    print("=" * 70)
    print(" SENTINEL-TRADER: PHASE 3 VERIFICATION SUITE")
    print(" Live Binance Testnet & Real LLM Integration Verification")
    print("=" * 70)

    config = Config()

    check_a_ok = await check_a_binance_testnet(config)
    check_b_ok = await check_b_live_llm_inference(config)
    check_c_ok = await check_c_benign_headline(config)

    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    print(f" Check A (Binance Testnet Live Connection): {'PASSED [OK]' if check_a_ok else 'FAILED [X]'}")
    print(f" Check B (Live LLM Panic-Sell Inference):   {'PASSED [OK]' if check_b_ok else 'FAILED [X]'}")
    print(f" Check C (Control Benign News HOLD):        {'PASSED [OK]' if check_c_ok else 'FAILED [X]'}")
    print("=" * 70)

    if check_a_ok and check_b_ok and check_c_ok:
        print("\n [SUCCESS] ALL PHASE 3 VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
        sys.exit(0)
    else:
        print("\n [ERROR] ONE OR MORE VERIFICATION CHECKS FAILED!\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
