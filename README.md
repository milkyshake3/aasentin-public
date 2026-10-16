# Sentinel-Trader

Sentinel-Trader is an educational Python project for experimenting with automated
crypto trading. It uses technical indicators, rule-based risk controls, and optional
AI-assisted news analysis.

**The default way to use this project is local simulation.** It is not financial
advice, it does not guarantee profits, and it should not be used with real money.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-success.svg?style=flat&logo=pytest&logoColor=white)](#run-the-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What it does

The bot follows a simple state machine:

1. Look for a possible entry using an EMA trend filter and RSI pullback signal.
2. Open a simulated or testnet position.
3. Protect the position with take-profit and stop-loss rules.
4. Optionally inspect crypto news for severe threats.
5. Record the result and continue scanning.

The AI is only used for news threat analysis. It does not choose position sizes,
calculate order prices, or receive exchange credentials directly.

## Quick start

### 1. Install

You need Python 3.10 or newer.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Create a local configuration

Copy the example file. On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

For a first run, use simulation mode. These settings do not need API keys:

```env
MOCK_EXCHANGE=True
LLM_PROVIDER=mock
BINANCE_API_KEY=replace_with_testnet_api_key
BINANCE_API_SECRET=replace_with_testnet_api_secret
LLM_API_KEY=replace_with_llm_api_key
BINANCE_SANDBOX=True
ALLOW_PRODUCTION_TRADING=False
```

Never commit `.env`, API keys, exchange secrets, or Discord webhook URLs.

### 3. Run a simulation

This example runs a complete simulated trading cycle:

```powershell
python -m sentinel_trader.main --scenario fsm_cycle --poll-interval 0.5
```

Other useful scenarios are:

```powershell
python -m sentinel_trader.main --scenario take_profit --poll-interval 0.5
python -m sentinel_trader.main --scenario stop_loss --poll-interval 0.5
python -m sentinel_trader.main --scenario ai_panic --poll-interval 0.5
python -m sentinel_trader.main --scenario failsafe --poll-interval 0.5
```

These scenarios use the in-memory mock exchange and do not trade real funds.

## Run the tests

```powershell
python -m pytest -q
```

The tests cover the strategy, exchange routing, risk controls, news handling,
notifications, persistence, and simulated trading flows.

## Project layout

```text
sentinel_trader/
├── main.py              # Command-line app and trading state machine
├── config.py            # Environment settings and safety validation
├── strategy.py          # RSI/EMA entry logic
├── exchange_client.py   # Mock and Binance exchange clients
├── risk_engine.py       # Stop-loss, take-profit, and liquidation rules
├── universe_screener.py # Selects liquid trading pairs
├── ai_sentinel.py       # Optional AI news threat analysis
├── news_watcher.py      # Crypto news collection and filtering
├── state_store.py       # Crash-safe local state
├── analytics.py         # Trade history and performance metrics
└── notifier.py          # Optional Discord alerts

test_*.py                # Automated tests
.env.example             # Safe configuration template
requirements.txt         # Python dependencies
```

Runtime state, trade history, logs, caches, virtual environments, and local
credentials are intentionally ignored by Git.

## How the strategy works

- **EMA trend filter:** the price must be above the configured EMA.
- **RSI pullback signal:** the strategy looks for an upward move from an oversold
  RSI value.
- **Take profit:** the default target is `+5%`.
- **Stop loss:** the default limit is `-2.5%`.
- **Cooldown:** after a position closes, the bot waits before scanning again.

These values are educational examples, not recommendations.

## Optional Binance Testnet use

After testing the simulations, you can connect to Binance Testnet with newly
generated testnet credentials:

1. Create keys at [Binance Testnet](https://testnet.binance.vision/).
2. Put them only in your local `.env`.
3. Keep `BINANCE_SANDBOX=True`.
4. Keep `ALLOW_PRODUCTION_TRADING=False`.
5. Verify the code and testnet behavior before considering anything else.

To check configured connections:

```powershell
python verify_connections.py
```

Do not use production credentials while learning or testing this project. Any
credentials that appeared in an older copy of this repository should be revoked
and regenerated before publishing or using them again.

## Configuration

The most useful settings are:

| Variable | Default | Purpose |
|---|---:|---|
| `MOCK_EXCHANGE` | `False` | Use the in-memory exchange when `True` |
| `BINANCE_SANDBOX` | `True` | Keep exchange orders on Binance Testnet |
| `ALLOW_PRODUCTION_TRADING` | `False` | Safety lock for production trading |
| `TARGET_SYMBOL` | `SOL/USDT` | Primary symbol |
| `TIMEFRAME` | `15m` | Candle timeframe |
| `TAKE_PROFIT_PCT` | `0.05` | Take-profit threshold |
| `STOP_LOSS_PCT` | `0.025` | Stop-loss threshold |
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai`, or `mock` |
| `DISCORD_WEBHOOK_URL` | empty | Optional alert webhook |

See [`.env.example`](.env.example) for the complete list.

## Safety notes

- This project is for education and experimentation.
- Cryptocurrency trading can result in the loss of money.
- Test with simulations and Binance Testnet first.
- Never commit credentials.
- Review the source code yourself before connecting any exchange account.
- Live trading requires explicit configuration and is disabled by default.

## License

This project is available under the [MIT License](LICENSE).
