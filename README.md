# NSE Momentum Paper Trader

A fully autonomous **paper trading** (simulated, no real money, no real
orders) system for the Indian equity market. It watches real-time NSE
prices, runs a **52-week-high momentum swing-trading strategy**, and
autonomously buys and sells a virtual portfolio during market hours —
persisting everything (cash, positions, trade log, equity curve) to a
local SQLite database.

> ⚠️ **This is a simulator for research/education only.** It does not
> place real orders, is not investment advice, and past/simulated
> performance is not indicative of future results.

## How it works

**Strategy — 52-Week-High Momentum:**
A stock is bought when it is (1) trading within a configurable band of
its 52-week high, (2) has strong trailing momentum (e.g. ≥15% return
over the last ~90 days), (3) is trading above its 50-day and 200-day
moving averages (trend confirmation, no death-cross), and (4) is liquid
enough to trade (average daily turnover filter). Candidates are ranked
by a momentum+proximity score and the top few are bought each scan,
sized as a fixed percentage of total equity.

Positions are exited on a **hard stop-loss** from entry, a **trailing
stop** from the highest close since entry, or a **momentum breakdown**
(close falls below the 50-day moving average) — whichever comes first.

**Data:** real-time quotes and 52-week high/low come from the NSE
website's own JSON API; historical daily bars used for moving averages
and momentum returns come from Yahoo Finance (`.NS` tickers). If NSE is
unreachable (its anti-bot layer often blocks data-center/CI IPs), the
system automatically falls back to Yahoo Finance for live quotes too.

**Engine:** during NSE trading hours (09:15–15:30 IST, Mon–Fri,
excluding holidays in `data/nse_holidays.csv`) it checks exits every few
minutes and scans the full universe for new entries on a longer
interval. Outside market hours it idles and rechecks periodically.

## Project layout

```
config.yaml                  All strategy/risk/engine parameters
data/universe.csv            Tradable NSE symbols (edit to change universe)
data/nse_holidays.csv        NSE trading holiday calendar
data/state.db                SQLite portfolio state (created on first run)
src/papertrader/
  config.py                  YAML config loader
  data/nse_client.py         NSE + Yahoo Finance data access, with fallback
  data/universe.py           Universe CSV loader
  strategy/momentum_52w_high.py   Entry/exit signal logic
  portfolio/{models,storage,broker}.py   Paper execution engine + persistence
  risk/risk_manager.py       Position sizing & exposure limits
  engine/{market_hours,scheduler}.py     Autonomous scan loop
  cli.py                     Command-line interface
main.py                      Entry point
tests/                       pytest unit tests (strategy, broker, calendar)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the autonomous loop (blocks, trades continuously during market hours):

```bash
python main.py run
```

Run a single scan-and-trade cycle (good for cron/systemd timers instead
of a long-running process):

```bash
python main.py once
```

View current holdings and P&L:

```bash
python main.py portfolio
```

View trade history:

```bash
python main.py history --limit 100
```

### Web dashboard

```bash
python main.py web                      # http://127.0.0.1:8000
python main.py web --host 0.0.0.0 --port 8080
```

A local, read-only dashboard (Flask) showing live-marked open positions,
unrealized P&L, an equity curve, and recent trade history, self-refreshing
every 10 seconds. It reads the same SQLite ledger the engine writes to —
run it alongside `python main.py run` (or a cron-driven `once`) in a
separate process/terminal; the dashboard never places trades itself.

### Running continuously

Two supported ways to keep it running unattended:

1. **Long-running process** — `python main.py run` under `systemd`,
   `tmux`/`screen`, `pm2`, or a Docker container with a restart policy.
   It sleeps outside market hours and resumes automatically.
2. **Scheduled invocation** — run `python main.py once` from `cron`
   every few minutes between 09:15–15:30 IST on weekdays; the engine
   checks the market calendar itself so extra invocations outside hours
   are harmless no-ops.

## Configuring the strategy

All thresholds live in `config.yaml`:

- `strategy.proximity_to_52w_high_pct` — how close to the 52-week high a stock must be to qualify.
- `strategy.min_momentum_return_pct` / `momentum_lookback_days` — trailing momentum filter.
- `strategy.fast_ma_days` / `slow_ma_days` — trend-confirmation moving averages.
- `risk.stop_loss_pct` / `trailing_stop_pct` — exit rules.
- `risk.position_size_pct_of_equity` / `max_open_positions` — sizing & diversification.
- `universe.file` — edit `data/universe.csv` to change which NSE stocks are scanned.

## Testing

```bash
pytest
```

Tests cover the strategy's entry/exit rules (using synthetic price
series, no network needed), the paper broker's order/cash/position
bookkeeping, and the NSE market-hours calendar.

## Notes & limitations

- The official NSE JSON API is unofficial/undocumented and frequently
  rate-limits or blocks requests from data-center IPs; run this from a
  residential/normal network for the most reliable live NSE access —
  the automatic Yahoo Finance fallback keeps things working either way,
  typically with a small delay.
- This system only ever simulates trades against its own local ledger;
  it has no brokerage integration and cannot place real orders.
