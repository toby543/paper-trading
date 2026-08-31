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

There are two selectable entry strategies (`strategy.mode` in
`config.yaml`, or the "Strategy mode" field in the dashboard's Edit
Settings panel). Both share the same liquidity/price-range/trend
filters, the market regime filter, and all exit rules below — they only
differ in how a stock qualifies as a BUY.

**Strategy — 52-Week-High Momentum** (`strategy.mode: 52w_high`, the default):
A stock is bought when it is (1) trading within a configurable band of
its 52-week high, (2) has strong trailing momentum (e.g. ≥15% return
over the last ~90 days), (3) is trading above its 50-day and 200-day
moving averages (trend confirmation, no death-cross), (4) is liquid
enough to trade (average daily turnover filter), (5) is **outperforming
the Nifty 50 by a minimum margin over the same window** (relative
strength — a stock merely drifting up with a rising market isn't the
same signal as one genuinely leading it), and (6) shows **recent trading
volume running hot relative to its own baseline** (volume confirmation —
a move near the 52-week high on light volume is a weaker signal than
one with real participation). Candidates are ranked by a
momentum+proximity+relative-strength score and the top few are bought
each scan, sized as a fixed percentage of total equity. Relative
strength and volume confirmation are each independently toggleable in
`config.yaml` (`strategy.min_relative_strength_pct`,
`strategy.volume_confirmation.enabled`) and fail open (skip the check
rather than blocking every trade) if the underlying index/volume data
can't be fetched for a given scan. An optional price range
(`strategy.min_ltp_inr`/`max_ltp_inr`, 0 = no bound) can also skip
penny stocks or steer clear of very high-priced names.

**Strategy — Cross-Sectional Momentum** (`strategy.mode: cross_sectional_momentum`):
a "dual momentum" approach: instead of judging each stock independently
against fixed thresholds, it computes every stock's trailing return over
`strategy.cross_sectional.lookback_days` (default 252 trading days,
i.e. ~12 months) ending `skip_recent_days` before today (default 21 —
the classic "12-1" momentum window, which skips the most recent month to
avoid short-term reversal noise), ranks the *whole universe* against
each other, and buys only the top `top_pct` percentile (default top
10%). A stock still has to pass the same trend-confirmation
(fast/slow moving average), liquidity, and price-range filters as the
52-week-high strategy, and is further required to have a *positive*
trailing return of its own even at a very wide `top_pct` — this
"absolute-return gate" is the other half of dual momentum (the market
regime filter below being the market-wide half of it), so nothing gets
bought purely for being the least-bad decliner in a falling universe.
This mode tends to be more robust across different market regimes than
proximity-to-52w-high (which favors strongly trending bull markets)
since it ranks stocks relative to their peers rather than against a
fixed level. Recommended if you want to compare the two head-to-head:
switch the mode, then run the dashboard's Backtest panel (or `python
main.py backtest`) against the same date range for both.

Positions are exited on a **hard stop-loss** from entry, a **trailing
stop** from the highest close since entry, a **momentum breakdown**
(close falls below the 50-day moving average), or an optional **take
profit** target — whichever comes first. Take profit is off by default
(`risk.take_profit_pct: 0`): momentum strategies are usually better
served by the trailing stop's "let winners run" behavior than by
capping upside at a fixed target, but it's there if you want a hard
sell target anyway. Exits always apply regardless of market regime
(see below) — only new entries are gated.

**Market regime filter:** momentum strategies get badly hurt trading
through a falling market, so by default new entries only happen while
the Nifty 50 index itself is trading above its own 200-day moving
average (`regime` section in `config.yaml`). When the index is below
that average, the engine keeps managing exits on existing positions as
normal but stops opening new ones until the index recovers. Disable
with `regime.enabled: false` if you'd rather trade through every
regime. The benchmark index (`regime.index_symbol`, also used for
relative strength) isn't limited to Nifty 50 — the dashboard's Edit
Settings panel offers common alternatives (Nifty Bank, Nifty IT, Nifty
500, several sector indexes, etc.) as autocomplete suggestions, or type
any other Yahoo Finance ticker directly; an unreachable/invalid ticker
just fails that check open rather than blocking trading.

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
  backtest/{engine,metrics}.py     Historical replay of the same strategy/exit logic
  cli.py                     Command-line interface
main.py                      Entry point
tests/                       pytest unit tests (strategy, broker, calendar)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### One-click launchers (Windows / macOS)

If you'd rather not run commands by hand, double-click:

- **`start_windows.bat`** on Windows
- **`start_mac.command`** on macOS

Either one sets up a virtual environment and installs dependencies the
first time you run it (every run after that starts straight up), walks
you through `setup-auth` the first time (creating the dashboard's admin
account), then starts the dashboard and opens it in your browser at
`http://127.0.0.1:8000`. Python 3 must already be installed on the
machine — these scripts automate the setup/run steps, they don't bundle
Python itself. Leave the terminal/command window open while it's
running; press Ctrl+C in it to stop the app.

### Always-on setup (Raspberry Pi)

Unlike the Windows/Mac launchers above (which run in a terminal you keep
open), a Raspberry Pi is normally headless and meant to run this 24/7 in
the background. `deploy/raspberrypi/setup_pi.sh` sets it up as a
`systemd` service instead — it starts on boot and restarts itself if it
ever crashes:

```bash
git clone <your-repo-url>
cd paper-trading
bash deploy/raspberrypi/setup_pi.sh
```

This installs a 1GB swap file (if one doesn't already exist — a Pi's
usable RAM is often tighter than advertised once Flask/pandas/numpy are
all loaded together), creates the virtual environment and installs
dependencies (Raspberry Pi OS's `pip` is preconfigured to use
[piwheels.org](https://www.piwheels.org/) for precompiled ARM wheels of
pandas/numpy, so this doesn't trigger a slow from-source build), walks
you through `setup-auth` the first time, then installs and starts
`deploy/raspberrypi/papertrader.service`. The dashboard binds to
`0.0.0.0:8000` (not just `127.0.0.1`) so it's reachable from other
devices on the same network — the script prints the URL to use
(`http://<pi's-IP>:8000`) at the end.

Useful commands afterward:

```bash
sudo systemctl status papertrader     # is it running?
journalctl -u papertrader -f          # follow the logs
sudo systemctl restart papertrader    # restart (e.g. after git pull)
```

To pick up a code update, `git pull` then `sudo systemctl restart
papertrader` — you don't need to re-run `setup_pi.sh` unless
`requirements.txt` changed or this is a fresh clone.

A Pi Zero W/Zero 2 W or an original Pi 1 Model B (256–512MB RAM,
single-core) will run this but slowly — consider pointing
`universe.file` at the smaller `data/universe.csv` (~105 symbols)
instead of the full Nifty 500 list on those boards. A Pi 2 Model B or
newer (1GB+ RAM, quad-core) handles the full Nifty 500 universe
comfortably.

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

### Run the engine and dashboard together

```bash
python main.py serve                    # http://127.0.0.1:8000
python main.py serve --host 0.0.0.0 --port 8080
```

Runs the autonomous trading loop in a background thread and the web
dashboard in the same process/terminal, sharing one `TradingEngine`
instance — one command instead of two terminals. Ctrl+C stops both.
This is the simplest way to run everything; use `run` and `web`
separately (below) only if you specifically want them as independent
processes (e.g. the engine on a server, the dashboard on your laptop).

### Web dashboard

```bash
python main.py web                      # http://127.0.0.1:8000
python main.py web --host 0.0.0.0 --port 8080
```

A local, read-only dashboard (Flask) showing live-marked open positions,
unrealized P&L, an equity curve, and recent trade history, self-refreshing
every 10 seconds, with a responsive layout that works on a phone.

Its **Backtest** panel runs the same replay as `python main.py backtest`
(below) without leaving the browser: pick a start/end date and universe
(default or Nifty 500), click "Run Backtest", and watch a live progress
bar (fetch phase, then day-by-day simulation) followed by the same
summary stats, equity curve, and trade log the CLI prints. It runs as a
background job on the server so the page stays responsive while a long
backtest is in flight — polling every 1.5s — and, like the Scanner
Watchlist, is deliberately on-demand rather than auto-run.

Its **Scanner Watchlist** panel previews which stocks currently qualify
for the next buy scan — without placing any trades — via a "Scan Now"
button. It's deliberately manual, not auto-refreshed like the rest of
the dashboard: evaluating the whole universe means a real NSE/Yahoo
network call per symbol, so polling it every 10 seconds the way the
other panels do would be slow and hammer the data source for no reason.
Useful for checking what changing a setting (e.g. the benchmark index)
would surface before it actually happens live.

**To open it from your phone:** start it with `--host 0.0.0.0`, find the
host machine's LAN IP (`ipconfig getifaddr en0` on Mac, `hostname -I` on
Linux, `ipconfig` on Windows), then visit `http://<that-ip>:8000` from a
phone on the **same Wi-Fi**. This is Flask's development server — fine
for personal/LAN use.

It reads the same SQLite ledger the engine writes to —
run it alongside `python main.py run` (or a cron-driven `once`) in a
separate process/terminal; the dashboard never places trades itself.

### Authentication (password + 2FA)

By default the dashboard has **no login at all** — fine for pure
localhost/LAN use, but the Edit Settings panel can change the live
strategy config and the Backtest panel can kick off expensive scans, so
anything reachable beyond your own network needs a real gate in front
of it (e.g. if you're running this on a cloud VM with a public IP).

Bootstrap the first (admin) account once with:

```bash
python main.py setup-auth
```

This just asks for a username and password — **not** a 2FA code. Every
account (this first one and any created later) sets up its own 2FA the
first time it logs in from the browser: on first login you're shown a
setup key/URI to add to an authenticator app (Google Authenticator,
Authy, 1Password, etc. — "enter setup key manually", no QR scanning
needed) and asked to enter one code back to confirm you saved it
correctly before 2FA is actually turned on for that account. From then
on, that account needs username + password + the current 6-digit code
on every login.

Credentials for every account live in `data/auth_secrets.json` —
**never** committed to git (already in `.gitignore`; this repo is
public, so double-check before ever force-adding files in `data/`).
Without that file present, the dashboard runs exactly as before
(unauthenticated), logging a startup warning either way.

**Adding more users:** log in as an admin and use the **Users** link in
the top bar (`/admin/users`, admin-only — a non-admin account can't
reach it). From there you can create additional accounts (optionally
also admins), delete one, or reset a user's 2FA if they lose their
phone (clears their enrollment so they set it up again on next login).
A new account you create there goes through the exact same first-login
2FA setup as the bootstrap admin did — you never see or transmit
anyone else's TOTP secret, since it's generated in their own browser
session at first login, not by you.

Failed login attempts are throttled (5 tries, then a 15-minute lockout
per source IP) since a login form is exactly what automated scanners
probe once something is reachable from the internet. Sessions last 12
hours; use the "Sign out" button in the top bar to end one early.
`python main.py setup-auth --force` wipes **every** account and 2FA
enrollment and starts over from a single new admin — only use it if
you're actually locked out, not to reset just one user (use the Users
page's "Reset 2FA" or delete-and-recreate for that instead).

**This alone is not a substitute for network-level restrictions** —
still lock down inbound access at the firewall/security-group level to
just your own IP where you can, and prefer HTTPS (e.g. via a reverse
proxy or a tunnel like Cloudflare Tunnel) over plain HTTP if this is
reachable from the public internet, since Flask's dev server has no
TLS of its own and a plain-HTTP login submits your password/code
unencrypted over the network.

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

- `strategy.mode` — `52w_high` (default) or `cross_sectional_momentum` (see above). A restart is required to switch, like any other engine-construction-time setting.
- `strategy.cross_sectional.lookback_days` / `skip_recent_days` / `top_pct` — only used in `cross_sectional_momentum` mode: the trailing-return ranking window and the top percentile bought.
- `strategy.proximity_to_52w_high_pct` — how close to the 52-week high a stock must be to qualify (52w_high mode only).
- `strategy.min_momentum_return_pct` / `momentum_lookback_days` — trailing momentum filter.
- `strategy.fast_ma_days` / `slow_ma_days` — trend-confirmation moving averages.
- `risk.stop_loss_pct` / `trailing_stop_pct` — exit rules.
- `risk.position_size_pct_of_equity` / `max_open_positions` — sizing & diversification.
- `universe.file` — edit `data/universe.csv` to change which NSE stocks are scanned,
  or point it at `data/universe_nifty500.csv` (~480 symbols, included) for much
  broader coverage:
  ```yaml
  universe:
    file: data/universe_nifty500.csv
  ```
  A bigger universe means more candidates to evaluate each scan (roughly
  proportionally slower — see the network/performance notes above) but a
  better chance of finding something that clears every filter on a given day.
  **Caveat:** this list was hand-compiled from training knowledge, not pulled
  live from NSE, so it's an approximation of the real Nifty 500 and will drift
  as the index gets periodically rebalanced. A stale/incorrect symbol just
  gets silently skipped during a scan (no crash) — but if you want it exact,
  cross-check or replace it with NSE's official current list (downloadable
  from niftyindices.com).

## Backtesting

```bash
python main.py backtest --start 2023-01-01 --end 2025-01-01
python main.py backtest --start 2023-01-01 --end 2025-01-01 --trades --export curve.csv
```

Replays the exact same strategy/exit code (`evaluate_candidate`,
`check_exit`, `rank_candidates`, `is_market_in_uptrend`) day-by-day
against historical daily bars fetched once per symbol up front, instead
of waiting on live scans over months to see whether a strategy or config
change actually works. It runs against a throwaway temp SQLite ledger
(never your real `data/state.db`) but reuses `PaperBroker`/`RiskManager`
unchanged, so position sizing, slippage, and charges match live trading
exactly — the only thing that differs from `python main.py run` is where
the price data comes from and that time is simulated instead of
wall-clock, so there's no risk of backtest and live behavior quietly
diverging.

Reports starting/ending equity, total return, CAGR, max drawdown, number
of round-trip trades, win rate, and average win/loss size. `--trades`
also prints the full trade log; `--export` saves the daily equity curve
to a CSV for charting elsewhere.

**Caveats:**
- It needs `momentum_lookback_days`/`slow_ma_days` worth of price history
  *before* `--start` to compute moving averages and momentum correctly —
  it fetches a ~420-day buffer automatically, but very early dates in a
  long backtest may still show few/no candidates simply because the
  buffer itself is still warming up.
- A larger universe (e.g. the Nifty 500 list) or longer date range means
  more symbols × more days to fetch and simulate — expect it to take a
  while for anything beyond a small universe or a few months. Fetches
  are paced (~0.2s apart) to avoid tripping Yahoo Finance's rate
  limiting, which otherwise surfaces as "possibly delisted" 404s on
  perfectly real, actively-traded stocks and is easy to mistake for a
  wrong ticker — if several genuinely-listed symbols fail together in
  the same run (especially ones that resolved fine before), suspect
  throttling before assuming the symbol is wrong.
- Past performance in a backtest is not a promise of future results —
  it validates that the *logic* behaves as intended against history, not
  that the strategy will keep working going forward.

## Testing

```bash
pytest
```

Tests cover the strategy's entry/exit rules (using synthetic price
series, no network needed), the paper broker's order/cash/position
bookkeeping, the NSE market-hours calendar, and the backtest's summary
statistics (drawdown, CAGR, win rate).

## Notes & limitations

- The official NSE JSON API is unofficial/undocumented and frequently
  rate-limits or blocks requests from data-center IPs; run this from a
  residential/normal network for the most reliable live NSE access —
  the automatic Yahoo Finance fallback keeps things working either way,
  typically with a small delay.
- This system only ever simulates trades against its own local ledger;
  it has no brokerage integration and cannot place real orders.
