"""Command-line interface for the paper trading system."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from tabulate import tabulate

from .config import Config
from .engine.scheduler import TradingEngine


def _setup_logging(cfg: Config) -> None:
    log_file = cfg.log_file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    level = getattr(logging, cfg.get("logging", "level", default="INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def cmd_run(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    engine = TradingEngine(cfg)
    engine.run_forever()


def cmd_once(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    engine = TradingEngine(cfg)
    engine.run_once()


def cmd_portfolio(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    engine = TradingEngine(cfg)
    positions = engine.broker.positions()
    cash = engine.broker.cash()

    rows = []
    total_mv = 0.0
    total_pnl = 0.0
    for symbol, pos in positions.items():
        try:
            quote = engine.data.get_quote(symbol)
            ltp = quote.ltp
            week52_high = quote.week52_high
        except Exception:
            ltp = pos.avg_price
            week52_high = None
        if week52_high and week52_high > 0:
            from_high = f"-{max(0.0, (week52_high - ltp) / week52_high * 100.0):.1f}%"
        else:
            from_high = "—"
        mv = pos.market_value(ltp)
        pnl = pos.unrealized_pnl(ltp)
        pnl_pct = pos.unrealized_pnl_pct(ltp)
        total_mv += mv
        total_pnl += pnl
        rows.append([symbol, pos.quantity, f"{pos.avg_price:.2f}", f"{ltp:.2f}", from_high, f"{mv:.2f}", f"{pnl:+.2f}", f"{pnl_pct:+.2f}%"])

    print(tabulate(rows, headers=["Symbol", "Qty", "Avg Price", "LTP", "From 52W High", "Mkt Value", "Unrl. P&L", "P&L %"], tablefmt="simple"))
    print()
    print(f"Cash:              ₹{cash:,.2f}")
    print(f"Positions value:   ₹{total_mv:,.2f}")
    print(f"Unrealized P&L:    ₹{total_pnl:+,.2f}")
    print(f"Total equity:      ₹{cash + total_mv:,.2f}")


def cmd_web(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    from .web.app import run_dashboard

    run_dashboard(cfg, host=args.host, port=args.port)


def cmd_serve(args: argparse.Namespace) -> None:
    """Run the autonomous trading loop and the web dashboard together in
    one process: the engine loop runs in a background thread, the
    dashboard's Flask server runs (blocking) in the main thread and
    shares the same TradingEngine instance, so there's nothing else to
    start separately -- Ctrl+C stops both."""
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    engine = TradingEngine(cfg)

    engine_thread = threading.Thread(target=engine.run_forever, name="trading-engine", daemon=True)
    engine_thread.start()

    from .web.app import create_app

    app = create_app(engine)
    # use_reloader must stay off: Flask's reloader forks a second process,
    # which would start a second copy of the engine thread too. threaded:
    # /api/candidates can take a while (network calls across the whole
    # universe) -- without it the dev server would serve one request at a
    # time and the rest of the dashboard would appear frozen during a scan.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    from .backtest.engine import Backtester

    bt = Backtester(cfg, start=args.start, end=args.end)
    print(f"Backtesting {args.start} -> {args.end} against {len(bt.universe)} symbols "
          f"(this fetches full history per symbol and can take a while)...")
    result = bt.run()

    print()
    print(f"Backtest: {result.start_date} -> {result.end_date} ({result.trading_days} trading days, "
          f"{result.symbols_with_data}/{len(bt.universe)} symbols had usable data)")
    print(f"Starting capital:    ₹{result.starting_capital:,.2f}")
    print(f"Ending equity:       ₹{result.ending_equity:,.2f}")
    print(f"Total return:        {result.total_return_pct:+.2f}%")
    print(f"CAGR:                {result.cagr_pct:+.2f}%")
    print(f"Max drawdown:        -{result.max_drawdown_pct:.2f}%")
    print(f"Round-trip trades:   {result.num_round_trips}")
    print(f"Win rate:            {result.win_rate_pct:.1f}%")
    print(f"Avg win / avg loss:  ₹{result.avg_win_inr:+,.2f} / ₹{result.avg_loss_inr:+,.2f}")
    if result.benchmark_total_return_pct is not None:
        beat = "beat" if result.total_return_pct > result.benchmark_total_return_pct else "lagged"
        print(f"Benchmark ({result.benchmark_symbol}) buy & hold: {result.benchmark_total_return_pct:+.2f}% "
              f"total return, {result.benchmark_cagr_pct:+.2f}% CAGR -- strategy {beat} it")
    else:
        print(f"Benchmark ({result.benchmark_symbol}) buy & hold: unavailable (could not fetch index history)")

    if args.trades:
        print("\nTrade log:")
        rows = [[t["date"], t["side"], t["symbol"], t["qty"], f"{t['price']:.2f}",
                 "—" if t["pnl"] is None else f"{t['pnl']:+.2f}", t["reason"]] for t in result.trade_log]
        print(tabulate(rows, headers=["Date", "Side", "Symbol", "Qty", "Price", "P&L", "Reason"], tablefmt="simple"))

    if args.export:
        import csv
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "equity"])
            writer.writerows(result.equity_curve)
        print(f"\nEquity curve written to {args.export}")


def cmd_setup_auth(args: argparse.Namespace) -> None:
    import getpass

    from .web import auth

    if auth.is_configured():
        if not args.force:
            print(f"Authentication is already configured ({auth.AUTH_FILE}).")
            print("This bootstraps the FIRST admin account only -- to add more users "
                  "(admin or not), log in as an admin and use the Admin > Users page.")
            print("Re-run with --force to wipe ALL existing users and start over "
                  "(irreversible -- only do this if you're locked out).")
            return
        print("--force: deleting the existing auth store and every user/2FA enrollment in it.")
        os.remove(auth.AUTH_FILE)

    print("Bootstrapping the first (admin) dashboard account. This does NOT affect")
    print("trading -- it only gates access to the web dashboard (python main.py web/serve).")
    print("Two-factor authentication is set up separately, the first time this account")
    print("logs in from the browser -- not here.\n")

    while True:
        username = input("Choose a username: ").strip()
        if username:
            break
        print("Username can't be empty.\n")

    while True:
        password = getpass.getpass("Choose a dashboard password: ")
        if len(password) < 8:
            print("Please use at least 8 characters.\n")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords didn't match, try again.\n")
            continue
        break

    auth.bootstrap_admin(username, password)

    print(f"\nSaved to {auth.AUTH_FILE} (never commit this file -- it's already in .gitignore).")
    print(f"\nStart the dashboard (`python main.py web` or `serve`) and log in as '{username}' with")
    print("that password -- you'll be walked through 2FA setup (scan/enter a code into an")
    print("authenticator app) right there on first login.")
    print("\nOnce logged in, use the Admin > Users page to create additional accounts -- each")
    print("new user goes through the same one-time 2FA setup on their own first login.")


def cmd_history(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    engine = TradingEngine(cfg)
    trades = engine.storage.get_trades(limit=args.limit)
    rows = [[t.timestamp, t.side, t.symbol, t.quantity, f"{t.price:.2f}", f"{t.charges:.2f}", t.reason] for t in trades]
    print(tabulate(rows, headers=["Timestamp", "Side", "Symbol", "Qty", "Price", "Charges", "Reason"], tablefmt="simple"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrader", description="Autonomous NSE momentum paper trading simulator")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run the autonomous trading loop (blocks until killed)").set_defaults(func=cmd_run)
    sub.add_parser("once", help="Run a single scan/trade cycle and exit").set_defaults(func=cmd_once)
    sub.add_parser("portfolio", help="Show current positions and P&L").set_defaults(func=cmd_portfolio)

    web = sub.add_parser("web", help="Launch the local read-only web dashboard")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(func=cmd_web)

    serve = sub.add_parser("serve", help="Run the autonomous trading loop AND the web dashboard together, in one command")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    hist = sub.add_parser("history", help="Show trade history")
    hist.add_argument("--limit", type=int, default=50)
    hist.set_defaults(func=cmd_history)

    bt = sub.add_parser("backtest", help="Replay the strategy against historical data instead of trading live")
    bt.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    bt.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    bt.add_argument("--trades", action="store_true", help="Also print the full trade log")
    bt.add_argument("--export", default=None, help="Optional CSV path to save the daily equity curve")
    bt.set_defaults(func=cmd_backtest)

    setup_auth = sub.add_parser("setup-auth", help="Bootstrap the dashboard's first admin login (username + password only; 2FA is set up on first web login)")
    setup_auth.add_argument("--force", action="store_true", help="Wipe ALL existing users/2FA and start over")
    setup_auth.set_defaults(func=cmd_setup_auth)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
