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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
