"""Command-line interface for the paper trading system."""
from __future__ import annotations

import argparse
import logging
import os
import sys

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
            ltp = engine.data.get_quote(symbol).ltp
        except Exception:
            ltp = pos.avg_price
        mv = pos.market_value(ltp)
        pnl = pos.unrealized_pnl(ltp)
        pnl_pct = pos.unrealized_pnl_pct(ltp)
        total_mv += mv
        total_pnl += pnl
        rows.append([symbol, pos.quantity, f"{pos.avg_price:.2f}", f"{ltp:.2f}", f"{mv:.2f}", f"{pnl:+.2f}", f"{pnl_pct:+.2f}%"])

    print(tabulate(rows, headers=["Symbol", "Qty", "Avg Price", "LTP", "Mkt Value", "Unrl. P&L", "P&L %"], tablefmt="simple"))
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
