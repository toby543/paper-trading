"""Local read-only web dashboard for the paper trading account.

Runs as its own process, reading the same SQLite ledger the trading
engine (`python main.py run`) writes to. It does not place trades
itself; it only displays live-marked positions, P&L, the equity curve
and the trade history, refreshing itself every few seconds in-browser.
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from ..config import Config
from ..engine.scheduler import TradingEngine
from .data_api import build_equity_curve, build_summary, build_trades

_HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(cfg: Config) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=os.path.join(_HERE, "static"),
    )
    engine = TradingEngine(cfg)
    app.config["ENGINE"] = engine

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/summary")
    def api_summary():
        return jsonify(build_summary(engine))

    @app.get("/api/trades")
    def api_trades():
        limit = request.args.get("limit", default=100, type=int)
        return jsonify(build_trades(engine, limit=limit))

    @app.get("/api/equity_curve")
    def api_equity_curve():
        limit = request.args.get("limit", default=500, type=int)
        return jsonify(build_equity_curve(engine, limit=limit))

    return app


def run_dashboard(cfg: Config, host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    app = create_app(cfg)
    app.run(host=host, port=port, debug=debug)
