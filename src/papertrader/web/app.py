"""Local read-only web dashboard for the paper trading account.

Normally runs as its own process, reading the same SQLite ledger the
trading engine (`python main.py run`) writes to -- it does not place
trades itself; it only displays live-marked positions, P&L, the equity
curve and the trade history, refreshing itself every few seconds in
the browser. `create_app()` takes an existing TradingEngine so it can
also be embedded in the same process as the engine (see
`python main.py serve`, cli.py) sharing one engine instance instead of
constructing a second one.
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from ..config import Config
from ..engine.scheduler import TradingEngine
from .data_api import build_equity_curve, build_summary, build_trades
from .filters import indian_currency

_HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(engine: TradingEngine) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=os.path.join(_HERE, "static"),
    )
    app.jinja_env.filters["inr"] = indian_currency
    app.config["ENGINE"] = engine
    cfg = engine.cfg

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            slippage_bps=cfg.get("execution", "slippage_bps", default=5.0),
            flat_charges_inr=cfg.get("execution", "flat_charges_inr", default=20.0),
            strategy=cfg.get("strategy", default={}),
            risk=cfg.get("risk", default={}),
            regime=cfg.get("regime", default={}),
            account=cfg.get("account", default={}),
            universe_cfg=cfg.get("universe", default={}),
            execution=cfg.get("execution", default={}),
            engine_cfg=cfg.get("engine", default={}),
            data_source=cfg.get("data_source", default={}),
            logging_cfg=cfg.get("logging", default={}),
        )

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
    engine = TradingEngine(cfg)
    app = create_app(engine)
    app.run(host=host, port=port, debug=debug)
