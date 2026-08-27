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
from ..config_editor import update_config_file
from ..engine.scheduler import TradingEngine
from .backtest_jobs import get_job, start_backtest_job
from .data_api import build_candidates, build_equity_curve, build_summary, build_trades
from .filters import indian_currency
from .settings_schema import EDITABLE_SETTINGS, coerce_and_validate, get_value

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

    @app.get("/api/candidates")
    def api_candidates():
        # Expensive (network calls across the whole universe) -- the
        # dashboard triggers this on demand (a "Scan Now" button), never
        # on the regular auto-refresh poll. threaded=True on app.run()
        # below keeps this from blocking the rest of the dashboard while
        # it runs.
        limit = request.args.get("limit", default=20, type=int)
        return jsonify(build_candidates(engine, limit=limit))

    @app.post("/api/backtest/run")
    def api_backtest_run():
        payload = request.get_json(silent=True) or {}
        start = (payload.get("start") or "").strip()
        end = (payload.get("end") or "").strip()
        universe_file = (payload.get("universe_file") or "").strip() or None
        if not start or not end:
            return jsonify({"ok": False, "error": "Start and end dates are both required."}), 400
        # Backtester itself validates start < end etc., but that happens
        # inside the background thread (see backtest_jobs.py) -- any such
        # error surfaces via the job's "error" status on the next poll,
        # not as a synchronous 400 here.
        job_id = start_backtest_job(cfg, start, end, universe_file=universe_file)
        return jsonify({"ok": True, "job_id": job_id})

    @app.get("/api/backtest/status/<job_id>")
    def api_backtest_status(job_id: str):
        job = get_job(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Unknown or expired job id."}), 404
        return jsonify({"ok": True, "job": job})

    @app.get("/api/settings")
    def api_get_settings():
        fields = []
        for spec in EDITABLE_SETTINGS:
            fields.append({**spec, "path": list(spec["path"]), "value": get_value(cfg.raw, spec["path"])})
        return jsonify({"fields": fields})

    @app.post("/api/settings")
    def api_update_settings():
        payload = request.get_json(silent=True) or {}
        raw_updates = payload.get("updates", [])
        if not isinstance(raw_updates, list) or not raw_updates:
            return jsonify({"ok": False, "errors": ["No changes submitted."]}), 400

        coerced = []
        errors = []
        for item in raw_updates:
            path = tuple(item.get("path", []))
            try:
                value = coerce_and_validate(path, item.get("value"))
                coerced.append((list(path), value))
            except ValueError as exc:
                errors.append(f"{'.'.join(path)}: {exc}")

        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        if not cfg.path:
            return jsonify({"ok": False, "errors": ["No config file path is known for this running instance."]}), 500

        try:
            update_config_file(cfg.path, coerced)
        except Exception as exc:  # noqa: BLE001 - surface any write failure to the UI, don't 500 silently
            return jsonify({"ok": False, "errors": [f"Failed to save config.yaml: {exc}"]}), 500

        # Reflect the change immediately in this process's in-memory config
        # so a page refresh shows the new values right away. The running
        # engine's already-constructed components (RiskManager, PaperBroker,
        # MarketCalendar, MarketDataClient) were built from copies of the
        # old values at startup and do NOT pick this up live -- a restart
        # is still required for the change to actually govern trading,
        # which the client surfaces after a successful save.
        for path, value in coerced:
            node = cfg.raw
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value

        return jsonify({"ok": True, "restart_required": True})

    return app


def run_dashboard(cfg: Config, host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    engine = TradingEngine(cfg)
    app = create_app(engine)
    # threaded=True: /api/candidates can take a while (network calls across
    # the whole universe) -- without this, Werkzeug's dev server serves one
    # request at a time and the rest of the dashboard would appear frozen
    # while a candidate scan is running.
    app.run(host=host, port=port, debug=debug, threaded=True)
