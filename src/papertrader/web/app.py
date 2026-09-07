"""Local read-only web dashboard for the paper trading account.

Normally runs as its own process, reading the same SQLite ledger the
trading engine (`python main.py run`) writes to -- it does not place
trades itself; it only displays live-marked positions, P&L, the equity
curve and the trade history, refreshing itself every few seconds in
the browser. `create_app()` takes a dict of already-constructed
TradingEngine instances (one per configured profile) so it can also be
embedded in the same process as the engine(s) (see `python main.py
serve`, cli.py) sharing those instances instead of constructing new
ones.

In multi_profile_mode all engines in the dict are already running
their own trading loop concurrently in separate threads -- the
dashboard's profile dropdown only changes which engine's data is
*displayed* (get_current_engine()), it never starts, stops, or
reinitializes an engine. Outside multi_profile_mode the dict holds a
single engine, and switching profiles instead calls
TradingEngine.reload_profile() to actually swap out that one engine's
ledger/strategy.
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from . import auth
from ..config import Config
from ..config_editor import update_config_file
from ..engine.scheduler import TradingEngine
from .backtest_jobs import get_job, start_backtest_job
from .data_api import build_candidates, build_equity_curve, build_index_charts, build_performance_comparison, build_summary, build_trades
from .filters import indian_currency
from .settings_schema import EDITABLE_SETTINGS, coerce_and_validate, get_value

_HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)


def _safe_next_path(value: str | None) -> str | None:
    """Only accept a same-site relative path for post-login redirect.
    `next` is attacker-controllable (anyone can send someone a
    `/login?next=...` link), so without this an absolute or
    protocol-relative URL here would be a classic open-redirect --
    e.g. `next=https://evil.example/phish` or `next=//evil.example`."""
    if not value or not value.startswith("/") or value.startswith("//"):
        return None
    return value


def create_app(engines: dict[str, TradingEngine], cfg: Config | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=os.path.join(_HERE, "static"),
    )
    app.jinja_env.filters["inr"] = indian_currency
    app.config["ENGINES"] = engines
    cfg = cfg or next(iter(engines.values())).cfg

    def get_current_engine() -> TradingEngine:
        """The engine whose data the dashboard is currently displaying.
        In multi_profile_mode every engine in `engines` is already
        running its own trading loop; this only picks which one's data
        to show, matching the active_profile dropdown selection."""
        active = cfg.get("active_profile", default="52w_high")
        return engines.get(active) or next(iter(engines.values()))

    def get_store():
        # Re-read on every call rather than caching at startup: an admin
        # creating/deleting a user via /admin/users must take effect
        # immediately for other sessions, without a server restart.
        return auth.load_auth_store()

    boot_store = get_store()
    if boot_store is None:
        # Backward-compatible: pure-localhost use never required a login.
        # A random per-process key still gets a real (if throwaway)
        # session secret rather than Flask's default of none.
        app.secret_key = os.urandom(32)
        log.warning(
            "No authentication configured (no data/auth_secrets.json) -- this dashboard is "
            "UNPROTECTED. Run `python main.py setup-auth` before exposing it beyond localhost."
        )
    else:
        app.secret_key = boot_store["flask_secret_key"]
    app.permanent_session_lifetime = timedelta(hours=12)

    def login_required(view):
        """For page routes: redirect to the login page (the browser is
        expecting an HTML response either way)."""
        @wraps(view)
        def wrapped(*args, **kwargs):
            if get_store() is None or session.get("authenticated"):
                return view(*args, **kwargs)
            return redirect(url_for("login", next=request.path))
        return wrapped

    def api_login_required(view):
        """For /api/* routes: a redirect would hand the frontend's
        fetch() an HTML login page instead of JSON, which just fails to
        parse -- return a 401 JSON error instead so the client can
        actually react to it."""
        @wraps(view)
        def wrapped(*args, **kwargs):
            if get_store() is None or session.get("authenticated"):
                return view(*args, **kwargs)
            return jsonify({"ok": False, "error": "Not authenticated. Please log in again."}), 401
        return wrapped

    def admin_required(view):
        """User-management pages: authenticated AND the logged-in
        account is flagged is_admin. Deliberately redirects to the
        ordinary dashboard rather than the login page for a non-admin
        who's otherwise logged in -- they're not unauthenticated, just
        not allowed here."""
        @wraps(view)
        def wrapped(*args, **kwargs):
            store = get_store()
            if store is None:
                return redirect(url_for("index"))
            if not session.get("authenticated"):
                return redirect(url_for("login", next=request.path))
            if not auth.is_admin(store, session.get("username", "")):
                return redirect(url_for("index"))
            return view(*args, **kwargs)
        return wrapped

    @app.get("/login")
    def login():
        store = get_store()
        if store is None or session.get("authenticated"):
            return redirect(url_for("index"))
        return render_template("login.html", error=None)

    @app.post("/login")
    def login_submit():
        store = get_store()
        if store is None:
            return redirect(url_for("index"))
        if auth.is_locked_out(request.remote_addr):
            return render_template(
                "login.html",
                error="Too many failed attempts. Wait 15 minutes before trying again.",
            ), 429

        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not auth.verify_password(store, username, password):
            auth.record_failed_attempt(request.remote_addr)
            return render_template("login.html", error="Incorrect username or password."), 401

        auth.clear_failed_attempts(request.remote_addr)
        _complete_login(username)
        return redirect(_safe_next_path(request.args.get("next")) or url_for("index"))

    def _complete_login(username: str) -> None:
        session.permanent = True
        session["authenticated"] = True
        session["username"] = username

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    def _users_view(store: dict) -> list[dict]:
        return [
            {"username": u, "is_admin": rec["is_admin"]}
            for u, rec in sorted(store["users"].items())
        ]

    @app.get("/admin/users")
    @admin_required
    def admin_users():
        store = get_store()
        return render_template(
            "admin_users.html",
            users=_users_view(store),
            current_username=session.get("username"),
            error=request.args.get("error"),
            message=request.args.get("message"),
        )

    @app.post("/admin/users")
    @admin_required
    def admin_users_create():
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        is_admin_flag = request.form.get("is_admin") == "on"
        if not username or len(password) < 8:
            return redirect(url_for("admin_users", error="Username is required and password must be at least 8 characters."))
        try:
            auth.create_user(username, password, is_admin=is_admin_flag)
        except auth.AuthError as exc:
            return redirect(url_for("admin_users", error=str(exc)))
        return redirect(url_for("admin_users", message=f"Created '{username}'."))

    @app.post("/admin/users/<username>/delete")
    @admin_required
    def admin_users_delete(username: str):
        if username == session.get("username"):
            return redirect(url_for("admin_users", error="You can't delete your own account while logged in as it."))
        try:
            auth.delete_user(username)
        except auth.AuthError as exc:
            return redirect(url_for("admin_users", error=str(exc)))
        return redirect(url_for("admin_users", message=f"Deleted '{username}'."))

    @app.get("/")
    @login_required
    def index():
        store = get_store()
        active_profile = cfg.get("active_profile", default="52w_high")
        active_strategy_mode = cfg.get_profile_strategy_mode(active_profile)
        active_state_file = cfg.get_profile_state_file(active_profile)
        active_state_file_short = os.path.basename(active_state_file)

        # Build strategy config with profile-specific mode
        strategy = cfg.get("strategy", default={})
        strategy = {**strategy, "mode": active_strategy_mode}

        return render_template(
            "index.html",
            is_admin=bool(store and auth.is_admin(store, session.get("username", ""))),
            slippage_bps=cfg.get("execution", "slippage_bps", default=5.0),
            flat_charges_inr=cfg.get("execution", "flat_charges_inr", default=20.0),
            strategy=strategy,
            risk=cfg.get("risk", default={}),
            regime=cfg.get("regime", default={}),
            account=cfg.get("account", default={}),
            universe_cfg=cfg.get("universe", default={}),
            execution=cfg.get("execution", default={}),
            engine_cfg=cfg.get("engine", default={}),
            data_source=cfg.get("data_source", default={}),
            logging_cfg=cfg.get("logging", default={}),
            profiles=cfg.list_profiles(),
            active_profile=active_profile,
            active_state_file=active_state_file,
            active_state_file_short=active_state_file_short,
            multi_profile_mode=cfg.is_multi_profile_mode(),
        )

    @app.get("/api/summary")
    @api_login_required
    def api_summary():
        return jsonify(build_summary(get_current_engine()))

    @app.get("/api/trades")
    @api_login_required
    def api_trades():
        limit = request.args.get("limit", default=100, type=int)
        return jsonify(build_trades(get_current_engine(), limit=limit))

    @app.get("/api/equity_curve")
    @api_login_required
    def api_equity_curve():
        limit = request.args.get("limit", default=500, type=int)
        return jsonify(build_equity_curve(get_current_engine(), limit=limit))

    @app.get("/api/indices")
    @api_login_required
    def api_indices():
        period = request.args.get("period", default="6mo")
        return jsonify(build_index_charts(get_current_engine(), period=period))

    @app.get("/api/performance")
    @api_login_required
    def api_performance():
        return jsonify(build_performance_comparison(get_current_engine()))

    @app.get("/api/candidates")
    @api_login_required
    def api_candidates():
        # Expensive (network calls across the whole universe) -- the
        # dashboard triggers this on demand (a "Scan Now" button), never
        # on the regular auto-refresh poll. threaded=True on app.run()
        # below keeps this from blocking the rest of the dashboard while
        # it runs.
        limit = request.args.get("limit", default=20, type=int)
        return jsonify(build_candidates(get_current_engine(), limit=limit))

    @app.post("/api/backtest/run")
    @api_login_required
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
    @api_login_required
    def api_backtest_status(job_id: str):
        job = get_job(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Unknown or expired job id."}), 404
        return jsonify({"ok": True, "job": job})

    @app.get("/api/settings")
    @api_login_required
    def api_get_settings():
        fields = []
        for spec in EDITABLE_SETTINGS:
            fields.append({**spec, "path": list(spec["path"]), "value": get_value(cfg.raw, spec["path"])})
        return jsonify({"fields": fields})

    @app.post("/api/settings")
    @api_login_required
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

    @app.post("/api/reload-config")
    @api_login_required
    def api_reload_config():
        """Hot-reload configuration from disk without restarting the service.
        Updates every running engine's components with the new config
        values (there can be more than one in multi_profile_mode)."""
        if not cfg.reload():
            return jsonify({"ok": False, "error": "No changes to config file"}), 400

        # Reload components that can be updated live (don't affect in-flight trades)
        try:
            for eng in engines.values():
                # Copy -- cfg.get() returns the live dict inside cfg.raw,
                # and every engine shares this same Config object, so
                # mutating it in place would leak one engine's "mode"
                # into all the others.
                strategy_cfg = dict(cfg.get("strategy", default={}))
                strategy_cfg["mode"] = cfg.get_profile_strategy_mode(eng.profile_name)
                # Under the engine's own lock so this can't interleave with
                # its background run_forever() loop or a reload_profile().
                with eng._state_lock:
                    eng.strategy_cfg = strategy_cfg
                    eng.risk_cfg = cfg.get("risk", default={})
                    eng.regime_cfg = cfg.get("regime", default={})

                    # Update RiskManager with new risk settings
                    eng.risk.max_open_positions = cfg.get("risk", "max_open_positions", default=10)
                    eng.risk.position_size_pct_of_equity = cfg.get("risk", "position_size_pct_of_equity", default=8.0)
                    eng.risk.max_cash_deployed_per_scan_pct = cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0)

                    # Update data source timeout settings
                    eng.data.timeout = cfg.get("data_source", "request_timeout_seconds", default=10)

            log.info("Configuration hot-reloaded successfully. Changes applied to %d running engine(s).", len(engines))
            return jsonify({
                "ok": True,
                "message": "Configuration reloaded successfully. Risk management and strategy settings are now active.",
                "note": "Each profile's strategy_mode (in config.yaml) is picked up live too. "
                        "Only the standalone strategy.mode field edited via 'Edit Settings' has no effect while any "
                        "profile is configured, since every profile always uses its own strategy_mode instead."
            }), 200
        except Exception as e:
            log.error("Failed to apply reloaded config to engine(s): %s", e)
            return jsonify({"ok": False, "error": f"Failed to apply config: {e}"}), 500

    @app.post("/api/set-profile")
    @api_login_required
    def api_set_profile():
        """Switch the active profile.

        In multi_profile_mode every profile's engine is already running
        and trading independently in its own background thread -- this
        only changes which one's data the dashboard *displays*. Outside
        multi_profile_mode there is a single engine, and this actually
        reloads it (new ledger, new strategy) to match the new profile."""
        payload = request.get_json(silent=True) or {}
        profile_name = payload.get("profile")

        if not profile_name:
            return jsonify({"ok": False, "error": "Profile name is required"}), 400

        profiles = cfg.get("profiles", default={})
        if profile_name not in profiles:
            return jsonify({"ok": False, "error": f"Unknown profile: {profile_name}"}), 400

        if not cfg.path:
            return jsonify({"ok": False, "error": "No config file path is known"}), 500

        try:
            from ..config_editor import update_config_file
            profile_cfg = profiles[profile_name]
            profile_strategy = profile_cfg.get("strategy_mode", "52w_high")
            display_name = profile_cfg.get("display_name", profile_name)

            # Persist which profile is active (view selection + fallback default)
            update_config_file(cfg.path, [(["active_profile"], profile_name)])
            cfg.set_active_profile(profile_name)

            if cfg.is_multi_profile_mode():
                # All profiles are already trading in parallel -- this is
                # purely a view switch, no engine touched.
                log.info("Switched dashboard view to profile: %s (all profiles trading in parallel)", profile_name)
                return jsonify({
                    "ok": True,
                    "message": f"Now viewing {display_name} (strategy: {profile_strategy}).",
                    "restart_required": False,
                    "note": "All 3 profiles are trading simultaneously -- this only changed which one you're viewing."
                }), 200

            # Single-profile mode: actually reload the one running engine
            # to the new profile's ledger/strategy.
            engine = next(iter(engines.values()), None)
            restart_required = True
            reload_msg = "Restart the engine to load the new profile's ledger, positions, and strategy."

            if engine:
                try:
                    old_key = engine.profile_name
                    engine.reload_profile()
                    # Engines is keyed by profile_name; re-key it to match.
                    # Insert the new key before removing the old one so a
                    # concurrent request's get_current_engine() never sees
                    # the dict transiently empty.
                    if engine.profile_name != old_key:
                        engines[engine.profile_name] = engine
                        del engines[old_key]
                    restart_required = False
                    reload_msg = "Engine reloaded automatically with new profile, ledger, and strategy."
                    log.info("Engine reloaded for profile: %s (strategy: %s)", profile_name, profile_strategy)
                except Exception as e:
                    log.exception("Could not auto-reload engine: %s (will need manual restart)", e)
                    # Even if reload fails, config was updated, so notify user

            log.info("Switched to profile: %s (strategy: %s)", profile_name, profile_strategy)
            return jsonify({
                "ok": True,
                "message": f"Switched to {display_name} profile (strategy: {profile_strategy}).",
                "restart_required": restart_required,
                "note": reload_msg
            }), 200
        except Exception as e:
            log.error("Failed to switch profile: %s", e)
            return jsonify({"ok": False, "error": f"Failed to switch profile: {e}"}), 500

    return app


def run_dashboard(cfg: Config, host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Read-only dashboard, no trading loop of its own: expects a
    separate `python main.py run` process to be writing to the same
    ledger file(s). Builds one (non-running) TradingEngine per profile
    in multi_profile_mode so it can read each profile's own ledger."""
    if cfg.is_multi_profile_mode():
        engines = {name: TradingEngine(cfg, profile_name=name) for name in cfg.list_profiles()}
    else:
        eng = TradingEngine(cfg)
        engines = {eng.profile_name: eng}
    app = create_app(engines, cfg)
    # threaded=True: /api/candidates can take a while (network calls across
    # the whole universe) -- without this, Werkzeug's dev server serves one
    # request at a time and the rest of the dashboard would appear frozen
    # while a candidate scan is running.
    app.run(host=host, port=port, debug=debug, threaded=True)
