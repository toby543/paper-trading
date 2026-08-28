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

import logging
import os
from datetime import timedelta
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from . import auth, qr
from ..config import Config
from ..config_editor import update_config_file
from ..engine.scheduler import TradingEngine
from .backtest_jobs import get_job, start_backtest_job
from .data_api import build_candidates, build_equity_curve, build_summary, build_trades
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


def create_app(engine: TradingEngine) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=os.path.join(_HERE, "static"),
    )
    app.jinja_env.filters["inr"] = indian_currency
    app.config["ENGINE"] = engine
    cfg = engine.cfg

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
        pending = session.get("pending_user")
        if pending and pending in store["users"]:
            return _render_pending_login_step(store, pending, error=None)
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

        session["pending_user"] = username
        session["pending_next"] = _safe_next_path(request.args.get("next"))
        return _render_pending_login_step(store, username, error=None)

    @app.post("/login/verify")
    def login_verify():
        """Step 2 for an already-enrolled account: check the 6-digit code."""
        pending = session.get("pending_user")
        store = get_store()
        if not pending or store is None or pending not in store["users"]:
            return redirect(url_for("login"))
        if auth.is_locked_out(request.remote_addr):
            return render_template("verify_2fa.html", error="Too many failed attempts. Wait 15 minutes before trying again."), 429

        code = request.form.get("code", "")
        if not auth.verify_totp(store, pending, code):
            auth.record_failed_attempt(request.remote_addr)
            return render_template("verify_2fa.html", error="Incorrect code."), 401

        auth.clear_failed_attempts(request.remote_addr)
        _complete_login(pending)
        return redirect(session.pop("pending_next", None) or url_for("index"))

    @app.post("/login/enroll")
    def login_enroll():
        """Step 2 the FIRST time this account ever logs in: confirm the
        user actually saved the just-shown TOTP secret before turning
        2FA on for their account."""
        pending = session.get("pending_user")
        store = get_store()
        if not pending or store is None or pending not in store["users"]:
            return redirect(url_for("login"))
        if auth.is_locked_out(request.remote_addr):
            return render_template("login.html", error="Too many failed attempts. Wait 15 minutes before trying again."), 429

        code = request.form.get("code", "")
        if not auth.confirm_totp_enrollment(store, pending, code):
            auth.record_failed_attempt(request.remote_addr)
            # Wrong code: keep the SAME pending secret (don't regenerate)
            # so the user can just retry against the QR/setup key they
            # already have open, instead of it becoming a moving target.
            uri = auth.start_totp_enrollment(store, pending)
            return render_template(
                "enroll_2fa.html", error="Incorrect code -- try again.",
                totp_uri=uri, totp_secret=store["users"][pending]["totp_secret_pending"], username=pending,
                qr_data_uri=qr.totp_qr_data_uri(uri),
            ), 401

        auth.clear_failed_attempts(request.remote_addr)
        _complete_login(pending)
        return redirect(session.pop("pending_next", None) or url_for("index"))

    def _render_pending_login_step(store: dict, username: str, error: str | None):
        user = store["users"][username]
        if user["totp_enrolled"]:
            return render_template("verify_2fa.html", error=error)
        uri = auth.start_totp_enrollment(store, username)
        return render_template(
            "enroll_2fa.html", error=error,
            totp_uri=uri, totp_secret=store["users"][username]["totp_secret_pending"], username=username,
            qr_data_uri=qr.totp_qr_data_uri(uri),
        )

    def _complete_login(username: str) -> None:
        session.pop("pending_user", None)
        session.permanent = True
        session["authenticated"] = True
        session["username"] = username

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    def _users_view(store: dict) -> list[dict]:
        return [
            {"username": u, "is_admin": rec["is_admin"], "totp_enrolled": rec["totp_enrolled"]}
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
        return redirect(url_for("admin_users", message=f"Created '{username}'. They'll set up 2FA on their own first login."))

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

    @app.post("/admin/users/<username>/reset-2fa")
    @admin_required
    def admin_users_reset_2fa(username: str):
        try:
            auth.reset_totp(username)
        except auth.AuthError as exc:
            return redirect(url_for("admin_users", error=str(exc)))
        return redirect(url_for("admin_users", message=f"Reset 2FA for '{username}' -- they'll set it up again on their next login."))

    @app.get("/")
    @login_required
    def index():
        store = get_store()
        return render_template(
            "index.html",
            is_admin=bool(store and auth.is_admin(store, session.get("username", ""))),
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
    @api_login_required
    def api_summary():
        return jsonify(build_summary(engine))

    @app.get("/api/trades")
    @api_login_required
    def api_trades():
        limit = request.args.get("limit", default=100, type=int)
        return jsonify(build_trades(engine, limit=limit))

    @app.get("/api/equity_curve")
    @api_login_required
    def api_equity_curve():
        limit = request.args.get("limit", default=500, type=int)
        return jsonify(build_equity_curve(engine, limit=limit))

    @app.get("/api/candidates")
    @api_login_required
    def api_candidates():
        # Expensive (network calls across the whole universe) -- the
        # dashboard triggers this on demand (a "Scan Now" button), never
        # on the regular auto-refresh poll. threaded=True on app.run()
        # below keeps this from blocking the rest of the dashboard while
        # it runs.
        limit = request.args.get("limit", default=20, type=int)
        return jsonify(build_candidates(engine, limit=limit))

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

    return app


def run_dashboard(cfg: Config, host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    engine = TradingEngine(cfg)
    app = create_app(engine)
    # threaded=True: /api/candidates can take a while (network calls across
    # the whole universe) -- without this, Werkzeug's dev server serves one
    # request at a time and the rest of the dashboard would appear frozen
    # while a candidate scan is running.
    app.run(host=host, port=port, debug=debug, threaded=True)
