"""Login + TOTP two-factor authentication for the web dashboard.

The dashboard has no built-in access control otherwise -- fine for
localhost-only use, but the Edit Settings panel can change the live
strategy config and the Backtest panel can kick off expensive scans, so
anything reachable from the internet needs a real gate in front of it.

Credentials live in `data/auth_secrets.json` (path resolved the same way
as everything else in config.py), generated once via `python main.py
setup-auth` -- never in config.yaml or anywhere else that gets committed
to git, since this repo is public. That file is `.gitignore`'d
explicitly; treat it like any other secret and never check it in.

If that file doesn't exist, the dashboard runs exactly as it always has
(unauthenticated) for backward compatibility with pure-localhost use --
but `create_app()` prints a loud warning at startup so it's never silently
left open by accident.
"""
from __future__ import annotations

import json
import os
import secrets
import time

import pyotp
from werkzeug.security import check_password_hash, generate_password_hash

from ..config import REPO_ROOT

AUTH_FILE = os.path.join(REPO_ROOT, "data", "auth_secrets.json")

# Simple in-memory brute-force throttle: the Flask dev server this app
# runs on has no rate limiting of its own, and a login form is exactly
# the kind of endpoint automated scanners probe. Keyed by source IP;
# reset on process restart, which is an acceptable tradeoff for a
# personal single-user dashboard.
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 15 * 60
_failed_attempts: dict[str, list[float]] = {}


def is_configured() -> bool:
    return os.path.exists(AUTH_FILE)


def load_auth_secrets() -> dict | None:
    if not is_configured():
        return None
    with open(AUTH_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_auth(username: str, password: str, account_label: str = "papertrader") -> dict:
    """Generate a new password hash + TOTP secret + Flask session secret
    key, write them to AUTH_FILE, and return the record (including the
    otpauth:// URI the caller should show the user once, for adding to
    an authenticator app)."""
    totp_secret = pyotp.random_base32()
    record = {
        "username": username,
        "password_hash": generate_password_hash(password),
        "totp_secret": totp_secret,
        "flask_secret_key": secrets.token_hex(32),
    }
    os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
    with open(AUTH_FILE, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    # 0600: this file holds a password hash and a TOTP secret.
    try:
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass  # best-effort on platforms where this doesn't apply (e.g. Windows)

    uri = pyotp.TOTP(totp_secret).provisioning_uri(name=account_label, issuer_name="Momentum Desk")
    return {**record, "totp_uri": uri}


def _throttle_key(remote_addr: str | None) -> str:
    return remote_addr or "unknown"


def is_locked_out(remote_addr: str | None) -> bool:
    key = _throttle_key(remote_addr)
    now = time.time()
    attempts = [t for t in _failed_attempts.get(key, []) if now - t < _LOCKOUT_SECONDS]
    _failed_attempts[key] = attempts
    return len(attempts) >= _MAX_ATTEMPTS


def record_failed_attempt(remote_addr: str | None) -> None:
    key = _throttle_key(remote_addr)
    _failed_attempts.setdefault(key, []).append(time.time())


def clear_failed_attempts(remote_addr: str | None) -> None:
    _failed_attempts.pop(_throttle_key(remote_addr), None)


def verify_credentials(username: str, password: str, totp_code: str, secrets_record: dict) -> bool:
    # Constant-time comparison for the username too -- it's paired here
    # with a hashed password + TOTP check, so there's no reason to let a
    # plain `==` leak timing information about it for free.
    if not secrets.compare_digest(username, secrets_record.get("username", "")):
        return False
    if not check_password_hash(secrets_record["password_hash"], password):
        return False
    totp = pyotp.TOTP(secrets_record["totp_secret"])
    # valid_window=1 tolerates ~30s of clock drift between the server and
    # the phone running the authenticator app, without widening the
    # acceptance window enough to meaningfully weaken the code.
    return totp.verify(totp_code, valid_window=1)
