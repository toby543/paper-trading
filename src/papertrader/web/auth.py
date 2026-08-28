"""Multi-user login + self-service TOTP two-factor auth for the dashboard.

The dashboard has no built-in access control otherwise -- fine for
localhost-only use, but the Edit Settings panel can change the live
strategy config and the Backtest panel can kick off expensive scans, so
anything reachable from the internet needs a real gate in front of it.

All accounts live in `data/auth_secrets.json` (path resolved the same way
as everything else in config.py) -- never in config.yaml or anywhere else
that gets committed to git, since this repo is public. That file is
`.gitignore`'d explicitly; treat it like any other secret and never check
it in.

`python main.py setup-auth` bootstraps the first (admin) account with a
username + password only -- no TOTP secret yet. Two-factor is enrolled
the first time that account logs in from the browser (see app.py's
/login/enroll route): the server generates a secret, shows it once, and
only marks the account as TOTP-enrolled after the user proves they
actually saved it by entering one correct code. The same self-enrollment
flow applies to every user an admin creates afterwards -- an admin never
needs to see or transmit anyone else's TOTP secret.

If AUTH_FILE doesn't exist at all, the dashboard runs exactly as it
always has (unauthenticated) for backward compatibility with pure-
localhost use -- but `create_app()` prints a loud warning at startup so
it's never silently left open by accident.
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
# personal/small-team dashboard.
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 15 * 60
_failed_attempts: dict[str, list[float]] = {}


class AuthError(Exception):
    """Raised for user-facing setup mistakes (duplicate username, deleting
    the last admin, etc.) -- distinct from a plain bug, so callers can
    show the message directly rather than a generic 500."""


# ---- store I/O -----------------------------------------------------
def is_configured() -> bool:
    store = load_auth_store()
    return bool(store and store.get("users"))


def load_auth_store() -> dict | None:
    if not os.path.exists(AUTH_FILE):
        return None
    with open(AUTH_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_auth_store(store: dict) -> None:
    os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
    tmp_path = AUTH_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp_path, AUTH_FILE)
    # 0600: this file holds password hashes and TOTP secrets.
    try:
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass  # best-effort on platforms where this doesn't apply (e.g. Windows)


def bootstrap_admin(username: str, password: str) -> dict:
    """Create AUTH_FILE from scratch with a single admin account (no
    TOTP secret yet -- enrolled on first web login). Refuses if the file
    already has any users; the caller (cmd_setup_auth --force) is
    responsible for deciding whether to wipe it first."""
    if is_configured():
        raise AuthError("Authentication is already configured.")
    store = {
        "flask_secret_key": secrets.token_hex(32),
        "users": {},
    }
    _add_user(store, username, password, is_admin=True)
    _save_auth_store(store)
    return store


def _add_user(store: dict, username: str, password: str, is_admin: bool) -> None:
    if username in store["users"]:
        raise AuthError(f"User '{username}' already exists.")
    store["users"][username] = {
        "password_hash": generate_password_hash(password),
        "is_admin": is_admin,
        "totp_secret": None,          # set once enrollment is confirmed
        "totp_secret_pending": None,  # set while enrollment is in progress
        "totp_enrolled": False,
    }


def create_user(username: str, password: str, is_admin: bool = False) -> None:
    store = load_auth_store()
    if store is None:
        raise AuthError("Run `python main.py setup-auth` first to create the admin account.")
    _add_user(store, username, password, is_admin)
    _save_auth_store(store)


def delete_user(username: str) -> None:
    store = load_auth_store()
    if store is None or username not in store["users"]:
        raise AuthError(f"User '{username}' does not exist.")
    remaining_admins = [
        u for u, rec in store["users"].items() if rec["is_admin"] and u != username
    ]
    if store["users"][username]["is_admin"] and not remaining_admins:
        raise AuthError("Can't delete the last remaining admin account.")
    del store["users"][username]
    _save_auth_store(store)


def reset_totp(username: str) -> None:
    """Admin-triggered recovery: clear a user's 2FA enrollment so they go
    through the enrollment flow again on their next login (e.g. they
    lost the phone their authenticator app was on)."""
    store = load_auth_store()
    if store is None or username not in store["users"]:
        raise AuthError(f"User '{username}' does not exist.")
    user = store["users"][username]
    user["totp_secret"] = None
    user["totp_secret_pending"] = None
    user["totp_enrolled"] = False
    _save_auth_store(store)


# ---- login: password check -> TOTP verify or first-time enrollment --
def verify_password(store: dict, username: str, password: str) -> bool:
    user = store["users"].get(username)
    if user is None:
        # Still run a hash comparison against a dummy hash so a
        # nonexistent-username response takes the same time as a
        # wrong-password one, rather than leaking which usernames exist
        # via a timing difference.
        check_password_hash(generate_password_hash("decoy"), password)
        return False
    return check_password_hash(user["password_hash"], password)


def verify_totp(store: dict, username: str, code: str) -> bool:
    user = store["users"].get(username)
    if user is None or not user["totp_enrolled"] or not user["totp_secret"]:
        return False
    # valid_window=1 tolerates ~30s of clock drift between the server and
    # the phone running the authenticator app, without widening the
    # acceptance window enough to meaningfully weaken the code.
    return pyotp.TOTP(user["totp_secret"]).verify(code, valid_window=1)


def start_totp_enrollment(store: dict, username: str) -> str:
    """First-login 2FA setup: generate a pending secret (reusing one
    already in progress rather than rotating it on every page load, so a
    user who takes a minute to scan/enter it doesn't get a moving
    target) and return the otpauth:// URI to show them."""
    user = store["users"][username]
    if not user["totp_secret_pending"]:
        user["totp_secret_pending"] = pyotp.random_base32()
        _save_auth_store(store)
    return pyotp.TOTP(user["totp_secret_pending"]).provisioning_uri(
        name=username, issuer_name="Momentum Desk"
    )


def confirm_totp_enrollment(store: dict, username: str, code: str) -> bool:
    """Verify the user actually saved the pending secret correctly
    before turning 2FA on for their account. Returns False (secret left
    pending, so they can just retry the same QR/setup key) on a wrong
    code."""
    user = store["users"][username]
    pending = user.get("totp_secret_pending")
    if not pending or not pyotp.TOTP(pending).verify(code, valid_window=1):
        return False
    user["totp_secret"] = pending
    user["totp_secret_pending"] = None
    user["totp_enrolled"] = True
    _save_auth_store(store)
    return True


def is_admin(store: dict, username: str) -> bool:
    user = store["users"].get(username)
    return bool(user and user["is_admin"])


# ---- per-IP brute-force throttle ------------------------------------
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
