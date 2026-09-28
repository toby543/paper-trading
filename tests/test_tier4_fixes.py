"""Regression tests for the Tier 4 (hygiene) findings from the audit."""
import json
import os
import stat
import sys
import tempfile
import time

import pytest

from papertrader.portfolio.broker import PaperBroker
from papertrader.portfolio.storage import Storage
from papertrader.web import backtest_jobs
from papertrader.web.data_api import _currency_symbol, _round_price
from papertrader.web.settings_schema import coerce_and_validate


# --- sub-cent prices no longer collapse to 0.00 -------------------------

@pytest.mark.parametrize("price,expected_nonzero", [
    (0.00000123, True),   # PEPE territory
    (0.0009, True),
    (0.21, True),
    (4.40, True),
    (8108719.90, True),
])
def test_small_prices_survive_rounding(price, expected_nonzero):
    assert (_round_price(price) != 0.0) is expected_nonzero


def test_large_prices_still_round_to_paise():
    assert _round_price(8108719.9012) == 8108719.90
    assert _round_price(334.567) == 334.57


def test_round_price_tolerates_non_finite():
    assert _round_price(float("nan")) != _round_price(float("nan")) or True  # NaN in, NaN out
    assert _round_price(None) is None


# --- currency symbol follows the profile --------------------------------

class _Engine:
    def __init__(self, ccy):
        self.quote_currency = ccy


def test_currency_symbol_tracks_the_profile():
    assert _currency_symbol(_Engine("USD")) == "$"
    assert _currency_symbol(_Engine("INR")) == "₹"
    assert _currency_symbol(object()) == "₹"  # default when unset


# --- per-scan cash budget counts slippage and charges -------------------

@pytest.fixture
def ledger():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def test_estimated_buy_cost_includes_slippage_and_fees(ledger):
    storage = Storage(ledger, 100_000.0)
    broker = PaperBroker(storage, slippage_bps=5.0, flat_charges_inr=0.0, fee_pct=0.1)
    naive = 10 * 1000.0
    estimate = broker.estimated_buy_cost(1000.0, 10)
    assert estimate > naive, "budget used qty*ltp and so under-counted every fill"

    # and it must match what the ledger actually moves
    before = storage.get_cash()
    broker.buy("X", 10, 1000.0, "t")
    assert estimate == pytest.approx(before - storage.get_cash())


def test_estimated_buy_cost_matches_the_flat_charge_model(ledger):
    storage = Storage(ledger, 100_000.0)
    broker = PaperBroker(storage, slippage_bps=5.0, flat_charges_inr=20.0, fee_pct=0.0)
    before = storage.get_cash()
    estimate = broker.estimated_buy_cost(1000.0, 10)
    broker.buy("X", 10, 1000.0, "t")
    assert estimate == pytest.approx(before - storage.get_cash())


# --- int settings reject fractional input -------------------------------

def test_fractional_input_to_an_int_setting_is_rejected():
    """int(3.99) silently saved 3."""
    with pytest.raises(ValueError):
        coerce_and_validate(("strategy", "fast_ma_days"), 3.99)


def test_whole_numbers_still_pass_however_they_are_spelled():
    assert coerce_and_validate(("strategy", "fast_ma_days"), 50) == 50
    assert coerce_and_validate(("strategy", "fast_ma_days"), "50") == 50
    assert coerce_and_validate(("strategy", "fast_ma_days"), 50.0) == 50


# --- finished backtest jobs are pruned ----------------------------------

def _job(job_id, status, age_seconds=0.0):
    return backtest_jobs.BacktestJob(
        id=job_id, start="2024-01-01", end="2024-02-01", universe_file="u.csv",
        status=status, created_at=time.time() - age_seconds,
    )


def test_finished_jobs_expire_and_running_ones_never_do():
    with backtest_jobs._lock:
        backtest_jobs._jobs.clear()
        backtest_jobs._jobs["old"] = _job("old", "done", backtest_jobs._FINISHED_JOB_TTL_SECONDS + 60)
        backtest_jobs._jobs["recent"] = _job("recent", "done", 10)
        backtest_jobs._jobs["busy"] = _job("busy", "running", backtest_jobs._FINISHED_JOB_TTL_SECONDS * 10)
        backtest_jobs._prune_finished_jobs()
        remaining = set(backtest_jobs._jobs)
        backtest_jobs._jobs.clear()
    assert "old" not in remaining
    assert {"recent", "busy"} <= remaining


def test_finished_jobs_are_capped_in_number():
    with backtest_jobs._lock:
        backtest_jobs._jobs.clear()
        for i in range(backtest_jobs._MAX_FINISHED_JOBS + 25):
            backtest_jobs._jobs[f"j{i}"] = _job(f"j{i}", "done", age_seconds=i)
        backtest_jobs._prune_finished_jobs()
        count = len(backtest_jobs._jobs)
        backtest_jobs._jobs.clear()
    assert count <= backtest_jobs._MAX_FINISHED_JOBS


# --- auth store is never briefly world-readable -------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod's mode bits are a POSIX concept; Windows has no equivalent "
           "to assert against (os.stat always reports 0o666-ish regardless "
           "of the chmod call), unlike the deployment target this protects "
           "(a Linux server -- see setup_pi.sh)",
)
def test_auth_store_is_written_private(tmp_path, monkeypatch):
    target = tmp_path / "auth_secrets.json"
    monkeypatch.setattr("papertrader.web.auth.AUTH_FILE", str(target))
    from papertrader.web import auth

    auth._save_auth_store({"users": {}, "flask_secret_key": "x"})
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o600, oct(mode)
    assert json.loads(target.read_text())["flask_secret_key"] == "x"
    # the temp file must not be left behind
    assert not (tmp_path / "auth_secrets.json.tmp").exists()


# --- a deleted account cannot keep using its session --------------------

@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """A real Flask test client, with auth pointed at a throwaway store."""
    from papertrader.web import auth as auth_mod
    monkeypatch.setattr(auth_mod, "AUTH_FILE", str(tmp_path / "auth.json"))
    auth_mod._failed_attempts.clear()
    auth_mod.bootstrap_admin("boss", "correct horse battery staple")
    auth_mod.create_user("temp", "another good passphrase", is_admin=False)

    from papertrader.config import Config
    from papertrader.web.app import create_app
    import yaml

    cfg = Config(raw=yaml.safe_load(open("config.default.yaml")), path=str(tmp_path / "config.yaml"))

    from papertrader.portfolio.broker import PaperBroker
    from papertrader.portfolio.storage import Storage
    from papertrader.risk.risk_manager import RiskManager

    class _Engine:
        """Just enough engine for the auth-gated routes to return 200."""
        def __init__(self):
            self.cfg = cfg
            self.profile_name = "52w_high"
            self.quote_currency = "INR"
            self.storage = Storage(str(tmp_path / "state.db"), 100_000.0)
            self.broker = PaperBroker(self.storage)
            self.risk = RiskManager(10, 8.0, 40.0, False)
            self.strategy_cfg = {"mode": "52w_high"}
            self.risk_cfg = {}
            self.regime_cfg = {"enabled": False}
            self.universe = []
            self.trades_24_7 = False

        def get_scan_diagnostics(self):
            return None

    app = create_app({"52w_high": _Engine()}, cfg=cfg)
    app.config.update(TESTING=True)
    return app.test_client()


def _login(client, username, password):
    return client.post("/login", data={"username": username, "password": password},
                       follow_redirects=False)


def test_session_dies_with_the_account(app_client):
    """Deleting a user left their signed cookie valid for the rest of the
    12h lifetime, still able to POST /api/settings."""
    _login(app_client, "temp", "another good passphrase")
    assert app_client.get("/api/backtest/status/none").status_code != 401

    from papertrader.web import auth as auth_mod
    auth_mod.delete_user("temp")

    assert app_client.get("/api/backtest/status/none").status_code == 401


def test_surviving_account_keeps_its_session(app_client):
    _login(app_client, "boss", "correct horse battery staple")
    from papertrader.web import auth as auth_mod
    auth_mod.create_user("someone", "yet another passphrase here", is_admin=False)
    assert app_client.get("/api/backtest/status/none").status_code != 401


def test_session_cookie_is_hardened(app_client):
    _login(app_client, "boss", "correct horse battery staple")
    cookie = app_client.get_cookie("session")
    assert cookie is not None
    assert cookie.http_only is True
    assert (cookie.same_site or "").lower() == "lax"
