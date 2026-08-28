import pyotp
import pytest

from papertrader.web import auth


@pytest.fixture(autouse=True)
def isolated_auth_file(tmp_path, monkeypatch):
    """Point AUTH_FILE at a throwaway path per test, and reset the
    in-memory lockout state (module-level, otherwise leaks between tests)."""
    monkeypatch.setattr(auth, "AUTH_FILE", str(tmp_path / "auth_secrets.json"))
    auth._failed_attempts.clear()
    yield


def test_not_configured_before_setup():
    assert auth.is_configured() is False
    assert auth.load_auth_secrets() is None


def test_setup_auth_writes_file_and_returns_totp_uri():
    record = auth.setup_auth("nash", "correct horse battery staple", account_label="tester")
    assert auth.is_configured() is True
    assert "totp_uri" in record
    assert "tester" in record["totp_uri"]

    loaded = auth.load_auth_secrets()
    assert loaded["username"] == "nash"
    assert loaded["totp_secret"] == record["totp_secret"]
    assert "flask_secret_key" in loaded
    assert len(loaded["flask_secret_key"]) >= 32  # hex-encoded 32 bytes -> 64 chars, sanity floor


def test_verify_credentials_requires_username_password_and_code():
    record = auth.setup_auth("nash", "correct horse battery staple")
    correct_code = pyotp.TOTP(record["totp_secret"]).now()

    assert auth.verify_credentials("wrong-user", "correct horse battery staple", correct_code, record) is False
    assert auth.verify_credentials("nash", "wrong password", correct_code, record) is False
    assert auth.verify_credentials("nash", "correct horse battery staple", "000000", record) is False
    assert auth.verify_credentials("nash", "correct horse battery staple", correct_code, record) is True


def test_lockout_after_max_failed_attempts():
    ip = "203.0.113.9"
    for _ in range(auth._MAX_ATTEMPTS):
        assert auth.is_locked_out(ip) is False
        auth.record_failed_attempt(ip)
    assert auth.is_locked_out(ip) is True


def test_clear_failed_attempts_lifts_lockout():
    ip = "203.0.113.10"
    for _ in range(auth._MAX_ATTEMPTS):
        auth.record_failed_attempt(ip)
    assert auth.is_locked_out(ip) is True
    auth.clear_failed_attempts(ip)
    assert auth.is_locked_out(ip) is False


def test_lockout_is_per_source_address():
    ip_a, ip_b = "203.0.113.11", "203.0.113.12"
    for _ in range(auth._MAX_ATTEMPTS):
        auth.record_failed_attempt(ip_a)
    assert auth.is_locked_out(ip_a) is True
    assert auth.is_locked_out(ip_b) is False
