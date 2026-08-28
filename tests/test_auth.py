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


def test_not_configured_before_bootstrap():
    assert auth.is_configured() is False
    assert auth.load_auth_store() is None


def test_bootstrap_admin_creates_store_with_one_admin_unenrolled():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    assert auth.is_configured() is True

    store = auth.load_auth_store()
    assert "nash" in store["users"]
    user = store["users"]["nash"]
    assert user["is_admin"] is True
    assert user["totp_enrolled"] is False
    assert user["totp_secret"] is None
    assert len(store["flask_secret_key"]) >= 32


def test_bootstrap_admin_refuses_if_already_configured():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    with pytest.raises(auth.AuthError):
        auth.bootstrap_admin("someone-else", "another password entirely")


def test_verify_password_rejects_wrong_password_and_unknown_user():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    store = auth.load_auth_store()
    assert auth.verify_password(store, "nash", "wrong password") is False
    assert auth.verify_password(store, "nobody", "correct horse battery staple") is False
    assert auth.verify_password(store, "nash", "correct horse battery staple") is True


def test_first_login_totp_enrollment_flow():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    store = auth.load_auth_store()

    uri = auth.start_totp_enrollment(store, "nash")
    assert "nash" in uri
    pending_secret = store["users"]["nash"]["totp_secret_pending"]
    assert pending_secret

    # Not yet enrolled -- verify_totp must not accept the pending secret's code.
    correct_code = pyotp.TOTP(pending_secret).now()
    assert auth.verify_totp(store, "nash", correct_code) is False

    # Wrong code during confirmation leaves it pending, not enrolled.
    assert auth.confirm_totp_enrollment(store, "nash", "000000") is False
    assert store["users"]["nash"]["totp_enrolled"] is False

    # Right code confirms it.
    assert auth.confirm_totp_enrollment(store, "nash", correct_code) is True
    assert store["users"]["nash"]["totp_enrolled"] is True
    assert store["users"]["nash"]["totp_secret"] == pending_secret
    assert store["users"]["nash"]["totp_secret_pending"] is None

    # Now verify_totp works against the now-active secret.
    assert auth.verify_totp(store, "nash", pyotp.TOTP(pending_secret).now()) is True


def test_start_totp_enrollment_reuses_pending_secret_across_calls():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    store = auth.load_auth_store()
    auth.start_totp_enrollment(store, "nash")
    first_secret = store["users"]["nash"]["totp_secret_pending"]
    auth.start_totp_enrollment(store, "nash")
    assert store["users"]["nash"]["totp_secret_pending"] == first_secret


def test_create_user_requires_existing_store():
    with pytest.raises(auth.AuthError):
        auth.create_user("newuser", "another password entirely", is_admin=False)


def test_create_user_and_duplicate_rejected():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    auth.create_user("teammate", "another password entirely", is_admin=False)
    store = auth.load_auth_store()
    assert "teammate" in store["users"]
    assert store["users"]["teammate"]["is_admin"] is False
    assert store["users"]["teammate"]["totp_enrolled"] is False

    with pytest.raises(auth.AuthError):
        auth.create_user("teammate", "yet another password", is_admin=False)


def test_delete_user():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    auth.create_user("teammate", "another password entirely", is_admin=False)
    auth.delete_user("teammate")
    store = auth.load_auth_store()
    assert "teammate" not in store["users"]


def test_cannot_delete_last_admin():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    with pytest.raises(auth.AuthError):
        auth.delete_user("nash")


def test_can_delete_admin_if_another_admin_remains():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    auth.create_user("otheradmin", "another password entirely", is_admin=True)
    auth.delete_user("nash")  # should not raise: otheradmin still exists
    store = auth.load_auth_store()
    assert "nash" not in store["users"]
    assert "otheradmin" in store["users"]


def test_delete_nonexistent_user_raises():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    with pytest.raises(auth.AuthError):
        auth.delete_user("ghost")


def test_reset_totp_clears_enrollment():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    store = auth.load_auth_store()
    uri = auth.start_totp_enrollment(store, "nash")
    secret = store["users"]["nash"]["totp_secret_pending"]
    auth.confirm_totp_enrollment(store, "nash", pyotp.TOTP(secret).now())
    assert store["users"]["nash"]["totp_enrolled"] is True

    auth.reset_totp("nash")
    store = auth.load_auth_store()
    assert store["users"]["nash"]["totp_enrolled"] is False
    assert store["users"]["nash"]["totp_secret"] is None
    assert store["users"]["nash"]["totp_secret_pending"] is None


def test_is_admin():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    auth.create_user("teammate", "another password entirely", is_admin=False)
    store = auth.load_auth_store()
    assert auth.is_admin(store, "nash") is True
    assert auth.is_admin(store, "teammate") is False
    assert auth.is_admin(store, "nobody") is False


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
