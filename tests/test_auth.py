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


def test_bootstrap_admin_creates_store_with_one_admin():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    assert auth.is_configured() is True

    store = auth.load_auth_store()
    assert "nash" in store["users"]
    user = store["users"]["nash"]
    assert user["is_admin"] is True
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


def test_create_user_requires_existing_store():
    with pytest.raises(auth.AuthError):
        auth.create_user("newuser", "another password entirely", is_admin=False)


def test_create_user_and_duplicate_rejected():
    auth.bootstrap_admin("nash", "correct horse battery staple")
    auth.create_user("teammate", "another password entirely", is_admin=False)
    store = auth.load_auth_store()
    assert "teammate" in store["users"]
    assert store["users"]["teammate"]["is_admin"] is False

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
