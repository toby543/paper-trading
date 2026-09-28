"""Edit Settings must only offer fields that govern the viewed profile.

There is no crypto strategy in this codebase any more (the crypto profiles
and strategies were removed), but the is_crypto/CRYPTO_INERT_PATHS
mechanism itself is kept as generic, profile-category-driven plumbing in
case a crypto category is reintroduced -- these tests exercise that
mechanism directly with a synthetic "is_crypto=True" profile rather than
naming a real strategy mode."""
from papertrader.web.settings_schema import CRYPTO_INERT_PATHS, applies_to_profile


def test_crypto_profile_hides_nse_only_globals():
    """A crypto profile would set its own universe, trade 24/7, and route
    quotes elsewhere -- so the NSE universe list, the NSE session window
    and the NSE/Yahoo source pair would govern nothing it does."""
    for path in [("universe", "file"), ("engine", "market_open"),
                 ("engine", "market_close"), ("data_source", "preferred"),
                 ("data_source", "fallback")]:
        assert applies_to_profile(path, "52w_high", is_crypto=True) is False


def test_equity_profile_still_shows_those_globals():
    for path in CRYPTO_INERT_PATHS:
        assert applies_to_profile(path, "52w_high", is_crypto=False) is True


def test_shared_scheduling_and_timeout_still_apply_to_crypto():
    """These live in the same groups as the hidden fields but are read by
    every profile's engine, so hiding the group wholesale would be wrong."""
    for path in [("engine", "scan_interval_minutes"),
                 ("engine", "exit_check_interval_minutes"),
                 ("data_source", "request_timeout_seconds"),
                 ("logging", "level")]:
        assert applies_to_profile(path, "52w_high", is_crypto=True) is True


def test_same_strategy_mode_differs_by_profile_category():
    """Mode alone cannot decide this -- only the profile's crypto-ness
    can, since the same mode could in principle run under either."""
    path = ("engine", "market_open")
    assert applies_to_profile(path, "consolidation_breakout", is_crypto=True) is False
    assert applies_to_profile(path, "consolidation_breakout", is_crypto=False) is True


def test_52w_high_only_fields_hidden_from_other_modes():
    for path in [("strategy", "proximity_to_52w_high_pct"),
                 ("strategy", "min_relative_strength_pct"),
                 ("strategy", "volume_confirmation", "enabled"),
                 ("risk", "exit_below_fast_ma")]:
        assert applies_to_profile(path, "pivot_supertrend", is_crypto=False) is False
