"""Edit Settings must only offer fields that govern the viewed profile."""
from papertrader.web.settings_schema import CRYPTO_INERT_PATHS, applies_to_profile


def test_crypto_profile_hides_nse_only_globals():
    """A crypto profile sets its own universe, trades 24/7, and routes
    quotes to Binance/Kraken -- so the NSE universe list, the NSE session
    window and the NSE/Yahoo source pair govern nothing it does."""
    for path in [("universe", "file"), ("engine", "market_open"),
                 ("engine", "market_close"), ("data_source", "preferred"),
                 ("data_source", "fallback")]:
        assert applies_to_profile(path, "crypto_momentum", is_crypto=True) is False


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
        assert applies_to_profile(path, "crypto_momentum", is_crypto=True) is True


def test_same_strategy_mode_differs_by_profile_category():
    """crypto_breakout and the NSE Consolidation Breakout profile share
    the consolidation_breakout MODE, so mode alone cannot decide this --
    only the profile's crypto-ness can."""
    path = ("engine", "market_open")
    assert applies_to_profile(path, "consolidation_breakout", is_crypto=True) is False
    assert applies_to_profile(path, "consolidation_breakout", is_crypto=False) is True


def test_min_rsi_is_crypto_momentum_only():
    path = ("strategy", "min_rsi")
    assert applies_to_profile(path, "crypto_momentum", is_crypto=True) is True
    assert applies_to_profile(path, "52w_high", is_crypto=False) is False


def test_52w_high_only_fields_hidden_from_crypto():
    for path in [("strategy", "proximity_to_52w_high_pct"),
                 ("strategy", "min_relative_strength_pct"),
                 ("strategy", "volume_confirmation", "enabled"),
                 ("strategy", "min_ltp_inr"),
                 ("risk", "exit_below_fast_ma")]:
        assert applies_to_profile(path, "crypto_momentum", is_crypto=True) is False
