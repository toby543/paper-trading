"""Every editable setting must actually do something.

The audit found eight settings that were displayed and editable but read
by nothing -- change them in the dashboard and the engine's behaviour was
identical. That is the single most repeated bug class in this codebase,
so the first test here is a standing guard against it rather than a check
of any one field: it walks the schema and fails on any strategy field
offered for a mode whose module never mentions it.
"""
import pathlib
import re

import pytest
import yaml

from papertrader.web.settings_schema import (
    EDITABLE_SETTINGS,
    _ALL_MODES,
    _FIELD_MODES,
    applies_to_mode,
    coerce_and_validate,
)

STRATEGY_DIR = pathlib.Path("src/papertrader/strategy")

# mode -> the module whose evaluate_candidate/check_exit that mode runs.
# cross_sectional_momentum has no check_exit of its own; the scheduler
# falls through to momentum_52w_high's, so its exit keys live there.
MODE_MODULE = {
    "52w_high": "momentum_52w_high",
    "cross_sectional_momentum": "cross_sectional_momentum",
    "consolidation_breakout": "consolidation_breakout",
    "ipo_base_breakout": "ipo_base_breakout",
    "pivot_supertrend": "pivot_supertrend",
    "trend_pullback": "trend_pullback",
    "long_term_trend": "long_term_trend",
    "crypto_momentum": "crypto_momentum",
    "crypto_breakout": "crypto_breakout",
    "crypto_institutional_swing": "crypto_institutional_swing",
    "crypto_mean_reversion": "crypto_mean_reversion",
    "crypto_trend_pullback": "crypto_trend_pullback",
    "crypto_breakout_retest": "crypto_breakout_retest",
    "crypto_pairs_trading": "crypto_pairs_trading",
}

# Exit keys that momentum_52w_high's check_exit reads on behalf of a mode
# with no check_exit of its own.
_SHARED_EXIT_FALLBACK = {"cross_sectional_momentum": "momentum_52w_high"}


def _source(mode: str, field: str) -> str:
    text = (STRATEGY_DIR / f"{MODE_MODULE[mode]}.py").read_text()
    fallback = _SHARED_EXIT_FALLBACK.get(mode)
    if fallback:
        text += (STRATEGY_DIR / f"{fallback}.py").read_text()
    return text


def _mentions(mode: str, field: str) -> bool:
    return bool(re.search(rf"\b{re.escape(field)}\b", _source(mode, field)))


def test_mode_table_covers_every_real_mode():
    assert set(MODE_MODULE) == set(_ALL_MODES)


@pytest.mark.parametrize("mode", sorted(MODE_MODULE))
def test_no_strategy_or_risk_field_is_offered_to_a_mode_that_ignores_it(mode):
    """A field shown for a mode whose module never names it cannot
    possibly change that mode's behaviour when saved."""
    phantom = []
    for entry in EDITABLE_SETTINGS:
        path = tuple(entry["path"])
        if path[0] not in ("strategy", "risk") or len(path) != 2:
            continue  # nested subsections are already mode-gated by name
        if path not in _FIELD_MODES:
            continue  # unrestricted by design (shared plumbing)
        if applies_to_mode(path, mode) and not _mentions(mode, path[1]):
            phantom.append(".".join(path))
    assert not phantom, f"{mode} is offered settings it never reads: {phantom}"


def test_every_key_a_profile_sets_is_reachable_or_read():
    """The inverse: a tunable a profile YAML sets should not be invisible
    to the UI *and* unread by the strategy (a dead key, like the old
    pivot_supertrend.min_pct_above_pivot)."""
    editable = {tuple(e["path"]) for e in EDITABLE_SETTINGS}
    dead = []
    for path in sorted(pathlib.Path("profiles").glob("*.yaml")):
        profile = yaml.safe_load(path.read_text()) or {}
        mode = profile.get("strategy_mode")
        subsection = (profile.get("strategy") or {}).get(mode)
        if mode not in MODE_MODULE or not isinstance(subsection, dict):
            continue
        for key in subsection:
            if ("strategy", mode, key) in editable:
                continue
            if not _mentions(mode, key):
                dead.append(f"{path.name}:strategy.{mode}.{key}")
    assert not dead, f"set in a profile but neither editable nor read: {dead}"


# --- the specific Tier 3 findings ---------------------------------------

def test_pivot_supertrend_is_not_offered_moving_average_fields():
    for field in ("fast_ma_days", "slow_ma_days", "momentum_lookback_days",
                  "min_momentum_return_pct"):
        assert not applies_to_mode(("strategy", field), "pivot_supertrend"), field


def test_long_term_trends_own_fast_ma_is_editable():
    """The profile runs fast_ma_days: 200 against an old ceiling of 100,
    so the panel displayed a value it refused to save."""
    configured = yaml.safe_load(open("profiles/long_term_trend.yaml"))["strategy"]["fast_ma_days"]
    assert coerce_and_validate(("strategy", "fast_ma_days"), configured) == configured


def test_trailing_stop_hidden_for_the_strategies_that_ignore_it():
    for mode in ("crypto_mean_reversion", "crypto_pairs_trading"):
        assert not applies_to_mode(("risk", "trailing_stop_pct"), mode), mode
    assert applies_to_mode(("risk", "trailing_stop_pct"), "crypto_momentum")


def test_take_profit_hidden_for_the_strategies_that_ignore_it():
    for mode in ("crypto_institutional_swing", "crypto_mean_reversion", "crypto_pairs_trading"):
        assert not applies_to_mode(("risk", "take_profit_pct"), mode), mode
    assert applies_to_mode(("risk", "take_profit_pct"), "crypto_momentum")


def test_equity_breakout_tightness_ceiling_is_editable():
    path = ("strategy", "consolidation_breakout", "max_consolidation_range_pct")
    assert path in {tuple(e["path"]) for e in EDITABLE_SETTINGS}
    assert applies_to_mode(path, "consolidation_breakout")
    assert not applies_to_mode(path, "crypto_breakout")


def test_max_retries_reaches_the_http_session():
    from papertrader.data.nse_client import MarketDataClient

    assert MarketDataClient(max_retries=7)._nse.max_attempts == 7
    assert MarketDataClient()._nse.max_attempts == 2
    # never zero attempts, whatever is configured
    assert MarketDataClient(max_retries=0)._nse.max_attempts == 1
