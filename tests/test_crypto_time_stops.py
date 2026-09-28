"""Crypto time stops must measure SIMULATED days held in a backtest.

Time stops exist only in the crypto strategies (settings_schema.py scopes
`time_stop_days` to exactly these three). They derive days-held from
`position.entry_date`, which PaperBroker stamps with wall-clock now() --
and PaperBroker is reused unchanged by the backtester. So a backtest that
left "now" as the wall clock measured real seconds elapsed since the run
started, not simulated days held: days_held was 0 for the entire run and
every one of these exits was silently unreachable, making backtested holds
run longer than live ones ever would. The `today` override is what keeps
the two paths in agreement.
"""
from datetime import datetime

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy import (
    crypto_institutional_swing,
    crypto_mean_reversion,
    crypto_pairs_trading,
)

SIM_ENTRY = pd.Timestamp("2023-05-01")


def _history(closes: list[float], end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=len(closes), freq="D")
    df = pd.DataFrame(
        {"Close": closes, "High": [c * 1.02 for c in closes], "Low": [c * 0.98 for c in closes]},
        index=idx,
    )
    df["Volume"] = 1_000_000.0
    return df


def _position(entry_date: str, avg_price: float = 100.0) -> Position:
    return Position(symbol="BTC-USD", quantity=1.0, avg_price=avg_price,
                    entry_date=entry_date, highest_close_since_entry=avg_price)


def _quote(ltp: float) -> Quote:
    return Quote(symbol="BTC-USD", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.5,
                 week52_low=ltp * 0.5, volume=10_000_000, timestamp=datetime.now(), source="test")


# A holding whose price is flat and slightly below entry: no other exit
# (stop-loss, profit target, trend break, reversion) can fire, so whether
# check_exit exits at all isolates the time stop.
FLAT = [99.0] * 120

# For mean reversion specifically, the 20-day mean has to stay ABOVE ltp,
# or "reverted_to_mean" exits first and the time stop is never reached.
BELOW_MEAN = [120.0] * 119 + [99.0]


def test_mean_reversion_time_stop_fires_on_simulated_days_held():
    day = SIM_ENTRY + pd.Timedelta(days=6)  # time_stop_days default is 5
    pos = _position(SIM_ENTRY.isoformat())
    hist = _history(BELOW_MEAN, end=day)
    should_exit, reason = crypto_mean_reversion.check_exit(
        pos, _quote(99.0), hist, {"time_stop_days": 5}, today=day,
    )
    assert should_exit
    assert reason.startswith("time_stop")
    assert "6 days" in reason


def test_mean_reversion_time_stop_holds_before_the_limit():
    day = SIM_ENTRY + pd.Timedelta(days=3)
    pos = _position(SIM_ENTRY.isoformat())
    hist = _history(BELOW_MEAN, end=day)
    should_exit, _ = crypto_mean_reversion.check_exit(
        pos, _quote(99.0), hist, {"time_stop_days": 5}, today=day,
    )
    assert not should_exit


def test_institutional_swing_time_stop_fires_on_simulated_days_held():
    day = SIM_ENTRY + pd.Timedelta(days=8)  # time_stop_days default is 7
    pos = _position(SIM_ENTRY.isoformat())
    # Rising history keeps price above the 20-day MA, so "trend_break"
    # (checked after the time stop) isn't what fires here.
    hist = _history([80.0 + i * 0.2 for i in range(120)], end=day)
    should_exit, reason = crypto_institutional_swing.check_exit(
        pos, _quote(99.0), hist, {"time_stop_days": 7}, today=day,
    )
    assert should_exit
    assert reason.startswith("time_stop")


def test_pairs_trading_time_stop_fires_on_simulated_days_held():
    day = SIM_ENTRY + pd.Timedelta(days=11)  # time_stop_days default is 10
    pos = _position(SIM_ENTRY.isoformat())
    should_exit, reason = crypto_pairs_trading.check_exit(
        pos, _quote(99.0), _history(FLAT, end=day), {"time_stop_days": 10},
        benchmark_history=None, today=day,
    )
    assert should_exit
    assert reason.startswith("time_stop")


def test_wall_clock_entry_would_never_trip_the_time_stop():
    """The exact shape of the bug: a simulated day far from the wall clock.

    With entry_date stamped by wall-clock now() (what PaperBroker does) and
    `today` left to default to the wall clock too, days_held collapses to 0
    however long the position is held in simulated time.
    """
    day = SIM_ENTRY + pd.Timedelta(days=90)
    pos = _position(pd.Timestamp.now().isoformat())
    hist = _history(BELOW_MEAN, end=day)
    should_exit, _ = crypto_mean_reversion.check_exit(
        pos, _quote(99.0), hist, {"time_stop_days": 5},
    )
    assert not should_exit, "wall-clock entry_date cannot express simulated days held"


def test_live_path_is_unchanged_when_today_is_omitted():
    """Omitting `today` keeps wall-clock semantics, so live behaviour holds."""
    pos = _position((pd.Timestamp.now() - pd.Timedelta(days=9)).isoformat())
    hist = _history(BELOW_MEAN, end=pd.Timestamp.now())
    should_exit, reason = crypto_mean_reversion.check_exit(
        pos, _quote(99.0), hist, {"time_stop_days": 5},
    )
    assert should_exit
    assert reason.startswith("time_stop")


def test_every_simple_crypto_strategy_accepts_the_shared_today_kwarg():
    """The backtester drives all six through one dispatch loop, so their
    signatures must stay uniform -- including the ones with no time stop."""
    from papertrader.backtest.engine import _SIMPLE_CRYPTO_STRATEGIES

    day = SIM_ENTRY + pd.Timedelta(days=3)
    pos = _position(SIM_ENTRY.isoformat())
    hist = _history(FLAT, end=day)
    for mode, module in _SIMPLE_CRYPTO_STRATEGIES.items():
        should_exit, reason = module.check_exit(pos, _quote(99.0), hist, {}, today=day)
        assert isinstance(should_exit, bool), mode
        assert isinstance(reason, str), mode
