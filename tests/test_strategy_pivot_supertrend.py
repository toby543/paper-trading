"""Tests for the pivot point + SuperTrend strategy.

The fixtures here were verified against a pure-Python (no pandas)
line-for-line re-implementation of the same algorithm before being
written as pandas tests -- a first attempt at a "breakout" fixture
(gentle uptrend the whole way through) never actually left an uptrend to
flip from, which caught a fixture-design mistake, not an algorithm bug,
before it could hide a real one. See that verification for the exact
numbers: a 25-day steady decline settles the indicator into a genuine
downtrend (the upper/"dn" band ratchets down, capping price from above),
then a single sharp reversal day clears that band and flips the trend to
up exactly on that day -- day 14 flips down, day 25 (last) flips back up,
nothing in between.
"""
from datetime import datetime

import pandas as pd
import pytest

from papertrader.data.nse_client import Quote
from papertrader.strategy.pivot_supertrend import (
    _atr,
    _prior_day_pivot,
    _supertrend,
    check_exit,
    evaluate_candidate,
)
from papertrader.portfolio.models import Position

CFG = {
    "min_avg_daily_turnover_inr": 50_000_000,
    "pivot_supertrend": {"atr_period": 10, "supertrend_multiplier": 3.0, "min_pct_above_pivot": 0.0},
}

PERIOD = 10
MULTIPLIER = 3.0


def _reversal_history(decline_days: int = 25, breakout_add: float = 6.0) -> pd.DataFrame:
    """A steady decline (settles SuperTrend into a real downtrend) then one
    sharp reversal bar that must clear the upper band and flip to an
    uptrend -- the exact setup this strategy looks for. Deterministic, no
    RNG, so the flip happens at a known, hand-verified bar."""
    close = [120.0]
    for _ in range(1, decline_days):
        close.append(close[-1] - 0.6)
    prev_close = close[-1]
    breakout_close = prev_close + breakout_add
    close.append(breakout_close)

    high = [c + 0.4 for c in close[:-1]] + [breakout_close + 0.4]
    low = [c - 0.4 for c in close[:-1]] + [prev_close - 0.4]

    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(close), freq="B")
    df = pd.DataFrame({"Close": close, "High": high, "Low": low}, index=idx)
    df["Volume"] = 1_000_000.0
    return df


def _quote(ltp: float) -> Quote:
    return Quote(symbol="TEST", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.2, week52_low=ltp * 0.8,
                 volume=1_000_000, timestamp=datetime.now(), source="test")


# --------------------------------------------------------------- indicators


def test_supertrend_settles_into_a_real_downtrend_then_flips_on_reversal():
    hist = _reversal_history()
    line, in_uptrend = _supertrend(hist, PERIOD, MULTIPLIER)
    assert line is not None
    # Mid-decline: genuinely in a downtrend, not just "hasn't flipped yet".
    assert not bool(in_uptrend.iloc[-5])
    # The reversal bar (last) flips it, and only it -- the bar before must
    # still show the downtrend.
    assert bool(in_uptrend.iloc[-1]) and not bool(in_uptrend.iloc[-2])


def test_supertrend_insufficient_history_returns_none():
    hist = _reversal_history().tail(5)  # far fewer bars than atr_period + 1
    line, in_uptrend = _supertrend(hist, PERIOD, MULTIPLIER)
    assert line is None and in_uptrend is None


def test_atr_is_positive_and_stable_once_seeded():
    hist = _reversal_history()
    atr = _atr(hist, PERIOD)
    assert atr is not None
    assert atr.iloc[-1] > 0
    # The decline's daily range is a constant 0.8 (close +/- 0.4), so ATR
    # should sit close to that away from the breakout bar.
    assert 0.5 < atr.iloc[-3] < 1.5


def test_pivot_uses_only_the_prior_bar():
    hist = _reversal_history()
    pivot = _prior_day_pivot(hist)
    prev = hist.iloc[-2]
    expected = (prev["High"] + prev["Low"] + prev["Close"]) / 3.0
    assert pivot == pytest.approx(expected)


# --------------------------------------------------------------- entries


def test_genuine_reversal_qualifies():
    hist = _reversal_history()
    breakout_close = float(hist["Close"].iloc[-1])
    cand = evaluate_candidate("TEST", _quote(breakout_close), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG)
    assert cand is not None
    assert cand.symbol == "TEST"
    assert cand.pct_above_pivot > 0
    assert cand.supertrend_value < breakout_close


def test_no_flip_is_rejected_with_reason():
    """Mid-decline, still in a downtrend -- must not qualify."""
    hist = _reversal_history().iloc[:-5]  # cut off before the reversal bar
    ltp = float(hist["Close"].iloc[-1])
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(ltp), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"no_supertrend_flip": 1}


def test_illiquid_symbol_rejected():
    hist = _reversal_history()
    ltp = float(hist["Close"].iloc[-1])
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(ltp), hist,
                              avg_daily_turnover=1_000.0, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"illiquid": 1}


def test_min_pct_above_pivot_filter():
    hist = _reversal_history()
    breakout_close = float(hist["Close"].iloc[-1])
    strict_cfg = {**CFG, "pivot_supertrend": {**CFG["pivot_supertrend"], "min_pct_above_pivot": 50.0}}
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(breakout_close), hist,
                              avg_daily_turnover=100_000_000, cfg=strict_cfg, reasons=reasons)
    assert cand is None
    assert reasons == {"below_pivot": 1}


def test_insufficient_history_rejected_not_crashed():
    hist = _reversal_history().tail(3)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(100.0), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"insufficient_history": 1}


# ---------------------------------------------------------------- exits


def _position(avg_price: float, highest_close: float) -> Position:
    return Position(symbol="TEST", quantity=10, avg_price=avg_price,
                    entry_date="2026-01-01", highest_close_since_entry=highest_close)


def test_stop_loss_exit():
    hist = _reversal_history()
    pos = _position(avg_price=200.0, highest_close=200.0)
    should_exit, reason = check_exit(pos, _quote(180.0), hist, {"stop_loss_pct": 7.0})
    assert should_exit and "stop_loss" in reason


def test_trailing_stop_exit():
    hist = _reversal_history()
    pos = _position(avg_price=100.0, highest_close=150.0)
    should_exit, reason = check_exit(pos, _quote(129.0), hist, {"trailing_stop_pct": 12.0})
    assert should_exit and "trailing_stop" in reason


def test_take_profit_exit():
    hist = _reversal_history()
    pos = _position(avg_price=100.0, highest_close=100.0)
    should_exit, reason = check_exit(pos, _quote(120.0), hist, {"take_profit_pct": 15.0})
    assert should_exit and "take_profit" in reason


def test_supertrend_flip_exit():
    """Price still in a downtrend (mid-decline) with no risk-based exit
    triggered -- the strategy-specific SuperTrend-flip exit must catch it."""
    hist = _reversal_history().iloc[:-3]
    ltp = float(hist["Close"].iloc[-1])
    pos = _position(avg_price=ltp * 1.001, highest_close=ltp * 1.01)  # tiny loss, no stop hit
    cfg = {"stop_loss_pct": 50.0, "trailing_stop_pct": 50.0, "pivot_supertrend": CFG["pivot_supertrend"]}
    should_exit, reason = check_exit(pos, _quote(ltp), hist, cfg)
    assert should_exit and "supertrend_flip" in reason


def test_no_exit_while_riding_the_uptrend():
    hist = _reversal_history()  # the reversal bar itself: fresh uptrend
    ltp = float(hist["Close"].iloc[-1])
    pos = _position(avg_price=ltp * 0.98, highest_close=ltp)
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 12.0, "pivot_supertrend": CFG["pivot_supertrend"]}
    should_exit, reason = check_exit(pos, _quote(ltp), hist, cfg)
    assert not should_exit
