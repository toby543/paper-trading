"""Tests for the trend-pullback strategy.

Every scenario here was verified first against a pure-Python (no pandas)
re-implementation of the same filter chain and exit logic, run standalone
in a sandbox without pandas installed -- see that transcript for the
fixture-design note below. This file is the permanent, pandas-based
version of that check.

One fixture design note worth keeping: a plain multi-day decline of more
than max_pullback_pct drags the moving averages down with the price, so
it gets rejected as not_in_uptrend before ever reaching the pullback-depth
filter -- the two filters overlap in coverage by design. The
"pullback too deep" test below isolates that filter specifically with a
fresh one-day spike (a new recent high that hasn't dragged the 20/50-day
averages with it yet), rather than an ordinary decline.
"""
from datetime import datetime

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.trend_pullback import (
    _moving_average,
    _recent_high,
    check_exit,
    evaluate_candidate,
)

CFG = {
    "fast_ma_days": 20,
    "slow_ma_days": 50,
    "min_avg_daily_turnover_inr": 50_000_000,
    "trend_pullback": {"pullback_lookback_days": 20, "min_pullback_pct": 3.0, "max_pullback_pct": 12.0},
}


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    """closes[-1] is "today" -- matches the established convention already
    used by consolidation_breakout/pivot_supertrend, where history's last
    row is the current/reference bar and quote.ltp equals its Close."""
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(closes), freq="B")
    high = [c * 1.005 for c in closes]
    low = [c * 0.995 for c in closes]
    df = pd.DataFrame({"Close": closes, "High": high, "Low": low}, index=idx)
    df["Volume"] = 1_000_000.0
    return df


def _quote(ltp: float) -> Quote:
    return Quote(symbol="TEST", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.3, week52_low=ltp * 0.7,
                 volume=1_000_000, timestamp=datetime.now(), source="test")


def test_moving_average_needs_the_full_window():
    hist = _history_from_closes([100.0] * 19)
    assert _moving_average(hist, 20) is None
    hist = _history_from_closes([100.0] * 20)
    assert _moving_average(hist, 20) == 100.0


def test_recent_high_is_the_max_close_including_today():
    closes = [100.0] * 19 + [150.0]  # today (the last close) is the new high
    hist = _history_from_closes(closes)
    assert _recent_high(hist, 20) == 150.0


def test_clean_pullback_and_turn_up_qualifies():
    closes = [100.0]
    for _ in range(58):
        closes.append(closes[-1] * 1.01)
    peak = closes[-1]
    closes.append(peak * 0.945)
    closes.append(peak * 0.945 * 1.01)
    hist = _history_from_closes(closes)
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG)
    assert cand is not None
    assert 3.0 <= cand.pct_from_high <= 12.0
    assert cand.day_change_pct > 0


def test_pullback_too_shallow_rejected():
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("pullback_too_shallow") == 1


def test_pullback_too_deep_rejected():
    closes = [100.0]
    for _ in range(55):
        closes.append(closes[-1] * 1.002)
    spike = closes[-1] * 1.20
    closes.append(spike)
    closes.append(spike * 0.90)
    closes.append(spike * 0.85)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("pullback_too_deep") == 1


def test_declining_stock_rejected_as_not_in_uptrend():
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * 0.995)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("not_in_uptrend") == 1


def test_still_falling_day_rejected_as_not_turning_up():
    closes = [100.0]
    for _ in range(58):
        closes.append(closes[-1] * 1.01)
    peak = closes[-1]
    closes.append(peak * 0.95)
    closes.append(peak * 0.93)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("not_turning_up") == 1


def test_illiquid_symbol_rejected():
    closes = [100.0] * 60
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(100.0), hist, 1000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("illiquid") == 1


def test_insufficient_history_rejected_not_crashed():
    hist = _history_from_closes([100.0, 101.0])
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(101.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("insufficient_history") == 1


def _position(avg_price: float, highest_close: float) -> Position:
    return Position(symbol="TEST", quantity=10, avg_price=avg_price,
                     entry_date="2024-01-01", highest_close_since_entry=highest_close)


def test_stop_loss_exit():
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(92.0), hist, CFG)
    assert should_exit
    assert "stop_loss" in reason


def test_trailing_stop_exit():
    pos = _position(avg_price=100.0, highest_close=150.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(128.0), hist, CFG)
    assert should_exit
    assert "trailing_stop" in reason


def test_take_profit_exit():
    cfg = {**CFG, "take_profit_pct": 25.0}
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(126.0), hist, cfg)
    assert should_exit
    assert "take_profit" in reason


def test_trend_break_exit():
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 50)
    should_exit, reason = check_exit(pos, _quote(96.0), hist, CFG)
    assert should_exit
    assert "trend_break" in reason


def test_no_exit_while_riding_the_position():
    pos = _position(avg_price=100.0, highest_close=105.0)
    hist = _history_from_closes([100.0] * 50)
    should_exit, reason = check_exit(pos, _quote(104.0), hist, CFG)
    assert not should_exit
