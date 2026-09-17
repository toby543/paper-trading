"""Tests for the long-term trend strategy.

Every scenario here was verified first against a pure-Python (no pandas)
re-implementation of the same filter chain and exit logic, run standalone
in a sandbox without pandas installed. This file is the permanent,
pandas-based version of that check.
"""
from datetime import datetime

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.long_term_trend import (
    _momentum_return_pct,
    _moving_average,
    check_exit,
    evaluate_candidate,
)

# Shrunk MA windows so fixtures stay small -- same shape as the real
# 200/400-day defaults, just faster to build and verify by hand.
CFG = {
    "fast_ma_days": 20,
    "slow_ma_days": 40,
    "min_avg_daily_turnover_inr": 50_000_000,
    "momentum_lookback_days": 30,
    "min_momentum_return_pct": 12.0,
}


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    """closes[-1] is "today" -- matches the established convention used by
    every other strategy in this app (history's last row is the current/
    reference bar, and quote.ltp equals its Close)."""
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


def test_momentum_return_pct_uses_the_lookback_window():
    closes = [100.0] * 30 + [110.0]  # 30-day-ago close was 100, today is 110
    hist = _history_from_closes(closes)
    assert _momentum_return_pct(hist, 30) == 10.0


def test_clean_long_uptrend_with_participation_qualifies():
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG)
    assert cand is not None
    assert cand.fast_ma > cand.slow_ma
    assert cand.momentum_return_pct >= 12.0


def test_flat_sideways_drift_rejected():
    closes = [100.0] * 60
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(100.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert "weak_participation" in reasons or "not_in_uptrend" in reasons


def test_declining_stock_rejected_as_not_in_uptrend():
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * 0.995)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("not_in_uptrend") == 1


def test_illiquid_symbol_rejected_even_with_a_great_trend():
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(closes[-1]), hist, 1000, CFG, reasons=reasons)
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


RISK_CFG = {"stop_loss_pct": 20.0, "trailing_stop_pct": 25.0, "take_profit_pct": 0, "fast_ma_days": 20}


def test_wide_stop_loss_still_triggers_on_a_real_breach():
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(79.0), hist, RISK_CFG)
    assert should_exit
    assert "stop_loss" in reason


def test_wide_trailing_stop_still_triggers_on_a_real_breach():
    pos = _position(avg_price=100.0, highest_close=200.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(148.0), hist, RISK_CFG)
    assert should_exit
    assert "trailing_stop" in reason


def test_ordinary_pullback_within_wide_stops_does_not_exit():
    """The whole point of profile-scoped, wider risk settings: a pullback
    that would stop a swing profile out shouldn't touch this one."""
    pos = _position(avg_price=100.0, highest_close=130.0)
    hist = _history_from_closes([100.0] * 20)
    should_exit, reason = check_exit(pos, _quote(115.0), hist, RISK_CFG)  # 11.5% below peak
    assert not should_exit


def test_trend_break_exit_below_the_long_moving_average():
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 20)
    should_exit, reason = check_exit(pos, _quote(96.0), hist, RISK_CFG)  # only 4% below entry, below flat 20DMA
    assert should_exit
    assert "trend_break" in reason
