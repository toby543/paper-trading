"""Tests for the crypto momentum strategy."""
from datetime import datetime

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.crypto_momentum import (
    _momentum_return_pct,
    _moving_average,
    _rsi,
    check_exit,
    evaluate_candidate,
)

# Crypto-focused config with 30-day lookback and lower momentum threshold
CFG = {
    "min_avg_daily_turnover_inr": 50_000_000,
    "momentum_lookback_days": 30,
    "min_momentum_return_pct": 5.0,
    "min_rsi": 50.0,
}


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    """closes[-1] is "today" (current/reference bar)."""
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(closes), freq="B")
    high = [c * 1.02 for c in closes]  # Crypto has wider wicks
    low = [c * 0.98 for c in closes]
    df = pd.DataFrame({"Close": closes, "High": high, "Low": low}, index=idx)
    df["Volume"] = 1_000_000.0
    return df


def _quote(ltp: float) -> Quote:
    return Quote(symbol="BTCUSD", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.5,
                 week52_low=ltp * 0.5, volume=10_000_000, timestamp=datetime.now(), source="test")


def test_moving_average_needs_the_full_window():
    hist = _history_from_closes([100.0] * 19)
    assert _moving_average(hist, 20) is None
    hist = _history_from_closes([100.0] * 20)
    assert _moving_average(hist, 20) == 100.0


def test_rsi_needs_sufficient_data():
    hist = _history_from_closes([100.0] * 14)
    assert _rsi(hist, 14) is None
    hist = _history_from_closes([100.0] * 15)
    assert _rsi(hist, 14) is not None


def test_momentum_return_pct_uses_the_lookback_window():
    closes = [100.0] * 30 + [105.0]  # 30-day return is 5%
    hist = _history_from_closes(closes)
    assert _momentum_return_pct(hist, 30) == 5.0


def test_uptrend_with_rsi_momentum_qualifies():
    """Steady uptrend with strong RSI should qualify."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)  # 1% daily gains
    hist = _history_from_closes(closes)
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 100_000_000, CFG)
    assert cand is not None
    assert cand["ma20"] > cand["ma50"]
    assert cand["rsi"] > CFG["min_rsi"]
    assert cand["momentum_return_pct"] >= CFG["min_momentum_return_pct"]


def test_flat_price_action_rejected():
    """Sideways movement lacks momentum."""
    closes = [100.0] * 60
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(100.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    # Should fail on either not_in_uptrend or weak_momentum since RSI is neutral
    assert "not_in_uptrend" in reasons or "weak_momentum" in reasons


def test_declining_trend_rejected():
    """Downtrend should not qualify even with liquidity."""
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * 0.99)  # 1% daily losses
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("not_in_uptrend") == 1


def test_illiquid_pair_rejected():
    """Low daily turnover disqualifies even with great trend."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 1000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("illiquid") == 1


def test_insufficient_history_rejected():
    """Need at least 50 bars for MA calculation."""
    hist = _history_from_closes([100.0, 101.0])
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(101.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("insufficient_history") == 1


def _position(avg_price: float, highest_close: float) -> Position:
    return Position(symbol="BTCUSD", quantity=0.5, avg_price=avg_price,
                    entry_date="2024-01-01", highest_close_since_entry=highest_close)


RISK_CFG = {"stop_loss_pct": 10.0, "trailing_stop_pct": 15.0, "take_profit_pct": 0}


def test_stop_loss_triggers():
    """Stop-loss should trigger at configured threshold."""
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(89.0), hist, RISK_CFG)  # 11% below entry
    assert should_exit
    assert "stop_loss" in reason


def test_trailing_stop_triggers():
    """Trailing-stop should trigger on retracement from peak."""
    pos = _position(avg_price=100.0, highest_close=150.0)  # Peak at 150
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(126.0), hist, RISK_CFG)  # 16% below peak
    assert should_exit
    assert "trailing_stop" in reason


def test_ordinary_pullback_within_stops():
    """Small pullback within stop thresholds should not exit."""
    pos = _position(avg_price=100.0, highest_close=120.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(110.0), hist, RISK_CFG)  # 8.3% below peak
    assert not should_exit


def test_rsi_momentum_loss_triggers_exit():
    """RSI < 40 signals momentum loss and exit."""
    pos = _position(avg_price=100.0, highest_close=100.0)
    # Create flat/declining closes to push RSI below 40
    closes = [100.0] * 30 + [99.0] * 20  # Recent decline
    hist = _history_from_closes(closes)
    should_exit, reason = check_exit(pos, _quote(98.0), hist, RISK_CFG)
    assert should_exit
    assert "rsi_momentum_loss" in reason


def test_trend_break_below_20ma():
    """Close below 20-day MA signals trend break."""
    pos = _position(avg_price=100.0, highest_close=110.0)
    closes = [100.0] * 20 + [99.0]  # 20-day MA is ~100, now at 99
    hist = _history_from_closes(closes)
    should_exit, reason = check_exit(pos, _quote(99.5), hist, RISK_CFG)
    assert should_exit
    assert "trend_break" in reason
