from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.momentum_52w_high import evaluate_candidate, check_exit, rank_candidates

STRATEGY_CFG = {
    "proximity_to_52w_high_pct": 5.0,
    "momentum_lookback_days": 90,
    "min_momentum_return_pct": 15.0,
    "fast_ma_days": 50,
    "slow_ma_days": 200,
    "min_avg_daily_turnover_inr": 50_000_000,
}


def _uptrend_history(days: int = 260, start: float = 100.0, drift: float = 0.006) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = [start]
    for _ in range(days - 1):
        closes.append(closes[-1] * (1 + drift + rng.normal(0, 0.004)))
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    df = pd.DataFrame({"Close": closes}, index=idx)
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    df["Volume"] = 1_000_000
    return df


def test_strong_uptrend_near_high_qualifies():
    hist = _uptrend_history()
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.01,
                  week52_low=ltp * 0.5, volume=1_000_000, timestamp=datetime.now(), source="test")
    cand = evaluate_candidate("TEST", quote, hist, avg_daily_turnover=100_000_000, cfg=STRATEGY_CFG)
    assert cand is not None
    assert cand.score > 0


def test_flat_stock_rejected():
    days = 260
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    df = pd.DataFrame({"Close": [100.0] * days}, index=idx)
    df["High"] = 101
    df["Low"] = 99
    df["Volume"] = 1_000_000
    quote = Quote(symbol="FLAT", ltp=100.0, prev_close=100.0, week52_high=101.0,
                  week52_low=99.0, volume=1_000_000, timestamp=datetime.now(), source="test")
    cand = evaluate_candidate("FLAT", quote, df, avg_daily_turnover=100_000_000, cfg=STRATEGY_CFG)
    assert cand is None


def test_far_from_high_rejected():
    hist = _uptrend_history()
    quote = Quote(symbol="TEST", ltp=float(hist["Close"].iloc[-1]) * 0.7, prev_close=90,
                  week52_high=float(hist["Close"].iloc[-1]) * 1.0, week52_low=50,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cand = evaluate_candidate("TEST", quote, hist, avg_daily_turnover=100_000_000, cfg=STRATEGY_CFG)
    assert cand is None


def test_illiquid_stock_rejected():
    hist = _uptrend_history()
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="ILLIQ", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.01,
                  week52_low=ltp * 0.5, volume=1000, timestamp=datetime.now(), source="test")
    cand = evaluate_candidate("ILLIQ", quote, hist, avg_daily_turnover=1_000_000, cfg=STRATEGY_CFG)
    assert cand is None


def test_stop_loss_triggers_exit():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=100.0)
    quote = Quote(symbol="TEST", ltp=92.0, prev_close=95, week52_high=110, week52_low=80,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 10.0, "exit_below_fast_ma": False}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert should_exit
    assert "stop_loss" in reason


def test_trailing_stop_triggers_exit():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=150.0)
    quote = Quote(symbol="TEST", ltp=130.0, prev_close=140, week52_high=160, week52_low=80,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 10.0, "exit_below_fast_ma": False}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert should_exit
    assert "trailing_stop" in reason


def test_healthy_position_does_not_exit():
    hist = _uptrend_history()
    ltp = float(hist["Close"].iloc[-1])
    pos = Position(symbol="TEST", quantity=10, avg_price=ltp * 0.9, entry_date="2026-01-01", highest_close_since_entry=ltp)
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.01, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 10.0, "exit_below_fast_ma": True, "fast_ma_days": 50}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert not should_exit


def test_rank_candidates_orders_by_score():
    hist = _uptrend_history()
    ltp = float(hist["Close"].iloc[-1])
    q1 = Quote(symbol="A", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.005, week52_low=ltp * 0.5,
               volume=1_000_000, timestamp=datetime.now(), source="test")
    q2 = Quote(symbol="B", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.04, week52_low=ltp * 0.5,
               volume=1_000_000, timestamp=datetime.now(), source="test")
    c1 = evaluate_candidate("A", q1, hist, 100_000_000, STRATEGY_CFG)
    c2 = evaluate_candidate("B", q2, hist, 100_000_000, STRATEGY_CFG)
    assert c1 and c2
    ranked = rank_candidates([c2, c1])
    assert ranked[0].symbol == "A"  # closer to its 52w high -> higher proximity bonus
