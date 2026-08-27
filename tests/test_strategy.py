from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.momentum_52w_high import evaluate_candidate, check_exit, rank_candidates, is_market_in_uptrend

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


def test_market_uptrend_detected():
    hist = _uptrend_history()
    assert is_market_in_uptrend(hist, ma_days=200) is True


def test_market_downtrend_detected():
    days = 260
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    rng = np.random.default_rng(7)
    closes = [20000.0]
    for _ in range(days - 1):
        closes.append(closes[-1] * (1 - 0.004 + rng.normal(0, 0.004)))
    df = pd.DataFrame({"Close": closes}, index=idx)
    assert is_market_in_uptrend(df, ma_days=200) is False


def test_market_regime_fails_open_without_enough_history():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=10, freq="B")
    df = pd.DataFrame({"Close": [20000.0] * 10}, index=idx)
    assert is_market_in_uptrend(df, ma_days=200) is True


def test_relative_strength_passes_when_outperforming_index():
    hist = _uptrend_history(drift=0.008)
    index_hist = _uptrend_history(drift=0.002)
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.005, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {**STRATEGY_CFG, "min_relative_strength_pct": 1.0}
    cand = evaluate_candidate("TEST", quote, hist, 100_000_000, cfg, index_history=index_hist)
    assert cand is not None
    assert cand.relative_strength_pct is not None and cand.relative_strength_pct > 1.0


def test_relative_strength_rejects_when_bar_too_high():
    hist = _uptrend_history(drift=0.008)
    index_hist = _uptrend_history(drift=0.002)
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.005, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {**STRATEGY_CFG, "min_relative_strength_pct": 500.0}
    cand = evaluate_candidate("TEST", quote, hist, 100_000_000, cfg, index_history=index_hist)
    assert cand is None


def test_volume_surge_passes_confirmation():
    hist = _uptrend_history()
    hist["Volume"] = [1_000_000] * (len(hist) - 10) + [2_500_000] * 10
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.005, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {**STRATEGY_CFG, "volume_confirmation": {"enabled": True, "recent_days": 10, "baseline_days": 50, "min_volume_multiple": 1.2}}
    cand = evaluate_candidate("TEST", quote, hist, 100_000_000, cfg)
    assert cand is not None
    assert cand.volume_multiple is not None and cand.volume_multiple > 1.2


def test_flat_volume_rejected_by_confirmation():
    hist = _uptrend_history()
    hist["Volume"] = [1_000_000] * len(hist)
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.005, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {**STRATEGY_CFG, "volume_confirmation": {"enabled": True, "recent_days": 10, "baseline_days": 50, "min_volume_multiple": 1.2}}
    cand = evaluate_candidate("TEST", quote, hist, 100_000_000, cfg)
    assert cand is None


def test_new_filters_are_backward_compatible_when_absent():
    hist = _uptrend_history()
    ltp = float(hist["Close"].iloc[-1])
    quote = Quote(symbol="TEST", ltp=ltp, prev_close=ltp * 0.99, week52_high=ltp * 1.01, week52_low=ltp * 0.5,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cand = evaluate_candidate("TEST", quote, hist, 100_000_000, STRATEGY_CFG)
    assert cand is not None
    assert cand.relative_strength_pct is None
    assert cand.volume_multiple is None


def test_take_profit_disabled_by_default_even_on_huge_gain():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=100.0)
    quote = Quote(symbol="TEST", ltp=200.0, prev_close=195, week52_high=210, week52_low=90,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 90.0, "exit_below_fast_ma": False, "take_profit_pct": 0}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert not should_exit


def test_take_profit_triggers_at_target():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=100.0)
    quote = Quote(symbol="TEST", ltp=125.0, prev_close=124, week52_high=130, week52_low=90,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 90.0, "exit_below_fast_ma": False, "take_profit_pct": 25.0}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert should_exit
    assert "take_profit" in reason


def test_take_profit_does_not_trigger_below_target():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=100.0)
    quote = Quote(symbol="TEST", ltp=120.0, prev_close=118, week52_high=130, week52_low=90,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 90.0, "exit_below_fast_ma": False, "take_profit_pct": 25.0}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert not should_exit


def test_stop_loss_still_works_when_take_profit_enabled():
    hist = _uptrend_history()
    pos = Position(symbol="TEST", quantity=10, avg_price=100.0, entry_date="2026-01-01", highest_close_since_entry=100.0)
    quote = Quote(symbol="TEST", ltp=90.0, prev_close=92, week52_high=105, week52_low=85,
                  volume=1_000_000, timestamp=datetime.now(), source="test")
    cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 90.0, "exit_below_fast_ma": False, "take_profit_pct": 25.0}
    should_exit, reason = check_exit(pos, quote, hist, cfg)
    assert should_exit
    assert "stop_loss" in reason
