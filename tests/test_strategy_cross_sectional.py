from datetime import datetime

import pandas as pd
import pytest

from papertrader.data.nse_client import Quote
from papertrader.strategy.cross_sectional_momentum import select_cross_sectional_candidates

CFG = {
    "fast_ma_days": 50,
    "slow_ma_days": 200,
    "min_avg_daily_turnover_inr": 1_000_000,
    "cross_sectional": {"lookback_days": 252, "skip_recent_days": 21, "top_pct": 20.0},
}


def _hist(days: int = 300, drift: float = 0.0) -> pd.DataFrame:
    closes = [100.0]
    for _ in range(days - 1):
        closes.append(closes[-1] * (1 + drift))
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    df = pd.DataFrame({"Close": closes}, index=idx)
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    df["Volume"] = 1_000_000
    return df


def _quote(symbol: str, ltp: float) -> Quote:
    return Quote(symbol=symbol, ltp=ltp, prev_close=ltp, week52_high=ltp * 1.1, week52_low=ltp * 0.8,
                 volume=1_000_000, timestamp=datetime.now(), source="test")


def _universe(specs: list[tuple[str, float]], turnover: float = 50_000_000) -> list[tuple[str, Quote, pd.DataFrame, float]]:
    data = []
    for symbol, drift in specs:
        hist = _hist(drift=drift)
        ltp = float(hist["Close"].iloc[-1])
        data.append((symbol, _quote(symbol, ltp), hist, turnover))
    return data


def test_only_top_percentile_qualifies():
    universe = _universe([
        ("LEADER1", 0.006), ("LEADER2", 0.005),
        ("LAGGARD1", 0.001), ("LAGGARD2", 0.0008),
        ("LOSER", -0.003), ("FLAT", 0.0),
    ])
    ranked = select_cross_sectional_candidates(universe, CFG)
    # top_pct=20% of 6 symbols -> round(1.2) -> 1
    assert len(ranked) == 1
    assert ranked[0].symbol == "LEADER1"


def test_absolute_return_gate_excludes_losers_and_flat_even_at_full_width():
    universe = _universe([
        ("LEADER1", 0.006), ("LEADER2", 0.005),
        ("LAGGARD1", 0.001), ("LAGGARD2", 0.0008),
        ("LOSER", -0.003), ("FLAT", 0.0),
    ])
    wide_cfg = {**CFG, "cross_sectional": {**CFG["cross_sectional"], "top_pct": 100.0}}
    ranked = select_cross_sectional_candidates(universe, wide_cfg)
    symbols = [c.symbol for c in ranked]
    assert "LOSER" not in symbols
    assert "FLAT" not in symbols
    assert symbols == ["LEADER1", "LEADER2", "LAGGARD1", "LAGGARD2"]


def test_scores_are_descending_percentiles():
    universe = _universe([("A", 0.006), ("B", 0.004), ("C", 0.002), ("D", 0.001)])
    wide_cfg = {**CFG, "cross_sectional": {**CFG["cross_sectional"], "top_pct": 100.0}}
    ranked = select_cross_sectional_candidates(universe, wide_cfg)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100.0


def test_illiquid_stock_excluded():
    universe = _universe([("ILLIQ", 0.01)], turnover=10.0)
    ranked = select_cross_sectional_candidates(universe, CFG)
    assert ranked == []


def test_insufficient_history_excluded_not_crashed():
    hist = _hist(days=100, drift=0.01)
    ltp = float(hist["Close"].iloc[-1])
    universe = [("SHORTHIST", _quote("SHORTHIST", ltp), hist, 50_000_000)]
    ranked = select_cross_sectional_candidates(universe, CFG)
    assert ranked == []


def test_below_trend_ma_excluded():
    # A single-day gap down on the very last close, well after the
    # lookback+skip window used for the momentum calculation (so trailing
    # return is still strongly positive) -- isolates the trend-confirmation
    # (fast/slow MA) filter from the absolute-return gate.
    hist = _hist(days=300, drift=0.004)
    hist.loc[hist.index[-1], "Close"] = hist["Close"].iloc[-2] * 0.5
    ltp = float(hist["Close"].iloc[-1])
    universe = [("CRASHED", _quote("CRASHED", ltp), hist, 50_000_000)]
    ranked = select_cross_sectional_candidates(universe, CFG)
    assert ranked == []


def test_price_range_filters_apply():
    universe = _universe([("CHEAP", 0.006), ("EXPENSIVE", 0.006)])
    cheap_ltp = universe[0][1].ltp
    expensive_ltp = universe[1][1].ltp
    cfg = {**CFG, "min_ltp_inr": cheap_ltp + 1, "max_ltp_inr": 0}
    ranked = select_cross_sectional_candidates(universe, cfg)
    symbols = [c.symbol for c in ranked]
    assert "CHEAP" not in symbols


def test_empty_universe_returns_empty_list():
    assert select_cross_sectional_candidates([], CFG) == []
