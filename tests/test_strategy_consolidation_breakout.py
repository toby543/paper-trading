"""Regression tests for the consolidation breakout strategy.

The headline case is `test_genuine_breakout_qualifies`: for a long time this
strategy produced zero buys on every scan because `_detect_consolidation`
measured the base high over a window that included the breakout bar itself.
Since a bar's Close can never exceed its own High, `close > base_high` was
false by construction for every symbol on every day. These tests pin the
corrected behaviour so that can't silently come back.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.strategy.consolidation_breakout import (
    _detect_breakout,
    _detect_consolidation,
    evaluate_candidate,
)

CFG = {
    "fast_ma_days": 50,
    "slow_ma_days": 200,
    "min_avg_daily_turnover_inr": 50_000_000,
    "momentum_lookback_days": 21,
    "min_momentum_return_pct": 5.0,
    "consolidation_breakout": {
        "consolidation_days": 10,
        "max_consolidation_range_pct": 3.0,
        "volume_multiple": 2.0,
    },
}

BASE_LEVEL = 100.0
BREAKOUT_CLOSE = 104.0


def _breakout_history(breakout_close: float = BREAKOUT_CLOSE,
                      breakout_volume: float = 3_000_000,
                      base_wobble: float = 0.0) -> pd.DataFrame:
    """260 bars: a long climb, an 11-bar run-up, a flat 10-bar base, then a
    breakout bar. Deterministic -- no RNG, so the thresholds under test are
    exercised at known distances rather than by luck."""
    closes = list(np.linspace(60.0, 94.0, 238))          # the long uptrend
    closes += list(np.linspace(94.0, 100.0, 11))         # run-up: bars -22..-12
    rng = np.random.default_rng(7)
    closes += [BASE_LEVEL + rng.uniform(-base_wobble, base_wobble) for _ in range(10)]  # base: -11..-2
    closes.append(breakout_close)                        # the breakout bar: -1
    assert len(closes) == 260

    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(closes), freq="B")
    df = pd.DataFrame({"Close": closes}, index=idx)
    df["High"] = df["Close"] * 1.005
    df["Low"] = df["Close"] * 0.995
    df["Volume"] = 1_000_000.0
    df.iloc[-1, df.columns.get_loc("Volume")] = breakout_volume
    return df


def _quote(ltp: float) -> Quote:
    return Quote(symbol="TEST", ltp=ltp, prev_close=BASE_LEVEL, week52_high=ltp,
                 week52_low=ltp * 0.5, volume=3_000_000,
                 timestamp=datetime.now(), source="test")


# ---------------------------------------------------------------- base window


def test_base_excludes_the_latest_bar():
    """The whole bug in one assertion: the base high must be measurable
    strictly below a breakout bar's close, which is only possible if that
    bar is excluded from the base window."""
    hist = _breakout_history()
    high, low, is_tight = _detect_consolidation(hist, 10)
    assert is_tight
    assert high < BREAKOUT_CLOSE
    assert low <= BASE_LEVEL <= high


def test_base_needs_enough_history():
    hist = _breakout_history().tail(10)  # 10 bars can't supply a 10-bar base + breakout bar
    assert _detect_consolidation(hist, 10) is None


def test_wide_base_is_not_tight():
    _, _, is_tight = _detect_consolidation(_breakout_history(base_wobble=4.0), 10)
    assert not is_tight


# ------------------------------------------------------------------- breakout


def test_breakout_detected_on_strong_volume():
    hist = _breakout_history()
    high, _, _ = _detect_consolidation(hist, 10)
    assert _detect_breakout(hist, high, volume_multiple=2.0)


def test_breakout_rejected_on_weak_volume():
    hist = _breakout_history(breakout_volume=1_200_000)
    high, _, _ = _detect_consolidation(hist, 10)
    assert not _detect_breakout(hist, high, volume_multiple=2.0)


def test_no_breakout_when_price_stays_inside_the_base():
    hist = _breakout_history(breakout_close=BASE_LEVEL)
    high, _, _ = _detect_consolidation(hist, 10)
    assert not _detect_breakout(hist, high, volume_multiple=2.0)


# ---------------------------------------------------------- end-to-end filter


def test_genuine_breakout_qualifies():
    """The regression that matters: this setup must produce a candidate.
    Before the fix it returned None, as did every other input."""
    hist = _breakout_history()
    cand = evaluate_candidate("TEST", _quote(BREAKOUT_CLOSE), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG)
    assert cand is not None
    assert cand.symbol == "TEST"
    assert cand.consolidation_high < BREAKOUT_CLOSE
    assert cand.momentum_return_pct >= CFG["min_momentum_return_pct"]
    assert cand.breakout_volume >= cand.avg_volume * 2.0


def test_rejection_reason_is_recorded():
    reasons: dict[str, int] = {}
    hist = _breakout_history(breakout_volume=1_200_000)
    cand = evaluate_candidate("TEST", _quote(BREAKOUT_CLOSE), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"no_breakout": 1}


def test_illiquid_symbol_rejected():
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("TEST", _quote(BREAKOUT_CLOSE), _breakout_history(),
                              avg_daily_turnover=1_000.0, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"illiquid": 1}


def test_downtrend_rejected():
    reasons: dict[str, int] = {}
    hist = _breakout_history()
    # Quote below both moving averages -- no longer an uptrend.
    cand = evaluate_candidate("TEST", _quote(50.0), hist,
                              avg_daily_turnover=100_000_000, cfg=CFG, reasons=reasons)
    assert cand is None
    assert reasons == {"not_in_uptrend": 1}
