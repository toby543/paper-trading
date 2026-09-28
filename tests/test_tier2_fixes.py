"""Regression tests for the Tier 2 bugs found in the full-codebase audit.

These are latent trading-logic defects: the two strategies involved were
not among the profiles running live, but each would have destroyed its
book the moment it was enabled.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from papertrader.data import crypto_client as cc
from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy import crypto_breakout_retest as brt
from papertrader.strategy import crypto_institutional_swing as swing


def _profile(name: str) -> dict:
    cfg = yaml.safe_load(open(f"profiles/{name}.yaml"))
    return {**cfg["risk"], **cfg["strategy"]}


def _quote(ltp: float, prev: float | None = None) -> Quote:
    return Quote("X-USD", ltp, prev if prev is not None else ltp, ltp * 1.5, ltp * 0.5,
                 1e6, datetime.now(), "test")


def _frame(closes, end, volumes=None) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=len(closes), freq="D")
    df = pd.DataFrame(
        {"Close": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes]},
        index=idx,
    )
    df["Volume"] = volumes if volumes is not None else 1e6
    return df


# --- 7. institutional swing: entry and exit must not contradict ----------

ENTRY_DAY = pd.Timestamp("2026-09-28")


def _swing_pullback_entry():
    """A dip to just under the 20-day MA that passes every entry gate --
    the setup this strategy exists to buy."""
    cfg = _profile("crypto_institutional_swing")
    for dip_len in range(3, 15):
        for dip_pct in np.arange(0.005, 0.09, 0.0025):
            base = [100 + i * 0.6 for i in range(300)]
            closes = base + [base[-1] * (1 - dip_pct * (k + 1) / dip_len) for k in range(dip_len)]
            hist = _frame(closes, ENTRY_DAY, volumes=[1e6] * (len(closes) - 1) + [2.5e6])
            ltp, ma20 = closes[-1], hist["Close"].tail(20).mean()
            if ltp >= ma20:
                continue  # we want a pullback strictly BELOW the MA
            if swing.evaluate_candidate("X-USD", _quote(ltp), hist, 9e8, cfg, reasons={}):
                return cfg, hist, ltp, ma20
    pytest.fail("no qualifying pullback-below-MA entry could be constructed")


def test_swing_does_not_sell_the_pullback_it_just_bought():
    cfg, hist, ltp, ma20 = _swing_pullback_entry()
    assert ltp < ma20
    position = Position("X-USD", 1.0, ltp, ENTRY_DAY.isoformat(), ltp)
    exited, reason = swing.check_exit(position, _quote(ltp), hist, cfg, today=ENTRY_DAY)
    assert not exited, f"sold on the entry bar: {reason}"


def test_swing_still_exits_when_the_pullback_actually_fails():
    """Below the MA by more than the entry tolerates, but inside the hard
    stop -- so this isolates trend_break rather than stop_loss."""
    cfg = _profile("crypto_institutional_swing")
    closes = [300.0] * 19 + [250.0]
    hist = _frame(closes, ENTRY_DAY)
    entry = 265.0
    ltp = entry * 0.94  # -6%, inside the -8% hard stop
    position = Position("X-USD", 1.0, entry, ENTRY_DAY.isoformat(), entry)
    exited, reason = swing.check_exit(position, _quote(ltp), hist, cfg, today=ENTRY_DAY)
    assert exited and reason.startswith("trend_break"), reason


def test_swing_profit_target_zero_means_disabled():
    """0 is this app's 'disabled' idiom everywhere else; ungated it meant
    'sell as soon as price reaches entry'."""
    cfg, hist, ltp, _ = _swing_pullback_entry()
    position = Position("X-USD", 1.0, ltp, ENTRY_DAY.isoformat(), ltp)
    exited, reason = swing.check_exit(
        position, _quote(ltp), hist, {**cfg, "profit_target_pct": 0}, today=ENTRY_DAY,
    )
    assert not exited, f"profit_target_pct=0 closed the position: {reason}"


# --- 8. breakout retest: the level is anchored at entry ------------------

def _retest_entry():
    """21 flat base bars at 100, a 9-bar breakout leg to 115, then a
    retest bar turning up off the base. A genuine entry."""
    cfg = _profile("crypto_breakout_retest")
    closes = [100.0] * 21 + [100 + (15 * (k + 1) / 9) for k in range(9)] + [100.6, 101.5]
    hist = _frame(closes, ENTRY_DAY)
    cand = brt.evaluate_candidate("X-USD", _quote(101.5, prev=100.6), hist, 9e8, cfg, reasons={})
    assert cand is not None and cand.breakout_level == pytest.approx(100.0)
    return cfg, closes


@pytest.mark.parametrize("days_held", [0, 1, 2, 4, 8, 14])
def test_retest_level_does_not_slide_forward_while_held(days_held):
    """Measured backwards from today, the level climbed the breakout leg
    and overtook the price, liquidating healthy positions on day two."""
    cfg, closes = _retest_entry()
    position = Position("X-USD", 1.0, 101.5, ENTRY_DAY.isoformat(), 101.5)
    hist = _frame(closes + [101.5] * days_held, ENTRY_DAY + pd.Timedelta(days=days_held))
    exited, reason = brt.check_exit(position, _quote(101.5), hist, cfg)
    assert not exited, f"held flat at 101.5 but exited after {days_held}d: {reason}"


def test_retest_still_exits_when_price_breaks_the_entry_level():
    cfg, closes = _retest_entry()
    position = Position("X-USD", 1.0, 101.5, ENTRY_DAY.isoformat(), 101.5)
    hist = _frame(closes + [101.5] * 3 + [98.0], ENTRY_DAY + pd.Timedelta(days=4))
    exited, reason = brt.check_exit(position, _quote(98.0), hist, cfg)
    assert exited and "retest_failed" in reason, reason


def test_retest_skips_the_level_check_when_entry_predates_the_history():
    """Rather than guessing a level, fall through to the other exits."""
    cfg, closes = _retest_entry()
    position = Position("X-USD", 1.0, 101.5, "2020-01-01T00:00:00", 101.5)
    hist = _frame(closes, ENTRY_DAY)
    exited, _ = brt.check_exit(position, _quote(101.5), hist, cfg)
    assert not exited


# --- 10. crypto 52-week range is a real 52 weeks -------------------------

class _StubBinance(cc.BinanceClient):
    """Real get_quote, stubbed network."""

    def __init__(self, history_works=True):
        super().__init__()
        self.history_calls = 0
        self._history_works = history_works

    def _get_json(self, url, params=None):
        return {"highPrice": "101", "lowPrice": "99", "lastPrice": "100",
                "prevClosePrice": "98", "volume": "1000"}

    def get_history(self, symbol, days=365, ttl_seconds=900):
        self.history_calls += 1
        if not self._history_works:
            raise cc.CryptoDataUnavailableError("klines unavailable")
        idx = pd.date_range(end="2026-09-28", periods=365, freq="D")
        frame = pd.DataFrame(
            {"Open": 1.0, "High": [140.0] * 365, "Low": [42.0] * 365, "Close": 100.0, "Volume": 1.0},
            index=idx,
        )
        self._history_cache[f"{cc.to_binance_pair(symbol)}:365"] = frame
        return frame


def test_cold_cache_still_yields_a_real_52_week_range():
    """get_quote runs before get_history in every scan loop, so the cache
    is cold on the first scan after each restart -- and the 24h ticker
    range was being reported as the 52-week range."""
    client = _StubBinance()
    quote = client.get_quote("BTC-USD")
    assert quote.week52_high == 140.0  # not the 101 24h high
    assert quote.week52_low == 42.0    # not the 99 24h low


def test_warm_cache_is_not_refetched():
    client = _StubBinance()
    client.get_quote("BTC-USD")
    client.get_quote("BTC-USD")
    assert client.history_calls == 1


def test_unavailable_history_falls_back_to_the_day_range():
    client = _StubBinance(history_works=False)
    quote = client.get_quote("BTC-USD")
    assert (quote.week52_high, quote.week52_low) == (101.0, 99.0)
