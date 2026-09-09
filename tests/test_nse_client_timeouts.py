"""Regression test for a real production incident: the live engine's
background thread went 1076 minutes (~18 hours) without completing a
scan, with no exception ever logged.

Root cause: every yfinance `.history()` call in this module was made with
no timeout. A stalled connection there hangs the call forever -- no
exception is raised, so run_forever()'s top-level try/except (added
earlier to survive a bad symbol crashing the loop) never fires, and
scan_for_entries() never even starts, so last_scan_at never updates
either. The engine doesn't crash and doesn't retry; it just freezes,
silently, until the process is restarted by hand.

These tests inject a fake `yfinance` module (via sys.modules, since
nse_client.py imports it lazily inside each method) that raises unless a
`timeout` kwarg is present, and assert every call site both passes one
and converts whatever the underlying network error is into the existing
DataUnavailableError convention -- so a genuine timeout degrades into
"skip this symbol" (already handled everywhere) instead of an unbounded
hang or an uncaught exception.

Needs pandas/requests/tenacity (this module's real imports), so -- like
the strategy tests -- this can't run in a pandas-free sandbox; it was
verified by hand against an equivalent stub before being committed.
"""
import sys
import types

import pytest

from papertrader.data.nse_client import DataUnavailableError, MarketDataClient


class _FakeEmptyFrame:
    """Stand-in for an empty pandas DataFrame -- only `.empty` is read
    before any of these calls would raise anyway."""
    empty = True


class _TimeoutRequiredTicker:
    """Raises unless called with a `timeout` kwarg -- proof the call site
    actually passes one, not just that it happens not to crash."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            raise AssertionError(f"history() called without a timeout for {self.symbol}: {kwargs}")
        raise TimeoutError(f"simulated stalled connection after {kwargs['timeout']}s")


@pytest.fixture
def client_with_timeout_ticker(monkeypatch):
    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = _TimeoutRequiredTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    client = MarketDataClient(preferred="yfinance", fallback="yfinance", timeout=7)
    client._nse_broken = True  # force every call down the yfinance path being tested
    return client


def test_get_quote_yfinance_fallback_passes_a_timeout(client_with_timeout_ticker):
    with pytest.raises(DataUnavailableError):
        client_with_timeout_ticker.get_quote("RELIANCE")


def test_get_history_passes_a_timeout_and_wraps_the_error(client_with_timeout_ticker):
    with pytest.raises(DataUnavailableError):
        client_with_timeout_ticker.get_history("RELIANCE")


def test_get_index_history_passes_a_timeout_and_wraps_the_error(client_with_timeout_ticker):
    with pytest.raises(DataUnavailableError):
        client_with_timeout_ticker.get_index_history("^NSEI")


def test_a_timeout_on_one_symbol_does_not_raise_a_raw_exception(monkeypatch):
    """Before this fix, get_history()'s yfinance call was unwrapped: any
    exception (a timeout included) would propagate as-is out of
    check_exits()'s per-symbol loop instead of being treated like every
    other "no data for this symbol" case."""
    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = _TimeoutRequiredTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    client = MarketDataClient(preferred="yfinance", fallback="yfinance", timeout=5)
    try:
        client.get_history("TCS")
        assert False, "expected DataUnavailableError"
    except DataUnavailableError:
        pass  # exactly the exception every existing call site already handles
    except TimeoutError:
        pytest.fail("raw TimeoutError leaked out instead of being wrapped in DataUnavailableError")
