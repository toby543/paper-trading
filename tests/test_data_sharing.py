"""One shared market-data client, short-lived quote reuse, and ledger journal mode."""
import sqlite3
import sys
import threading
import time
import types
from datetime import datetime

import pandas as pd
import pytest

from papertrader.data.nse_client import DataClientPool, DataUnavailableError, MarketDataClient, Quote
from papertrader.portfolio.storage import Storage


def _quote(symbol="ABC", price=100.0):
    return Quote(symbol=symbol, ltp=price, prev_close=price, week52_high=price, week52_low=price,
                 volume=1.0, timestamp=datetime.now(), source="test")


def test_a_quote_is_reused_within_its_ttl_and_refetched_after():
    client = MarketDataClient(quote_ttl_seconds=0.3)
    calls = []
    client._get_quote = lambda symbol: calls.append(symbol) or _quote(symbol, 100.0 + len(calls))
    first = client.get_quote("ABC")
    assert client.get_quote("ABC").ltp == first.ltp and len(calls) == 1
    client.get_quote("XYZ")
    assert len(calls) == 2                                  # per symbol
    time.sleep(0.35)
    assert client.get_quote("ABC").ltp != first.ltp and len(calls) == 3


def test_quote_caching_is_off_by_default_and_never_caches_a_failure():
    plain = MarketDataClient()
    calls = []
    plain._get_quote = lambda symbol: calls.append(1) or _quote(symbol)
    plain.get_quote("ABC"), plain.get_quote("ABC")
    assert len(calls) == 2

    cached = MarketDataClient(quote_ttl_seconds=30)
    outcomes = [DataUnavailableError("down"), _quote("ABC", 55.0)]

    def flaky(symbol):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    cached._get_quote = flaky
    with pytest.raises(DataUnavailableError):
        cached.get_quote("ABC")
    assert cached.get_quote("ABC").ltp == 55.0              # the failure was not remembered


def test_a_cached_quote_cannot_be_changed_by_its_caller():
    client = MarketDataClient(quote_ttl_seconds=30)
    client._get_quote = lambda symbol: _quote(symbol, 100.0)
    client.get_quote("ABC").ltp = 1.0
    assert client.get_quote("ABC").ltp == 100.0


def test_concurrent_requests_for_one_history_download_it_once(monkeypatch):
    downloads = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):
            downloads.append(1)
            time.sleep(0.15)
            return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
                                index=pd.to_datetime(["2026-11-02"]))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    client = MarketDataClient()
    client._YFINANCE_MIN_INTERVAL_SECONDS = 0
    results = []
    threads = [threading.Thread(target=lambda: results.append(client.get_history("ABC", period="2y")))
               for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(downloads) == 1 and len(results) == 5


def _fake_yahoo(monkeypatch):
    downloads = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period, **kwargs):
            downloads.append(period)
            close = 100.0 + len(downloads)
            return pd.DataFrame({"Open": [close] * 2, "High": [close] * 2, "Low": [close] * 2,
                                 "Close": [close - 1, close], "Volume": [1.0] * 2},
                                index=pd.to_datetime(["2026-11-11", "2026-11-12"]))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    client = MarketDataClient(preferred="yfinance", quote_ttl_seconds=20)
    client._YFINANCE_MIN_INTERVAL_SECONDS = 0
    return client, downloads


def test_a_yahoo_quote_and_a_one_year_history_share_one_download(monkeypatch):
    client, downloads = _fake_yahoo(monkeypatch)
    quote = client.get_quote("ABC")
    history = client.get_history("ABC", period="1y")
    assert downloads == ["1y"] and float(history["Close"].iloc[-1]) == quote.ltp
    client.get_history("ABC", period="2y")                   # a different span is its own download
    assert downloads == ["1y", "2y"]


def test_a_fill_never_reuses_a_download_and_history_keeps_its_download_time(monkeypatch):
    client, downloads = _fake_yahoo(monkeypatch)
    client.get_quote("ABC")
    fresh = client.get_fresh_quote("ABC")
    assert downloads == ["1y", "1y"] and fresh.ltp == 102.0
    client._raw_cache[("ABC", "1y")] = (time.time() - 600, client._raw_cache[("ABC", "1y")][1])
    history = client.get_history("ABC", period="1y")          # reuses the 10-minute-old download
    assert downloads == ["1y", "1y"]
    assert time.time() - history.attrs["_fetched_at"] >= 600  # so its TTL runs from then, not now


def test_yahoo_pacing_is_shared_by_every_caller():
    client = MarketDataClient()
    client._YFINANCE_MIN_INTERVAL_SECONDS = 0.1
    started = time.time()
    threads = [threading.Thread(target=client._throttle_yfinance) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert time.time() - started >= 0.3 - 0.02              # four calls, three enforced gaps


def test_a_fresh_quote_skips_the_cache_and_refreshes_it():
    client = MarketDataClient(quote_ttl_seconds=30)
    prices = iter([100.0, 101.0, 102.0])
    client._get_quote = lambda symbol: _quote(symbol, next(prices))
    assert client.get_quote("ABC").ltp == 100.0
    assert client.get_fresh_quote("ABC").ltp == 101.0
    assert client.get_quote("ABC").ltp == 101.0            # readers now see the fresher price


def test_manual_fills_use_a_fresh_price_and_its_dashboard_a_cached_one(tmp_path, monkeypatch):
    manual = _engine(tmp_path, monkeypatch, "manual_swing", DataClientPool())
    prices = iter([100.0, 105.0])
    manual.data._get_quote = lambda symbol: _quote(symbol, next(prices))
    assert manual.data.get_quote("ABC").ltp == 100.0        # dashboard read, now cached
    assert manual.manual_trader()._live_price("ABC").ltp == 105.0


def test_the_pool_shares_a_client_only_for_identical_settings():
    pool = DataClientPool()
    a = pool.get(quote_currency="INR", timeout=10, quote_ttl_seconds=20)
    assert pool.get(quote_currency="INR", timeout=10, quote_ttl_seconds=20) is a
    assert pool.get(quote_currency="INR", timeout=10, quote_ttl_seconds=0) is not a


def _engine(tmp_path, monkeypatch, profile, pool):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / f"{name or profile}.db"))
    return TradingEngine(cfg, profile_name=profile, data_pool=pool)


def test_automated_profiles_share_one_client_and_a_manual_book_keeps_its_own(tmp_path, monkeypatch):
    pool = DataClientPool()
    swing = _engine(tmp_path, monkeypatch, "52w_high", pool)
    crypto = _engine(tmp_path, monkeypatch, "crypto_momentum", pool)
    manual = _engine(tmp_path, monkeypatch, "manual_swing", pool)
    assert swing.data is crypto.data and swing.data.quote_ttl_seconds > 0
    assert manual.data is not swing.data                    # it may use the end-of-day fallback
    assert manual.data.allow_eod_fallback is True
    assert swing.data.allow_eod_fallback is False
    alone = _engine(tmp_path, monkeypatch, "52w_high", None)
    assert alone.data is not swing.data


def test_a_ledger_left_in_wal_mode_is_switched_back(tmp_path):
    path = str(tmp_path / "ledger.db")
    Storage(path, 1000.0).record_equity(1000.0, 0.0)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.close()
    ledger = Storage(path, 1000.0)
    assert sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert ledger.get_cash() == 1000.0 and len(ledger.get_equity_curve()) == 1


def test_a_reader_is_not_blocked_by_an_open_write_transaction(tmp_path):
    ledger = Storage(str(tmp_path / "w.db"), 1000.0)
    writer = sqlite3.connect(str(tmp_path / "w.db"))
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE account SET cash = 5 WHERE id = 1")
    started = time.time()
    assert ledger.get_cash() == 1000.0                      # sees the last committed value, at once
    assert time.time() - started < 1.0
    writer.rollback()
    writer.close()
