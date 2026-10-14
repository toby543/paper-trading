"""Crypto prices must not come from the wrong coin, and an exchange hiccup
must not push every price onto the fallback for the rest of the run.

Found on the Pi: Arbitrum was quoted from Yahoo's "ARB-USD", which is a
different coin (~$0.0006), so a stop-loss sold a flat position at a 99.7% loss.
"""
from datetime import datetime

import pandas as pd
import pytest
import requests

from papertrader.data import crypto_client
from papertrader.data.crypto_client import BinanceClient, CryptoDataUnavailableError, KrakenClient
from papertrader.data.nse_client import DataUnavailableError, MarketDataClient, yahoo_ticker


def test_yahoo_ticker_uses_the_right_coin():
    assert yahoo_ticker("ARB-USD") == "ARB11841-USD"
    assert yahoo_ticker("UNI-USD") == "UNI7083-USD"
    assert yahoo_ticker("PEPE-USD") == "PEPE24478-USD"
    assert yahoo_ticker("BTC-USD") == "BTC-USD"          # unambiguous coins pass through
    assert yahoo_ticker("RELIANCE") == "RELIANCE.NS"
    assert yahoo_ticker("ABB.BO") == "ABB.BO"


def test_the_crypto_universe_has_no_dead_matic_pair():
    import csv

    symbols = [r["symbol"] for r in csv.DictReader(open("data/universe_crypto.csv"))]
    assert "MATIC-USD" not in symbols and "POL-USD" in symbols


# --- circuit breaker ------------------------------------------------------------------

def _http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status}", response=resp)


def test_a_tripped_exchange_comes_back_after_the_cooldown(monkeypatch):
    client = BinanceClient()
    now = [1_000_000.0]
    monkeypatch.setattr(crypto_client.time, "time", lambda: now[0])
    client._trip(RuntimeError("connection reset"))
    assert client.broken
    now[0] += crypto_client.BREAKER_COOLDOWN_SECONDS - 1
    assert client.broken
    now[0] += 2
    assert not client.broken                              # not "for the rest of the process"


@pytest.mark.parametrize("status", [400, 404])
def test_one_unknown_symbol_does_not_switch_the_exchange_off(status):
    client = BinanceClient()
    client._trip(_http_error(status))
    assert not client.broken


def test_a_server_error_does_switch_it_off_for_a_while():
    client = KrakenClient()
    client._trip(_http_error(503))
    assert client.broken


def test_binance_quote_failure_on_one_symbol_keeps_binance_for_the_others(monkeypatch):
    client = BinanceClient()

    def fail(url, params=None):
        raise _http_error(400)

    monkeypatch.setattr(client, "_get_json", fail)
    with pytest.raises(CryptoDataUnavailableError):
        client.get_quote("NOPE-USD")
    assert not client.broken


# --- the fallback price is vetted against the exchanges -----------------------------

def _yahoo_returning(monkeypatch, price):
    frame = pd.DataFrame({"Open": [price], "High": [price], "Low": [price], "Close": [price], "Volume": [1.0]},
                         index=pd.to_datetime([datetime.now().date()]))

    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return frame

    monkeypatch.setattr("yfinance.Ticker", _Ticker)


def _exchanges_down(client):
    client._binance.broken = True
    client._kraken.broken = True


def test_a_wrong_coin_fallback_price_is_refused(monkeypatch):
    client = MarketDataClient(quote_currency="USD")
    client._last_exchange_price["ARB-USD"] = (0.2036, __import__("time").time())
    _exchanges_down(client)
    _yahoo_returning(monkeypatch, 0.000629)               # "ARbit", not Arbitrum
    with pytest.raises(DataUnavailableError, match="implausible"):
        client.get_quote("ARB-USD")


def test_a_plausible_fallback_price_is_used(monkeypatch):
    client = MarketDataClient(quote_currency="USD")
    client._last_exchange_price["ARB-USD"] = (0.2036, __import__("time").time())
    _exchanges_down(client)
    _yahoo_returning(monkeypatch, 0.19)
    assert client.get_quote("ARB-USD").ltp == pytest.approx(0.19)


def test_without_a_reference_the_fallback_is_used(monkeypatch):
    client = MarketDataClient(quote_currency="USD")
    _exchanges_down(client)
    _yahoo_returning(monkeypatch, 0.19)
    assert client.get_quote("ARB-USD").ltp == pytest.approx(0.19)


def test_a_stale_reference_is_ignored(monkeypatch):
    import time as _time

    client = MarketDataClient(quote_currency="USD")
    client._last_exchange_price["ARB-USD"] = (0.2036, _time.time() - 4 * 86400)
    _exchanges_down(client)
    _yahoo_returning(monkeypatch, 0.12)                   # a real 40% fall over four days is possible
    assert client.get_quote("ARB-USD").ltp == pytest.approx(0.12)


def test_good_exchange_quotes_are_remembered(monkeypatch):
    client = MarketDataClient(quote_currency="USD")
    from papertrader.data.crypto_client import CryptoQuote

    monkeypatch.setattr(client._binance, "get_quote", lambda s: CryptoQuote(
        symbol=s, ltp=0.2, prev_close=0.2, week52_high=0.3, week52_low=0.1, volume=1.0,
        timestamp=datetime.now()))
    client.get_quote("ARB-USD")
    assert client._last_exchange_price["ARB-USD"][0] == 0.2


# --- scripts/correct_trade_price.py ---------------------------------------------------

def _correct_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("correct_trade_price", "scripts/correct_trade_price.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bad_arb_ledger(tmp_path):
    from papertrader.portfolio.broker import PaperBroker
    from papertrader.portfolio.storage import Storage

    storage = Storage(str(tmp_path / "tp.db"), 100_000.0)
    broker = PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=0.0, fee_pct=0.1)
    broker.buy("ARB-USD", 300, 19.71, "entry")
    broker.sell("ARB-USD", 300, 0.06, "stop_loss (8.0% below entry 19.71)")   # the bad print
    sell = [t for t in storage.get_trades(limit=5) if t.side == "SELL"][0]
    return storage, broker, sell


def test_correcting_a_bad_sale_fixes_its_pnl_and_the_cash(tmp_path):
    mod = _correct_module()
    storage, broker, sell = _bad_arb_ledger(tmp_path)
    cash_before = storage.get_cash()
    assert sell.realized_pnl < -5_800

    p = mod.plan(storage.db_path, sell.id, 19.40, fee_pct=0.1, flat_charges=0.0)
    mod.apply(storage.db_path, p)

    fixed = [t for t in storage.get_trades(limit=5) if t.side == "SELL"][0]
    # what the sale would have been at the real price: (19.40 - 19.71) x 300 less both fees
    expected = (19.40 - 19.71) * 300 - 19.71 * 300 * 0.001 - 19.40 * 300 * 0.001
    assert fixed.realized_pnl == pytest.approx(expected, abs=0.01)
    assert fixed.price == 19.40 and "price corrected from 0.06" in fixed.reason
    assert storage.get_cash() == pytest.approx(cash_before + p["cash_delta"], abs=0.01)
    # the books balance: cash = starting capital + realized P&L, with nothing open
    assert storage.get_cash() == pytest.approx(100_000.0 + storage.get_total_realized_pnl(), abs=0.01)


def test_a_correction_cannot_be_applied_twice_or_to_a_buy(tmp_path):
    mod = _correct_module()
    storage, _, sell = _bad_arb_ledger(tmp_path)
    mod.apply(storage.db_path, mod.plan(storage.db_path, sell.id, 19.40, 0.1, 0.0))
    with pytest.raises(ValueError, match="already been corrected"):
        mod.plan(storage.db_path, sell.id, 19.40, 0.1, 0.0)
    buy = [t for t in storage.get_trades(limit=5) if t.side == "BUY"][0]
    with pytest.raises(ValueError, match="only a SELL"):
        mod.plan(storage.db_path, buy.id, 19.40, 0.1, 0.0)
    with pytest.raises(ValueError, match="no trade"):
        mod.plan(storage.db_path, 9999, 19.40, 0.1, 0.0)


def test_a_dry_run_changes_nothing(tmp_path):
    mod = _correct_module()
    storage, _, sell = _bad_arb_ledger(tmp_path)
    cash = storage.get_cash()
    mod.plan(storage.db_path, sell.id, 19.40, 0.1, 0.0)
    assert storage.get_cash() == cash
    assert [t for t in storage.get_trades(limit=5) if t.side == "SELL"][0].price == pytest.approx(0.06)
