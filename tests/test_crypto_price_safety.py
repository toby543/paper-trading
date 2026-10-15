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


# --- strategy fixes -------------------------------------------------------------------

def test_breakout_retest_exit_waits_for_the_buffer(monkeypatch):
    from papertrader.data.nse_client import Quote
    from papertrader.portfolio.models import Position
    from papertrader.strategy import crypto_breakout_retest as br

    monkeypatch.setattr(br, "_history_as_of", lambda history, when: pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(br, "_find_base_and_breakout", lambda *a, **k: 100.0)   # the entry's breakout level
    pos = Position(symbol="X-USD", quantity=1, avg_price=101.0, entry_date="2026-10-01T00:00:00",
                   highest_close_since_entry=101.0)

    def exit_at(price, buffer):
        q = Quote("X-USD", price, 100.0, 120.0, 80.0, 1.0, datetime.now(), "test")
        cfg = {"stop_loss_pct": 50, "trailing_stop_pct": 50,
               "crypto_breakout_retest": {"retest_exit_buffer_pct": buffer}}
        return br.check_exit(pos, q, pd.DataFrame(), cfg)

    assert exit_at(98.0, 0)[0]                         # old behaviour: any dip under the level sells
    assert not exit_at(98.0, 5)[0]                     # a 2% wick is now ridden out
    sold, reason = exit_at(94.0, 5)
    assert sold and "by more than 5%" in reason


def test_crypto_trend_profiles_use_the_bitcoin_filter_and_the_tested_settings():
    import yaml

    def load(name):
        return yaml.safe_load(open(f"profiles/{name}.yaml", encoding="utf-8"))

    for name in ("crypto_trend_pullback", "crypto_momentum", "crypto_breakout", "crypto_breakout_retest"):
        assert load(name)["regime"] == {"enabled": True, "index_symbol": "BTC-USD", "ma_days": 100}, name
    for name in ("crypto_mean_reversion", "crypto_pairs_trading"):   # buy-the-dip / market-neutral
        assert load(name)["regime"]["enabled"] is False, name
    assert load("crypto_trend_pullback")["risk"]["take_profit_pct"] == 20.0
    assert load("crypto_breakout_retest")["strategy"]["crypto_breakout_retest"]["retest_exit_buffer_pct"] == 5.0


def test_the_live_engine_blocks_crypto_buys_while_bitcoin_is_below_its_100_day_average(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "m.db"))
    engine = TradingEngine(cfg, profile_name="crypto_momentum")
    falling = pd.DataFrame({"Close": [200.0 - i for i in range(150)]})      # ends well below its 100-day mean
    rising = pd.DataFrame({"Close": [50.0 + i for i in range(150)]})
    engine.data.get_index_history = lambda symbol, *a, **k: falling
    assert engine.market_regime_ok() is False
    engine.data.get_index_history = lambda symbol, *a, **k: rising
    assert engine.market_regime_ok() is True


def test_correction_also_shifts_the_equity_curve_from_the_sale_on(tmp_path):
    import sqlite3

    mod = _correct_module()
    storage, _, sell = _bad_arb_ledger(tmp_path)
    conn = sqlite3.connect(storage.db_path)
    with conn:
        conn.executemany("INSERT INTO equity_curve VALUES (?, ?, ?, ?)", [
            ("2000-01-01T00:00:00", 100_000.0, 0.0, 100_000.0),                    # before the sale: untouched
            ("2999-01-01T00:00:00", 94_000.0, 0.0, 94_000.0),                      # after it: shifted
        ])
    conn.close()
    p = mod.plan(storage.db_path, sell.id, 19.40, 0.1, 0.0)
    mod.apply(storage.db_path, p)
    curve = {r["timestamp"]: r["total_equity"] for r in storage.get_equity_curve(limit=10)}
    assert curve["2000-01-01T00:00:00"] == pytest.approx(100_000.0)
    assert curve["2999-01-01T00:00:00"] == pytest.approx(94_000.0 + p["cash_delta"])


def test_curve_repair_for_a_trade_corrected_by_the_earlier_script(tmp_path):
    """The Pi's ARB trade was re-priced before the script shifted the curve."""
    import sqlite3

    mod = _correct_module()
    storage, _, sell = _bad_arb_ledger(tmp_path)
    p = mod.plan(storage.db_path, sell.id, 19.40, 0.1, 0.0)
    # what the earlier version did: trade and cash only, no curve shift, no curve mark
    conn = sqlite3.connect(storage.db_path)
    with conn:
        conn.execute("UPDATE trades SET price = ?, charges = ?, realized_pnl = ?, reason = ? WHERE id = ?",
                     (p["new_price"], p["new_charges"], p["new_realized"],
                      p["new_reason"].replace(mod.CURVE_MARK, ""), sell.id))
        conn.execute("UPDATE account SET cash = cash + ?", (p["cash_delta"],))
        conn.execute("INSERT INTO equity_curve VALUES ('2999-01-01T00:00:00', 94000.0, 0.0, 94000.0)")
    conn.close()

    r = mod.plan_curve_repair(storage.db_path, sell.id, 0.1, 0.0)
    assert r["cash_delta"] == pytest.approx(p["cash_delta"], abs=0.01)     # recovered from the reason text
    mod.apply_curve_repair(storage.db_path, r)
    assert storage.get_equity_curve(limit=1)[0]["total_equity"] == pytest.approx(94_000.0 + p["cash_delta"], abs=0.01)
    with pytest.raises(ValueError, match="already been adjusted"):
        mod.plan_curve_repair(storage.db_path, sell.id, 0.1, 0.0)


def test_price_cache_never_reuses_history_cached_under_the_wrong_coin(tmp_path, monkeypatch):
    from papertrader.data import price_cache

    monkeypatch.setattr(price_cache, "CACHE_DIR", str(tmp_path))
    assert price_cache._cache_path("ARB-USD").endswith("ARB11841-USD.csv")
    assert price_cache._cache_path("BTC-USD").endswith("BTC-USD.csv")
    assert price_cache._cache_path("RELIANCE").endswith("RELIANCE.csv")


# --- fixes from the advanced review ------------------------------------------------------

def test_the_nse_api_comes_back_after_its_cooldown(monkeypatch):
    from papertrader.data import nse_client

    client = MarketDataClient()
    now = [1_000_000.0]
    monkeypatch.setattr(nse_client.time, "time", lambda: now[0])
    client._nse_broken = True
    assert client._nse_broken
    now[0] += client._NSE_COOLDOWN_SECONDS + 1
    assert not client._nse_broken                            # no longer "for the rest of this run"


def test_history_drops_a_trailing_bar_that_has_no_prices(monkeypatch):
    nan = float("nan")
    frame = pd.DataFrame(
        {"Open": [1.0, 2.0, nan], "High": [1.0, 2.0, nan], "Low": [1.0, 2.0, nan], "Close": [1.0, 2.0, nan],
         "Volume": [10.0, 10.0, 5.0]}, index=pd.to_datetime(["2026-10-01", "2026-10-05", "2026-10-06"]))

    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return frame

    monkeypatch.setattr("yfinance.Ticker", _Ticker)
    df = MarketDataClient().get_history("RPEL")
    assert list(df["Close"]) == [1.0, 2.0]


def _stale_quote_engine(tmp_path, monkeypatch, profile):
    from datetime import timedelta
    from papertrader.config import Config
    from papertrader.data.nse_client import Quote
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "e.db"))
    engine = TradingEngine(cfg, profile_name=profile)
    yesterday = datetime.now().date() - timedelta(days=1)

    class _Data:
        allow_eod_fallback = False

        def get_quote(self, symbol):
            return Quote(symbol, 50.0, 100.0, 120.0, 40.0, 1.0, datetime.now(), "yfinance", yesterday)

        def get_history(self, symbol, period="1y"):
            return pd.DataFrame({"Close": [100.0] * 60, "High": [101.0] * 60, "Low": [99.0] * 60,
                                 "Open": [100.0] * 60, "Volume": [1e6] * 60},
                                index=pd.date_range(end="2026-10-05", periods=60))

    engine.data = _Data()
    return engine


def test_an_automated_stop_does_not_fire_on_an_earlier_days_price(tmp_path, monkeypatch):
    engine = _stale_quote_engine(tmp_path, monkeypatch, "trend_pullback")
    engine.trades_24_7 = False
    engine.broker.buy("AAA", 10, 100.0, "entry")             # a 50 quote is far below the 8% stop
    engine.check_exits()
    assert "AAA" in engine.broker.positions()                # skipped: the price is from yesterday


def test_manual_stops_do_not_fire_on_an_earlier_days_price(tmp_path, monkeypatch):
    engine = _stale_quote_engine(tmp_path, monkeypatch, "manual_swing")
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.storage.set_levels("AAA", 95.0, 130.0)
    engine.check_exits()
    assert "AAA" in engine.broker.positions()


def test_only_a_manual_engine_opts_into_the_end_of_day_fallback(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "x.db"))
    assert TradingEngine(cfg, profile_name="manual_swing").data.allow_eod_fallback is True
    assert TradingEngine(cfg, profile_name="trend_pullback").data.allow_eod_fallback is False


def test_settings_save_takes_the_engine_lock_for_the_capital_sync():
    source = open("src/papertrader/web/app.py", encoding="utf-8").read()
    assert "with engine._state_lock:\n                        engine.sync_untouched_capital()" in source


def test_capital_adoption_takes_the_write_lock_before_it_looks(tmp_path):
    import sqlite3

    from papertrader.portfolio.storage import Storage

    storage = Storage(str(tmp_path / "w.db"), 100_000.0)
    blocker = sqlite3.connect(storage.db_path, timeout=0.2)
    blocker.execute("BEGIN IMMEDIATE")                       # another process holds the write lock
    try:
        with pytest.raises(sqlite3.OperationalError):
            storage.adopt_starting_capital_if_untouched(500_000.0)   # waits for it, instead of racing it
    finally:
        blocker.rollback()
        blocker.close()
    assert storage.adopt_starting_capital_if_untouched(500_000.0) is True
