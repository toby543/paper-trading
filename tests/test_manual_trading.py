"""Manual Swing profile: stock search, order validation, fills, and the API."""
import csv
import os
import tempfile
import threading
from datetime import datetime

import pytest

from papertrader.data.nse_client import DataUnavailableError, Quote
from papertrader.data import symbol_search
from papertrader.engine.manual import ManualOrderError, ManualTrader
from papertrader.portfolio.broker import PaperBroker
from papertrader.portfolio.storage import Storage
from papertrader.risk.risk_manager import RiskManager
from papertrader.web.settings_schema import applies_to_mode


# --- search ---------------------------------------------------------------

@pytest.fixture
def directory(tmp_path):
    path = tmp_path / "dir.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "name", "exchange"])
        w.writerows([
            ("RELIANCE", "Reliance Industries Limited", "NSE"),
            ("RELINFRA", "Reliance Infrastructure Limited", "NSE"),
            ("TATAMOTORS", "Tata Motors Limited", "NSE"),
            ("TATAPOWER", "The Tata Power Company Limited", "NSE"),
            ("ABB.BO", "ABB Something Limited", "BSE"),
            ("ABB", "ABB India Limited", "NSE"),
        ])
    return str(path)


def test_search_ranks_exact_symbol_first(directory):
    assert symbol_search.search("abb", path=directory)[0].symbol == "ABB"


def test_search_matches_company_name_words(directory):
    symbols = [e.symbol for e in symbol_search.search("power", path=directory)]
    assert symbols == ["TATAPOWER"]


def test_search_prefers_nse_over_bse_at_equal_rank(directory):
    symbols = [e.symbol for e in symbol_search.search("abb", path=directory)]
    assert symbols.index("ABB") < symbols.index("ABB.BO")


def test_name_starting_with_the_query_beats_a_name_merely_containing_it(tmp_path):
    path = tmp_path / "d.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "name", "exchange"])
        w.writerows([("HCL-INSYS", "HCL Infosystems Limited", "NSE"), ("INFY", "Infosys Limited", "NSE")])
    assert symbol_search.search("infosys", path=str(path))[0].symbol == "INFY"


def test_blank_query_returns_nothing(directory):
    assert symbol_search.search("   ", path=directory) == []


def test_shipped_directory_includes_etfs():
    for query in ("niftybees", "nippon india etf nifty 50 bees", "nifty 50 bees"):
        assert symbol_search.search(query)[0].symbol == "NIFTYBEES", query


def test_shipped_directory_finds_a_known_stock():
    assert any(e.symbol == "RELIANCE" for e in symbol_search.search("reliance"))
    assert symbol_search.find("RELIANCE") is not None
    assert symbol_search.find("NOSUCHSYMBOL123") is None


# --- trader ---------------------------------------------------------------

class _FakeData:
    def __init__(self, ltp=100.0):
        self.ltp = ltp
        self.fail = False
        self.bar_date = None

    def get_quote(self, symbol):
        if self.fail:
            raise DataUnavailableError("down")
        return Quote(symbol, self.ltp, 99.0, 120.0, 80.0, 1000.0, datetime.now(), "fake", self.bar_date)


@pytest.fixture
def trader_env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = Storage(path, 100_000.0)
    broker = PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0)
    risk = RiskManager(max_open_positions=2, position_size_pct_of_equity=8.0,
                       max_cash_deployed_per_scan_pct=40.0, fractional_quantities=False)
    data = _FakeData()
    state = {"open": True}
    trader = ManualTrader(
        broker=broker, data=data, risk=risk, is_market_open=lambda: state["open"],
        lock=threading.Lock(), symbol_exists=lambda s: s in {"AAA", "BBB", "CCC"},
    )
    yield trader, broker, data, state
    del storage
    try:
        os.unlink(path)
    except PermissionError:
        pass


def test_buy_then_sell_round_trip(trader_env):
    trader, broker, data, _ = trader_env
    out = trader.place_order("aaa", "buy", 10, "breakout watch")
    assert out["symbol"] == "AAA" and out["quantity"] == 10
    assert broker.positions()["AAA"].quantity == 10
    data.ltp = 110.0
    sold = trader.place_order("AAA", "SELL", 10)
    assert sold["realized_pnl"] > 0
    assert "AAA" not in broker.positions()


def test_note_is_recorded_in_the_trade_reason(trader_env):
    trader, broker, _, _ = trader_env
    trader.place_order("AAA", "BUY", 1, "  earnings\n gap  ")
    reasons = [t.reason for t in broker.storage.get_trades(limit=5)]
    assert reasons == ["manual: earnings gap"]


@pytest.mark.parametrize("qty", [0, -3, 2.5, "abc", None, float("nan"), float("inf")])
def test_bad_quantities_are_rejected(trader_env, qty):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError):
        trader.place_order("AAA", "BUY", qty)
    assert not broker.positions()


def test_unknown_symbol_is_rejected(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="Unknown symbol"):
        trader.place_order("ZZZ", "BUY", 1)
    assert not broker.positions()


def test_market_closed_schedules_the_order_unless_after_hours_fills_are_allowed(trader_env):
    trader, broker, _, state = trader_env
    state["open"] = False
    out = trader.place_order("AAA", "BUY", 1)
    assert out["status"] == "scheduled" and not broker.positions()
    trader.allow_when_market_closed = True
    assert trader.place_order("BBB", "BUY", 1)["status"] == "filled"
    assert "BBB" in broker.positions()


def test_cannot_sell_what_you_do_not_hold_or_more_than_you_hold(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="don't hold"):
        trader.place_order("AAA", "SELL", 1)
    trader.place_order("AAA", "BUY", 5)
    with pytest.raises(ManualOrderError, match="only hold"):
        trader.place_order("AAA", "SELL", 6)
    assert broker.positions()["AAA"].quantity == 5


def test_insufficient_funds_is_a_clean_rejection(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="Need"):
        trader.place_order("AAA", "BUY", 1_000_000)
    assert not broker.positions()


def test_max_open_positions_applies_to_new_symbols_only(trader_env):
    trader, broker, _, _ = trader_env
    trader.place_order("AAA", "BUY", 1)
    trader.place_order("BBB", "BUY", 1)
    with pytest.raises(ManualOrderError, match="Maximum open positions"):
        trader.place_order("CCC", "BUY", 1)
    trader.place_order("AAA", "BUY", 1)  # adding to an existing holding is fine
    assert broker.positions()["AAA"].quantity == 2


def test_no_price_means_no_trade(trader_env):
    trader, broker, data, _ = trader_env
    data.fail = True
    with pytest.raises(ManualOrderError, match="No live price"):
        trader.place_order("AAA", "BUY", 1)
    data.fail = False
    data.ltp = float("nan")
    with pytest.raises(ManualOrderError, match="No valid price"):
        trader.place_order("AAA", "BUY", 1)
    assert not broker.positions()


def test_quote_reports_holding_and_cost(trader_env):
    trader, _, _, _ = trader_env
    trader.place_order("AAA", "BUY", 4)
    q = trader.quote("AAA")
    assert q["held_quantity"] == 4 and q["ltp"] == 100.0
    # the estimate for any quantity must equal what the broker would really charge
    for qty in (1, 7, 250):
        estimate = q["buy_cost_per_share"] * qty + q["buy_cost_fixed"]
        assert estimate == pytest.approx(trader.broker.estimated_buy_cost(100.0, qty))
    assert q["buy_cost_fixed"] == pytest.approx(10.0)  # the flat charge, counted once


# --- settings visibility ----------------------------------------------------

def test_manual_profile_only_offers_settings_that_apply():
    assert not applies_to_mode(("risk", "stop_loss_pct"), "manual")
    assert not applies_to_mode(("strategy", "min_ltp_inr"), "manual")
    assert not applies_to_mode(("engine", "scan_interval_minutes"), "manual")
    assert applies_to_mode(("risk", "max_open_positions"), "manual")
    assert applies_to_mode(("execution", "slippage_bps"), "manual")


def test_manual_profile_file_is_configured_as_manual():
    import yaml

    profile = yaml.safe_load(open("profiles/manual_swing.yaml"))
    assert profile["strategy_mode"] == "manual"
    assert profile["category"] == "manual"  # its own dashboard tab


# --- HTTP API -----------------------------------------------------------------

class _FakeEngine:
    def __init__(self, name, manual, trader=None):
        self.profile_name = name
        self.is_manual = manual
        self._trader = trader

    def manual_trader(self):
        return self._trader


@pytest.fixture
def client(trader_env, monkeypatch):
    from papertrader.config import Config
    from papertrader.web import auth
    from papertrader.web.app import create_app

    monkeypatch.setattr(auth, "load_auth_store", lambda: None)
    trader = trader_env[0]
    engines = {
        "manual_swing": _FakeEngine("manual_swing", True, trader),
        "auto": _FakeEngine("auto", False),
    }
    cfg = Config.load()
    cfg.raw["active_profile"] = "manual_swing"
    app = create_app(engines, cfg)
    return app.test_client()


def test_api_search(client):
    data = client.get("/api/manual/search?q=reliance").get_json()
    assert any(r["symbol"] == "RELIANCE" for r in data["results"])


def test_api_order_and_quote(client):
    quote = client.get("/api/manual/quote?profile=manual_swing&symbol=AAA").get_json()
    assert quote["ok"] and quote["ltp"] == 100.0
    out = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 3}).get_json()
    assert out["ok"] and out["quantity"] == 3


def test_api_rejects_non_manual_profiles_and_bad_orders(client):
    r = client.post("/api/manual/order", json={
        "profile": "auto", "symbol": "AAA", "side": "BUY", "quantity": 1})
    assert r.status_code == 400
    r = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 0})
    assert r.status_code == 400 and not r.get_json()["ok"]
    r = client.get("/api/manual/quote?profile=auto&symbol=AAA")
    assert r.status_code == 400


def test_api_order_requires_json(client):
    r = client.post("/api/manual/order", data="profile=manual_swing&symbol=AAA&side=BUY&quantity=1",
                    content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415


# --- engine: a manual profile never trades on its own ------------------------

def test_manual_engine_places_no_automatic_entries_or_exits(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    assert engine.is_manual

    engine.data = _FakeData(ltp=40.0)  # 60% below the entry: any auto stop-loss would fire
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.check_exits()
    assert "AAA" in engine.broker.positions()

    engine.scan_for_entries()
    assert list(engine.broker.positions()) == ["AAA"]
    assert engine.find_candidates() == []
    assert engine.storage.get_last_scan_at()  # still reports as alive on the dashboard


def test_manual_engine_keeps_the_peak_price_current(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    engine.data = _FakeData(ltp=100.0)
    engine.broker.buy("AAA", 1, 100.0, "manual")
    engine.data.ltp = 130.0
    engine.check_exits()
    assert engine.broker.positions()["AAA"].highest_close_since_entry == 130.0


# --- bug-check regressions -----------------------------------------------------

def test_quote_payload_never_contains_nan(trader_env):
    """NaN is not valid JSON; a missing previous close or 52-week figure
    (thin or new listings) used to make the browser's res.json() throw."""
    import json

    trader, _, data, _ = trader_env
    data.get_quote = lambda s: Quote(s, 100.0, float("nan"), float("nan"), float("inf"), 1.0, datetime.now(), "x")
    q = trader.quote("AAA")
    assert q["prev_close"] is None and q["week52_high"] is None and q["week52_low"] is None
    json.dumps(q, allow_nan=False)  # raises on NaN/Infinity


def test_order_is_refused_when_the_price_moved_since_the_quote(trader_env):
    trader, broker, data, _ = trader_env
    data.ltp = 105.0  # +5% from the 100.0 the user saw
    with pytest.raises(ManualOrderError, match="price moved"):
        trader.place_order("AAA", "BUY", 1, expected_price=100.0)
    assert not broker.positions()
    trader.place_order("AAA", "BUY", 1, expected_price=104.0)  # within tolerance
    assert "AAA" in broker.positions()


def test_expected_price_is_optional_and_validated(trader_env):
    trader, broker, _, _ = trader_env
    trader.place_order("AAA", "BUY", 1)
    with pytest.raises(ManualOrderError, match="Expected price"):
        trader.place_order("AAA", "BUY", 1, expected_price="abc")


def test_user_text_is_escaped_before_it_reaches_the_page():
    """The note is stored in the trade reason and the trade log renders it
    via innerHTML, so it must go through escapeText."""
    source = open("src/papertrader/web/templates/index.html", encoding="utf-8").read()
    assert "${escapeText(t.reason)}" in source
    assert '<div class="ticker-reason">${t.reason}</div>' not in source
    assert "${escapeText(r.name)}" in source


def test_api_passes_expected_price_through(client):
    r = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 1, "expected_price": 50})
    assert r.status_code == 400 and "price moved" in r.get_json()["error"]


def test_manual_profile_hides_every_strategy_panel_and_the_regime_pill():
    from papertrader.web.data_api import _market_regime

    class _E:
        is_manual = True
        regime_cfg = {"enabled": True, "index_symbol": "^NSEI", "ma_days": 200}

    assert _market_regime(_E())["enabled"] is False


def test_manual_page_hides_strategy_panels(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="manualPanel"' in html
    hidden = html.split("A manual profile runs no strategy")[1].split("</style>")[0]
    for panel in ("watchlistPanel", "backtestPanel", "rulesPanel", "settingsPanel", "comparePanel", "allUnrealizedWrap"):
        assert f"#{panel}" in hidden


def test_manual_profile_gets_its_own_dashboard_tab(client):
    html = client.get("/").get_data(as_text=True)
    assert 'data-category="manual"' in html and "Manual Trading" in html
    assert '"manual_swing": "manual"' in html  # PROFILE_CATEGORIES maps it to that tab


# --- entry price, stop loss, target, limit orders --------------------------------

def _storage(broker):
    return broker.storage


def test_buy_records_stop_loss_and_target(trader_env):
    trader, broker, _, _ = trader_env
    out = trader.place_order("AAA", "BUY", 10, stop_loss=95, target_price=120)
    assert out["status"] == "filled" and out["stop_loss"] == 95
    assert _storage(broker).get_levels()["AAA"] == {"stop_loss": 95.0, "target_price": 120.0}
    assert trader.quote("AAA")["held_stop_loss"] == 95.0


@pytest.mark.parametrize("sl,tg,msg", [(100, None, "Stop loss must be below"), (105, None, "Stop loss must be below"),
                                       (None, 100, "Target must be above"), (None, 90, "Target must be above"),
                                       (-5, None, "positive"), ("abc", None, "number")])
def test_bad_levels_are_rejected_and_nothing_trades(trader_env, sl, tg, msg):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match=msg):
        trader.place_order("AAA", "BUY", 1, stop_loss=sl, target_price=tg)
    assert not broker.positions()


def test_levels_cannot_be_sent_with_a_sell(trader_env):
    trader, _, _, _ = trader_env
    trader.place_order("AAA", "BUY", 2)
    with pytest.raises(ManualOrderError, match="buy orders only"):
        trader.place_order("AAA", "SELL", 1, stop_loss=90)


def test_entry_at_or_above_the_live_price_is_a_market_buy(trader_env):
    trader, broker, _, _ = trader_env
    assert trader.place_order("AAA", "BUY", 1, entry_price=100.0)["status"] == "filled"
    assert trader.place_order("BBB", "BUY", 1, entry_price=130.0)["status"] == "filled"
    assert set(broker.positions()) == {"AAA", "BBB"}


def test_entry_below_the_live_price_places_a_pending_limit_order(trader_env):
    trader, broker, _, _ = trader_env
    cash = broker.cash()
    out = trader.place_order("AAA", "BUY", 10, entry_price=90, stop_loss=85, target_price=110)
    assert out["status"] == "pending" and out["limit_price"] == 90.0
    assert not broker.positions() and broker.cash() == cash
    assert [o["symbol"] for o in broker.storage.get_pending_orders()] == ["AAA"]


def test_limit_order_levels_are_checked_against_the_entry_price(trader_env):
    trader, _, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="below the entry price"):
        trader.place_order("AAA", "BUY", 1, entry_price=90, stop_loss=92)
    with pytest.raises(ManualOrderError, match="above the entry price"):
        trader.place_order("AAA", "BUY", 1, entry_price=90, target_price=89)


def test_limit_order_is_refused_when_it_could_never_be_afforded(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="would need about"):
        trader.place_order("AAA", "BUY", 5_000, entry_price=90)
    assert not broker.storage.get_pending_orders()


def test_stop_loss_sells_the_whole_position_automatically(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10, stop_loss=95, target_price=120)
    data.ltp = 94.0
    trader.run_automation()
    assert "AAA" not in broker.positions()
    last = broker.storage.get_trades(limit=1)[0]
    assert last.side == "SELL" and "stop-loss" in last.reason and last.realized_pnl < 0
    assert broker.storage.get_levels() == {}  # levels die with the position


def test_target_sells_automatically_and_in_between_does_nothing(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10, stop_loss=95, target_price=120)
    data.ltp = 110.0
    trader.run_automation()
    assert "AAA" in broker.positions()
    data.ltp = 121.0
    trader.run_automation()
    assert "AAA" not in broker.positions()
    assert "target" in broker.storage.get_trades(limit=1)[0].reason


def test_a_position_without_levels_is_never_sold_automatically(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10)
    data.ltp = 1.0
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_a_reentry_does_not_inherit_the_previous_positions_stop(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 5, stop_loss=90)
    trader.place_order("AAA", "SELL", 5)
    trader.place_order("AAA", "BUY", 5)
    assert broker.storage.get_levels() == {}


def test_partial_sell_keeps_the_levels(trader_env):
    trader, broker, _, _ = trader_env
    trader.place_order("AAA", "BUY", 10, stop_loss=90, target_price=130)
    trader.place_order("AAA", "SELL", 4)
    assert broker.storage.get_levels()["AAA"]["stop_loss"] == 90.0


def test_adding_to_a_holding_keeps_levels_left_blank_and_replaces_given_ones(trader_env):
    trader, broker, _, _ = trader_env
    trader.place_order("AAA", "BUY", 5, stop_loss=90, target_price=130)
    trader.place_order("AAA", "BUY", 5, stop_loss=92)
    assert broker.storage.get_levels()["AAA"] == {"stop_loss": 92.0, "target_price": 130.0}


def test_set_levels_updates_and_clears_and_validates(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 5)
    trader.set_levels("AAA", 92, 140)
    assert broker.storage.get_levels()["AAA"] == {"stop_loss": 92.0, "target_price": 140.0}
    with pytest.raises(ManualOrderError, match="below the current price"):
        trader.set_levels("AAA", 100, None)
    trader.set_levels("AAA", None, None)
    assert broker.storage.get_levels() == {}
    with pytest.raises(ManualOrderError, match="don't hold"):
        trader.set_levels("BBB", 90, None)


def _pending(broker):
    return broker.storage.get_pending_orders()


def test_limit_order_fills_when_the_price_reaches_it_and_applies_its_levels(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10, entry_price=90, stop_loss=85, target_price=110, note="dip")
    trader.run_automation()  # price still 100: nothing happens
    assert not broker.positions() and len(_pending(broker)) == 1

    data.ltp = 89.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 10
    assert broker.storage.get_levels()["AAA"] == {"stop_loss": 85.0, "target_price": 110.0}
    assert not _pending(broker)
    assert broker.storage.get_recent_orders()[0]["status"] == "filled"
    assert broker.storage.get_trades(limit=1)[0].reason == "manual limit buy: dip"


def test_limit_order_cannot_fill_twice(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10, entry_price=90)
    data.ltp = 80.0
    trader.run_automation()
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 10


def test_limit_order_is_cancelled_with_a_reason_if_cash_ran_out(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 900, entry_price=90)   # affordable now (about 81k of 100k)
    trader.place_order("BBB", "BUY", 800)                   # a market buy uses the cash first
    data.ltp = 80.0
    trader.run_automation()
    order = [o for o in broker.storage.get_recent_orders() if o["symbol"] == "AAA"][0]
    assert order["status"] == "cancelled" and "Need" in order["detail"]
    assert "AAA" not in broker.positions()


def test_limit_order_is_cancelled_when_the_position_limit_filled_up(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 1, entry_price=90)  # pending
    trader.place_order("BBB", "BUY", 1)
    trader.place_order("CCC", "BUY", 1)                  # fixture allows 2 open positions: these fill it
    data.ltp = 85.0
    trader.run_automation()
    order = broker.storage.get_recent_orders()[0]
    assert order["status"] == "cancelled" and "maximum open positions" in order["detail"]
    assert "AAA" not in broker.positions()


def test_cancel_order(trader_env):
    trader, broker, data, _ = trader_env
    oid = trader.place_order("AAA", "BUY", 1, entry_price=90)["order_id"]
    assert trader.cancel_order(oid)["status"] == "cancelled"
    data.ltp = 50.0
    trader.run_automation()
    assert not broker.positions()
    with pytest.raises(ManualOrderError, match="no longer pending"):
        trader.cancel_order(oid)


def test_manual_engine_sells_on_a_stop_loss_during_its_normal_exit_check(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    engine.data = _FakeData(ltp=100.0)
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.storage.set_levels("AAA", 95.0, 120.0)

    engine.data.ltp = 94.0
    engine.check_exits()  # the call run_forever() makes, with the lock already held
    assert "AAA" not in engine.broker.positions()


def test_summary_rows_carry_levels_and_insights_ignore_phantom_global_stops(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    from papertrader.web.data_api import build_summary

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    engine.data = _FakeData(ltp=100.0)
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.storage.set_levels("AAA", 99.0, 130.0)

    summary = build_summary(engine)
    row = summary["positions"][0]
    assert row["stop_loss"] == 99.0 and row["target_price"] == 130.0
    titles = [i["title"] for i in summary["insights"]]
    assert any("above its stop-loss" in t for t in titles)  # the user's own stop, about 1% away

    engine.storage.set_levels("AAA", None, None)
    titles = [i["title"] for i in build_summary(engine)["insights"]]
    assert not any("stop" in t for t in titles)  # no phantom stop from the shared defaults


def test_api_levels_orders_and_cancel(client):
    ok = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 2,
        "stop_loss": 90, "target_price": 130}).get_json()
    assert ok["ok"] and ok["status"] == "filled"
    lv = client.post("/api/manual/levels", json={
        "profile": "manual_swing", "symbol": "AAA", "stop_loss": 92, "target_price": None}).get_json()
    assert lv["ok"] and lv["stop_loss"] == 92.0 and lv["target_price"] is None

    pend = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "BBB", "side": "BUY", "quantity": 1, "entry_price": 80}).get_json()
    assert pend["status"] == "pending"
    orders = client.get("/api/manual/orders?profile=manual_swing").get_json()["orders"]
    assert [o["symbol"] for o in orders] == ["BBB"]
    assert client.post("/api/manual/cancel", json={
        "profile": "manual_swing", "order_id": pend["order_id"]}).get_json()["ok"]
    assert client.get("/api/manual/orders?profile=auto").status_code == 400


# --- scheduling orders while the market is closed -----------------------------------

@pytest.fixture
def closed(trader_env):
    trader, broker, data, state = trader_env
    state["open"] = False
    return trader, broker, data, state


def _open_market(state, data, price=None):
    state["open"] = True
    if price is not None:
        data.ltp = price


def test_a_closed_market_buy_is_queued_without_touching_cash_or_positions(closed):
    trader, broker, _, _ = closed
    cash = broker.cash()
    out = trader.place_order("AAA", "BUY", 10, stop_loss=95, target_price=120)
    assert out["status"] == "scheduled" and out["last_price"] == 100.0
    assert not broker.positions() and broker.cash() == cash
    order = broker.storage.get_pending_orders()[0]
    assert (order["side"], order["order_type"], order["stop_loss"]) == ("BUY", "market_open", 95.0)


def test_scheduled_buy_runs_at_the_open_price_and_applies_its_levels(closed):
    trader, broker, data, state = closed
    trader.place_order("AAA", "BUY", 10, stop_loss=95, target_price=120, note="gap play")
    _open_market(state, data, price=103.0)
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 10
    assert broker.storage.get_levels()["AAA"] == {"stop_loss": 95.0, "target_price": 120.0}
    order = broker.storage.get_recent_orders()[0]
    assert order["status"] == "filled" and "at the open" in order["detail"]
    assert broker.storage.get_trades(limit=1)[0].reason == "manual scheduled buy: gap play"


def test_scheduled_order_waits_while_the_price_is_still_yesterdays(closed):
    from datetime import timedelta

    trader, broker, data, state = closed
    trader.place_order("AAA", "BUY", 5)
    _open_market(state, data)
    data.bar_date = datetime.now().date() - timedelta(days=1)  # Yahoo has not produced today's bar yet
    trader.run_automation()
    assert not broker.positions() and len(broker.storage.get_pending_orders()) == 1
    data.bar_date = datetime.now().date()
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_scheduled_buy_is_cancelled_if_the_open_gaps_through_its_stop_or_target(closed):
    trader, broker, data, state = closed
    trader.place_order("AAA", "BUY", 5, stop_loss=95)
    trader.place_order("BBB", "BUY", 5, target_price=110)
    _open_market(state, data, price=90.0)   # below the stop, not above the target
    trader.run_automation()
    orders = {o["symbol"]: o for o in broker.storage.get_recent_orders()}
    assert orders["AAA"]["status"] == "cancelled" and "stop loss" in orders["AAA"]["detail"]
    assert orders["BBB"]["status"] == "filled"

    state["open"] = False
    data.ltp = 100.0
    trader.place_order("CCC", "BUY", 1, target_price=105)   # queued again for the next open
    _open_market(state, data, price=107.0)
    trader.run_automation()
    assert "target" in [o for o in broker.storage.get_recent_orders() if o["symbol"] == "CCC"][0]["detail"]


def test_scheduled_buy_is_cancelled_with_a_reason_if_cash_is_gone_by_the_open(closed):
    trader, broker, data, state = closed
    trader.place_order("AAA", "BUY", 900)          # about 90k of 100k at the last price
    state["open"] = True
    trader.place_order("BBB", "BUY", 800)          # a live market buy spends it first
    state["open"] = False
    _open_market(state, data)
    trader.run_automation()
    order = [o for o in broker.storage.get_recent_orders() if o["symbol"] == "AAA"][0]
    assert order["status"] == "cancelled" and "Need" in order["detail"]


def test_scheduled_sell_runs_at_the_open(closed):
    trader, broker, data, state = closed
    state["open"] = True
    trader.place_order("AAA", "BUY", 10)
    state["open"] = False
    out = trader.place_order("AAA", "SELL", 4)
    assert out["status"] == "scheduled" and broker.positions()["AAA"].quantity == 10
    _open_market(state, data, price=108.0)
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 6
    assert broker.storage.get_trades(limit=1)[0].reason == "manual scheduled sell"


def test_scheduled_sell_is_cancelled_if_the_shares_are_already_gone(closed):
    trader, broker, data, state = closed
    state["open"] = True
    trader.place_order("AAA", "BUY", 5, stop_loss=95)
    state["open"] = False
    trader.place_order("AAA", "SELL", 5)
    _open_market(state, data, price=90.0)   # the stop sells it first (levels run before queued orders)
    trader.run_automation()
    order = broker.storage.get_recent_orders()[0]
    assert order["status"] == "cancelled" and "no longer hold" in order["detail"]


def test_closed_market_validation_still_applies(closed):
    trader, broker, _, _ = closed
    with pytest.raises(ManualOrderError, match="Stop loss must be below the last price"):
        trader.place_order("AAA", "BUY", 1, stop_loss=101)
    with pytest.raises(ManualOrderError, match="would need about"):
        trader.place_order("AAA", "BUY", 5_000)
    with pytest.raises(ManualOrderError, match="don't hold"):
        trader.place_order("AAA", "SELL", 1)
    with pytest.raises(ManualOrderError, match="Unknown symbol"):
        trader.place_order("ZZZ", "BUY", 1)
    assert not broker.storage.get_pending_orders()


def test_closed_market_entry_below_the_last_price_is_a_limit_order(closed):
    trader, broker, data, state = closed
    out = trader.place_order("AAA", "BUY", 5, entry_price=95)
    assert out["status"] == "pending" and out["limit_price"] == 95.0
    _open_market(state, data, price=97.0)
    trader.run_automation()
    assert not broker.positions()        # opened above the limit: keeps waiting
    data.ltp = 94.0
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_a_scheduled_order_can_be_cancelled_before_the_open(closed):
    trader, broker, data, state = closed
    oid = trader.place_order("AAA", "BUY", 5)["order_id"]
    trader.cancel_order(oid)
    _open_market(state, data)
    trader.run_automation()
    assert not broker.positions()


def test_pending_orders_table_from_the_earlier_version_is_migrated(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE pending_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, quantity REAL NOT NULL,
            limit_price REAL NOT NULL, stop_loss REAL, target_price REAL, note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            detail TEXT NOT NULL DEFAULT '', resolved_at TEXT);
        INSERT INTO pending_orders (symbol, quantity, limit_price, created_at) VALUES ('AAA', 3, 90, 'x');
    """)
    conn.commit()
    conn.close()
    storage = Storage(path, 100_000.0)
    order = storage.get_pending_orders()[0]
    assert (order["side"], order["order_type"]) == ("BUY", "limit")  # old orders stay limit buys
    storage.add_pending_order("BBB", 1, 0.0, None, None, "", "SELL", "market_open")
    assert len(storage.get_pending_orders()) == 2


def test_quote_tells_the_page_whether_closed_orders_queue(closed):
    trader, _, _, _ = closed
    q = trader.quote("AAA")
    assert q["market_open"] is False and q["queue_when_closed"] is True


def test_api_schedules_when_closed(client, trader_env):
    trader_env[3]["open"] = False
    out = client.post("/api/manual/order", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 2}).get_json()
    assert out["ok"] and out["status"] == "scheduled"


# --- a scheduled buy honours its entry price ------------------------------------------

def test_scheduled_buy_with_an_entry_price_never_pays_more_than_it(closed):
    trader, broker, data, state = closed
    out = trader.place_order("AAA", "BUY", 5, entry_price=100.0)   # equal to the last price
    assert out["status"] == "pending" and out["limit_price"] == 100.0
    assert broker.storage.get_pending_orders()[0]["order_type"] == "limit"

    _open_market(state, data, price=106.0)        # the open gaps above the entry
    trader.run_automation()
    assert not broker.positions()                 # not bought at 106
    data.ltp = 100.0
    trader.run_automation()
    assert broker.positions()["AAA"].avg_price <= 100.1   # entry price, plus slippage only


def test_scheduled_buy_fills_at_the_open_when_that_is_at_or_below_the_entry(closed):
    trader, broker, data, state = closed
    trader.place_order("AAA", "BUY", 5, entry_price=100.0)
    _open_market(state, data, price=98.0)
    trader.run_automation()
    assert "AAA" in broker.positions()
    assert broker.positions()["AAA"].avg_price < 100.0


def test_an_entry_above_the_last_price_is_still_a_ceiling_not_ignored(closed):
    trader, broker, data, state = closed
    out = trader.place_order("AAA", "BUY", 5, entry_price=110.0, stop_loss=105, target_price=130)
    assert out["status"] == "pending" and out["limit_price"] == 110.0
    _open_market(state, data, price=112.0)
    trader.run_automation()
    assert not broker.positions()                 # 112 is above the 110 ceiling
    data.ltp = 109.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 5


def test_scheduled_entry_levels_are_checked_against_the_entry_price(closed):
    trader, _, _, _ = closed
    with pytest.raises(ManualOrderError, match="below the entry price"):
        trader.place_order("AAA", "BUY", 1, entry_price=110.0, stop_loss=105.0 + 6)
    with pytest.raises(ManualOrderError, match="above the entry price"):
        trader.place_order("AAA", "BUY", 1, entry_price=110.0, target_price=109)


def test_scheduled_buy_without_an_entry_price_still_runs_at_the_open(closed):
    trader, broker, data, state = closed
    assert trader.place_order("AAA", "BUY", 5)["status"] == "scheduled"
    _open_market(state, data, price=106.0)
    trader.run_automation()
    assert "AAA" in broker.positions()


# --- a half-formed last bar from Yahoo (the RPEL case) ---------------------------------

def _bars(rows):
    import pandas as pd

    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[1] for r in rows], "Low": [r[1] for r in rows],
         "Close": [r[1] for r in rows], "Volume": [r[2] for r in rows]}, index=idx)


def test_quote_skips_a_last_bar_that_has_volume_but_no_prices(monkeypatch):
    """Yahoo ended RPEL's history with a row of NaN prices and a volume. The
    quote took that row's close, so the ticket said "No valid price"."""
    import math

    from papertrader.data.nse_client import MarketDataClient

    nan = float("nan")
    frame = _bars([("2026-10-01", 1818.9, 100), ("2026-10-05", 1825.3, 83), ("2026-10-06", nan, 51)])

    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return frame

    monkeypatch.setattr("yfinance.Ticker", _Ticker)
    q = MarketDataClient()._quote_from_yfinance("RPEL")
    assert q.ltp == pytest.approx(1825.3) and not math.isnan(q.ltp)
    assert q.prev_close == pytest.approx(1818.9)
    assert str(q.bar_date) == "2026-10-05"


def test_quote_with_no_usable_price_at_all_is_unavailable(monkeypatch):
    from papertrader.data.nse_client import MarketDataClient

    frame = _bars([("2026-10-06", float("nan"), 51)])

    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return frame

    monkeypatch.setattr("yfinance.Ticker", _Ticker)
    with pytest.raises(DataUnavailableError, match="no usable prices"):
        MarketDataClient()._quote_from_yfinance("RPEL")


def test_an_immediate_order_will_not_fill_on_an_earlier_days_price(trader_env):
    from datetime import timedelta

    trader, broker, data, _ = trader_env
    data.bar_date = datetime.now().date() - timedelta(days=2)
    with pytest.raises(ManualOrderError, match="not today"):
        trader.place_order("AAA", "BUY", 1)
    assert not broker.positions()
    trader.allow_when_market_closed = True        # after-hours testing mode skips the check
    assert trader.place_order("AAA", "BUY", 1)["status"] == "filled"


def test_quote_reports_which_day_its_price_is_from(trader_env):
    trader, _, data, _ = trader_env
    assert trader.quote("AAA")["price_date"] is None
    data.bar_date = datetime(2026, 10, 5).date()
    assert trader.quote("AAA")["price_date"] == "2026-10-05"


# --- price fallback chain: live NSE -> Yahoo -> exchange end-of-day file ---------------

NSE_FILE = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
    "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RPEL, EQ, 06-Oct-2026, 1825.30, 1851.20, 1860.00, 1799.00, 1845.00, 1835.60, 1828.72, 51137, 935.15, 7827, 22826, 44.64\n"
    "RPEL, BE, 06-Oct-2026, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1, 1, 1, 1, 1\n"
    "NOCLOSE, EQ, 06-Oct-2026, 10.00, -, -, -, -, -, -, 0, 0, 0, 0, 0\n"
)
BSE_FILE = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,"
    "FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
    "TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2026-10-06,2026-10-06,CM,BSE,STK,500002,INE117A01022,ABB,A,,,,,ABB INDIA LIMITED,7216.95,7300.00,7022.75,7057.90,"
    "7057.90,7115.00,,7052.88,,,2883,20467777.00,620,F1,1,,,,,\n"
)


def test_end_of_day_files_are_parsed_for_both_exchanges():
    from papertrader.data import eod_prices

    nse = eod_prices.parse_nse(NSE_FILE)
    assert nse["RPEL"]["close"] == pytest.approx(1835.60)      # the EQ row, not the BE one
    assert nse["RPEL"]["prev_close"] == pytest.approx(1825.30)
    assert "NOCLOSE" not in nse                                # no price, no entry
    bse = eod_prices.parse_bse(BSE_FILE)
    assert bse["ABB"]["close"] == pytest.approx(7057.90) and bse["ABB"]["high"] == pytest.approx(7300.00)


def test_lookup_routes_bse_symbols_to_the_bse_file(monkeypatch):
    from papertrader.data import eod_prices

    seen = []

    def load(exchange, day, timeout):
        seen.append(exchange)
        return {"ABB": {"close": 1.0}} if exchange == "BSE" else {"RPEL": {"close": 2.0}}

    monkeypatch.setattr(eod_prices, "_load", load)
    assert eod_prices.lookup("ABB.BO", datetime(2026, 10, 6).date())["close"] == 1.0
    assert eod_prices.lookup("RPEL", datetime(2026, 10, 6).date())["close"] == 2.0
    assert seen == ["BSE", "NSE"]


def test_latest_walks_back_over_weekends_and_unpublished_days(monkeypatch):
    from papertrader.data import eod_prices

    published = {datetime(2026, 10, 5).date()}   # Monday only; Tuesday-Thursday not published

    monkeypatch.setattr(eod_prices, "_load",
                        lambda ex, day, t: {"RPEL": {"close": 9.0}} if day in published else None)
    day, row = eod_prices.latest("RPEL", datetime(2026, 10, 8).date())
    assert day == datetime(2026, 10, 5).date() and row["close"] == 9.0
    assert eod_prices.latest("NOSUCH", datetime(2026, 10, 8).date()) is None


def _patch_yahoo(monkeypatch, frame):
    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return frame

    monkeypatch.setattr("yfinance.Ticker", _Ticker)


def test_an_empty_trailing_yahoo_bar_is_filled_from_the_exchange_file(monkeypatch):
    """The RPEL case: Yahoo had no prices for 6 Oct, NSE's file has them."""
    from papertrader.data import eod_prices
    from papertrader.data.nse_client import MarketDataClient

    nan = float("nan")
    _patch_yahoo(monkeypatch, _bars([("2026-10-01", 1818.9, 100), ("2026-10-05", 1825.3, 83), ("2026-10-06", nan, 51)]))
    table = eod_prices.parse_nse(NSE_FILE)
    monkeypatch.setattr(eod_prices, "_load", lambda ex, day, t: table if str(day) == "2026-10-06" else None)

    q = MarketDataClient()._quote_from_yfinance("RPEL")
    assert q.ltp == pytest.approx(1835.60) and q.prev_close == pytest.approx(1825.30)
    assert str(q.bar_date) == "2026-10-06" and q.source == "yfinance+eod"
    assert q.week52_high >= 1860.0                             # the day's high is included


def test_get_quote_falls_back_to_the_exchange_file_when_yahoo_has_nothing(monkeypatch):
    from papertrader.data import eod_prices
    from papertrader.data.nse_client import MarketDataClient

    class _Dead:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            raise RuntimeError("yahoo is down")

    monkeypatch.setattr("yfinance.Ticker", _Dead)
    table = eod_prices.parse_nse(NSE_FILE)
    monkeypatch.setattr(eod_prices, "_load", lambda ex, day, t: table)

    client = MarketDataClient(preferred="yfinance")
    q = client.get_quote("RPEL")
    assert q.source == "eod" and q.ltp == pytest.approx(1835.60)
    assert q.week52_high == 0.0                                # unknown, not invented


def test_get_quote_still_raises_when_every_source_fails(monkeypatch):
    from papertrader.data.nse_client import MarketDataClient

    class _Dead:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            raise RuntimeError("yahoo is down")

    monkeypatch.setattr("yfinance.Ticker", _Dead)
    with pytest.raises(DataUnavailableError):
        MarketDataClient(preferred="yfinance").get_quote("RPEL")


def test_manual_quote_treats_the_unknown_52_week_range_as_missing(trader_env):
    trader, _, data, _ = trader_env
    data.get_quote = lambda s: Quote(s, 100.0, 99.0, 0.0, 0.0, 1.0, datetime.now(), "eod", None)
    q = trader.quote("AAA")
    assert q["week52_high"] is None and q["week52_low"] is None


# --- GTT: good till triggered -----------------------------------------------------------

def _gtts(broker):
    return broker.storage.get_active_gtts()


def test_a_single_gtt_waits_without_touching_cash_or_positions(trader_env):
    trader, broker, _, _ = trader_env
    cash = broker.cash()
    out = trader.create_gtt("AAA", "BUY", 10, trigger_price=90)
    assert out["direction"] == "down" and out["kind"] == "single"
    assert not broker.positions() and broker.cash() == cash
    assert len(_gtts(broker)) == 1


def test_trigger_direction_follows_where_it_sits_against_the_current_price(trader_env):
    trader, _, _, _ = trader_env
    assert trader.create_gtt("AAA", "BUY", 1, trigger_price=90)["direction"] == "down"   # price is 100
    assert trader.create_gtt("BBB", "BUY", 1, trigger_price=110)["direction"] == "up"    # a breakout buy


def test_a_trigger_at_the_current_price_is_refused(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="regular order"):
        trader.create_gtt("AAA", "BUY", 1, trigger_price=100.0)
    assert not _gtts(broker)


def test_a_breakout_gtt_buys_when_the_price_rises_to_the_trigger(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 10, trigger_price=110, stop_loss=104, target_price=130, note="breakout")
    data.ltp = 108.0
    trader.run_automation()
    assert not broker.positions()
    data.ltp = 111.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 10
    assert broker.storage.get_levels()["AAA"] == {"stop_loss": 104.0, "target_price": 130.0}
    gtt = broker.storage.get_recent_gtts()[0]
    assert gtt["status"] == "triggered" and "111.00" in gtt["detail"]
    assert broker.storage.get_trades(limit=1)[0].reason.startswith("manual gtt buy (trigger 110.00): breakout")


def test_a_buy_the_dip_gtt_fires_when_the_price_falls_to_the_trigger(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5, trigger_price=90)
    data.ltp = 95.0
    trader.run_automation()
    assert not broker.positions()
    data.ltp = 89.0
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_a_gtt_fires_only_once(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5, trigger_price=90)
    data.ltp = 85.0
    trader.run_automation()
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 5


def test_a_limit_price_stops_the_order_when_the_price_gaps_past_it(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5, trigger_price=110, limit_price=112)
    data.ltp = 118.0                                   # jumped through both
    trader.run_automation()
    assert not broker.positions()
    gtt = broker.storage.get_recent_gtts()[0]
    assert gtt["status"] == "triggered" and "beyond your limit" in gtt["detail"]


def test_limit_price_rules(trader_env):
    trader, _, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="at or above its trigger"):
        trader.create_gtt("AAA", "BUY", 1, trigger_price=110, limit_price=108)
    trader.place_order("BBB", "BUY", 5)
    with pytest.raises(ManualOrderError, match="at or below its trigger"):
        trader.create_gtt("BBB", "SELL", 5, trigger_price=110, limit_price=112)


def test_a_sell_gtt_needs_the_shares_and_runs_on_its_trigger(trader_env):
    trader, broker, data, _ = trader_env
    with pytest.raises(ManualOrderError, match="don't hold"):
        trader.create_gtt("AAA", "SELL", 1, trigger_price=120)
    trader.place_order("AAA", "BUY", 10)
    with pytest.raises(ManualOrderError, match="only hold"):
        trader.create_gtt("AAA", "SELL", 11, trigger_price=120)
    trader.create_gtt("AAA", "SELL", 4, trigger_price=120)
    data.ltp = 121.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 6


def test_two_leg_gtt_sells_at_the_stop_and_drops_the_target(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10)
    out = trader.create_gtt("AAA", "SELL", 6, trigger_price=95, oco=True, target_trigger=120)
    assert out["kind"] == "oco"
    data.ltp = 96.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 10          # between the two legs: nothing yet
    data.ltp = 94.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 4
    assert "gtt stop-loss" in broker.storage.get_trades(limit=1)[0].reason
    data.ltp = 130.0
    trader.run_automation()
    assert broker.positions()["AAA"].quantity == 4           # the target leg died with the stop leg


def test_two_leg_gtt_sells_at_the_target(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 10)
    trader.create_gtt("AAA", "SELL", 10, trigger_price=95, oco=True, target_trigger=120)
    data.ltp = 121.0
    trader.run_automation()
    assert "AAA" not in broker.positions()
    assert "gtt target" in broker.storage.get_trades(limit=1)[0].reason


def test_two_leg_gtt_validation(trader_env):
    trader, _, _, _ = trader_env
    trader.place_order("AAA", "BUY", 10)
    with pytest.raises(ManualOrderError, match="stop-loss trigger must be below"):
        trader.create_gtt("AAA", "SELL", 5, trigger_price=105, oco=True, target_trigger=120)
    with pytest.raises(ManualOrderError, match="target trigger must be above"):
        trader.create_gtt("AAA", "SELL", 5, trigger_price=95, oco=True, target_trigger=99)
    with pytest.raises(ManualOrderError, match="Enter the target trigger"):
        trader.create_gtt("AAA", "SELL", 5, trigger_price=95, oco=True)
    with pytest.raises(ManualOrderError, match="leave the limit price blank"):
        trader.create_gtt("AAA", "SELL", 5, trigger_price=95, oco=True, target_trigger=120, limit_price=94)
    with pytest.raises(ManualOrderError, match="for selling a holding"):
        trader.create_gtt("BBB", "BUY", 5, trigger_price=95, oco=True, target_trigger=120)


def test_a_gtt_whose_shares_are_already_gone_is_cancelled_not_oversold(trader_env):
    trader, broker, data, _ = trader_env
    trader.place_order("AAA", "BUY", 5)
    trader.create_gtt("AAA", "SELL", 5, trigger_price=120)
    trader.place_order("AAA", "SELL", 5)
    data.ltp = 125.0
    trader.run_automation()
    assert broker.storage.get_recent_gtts()[0]["status"] == "cancelled"


def test_a_gtt_buy_that_cannot_be_afforded_resolves_with_the_reason(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5_000, trigger_price=90)
    data.ltp = 85.0
    trader.run_automation()
    gtt = broker.storage.get_recent_gtts()[0]
    assert gtt["status"] == "triggered" and "order failed" in gtt["detail"]
    assert not broker.positions()


def test_gtt_waits_while_the_price_is_still_yesterdays(trader_env):
    from datetime import timedelta

    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5, trigger_price=90)
    data.ltp = 80.0
    data.bar_date = datetime.now().date() - timedelta(days=1)
    trader.run_automation()
    assert not broker.positions()
    data.bar_date = datetime.now().date()
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_a_gtt_expires_after_a_year(trader_env):
    trader, broker, data, _ = trader_env
    trader.create_gtt("AAA", "BUY", 5, trigger_price=90)
    gid = _gtts(broker)[0]["id"]
    import sqlite3

    conn = sqlite3.connect(broker.storage.db_path)
    conn.execute("UPDATE gtt_orders SET expires_at = '2020-01-01T00:00:00' WHERE id = ?", (gid,))
    conn.commit()
    conn.close()
    data.ltp = 50.0                                  # would have triggered
    trader.run_automation()
    assert broker.storage.get_recent_gtts()[0]["status"] == "expired" and not broker.positions()


def test_gtt_is_valid_for_a_year(trader_env):
    trader, _, _, _ = trader_env
    expires = datetime.fromisoformat(trader.create_gtt("AAA", "BUY", 1, trigger_price=90)["expires_at"])
    assert 364 <= (expires - datetime.now()).days <= 365


def test_cancel_gtt(trader_env):
    trader, broker, data, _ = trader_env
    gid = trader.create_gtt("AAA", "BUY", 5, trigger_price=90)["gtt_id"]
    assert trader.cancel_gtt(gid)["status"] == "cancelled"
    data.ltp = 50.0
    trader.run_automation()
    assert not broker.positions()
    with pytest.raises(ManualOrderError, match="no longer active"):
        trader.cancel_gtt(gid)


def test_gtt_input_validation(trader_env):
    trader, broker, _, _ = trader_env
    with pytest.raises(ManualOrderError, match="trigger price"):
        trader.create_gtt("AAA", "BUY", 1)
    with pytest.raises(ManualOrderError, match="Unknown symbol"):
        trader.create_gtt("ZZZ", "BUY", 1, trigger_price=90)
    with pytest.raises(ManualOrderError, match="whole number"):
        trader.create_gtt("AAA", "BUY", 1.5, trigger_price=90)
    with pytest.raises(ManualOrderError, match="Stop loss must be below"):
        trader.create_gtt("AAA", "BUY", 1, trigger_price=90, stop_loss=95)
    assert not _gtts(broker)


def test_gtt_can_be_created_while_the_market_is_closed(trader_env):
    trader, broker, data, state = trader_env
    state["open"] = False
    assert trader.create_gtt("AAA", "BUY", 5, trigger_price=90)["gtt_id"]
    state["open"] = True
    data.ltp = 85.0
    trader.run_automation()
    assert "AAA" in broker.positions()


def test_api_gtt_create_list_cancel(client):
    made = client.post("/api/manual/gtt", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 3, "trigger_price": 90}).get_json()
    assert made["ok"] and made["direction"] == "down"
    listed = client.get("/api/manual/gtt?profile=manual_swing").get_json()["gtts"]
    assert [g["symbol"] for g in listed] == ["AAA"] and listed[0]["status"] == "active"
    assert client.post("/api/manual/gtt/cancel", json={
        "profile": "manual_swing", "gtt_id": made["gtt_id"]}).get_json()["ok"]
    bad = client.post("/api/manual/gtt", json={
        "profile": "manual_swing", "symbol": "AAA", "side": "BUY", "quantity": 3, "trigger_price": 100})
    assert bad.status_code == 400
    assert client.get("/api/manual/gtt?profile=auto").status_code == 400


# --- starting capital follows the setting until the first trade -----------------------

def _fresh_ledger(tmp_path, capital=100_000.0):
    return Storage(str(tmp_path / "ledger.db"), capital)


def test_an_untouched_ledger_adopts_a_changed_starting_capital(tmp_path):
    storage = _fresh_ledger(tmp_path)
    storage.record_equity(100_000.0, 0.0)                    # a flat snapshot of the old amount
    assert storage.adopt_starting_capital_if_untouched(500_000.0) is True
    assert storage.get_cash() == 500_000.0 and storage.get_starting_capital() == 500_000.0
    assert storage.get_equity_curve(limit=10) == []          # no jump on the curve
    assert storage.adopt_starting_capital_if_untouched(500_000.0) is False   # nothing left to change


def test_a_ledger_that_has_traded_keeps_its_original_capital(tmp_path):
    storage = _fresh_ledger(tmp_path)
    broker = PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0)
    broker.buy("AAA", 5, 100.0, "t")
    cash = storage.get_cash()
    assert storage.adopt_starting_capital_if_untouched(500_000.0) is False
    assert storage.get_starting_capital() == 100_000.0 and storage.get_cash() == cash


def test_a_closed_out_ledger_is_still_a_track_record(tmp_path):
    storage = _fresh_ledger(tmp_path)
    broker = PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0)
    broker.buy("AAA", 5, 100.0, "t")
    broker.sell("AAA", 5, 100.0, "t")                        # no open position, but it has traded
    assert storage.adopt_starting_capital_if_untouched(500_000.0) is False
    assert storage.get_starting_capital() == 100_000.0


@pytest.mark.parametrize("bad", [0, -5, None])
def test_a_missing_or_nonpositive_capital_is_ignored(tmp_path, bad):
    storage = _fresh_ledger(tmp_path)
    assert storage.adopt_starting_capital_if_untouched(bad) is False
    assert storage.get_cash() == 100_000.0


def test_pending_orders_and_gtts_do_not_count_as_trading(tmp_path):
    storage = _fresh_ledger(tmp_path)
    storage.add_pending_order("AAA", 1, 90.0, None, None, "")
    assert storage.adopt_starting_capital_if_untouched(250_000.0) is True
    assert len(storage.get_pending_orders()) == 1            # kept: still valid orders


def test_engine_picks_up_a_changed_capital_for_an_untouched_manual_ledger(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    state = str(tmp_path / "manual.db")
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: state)
    monkeypatch.setattr(cfg, "get_profile_starting_capital", lambda name=None: 100_000.0)
    engine = TradingEngine(cfg, profile_name="manual_swing")
    assert engine.broker.cash() == 100_000.0

    monkeypatch.setattr(cfg, "get_profile_starting_capital", lambda name=None: 500_000.0)
    engine.sync_untouched_capital()                          # what the loop and the settings save call
    assert engine.broker.cash() == 500_000.0
    assert engine.storage.get_starting_capital() == 500_000.0

    restarted = TradingEngine(cfg, profile_name="manual_swing")   # a restart keeps it
    assert restarted.broker.cash() == 500_000.0

    restarted.broker.buy("AAA", 1, 100.0, "manual")          # after the first trade it is fixed
    monkeypatch.setattr(cfg, "get_profile_starting_capital", lambda name=None: 900_000.0)
    restarted.sync_untouched_capital()
    assert restarted.storage.get_starting_capital() == 500_000.0


# --- what is invested, counting the waiting buys and GTTs ------------------------------

def test_commitments_are_empty_with_nothing_waiting(trader_env):
    trader, _, _, _ = trader_env
    c = trader.commitments()
    assert c["total"] == 0 and c["count"] == 0 and c["cash_after"] == trader.broker.cash()


def test_commitments_cost_each_kind_of_waiting_buy(trader_env):
    trader, broker, data, state = trader_env
    state["open"] = False
    trader.place_order("AAA", "BUY", 10, entry_price=90)            # limit order, 90 each
    trader.place_order("BBB", "BUY", 5)                              # scheduled at the open, last price 100
    trader.create_gtt("CCC", "BUY", 4, trigger_price=110)            # breakout GTT, 110 each
    trader.create_gtt("CCC", "BUY", 2, trigger_price=80, limit_price=82)   # GTT with a limit: costed at 82

    c = trader.commitments()
    assert c["limit_orders"] == pytest.approx(broker.estimated_buy_cost(90, 10), abs=0.01)
    assert c["scheduled_buys"] == pytest.approx(broker.estimated_buy_cost(100, 5), abs=0.01)
    assert c["gtt_buys"] == pytest.approx(
        broker.estimated_buy_cost(110, 4) + broker.estimated_buy_cost(82, 2), abs=0.01)
    assert c["count"] == 4 and c["unpriced"] == 0
    assert c["total"] == pytest.approx(c["limit_orders"] + c["scheduled_buys"] + c["gtt_buys"], abs=0.02)
    assert c["cash_after"] == pytest.approx(broker.cash() - c["total"], abs=0.02)


def test_sells_free_money_so_they_are_not_counted(trader_env):
    trader, _, _, state = trader_env
    trader.place_order("AAA", "BUY", 10)
    state["open"] = False
    trader.place_order("AAA", "SELL", 5)                             # scheduled sell
    trader.create_gtt("AAA", "SELL", 5, trigger_price=120)
    trader.create_gtt("AAA", "SELL", 5, trigger_price=95, oco=True, target_trigger=130)
    assert trader.commitments()["total"] == 0


def test_resolved_orders_stop_counting(trader_env):
    trader, broker, data, state = trader_env
    trader.place_order("AAA", "BUY", 10, entry_price=90)
    gid = trader.create_gtt("BBB", "BUY", 5, trigger_price=110)["gtt_id"]
    assert trader.commitments()["count"] == 2
    trader.cancel_gtt(gid)
    data.ltp = 85.0
    trader.run_automation()                                          # the limit order fills
    assert trader.commitments()["count"] == 0


def test_an_unpriceable_scheduled_buy_is_counted_but_flagged(trader_env):
    trader, _, data, state = trader_env
    state["open"] = False
    trader.place_order("AAA", "BUY", 5)
    data.fail = True
    c = trader.commitments()
    assert c["count"] == 1 and c["unpriced"] == 1 and c["scheduled_buys"] == 0


def test_cash_after_goes_negative_when_waiting_buys_exceed_cash(trader_env):
    trader, _, _, _ = trader_env
    trader.create_gtt("AAA", "BUY", 600, trigger_price=110)          # about 66,000
    trader.create_gtt("BBB", "BUY", 600, trigger_price=90)           # about 54,000: 120,000 > 100,000 cash
    assert trader.commitments()["cash_after"] < 0


def test_summary_reports_invested_and_commitments_for_a_manual_profile(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    from papertrader.web.data_api import build_summary

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    engine.data = _FakeData(ltp=100.0)
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.storage.add_gtt(symbol="BBB", side="BUY", kind="single", quantity=5, trigger_price=110.0,
                           direction="up", limit_price=None, target_trigger=None, stop_loss=None,
                           target_price=None, note="", expires_at="2099-01-01T00:00:00")
    summary = build_summary(engine)
    assert summary["invested"] == pytest.approx(10 * 100.0, abs=1.0)
    assert summary["commitments"]["count"] == 1 and summary["commitments"]["gtt_buys"] > 5 * 110 - 1


def test_an_automated_profile_has_no_commitments_but_still_reports_invested(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    from papertrader.web.data_api import build_summary

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "auto.db"))
    engine = TradingEngine(cfg, profile_name="trend_pullback")
    engine.data = _FakeData(ltp=100.0)
    summary = build_summary(engine)
    assert summary["commitments"] is None and summary["invested"] == 0


# --- scripts/check_capital.py --------------------------------------------------------

def _check_capital_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_capital", "scripts/check_capital.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_capital_classifies_each_ledger(tmp_path):
    describe = _check_capital_module().describe
    assert describe(str(tmp_path / "missing.db"), 500_000.0)["status"] == "no_ledger"

    untouched = Storage(str(tmp_path / "a.db"), 100_000.0)
    assert describe(untouched.db_path, 100_000.0)["status"] == "ok"
    assert describe(untouched.db_path, 500_000.0)["status"] == "will_update"

    traded = Storage(str(tmp_path / "b.db"), 100_000.0)
    PaperBroker(traded, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0).buy("AAA", 1, 100.0, "t")
    info = describe(traded.db_path, 500_000.0)
    assert info["status"] == "locked" and info["trades"] == 1 and info["positions"] == 1
    assert describe(traded.db_path, 100_000.0)["status"] == "ok"


def test_applying_what_check_capital_reports_brings_the_ledger_to_ok(tmp_path):
    describe = _check_capital_module().describe
    storage = Storage(str(tmp_path / "a.db"), 100_000.0)
    assert describe(storage.db_path, 500_000.0)["status"] == "will_update"
    Storage(storage.db_path, 500_000.0).adopt_starting_capital_if_untouched(500_000.0)
    assert describe(storage.db_path, 500_000.0)["status"] == "ok"
