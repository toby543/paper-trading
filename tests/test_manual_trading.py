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

    def get_quote(self, symbol):
        if self.fail:
            raise DataUnavailableError("down")
        return Quote(symbol, self.ltp, 99.0, 120.0, 80.0, 1000.0, datetime.now(), "fake")


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


def test_market_closed_blocks_orders_unless_allowed(trader_env):
    trader, broker, _, state = trader_env
    state["open"] = False
    with pytest.raises(ManualOrderError, match="market is closed"):
        trader.place_order("AAA", "BUY", 1)
    trader.allow_when_market_closed = True
    trader.place_order("AAA", "BUY", 1)
    assert "AAA" in broker.positions()


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
