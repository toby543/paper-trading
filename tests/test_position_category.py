"""Long term / short term: the category the user gives a manual position, and
the tax term (LTCG / STCG) from how long a position was held."""
import sqlite3
import threading
from datetime import datetime

import pytest

from papertrader.engine.manual import ManualOrderError, ManualTrader
from papertrader.portfolio.broker import PaperBroker
from papertrader.portfolio.storage import Storage
from papertrader.risk.risk_manager import RiskManager
from papertrader.web.data_api import build_trades, holding_period
from tests.test_manual_trading import _FakeData


@pytest.fixture
def env(tmp_path):
    storage = Storage(str(tmp_path / "m.db"), 100_000.0)
    broker = PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0)
    risk = RiskManager(max_open_positions=3, position_size_pct_of_equity=8.0,
                       max_cash_deployed_per_scan_pct=40.0, fractional_quantities=False)
    data = _FakeData()
    state = {"open": True}
    trader = ManualTrader(broker=broker, data=data, risk=risk, is_market_open=lambda: state["open"],
                          lock=threading.Lock(), symbol_exists=lambda s: s in {"AAA", "BBB"})
    return trader, broker, data, state


def test_a_buy_tags_the_position_and_a_sell_records_and_clears_it(env):
    trader, broker, data, _ = env
    out = trader.place_order("AAA", "BUY", 10, horizon="long")
    assert out["horizon"] == "long"
    assert broker.storage.get_horizons() == {"AAA": "long"}
    assert trader.quote("AAA")["held_horizon"] == "long"

    entered = broker.positions()["AAA"].entry_date
    trader.place_order("AAA", "SELL", 4)
    assert broker.storage.get_horizons() == {"AAA": "long"}      # partial sell keeps it
    trader.place_order("AAA", "SELL", 6)
    assert broker.storage.get_horizons() == {}                   # gone with the position
    sells = [t for t in broker.storage.get_trades() if t.side == "SELL"]
    assert [(t.horizon, t.entry_date) for t in sells] == [("long", entered), ("long", entered)]
    assert broker.storage.get_trades()[-1].entry_date is None    # the BUY row has none


def test_adding_to_a_holding_keeps_the_category_unless_a_new_one_is_given(env):
    trader, broker, _, _ = env
    trader.place_order("AAA", "BUY", 1, horizon="short")
    trader.place_order("AAA", "BUY", 1)
    assert broker.storage.get_horizons()["AAA"] == "short"
    trader.place_order("AAA", "BUY", 1, horizon="Long term")
    assert broker.storage.get_horizons()["AAA"] == "long"


def test_a_bad_category_is_rejected_and_nothing_trades(env):
    trader, broker, _, _ = env
    with pytest.raises(ManualOrderError, match="Category"):
        trader.place_order("AAA", "BUY", 1, horizon="forever")
    assert not broker.positions()


def test_set_levels_changes_the_category_only_when_one_is_sent(env):
    trader, broker, _, _ = env
    trader.place_order("AAA", "BUY", 5, horizon="short")
    trader.set_levels("AAA", 90, None)
    assert broker.storage.get_horizons()["AAA"] == "short"
    assert trader.set_levels("AAA", 90, None, horizon="long")["horizon"] == "long"
    assert trader.set_levels("AAA", None, None, horizon="")["horizon"] is None
    assert broker.storage.get_horizons() == {}


def test_waiting_orders_and_gtts_carry_the_category_to_the_fill(env):
    trader, broker, data, state = env
    trader.place_order("AAA", "BUY", 1, entry_price=90, horizon="long")      # limit order
    trader.create_gtt("BBB", "BUY", 1, trigger_price=110, horizon="short")   # buys when it rises to 110
    data.ltp = 89.0
    trader.run_automation()
    assert broker.storage.get_horizons() == {"AAA": "long"}
    data.ltp = 111.0
    trader.run_automation()
    assert broker.storage.get_horizons() == {"AAA": "long", "BBB": "short"}


def test_the_tax_term_turns_long_after_twelve_months():
    assert holding_period("2025-11-12T10:00:00", "2026-11-12T10:00:00")["tax_term"] == "short"
    late = holding_period("2025-11-12T10:00:00", "2026-11-13T10:00:00")
    assert late["tax_term"] == "long" and late["held_days"] == 366
    leap = holding_period("2024-02-29T10:00:00", "2025-03-01T10:00:00")
    assert leap["tax_term"] == "short" and leap["long_term_from"] == "2025-03-02"
    assert holding_period("2020-01-01", "2026-01-01", crypto=True)["tax_term"] is None
    assert holding_period(None)["tax_term"] is None


def test_an_old_ledger_gets_entry_dates_on_its_past_sells(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side TEXT NOT NULL,
            quantity REAL NOT NULL, price REAL NOT NULL, charges REAL NOT NULL, reason TEXT NOT NULL,
            timestamp TEXT NOT NULL, realized_pnl REAL);
        INSERT INTO trades VALUES (1, 'AAA', 'BUY', 10, 100, 0, 'e', '2024-01-05T10:00:00', NULL);
        INSERT INTO trades VALUES (2, 'AAA', 'BUY', 5, 100, 0, 'e', '2024-03-01T10:00:00', NULL);
        INSERT INTO trades VALUES (3, 'AAA', 'SELL', 15, 120, 0, 'x', '2025-02-01T10:00:00', 300);
        INSERT INTO trades VALUES (4, 'AAA', 'BUY', 2, 100, 0, 'e', '2025-06-01T10:00:00', NULL);
        INSERT INTO trades VALUES (5, 'AAA', 'SELL', 1, 90, 0, 'x', '2025-07-01T10:00:00', -10);
    """)
    conn.commit()
    conn.close()
    storage = Storage(path, 100_000.0)
    sells = {t.id: t.entry_date for t in storage.get_trades() if t.side == "SELL"}
    assert sells == {3: "2024-01-05T10:00:00", 5: "2025-06-01T10:00:00"}

    class Engine:
        trades_24_7 = False
    Engine.storage = storage
    rows = {r["id"]: r for r in build_trades(Engine(), 10)}
    assert rows[3]["tax_term"] == "long" and rows[5]["tax_term"] == "short"
    assert rows[4]["tax_term"] is None


def test_the_api_takes_a_category_and_the_summary_shows_it(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    from papertrader.web.data_api import build_summary

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "manual.db"))
    engine = TradingEngine(cfg, profile_name="manual_swing")
    engine.data = _FakeData(ltp=100.0)
    engine.broker.buy("AAA", 10, 100.0, "manual")
    engine.storage.set_horizon("AAA", "long")
    row = build_summary(engine)["positions"][0]
    assert row["horizon"] == "long" and row["tax_term"] == "short" and row["held_days"] == 0
    assert row["long_term_from"] > datetime.now().date().isoformat()
