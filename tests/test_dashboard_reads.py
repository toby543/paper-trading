"""The dashboard's ledger reads: one-row summaries, single-step row reads, cached comparison."""
import sqlite3

import pytest

from papertrader.portfolio.models import Position, Trade
from papertrader.portfolio.storage import Storage


@pytest.fixture
def ledger(tmp_path):
    s = Storage(str(tmp_path / "l.db"), 100000.0)
    trades = [
        Trade(None, "AAA", "BUY", 10, 100.0, 20.0, "entry", "2026-11-01T10:00:00"),
        Trade(None, "AAA", "SELL", 10, 110.0, 20.0, "target", "2026-11-02T10:00:00", realized_pnl=60.0),
        Trade(None, "BBB", "BUY", 0.123456789, 0.00001234, 0.5, "entry", "2026-11-03T10:00:00"),
        Trade(None, "BBB", "SELL", 0.123456789, 0.00001, 0.5, "stop", "2026-11-04T10:00:00", realized_pnl=-1.25),
        Trade(None, "CCC", "BUY", 5, 200.0, 20.0, "entry", "2026-11-05T10:00:00"),
    ]
    for t in trades:
        s.record_trade(t)
    s.upsert_position(Position("CCC", 5, 200.0, "2026-11-05", 210.0, entry_charges=20.0))
    s.set_cash(98000.0)
    s.set_last_scan_at("2026-11-05T11:00:00")
    conn = sqlite3.connect(s.db_path)
    for i, equity in enumerate((100000.0, 99500.0, 101000.0)):
        conn.execute("INSERT INTO equity_curve VALUES (?, ?, ?, ?)",
                     (f"2026-11-0{i + 1}T15:30:00", 98000.0, equity - 98000.0, equity))
    conn.commit()
    conn.close()
    return s


def test_trades_and_the_equity_curve_read_back_exactly_newest_first(ledger):
    trades = ledger.get_trades()
    assert [t.symbol for t in trades] == ["CCC", "BBB", "BBB", "AAA", "AAA"]
    assert [t.id for t in trades] == sorted((t.id for t in trades), reverse=True)
    bbb_buy = trades[2]
    assert bbb_buy.quantity == 0.123456789 and bbb_buy.price == 0.00001234 and bbb_buy.realized_pnl is None
    assert trades[1].realized_pnl == -1.25
    assert [t.symbol for t in ledger.get_trades(limit=2)] == ["CCC", "BBB"]

    curve = ledger.get_equity_curve(limit=2)
    assert [r["total_equity"] for r in curve] == [101000.0, 99500.0]
    assert set(curve[0]) == {"timestamp", "cash", "positions_value", "total_equity"}


def test_the_summary_row_matches_what_the_full_reads_add_up_to(ledger):
    row = ledger.get_summary_row()
    trades = ledger.get_trades()
    closed = [t for t in trades if t.side == "SELL" and t.realized_pnl is not None]
    positions = ledger.get_positions()
    assert row["total_trades"] == len(trades) == 5
    assert row["closed_trades"] == len(closed) == 2
    assert row["winning_trades"] == 1
    assert row["realized_pnl"] == pytest.approx(ledger.get_total_realized_pnl()) == 58.75
    assert row["open_positions"] == len(positions) == 1
    assert row["cost_basis"] == pytest.approx(sum(p.cost_basis for p in positions.values()))
    assert row["entry_charges"] == 20.0
    assert (row["starting_capital"], row["cash"], row["last_scan_at"]) == (100000.0, 98000.0, "2026-11-05T11:00:00")
    assert (row["last_equity"], row["last_equity_at"]) == (101000.0, "2026-11-03T15:30:00")


def test_an_untouched_ledger_has_an_empty_summary_and_no_equity_endpoints(tmp_path):
    fresh = Storage(str(tmp_path / "new.db"), 5000.0)
    row = fresh.get_summary_row()
    assert row["last_equity"] is None and row["total_trades"] == 0 and row["cost_basis"] == 0
    assert fresh.get_equity_endpoints() is None
    assert fresh.get_trades() == [] and fresh.get_equity_curve() == []


def test_equity_endpoints_are_the_first_and_latest_points(ledger):
    first, last = ledger.get_equity_endpoints()
    assert (first["total_equity"], last["total_equity"]) == (100000.0, 101000.0)


def test_rows_fall_back_to_plain_reads_without_json_support(ledger, monkeypatch):
    real = sqlite3.Connection.execute

    class NoJson(sqlite3.Connection):
        def execute(self, sql, *args):
            if "json_group_array" in sql:
                raise sqlite3.OperationalError("no such function: json_group_array")
            return real(self, sql, *args)

    def opener():
        conn = sqlite3.connect(ledger.db_path, factory=NoJson)
        conn.row_factory = sqlite3.Row
        return conn
    monkeypatch.setattr(ledger, "_open", opener)
    assert [t.symbol for t in ledger.get_trades()] == ["CCC", "BBB", "BBB", "AAA", "AAA"]
    assert len(ledger.get_equity_curve()) == 3


def test_the_strategy_comparison_is_built_once_per_cache_window(monkeypatch):
    from papertrader.config import Config
    from papertrader.web import app as web_app
    from papertrader.web import auth

    builds = []
    monkeypatch.setattr(auth, "load_auth_store", lambda: None)
    monkeypatch.setattr(web_app, "build_strategy_comparison",
                        lambda engines, cfg: builds.append(1) or {"profiles": [], "n": len(builds)})

    class Engine:
        profile_name = "52w_high"
        is_manual = False
    client = web_app.create_app({"52w_high": Engine()}, Config.load()).test_client()
    assert client.get("/api/strategy-comparison").get_json()["n"] == 1
    assert client.get("/api/strategy-comparison").get_json()["n"] == 1
    monkeypatch.setattr(web_app, "COMPARISON_CACHE_SECONDS", 0)
    assert client.get("/api/strategy-comparison").get_json()["n"] == 2
