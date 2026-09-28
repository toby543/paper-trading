"""Regression tests for the Tier 1 bugs found in the full-codebase audit.

Each test below fails against the code as it stood before the audit.
"""
import os
import sqlite3
import tempfile

import pytest

from papertrader.portfolio.broker import InsufficientFundsError, PaperBroker
from papertrader.portfolio.storage import Storage
from papertrader.web.settings_schema import coerce_and_validate


@pytest.fixture
def ledger():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def _broker(storage, fee_pct=0.1, flat=0.0):
    return PaperBroker(storage, slippage_bps=5.0, flat_charges_inr=flat, fee_pct=fee_pct)


# --- 2. atomic ledger writes ---------------------------------------------

def test_buy_is_all_or_nothing(ledger, monkeypatch):
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage)
    cash_before = storage.get_cash()

    def boom(_trade):
        raise RuntimeError("write failed midway")

    monkeypatch.setattr(storage, "record_trade", boom)
    with pytest.raises(RuntimeError):
        broker.buy("X", 10, 1000.0, "t")

    assert storage.get_cash() == cash_before
    assert storage.get_positions() == {}
    assert storage.get_trades() == []


def test_sell_is_all_or_nothing(ledger, monkeypatch):
    """The worst shape of this bug: position gone and cash credited but no
    SELL row, which also loses the realized P&L and the re-entry cooldown."""
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage)
    broker.buy("X", 10, 1000.0, "t")
    cash_after_buy = storage.get_cash()

    monkeypatch.setattr(storage, "record_trade", lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        broker.sell("X", 10, 1100.0, "t")

    assert "X" in storage.get_positions()
    assert storage.get_cash() == cash_after_buy


def test_insufficient_funds_writes_nothing(ledger):
    storage = Storage(ledger, 100.0)
    broker = _broker(storage)
    with pytest.raises(InsufficientFundsError):
        broker.buy("X", 10, 1000.0, "t")
    assert storage.get_cash() == 100.0
    assert storage.get_positions() == {}
    assert storage.get_trades() == []


def test_sell_rejects_non_positive_quantity(ledger):
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage)
    broker.buy("X", 10, 1000.0, "t")
    with pytest.raises(ValueError):
        broker.sell("X", -5, 1000.0, "t")


# --- 3. P&L net of entry charges -----------------------------------------

@pytest.mark.parametrize("fee_pct,flat", [(0.0, 20.0), (0.1, 0.0)])
def test_realized_pnl_matches_actual_cash_movement(ledger, fee_pct, flat):
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage, fee_pct=fee_pct, flat=flat)
    before = storage.get_cash()
    broker.buy("X", 10, 1000.0, "t")
    trade = broker.sell("X", 10, 1000.0, "t")
    assert trade.realized_pnl == pytest.approx(storage.get_cash() - before)


def test_realized_plus_unrealized_reconciles_with_equity(ledger):
    """The user-visible symptom: two figures on the same dashboard that
    disagreed by every rupee of brokerage ever paid to open a position."""
    start = 100_000.0
    storage = Storage(ledger, start)
    broker = _broker(storage)
    broker.buy("A", 10, 1000.0, "x")
    broker.sell("A", 10, 1100.0, "x")
    broker.buy("B", 10, 1000.0, "x")

    ltp = 1050.0
    position = storage.get_positions()["B"]
    equity = storage.get_cash() + position.market_value(ltp)
    booked = storage.get_total_realized_pnl() + position.unrealized_pnl(ltp)
    assert booked == pytest.approx(equity - start)


def test_partial_sell_amortises_entry_charges(ledger):
    start = 100_000.0
    storage = Storage(ledger, start)
    broker = _broker(storage)
    broker.buy("B", 10, 1000.0, "x")
    opening_charges = storage.get_positions()["B"].entry_charges
    assert opening_charges > 0

    broker.sell("B", 4, 1050.0, "partial")
    left = storage.get_positions()["B"]
    assert left.quantity == pytest.approx(6.0)
    assert left.entry_charges == pytest.approx(opening_charges * 0.6)

    ltp = 1050.0
    equity = storage.get_cash() + left.market_value(ltp)
    booked = storage.get_total_realized_pnl() + left.unrealized_pnl(ltp)
    assert booked == pytest.approx(equity - start)


def test_entry_charges_column_is_added_to_an_existing_ledger(ledger):
    """Ledgers created before this column existed must keep working."""
    conn = sqlite3.connect(ledger)
    conn.executescript(
        """CREATE TABLE positions (symbol TEXT PRIMARY KEY, quantity REAL NOT NULL,
             avg_price REAL NOT NULL, entry_date TEXT NOT NULL,
             highest_close_since_entry REAL NOT NULL);
           INSERT INTO positions VALUES ('OLD-USD', 2.0, 100.0, '2026-01-01T00:00:00', 110.0);"""
    )
    conn.commit()
    conn.close()

    storage = Storage(ledger, 100_000.0)
    old = storage.get_positions()["OLD-USD"]
    assert old.entry_charges == 0.0  # unknown historically; don't invent one
    assert old.unrealized_pnl(120.0) == pytest.approx(40.0)


# --- 5. settings validation ----------------------------------------------

@pytest.mark.parametrize("path", [
    ("risk", "position_size_pct_of_equity"),
    ("risk", "stop_loss_pct"),
    ("risk", "trailing_stop_pct"),
])
def test_nan_is_rejected_by_settings_validation(path):
    """NaN compares False against every bound, so it used to pass both the
    min and max checks and persist -- then either killed the profile's scan
    thread or made the stop silently unreachable."""
    with pytest.raises(ValueError):
        coerce_and_validate(path, "nan")


def test_infinity_is_rejected_by_settings_validation():
    with pytest.raises(ValueError):
        coerce_and_validate(("risk", "position_size_pct_of_equity"), "inf")
    with pytest.raises(ValueError):
        coerce_and_validate(("risk", "position_size_pct_of_equity"), "-inf")


def test_ordinary_values_still_pass():
    assert coerce_and_validate(("risk", "position_size_pct_of_equity"), "8.0") == 8.0
