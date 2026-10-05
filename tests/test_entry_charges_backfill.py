"""Positions opened before entry_charges was tracked recorded 0; the fee is
recoverable from the trades table and must be rebuilt."""
import os
import sqlite3
import tempfile

import pytest

from papertrader.portfolio.broker import PaperBroker
from papertrader.portfolio.storage import Storage


@pytest.fixture
def ledger():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def _zero_entry_charges(path, symbol=None):
    conn = sqlite3.connect(path)
    try:
        with conn:
            if symbol:
                conn.execute("UPDATE positions SET entry_charges = 0 WHERE symbol = ?", (symbol,))
            else:
                conn.execute("UPDATE positions SET entry_charges = 0")
    finally:
        conn.close()


def _broker(storage):
    return PaperBroker(storage, slippage_bps=0.0, flat_charges_inr=10.0, fee_pct=0.0)


def test_backfill_restores_the_entry_fee_of_a_legacy_position(ledger):
    storage = Storage(ledger, 100_000.0)
    _broker(storage).buy("AAA", 10, 100.0, "t")
    real = storage.get_positions()["AAA"].entry_charges
    assert real > 0
    _zero_entry_charges(ledger)

    assert storage.backfill_entry_charges() == 1
    assert storage.get_positions()["AAA"].entry_charges == pytest.approx(real)


def test_backfill_follows_a_partial_sell(ledger):
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage)
    broker.buy("AAA", 10, 100.0, "t")
    broker.sell("AAA", 4, 100.0, "t")
    expected = storage.get_positions()["AAA"].entry_charges  # what the broker itself kept
    _zero_entry_charges(ledger)

    storage.backfill_entry_charges()
    assert storage.get_positions()["AAA"].entry_charges == pytest.approx(expected)


def test_backfill_ignores_a_closed_then_reopened_symbol(ledger):
    storage = Storage(ledger, 100_000.0)
    broker = _broker(storage)
    broker.buy("AAA", 10, 100.0, "t")
    broker.sell("AAA", 10, 100.0, "t")
    broker.buy("AAA", 5, 100.0, "t")
    expected = storage.get_positions()["AAA"].entry_charges
    _zero_entry_charges(ledger)

    storage.backfill_entry_charges()
    assert storage.get_positions()["AAA"].entry_charges == pytest.approx(expected)


def test_backfill_leaves_a_correct_position_alone(ledger):
    storage = Storage(ledger, 100_000.0)
    _broker(storage).buy("AAA", 10, 100.0, "t")
    before = storage.get_positions()["AAA"].entry_charges
    assert storage.backfill_entry_charges() == 0
    assert storage.get_positions()["AAA"].entry_charges == before


def test_unrealized_pnl_is_net_of_the_backfilled_fee(ledger):
    storage = Storage(ledger, 100_000.0)
    _broker(storage).buy("AAA", 10, 100.0, "t")
    _zero_entry_charges(ledger)
    storage.backfill_entry_charges()
    pos = storage.get_positions()["AAA"]
    assert pos.unrealized_pnl(100.0) == pytest.approx(-pos.entry_charges)
