import os

import pytest

from papertrader.portfolio.broker import PaperBroker, InsufficientFundsError
from papertrader.portfolio.storage import Storage


@pytest.fixture
def broker(tmp_path):
    db_path = os.path.join(tmp_path, "state.db")
    storage = Storage(db_path, starting_capital=100_000.0)
    return PaperBroker(storage, slippage_bps=0, flat_charges_inr=0)


def test_starting_cash(broker):
    assert broker.cash() == 100_000.0


def test_buy_reduces_cash_and_opens_position(broker):
    broker.buy("RELIANCE", 10, 2500.0, reason="test")
    assert broker.cash() == 100_000.0 - 25_000.0
    pos = broker.positions()["RELIANCE"]
    assert pos.quantity == 10
    assert pos.avg_price == 2500.0


def test_buy_insufficient_funds_raises(broker):
    with pytest.raises(InsufficientFundsError):
        broker.buy("RELIANCE", 1000, 2500.0, reason="test")


def test_averaging_up(broker):
    broker.buy("TCS", 10, 3000.0, reason="t1")
    broker.buy("TCS", 10, 3200.0, reason="t2")
    pos = broker.positions()["TCS"]
    assert pos.quantity == 20
    assert pos.avg_price == pytest.approx(3100.0)


def test_full_sell_closes_position_and_credits_cash(broker):
    broker.buy("INFY", 10, 1500.0, reason="t1")
    broker.sell("INFY", 10, 1600.0, reason="exit")
    assert "INFY" not in broker.positions()
    assert broker.cash() == pytest.approx(100_000.0 - 15_000.0 + 16_000.0)


def test_partial_sell_reduces_quantity(broker):
    broker.buy("WIPRO", 20, 400.0, reason="t1")
    broker.sell("WIPRO", 5, 420.0, reason="partial")
    pos = broker.positions()["WIPRO"]
    assert pos.quantity == 15


def test_sell_more_than_held_raises(broker):
    broker.buy("ITC", 5, 400.0, reason="t1")
    with pytest.raises(ValueError):
        broker.sell("ITC", 10, 420.0, reason="oops")


def test_trade_history_recorded(broker):
    broker.buy("HDFCBANK", 5, 1600.0, reason="t1")
    broker.sell("HDFCBANK", 5, 1650.0, reason="exit")
    trades = broker.storage.get_trades()
    assert len(trades) == 2
    assert trades[0].side == "SELL"  # most recent first
    assert trades[1].side == "BUY"
