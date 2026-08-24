"""Paper broker: simulates order execution, fees and slippage against the
persisted account state. No real orders are ever sent anywhere."""
from __future__ import annotations

import logging
from datetime import datetime

from .models import Position, Trade
from .storage import Storage

log = logging.getLogger(__name__)


class InsufficientFundsError(RuntimeError):
    pass


class PaperBroker:
    def __init__(self, storage: Storage, slippage_bps: float = 5.0, flat_charges_inr: float = 20.0):
        self.storage = storage
        self.slippage_bps = slippage_bps
        self.flat_charges_inr = flat_charges_inr

    def _fill_price(self, ltp: float, side: str) -> float:
        slip = ltp * (self.slippage_bps / 10_000.0)
        return ltp + slip if side == "BUY" else ltp - slip

    def cash(self) -> float:
        return self.storage.get_cash()

    def positions(self) -> dict[str, Position]:
        return self.storage.get_positions()

    def equity(self, quotes: dict[str, float]) -> float:
        positions_value = sum(p.quantity * quotes.get(p.symbol, p.avg_price) for p in self.positions().values())
        return self.cash() + positions_value

    def buy(self, symbol: str, quantity: int, ltp: float, reason: str) -> Trade:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        fill_price = self._fill_price(ltp, "BUY")
        cost = fill_price * quantity + self.flat_charges_inr
        cash = self.storage.get_cash()
        if cost > cash:
            raise InsufficientFundsError(f"Need {cost:.2f}, have {cash:.2f} for {symbol}")

        positions = self.storage.get_positions()
        existing = positions.get(symbol)
        now = datetime.now().isoformat(timespec="seconds")
        if existing:
            total_qty = existing.quantity + quantity
            new_avg = (existing.avg_price * existing.quantity + fill_price * quantity) / total_qty
            pos = Position(
                symbol=symbol, quantity=total_qty, avg_price=new_avg,
                entry_date=existing.entry_date, highest_close_since_entry=max(existing.highest_close_since_entry, ltp),
            )
        else:
            pos = Position(symbol=symbol, quantity=quantity, avg_price=fill_price, entry_date=now, highest_close_since_entry=ltp)

        self.storage.upsert_position(pos)
        self.storage.set_cash(cash - cost)
        trade = Trade(id=None, symbol=symbol, side="BUY", quantity=quantity, price=fill_price, charges=self.flat_charges_inr, reason=reason, timestamp=now)
        self.storage.record_trade(trade)
        log.info("BUY  %-10s qty=%-6d price=%-10.2f reason=%s", symbol, quantity, fill_price, reason)
        return trade

    def sell(self, symbol: str, quantity: int, ltp: float, reason: str) -> Trade:
        positions = self.storage.get_positions()
        existing = positions.get(symbol)
        if not existing or existing.quantity < quantity:
            raise ValueError(f"Cannot sell {quantity} of {symbol}; held={existing.quantity if existing else 0}")

        fill_price = self._fill_price(ltp, "SELL")
        proceeds = fill_price * quantity - self.flat_charges_inr
        now = datetime.now().isoformat(timespec="seconds")

        remaining = existing.quantity - quantity
        if remaining == 0:
            self.storage.delete_position(symbol)
        else:
            existing.quantity = remaining
            self.storage.upsert_position(existing)

        self.storage.set_cash(self.storage.get_cash() + proceeds)
        trade = Trade(id=None, symbol=symbol, side="SELL", quantity=quantity, price=fill_price, charges=self.flat_charges_inr, reason=reason, timestamp=now)
        self.storage.record_trade(trade)
        log.info("SELL %-10s qty=%-6d price=%-10.2f reason=%s", symbol, quantity, fill_price, reason)
        return trade

    def update_trailing_high(self, symbol: str, ltp: float) -> None:
        positions = self.storage.get_positions()
        pos = positions.get(symbol)
        if pos and ltp > pos.highest_close_since_entry:
            pos.highest_close_since_entry = ltp
            self.storage.upsert_position(pos)
