"""Paper broker: simulates order execution, fees and slippage against the
persisted account state. No real orders are ever sent anywhere."""
from __future__ import annotations

import logging
from datetime import datetime

from .models import Position, Trade
from .storage import Storage

log = logging.getLogger(__name__)

# Anything at or below this counts as a fully-closed position rather than
# a real holding. Sits an order of magnitude under the 1e-8 (satoshi)
# precision RiskManager sizes crypto to, so it only ever swallows
# floating-point residue, never a quantity anyone actually bought.
_DUST = 1e-9


class InsufficientFundsError(RuntimeError):
    pass


class PaperBroker:
    def __init__(self, storage: Storage, slippage_bps: float = 5.0, flat_charges_inr: float = 20.0,
                 fee_pct: float = 0.0):
        """fee_pct: percentage-of-notional fee charged on each fill, which
        is how crypto exchanges actually price (~0.1% taker), as opposed
        to flat_charges_inr's flat per-order brokerage+STT, which is how
        an Indian equity broker prices. A profile sets one or the other:
        crypto uses fee_pct with flat_charges_inr at 0, equities keep the
        flat charge with fee_pct at 0. Both are honoured if both are set."""
        self.storage = storage
        self.fee_pct = fee_pct
        self.slippage_bps = slippage_bps
        self.flat_charges_inr = flat_charges_inr

    def _fill_price(self, ltp: float, side: str) -> float:
        slip = ltp * (self.slippage_bps / 10_000.0)
        return ltp + slip if side == "BUY" else ltp - slip

    def _charges(self, fill_price: float, quantity: float) -> float:
        """Total cost of one fill under whichever fee model this profile
        uses -- see __init__. A crypto profile's percentage fee scales
        with the notional the way a real exchange charges; an equity
        profile's flat charge does not."""
        return self.flat_charges_inr + fill_price * quantity * (self.fee_pct / 100.0)

    def cash(self) -> float:
        return self.storage.get_cash()

    def positions(self) -> dict[str, Position]:
        return self.storage.get_positions()

    def equity(self, quotes: dict[str, float]) -> float:
        positions_value = sum(p.quantity * quotes.get(p.symbol, p.avg_price) for p in self.positions().values())
        return self.cash() + positions_value

    def buy(self, symbol: str, quantity: float, ltp: float, reason: str) -> Trade:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        fill_price = self._fill_price(ltp, "BUY")
        charges = self._charges(fill_price, quantity)
        cost = fill_price * quantity + charges
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
        trade = Trade(id=None, symbol=symbol, side="BUY", quantity=quantity, price=fill_price, charges=charges, reason=reason, timestamp=now)
        self.storage.record_trade(trade)
        log.info("BUY  %-10s qty=%-12.8g price=%-10.2f reason=%s", symbol, quantity, fill_price, reason)
        return trade

    def sell(self, symbol: str, quantity: float, ltp: float, reason: str) -> Trade:
        positions = self.storage.get_positions()
        existing = positions.get(symbol)
        # Tolerance, not a bare `<`: with fractional crypto quantities a
        # full exit computed from the held size can come back a few
        # float-ULPs above it, which would otherwise refuse to sell a
        # position the caller is holding in its entirety.
        if not existing or existing.quantity < quantity - _DUST:
            raise ValueError(f"Cannot sell {quantity} of {symbol}; held={existing.quantity if existing else 0}")

        fill_price = self._fill_price(ltp, "SELL")
        charges = self._charges(fill_price, quantity)
        proceeds = fill_price * quantity - charges
        now = datetime.now().isoformat(timespec="seconds")
        realized_pnl = (fill_price - existing.avg_price) * quantity - charges

        remaining = existing.quantity - quantity
        # `== 0` would strand an un-closeable dust position: subtracting
        # fractional quantities rarely lands on exactly 0.0, so a full
        # exit could leave e.g. 1e-17 BTC behind, which then keeps the
        # symbol "held" forever (blocking re-entry) while being far too
        # small to ever sell again.
        if remaining <= _DUST:
            self.storage.delete_position(symbol)
        else:
            existing.quantity = remaining
            self.storage.upsert_position(existing)

        self.storage.set_cash(self.storage.get_cash() + proceeds)
        trade = Trade(id=None, symbol=symbol, side="SELL", quantity=quantity, price=fill_price, charges=charges, reason=reason, timestamp=now, realized_pnl=realized_pnl)
        self.storage.record_trade(trade)
        log.info("SELL %-10s qty=%-12.8g price=%-10.2f reason=%s", symbol, quantity, fill_price, reason)
        return trade

    def update_trailing_high(self, symbol: str, ltp: float) -> None:
        positions = self.storage.get_positions()
        pos = positions.get(symbol)
        if pos and ltp > pos.highest_close_since_entry:
            pos.highest_close_since_entry = ltp
            self.storage.upsert_position(pos)
