"""Manual order entry for a profile whose strategy_mode is "manual".

The engine places no entries and no exits for such a profile -- every trade
comes from here, fills through the same PaperBroker (same slippage, charges
and ledger) as the automated profiles.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable

from ..data.nse_client import DataUnavailableError
from ..data.symbol_search import find
from ..portfolio.broker import InsufficientFundsError

MANUAL_MODE = "manual"
MAX_NOTE_LENGTH = 100
# A fill more than this far from the price the user saw is refused: the
# quote on screen can be minutes old by the time Place order is clicked.
PRICE_TOLERANCE_PCT = 2.0


class ManualOrderError(ValueError):
    """A rejected order; the message is safe to show the user."""


@dataclass
class ManualTrader:
    broker: object
    data: object
    risk: object
    is_market_open: Callable[[], bool]
    lock: threading.Lock
    allow_when_market_closed: bool = False
    # (symbol) -> bool; None skips the directory check (tests)
    symbol_exists: Callable[[str], bool] | None = None

    def _symbol_exists(self, symbol: str) -> bool:
        check = self.symbol_exists or (lambda s: find(s) is not None)
        return check(symbol)

    def _live_price(self, symbol: str):
        try:
            quote = self.data.get_quote(symbol)
        except DataUnavailableError as exc:
            raise ManualOrderError(f"No live price available for {symbol} right now ({exc}).") from exc
        if quote is None or not math.isfinite(quote.ltp) or quote.ltp <= 0:
            raise ManualOrderError(f"No valid price for {symbol} right now.")
        return quote

    def quote(self, symbol: str) -> dict:
        symbol = (symbol or "").strip().upper()
        held = self.broker.positions().get(symbol)
        if not held and not self._symbol_exists(symbol):
            raise ManualOrderError(f"Unknown symbol: {symbol or '(empty)'}")
        q = self._live_price(symbol)
        # Buy cost is linear in quantity (slippage and percentage fee scale
        # with it, the flat charge does not): cost(n) = slope * n + fixed.
        one, two = self.broker.estimated_buy_cost(q.ltp, 1), self.broker.estimated_buy_cost(q.ltp, 2)
        slope = two - one
        def finite(value):
            # NaN is not valid JSON, and thin or newly listed stocks do
            # produce a missing previous close or 52-week figure.
            return value if value is not None and math.isfinite(value) else None

        return {
            "symbol": symbol,
            "ltp": q.ltp,
            "prev_close": finite(q.prev_close),
            "week52_high": finite(q.week52_high),
            "week52_low": finite(q.week52_low),
            "source": q.source,
            "market_open": bool(self.is_market_open()),
            "cash": self.broker.cash(),
            "held_quantity": held.quantity if held else 0,
            "held_avg_price": held.avg_price if held else None,
            "buy_cost_per_share": slope,
            "buy_cost_fixed": one - slope,
        }

    def place_order(self, symbol: str, side: str, quantity, note: str = "", expected_price=None) -> dict:
        symbol = (symbol or "").strip().upper()
        side = (side or "").strip().upper()
        if side not in ("BUY", "SELL"):
            raise ManualOrderError("Side must be BUY or SELL.")
        try:
            qty = float(quantity)
        except (TypeError, ValueError):
            raise ManualOrderError("Quantity must be a number.") from None
        if not math.isfinite(qty) or qty <= 0 or qty != int(qty):
            raise ManualOrderError("Quantity must be a whole number of shares, at least 1.")
        qty = int(qty)
        if not (self.allow_when_market_closed or self.is_market_open()):
            raise ManualOrderError("The market is closed, so no fill price is available. Try again during market hours.")
        note = " ".join((note or "").split())[:MAX_NOTE_LENGTH]
        reason = "manual" + (f": {note}" if note else "")

        with self.lock:
            positions = self.broker.positions()
            held = positions.get(symbol)
            if side == "BUY":
                if not held and not self._symbol_exists(symbol):
                    raise ManualOrderError(f"Unknown symbol: {symbol or '(empty)'}")
                if not held and self.risk.room_for_new_positions(len(positions)) <= 0:
                    raise ManualOrderError(
                        f"Maximum open positions reached ({self.risk.max_open_positions}); sell one first.")
            else:
                if not held:
                    raise ManualOrderError(f"You don't hold {symbol}.")
                if qty > held.quantity:
                    raise ManualOrderError(f"You only hold {held.quantity:g} of {symbol}.")
            q = self._live_price(symbol)
            if expected_price is not None:
                try:
                    seen = float(expected_price)
                except (TypeError, ValueError):
                    raise ManualOrderError("Expected price must be a number.") from None
                if math.isfinite(seen) and seen > 0 and abs(q.ltp - seen) / seen * 100.0 > PRICE_TOLERANCE_PCT:
                    raise ManualOrderError(
                        f"The price moved from {seen:,.2f} to {q.ltp:,.2f} since you looked. "
                        "Nothing was traded; review the new price and place the order again.")
            try:
                trade = (self.broker.buy if side == "BUY" else self.broker.sell)(symbol, qty, q.ltp, reason)
            except InsufficientFundsError as exc:
                raise ManualOrderError(str(exc)) from exc
            except ValueError as exc:
                raise ManualOrderError(str(exc)) from exc
            return {
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": trade.price,
                "charges": trade.charges,
                "realized_pnl": trade.realized_pnl,
                "cash": self.broker.cash(),
            }
