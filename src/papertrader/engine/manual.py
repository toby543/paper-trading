"""Manual order entry for a profile whose strategy_mode is "manual".

The engine runs no strategy for such a profile. Entries are placed here, at
the live price or as a limit order that waits for a lower price; each
position may carry a stop loss and a target the user chose, and the engine
sells it when the live price reaches either. Everything fills through the
same PaperBroker (same slippage, charges and ledger) as the automated
profiles.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from ..data.nse_client import DataUnavailableError
from ..data.symbol_search import find
from ..portfolio.broker import InsufficientFundsError

log = logging.getLogger(__name__)

MANUAL_MODE = "manual"
MAX_NOTE_LENGTH = 100
# A fill more than this far from the price the user saw is refused: the
# quote on screen can be minutes old by the time Place order is clicked.
PRICE_TOLERANCE_PCT = 2.0
# An entry price this close to (or above) the live price is just a market buy.
MARKET_ENTRY_TOLERANCE = 0.0005
# GTT orders live this long (Zerodha's limit), and a trigger this close to the
# current price would fire at once, which is what a regular order is for.
GTT_VALIDITY_DAYS = 365
GTT_MIN_DISTANCE = 0.0005


class ManualOrderError(ValueError):
    """A rejected order; the message is safe to show the user."""


def _optional_price(value, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ManualOrderError(f"{label} must be a number.") from None
    if not math.isfinite(number) or number <= 0:
        raise ManualOrderError(f"{label} must be a positive price.")
    return number


@dataclass
class ManualTrader:
    broker: object
    data: object
    risk: object
    is_market_open: Callable[[], bool]
    lock: threading.Lock
    allow_when_market_closed: bool = False
    # (symbol) -> bool; None uses the shipped symbol directory
    symbol_exists: Callable[[str], bool] | None = None

    @property
    def storage(self):
        return self.broker.storage

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

    # ------------------------------------------------------------------
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
            # produce a missing previous close or 52-week figure. A 0 means
            # "unknown" (the end-of-day fallback has no 52-week range).
            return value if value is not None and math.isfinite(value) and value > 0 else None

        levels = self.storage.get_levels().get(symbol, {}) if held else {}
        return {
            "symbol": symbol,
            "ltp": q.ltp,
            "prev_close": finite(q.prev_close),
            "week52_high": finite(q.week52_high),
            "week52_low": finite(q.week52_low),
            "source": q.source,
            "price_date": q.bar_date.isoformat() if getattr(q, "bar_date", None) else None,
            "market_open": bool(self.is_market_open()),
            "queue_when_closed": not self.allow_when_market_closed,
            "cash": self.broker.cash(),
            "held_quantity": held.quantity if held else 0,
            "held_avg_price": held.avg_price if held else None,
            "held_stop_loss": levels.get("stop_loss"),
            "held_target_price": levels.get("target_price"),
            "buy_cost_per_share": slope,
            "buy_cost_fixed": one - slope,
        }

    # ------------------------------------------------------------------
    def place_order(self, symbol: str, side: str, quantity, note: str = "", expected_price=None,
                    entry_price=None, stop_loss=None, target_price=None) -> dict:
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
        entry = _optional_price(entry_price, "Entry price")
        sl = _optional_price(stop_loss, "Stop loss")
        target = _optional_price(target_price, "Target price")
        if side == "SELL" and (entry is not None or sl is not None or target is not None):
            raise ManualOrderError("Entry price, stop loss and target apply to buy orders only.")
        market_open = self.allow_when_market_closed or self.is_market_open()
        note = " ".join((note or "").split())[:MAX_NOTE_LENGTH]

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
            if not market_open:
                return self._schedule_for_open(symbol, side, qty, entry, sl, target, note, held)
            q = self._live_price(symbol)
            if not self.allow_when_market_closed and not self._is_fresh(q):
                raise ManualOrderError(
                    f"The latest price available for {symbol} is from {q.bar_date:%d %b}, not today, "
                    "so it can't be traded on yet. Try again in a minute, or schedule the order.")
            self._check_price_protection(q.ltp, expected_price)

            if side == "SELL":
                trade = self._fill(symbol, "SELL", qty, q.ltp, "manual" + (f": {note}" if note else ""))
                return self._result(symbol, "SELL", qty, trade)

            limit_order = entry is not None and entry < q.ltp * (1 - MARKET_ENTRY_TOLERANCE)
            reference = entry if limit_order else q.ltp
            self._check_levels(reference, sl, target, "entry price" if limit_order else "current price")

            if limit_order:
                cost = self.broker.estimated_buy_cost(entry, qty)
                if cost > self.broker.cash():
                    raise ManualOrderError(
                        f"A limit order for {qty} × {symbol} at {entry:,.2f} would need about {cost:,.2f}; "
                        f"you have {self.broker.cash():,.2f}.")
                order_id = self.storage.add_pending_order(symbol, qty, entry, sl, target, note)
                return {"symbol": symbol, "side": "BUY", "status": "pending", "order_id": order_id,
                        "quantity": qty, "limit_price": entry, "stop_loss": sl, "target_price": target,
                        "cash": self.broker.cash()}

            reason = "manual" + (f": {note}" if note else "")
            with self.storage.transaction():
                trade = self._fill(symbol, "BUY", qty, q.ltp, reason)
                self._apply_levels(symbol, sl, target)
            out = self._result(symbol, "BUY", qty, trade)
            out["stop_loss"], out["target_price"] = sl, target
            return out

    def _schedule_for_open(self, symbol, side, qty, entry, sl, target, note, held) -> dict:
        """The market is closed: queue the order. A buy with an entry price
        buys at that price or better and never above it: it fills once the
        market is open and the price is at or below the entry, however far
        that is from the last price. Only a buy with no entry price runs at
        whatever the open is. The engine only acts while the market is open,
        so the order waits through nights, weekends and holidays by itself."""
        try:
            last = self._live_price(symbol).ltp  # the last traded price (previous close)
        except ManualOrderError:
            last = None
        if side == "SELL":
            order_id = self.storage.add_pending_order(symbol, qty, 0.0, None, None, note, "SELL", "market_open")
            return {"symbol": symbol, "side": "SELL", "status": "scheduled", "order_id": order_id,
                    "quantity": qty, "cash": self.broker.cash()}

        limit_order = entry is not None
        reference = entry if limit_order else last
        if reference is not None:
            self._check_levels(reference, sl, target, "entry price" if limit_order else "last price")
            cost = self.broker.estimated_buy_cost(reference, qty)
            if cost > self.broker.cash():
                raise ManualOrderError(
                    f"{qty} × {symbol} at about {reference:,.2f} would need about {cost:,.2f}; "
                    f"you have {self.broker.cash():,.2f}.")
        if limit_order:
            order_id = self.storage.add_pending_order(symbol, qty, entry, sl, target, note)
            return {"symbol": symbol, "side": "BUY", "status": "pending", "order_id": order_id,
                    "quantity": qty, "limit_price": entry, "stop_loss": sl, "target_price": target,
                    "cash": self.broker.cash()}
        order_id = self.storage.add_pending_order(symbol, qty, 0.0, sl, target, note, "BUY", "market_open")
        return {"symbol": symbol, "side": "BUY", "status": "scheduled", "order_id": order_id,
                "quantity": qty, "stop_loss": sl, "target_price": target, "last_price": last,
                "cash": self.broker.cash()}

    # ------------------------------------------------------------------
    # GTT: good till triggered
    def create_gtt(self, symbol: str, side: str, quantity, trigger_price=None, limit_price=None,
                   note: str = "", stop_loss=None, target_price=None, oco: bool = False,
                   target_trigger=None) -> dict:
        """A single-trigger GTT (buy or sell once the price falls or rises to
        trigger_price, optionally capped by limit_price), or with oco=True a
        two-leg sell of a holding: trigger_price is the stop loss and
        target_trigger the target, and whichever is reached first sells while
        the other is dropped. Valid for GTT_VALIDITY_DAYS. Nothing is bought,
        sold or reserved until it triggers."""
        symbol = (symbol or "").strip().upper()
        side = (side or "").strip().upper()
        if side not in ("BUY", "SELL"):
            raise ManualOrderError("Side must be BUY or SELL.")
        qty = self._whole_quantity(quantity)
        trigger = _optional_price(trigger_price, "Trigger price")
        if trigger is None:
            raise ManualOrderError("Enter a trigger price.")
        limit = _optional_price(limit_price, "Limit price")
        target_trig = _optional_price(target_trigger, "Target trigger")
        sl = _optional_price(stop_loss, "Stop loss")
        target = _optional_price(target_price, "Target price")
        note = " ".join((note or "").split())[:MAX_NOTE_LENGTH]
        if oco and side != "SELL":
            raise ManualOrderError("A two-leg GTT (stop loss + target) is for selling a holding.")
        if oco and target_trig is None:
            raise ManualOrderError("Enter the target trigger for the second leg.")
        if oco and limit is not None:
            raise ManualOrderError("Two-leg GTT legs sell at the market price when triggered; leave the limit price blank.")
        if side == "SELL" and (sl is not None or target is not None):
            raise ManualOrderError("Stop loss and target apply to buys; a sell GTT closes shares.")

        with self.lock:
            positions = self.broker.positions()
            held = positions.get(symbol)
            if side == "BUY":
                if not held and not self._symbol_exists(symbol):
                    raise ManualOrderError(f"Unknown symbol: {symbol or '(empty)'}")
            else:
                if not held:
                    raise ManualOrderError(f"You don't hold {symbol}.")
                if qty > held.quantity:
                    raise ManualOrderError(f"You only hold {held.quantity:g} of {symbol}.")
            last = self._live_price(symbol).ltp
            tol = GTT_MIN_DISTANCE

            if oco:
                if not trigger < last * (1 - tol):
                    raise ManualOrderError(f"The stop-loss trigger must be below the current price ({last:,.2f}).")
                if not target_trig > last * (1 + tol):
                    raise ManualOrderError(f"The target trigger must be above the current price ({last:,.2f}).")
                direction = "down"
            else:
                if trigger < last * (1 - tol):
                    direction = "down"
                elif trigger > last * (1 + tol):
                    direction = "up"
                else:
                    raise ManualOrderError(
                        f"The trigger is at the current price ({last:,.2f}); place a regular order instead.")
                if limit is not None:
                    if side == "BUY" and limit < trigger:
                        raise ManualOrderError("A buy's limit price must be at or above its trigger, or it could never fill.")
                    if side == "SELL" and limit > trigger:
                        raise ManualOrderError("A sell's limit price must be at or below its trigger, or it could never fill.")
                if side == "BUY":
                    self._check_levels(limit if limit is not None else trigger, sl, target, "trigger price")

            expires = (datetime.now() + timedelta(days=GTT_VALIDITY_DAYS)).isoformat(timespec="seconds")
            gtt_id = self.storage.add_gtt(
                symbol=symbol, side=side, kind="oco" if oco else "single", quantity=qty,
                trigger_price=trigger, direction=direction, limit_price=limit,
                target_trigger=target_trig if oco else None, stop_loss=sl, target_price=target,
                note=note, expires_at=expires)
            return {"gtt_id": gtt_id, "symbol": symbol, "side": side, "kind": "oco" if oco else "single",
                    "quantity": qty, "trigger_price": trigger, "direction": direction, "limit_price": limit,
                    "target_trigger": target_trig if oco else None, "expires_at": expires, "last_price": last}

    def cancel_gtt(self, gtt_id) -> dict:
        try:
            gid = int(gtt_id)
        except (TypeError, ValueError):
            raise ManualOrderError("Unknown GTT.") from None
        with self.lock:
            if not self.storage.resolve_gtt(gid, "cancelled", "cancelled by you"):
                raise ManualOrderError("That GTT is no longer active.")
        return {"gtt_id": gid, "status": "cancelled"}

    def gtts(self) -> list[dict]:
        return self.storage.get_recent_gtts(15)

    @staticmethod
    def _whole_quantity(quantity) -> int:
        try:
            qty = float(quantity)
        except (TypeError, ValueError):
            raise ManualOrderError("Quantity must be a number.") from None
        if not math.isfinite(qty) or qty <= 0 or qty != int(qty):
            raise ManualOrderError("Quantity must be a whole number of shares, at least 1.")
        return int(qty)

    def _process_gtts(self) -> None:
        now = datetime.now()
        for g in self.storage.get_active_gtts():
            try:
                if g["expires_at"] and datetime.fromisoformat(g["expires_at"]) < now:
                    self.storage.resolve_gtt(g["id"], "expired", "expired after one year without triggering")
                    continue
                q = self._live_price(g["symbol"])
            except ManualOrderError:
                continue
            if not self._is_fresh(q):
                continue
            if g["kind"] == "oco":
                if q.ltp <= g["trigger_price"]:
                    self._fire_gtt(g, q, f"stop-loss trigger {g['trigger_price']:,.2f}")
                elif q.ltp >= g["target_trigger"]:
                    self._fire_gtt(g, q, f"target trigger {g['target_trigger']:,.2f}")
            elif (g["direction"] == "down" and q.ltp <= g["trigger_price"]) or \
                    (g["direction"] == "up" and q.ltp >= g["trigger_price"]):
                self._fire_gtt(g, q, f"trigger {g['trigger_price']:,.2f}")

    def _fire_gtt(self, g: dict, q, why: str) -> None:
        symbol, side, qty = g["symbol"], g["side"], g["quantity"]

        def finish(status: str, detail: str) -> None:
            self.storage.resolve_gtt(g["id"], status, detail)

        limit = g["limit_price"]
        if limit is not None and ((side == "BUY" and q.ltp > limit) or (side == "SELL" and q.ltp < limit)):
            finish("triggered", f"{why} reached at {q.ltp:,.2f}, but that is beyond your limit price "
                                f"({limit:,.2f}); no order was placed")
            return
        positions = self.broker.positions()
        if side == "SELL":
            held = positions.get(symbol)
            if held is None:
                finish("cancelled", "you no longer hold this stock")
                return
            qty = min(qty, held.quantity)
        else:
            if symbol not in positions and self.risk.room_for_new_positions(len(positions)) <= 0:
                finish("triggered", f"{why} reached, but the maximum open positions is reached; no order placed")
                return
        kind = "gtt " + ("stop-loss" if g["kind"] == "oco" and q.ltp <= g["trigger_price"]
                         else "target" if g["kind"] == "oco" else side.lower())
        reason = f"manual {kind} ({why})" + (f": {g['note']}" if g["note"] else "")
        try:
            with self.storage.transaction():
                # Claim first: a failure below rolls the claim back, so a GTT
                # can never place its order twice.
                if not self.storage.resolve_gtt(
                        g["id"], "triggered", f"{why} reached; {side.lower()} {qty:g} at {q.ltp:,.2f}"):
                    return
                self._fill(symbol, side, qty, q.ltp, reason)
                if side == "BUY":
                    self._apply_levels(symbol, g["stop_loss"], g["target_price"])
            log.info("GTT #%s triggered: %s %s", g["id"], side, symbol)
        except ManualOrderError as exc:
            finish("triggered", f"{why} reached, but the order failed: {exc}")
        except Exception:  # noqa: BLE001 - one bad GTT must not stop the others
            log.exception("Could not run GTT #%s", g["id"])

    # ------------------------------------------------------------------
    def set_levels(self, symbol: str, stop_loss, target_price) -> dict:
        """Replace the stop loss / target of a position already held. A
        blank value clears that level."""
        symbol = (symbol or "").strip().upper()
        sl = _optional_price(stop_loss, "Stop loss")
        target = _optional_price(target_price, "Target price")
        with self.lock:
            if symbol not in self.broker.positions():
                raise ManualOrderError(f"You don't hold {symbol}.")
            q = self._live_price(symbol)
            self._check_levels(q.ltp, sl, target, "current price")
            self.storage.set_levels(symbol, sl, target)
        return {"symbol": symbol, "stop_loss": sl, "target_price": target}

    def cancel_order(self, order_id) -> dict:
        try:
            oid = int(order_id)
        except (TypeError, ValueError):
            raise ManualOrderError("Unknown order.") from None
        with self.lock:
            if not self.storage.resolve_pending_order(oid, "cancelled", "cancelled by you"):
                raise ManualOrderError("That order is no longer pending.")
        return {"order_id": oid, "status": "cancelled"}

    def orders(self) -> list[dict]:
        return self.storage.get_recent_orders(15)

    # ------------------------------------------------------------------
    def run_automation(self) -> None:
        """Sell positions whose stop loss or target has been reached, then
        fill any limit order the price has come down to. Called by the engine
        each exit-check cycle with the engine lock ALREADY held (the lock is
        not reentrant, so this must not take it)."""
        positions = self.broker.positions()
        levels = self.storage.get_levels()
        for symbol, pos in positions.items():
            try:
                q = self._live_price(symbol)
            except ManualOrderError as exc:
                log.warning("Skipping manual level check for %s: %s", symbol, exc)
                continue
            self.broker.update_trailing_high(symbol, q.ltp)
            lv = levels.get(symbol) or {}
            stop, target = lv.get("stop_loss"), lv.get("target_price")
            if stop and q.ltp <= stop:
                reason = f"manual stop-loss hit (stop {stop:g})"
            elif target and q.ltp >= target:
                reason = f"manual target hit (target {target:g})"
            else:
                continue
            try:
                self.broker.sell(symbol, pos.quantity, q.ltp, reason)
                log.info("%s: %s", symbol, reason)
            except ValueError as exc:
                log.error("Failed to sell %s on its manual level: %s", symbol, exc)

        for order in self.storage.get_pending_orders():
            self._try_fill_order(order)
        self._process_gtts()

    @staticmethod
    def _is_fresh(quote) -> bool:
        """False when the price is built from an earlier day's bar, as the
        Yahoo fallback is for the first minutes after the open: a scheduled
        order must not fill on yesterday's close."""
        bar = getattr(quote, "bar_date", None)
        return bar is None or (isinstance(bar, date) and bar >= datetime.now().date())

    def _try_fill_order(self, order: dict) -> None:
        symbol = order["symbol"]
        side = order.get("side", "BUY")
        try:
            q = self._live_price(symbol)
        except ManualOrderError:
            return
        if not self._is_fresh(q):
            return  # wait for today's price
        is_limit = order.get("order_type", "limit") == "limit"
        if is_limit and q.ltp > order["limit_price"]:
            return

        def cancel(why: str) -> None:
            self.storage.resolve_pending_order(order["id"], "cancelled", why)

        positions = self.broker.positions()
        qty = order["quantity"]
        if side == "SELL":
            held = positions.get(symbol)
            if held is None or held.quantity < qty:
                cancel("you no longer hold enough shares")
                return
        else:
            if symbol not in positions and self.risk.room_for_new_positions(len(positions)) <= 0:
                cancel("maximum open positions reached")
                return
            # A price that gapped through the order's own stop or target would
            # buy a position that exits on the very next check.
            if order["stop_loss"] and q.ltp <= order["stop_loss"]:
                cancel(f"the price opened at {q.ltp:,.2f}, at or below your stop loss ({order['stop_loss']:,.2f})")
                return
            if order["target_price"] and q.ltp >= order["target_price"]:
                cancel(f"the price opened at {q.ltp:,.2f}, at or above your target ({order['target_price']:,.2f})")
                return
        kind = "limit" if is_limit else "scheduled"
        reason = f"manual {kind} {side.lower()}" + (f": {order['note']}" if order["note"] else "")
        detail = f"filled at {q.ltp:,.2f}" + (f" (limit {order['limit_price']:,.2f})" if is_limit else " at the open")
        try:
            with self.storage.transaction():
                # Claim the order first: if anything below fails the claim
                # rolls back with it, so an order can never fill twice.
                if not self.storage.resolve_pending_order(order["id"], "filled", detail):
                    return
                self._fill(symbol, side, qty, q.ltp, reason)
                if side == "BUY":
                    self._apply_levels(symbol, order["stop_loss"], order["target_price"])
            log.info("Order #%s filled: %s %s", order["id"], side, symbol)
        except ManualOrderError as exc:
            cancel(str(exc))
        except Exception:  # noqa: BLE001 - one bad order must not stop the others
            log.exception("Could not fill order #%s", order["id"])

    # ------------------------------------------------------------------
    def _check_price_protection(self, live: float, expected_price) -> None:
        if expected_price is None:
            return
        try:
            seen = float(expected_price)
        except (TypeError, ValueError):
            raise ManualOrderError("Expected price must be a number.") from None
        if math.isfinite(seen) and seen > 0 and abs(live - seen) / seen * 100.0 > PRICE_TOLERANCE_PCT:
            raise ManualOrderError(
                f"The price moved from {seen:,.2f} to {live:,.2f} since you looked. "
                "Nothing was traded; review the new price and place the order again.")

    @staticmethod
    def _check_levels(reference: float, stop_loss, target, reference_name: str) -> None:
        if stop_loss is not None and stop_loss >= reference:
            raise ManualOrderError(f"Stop loss must be below the {reference_name} ({reference:,.2f}).")
        if target is not None and target <= reference:
            raise ManualOrderError(f"Target must be above the {reference_name} ({reference:,.2f}).")

    def _fill(self, symbol: str, side: str, qty: int, price: float, reason: str):
        try:
            return (self.broker.buy if side == "BUY" else self.broker.sell)(symbol, qty, price, reason)
        except InsufficientFundsError as exc:
            raise ManualOrderError(str(exc)) from exc
        except ValueError as exc:
            raise ManualOrderError(str(exc)) from exc

    def _apply_levels(self, symbol: str, stop_loss, target) -> None:
        """Levels given with a buy replace the ones given; one left blank keeps
        what the position already had (a first entry has none)."""
        existing = self.storage.get_levels().get(symbol, {})
        self.storage.set_levels(
            symbol,
            stop_loss if stop_loss is not None else existing.get("stop_loss"),
            target if target is not None else existing.get("target_price"),
        )

    def _result(self, symbol: str, side: str, qty: int, trade) -> dict:
        return {
            "symbol": symbol,
            "side": side,
            "status": "filled",
            "quantity": qty,
            "price": trade.price,
            "charges": trade.charges,
            "realized_pnl": trade.realized_pnl,
            "cash": self.broker.cash(),
        }
