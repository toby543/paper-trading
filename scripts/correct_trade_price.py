"""Re-price one SELL that filled on a bad price, and fix the cash it moved.

For a sale recorded at a price that was a data error (on the Pi: Arbitrum sold
at 0.06 because the price came from a different coin). The trade keeps its
time, quantity and reason; its price, charges and realized P&L are recomputed
at the corrected price, and the account's cash is adjusted by the difference.
Shows what it would change unless --apply is given; backs up the ledger first.
Stop the app before applying.

    python scripts/correct_trade_price.py --profile crypto_trend_pullback --trade-id 25 --price 19.40
    python scripts/correct_trade_price.py --profile crypto_trend_pullback --trade-id 25 --price 19.40 --apply

The equity-curve history from the sale onwards is shifted by the same amount,
so the chart and drawdown don't show a drop that never happened. For a trade
corrected before that was done, repair just the curve:

    python scripts/correct_trade_price.py --profile crypto_trend_pullback --trade-id 25 --repair-curve [--apply]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from papertrader.config import Config  # noqa: E402

MARK = " [price corrected from "
CURVE_MARK = " [equity curve adjusted]"


def plan(db_path: str, trade_id: int, price: float, fee_pct: float, flat_charges: float) -> dict:
    """The correction for one SELL, without writing anything."""
    if price <= 0:
        raise ValueError("price must be positive")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        t = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    finally:
        conn.close()
    if t is None:
        raise ValueError(f"no trade #{trade_id} in {os.path.basename(db_path)}")
    if t["side"] != "SELL":
        raise ValueError(f"trade #{trade_id} is a {t['side']}; only a SELL can be re-priced")
    if MARK in (t["reason"] or ""):
        raise ValueError(f"trade #{trade_id} has already been corrected")
    qty = float(t["quantity"])
    new_charges = flat_charges + price * qty * fee_pct / 100.0
    old_net = float(t["price"]) * qty - float(t["charges"])
    new_net = price * qty - new_charges
    cash_delta = new_net - old_net
    return {
        "trade_id": trade_id, "symbol": t["symbol"], "timestamp": t["timestamp"], "quantity": qty,
        "old_price": float(t["price"]), "new_price": price,
        "old_charges": float(t["charges"]), "new_charges": new_charges,
        "old_realized": t["realized_pnl"],
        "new_realized": (t["realized_pnl"] or 0.0) + cash_delta,
        "cash_delta": cash_delta,
        "new_reason": f"{t['reason']}{MARK}{float(t['price']):g}: bad price data]{CURVE_MARK}",
    }


def plan_curve_repair(db_path: str, trade_id: int, fee_pct: float, flat_charges: float) -> dict:
    """For a trade already re-priced by an earlier version of this script,
    which fixed the cash but not the equity curve: the same shift, worked out
    from the original price recorded in the trade's reason."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        t = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    finally:
        conn.close()
    if t is None:
        raise ValueError(f"no trade #{trade_id} in {os.path.basename(db_path)}")
    reason = t["reason"] or ""
    if MARK not in reason:
        raise ValueError(f"trade #{trade_id} has not been price-corrected")
    if CURVE_MARK in reason:
        raise ValueError(f"trade #{trade_id}'s equity curve has already been adjusted")
    old_price = float(reason.split(MARK, 1)[1].split(":", 1)[0])
    qty = float(t["quantity"])
    old_net = old_price * qty - (flat_charges + old_price * qty * fee_pct / 100.0)
    new_net = float(t["price"]) * qty - float(t["charges"])
    return {"trade_id": trade_id, "symbol": t["symbol"], "timestamp": t["timestamp"],
            "cash_delta": new_net - old_net, "new_reason": reason + CURVE_MARK}


def shift_curve(conn, since: str, delta: float) -> int:
    """Add `delta` to every equity-curve point from `since` on."""
    cur = conn.execute(
        "UPDATE equity_curve SET cash = cash + ?, total_equity = total_equity + ? WHERE timestamp >= ?",
        (delta, delta, since))
    return cur.rowcount


def apply_curve_repair(db_path: str, p: dict) -> tuple[str, int]:
    backup = f"{db_path}.bak-curve-{p['trade_id']}"
    shutil.copy2(db_path, backup)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            rows = shift_curve(conn, p["timestamp"], p["cash_delta"])
            conn.execute("UPDATE trades SET reason = ? WHERE id = ?", (p["new_reason"], p["trade_id"]))
    finally:
        conn.close()
    return backup, rows


def apply(db_path: str, p: dict) -> str:
    backup = f"{db_path}.bak-trade-{p['trade_id']}"
    shutil.copy2(db_path, backup)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE trades SET price = ?, charges = ?, realized_pnl = ?, reason = ? WHERE id = ? AND side = 'SELL'",
                (p["new_price"], p["new_charges"], p["new_realized"], p["new_reason"], p["trade_id"]))
            conn.execute("UPDATE account SET cash = cash + ? WHERE id = 1", (p["cash_delta"],))
            shift_curve(conn, p["timestamp"], p["cash_delta"])
    finally:
        conn.close()
    return backup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--trade-id", type=int, required=True)
    ap.add_argument("--price", type=float, help="the price it should have filled at")
    ap.add_argument("--repair-curve", action="store_true",
                    help="only adjust the equity curve of a trade already re-priced by an earlier version")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    if args.profile not in cfg.list_profiles():
        print(f"Unknown profile {args.profile}")
        return 1
    db = cfg.get_profile_state_file(args.profile)
    ex = cfg.get_profile_execution_config(args.profile)
    fee, flat = float(ex.get("fee_pct", 0.0)), float(ex.get("flat_charges_inr", 0.0))
    if args.repair_curve:
        try:
            p = plan_curve_repair(db, args.trade_id, fee, flat)
        except ValueError as exc:
            print(exc)
            return 1
        print(f"Trade #{p['trade_id']} {p['symbol']} at {p['timestamp']}: shift the equity curve from then on "
              f"by {p['cash_delta']:+,.2f}")
        if not args.apply:
            print("Dry run: nothing changed. Add --apply (with the app stopped) to adjust the curve.")
            return 0
        backup, rows = apply_curve_repair(db, p)
        print(f"Adjusted {rows} equity-curve points. Backup of the ledger before the change: {backup}")
        return 0
    if args.price is None:
        print("--price is required (or use --repair-curve)")
        return 1
    try:
        p = plan(db, args.trade_id, args.price, float(ex.get("fee_pct", 0.0)), float(ex.get("flat_charges_inr", 0.0)))
    except ValueError as exc:
        print(exc)
        return 1
    print(f"Trade #{p['trade_id']} SELL {p['quantity']:g} {p['symbol']} at {p['timestamp']}")
    print(f"  price     {p['old_price']:,.4f} -> {p['new_price']:,.4f}")
    print(f"  charges   {p['old_charges']:,.2f} -> {p['new_charges']:,.2f}")
    print(f"  realized  {p['old_realized']:,.2f} -> {p['new_realized']:,.2f}")
    print(f"  cash      {p['cash_delta']:+,.2f}")
    if not args.apply:
        print("Dry run: nothing changed. Add --apply (with the app stopped) to make the correction.")
        return 0
    backup = apply(db, p)
    print(f"Corrected. Backup of the ledger before the change: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
