"""Plain (Flask-free) functions that turn engine/storage state into JSON-
serializable dicts for the web dashboard. Kept separate from app.py so
this logic can be exercised without a running Flask server."""
from __future__ import annotations

from datetime import datetime

def _safe_ltp(engine, symbol: str, fallback: float) -> float:
    try:
        return engine.data.get_quote(symbol).ltp
    except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky quote
        return fallback


def build_summary(engine) -> dict:
    positions = engine.broker.positions()
    cash = engine.broker.cash()
    starting_capital = float(engine.cfg.get("account", "starting_capital", default=0.0))

    position_rows = []
    positions_value = 0.0
    for symbol, pos in positions.items():
        ltp = _safe_ltp(engine, symbol, pos.avg_price)
        mv = pos.market_value(ltp)
        positions_value += mv
        position_rows.append({
            "symbol": symbol,
            "quantity": pos.quantity,
            "avg_price": round(pos.avg_price, 2),
            "ltp": round(ltp, 2),
            "market_value": round(mv, 2),
            "unrealized_pnl": round(pos.unrealized_pnl(ltp), 2),
            "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(ltp), 2),
            "entry_date": pos.entry_date,
            "highest_close_since_entry": round(pos.highest_close_since_entry, 2),
        })

    total_equity = cash + positions_value
    total_pnl = total_equity - starting_capital
    total_pnl_pct = (total_pnl / starting_capital * 100.0) if starting_capital else 0.0

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "market_open": engine.calendar.is_market_open(),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_equity": round(total_equity, 2),
        "starting_capital": round(starting_capital, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "open_positions": len(positions),
        "max_positions": engine.risk.max_open_positions,
        "universe_size": len(engine.universe),
        "positions": sorted(position_rows, key=lambda r: r["market_value"], reverse=True),
    }


def build_trades(engine, limit: int = 100) -> list[dict]:
    trades = engine.storage.get_trades(limit=limit)
    return [
        {
            "id": t.id,
            "timestamp": t.timestamp,
            "side": t.side,
            "symbol": t.symbol,
            "quantity": t.quantity,
            "price": round(t.price, 2),
            "charges": round(t.charges, 2),
            "value": round(t.gross_value, 2),
            "reason": t.reason,
        }
        for t in trades
    ]


def build_equity_curve(engine, limit: int = 500) -> list[dict]:
    rows = engine.storage.get_equity_curve(limit=limit)
    curve = [
        {
            "timestamp": r["timestamp"],
            "cash": round(r["cash"], 2),
            "positions_value": round(r["positions_value"], 2),
            "total_equity": round(r["total_equity"], 2),
        }
        for r in rows
    ]
    curve.reverse()  # storage returns newest-first; charts want chronological order
    return curve
