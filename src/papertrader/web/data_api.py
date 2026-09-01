"""Plain (Flask-free) functions that turn engine/storage state into JSON-
serializable dicts for the web dashboard. Kept separate from app.py so
this logic can be exercised without a running Flask server."""
from __future__ import annotations

from datetime import datetime

from ..strategy.momentum_52w_high import is_market_in_uptrend

# label, Yahoo Finance ticker -- shown as sparkline cards on the dashboard,
# independent of whatever regime.index_symbol is currently configured as
# the strategy's benchmark.
_INDEX_TILES = [
    ("Nifty 50", "^NSEI"),
    ("Nifty 500", "^CRSLDX"),
    ("Sensex", "^BSESN"),
]


def _safe_quote(engine, symbol: str):
    try:
        return engine.data.get_quote(symbol)
    except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky quote
        return None


def _pct_from_52w_high(week52_high: float | None, ltp: float) -> float | None:
    """How far below its 52-week high the stock is trading, in percent.
    Clamped at 0 if the quote's LTP is at/above the recorded high (a
    fresh high the cached 52w figure hasn't caught up to yet)."""
    if not week52_high or week52_high <= 0:
        return None
    return max(0.0, (week52_high - ltp) / week52_high * 100.0)


def _market_regime(engine) -> dict:
    regime_cfg = engine.cfg.get("regime", default={}) or {}
    index_symbol = regime_cfg.get("index_symbol", "^NSEI")
    ma_days = regime_cfg.get("ma_days", 200)
    if not regime_cfg.get("enabled", False):
        return {"enabled": False, "status": None, "index_symbol": index_symbol, "ma_days": ma_days}
    try:
        index_history = engine.data.get_index_history(index_symbol)
        status = "up" if is_market_in_uptrend(index_history, ma_days) else "down"
    except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky index fetch
        status = None
    return {"enabled": True, "status": status, "index_symbol": index_symbol, "ma_days": ma_days}


def build_summary(engine) -> dict:
    positions = engine.broker.positions()
    cash = engine.broker.cash()
    starting_capital = float(engine.cfg.get("account", "starting_capital", default=0.0))

    position_rows = []
    positions_value = 0.0
    for symbol, pos in positions.items():
        quote = _safe_quote(engine, symbol)
        ltp = quote.ltp if quote else pos.avg_price
        week52_high = quote.week52_high if quote else None
        pct_from_high = _pct_from_52w_high(week52_high, ltp)
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
            "week52_high": round(week52_high, 2) if week52_high else None,
            "pct_from_52w_high": round(pct_from_high, 2) if pct_from_high is not None else None,
        })

    total_equity = cash + positions_value
    total_pnl = total_equity - starting_capital
    total_pnl_pct = (total_pnl / starting_capital * 100.0) if starting_capital else 0.0
    total_realized_pnl = engine.storage.get_total_realized_pnl()

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "market_open": engine.calendar.is_market_open(),
        "regime": _market_regime(engine),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_equity": round(total_equity, 2),
        "starting_capital": round(starting_capital, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "open_positions": len(positions),
        "max_positions": engine.risk.max_open_positions,
        "universe_size": len(engine.universe),
        "last_scan_at": engine.storage.get_last_scan_at(),
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
            "realized_pnl": round(t.realized_pnl, 2) if t.realized_pnl is not None else None,
        }
        for t in trades
    ]


def build_candidates(engine, limit: int = 20) -> dict:
    """Preview of what the strategy currently finds worth buying, without
    placing any trades. Expensive (network calls across the whole
    universe) -- meant to be triggered on demand from the dashboard, not
    auto-polled like the rest of /api/*."""
    positions = engine.broker.positions()
    room = engine.risk.room_for_new_positions(len(positions))
    regime = _market_regime(engine)
    regime_blocking = regime["enabled"] and regime["status"] == "down"

    ranked = engine.find_candidates(exclude_symbols=set(positions))

    rows = []
    for cand in ranked[:limit]:
        rows.append({
            "symbol": cand.symbol,
            "ltp": round(cand.ltp, 2),
            "week52_high": round(cand.week52_high, 2),
            "pct_from_52w_high": round(cand.pct_from_52w_high, 2),
            "momentum_return_pct": round(cand.momentum_return_pct, 2),
            "relative_strength_pct": round(cand.relative_strength_pct, 2) if cand.relative_strength_pct is not None else None,
            "volume_multiple": round(cand.volume_multiple, 2) if cand.volume_multiple is not None else None,
            "score": round(cand.score, 2),
        })

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "index_symbol": regime["index_symbol"],
        "regime_blocking": regime_blocking,
        "room_available": room,
        "max_positions": engine.risk.max_open_positions,
        "total_qualifying": len(ranked),
        "candidates": rows,
    }


def build_index_charts(engine, period: str = "6mo") -> list[dict]:
    """Sparkline data for the Nifty 50 / Nifty 500 / Sensex cards: each
    stock's own price scale is wildly different (~25,000 vs. ~80,000), so
    the series is normalized to % change from the first close in the
    window rather than plotted at absolute levels."""
    charts = []
    for label, symbol in _INDEX_TILES:
        try:
            history = engine.data.get_index_history(symbol, period=period)
        except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky index fetch
            charts.append({"label": label, "symbol": symbol, "available": False, "points": [], "last_value": None, "total_return_pct": None})
            continue

        closes = history["Close"].dropna()
        if len(closes) < 2:
            charts.append({"label": label, "symbol": symbol, "available": False, "points": [], "last_value": None, "total_return_pct": None})
            continue

        base = float(closes.iloc[0])
        points = [
            {"date": idx.strftime("%Y-%m-%d"), "pct": round((float(v) - base) / base * 100.0, 2)}
            for idx, v in closes.items()
        ]
        charts.append({
            "label": label,
            "symbol": symbol,
            "available": True,
            "points": points,
            "last_value": round(float(closes.iloc[-1]), 2),
            "total_return_pct": points[-1]["pct"],
        })
    return charts


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


def build_performance_comparison(engine) -> dict:
    """Compares the live strategy's total-equity return since its first
    recorded equity-curve point against a plain buy-and-hold return of the
    configured benchmark index over that same window -- answers "is this
    actually beating the market" rather than just showing the raw equity
    curve on its own. Never fabricates a benchmark return when the index
    history can't be fetched; the strategy's own return is still shown."""
    rows = engine.storage.get_equity_curve(limit=100000)
    if len(rows) < 2:
        return {"available": False}

    first, last = rows[-1], rows[0]  # storage returns newest-first
    starting_equity = float(first["total_equity"])
    current_equity = float(last["total_equity"])
    if starting_equity <= 0:
        return {"available": False}
    strategy_return_pct = round((current_equity - starting_equity) / starting_equity * 100.0, 2)

    index_symbol = engine.regime_cfg.get("index_symbol", "^NSEI")
    benchmark_return_pct = None
    benchmark_available = False
    try:
        history = engine.data.get_index_history(index_symbol, period="2y")
        closes = history["Close"].dropna()
        if len(closes) >= 2:
            target_date = str(first["timestamp"])[:10]
            start_price = None
            for idx, price in closes.items():
                if idx.strftime("%Y-%m-%d") >= target_date:
                    start_price = float(price)
                    break
            if start_price is None:
                start_price = float(closes.iloc[0])
            end_price = float(closes.iloc[-1])
            if start_price > 0:
                benchmark_return_pct = round((end_price - start_price) / start_price * 100.0, 2)
                benchmark_available = True
    except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky index fetch
        pass

    return {
        "available": True,
        "since": first["timestamp"],
        "strategy_return_pct": strategy_return_pct,
        "benchmark_symbol": index_symbol,
        "benchmark_available": benchmark_available,
        "benchmark_return_pct": benchmark_return_pct,
    }
