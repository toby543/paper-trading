"""Plain (Flask-free) functions that turn engine/storage state into JSON-
serializable dicts for the web dashboard. Kept separate from app.py so
this logic can be exercised without a running Flask server."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from ..strategy.momentum_52w_high import is_market_in_uptrend

log = logging.getLogger(__name__)

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
        quote = engine.data.get_quote(symbol)
    except Exception:  # noqa: BLE001 - dashboard must never 500 on a flaky quote
        return None
    # A suspended/halted symbol can make the yfinance fallback return a
    # quote object whose LTP is NaN (its most recent daily bar has no
    # trades) while week52_high -- computed from the whole history, which
    # pandas' max() skips NaNs in -- still looks perfectly valid. NaN
    # compares False against everything ("nan <= x" and "nan > x" are both
    # False in Python), so it silently slips past every check downstream
    # and then poisons the running positions_value/total_equity sum for
    # the *entire* portfolio, not just this one symbol -- and since NaN
    # isn't valid JSON, the browser's JSON.parse() then throws on the
    # whole /api/summary response, breaking every panel on the dashboard,
    # not just this position's row. Treat it the same as "no quote".
    if quote is None or math.isnan(quote.ltp):
        return None
    return quote


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


def _parse_ts(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _build_insights(engine, position_rows: list[dict], cash: float, total_equity: float,
                    market_open: bool, last_scan_at) -> list[dict]:
    """Derived observations about the current book -- things worth noticing
    that aren't obvious from the raw numbers alone. Computed entirely from
    data build_summary() already has in hand, so this adds no extra network
    calls or queries. Each insight is {level, title, detail}, where level is
    warn (wants attention), info (worth knowing), or good (working well)."""
    insights: list[dict] = []
    risk_cfg = engine.risk_cfg or {}
    stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0) or 0)
    trailing_stop_pct = float(risk_cfg.get("trailing_stop_pct", 0) or 0)
    max_positions = engine.risk.max_open_positions

    # --- Engine health: is this profile's loop actually still scanning? ---
    # Each profile runs its own background thread; if one stops (a crash, a
    # hung fetch), the dashboard would otherwise look completely normal --
    # just quietly never trading again. Surface that explicitly.
    scan_interval_min = engine.cfg.get("engine", "scan_interval_minutes", default=15)
    scan_dt = _parse_ts(last_scan_at)
    if market_open:
        if scan_dt is None:
            insights.append({
                "level": "warn",
                "title": "No scan recorded yet",
                "detail": "This profile's engine hasn't completed a scan cycle. If the market has been "
                          "open a while, check the logs for errors in its trading loop.",
            })
        else:
            stale_after = timedelta(minutes=scan_interval_min * 2.5)
            behind = datetime.now() - scan_dt
            if behind > stale_after:
                mins = int(behind.total_seconds() // 60)
                insights.append({
                    "level": "warn",
                    "title": f"Engine may have stopped scanning ({mins}m since last scan)",
                    "detail": f"Expected a scan about every {scan_interval_min}m while the market is open. "
                              "Check data/papertrader.log for errors in this profile's trading loop.",
                })

    # --- Concentration risk ---
    if total_equity > 0:
        for row in position_rows:
            weight = row["market_value"] / total_equity * 100.0
            if weight >= 25.0:
                insights.append({
                    "level": "warn",
                    "title": f"{row['symbol']} is {weight:.0f}% of the book",
                    "detail": "A single position this large means one name drives most of the P&L. "
                              "Position sizing is capped per entry, but price appreciation can drift it up.",
                })

    # --- Positions close to being stopped out ---
    for row in position_rows:
        ltp = row["ltp"]
        triggers = []
        if stop_loss_pct > 0:
            triggers.append(("stop-loss", row["avg_price"] * (1 - stop_loss_pct / 100.0)))
        if trailing_stop_pct > 0:
            triggers.append(("trailing stop", row["highest_close_since_entry"] * (1 - trailing_stop_pct / 100.0)))
        if not triggers or ltp <= 0:
            continue
        # The binding stop is whichever sits highest (closest below price).
        name, level = max(triggers, key=lambda t: t[1])
        headroom_pct = (ltp - level) / ltp * 100.0
        if 0 < headroom_pct <= 2.0:
            insights.append({
                "level": "warn",
                "title": f"{row['symbol']} is {headroom_pct:.1f}% above its {name}",
                "detail": f"Currently ₹{ltp:,.2f} against a {name} around ₹{level:,.2f}. "
                          "A small move down would trigger an exit on the next check.",
            })

    # --- Idle capital ---
    free_slots = max_positions - len(position_rows)
    cash_pct = (cash / total_equity * 100.0) if total_equity > 0 else 0.0
    if market_open and free_slots > 0 and cash_pct >= 60.0:
        insights.append({
            "level": "info",
            "title": f"{cash_pct:.0f}% in cash with {free_slots} position slot(s) free",
            "detail": "Either nothing is currently passing this strategy's entry filters, or the market "
                      "regime filter is blocking new buys. Both are normal in a weak tape.",
        })
    elif free_slots == 0:
        insights.append({
            "level": "info",
            "title": "Fully invested",
            "detail": f"All {max_positions} position slots are in use, so no new entries will be taken "
                      "until something exits.",
        })

    # --- Standout winners ---
    for row in position_rows:
        if row["unrealized_pnl_pct"] >= 20.0:
            insights.append({
                "level": "good",
                "title": f"{row['symbol']} up {row['unrealized_pnl_pct']:.1f}%",
                "detail": f"Trailing stop is doing the work here -- it now sits {trailing_stop_pct:.0f}% "
                          f"below the ₹{row['highest_close_since_entry']:,.2f} peak since entry."
                          if trailing_stop_pct > 0 else "One of the stronger open positions.",
            })

    # --- Recent realised-loss streak ---
    try:
        recent_sells = [t for t in engine.storage.get_trades(limit=40)
                        if t.side == "SELL" and t.realized_pnl is not None][:5]
    except Exception:  # noqa: BLE001 - insights must never break the summary endpoint
        recent_sells = []
    if len(recent_sells) >= 3 and all(t.realized_pnl < 0 for t in recent_sells[:3]):
        insights.append({
            "level": "warn",
            "title": f"Last {len(recent_sells[:3])} closed trades were losses",
            "detail": "Repeated stop-outs often mean the strategy is fighting the prevailing trend. "
                      "Worth checking whether the regime filter is doing its job.",
        })

    return insights


def build_summary(engine) -> dict:
    positions = engine.broker.positions()
    cash = engine.broker.cash()
    starting_capital = engine.cfg.get_profile_starting_capital(engine.profile_name)

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
    total_realized_pnl_pct = (total_realized_pnl / starting_capital * 100.0) if starting_capital else 0.0

    market_open = engine.calendar.is_market_open()
    last_scan_at = engine.storage.get_last_scan_at()
    sorted_positions = sorted(position_rows, key=lambda r: r["market_value"], reverse=True)
    try:
        insights = _build_insights(engine, sorted_positions, cash, total_equity, market_open, last_scan_at)
    except Exception:  # noqa: BLE001 - a derived-insight bug must never break the main summary
        log.exception("Failed to build insights for profile %s", engine.profile_name)
        insights = []

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "market_open": market_open,
        "regime": _market_regime(engine),
        "insights": insights,
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_equity": round(total_equity, 2),
        "starting_capital": round(starting_capital, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "total_realized_pnl_pct": round(total_realized_pnl_pct, 2),
        "open_positions": len(positions),
        "max_positions": engine.risk.max_open_positions,
        "universe_size": len(engine.universe),
        "last_scan_at": last_scan_at,
        "positions": sorted_positions,
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
            "rationale": t.rationale,
        }
        for t in trades
    ]


def build_strategy_comparison(engines: dict, cfg) -> dict:
    """Side-by-side scoreboard of every running profile, so the three
    strategies can be compared directly instead of switching the profile
    dropdown back and forth and holding numbers in your head.

    Deliberately cheap: reads each profile's own SQLite ledger only --
    no live quotes, no network. Equity comes from the last row each
    engine wrote to its equity_curve during mark_to_market (refreshed
    every exit-check cycle), which is already live-marked."""
    display_names = cfg.list_profiles()
    rows = []

    for profile_name, engine in engines.items():
        try:
            starting_capital = cfg.get_profile_starting_capital(profile_name)
            curve = engine.storage.get_equity_curve(limit=1)
            cash = engine.storage.get_cash()
            positions = engine.storage.get_positions()
            # Fall back to cash + cost basis when no equity point exists yet
            # (a profile that hasn't completed its first mark-to-market).
            if curve:
                total_equity = float(curve[0]["total_equity"])
                as_of = str(curve[0]["timestamp"])
            else:
                total_equity = cash + sum(p.cost_basis for p in positions.values())
                as_of = None

            trades = engine.storage.get_trades(limit=100000)
            closed = [t for t in trades if t.side == "SELL" and t.realized_pnl is not None]
            wins = [t for t in closed if t.realized_pnl > 0]
            total_return_pct = ((total_equity - starting_capital) / starting_capital * 100.0) if starting_capital else 0.0

            rows.append({
                "profile": profile_name,
                "display_name": display_names.get(profile_name, profile_name),
                "strategy_mode": cfg.get_profile_strategy_mode(profile_name),
                "starting_capital": round(starting_capital, 2),
                "total_equity": round(total_equity, 2),
                "total_return_pct": round(total_return_pct, 2),
                "realized_pnl": round(engine.storage.get_total_realized_pnl(), 2),
                "cash": round(cash, 2),
                "open_positions": len(positions),
                "total_trades": len(trades),
                "closed_trades": len(closed),
                "win_rate_pct": round(len(wins) / len(closed) * 100.0, 1) if closed else None,
                "last_scan_at": engine.storage.get_last_scan_at(),
                "equity_as_of": as_of,
            })
        except Exception:  # noqa: BLE001 - one unreadable ledger must not blank the whole board
            log.exception("Could not build comparison row for profile %s", profile_name)

    # Best performer first, so the ranking is the point of the panel.
    rows.sort(key=lambda r: r["total_return_pct"], reverse=True)
    return {"profiles": rows, "as_of": datetime.now().isoformat(timespec="seconds")}


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
