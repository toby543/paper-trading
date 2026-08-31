"""SQLite-backed persistence for cash, positions, trades and equity curve."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .models import Position, Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    highest_close_since_entry REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    charges REAL NOT NULL,
    reason TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    realized_pnl REAL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    timestamp TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_equity REAL NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str, starting_capital: float):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._init_db(starting_capital)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self, starting_capital: float) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            if row is None:
                conn.execute("INSERT INTO account (id, cash) VALUES (1, ?)", (starting_capital,))
            existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "realized_pnl" not in existing_cols:
                conn.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL")
            self._backfill_realized_pnl(conn)

    def _backfill_realized_pnl(self, conn) -> None:
        """Fill in realized_pnl for SELL trades recorded before that column
        existed, by replaying the full trade history per symbol to
        reconstruct the weighted-average cost at the time of each sell."""
        rows = conn.execute(
            "SELECT id, symbol, side, quantity, price, charges, realized_pnl FROM trades ORDER BY id ASC"
        ).fetchall()
        if not any(r["side"] == "SELL" and r["realized_pnl"] is None for r in rows):
            return
        avg_cost: dict[str, tuple[int, float]] = {}
        for r in rows:
            qty, avg = avg_cost.get(r["symbol"], (0, 0.0))
            if r["side"] == "BUY":
                new_qty = qty + r["quantity"]
                new_avg = (avg * qty + r["price"] * r["quantity"]) / new_qty if new_qty else 0.0
                avg_cost[r["symbol"]] = (new_qty, new_avg)
            else:  # SELL
                if r["realized_pnl"] is None:
                    realized = (r["price"] - avg) * r["quantity"] - r["charges"]
                    conn.execute("UPDATE trades SET realized_pnl = ? WHERE id = ?", (realized, r["id"]))
                avg_cost[r["symbol"]] = (qty - r["quantity"], avg)

    # ---- account -----------------------------------------------------
    def get_cash(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            return float(row["cash"])

    def set_cash(self, cash: float) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE account SET cash = ? WHERE id = 1", (cash,))

    # ---- positions -----------------------------------------------------
    def get_positions(self) -> dict[str, Position]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
        return {
            r["symbol"]: Position(
                symbol=r["symbol"],
                quantity=r["quantity"],
                avg_price=r["avg_price"],
                entry_date=r["entry_date"],
                highest_close_since_entry=r["highest_close_since_entry"],
            )
            for r in rows
        }

    def upsert_position(self, pos: Position) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO positions (symbol, quantity, avg_price, entry_date, highest_close_since_entry)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(symbol) DO UPDATE SET
                     quantity=excluded.quantity,
                     avg_price=excluded.avg_price,
                     entry_date=excluded.entry_date,
                     highest_close_since_entry=excluded.highest_close_since_entry""",
                (pos.symbol, pos.quantity, pos.avg_price, pos.entry_date, pos.highest_close_since_entry),
            )

    def delete_position(self, symbol: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    # ---- trades -----------------------------------------------------
    def record_trade(self, trade: Trade) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO trades (symbol, side, quantity, price, charges, reason, timestamp, realized_pnl)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade.symbol, trade.side, trade.quantity, trade.price, trade.charges, trade.reason,
                 trade.timestamp, trade.realized_pnl),
            )

    def get_trades(self, limit: int | None = None) -> list[Trade]:
        query = "SELECT * FROM trades ORDER BY id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self._conn() as conn:
            rows = conn.execute(query).fetchall()
        return [
            Trade(
                id=r["id"], symbol=r["symbol"], side=r["side"], quantity=r["quantity"],
                price=r["price"], charges=r["charges"], reason=r["reason"], timestamp=r["timestamp"],
                realized_pnl=r["realized_pnl"] if "realized_pnl" in r.keys() else None,
            )
            for r in rows
        ]

    def get_total_realized_pnl(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT SUM(realized_pnl) AS total FROM trades WHERE side = 'SELL'").fetchone()
        return float(row["total"]) if row and row["total"] is not None else 0.0

    # ---- equity curve -----------------------------------------------------
    def record_equity(self, cash: float, positions_value: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO equity_curve (timestamp, cash, positions_value, total_equity)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(timestamp) DO UPDATE SET
                     cash=excluded.cash, positions_value=excluded.positions_value, total_equity=excluded.total_equity""",
                (datetime.now().isoformat(timespec="seconds"), cash, positions_value, cash + positions_value),
            )

    def get_equity_curve(self, limit: int = 200) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
