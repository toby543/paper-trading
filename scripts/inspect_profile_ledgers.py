#!/usr/bin/env python3
"""Inspect all profile ledger files to see what's actually in each one.

Run this after pulling the multi-profile bug fix to check whether your
data/state_*.db files contain trades from a single mixed-up strategy
(the bug) or look correctly separated.

Usage:
    python scripts/inspect_profile_ledgers.py
"""
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


def inspect(db_path: Path) -> None:
    print(f"\n=== {db_path.name} ===")
    if not db_path.exists():
        print("  (file does not exist yet)")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cash_row = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
        print(f"  cash: {cash_row['cash'] if cash_row else 'N/A'}")

        positions = conn.execute("SELECT symbol, quantity, avg_price, entry_date FROM positions").fetchall()
        print(f"  open positions: {len(positions)}")
        for p in positions:
            print(f"    {p['symbol']:12s} qty={p['quantity']:6d} avg={p['avg_price']:10.2f} entry={p['entry_date']}")

        trades = conn.execute(
            "SELECT symbol, side, quantity, price, reason, timestamp FROM trades ORDER BY id"
        ).fetchall()
        print(f"  total trades: {len(trades)}")
        # Show reasons seen -- if trades from multiple strategies got
        # mixed into one file, you'll often see reason strings typical
        # of more than one strategy module here (e.g. both 52w-high-style
        # and consolidation-breakout-style exit reasons).
        reasons = sorted({t["reason"] for t in trades})
        if reasons:
            print(f"  distinct trade reasons seen: {reasons}")
        for t in trades[:10]:
            print(f"    {t['timestamp']}  {t['side']:4s} {t['symbol']:12s} qty={t['quantity']:6d} @ {t['price']:10.2f}  ({t['reason']})")
        if len(trades) > 10:
            print(f"    ... and {len(trades) - 10} more")
    finally:
        conn.close()


if __name__ == "__main__":
    for name in ["state_52w_high.db", "state_cross_sectional.db", "state_consolidation.db", "state.db"]:
        inspect(DATA_DIR / name)
