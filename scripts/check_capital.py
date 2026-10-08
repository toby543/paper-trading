"""Compare each profile's configured starting capital with what its ledger holds.

A ledger keeps the capital it was created with, but one that has never traded
(no trades, no open positions) follows the setting. This shows which profiles
differ and why, and with --apply updates the untouched ones right now instead
of waiting for the app to restart.

    python scripts/check_capital.py                       # report
    python scripts/check_capital.py --apply               # update untouched ledgers now
    python scripts/check_capital.py --profile manual_swing --apply
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from papertrader.config import Config  # noqa: E402
from papertrader.portfolio.storage import Storage  # noqa: E402


def describe(db_path: str, configured: float) -> dict:
    """What the ledger at db_path holds against the configured capital.
    status: no_ledger | ok | will_update | locked"""
    if not os.path.exists(db_path):
        return {"status": "no_ledger", "configured": configured}
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT cash, starting_capital FROM account WHERE id = 1").fetchone()
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    finally:
        conn.close()
    cash, start = (row[0], row[1]) if row else (None, None)
    info = {"configured": configured, "cash": cash, "starting_capital": start, "trades": trades, "positions": positions}
    if trades or positions:
        info["status"] = "ok" if start == configured else "locked"
    else:
        info["status"] = "ok" if (cash == configured and start == configured) else "will_update"
    return info


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - purely informational
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="update untouched ledgers now")
    ap.add_argument("--profile", help="only this profile")
    args = ap.parse_args()

    print(f"Code: branch {_git('rev-parse', '--abbrev-ref', 'HEAD')}, commit {_git('rev-parse', '--short', 'HEAD')}")
    cfg = Config.load()
    for name in cfg.list_profiles():
        if args.profile and name != args.profile:
            continue
        path = cfg.get_profile_state_file(name)
        configured = cfg.get_profile_starting_capital(name)
        info = describe(path, configured)
        line = f"{name:28s} configured {configured:>12,.2f}"
        if info["status"] == "no_ledger":
            print(f"{line}  no ledger yet -- it will be created with this capital")
            continue
        line += f" | ledger cash {info['cash']:>12,.2f}, capital {info['starting_capital']:>12,.2f}" \
                f" | {info['trades']} trades, {info['positions']} positions"
        if info["status"] == "ok":
            print(f"{line}  OK")
        elif info["status"] == "locked":
            print(f"{line}  LOCKED: it has traded, so it keeps the capital it started with")
        elif args.apply:
            changed = Storage(path, configured).adopt_starting_capital_if_untouched(configured)
            print(f"{line}  UPDATED to {configured:,.2f}" if changed else f"{line}  nothing to change")
        else:
            print(f"{line}  DIFFERS: untouched, so it updates on the next start (or run with --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
