"""One-off: fill in entry_charges for open positions opened before it was tracked.

Those positions recorded 0, so their unrealized P&L (and the realized P&L when
they are eventually sold) ignores the fee paid to open them. The fee is in the
trades table, so it is rebuilt from there. Each ledger is copied to
<name>.bak-entry-charges before it is touched. Stop the app first.

    python scripts/backfill_entry_charges.py            # every data/state_*.db
    python scripts/backfill_entry_charges.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from papertrader.portfolio.storage import Storage  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "state_*.db"))):
        name = os.path.basename(path)
        probe = sqlite3.connect(path)
        try:
            todo = probe.execute("SELECT COUNT(*) FROM positions WHERE entry_charges = 0").fetchone()[0]
        finally:
            probe.close()
        if not todo:
            print(f"{name}: nothing to backfill")
            continue
        if args.dry_run:
            print(f"{name}: {todo} position(s) would be checked")
            continue
        shutil.copy2(path, path + ".bak-entry-charges")
        updated = Storage(path, 0.0).backfill_entry_charges()
        print(f"{name}: updated {updated} of {todo} position(s) (backup saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
