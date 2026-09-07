#!/usr/bin/env python3
"""Reset one or more profile ledgers back to a fresh starting-capital state.

Use this after running inspect_profile_ledgers.py to identify which
ledger(s) got contaminated by the multi-profile ledger-sharing bug (all
engines sharing one SQLite file before the fix). This deletes the
positions/trades/equity_curve rows and resets cash to the profile's
configured starting_capital -- it does NOT touch any ledger you don't
name, so the one you want to keep is left completely untouched.

Usage:
    python scripts/reset_profile_ledger.py cross_sectional consolidation_breakout
    python scripts/reset_profile_ledger.py --all        # reset every configured profile
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papertrader.config import Config  # noqa: E402


def reset(db_path: Path, starting_capital: float) -> None:
    if not db_path.exists():
        print(f"  {db_path.name}: does not exist yet, nothing to reset")
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM equity_curve")
        conn.execute("UPDATE account SET cash = ?, last_scan_at = NULL WHERE id = 1", (starting_capital,))
        conn.commit()
        print(f"  {db_path.name}: reset to cash={starting_capital:,.2f}, positions/trades/equity_curve cleared")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="*", help="Profile name(s) to reset, e.g. cross_sectional consolidation_breakout")
    parser.add_argument("--all", action="store_true", help="Reset every profile configured in config.yaml")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    all_profiles = cfg.list_profiles()

    if args.all:
        targets = list(all_profiles)
    elif args.profiles:
        unknown = [p for p in args.profiles if p not in all_profiles]
        if unknown:
            print(f"Unknown profile(s): {unknown}. Configured profiles: {list(all_profiles)}")
            sys.exit(1)
        targets = args.profiles
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Resetting {len(targets)} profile(s): {targets}\n")
    for profile_name in targets:
        db_path = Path(cfg.get_profile_state_file(profile_name))
        starting_capital = cfg.get_profile_starting_capital(profile_name)
        reset(db_path, starting_capital)

    print("\nDone. Any profile not named above was left untouched.")


if __name__ == "__main__":
    main()
