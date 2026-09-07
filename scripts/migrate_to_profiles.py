#!/usr/bin/env python3
"""
Migrate from single state.db to profile-based ledgers.

If you have an existing state.db file from before the profile system,
this script will move it to the 52w_high profile to preserve your trades.

Usage:
    python scripts/migrate_to_profiles.py
"""
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

OLD_STATE_FILE = DATA_DIR / "state.db"
PROFILE_52W_HIGH = DATA_DIR / "state_52w_high.db"
PROFILE_CROSS_SECTIONAL = DATA_DIR / "state_cross_sectional.db"
PROFILE_CONSOLIDATION = DATA_DIR / "state_consolidation.db"


def migrate():
    """Migrate existing state.db to 52w_high profile."""
    print("Profile Migration Tool")
    print("=" * 60)

    # Check if old state file exists
    if OLD_STATE_FILE.exists():
        print(f"\n✓ Found existing ledger: {OLD_STATE_FILE}")
        print(f"  This will be migrated to: {PROFILE_52W_HIGH}")
        print(f"  (Preserving all your existing trades under 52w_high profile)")

        if PROFILE_52W_HIGH.exists():
            print(f"\n⚠ Warning: {PROFILE_52W_HIGH} already exists!")
            response = input("Overwrite it with your existing trades? (y/n): ")
            if response.lower() != "y":
                print("Migration cancelled.")
                return

        shutil.move(str(OLD_STATE_FILE), str(PROFILE_52W_HIGH))
        print(f"\n✓ Migrated: {OLD_STATE_FILE} → {PROFILE_52W_HIGH}")
    else:
        print(f"\n✓ No existing ledger found at {OLD_STATE_FILE}")
        print("  All profiles will start with fresh ledgers.")

    # Verify profile ledger paths
    print("\n" + "=" * 60)
    print("Profile Ledgers:")
    print(f"  52w_high:              {PROFILE_52W_HIGH}")
    print(f"  cross_sectional:       {PROFILE_CROSS_SECTIONAL}")
    print(f"  consolidation_breakout: {PROFILE_CONSOLIDATION}")

    if PROFILE_52W_HIGH.exists():
        size_mb = PROFILE_52W_HIGH.stat().st_size / (1024 * 1024)
        print(f"\n✓ 52w_high has existing trades ({size_mb:.2f} MB)")
    else:
        print(f"\n  52w_high ledger will be created on first run")

    if PROFILE_CROSS_SECTIONAL.exists():
        size_mb = PROFILE_CROSS_SECTIONAL.stat().st_size / (1024 * 1024)
        print(f"✓ cross_sectional has trades ({size_mb:.2f} MB)")
    else:
        print(f"  cross_sectional ledger will be created on first run")

    if PROFILE_CONSOLIDATION.exists():
        size_mb = PROFILE_CONSOLIDATION.stat().st_size / (1024 * 1024)
        print(f"✓ consolidation_breakout has trades ({size_mb:.2f} MB)")
    else:
        print(f"  consolidation_breakout ledger will be created on first run")

    print("\n" + "=" * 60)
    print("Ready! You can now:")
    print("  1. Run '52w_high' profile to continue with existing trades")
    print("  2. Switch to other profiles for independent testing")
    print("  3. Enable multi_profile_mode in config.yaml to run all 3")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
