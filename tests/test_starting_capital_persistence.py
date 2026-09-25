"""starting_capital must be fixed at account creation and never drift.

Every %-return / PnL% figure the dashboard shows (build_summary,
build_strategy_comparison) is computed against this value as the
denominator. It used to be re-read from Config.get_profile_starting_capital()
on every single calculation -- the live config file, editable at any time
via Edit Settings. A profile's "Starting capital" setting is documented as
only applying "the first time its ledger database is created", but nothing
enforced that: editing it for a profile that had already traded silently
recomputed that profile's entire PnL history against a baseline it never
actually started from, permanently, with no way to detect or recover the
true original value (the database only ever stored `cash`, which drifts
with every trade).
"""
import os
import sqlite3

import pytest

from papertrader.portfolio.storage import Storage


def test_starting_capital_persists_across_reopens(tmp_path):
    db_path = os.path.join(tmp_path, "state.db")
    Storage(db_path, starting_capital=100_000.0)

    reopened = Storage(db_path, starting_capital=100_000.0)
    assert reopened.get_starting_capital() == 100_000.0


def test_starting_capital_does_not_drift_when_config_changes(tmp_path):
    """The actual bug: a later config edit must not retroactively change
    what an already-trading profile's return% is measured against."""
    db_path = os.path.join(tmp_path, "state.db")
    Storage(db_path, starting_capital=100_000.0)

    # Simulate the server restarting after someone edited "Starting
    # capital" in Edit Settings -- Storage is reconstructed with whatever
    # config.get_profile_starting_capital() now returns.
    drifted = Storage(db_path, starting_capital=250_000.0)

    assert drifted.get_starting_capital() == 100_000.0
    # And cash is untouched by the reopen either way.
    assert drifted.get_cash() == 100_000.0


def test_legacy_database_without_the_column_is_backfilled(tmp_path):
    """A ledger created before this column existed (or migrated via
    ALTER TABLE with the column still NULL) must be backfilled once from
    whatever value it's opened with -- preserving today's behavior exactly
    -- without disturbing `cash`, which has already drifted through real
    trades and must never be touched by this migration."""
    db_path = os.path.join(tmp_path, "state.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "cash REAL NOT NULL, last_scan_at TEXT)"
    )
    conn.execute("INSERT INTO account (id, cash) VALUES (1, 87654.0)")
    conn.commit()
    conn.close()

    migrated = Storage(db_path, starting_capital=100_000.0)

    assert migrated.get_starting_capital() == 100_000.0
    assert migrated.get_cash() == 87654.0  # untouched


def test_backfilled_value_locks_in_and_does_not_drift_either(tmp_path):
    """The migration in the previous test must itself be one-time -- a
    legacy DB, once backfilled, behaves identically to a DB that had the
    column from the start: immune to later config changes."""
    db_path = os.path.join(tmp_path, "state.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "cash REAL NOT NULL, last_scan_at TEXT)"
    )
    conn.execute("INSERT INTO account (id, cash) VALUES (1, 87654.0)")
    conn.commit()
    conn.close()

    Storage(db_path, starting_capital=100_000.0)  # backfills to 100,000
    reopened_again = Storage(db_path, starting_capital=999_999.0)  # config drifted further

    assert reopened_again.get_starting_capital() == 100_000.0
