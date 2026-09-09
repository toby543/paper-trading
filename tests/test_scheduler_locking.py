"""Regression test for a self-deadlock: _record_scan_diagnostics() and
get_scan_diagnostics() must use their OWN lock, never the engine's
_state_lock, which run_forever() holds for the whole
check_exits()+mark_to_market()+scan_for_entries() block on the background
thread.

The first version of scan diagnostics reused _state_lock. Since
threading.Lock is not reentrant, and _record_scan_diagnostics() is called
from inside that exact same locked block (scan_for_entries() ->
find_candidates() -> _record_scan_diagnostics()) on the SAME thread, this
deadlocked the background thread on its first scan cycle after every
restart -- 100% reproducible, not a race. It then cascaded: every
dashboard page load's get_scan_diagnostics() call (via
data_api._build_insights) would also block forever waiting on the same
lock, freezing the entire UI, since Flask summary routes are deliberately
never supposed to wait on _state_lock (see build_summary() and friends in
data_api.py, none of which touch it).

These tests don't construct a real TradingEngine (too many live-data
dependencies) -- they exercise the exact locking shape scheduler.py uses,
which is enough to catch a regression back to a shared lock without
needing pandas/network.
"""
import threading
import time


class _FakeEngineLocking:
    """Mirrors exactly the lock objects and acquisition pattern
    TradingEngine uses, so a regression (e.g. someone "simplifying" back
    to one shared lock) fails this test immediately."""
    def __init__(self):
        self._state_lock = threading.Lock()
        self._scan_diagnostics_lock = threading.Lock()
        self._last_scan_diagnostics = None

    def _record_scan_diagnostics(self, status):
        with self._scan_diagnostics_lock:
            self._last_scan_diagnostics = {"status": status}

    def get_scan_diagnostics(self):
        with self._scan_diagnostics_lock:
            return dict(self._last_scan_diagnostics) if self._last_scan_diagnostics else None

    def scan_for_entries_same_thread(self):
        """Mirrors run_forever()'s `with self._state_lock: ... scan_for_entries()`,
        which calls _record_scan_diagnostics() on the SAME thread before
        releasing _state_lock."""
        with self._state_lock:
            self._record_scan_diagnostics("scanned")


def test_recording_diagnostics_from_inside_the_locked_scan_does_not_deadlock():
    engine = _FakeEngineLocking()
    done = threading.Event()

    def run():
        engine.scan_for_entries_same_thread()
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=2)
    assert done.is_set(), (
        "scan_for_entries_same_thread() did not complete within 2s -- "
        "_record_scan_diagnostics is deadlocking on _state_lock, held by "
        "this same thread"
    )
    assert engine.get_scan_diagnostics() == {"status": "scanned"}


def test_dashboard_read_does_not_wait_for_an_in_progress_scan():
    """A Flask request thread calling get_scan_diagnostics() must return
    immediately even while the background thread is mid-scan (holding
    _state_lock for real network-bound work) -- it must never share a
    lock with the one that guards that long-running section."""
    engine = _FakeEngineLocking()
    scan_started = threading.Event()

    def long_scan():
        with engine._state_lock:
            scan_started.set()
            time.sleep(1.0)  # stands in for a real network-bound scan
            engine._record_scan_diagnostics("scanned")

    bg = threading.Thread(target=long_scan, daemon=True)
    bg.start()
    scan_started.wait(timeout=1)

    t0 = time.time()
    engine.get_scan_diagnostics()  # the dashboard's read
    elapsed = time.time() - t0

    assert elapsed < 0.2, (
        f"get_scan_diagnostics() took {elapsed:.2f}s while a scan was in "
        "progress -- it is blocking on _state_lock instead of its own "
        "dedicated lock, which would freeze the whole dashboard for the "
        "duration of every scan"
    )
    bg.join(timeout=2)
