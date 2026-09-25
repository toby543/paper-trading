"""Every profile's TradingEngine shares ONE Config object (see how
`engines` is built in cli.py's serve command and app.py's create_app) --
not one Config instance per engine. Config.reload()'s "has the file
changed since I last saw it" check is state on THAT SHARED OBJECT
(_last_modified), not per caller.

run_forever() used to gate its whole per-engine settings refresh behind
`if self.cfg.reload():`. With N engines sharing one Config, only whichever
engine's background thread happened to poll first after an edit ever saw
reload() return True and actually refreshed itself; the other N-1 calls
all returned False (someone else already reloaded it moments earlier) and
those engines silently kept running on stale strategy/risk/execution/
scan-interval settings indefinitely. Which single engine "won" was
effectively random, depending on each engine's own sleep timing -- not
something a user editing settings for one specific profile could rely on,
or even know happened.

The fix: call reload() unconditionally for its side effect (it updates the
shared cfg.raw if the file changed, regardless of who calls it or what it
returns to them), then have every engine unconditionally re-derive its own
state from cfg on every loop iteration -- never gated on that call's own
return value.
"""
import os
import time

from papertrader.config import Config


def test_second_caller_sees_fresh_data_even_though_reload_returns_false(tmp_path):
    """The actual race, reproduced directly: two callers share one Config
    object, as two engines do. The first call after an edit legitimately
    returns True; the second returns False purely because the first
    caller already consumed the "did it change" signal -- but the
    underlying data (cfg.raw) is shared state and reflects the edit
    either way. A caller that only refreshes itself when its OWN call
    returned True misses the update entirely; one that always re-reads
    cfg.raw regardless never does."""
    path = tmp_path / "config.yaml"
    path.write_text("engine:\n  scan_interval_minutes: 15\n", encoding="utf-8")
    cfg = Config.load(str(path))
    assert cfg.get("engine", "scan_interval_minutes") == 15

    # A settings save changes the file on disk.
    time.sleep(1.1)  # mtime resolution on some filesystems is ~1s
    path.write_text("engine:\n  scan_interval_minutes: 2\n", encoding="utf-8")

    # "Engine A" polls first.
    assert cfg.reload() is True
    assert cfg.get("engine", "scan_interval_minutes") == 2

    # "Engine B" polls moments later, same shared object, nothing has
    # changed since A already reloaded it -- reload() correctly reports
    # nothing NEW for B to react to...
    assert cfg.reload() is False
    # ...but the data B would actually use is not gated on that return
    # value, and already reflects the edit -- this is what a caller doing
    # an unconditional re-read (the fix) sees.
    assert cfg.get("engine", "scan_interval_minutes") == 2


def test_engines_dict_shares_one_config_object():
    """Guards the premise the bug depends on: if cli.py/app.py ever
    switched to constructing a separate Config per engine, this whole
    failure mode would stop applying and the test above would no longer
    describe the real system. Reads the actual construction sites as
    source rather than importing and running them (both need a real
    config.yaml and a running Flask app to instantiate)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli_source = open(os.path.join(repo_root, "src", "papertrader", "cli.py"), encoding="utf-8").read()
    app_source = open(os.path.join(repo_root, "src", "papertrader", "web", "app.py"), encoding="utf-8").read()

    assert "TradingEngine(cfg, profile_name=p) for p in profiles" in cli_source
    assert "TradingEngine(cfg, profile_name=name) for name in cfg.list_profiles()" in app_source


def test_run_forever_does_not_gate_the_refresh_behind_reload_return_value():
    """Structural regression guard: run_forever() must call
    self.cfg.reload() unconditionally, for its side effect, NOT as the
    condition of an `if` -- gating the per-engine settings refresh behind
    `if self.cfg.reload():` is the exact shape of the bug (see this
    module's docstring). Reads the method's source directly since actually
    exercising the infinite loop's multi-engine race in a unit test isn't
    practical.

    A weaker, indentation-based version of this check was tried first and
    did not actually fail when the bug was manually reintroduced (a
    single-level nest can read as "close enough" under a fuzzy indent
    comparison) -- this checks for the literal conditional construct
    instead, which is unambiguous and was verified to fail correctly when
    the bug was reintroduced.
    """
    import inspect

    from papertrader.engine.scheduler import TradingEngine

    source = inspect.getsource(TradingEngine.run_forever)
    assert "if self.cfg.reload():" not in source, (
        "self.cfg.reload() must be called unconditionally, not as an `if` "
        "condition -- self.cfg is shared across every profile's engine, so "
        "gating the settings refresh on THIS call's own True/False means "
        "only whichever engine polls first after an edit ever refreshes "
        "itself; see this module's docstring for the full failure mode"
    )
