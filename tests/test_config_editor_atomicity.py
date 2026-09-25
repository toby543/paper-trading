"""Atomicity guarantees for config_editor.py's file-writing functions.

update_config_file() writes one file via temp-file-then-atomic-replace, so
a crash mid-write can never leave that one file half-written. But saving
settings from the dashboard routinely touches TWO files in one action --
the Edit Settings panel shows global fields (data_source.*, engine.*) and
profile-scoped fields (strategy.*, risk.*, execution.*, regime.*) on the
same screen, and they land in config.yaml and profiles/<name>.yaml
respectively. Calling update_config_file() once per file let a failure on
the SECOND file leave the FIRST file's write already committed: the save
reports total failure while part of it silently succeeded.
update_config_files() exists to close that gap.
"""
import os

import pytest
import yaml as pyyaml

from papertrader.config_editor import update_config_file, update_config_files


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return pyyaml.safe_load(fh)


@pytest.fixture
def two_files(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("strategy:\n  fast_ma_days: 20\n", encoding="utf-8")
    b.write_text("risk:\n  stop_loss_pct: 8.0\n", encoding="utf-8")
    return str(a), str(b)


# --------------------------------------------------------- single-file
def test_update_config_file_leaves_original_untouched_on_bad_yaml(tmp_path):
    """A read failure (malformed existing YAML) must raise before any
    write -- the file that failed to parse is never touched, let alone
    truncated."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("strategy: [unterminated\n", encoding="utf-8")
    original = bad.read_text(encoding="utf-8")

    with pytest.raises(Exception):
        update_config_file(str(bad), [(["strategy", "x"], 1)])

    assert bad.read_text(encoding="utf-8") == original
    assert not (tmp_path / "broken.yaml.tmp").exists()


def test_update_config_file_writes_backup_and_new_content(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("strategy:\n  fast_ma_days: 20\n", encoding="utf-8")

    update_config_file(str(p), [(["strategy", "fast_ma_days"], 30)])

    assert _read(str(p))["strategy"]["fast_ma_days"] == 30
    assert _read(str(p) + ".bak")["strategy"]["fast_ma_days"] == 20  # pre-update snapshot
    assert not (tmp_path / "cfg.yaml.tmp").exists()  # renamed away, not left behind


# ------------------------------------------------------------ multi-file
def test_update_config_files_commits_all_targets(two_files):
    path_a, path_b = two_files
    update_config_files({
        path_a: [(["strategy", "fast_ma_days"], 99)],
        path_b: [(["risk", "stop_loss_pct"], 12.5)],
    })
    assert _read(path_a)["strategy"]["fast_ma_days"] == 99
    assert _read(path_b)["risk"]["stop_loss_pct"] == 12.5


def test_update_config_files_is_a_noop_for_empty_updates(two_files):
    """A path with an empty update list must not be touched at all --
    callers build this dict unconditionally (see api_update_settings),
    so a no-op path is the common case, not an edge case."""
    path_a, path_b = two_files
    before_a, before_b = _read(path_a), _read(path_b)

    update_config_files({path_a: [], path_b: []})

    assert _read(path_a) == before_a
    assert _read(path_b) == before_b
    assert not os.path.exists(path_a + ".bak")
    assert not os.path.exists(path_b + ".bak")


def test_update_config_files_all_or_nothing_when_one_file_is_malformed(two_files):
    """The regression this function exists to fix: if the SECOND file
    fails to parse, the FIRST file -- whose write would otherwise already
    have been committed by the old one-call-per-file approach -- must be
    left completely untouched, not silently updated while the response
    reports failure."""
    path_a, path_b = two_files
    with open(path_b, "w", encoding="utf-8") as fh:
        fh.write("risk: [unterminated\n")  # now invalid YAML
    original_a = open(path_a, encoding="utf-8").read()

    with pytest.raises(Exception):
        update_config_files({
            path_a: [(["strategy", "fast_ma_days"], 999)],
            path_b: [(["risk", "stop_loss_pct"], 5.0)],
        })

    # path_a must be byte-for-byte untouched -- not just "value unchanged",
    # since even a rewritten-but-equivalent file would indicate phase 2
    # started committing before phase 1 finished validating everything.
    assert open(path_a, encoding="utf-8").read() == original_a
    assert not os.path.exists(path_a + ".bak")
    assert not os.path.exists(path_a + ".tmp")


def test_update_config_files_leaves_no_tmp_litter_on_failure(two_files):
    """Phase 1 can create a.yaml.tmp before failing on b.yaml -- that temp
    file must be cleaned up, not left behind on every failed save."""
    path_a, path_b = two_files
    with open(path_b, "w", encoding="utf-8") as fh:
        fh.write("risk: [unterminated\n")

    with pytest.raises(Exception):
        update_config_files({
            path_a: [(["strategy", "fast_ma_days"], 999)],
            path_b: [(["risk", "stop_loss_pct"], 5.0)],
        })

    assert not os.path.exists(path_a + ".tmp")


def test_update_config_files_demonstrates_the_bug_the_old_call_pattern_had(two_files):
    """Not a test of update_config_files -- a test of what calling
    update_config_file() once per file (the old api_update_settings code)
    actually did: committed the first file, then raised on the second,
    leaving a partial save with no way for the caller to tell from the
    exception alone. This pins that old behavior down as the regression
    update_config_files() fixes; if this test ever starts failing, the
    single-file function's atomicity guarantee has changed underneath it.
    """
    path_a, path_b = two_files
    with open(path_b, "w", encoding="utf-8") as fh:
        fh.write("risk: [unterminated\n")

    update_config_file(path_a, [(["strategy", "fast_ma_days"], 999)])  # succeeds
    with pytest.raises(Exception):
        update_config_file(path_b, [(["risk", "stop_loss_pct"], 5.0)])  # then this fails

    # This is the bug: path_a is now silently updated even though the
    # overall "save settings" action the caller intended should have
    # failed as a whole.
    assert _read(path_a)["strategy"]["fast_ma_days"] == 999
