"""Tests for universe repair. The probe is injected, so these run offline."""
import csv

from papertrader.data.universe_doctor import KNOWN_RENAMES, apply_report, diagnose


def _probe(live: set[str]):
    return lambda sym: sym in live


def test_stale_alias_is_reported_as_duplicate_not_rename():
    """NALCO is dead but NATIONALUM -- the same company -- is already in the
    file, so the entry should be dropped, never 'renamed' onto a symbol that
    is already there."""
    universe = ["RELIANCE", "NALCO", "NATIONALUM"]
    r = diagnose(universe, probe=_probe({"RELIANCE", "NATIONALUM"}), pause_seconds=0)
    assert r.duplicates == [("NALCO", "NATIONALUM")]
    assert r.renamed == []
    assert r.unresolved == []


def test_verified_rename_is_applied():
    universe = ["RELIANCE", "BHARATFORGE"]
    r = diagnose(universe, probe=_probe({"RELIANCE", "BHARATFORG"}), pause_seconds=0)
    assert r.renamed == [("BHARATFORGE", "BHARATFORG")]
    assert r.duplicates == []


def test_unverifiable_rename_is_never_applied():
    """The candidate mappings are guesses. If the replacement doesn't probe
    clean, the symbol must land in `unresolved` -- never be written."""
    universe = ["BHARATFORGE"]
    r = diagnose(universe, probe=_probe(set()), pause_seconds=0)
    assert r.renamed == []
    assert r.unresolved == ["BHARATFORGE"]


def test_unknown_dead_symbol_is_unresolved():
    r = diagnose(["ZZZDEAD"], probe=_probe(set()), pause_seconds=0)
    assert r.unresolved == ["ZZZDEAD"]


def _write(path, symbols):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol"])
        w.writeheader()
        w.writerows({"symbol": s} for s in symbols)


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return [r["symbol"] for r in csv.DictReader(fh)]


def test_apply_drops_aliases_and_renames_in_place(tmp_path):
    p = tmp_path / "u.csv"
    _write(p, ["RELIANCE", "NALCO", "NATIONALUM", "BHARATFORGE"])
    live = {"RELIANCE", "NATIONALUM", "BHARATFORG"}
    r = diagnose(_read(p), probe=_probe(live), pause_seconds=0)
    apply_report(str(p), r)
    assert _read(p) == ["RELIANCE", "NATIONALUM", "BHARATFORG"]


def test_apply_keeps_unresolved_unless_asked(tmp_path):
    p = tmp_path / "u.csv"
    _write(p, ["RELIANCE", "ZZZDEAD"])
    r = diagnose(_read(p), probe=_probe({"RELIANCE"}), pause_seconds=0)
    apply_report(str(p), r)
    assert _read(p) == ["RELIANCE", "ZZZDEAD"], "one failed probe must not shrink the universe"
    apply_report(str(p), r, drop_unresolved=True)
    assert _read(p) == ["RELIANCE"]


def test_rename_colliding_with_existing_entry_does_not_duplicate(tmp_path):
    p = tmp_path / "u.csv"
    _write(p, ["BHARATFORGE", "BHARATFORG"])
    # BHARATFORG present, so BHARATFORGE is an alias duplicate, not a rename.
    r = diagnose(_read(p), probe=_probe({"BHARATFORG"}), pause_seconds=0)
    apply_report(str(p), r)
    assert _read(p) == ["BHARATFORG"]


def test_rename_map_never_maps_a_symbol_to_itself():
    for stale, (new, _note) in KNOWN_RENAMES.items():
        assert stale != new, f"{stale} maps to itself"
        assert new, f"{stale} has an empty replacement"
