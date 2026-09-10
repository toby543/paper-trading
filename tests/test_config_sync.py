"""Tests for find_missing_keys -- the merge logic behind `sync-config`.

Needs ruamel.yaml (config_editor.py's import for update_config_file, which
this file is imported alongside), so it can't run in an environment
without that installed. Every assertion here was verified first against
find_missing_keys's exact source, extracted and executed standalone
(no ruamel needed) against both synthetic fixtures and the real
config.default.yaml -- see the "Add a way to..." commit for that
transcript. This file is the permanent, executable version of that check.
"""
import copy
import os

import yaml

from papertrader.config_editor import find_missing_keys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_new_top_level_key_added_wholesale():
    """A brand-new profile (or any other whole new top-level block) is
    added in one piece -- never partially, which could leave a malformed
    fragment (e.g. a profile with a display_name but no strategy:)."""
    default = {"profiles": {"a": {"x": 1}, "b": {"y": 2, "z": 3}}}
    live = {"profiles": {"a": {"x": 1}}}
    missing = find_missing_keys(default, live)
    assert missing == [(["profiles", "b"], {"y": 2, "z": 3})]


def test_missing_nested_key_in_a_partially_customized_block_is_added():
    default = {"profiles": {"a": {"strategy": {"x": 1, "y": 2}}}}
    live = {"profiles": {"a": {"strategy": {"x": 999}}}}  # user customized x, never had y
    missing = find_missing_keys(default, live)
    assert missing == [(["profiles", "a", "strategy", "y"], 2)]


def test_customized_value_is_never_reported_or_touched():
    """The whole point: a key the user already has -- however they set
    it, even if it differs from the current default -- must never appear
    in the diff, full stop."""
    default = {"risk": {"stop_loss_pct": 7.0}}
    live = {"risk": {"stop_loss_pct": 3.5}}
    assert find_missing_keys(default, live) == []


def test_live_only_key_is_never_flagged_for_removal():
    """This function only ever adds. A setting present in the user's file
    but absent from the (newer) template -- e.g. something removed
    upstream -- must never show up in the diff; sync-config has no
    delete path at all."""
    default = {"risk": {"stop_loss_pct": 7.0}}
    live = {"risk": {"stop_loss_pct": 7.0}, "legacy_setting": "still here"}
    assert find_missing_keys(default, live) == []


def test_scalar_in_live_where_default_has_a_dict_is_left_alone():
    """If the user's value isn't a dict where the default's is, don't
    recurse into it or replace it -- whatever they have is final,
    regardless of its shape."""
    default = {"strategy": {"mode": "52w_high", "extra": {"deep": 1}}}
    live = {"strategy": {"mode": "52w_high", "extra": "not_a_dict_but_thats_fine"}}
    missing = find_missing_keys(default, live)
    assert missing == []  # "extra" key IS present in live -> never touched, never recursed into


def test_empty_when_already_in_sync():
    default = {"a": 1, "b": {"c": 2}}
    live = copy.deepcopy(default)
    assert find_missing_keys(default, live) == []


def test_real_config_default_yaml_adding_a_new_profile():
    """End-to-end against the actual shipped template: simulate a live
    config from before pivot_supertrend existed, with a real user
    customization elsewhere, and confirm sync-config would add exactly
    the new profile and nothing else."""
    default_path = os.path.join(_REPO_ROOT, "config.default.yaml")
    default = yaml.safe_load(open(default_path, encoding="utf-8"))
    live = copy.deepcopy(default)
    del live["profiles"]["pivot_supertrend"]
    live["profiles"]["52w_high"]["strategy"]["proximity_to_52w_high_pct"] = 6.5

    missing = find_missing_keys(default, live)
    paths = {tuple(p) for p, _ in missing}

    assert ("profiles", "pivot_supertrend") in paths
    assert ("profiles", "52w_high", "strategy", "proximity_to_52w_high_pct") not in paths
    # Exactly one thing changed (the new profile) -- nothing else should differ.
    assert len(missing) == 1
