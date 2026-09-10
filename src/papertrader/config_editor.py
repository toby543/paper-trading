"""Safely apply setting changes to config.yaml.

Uses ruamel.yaml's round-trip mode instead of a plain load-then-dump with
PyYAML, which would silently strip every explanatory comment the file
relies on (config.yaml is meant to be read, not just parsed). Writes to a
temp file and atomically replaces the original, and keeps a one-generation
`.bak` backup, so a crash mid-write or an unexpected library edge case
can't leave config.yaml corrupted or unrecoverable.
"""
from __future__ import annotations

import os
import shutil
import threading

from ruamel.yaml import YAML

_lock = threading.Lock()


def update_config_file(path: str, updates: list[tuple[list[str], object]]) -> None:
    """Apply `updates` -- a list of (key_path, new_value) pairs, e.g.
    (["strategy", "min_momentum_return_pct"], 20.0) -- to the YAML file at
    `path`, in place, preserving comments and formatting."""
    if not updates:
        return

    import logging
    log = logging.getLogger(__name__)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # don't let ruamel line-wrap long comments

    with _lock:
        try:
            # Read current config
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.load(fh)
            log.info("Loaded config from %s", path)

            # Apply updates. setdefault (not plain indexing) so a path
            # whose intermediate dicts don't exist yet in the file gets
            # them created on the fly -- e.g. the first time a setting is
            # saved for a newly-added profile that doesn't have a full
            # strategy: block pre-populated yet.
            for key_path, value in updates:
                node = data
                for key in key_path[:-1]:
                    node = node.setdefault(key, {})
                old_value = node.get(key_path[-1])
                node[key_path[-1]] = value
                log.info("Updated %s: %s -> %s", ".".join(key_path), old_value, value)

            # Backup and write
            shutil.copyfile(path, path + ".bak")
            log.info("Created backup: %s.bak", path)

            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh)
            log.info("Wrote temp file: %s", tmp_path)

            os.replace(tmp_path, path)
            log.info("Replaced config file: %s", path)
        except Exception as e:
            log.error("Failed to update config file: %s", e)
            raise


def find_missing_keys(
    default: dict, live: dict, _prefix: list[str] | None = None,
) -> list[tuple[list[str], object]]:
    """Recursively find keys present in `default` (config.default.yaml,
    the tracked template) but absent from `live` (the user's actual
    config.yaml), returned as (key_path, value) pairs ready for
    update_config_file -- this is the whole of `sync-config`.

    Deliberately additive-only, in both directions:
      - A key missing from `live` entirely gets added, whole subtree and
        all (e.g. a brand-new profile block) -- no partial copy that
        could leave a malformed fragment behind.
      - A key `live` already has is NEVER touched, even if the default's
        value has since changed -- the user's customization always wins,
        full stop. This is the entire point: sync-config must be safe to
        run blindly after every pull without re-reviewing every setting.
      - We only recurse into a key when BOTH sides have a dict there (a
        profile the user has partially customized still picks up any new
        field added to its strategy: block, without touching the fields
        they already set). If the user's value isn't a dict where the
        default's is, we leave it alone rather than recursing into or
        replacing it -- their value, whatever shape it is, is final.
      - Nothing in `live` but absent from `default` is ever reported or
        touched -- this only ever adds, never removes a setting the user
        has (even one from an older template that this version of
        config.default.yaml no longer defines).
    """
    prefix = _prefix or []
    missing: list[tuple[list[str], object]] = []
    for key, default_value in default.items():
        path = prefix + [key]
        if key not in live:
            missing.append((path, default_value))
        elif isinstance(default_value, dict) and isinstance(live.get(key), dict):
            missing.extend(find_missing_keys(default_value, live[key], path))
        # else: `live` already has this key (as a non-dict, or `default`
        # isn't a dict here) -- it wins, untouched, no recursion.
    return missing
