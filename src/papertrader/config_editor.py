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

            # Apply updates
            for key_path, value in updates:
                node = data
                for key in key_path[:-1]:
                    node = node[key]
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
