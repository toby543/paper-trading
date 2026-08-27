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

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # don't let ruamel line-wrap long comments

    with _lock:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.load(fh)

        for key_path, value in updates:
            node = data
            for key in key_path[:-1]:
                node = node[key]
            node[key_path[-1]] = value

        shutil.copyfile(path, path + ".bak")

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        os.replace(tmp_path, path)
