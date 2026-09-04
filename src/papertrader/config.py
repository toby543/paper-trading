"""Configuration loading for the paper trading system."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


@dataclass
class Config:
    raw: dict[str, Any] = field(repr=False)
    path: str = field(default="")
    _last_modified: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.path and os.path.exists(self.path):
            self._last_modified = os.path.getmtime(self.path)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or os.path.join(REPO_ROOT, "config.yaml")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw=raw, path=path)

    def reload(self) -> bool:
        """Reload config from file if it has changed. Returns True if reloaded."""
        if not self.path or not os.path.exists(self.path):
            return False

        current_mtime = os.path.getmtime(self.path)
        if current_mtime <= self._last_modified:
            return False

        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                new_raw = yaml.safe_load(fh)
            self.raw = new_raw
            self._last_modified = current_mtime
            log.info("Configuration reloaded successfully from %s", self.path)
            return True
        except Exception as e:
            log.error("Failed to reload configuration: %s", e)
            return False

    def has_changed(self) -> bool:
        """Check if config file has been modified without reloading."""
        if not self.path or not os.path.exists(self.path):
            return False
        return os.path.getmtime(self.path) > self._last_modified

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def state_file(self) -> str:
        return _resolve(self.get("account", "state_file", default="data/state.db"))

    @property
    def universe_file(self) -> str:
        return _resolve(self.get("universe", "file", default="data/universe.csv"))

    @property
    def holidays_file(self) -> str:
        return _resolve(self.get("engine", "holidays_file", default="data/nse_holidays.csv"))

    @property
    def log_file(self) -> str:
        return _resolve(self.get("logging", "file", default="data/papertrader.log"))
