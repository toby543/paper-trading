"""Configuration loading for the paper trading system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


@dataclass
class Config:
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or os.path.join(REPO_ROOT, "config.yaml")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw=raw)

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
