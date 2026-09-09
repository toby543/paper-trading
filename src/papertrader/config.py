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


def _deep_merge(base: dict, override: dict) -> dict:
    """New dict with `override` layered onto `base`: a nested dict value
    (e.g. volume_confirmation, cross_sectional) is merged key-by-key
    rather than replaced wholesale, so a profile can tweak just one
    nested field without needing to repeat the whole sub-block."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

    def get_active_profile(self) -> dict[str, Any]:
        """Get the configuration for the currently active profile.

        Returns profile config from profiles section, or account section as fallback.
        """
        active_profile_name = self.get("active_profile")
        profiles = self.get("profiles", default={})

        if active_profile_name and active_profile_name in profiles:
            return profiles[active_profile_name]

        # Fallback to account.state_file if profiles not defined or profile not found
        return {}

    def set_active_profile(self, profile_name: str) -> None:
        """Set the active profile."""
        profiles = self.get("profiles", default={})
        if profile_name in profiles:
            self.raw["active_profile"] = profile_name

    def list_profiles(self) -> dict[str, str]:
        """Return dict of profile_name -> display_name for all configured profiles."""
        profiles = self.get("profiles", default={})
        result = {}
        for name, cfg in profiles.items():
            result[name] = cfg.get("display_name", name)
        return result

    def get_profile_strategy_mode(self, profile_name: str | None = None) -> str:
        """Get the strategy mode for a specific profile (or active profile if not specified)."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        profiles = self.get("profiles", default={})
        if profile_name and profile_name in profiles:
            profile_cfg = profiles[profile_name]
            return profile_cfg.get("strategy_mode", "52w_high")

        # Fallback to global strategy mode
        return self.get("strategy", "mode", default="52w_high")

    def get_profile_strategy_config(self, profile_name: str | None = None) -> dict[str, Any]:
        """Full strategy parameter set for a profile: the top-level
        `strategy:` section as a base, with that profile's own
        `profiles.<name>.strategy:` block layered on top (only the
        fields it actually overrides -- anything it doesn't set falls
        through to the shared base). This is what lets each profile
        tune its own moving averages, momentum thresholds, etc.
        independently instead of every profile being forced to share
        one global set of values. Always returns a fresh dict, safe to
        mutate -- never a live reference into cfg.raw."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        base = dict(self.get("strategy", default={}))
        profiles = self.get("profiles", default={})
        profile_cfg = (profiles.get(profile_name) or {}) if profile_name else {}
        overrides = profile_cfg.get("strategy") or {}
        merged = _deep_merge(base, overrides)
        merged["mode"] = self.get_profile_strategy_mode(profile_name)
        return merged

    def is_multi_profile_mode(self) -> bool:
        """Check if multi-profile mode is enabled (run all profiles simultaneously)."""
        return self.get("multi_profile_mode", default=False)

    def is_intelligence_enabled(self) -> bool:
        """Whether per-trade LLM rationale generation is turned on. Off by
        default -- it's a nice-to-have annotation on the trade log, not
        something any trade decision depends on, and it costs real money
        per call once enabled. See intelligence/rationale.py."""
        return bool(self.get("intelligence", "enabled", default=False))

    def get_intelligence_model(self) -> str:
        return self.get("intelligence", "model", default="claude-opus-5")

    def get_intelligence_timeout_seconds(self) -> float:
        return float(self.get("intelligence", "timeout_seconds", default=8.0))

    def get_profile_starting_capital(self, profile_name: str | None = None) -> float:
        """Get starting capital for a specific profile (or active profile if not specified)."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        profiles = self.get("profiles", default={})
        if profile_name and profile_name in profiles:
            profile_cfg = profiles[profile_name]
            return float(profile_cfg.get("starting_capital", 100000.0))

        # Fallback to global starting capital
        return float(self.get("account", "starting_capital", default=1_000_000.0))

    def get_profile_state_file(self, profile_name: str | None = None) -> str:
        """Ledger path for a specific profile (or active profile if not
        specified). Unlike the state_file property below, this always
        resolves against the requested profile_name -- needed so that in
        multi_profile_mode, where several TradingEngine instances share
        one Config object, each engine gets ITS OWN profile's ledger
        instead of whatever the config's single global active_profile
        happens to be set to."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        profiles = self.get("profiles", default={})
        if profile_name and profile_name in profiles and "state_file" in profiles[profile_name]:
            return _resolve(profiles[profile_name]["state_file"])

        # Fallback to account.state_file for backward compatibility
        return _resolve(self.get("account", "state_file", default="data/state.db"))

    @property
    def state_file(self) -> str:
        # Use profile-specific state file if profiles are configured
        profile_cfg = self.get_active_profile()
        if profile_cfg and "state_file" in profile_cfg:
            return _resolve(profile_cfg["state_file"])

        # Fallback to account.state_file for backward compatibility
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
