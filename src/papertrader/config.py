"""Configuration loading for the paper trading system."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _bootstrap_from_default(path: str) -> None:
    """First-run convenience: if `path` doesn't exist yet but a sibling
    config.default.yaml does, copy it over before loading.

    config.yaml is gitignored on purpose (see .gitignore and
    config.default.yaml's own header comment) so dashboard/hand edits are
    never at risk from a `git pull` -- but that means a completely fresh
    clone has no config.yaml at all, and `python main.py ...` should still
    work immediately rather than failing with a bare FileNotFoundError
    and no indication of what to do about it. If no default exists either
    (a custom --config path pointing somewhere else entirely, or someone
    removed config.default.yaml too), do nothing and let the normal
    FileNotFoundError from the caller's own `open()` explain the problem
    -- inventing a file here would be worse than a clear error.
    """
    if os.path.exists(path):
        return
    default_path = os.path.join(os.path.dirname(path) or ".", "config.default.yaml")
    if not os.path.exists(default_path):
        return
    shutil.copyfile(default_path, path)
    log.info("No %s found -- created it from %s (first run). Edit it, or use the "
             "dashboard's Edit Settings panel, freely: this file is yours now and "
             "`git pull` will never touch it again.", path, default_path)


def _profiles_dir_mtime(profiles_dir: str) -> float:
    """Latest mtime across every file in profiles_dir, or 0.0 if the
    directory doesn't exist -- folded into Config's own _last_modified so
    reload()/has_changed() notice a profile file edited directly (or by
    the dashboard's per-profile write) exactly like a config.yaml edit."""
    if not os.path.isdir(profiles_dir):
        return 0.0
    latest = 0.0
    for fname in os.listdir(profiles_dir):
        if fname.endswith((".yaml", ".yml")):
            latest = max(latest, os.path.getmtime(os.path.join(profiles_dir, fname)))
    return latest


def _load_profiles_dir(profiles_dir: str) -> dict[str, Any]:
    """Every profiles_dir/<name>.yaml, keyed by filename stem -- shaped
    exactly like the old inline `profiles:` dict so every existing
    consumer (get_profile_strategy_config, the scheduler, the dashboard,
    ...) keeps working unchanged against cfg.raw["profiles"]."""
    profiles: dict[str, Any] = {}
    if not os.path.isdir(profiles_dir):
        return profiles
    for fname in sorted(os.listdir(profiles_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(profiles_dir, fname), "r", encoding="utf-8") as fh:
            profiles[name] = yaml.safe_load(fh) or {}
    return profiles


def _ensure_profiles_dir(config_path: str, profiles_dir: str) -> None:
    """Make sure profiles_dir exists and holds one file per profile,
    migrating from wherever profile data currently lives:

      1. If config_path still has an inline `profiles:` block -- either
         the old single-file layout, or a config.yaml freshly
         bootstrapped from config.default.yaml (which still has one) --
         split it out into profiles_dir/<name>.yaml, preserving whatever
         is actually there (a user's live, possibly customized values,
         not just template defaults), then strip `profiles:` back out of
         config_path. Uses ruamel's round-trip mode so config_path's own
         comments and each new per-profile file's comments both survive.
      2. Otherwise (config_path has no profiles: key and profiles_dir
         doesn't exist -- e.g. it was deleted by hand), seed profiles_dir
         from config.default.yaml's own `profiles:` block instead, the
         same first-run convenience _bootstrap_from_default provides for
         config.yaml itself.

    A no-op once profiles_dir already exists -- never re-splits or
    touches a file that's already there, so this is safe to call on
    every Config.load()."""
    if os.path.isdir(profiles_dir):
        return

    from ruamel.yaml import YAML
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.width = 4096

    data = None
    profiles = None
    source_desc = ""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml_rt.load(fh)
        if data and data.get("profiles"):
            profiles = data["profiles"]
            source_desc = config_path

    if profiles is None:
        default_path = os.path.join(os.path.dirname(config_path) or ".", "config.default.yaml")
        if not os.path.exists(default_path):
            return
        with open(default_path, "r", encoding="utf-8") as fh:
            default_data = yaml_rt.load(fh)
        profiles = (default_data or {}).get("profiles") or {}
        source_desc = default_path
        data = None  # nothing to strip out of config_path in this branch

    if not profiles:
        return

    os.makedirs(profiles_dir, exist_ok=True)
    for name, profile_data in list(profiles.items()):
        profile_path = os.path.join(profiles_dir, f"{name}.yaml")
        with open(profile_path, "w", encoding="utf-8") as fh:
            yaml_rt.dump(profile_data, fh)
    log.info("Populated %s from %s's profiles (%d profile file(s) created).",
              profiles_dir, source_desc, len(profiles))

    if data is not None and "profiles" in data:
        del data["profiles"]
        tmp_path = config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml_rt.dump(data, fh)
        os.replace(tmp_path, config_path)
        log.info("Removed inline profiles: block from %s (now split into %s/*.yaml).",
                  config_path, profiles_dir)


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
    _profiles_dir: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if self.path and os.path.exists(self.path):
            self._last_modified = os.path.getmtime(self.path)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or os.path.join(REPO_ROOT, "config.yaml")
        _bootstrap_from_default(path)
        # Each profile lives in its own profiles/<name>.yaml now, not an
        # inline `profiles:` block in config.yaml -- see _ensure_profiles_dir's
        # docstring. Merged into raw["profiles"] below so every existing
        # consumer keeps working against the same shape as before.
        profiles_dir = os.path.join(os.path.dirname(path) or ".", "profiles")
        _ensure_profiles_dir(path, profiles_dir)
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["profiles"] = _load_profiles_dir(profiles_dir)
        cfg = cls(raw=raw, path=path)
        cfg._profiles_dir = profiles_dir
        cfg._last_modified = max(cfg._last_modified, _profiles_dir_mtime(profiles_dir))
        return cfg

    def profile_file_path(self, profile_name: str) -> str:
        """Path to profile_name's own config file under profiles/ -- where
        Edit Settings' profile-scoped writes (strategy.*, risk.*,
        starting_capital) actually land now, instead of inside config.yaml's
        old inline profiles.<name>.* block."""
        return os.path.join(self._profiles_dir, f"{profile_name}.yaml")

    def reload(self) -> bool:
        """Reload config from file (and every profiles/*.yaml) if either
        has changed. Returns True if reloaded."""
        if not self.path or not os.path.exists(self.path):
            return False

        current_mtime = max(os.path.getmtime(self.path), _profiles_dir_mtime(self._profiles_dir))
        if current_mtime <= self._last_modified:
            return False

        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                new_raw = yaml.safe_load(fh)
            new_raw["profiles"] = _load_profiles_dir(self._profiles_dir)
            self.raw = new_raw
            self._last_modified = current_mtime
            log.info("Configuration reloaded successfully from %s (+ %s/*.yaml)", self.path, self._profiles_dir)
            return True
        except Exception as e:
            log.error("Failed to reload configuration: %s", e)
            return False

    def has_changed(self) -> bool:
        """Check if config.yaml or any profiles/*.yaml file has been
        modified without reloading."""
        if not self.path or not os.path.exists(self.path):
            return False
        current_mtime = max(os.path.getmtime(self.path), _profiles_dir_mtime(self._profiles_dir))
        return current_mtime > self._last_modified

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

    def list_profile_categories(self) -> dict[str, str]:
        """Return dict of profile_name -> category ("swing" or "long_term")
        for all configured profiles, used to filter the dashboard's tabs.
        A profile without an explicit `category` falls back to "swing" --
        every profile that existed before the long-term tab was added is
        a swing strategy, so this keeps them showing up correctly without
        requiring every existing config.yaml to be touched."""
        profiles = self.get("profiles", default={})
        return {name: cfg.get("category", "swing") for name, cfg in profiles.items()}

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

    def get_profile_risk_config(self, profile_name: str | None = None) -> dict[str, Any]:
        """Full risk parameter set for a profile: the top-level `risk:`
        section as a base, with that profile's own `profiles.<name>.risk:`
        block layered on top -- same pattern as get_profile_strategy_config.
        Lets a long-horizon profile run much wider stops (it needs to
        survive an ordinary pullback within a multi-year uptrend without
        being stopped out early) than a swing profile, instead of every
        profile being forced to share one global stop-loss/trailing-stop.
        Always returns a fresh dict, safe to mutate."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        base = dict(self.get("risk", default={}))
        profiles = self.get("profiles", default={})
        profile_cfg = (profiles.get(profile_name) or {}) if profile_name else {}
        overrides = profile_cfg.get("risk") or {}
        return _deep_merge(base, overrides)

    def get_profile_regime_config(self, profile_name: str | None = None) -> dict[str, Any]:
        """Full regime-filter parameter set for a profile: the top-level
        `regime:` section as a base, with that profile's own
        `profiles.<name>.regime:` block layered on top -- same pattern as
        get_profile_strategy_config/get_profile_risk_config. Lets a crypto
        profile disable the Nifty 50 uptrend filter entirely (crypto trades
        24/7 with its own cycles, uncorrelated with NSE) without touching
        the global regime block that equity profiles still rely on.
        Always returns a fresh dict, safe to mutate."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        base = dict(self.get("regime", default={}))
        profiles = self.get("profiles", default={})
        profile_cfg = (profiles.get(profile_name) or {}) if profile_name else {}
        overrides = profile_cfg.get("regime") or {}
        return _deep_merge(base, overrides)

    def is_multi_profile_mode(self) -> bool:
        """Check if multi-profile mode is enabled (run all profiles simultaneously)."""
        return self.get("multi_profile_mode", default=False)

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

    def get_profile_config(self, profile_name: str | None = None) -> dict:
        """Get the full profile configuration dict for a specific profile.
        Returns {} if profile not found."""
        if profile_name is None:
            profile_name = self.get("active_profile")

        profiles = self.get("profiles", default={})
        return profiles.get(profile_name, {}) if profile_name else {}

    def get_profile_universe_file(self, profile_name: str | None = None) -> str:
        """Universe file path for a specific profile (or active profile if
        not specified). Lets a profile scan its own symbol list -- e.g. a
        crypto profile's `universe_file: data/universe_crypto.csv` instead
        of the shared NSE universe every equity profile scans -- falling
        back to the global `universe.file` setting when the profile
        doesn't override it."""
        profile_cfg = self.get_profile_config(profile_name)
        return profile_cfg.get("universe_file") or self.universe_file

    def get_profile_trades_24_7(self, profile_name: str | None = None) -> bool:
        """True if this profile trades around the clock (crypto) and must
        never be gated by the NSE calendar's open/close hours, weekends, or
        holidays. Derived from the profile's `category` field."""
        profile_cfg = self.get_profile_config(profile_name)
        return profile_cfg.get("category") == "crypto"

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
