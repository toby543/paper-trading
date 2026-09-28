"""Which config.yaml settings the dashboard allows editing, how to
validate a submitted value for each, and the metadata used to render the
edit form. This is an explicit allowlist, not a denylist -- a new,
unrelated config key never becomes editable (and thus writable from a
browser) just by existing in config.yaml.
"""
from __future__ import annotations

import math
import re

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

# path: dotted tuple into config.yaml. group/label/desc/unit: for the UI.
# type: bool | int | float | choice | time | str.
EDITABLE_SETTINGS: list[dict] = [
    {"path": ("universe", "file"), "type": "choice",
     "choices": ["data/universe.csv", "data/universe_nifty500.csv"],
     "group": "Universe", "label": "Universe file", "unit": "",
     "desc": "Which symbol list to scan: data/universe.csv is a smaller ~100-symbol starter "
             "list, data/universe_nifty500.csv is the full cleaned Nifty 500. Loaded once at "
             "startup -- restart required for a change to take effect. Shared by every "
             "profile, not specific to any one of them."},
    {"path": ("strategy", "proximity_to_52w_high_pct"), "type": "float", "min": 0, "max": 50,
     "group": "Strategy — Entry", "label": "Proximity to 52W high", "unit": "%",
     "desc": "How close to its 52-week high a stock must be trading to qualify."},
    {"path": ("strategy", "min_momentum_return_pct"), "type": "float", "min": -100, "max": 500,
     "group": "Strategy — Entry", "label": "Min momentum return", "unit": "%",
     "desc": "Minimum return required over the lookback window below."},
    {"path": ("strategy", "momentum_lookback_days"), "type": "int", "min": 5, "max": 365,
     "group": "Strategy — Entry", "label": "Momentum lookback", "unit": "days",
     "desc": "Window the momentum return above is measured over."},
    {"path": ("strategy", "fast_ma_days"), "type": "int", "min": 2, "max": 400,
     "group": "Strategy — Entry", "label": "Fast moving average", "unit": "days",
     "desc": "Shorter trend-confirmation average; also used for the momentum-breakdown exit."},
    {"path": ("strategy", "slow_ma_days"), "type": "int", "min": 10, "max": 400,
     "group": "Strategy — Entry", "label": "Slow moving average", "unit": "days",
     "desc": "Longer trend-confirmation average."},
    {"path": ("strategy", "min_avg_daily_turnover_inr"), "type": "float", "min": 0,
     "group": "Strategy — Entry", "label": "Min liquidity", "unit": "₹/day",
     "desc": "Minimum average daily traded value over the last 20 sessions."},
    {"path": ("strategy", "min_relative_strength_pct"), "type": "float", "min": -100, "max": 500,
     "group": "Strategy — Entry", "label": "Min relative strength", "unit": "pp vs index",
     "desc": "How much a stock must beat the benchmark index's own return by."},
    {"path": ("strategy", "volume_confirmation", "enabled"), "type": "bool",
     "group": "Strategy — Entry", "label": "Volume confirmation", "unit": "",
     "desc": "Require recent volume to be running hot vs. its own baseline before buying."},
    {"path": ("strategy", "volume_confirmation", "min_volume_multiple"), "type": "float", "min": 0, "max": 50,
     "group": "Strategy — Entry", "label": "Min volume multiple", "unit": "×",
     "desc": "Recent average volume must be at least this multiple of the baseline."},
    {"path": ("strategy", "volume_confirmation", "recent_days"), "type": "int", "min": 1, "max": 250,
     "group": "Strategy — Entry", "label": "Volume: recent window", "unit": "days", "desc": ""},
    {"path": ("strategy", "volume_confirmation", "baseline_days"), "type": "int", "min": 2, "max": 400,
     "group": "Strategy — Entry", "label": "Volume: baseline window", "unit": "days", "desc": ""},
    {"path": ("strategy", "max_new_positions_per_scan"), "type": "int", "min": 1, "max": 50,
     "group": "Strategy — Entry", "label": "New positions per scan", "unit": "",
     "desc": "Caps how many top-ranked candidates get bought in a single scan."},
    {"path": ("strategy", "min_ltp_inr"), "type": "float", "min": 0,
     "group": "Strategy — Entry", "label": "Min price", "unit": "₹ (0 = off)",
     "desc": "Skip stocks trading below this price, e.g. to avoid penny stocks."},
    {"path": ("strategy", "max_ltp_inr"), "type": "float", "min": 0,
     "group": "Strategy — Entry", "label": "Max price", "unit": "₹ (0 = off)",
     "desc": "Skip stocks trading above this price."},

    {"path": ("strategy", "cross_sectional", "lookback_days"), "type": "int", "min": 5, "max": 500,
     "group": "Cross-sectional momentum", "label": "Lookback window", "unit": "days",
     "desc": "Trailing-return window used to rank the universe. Only applies when strategy mode is cross_sectional_momentum."},
    {"path": ("strategy", "cross_sectional", "skip_recent_days"), "type": "int", "min": 0, "max": 90,
     "group": "Cross-sectional momentum", "label": "Skip recent days", "unit": "days",
     "desc": "\"12-1\" style: ends the lookback this many days before today, to avoid short-term reversal noise. 0 disables the skip."},
    {"path": ("strategy", "cross_sectional", "top_pct"), "type": "float", "min": 1, "max": 100,
     "group": "Cross-sectional momentum", "label": "Top percentile", "unit": "% of universe",
     "desc": "Buy only stocks ranked in this top percentile by trailing return."},

    {"path": ("strategy", "consolidation_breakout", "consolidation_days"), "type": "int", "min": 2, "max": 50,
     "group": "Consolidation breakout", "label": "Consolidation period", "unit": "days",
     "desc": "How many days the stock must hold a tight consolidation before breaking out. Only applies when strategy mode is consolidation_breakout."},
    {"path": ("strategy", "consolidation_breakout", "max_consolidation_range_pct"), "type": "float", "min": 0.5, "max": 100.0,
     "group": "Consolidation breakout", "label": "Max base range", "unit": "%",
     "desc": "How wide (high-to-low as % of average close) the base is allowed to be and still count as a tight consolidation. Only applies when strategy mode is consolidation_breakout."},
    {"path": ("strategy", "consolidation_breakout", "volume_multiple"), "type": "float", "min": 1.0, "max": 10.0,
     "group": "Consolidation breakout", "label": "Breakout volume", "unit": "× baseline",
     "desc": "Breakout must occur on volume at least this multiple of the 20-day average. Only applies when strategy mode is consolidation_breakout."},

    {"path": ("strategy", "ipo_base_breakout", "min_listing_days"), "type": "int", "min": 5, "max": 500,
     "group": "IPO Base Breakout", "label": "Min listing age", "unit": "trading days",
     "desc": "The stock must have at least this many days of trading history -- enough to have formed a real base. Only applies when strategy mode is ipo_base_breakout."},
    {"path": ("strategy", "ipo_base_breakout", "max_listing_days"), "type": "int", "min": 60, "max": 1500,
     "group": "IPO Base Breakout", "label": "Max listing age", "unit": "trading days",
     "desc": "Beyond this many days of history the stock no longer counts as a recent IPO -- the no-overhead-resistance advantage this strategy trades on is gone. Only applies when strategy mode is ipo_base_breakout."},
    {"path": ("strategy", "ipo_base_breakout", "base_days"), "type": "int", "min": 2, "max": 50,
     "group": "IPO Base Breakout", "label": "Base period", "unit": "days",
     "desc": "How many days the stock must hold a tight consolidation before breaking out. Only applies when strategy mode is ipo_base_breakout."},
    {"path": ("strategy", "ipo_base_breakout", "max_base_range_pct"), "type": "float", "min": 0.5, "max": 100.0,
     "group": "IPO Base Breakout", "label": "Max base range", "unit": "%",
     "desc": "How wide (high-to-low as % of average close) the base is allowed to be and still count as tight. Wider than an established stock's, since a recent listing naturally swings more even while consolidating. Only applies when strategy mode is ipo_base_breakout."},
    {"path": ("strategy", "ipo_base_breakout", "volume_multiple"), "type": "float", "min": 1.0, "max": 10.0,
     "group": "IPO Base Breakout", "label": "Breakout volume", "unit": "× baseline",
     "desc": "Breakout must occur on volume at least this multiple of the 20-day average. Only applies when strategy mode is ipo_base_breakout."},

    {"path": ("strategy", "pivot_supertrend", "atr_period"), "type": "int", "min": 2, "max": 50,
     "group": "Pivot point + SuperTrend", "label": "ATR period", "unit": "days",
     "desc": "Wilder-smoothed ATR window the SuperTrend bands are built on. Only applies when strategy mode is pivot_supertrend."},
    {"path": ("strategy", "pivot_supertrend", "supertrend_multiplier"), "type": "float", "min": 0.5, "max": 10.0,
     "group": "Pivot point + SuperTrend", "label": "SuperTrend multiplier", "unit": "× ATR",
     "desc": "Band width either side of the midpoint. Wider means fewer, more decisive flips (less whipsaw, later entries)."},
    {"path": ("strategy", "pivot_supertrend", "min_pct_above_r1"), "type": "float", "min": 0, "max": 50,
     "group": "Pivot point + SuperTrend", "label": "Min % above R1", "unit": "%",
     "desc": "How far above the prior day's R1 (first resistance) a SuperTrend flip must occur to qualify."},

    {"path": ("strategy", "trend_pullback", "pullback_lookback_days"), "type": "int", "min": 3, "max": 100,
     "group": "Trend Pullback", "label": "Pullback lookback", "unit": "days",
     "desc": "Window used to find the recent high a pullback is measured from. Only applies when strategy mode is trend_pullback."},
    {"path": ("strategy", "trend_pullback", "min_pullback_pct"), "type": "float", "min": 0, "max": 50,
     "group": "Trend Pullback", "label": "Min pullback depth", "unit": "%",
     "desc": "Today's close must sit at least this far below the recent high to count as a real pullback, not noise."},
    {"path": ("strategy", "trend_pullback", "max_pullback_pct"), "type": "float", "min": 0.5, "max": 90,
     "group": "Trend Pullback", "label": "Max pullback depth", "unit": "%",
     "desc": "Deeper than this and the uptrend may already be broken rather than just pulling back."},

    {"path": ("regime", "enabled"), "type": "bool",
     "group": "Market regime", "label": "Regime filter", "unit": "",
     "desc": "Block new entries while the benchmark index is below its own moving average."},
    {"path": ("regime", "index_symbol"), "type": "str", "max_len": 20,
     "group": "Market regime", "label": "Benchmark index", "unit": "",
     "desc": "Yahoo Finance ticker for the regime filter and relative-strength "
             "benchmark. Pick a suggestion or type any other ticker -- if it "
             "can't be fetched, both checks fail open (skip themselves) rather "
             "than blocking trading.",
     "suggestions": [
         ("^NSEI", "Nifty 50"),
         ("^NSEBANK", "Nifty Bank"),
         ("^CNXIT", "Nifty IT"),
         ("^CNX100", "Nifty 100"),
         ("^CNX200", "Nifty 200"),
         ("^CRSLDX", "Nifty 500"),
         ("^NSMIDCP", "Nifty Midcap 100"),
         ("^CNXFMCG", "Nifty FMCG"),
         ("^CNXPHARMA", "Nifty Pharma"),
         ("^CNXAUTO", "Nifty Auto"),
         ("^CNXMETAL", "Nifty Metal"),
         ("^CNXREALTY", "Nifty Realty"),
         ("^CNXENERGY", "Nifty Energy"),
         ("^CNXPSE", "Nifty PSE"),
         ("^CNXINFRA", "Nifty Infrastructure"),
     ]},
    {"path": ("regime", "ma_days"), "type": "int", "min": 5, "max": 400,
     "group": "Market regime", "label": "Regime moving average", "unit": "days", "desc": ""},

    {"path": ("risk", "stop_loss_pct"), "type": "float", "min": 0.1, "max": 90,
     "group": "Risk — Exit & sizing", "label": "Stop loss", "unit": "% from entry", "desc": ""},
    {"path": ("risk", "trailing_stop_pct"), "type": "float", "min": 0.1, "max": 90,
     "group": "Risk — Exit & sizing", "label": "Trailing stop", "unit": "% from peak", "desc": ""},
    {"path": ("risk", "take_profit_pct"), "type": "float", "min": 0, "max": 1000,
     "group": "Risk — Exit & sizing", "label": "Take profit", "unit": "% above entry (0 = off)",
     "desc": "Optional hard sell target. 0 disables it and lets the trailing stop manage exits instead."},
    {"path": ("risk", "exit_below_fast_ma"), "type": "bool",
     "group": "Risk — Exit & sizing", "label": "Momentum-breakdown exit", "unit": "",
     "desc": "Sell if price closes below the fast moving average."},
    {"path": ("risk", "position_size_pct_of_equity"), "type": "float", "min": 0.1, "max": 100,
     "group": "Risk — Exit & sizing", "label": "Position size", "unit": "% of equity", "desc": ""},
    {"path": ("risk", "max_open_positions"), "type": "int", "min": 1, "max": 200,
     "group": "Risk — Exit & sizing", "label": "Max open positions", "unit": "", "desc": ""},
    {"path": ("risk", "max_cash_deployed_per_scan_pct"), "type": "float", "min": 1, "max": 100,
     "group": "Risk — Exit & sizing", "label": "Max cash deployed / scan", "unit": "% of free cash", "desc": ""},
    {"path": ("risk", "reentry_cooldown_days"), "type": "int", "min": 0, "max": 60,
     "group": "Risk — Exit & sizing", "label": "Re-entry cooldown", "unit": "days (0 = off)",
     "desc": "Blocks buying a symbol again for this many days after selling it -- prevents a "
             "stock stopped out on a dip from being immediately rebought the same scan."},

    {"path": ("execution", "slippage_bps"), "type": "float", "min": 0, "max": 1000,
     "group": "Execution realism", "label": "Slippage", "unit": "bps", "desc": ""},
    {"path": ("execution", "flat_charges_inr"), "type": "float", "min": 0, "max": 100000,
     "group": "Execution realism", "label": "Flat charges", "unit": "₹/order",
     "desc": "Fixed brokerage + STT charged per order, the way an Indian equity "
             "broker prices. Crypto profiles set this to 0 and use the percentage "
             "fee below instead."},
    {"path": ("execution", "fee_pct"), "type": "float", "min": 0, "max": 5,
     "group": "Execution realism", "label": "Exchange fee", "unit": "% of trade value",
     "desc": "Percentage of the traded amount charged on every fill, the way a "
             "crypto exchange prices (~0.1% taker). Scales with position size, "
             "unlike the flat charge above. Equity profiles leave this at 0."},

    {"path": ("engine", "market_open"), "type": "time",
     "group": "Market hours & scheduling", "label": "Market open", "unit": "HH:MM", "desc": ""},
    {"path": ("engine", "market_close"), "type": "time",
     "group": "Market hours & scheduling", "label": "Market close", "unit": "HH:MM", "desc": ""},
    {"path": ("engine", "scan_interval_minutes"), "type": "int", "min": 1, "max": 1440,
     "group": "Market hours & scheduling", "label": "Entry scan interval", "unit": "min", "desc": ""},
    {"path": ("engine", "exit_check_interval_minutes"), "type": "int", "min": 1, "max": 1440,
     "group": "Market hours & scheduling", "label": "Exit check interval", "unit": "min", "desc": ""},

    {"path": ("data_source", "preferred"), "type": "choice", "choices": ["nse", "yfinance"],
     "group": "Data source", "label": "Preferred source", "unit": "", "desc": ""},
    {"path": ("data_source", "fallback"), "type": "choice", "choices": ["nse", "yfinance"],
     "group": "Data source", "label": "Fallback source", "unit": "", "desc": ""},
    {"path": ("data_source", "request_timeout_seconds"), "type": "int", "min": 1, "max": 120,
     "group": "Data source", "label": "Request timeout", "unit": "sec", "desc": ""},
    {"path": ("data_source", "max_retries"), "type": "int", "min": 0, "max": 10,
     "group": "Data source", "label": "Max retries", "unit": "", "desc": ""},

    {"path": ("logging", "level"), "type": "choice", "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
     "group": "Logging", "label": "Log level", "unit": "", "desc": ""},
    {"path": ("account", "starting_capital"), "type": "float", "min": 0,
     "group": "Account", "label": "Starting capital", "unit": "₹",
     "desc": "This profile's own starting capital. Only applies the first time its ledger "
             "database is created -- has no effect on a profile that's already traded."},
]

_BY_PATH = {tuple(entry["path"]): entry for entry in EDITABLE_SETTINGS}

# Which strategy_mode a strategy.<subsection>.* field's settings actually
# govern. Everything else (universe, the shared strategy.* entry filters,
# regime, risk, execution, engine, data_source, logging, account) applies
# to every mode, so isn't listed here.
_STRATEGY_SUBSECTION_MODE = {
    "cross_sectional": "cross_sectional_momentum",
    "consolidation_breakout": "consolidation_breakout",
    "ipo_base_breakout": "ipo_base_breakout",
    "pivot_supertrend": "pivot_supertrend",
    "trend_pullback": "trend_pullback",
}

# Shared strategy.*/risk.* fields that only SOME strategies read, mapped
# to the modes that actually read them.
#
# The shared "Strategy — Entry" block was assumed universal, but that is
# only true of the 52-week-high strategy it grew out of. Anything absent
# from this map is genuinely universal and always shown.
# Every strategy mode a profile can run. Lets the risk-section entries
# below be written as "all modes except ...", which stays correct when a
# mode is added rather than silently omitting it.
_ALL_MODES = frozenset({
    "52w_high", "cross_sectional_momentum", "consolidation_breakout", "pivot_supertrend",
    "trend_pullback", "ipo_base_breakout",
})

_FIELD_MODES = {
    ("strategy", "proximity_to_52w_high_pct"): {"52w_high"},
    ("strategy", "min_relative_strength_pct"): {"52w_high"},
    ("strategy", "volume_confirmation", "enabled"): {"52w_high"},
    ("strategy", "volume_confirmation", "min_volume_multiple"): {"52w_high"},
    ("strategy", "volume_confirmation", "recent_days"): {"52w_high"},
    ("strategy", "volume_confirmation", "baseline_days"): {"52w_high"},
    ("strategy", "min_ltp_inr"): {
        "52w_high", "cross_sectional_momentum",
        "pivot_supertrend", "trend_pullback", "ipo_base_breakout",
    },
    ("strategy", "max_ltp_inr"): {
        "52w_high", "cross_sectional_momentum",
        "pivot_supertrend", "trend_pullback", "ipo_base_breakout",
    },
    ("risk", "exit_below_fast_ma"): {"52w_high", "consolidation_breakout", "ipo_base_breakout"},
    # pivot_supertrend, trend_pullback and cross_sectional_momentum are
    # deliberately absent: none of the three mentions either key anywhere
    # (cross_sectional has its own strategy.cross_sectional.lookback_days;
    # trend_pullback measures its own pullback window). Listing them here
    # put four editable fields on those profiles' panels that changed
    # nothing at all when saved.
    ("strategy", "min_momentum_return_pct"): {
        "52w_high", "consolidation_breakout", "ipo_base_breakout",
    },
    ("strategy", "momentum_lookback_days"): {
        "52w_high", "consolidation_breakout", "ipo_base_breakout",
    },
    # pivot_supertrend reads neither: its trend is the SuperTrend band, not
    # a moving-average pair, and it mentions neither key.
    ("strategy", "fast_ma_days"): {
        "52w_high", "cross_sectional_momentum", "trend_pullback",
        "consolidation_breakout", "ipo_base_breakout",
    },
    ("strategy", "slow_ma_days"): {
        "52w_high", "cross_sectional_momentum", "trend_pullback",
        "consolidation_breakout", "ipo_base_breakout",
    },
}


# Global settings a crypto profile never reads, so the Edit Settings
# panel must not offer them while one is being viewed.
#
# These are keyed off the profile being crypto rather than off its
# strategy mode, because mode cannot distinguish them: crypto_breakout
# and the NSE "Consolidation Breakout" profile run the SAME
# consolidation_breakout mode, and these settings apply to one and not
# the other. Each is genuinely dead for crypto rather than merely
# unusual:
#   - universe.file          crypto profiles set their own universe_file
#   - engine.market_open/close  crypto trades 24/7; the NSE session gate
#                               is skipped entirely (see trades_24_7)
#   - data_source.preferred/fallback  crypto symbols route to
#                               Binance/Kraken and never touch the
#                               NSE/Yahoo pair
# Everything else in those groups -- scan/exit intervals, the request
# timeout, the log level -- does govern a crypto profile and stays.
CRYPTO_INERT_PATHS = {
    ("universe", "file"),
    ("engine", "market_open"),
    ("engine", "market_close"),
    ("data_source", "preferred"),
    ("data_source", "fallback"),
}


def applies_to_profile(path: tuple[str, ...], mode: str, is_crypto: bool) -> bool:
    """Whether this field governs the profile currently being viewed --
    applies_to_mode plus the crypto-inert globals above."""
    if is_crypto and tuple(path) in CRYPTO_INERT_PATHS:
        return False
    return applies_to_mode(path, mode)


def applies_to_mode(path: tuple[str, ...], mode: str) -> bool:
    """False for a field the active strategy does not actually read --
    either because it belongs to a *different* strategy's own subsection
    (ATR period means nothing outside pivot_supertrend) or because it is
    one of the shared fields only some strategies consult (see
    _FIELD_MODES). Used to filter the Edit Settings panel down to what
    the active profile's strategy genuinely governs, so every field shown
    is one that changes its behaviour."""
    if len(path) >= 2 and path[0] == "strategy" and path[1] in _STRATEGY_SUBSECTION_MODE:
        return _STRATEGY_SUBSECTION_MODE[path[1]] == mode
    modes = _FIELD_MODES.get(tuple(path))
    if modes is not None:
        return mode in modes
    return True


def get_value(raw_cfg: dict, path: tuple[str, ...]):
    node = raw_cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def coerce_and_validate(path: tuple[str, ...], raw_value):
    """Return the properly-typed value for `path`, or raise ValueError."""
    spec = _BY_PATH.get(tuple(path))
    if spec is None:
        raise ValueError(f"'{'.'.join(path)}' is not an editable setting")

    kind = spec["type"]
    if kind == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str) and raw_value.lower() in ("true", "false"):
            return raw_value.lower() == "true"
        raise ValueError("expected true/false")

    if kind in ("int", "float"):
        try:
            if kind == "int":
                # via float first, so a fractional value is REJECTED rather
                # than silently truncated: int(3.99) is 3, and a user who
                # typed 3.99 for a day count got 3 saved with no warning.
                as_float = float(raw_value)
                if not as_float.is_integer():
                    raise ValueError("expected a whole number")
                value = int(as_float)
            else:
                value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc) if "whole number" in str(exc) else "expected a number") from None
        # Explicitly, before the bounds below: every comparison against NaN
        # is False, so "nan" would satisfy both of them and be persisted to
        # the profile. It then either kills that profile's scan thread on
        # every cycle (int(nan // price) raises) or, on a stop-loss, makes
        # the stop silently unreachable for the same reason.
        if not math.isfinite(value):
            raise ValueError("expected a finite number")
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"must be >= {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"must be <= {spec['max']}")
        return value

    if kind == "choice":
        value = str(raw_value)
        if value not in spec["choices"]:
            raise ValueError(f"must be one of {spec['choices']}")
        return value

    if kind == "time":
        value = str(raw_value)
        if not _TIME_RE.match(value):
            raise ValueError("expected 24-hour HH:MM")
        return value

    if kind == "str":
        value = str(raw_value)
        if "max_len" in spec and len(value) > spec["max_len"]:
            raise ValueError(f"too long (max {spec['max_len']} characters)")
        return value

    raise ValueError(f"unsupported type {kind!r}")  # pragma: no cover - schema bug, not user input
