"""Which config.yaml settings the dashboard allows editing, how to
validate a submitted value for each, and the metadata used to render the
edit form. This is an explicit allowlist, not a denylist -- a new,
unrelated config key never becomes editable (and thus writable from a
browser) just by existing in config.yaml.
"""
from __future__ import annotations

import re

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

# path: dotted tuple into config.yaml. group/label/desc/unit: for the UI.
# type: bool | int | float | choice | time | str.
EDITABLE_SETTINGS: list[dict] = [
    {"path": ("strategy", "proximity_to_52w_high_pct"), "type": "float", "min": 0, "max": 50,
     "group": "Strategy — Entry", "label": "Proximity to 52W high", "unit": "%",
     "desc": "How close to its 52-week high a stock must be trading to qualify."},
    {"path": ("strategy", "min_momentum_return_pct"), "type": "float", "min": -100, "max": 500,
     "group": "Strategy — Entry", "label": "Min momentum return", "unit": "%",
     "desc": "Minimum return required over the lookback window below."},
    {"path": ("strategy", "momentum_lookback_days"), "type": "int", "min": 5, "max": 365,
     "group": "Strategy — Entry", "label": "Momentum lookback", "unit": "days",
     "desc": "Window the momentum return above is measured over."},
    {"path": ("strategy", "fast_ma_days"), "type": "int", "min": 2, "max": 100,
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

    {"path": ("execution", "slippage_bps"), "type": "float", "min": 0, "max": 1000,
     "group": "Execution realism", "label": "Slippage", "unit": "bps", "desc": ""},
    {"path": ("execution", "flat_charges_inr"), "type": "float", "min": 0, "max": 100000,
     "group": "Execution realism", "label": "Flat charges", "unit": "₹/order", "desc": ""},

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
     "desc": "Only applies the first time data/state.db is created."},
]

_BY_PATH = {tuple(entry["path"]): entry for entry in EDITABLE_SETTINGS}


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
            value = int(raw_value) if kind == "int" else float(raw_value)
        except (TypeError, ValueError):
            raise ValueError("expected a number") from None
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
