"""Position sizing and portfolio-level risk limits."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskManager:
    max_open_positions: int
    position_size_pct_of_equity: float
    max_cash_deployed_per_scan_pct: float
    # Crypto is divisible, NSE equities are not -- see
    # Config.get_profile_fractional_quantities for why whole-share
    # rounding makes the largest coins unbuyable outright.
    fractional_quantities: bool = False

    def room_for_new_positions(self, open_position_count: int) -> int:
        return max(0, self.max_open_positions - open_position_count)

    def position_size_shares(self, equity: float, price: float) -> float:
        if price <= 0:
            return 0
        budget = equity * (self.position_size_pct_of_equity / 100.0)
        if self.fractional_quantities:
            # Round DOWN to 8 decimals (the satoshi-level precision real
            # exchanges quote) rather than flooring to a whole unit, so
            # the full budget gets deployed instead of whatever a whole
            # multiple of the price happens to leave behind.
            return math.floor(budget / price * 1e8) / 1e8
        return int(budget // price)

    def scan_cash_budget(self, free_cash: float) -> float:
        return free_cash * (self.max_cash_deployed_per_scan_pct / 100.0)
