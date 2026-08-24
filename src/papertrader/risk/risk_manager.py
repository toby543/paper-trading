"""Position sizing and portfolio-level risk limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    max_open_positions: int
    position_size_pct_of_equity: float
    max_cash_deployed_per_scan_pct: float

    def room_for_new_positions(self, open_position_count: int) -> int:
        return max(0, self.max_open_positions - open_position_count)

    def position_size_shares(self, equity: float, price: float) -> int:
        if price <= 0:
            return 0
        budget = equity * (self.position_size_pct_of_equity / 100.0)
        return int(budget // price)

    def scan_cash_budget(self, free_cash: float) -> float:
        return free_cash * (self.max_cash_deployed_per_scan_pct / 100.0)
