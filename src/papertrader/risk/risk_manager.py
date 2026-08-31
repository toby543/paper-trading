"""Position sizing and portfolio-level risk limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    max_open_positions: int
    position_size_pct_of_equity: float
    max_cash_deployed_per_scan_pct: float
    # Cap on how many currently-open positions may share the same sector,
    # so a momentum rally concentrated in one sector (e.g. all-IT or
    # all-PSU-bank runs) can't quietly turn into an undiversified,
    # correlated book. 0/None = disabled.
    max_positions_per_sector: int | None = None

    def room_for_new_positions(self, open_position_count: int) -> int:
        return max(0, self.max_open_positions - open_position_count)

    def position_size_shares(self, equity: float, price: float) -> int:
        if price <= 0:
            return 0
        budget = equity * (self.position_size_pct_of_equity / 100.0)
        return int(budget // price)

    def scan_cash_budget(self, free_cash: float) -> float:
        return free_cash * (self.max_cash_deployed_per_scan_pct / 100.0)

    def sector_cap_reached(self, sector_counts: dict[str, int], sector: str | None) -> bool:
        """True if buying another position in `sector` would exceed
        max_positions_per_sector. An unknown sector (None -- the data
        source couldn't classify the symbol) never blocks a trade: failing
        open here matches this codebase's existing philosophy of not
        freezing the whole system over a filter whose data is unavailable."""
        if not self.max_positions_per_sector or sector is None:
            return False
        return sector_counts.get(sector, 0) >= self.max_positions_per_sector
