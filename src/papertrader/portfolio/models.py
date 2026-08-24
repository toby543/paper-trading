"""Plain data models used across the portfolio layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    entry_date: str
    highest_close_since_entry: float

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    def market_value(self, ltp: float) -> float:
        return self.quantity * ltp

    def unrealized_pnl(self, ltp: float) -> float:
        return self.market_value(ltp) - self.cost_basis

    def unrealized_pnl_pct(self, ltp: float) -> float:
        if self.avg_price == 0:
            return 0.0
        return (ltp - self.avg_price) / self.avg_price * 100.0


@dataclass
class Trade:
    id: int | None
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    charges: float
    reason: str
    timestamp: str

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price
