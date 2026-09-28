"""Plain data models used across the portfolio layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    symbol: str
    # float, not int: crypto trades in fractions (0.095 BTC is an
    # ordinary size). An equity profile still only ever puts whole
    # numbers in here -- see RiskManager.fractional_quantities.
    quantity: float
    avg_price: float
    entry_date: str
    highest_close_since_entry: float
    # Brokerage/exchange fees paid to open this position. Held separately
    # rather than folded into avg_price, because every strategy's stop and
    # target is computed against avg_price and must keep meaning "the
    # price we filled at". Defaults to 0 for ledger rows written before
    # this was tracked.
    entry_charges: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    def market_value(self, ltp: float) -> float:
        return self.quantity * ltp

    def unrealized_pnl(self, ltp: float) -> float:
        """Net of what it cost to open, so that realized + unrealized
        reconciles with (total equity - starting capital). Gross of entry
        charges, the two disagree by every fee ever paid, and always in
        the flattering direction."""
        return self.market_value(ltp) - self.cost_basis - self.entry_charges

    def unrealized_pnl_pct(self, ltp: float) -> float:
        """Deliberately the raw PRICE return, unlike unrealized_pnl: this
        is the number to read against a stop or target, which the exit
        logic evaluates on price versus avg_price alone."""
        if self.avg_price == 0:
            return 0.0
        return (ltp - self.avg_price) / self.avg_price * 100.0


@dataclass
class Trade:
    id: int | None
    symbol: str
    side: str  # BUY or SELL
    quantity: float  # fractional for crypto -- see Position.quantity
    price: float
    charges: float
    reason: str
    timestamp: str
    realized_pnl: float | None = None  # set for SELL trades only

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price
