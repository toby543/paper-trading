"""The autonomous trading loop.

During NSE market hours it periodically:
  1. Checks open positions for stop-loss / trailing-stop / momentum-breakdown
     exits and sells (paper) any that trigger.
  2. Scans the configured universe for new 52-week-high momentum
     candidates and buys (paper) the top-ranked ones, sized by the risk
     manager, subject to max open positions and per-scan cash limits.
Outside market hours it sleeps until the next open.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from ..config import Config
from ..data.nse_client import MarketDataClient, DataUnavailableError
from ..data.universe import load_universe
from ..portfolio.broker import PaperBroker, InsufficientFundsError
from ..portfolio.storage import Storage
from ..risk.risk_manager import RiskManager
from ..strategy.momentum_52w_high import evaluate_candidate, rank_candidates, check_exit
from .market_hours import MarketCalendar

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.calendar = MarketCalendar(
            timezone=cfg.get("engine", "timezone", default="Asia/Kolkata"),
            market_open=cfg.get("engine", "market_open", default="09:15"),
            market_close=cfg.get("engine", "market_close", default="15:30"),
            holidays_file=cfg.holidays_file,
        )
        self.storage = Storage(cfg.state_file, cfg.get("account", "starting_capital", default=1_000_000.0))
        self.broker = PaperBroker(
            self.storage,
            slippage_bps=cfg.get("execution", "slippage_bps", default=5.0),
            flat_charges_inr=cfg.get("execution", "flat_charges_inr", default=20.0),
        )
        self.risk = RiskManager(
            max_open_positions=cfg.get("risk", "max_open_positions", default=10),
            position_size_pct_of_equity=cfg.get("risk", "position_size_pct_of_equity", default=8.0),
            max_cash_deployed_per_scan_pct=cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0),
        )
        self.data = MarketDataClient(
            preferred=cfg.get("data_source", "preferred", default="nse"),
            fallback=cfg.get("data_source", "fallback", default="yfinance"),
            timeout=cfg.get("data_source", "request_timeout_seconds", default=10),
        )
        self.universe = load_universe(cfg.universe_file)
        self.strategy_cfg = cfg.get("strategy", default={})
        self.risk_cfg = cfg.get("risk", default={})

    # ------------------------------------------------------------------
    def check_exits(self) -> None:
        positions = self.broker.positions()
        for symbol, pos in positions.items():
            try:
                quote = self.data.get_quote(symbol)
                history = self.data.get_history(symbol, period="1y")
            except DataUnavailableError as exc:
                log.warning("Skipping exit-check for %s: %s", symbol, exc)
                continue

            self.broker.update_trailing_high(symbol, quote.ltp)
            should_exit, reason = check_exit(pos, quote, history, {**self.risk_cfg, **self.strategy_cfg})
            if should_exit:
                try:
                    self.broker.sell(symbol, pos.quantity, quote.ltp, reason)
                except ValueError as exc:
                    log.error("Failed to sell %s: %s", symbol, exc)

    def scan_for_entries(self) -> None:
        positions = self.broker.positions()
        room = self.risk.room_for_new_positions(len(positions))
        if room <= 0:
            log.info("Max open positions reached (%d); skipping entry scan", self.risk.max_open_positions)
            return

        candidates = []
        for symbol in self.universe:
            if symbol in positions:
                continue
            try:
                quote = self.data.get_quote(symbol)
                history = self.data.get_history(symbol, period="1y")
                turnover = self.data.get_avg_daily_turnover(symbol, history=history)
            except DataUnavailableError as exc:
                log.debug("Skipping %s: %s", symbol, exc)
                continue

            cand = evaluate_candidate(symbol, quote, history, turnover, self.strategy_cfg)
            if cand:
                candidates.append(cand)

        ranked = rank_candidates(candidates)
        max_new = min(room, self.strategy_cfg.get("max_new_positions_per_scan", 3))
        ranked = ranked[:max_new]

        if not ranked:
            log.info("No qualifying 52-week-high momentum candidates this scan.")
            return

        free_cash = self.broker.cash()
        scan_budget = self.risk.scan_cash_budget(free_cash)
        # Approximate total equity once per scan (existing positions valued
        # at their average cost) for consistent position sizing across the
        # candidates picked in this pass.
        equity = self.broker.equity({})
        spent = 0.0
        for cand in ranked:
            qty = self.risk.position_size_shares(equity, cand.ltp)
            if qty <= 0:
                continue
            cost_estimate = qty * cand.ltp
            if spent + cost_estimate > scan_budget:
                log.info("Per-scan cash budget reached; deferring %s to next scan", cand.symbol)
                continue
            try:
                self.broker.buy(
                    cand.symbol, qty, cand.ltp,
                    reason=(
                        f"momentum_52w_high score={cand.score:.1f} "
                        f"{cand.pct_from_52w_high:.1f}% off 52w-high, "
                        f"{cand.momentum_return_pct:.1f}% {self.strategy_cfg.get('momentum_lookback_days')}d return"
                    ),
                )
                spent += cost_estimate
            except InsufficientFundsError as exc:
                log.warning("Insufficient funds for %s: %s", cand.symbol, exc)

    def mark_to_market(self) -> None:
        quotes = {}
        for symbol in self.broker.positions():
            try:
                quotes[symbol] = self.data.get_quote(symbol).ltp
            except DataUnavailableError:
                continue
        positions_value = sum(p.quantity * quotes.get(s, p.avg_price) for s, p in self.broker.positions().items())
        self.storage.record_equity(self.broker.cash(), positions_value)

    def run_once(self) -> None:
        log.info("=== Scan started %s ===", datetime.now().isoformat(timespec="seconds"))
        self.check_exits()
        self.scan_for_entries()
        self.mark_to_market()
        log.info("=== Scan complete. Cash=%.2f, Open positions=%d ===", self.broker.cash(), len(self.broker.positions()))

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        scan_interval = self.cfg.get("engine", "scan_interval_minutes", default=15) * 60
        exit_interval = self.cfg.get("engine", "exit_check_interval_minutes", default=5) * 60
        last_full_scan = 0.0
        log.info("Autonomous trading engine started. Universe size=%d", len(self.universe))
        while True:
            now_dt = self.calendar.now()
            if not self.calendar.is_market_open(now_dt):
                log.info("Market closed (%s). Sleeping 5 minutes...", now_dt.strftime("%Y-%m-%d %H:%M %Z"))
                time.sleep(300)
                continue

            self.check_exits()
            self.mark_to_market()

            if time.time() - last_full_scan >= scan_interval:
                self.scan_for_entries()
                last_full_scan = time.time()

            time.sleep(exit_interval)
