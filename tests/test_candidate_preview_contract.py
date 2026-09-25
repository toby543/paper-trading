"""Every strategy mode must survive the candidate-preview path, end to end.

Each strategy returns its own Candidate dataclass with its own fields, and
three separate layers have to agree about that shape:

  1. TradingEngine.find_candidates() -- returns the dataclass
  2. data_api.build_candidates()     -- turns it into a row dict per mode
  3. watchlistRowHtml() in index.html -- renders that row dict per mode

A mode missing from layer 2 or 3 does not fail loudly: it silently falls
through to the 52w_high branch, which reads week52_high /
pct_from_52w_high / relative_strength_pct / volume_multiple -- fields no
other strategy's Candidate carries. The result is an AttributeError (or a
JS TypeError on undefined) that only fires once that strategy actually
finds a candidate, so a mode can look completely healthy for weeks while
reporting zero candidates and then break the moment it finds one.

That exact drift has now happened four times in this codebase -- in the
live scheduler's buy-reason builder, in build_candidates(), in the
backtest engine's candidate adapter, and in the dashboard's watchlist
renderer -- each time for a newly added strategy nobody remembered to
register in every layer. These tests walk EVERY mode so the fifth one is
caught automatically rather than in production.
"""
from __future__ import annotations

import dataclasses
import os
import re

import pytest

from papertrader.strategy import (
    consolidation_breakout,
    crypto_breakout,
    crypto_breakout_retest,
    crypto_institutional_swing,
    crypto_mean_reversion,
    crypto_momentum,
    crypto_pairs_trading,
    crypto_trend_pullback,
    long_term_trend,
    momentum_52w_high,
    pivot_supertrend,
    trend_pullback,
)
from papertrader.web.data_api import build_candidates

_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "papertrader", "web", "templates", "index.html",
)

# Every strategy_mode the engine can run, mapped to the module whose
# Candidate that mode's find_candidates() actually returns. Adding a
# strategy without adding it here fails test_every_strategy_mode_is_covered.
MODE_TO_STRATEGY = {
    "52w_high": momentum_52w_high,
    "cross_sectional_momentum": momentum_52w_high,  # imports 52w_high's Candidate directly
    "consolidation_breakout": consolidation_breakout,
    "pivot_supertrend": pivot_supertrend,
    "trend_pullback": trend_pullback,
    "long_term_trend": long_term_trend,
    "crypto_momentum": crypto_momentum,
    "crypto_breakout": crypto_breakout,
    "crypto_institutional_swing": crypto_institutional_swing,
    "crypto_mean_reversion": crypto_mean_reversion,
    "crypto_trend_pullback": crypto_trend_pullback,
    "crypto_breakout_retest": crypto_breakout_retest,
    "crypto_pairs_trading": crypto_pairs_trading,
}


def _candidate_for(strategy):
    """That strategy's Candidate with every numeric field populated.

    Values are arbitrary but non-zero: the point is that every declared
    field EXISTS, so any layer reading a field this dataclass doesn't have
    raises instead of silently reading a default.
    """
    cls = strategy.Candidate
    values = {}
    for field in dataclasses.fields(cls):
        values[field.name] = "X-USD" if field.name == "symbol" else 1.5
    return cls(**values)


class _FakeStorage:
    def get_positions(self):
        return {}


class _FakeBroker:
    def positions(self):
        return {}


class _FakeRisk:
    max_open_positions = 10

    def room_for_new_positions(self, open_position_count):
        return 10


class _FakeEngine:
    """Just enough engine for build_candidates: it reads the mode, the
    regime, position/room counts, and calls find_candidates()."""

    def __init__(self, mode, candidate):
        self.strategy_cfg = {"mode": mode}
        self.regime_cfg = {"enabled": False, "index_symbol": "X"}
        self.broker = _FakeBroker()
        self.storage = _FakeStorage()
        self.risk = _FakeRisk()
        self.trades_24_7 = True
        self._candidate = candidate

    def find_candidates(self, exclude_symbols=None):
        return [self._candidate]


def _frontend_field_reads():
    """{mode: {fields read}} from watchlistRowHtml's per-mode branches,
    plus the trailing fallback branch under the "default" key."""
    source = open(_TEMPLATE, encoding="utf-8").read()
    body = re.search(r"function watchlistRowHtml\(mode, c\) \{(.*?)\n\}", source, re.S).group(1)
    reads = {}
    for match in re.finditer(r'if \(mode === "(\w+)"\) \{\s*return `(.*?)`;\s*\}', body, re.S):
        reads[match.group(1)] = set(re.findall(r"\bc\.(\w+)", match.group(2)))
    reads["default"] = set(re.findall(r"\bc\.(\w+)", body.rsplit("return `", 1)[1]))
    # These tests read the template as text, so a reformat that defeats the
    # patterns above would leave them comparing empty sets and passing
    # vacuously -- exactly the silent hole they exist to close.
    assert reads["default"], "could not parse the fallback branch's field reads"
    return reads


def _frontend_header_columns():
    """{mode: <th> count} from the WATCHLIST_HEAD map."""
    source = open(_TEMPLATE, encoding="utf-8").read()
    block = re.search(r"const WATCHLIST_HEAD = \{(.*?)\n\};", source, re.S).group(1)
    return {
        m.group(1): m.group(2).count("<th>")
        for m in re.finditer(r"^\s*(\w+):\s*`(.*?)`,\s*$", block, re.M)
    }


def _frontend_row_cells():
    """{mode: <td> count} from watchlistRowHtml's per-mode branches."""
    source = open(_TEMPLATE, encoding="utf-8").read()
    body = re.search(r"function watchlistRowHtml\(mode, c\) \{(.*?)\n\}", source, re.S).group(1)
    cells = {
        m.group(1): m.group(2).count("<td")
        for m in re.finditer(r'if \(mode === "(\w+)"\) \{\s*return `(.*?)`;\s*\}', body, re.S)
    }
    cells["default"] = body.rsplit("return `", 1)[1].count("<td")
    return cells


@pytest.mark.parametrize("mode", sorted(MODE_TO_STRATEGY))
def test_build_candidates_handles_every_mode(mode):
    """build_candidates must produce a row for whatever Candidate the mode
    returns. A mode with no branch falls through to the 52w_high shape and
    raises AttributeError on week52_high -- which is how this broke for
    crypto_momentum, crypto_breakout and crypto_institutional_swing."""
    engine = _FakeEngine(mode, _candidate_for(MODE_TO_STRATEGY[mode]))

    result = build_candidates(engine, limit=5)

    assert result["mode"] == mode
    assert len(result["candidates"]) == 1
    row = result["candidates"][0]
    assert row["symbol"] == "X-USD"
    assert "score" in row


@pytest.mark.parametrize("mode", sorted(MODE_TO_STRATEGY))
def test_frontend_reads_only_fields_the_backend_sends(mode):
    """The dashboard renders each row dict by mode. Reading a field the
    backend never puts in that row throws TypeError on undefined in the
    browser and blanks the whole preview table."""
    engine = _FakeEngine(mode, _candidate_for(MODE_TO_STRATEGY[mode]))
    backend_fields = set(build_candidates(engine, limit=5)["candidates"][0])

    reads = _frontend_field_reads()
    # A mode with no branch of its own is rendered by the fallback.
    rendered_by = reads.get(mode, reads["default"])

    missing = rendered_by - backend_fields
    assert not missing, (
        f"the watchlist renderer for {mode} reads {sorted(missing)}, which "
        f"build_candidates() does not send for that mode (it sends "
        f"{sorted(backend_fields)})"
    )


def test_every_strategy_mode_is_covered_by_this_test():
    """Guards the guard: a new strategy registered in the scheduler but not
    in MODE_TO_STRATEGY would otherwise skip every check above."""
    import papertrader.engine.scheduler as scheduler

    source = open(scheduler.__file__, encoding="utf-8").read()
    routed = set(re.findall(r'mode == "(\w+)"', source))
    # Without this the set-difference below could pass on an empty set if the
    # pattern ever stopped matching, silently testing nothing.
    assert len(routed) > 5, f"only discovered {sorted(routed)} in the scheduler -- parser likely broken"
    # 52w_high is the else-branch default, so it never appears as a literal.
    unknown = routed - set(MODE_TO_STRATEGY)
    assert not unknown, (
        f"scheduler routes {sorted(unknown)} but MODE_TO_STRATEGY does not list "
        f"them, so the candidate-preview contract is untested for those modes"
    )


def test_watchlist_headers_and_rows_have_matching_column_counts():
    """A header with more/fewer <th> than the row's <td> renders a visibly
    misaligned table -- cells shift under the wrong headings."""
    headers = _frontend_header_columns()
    rows = _frontend_row_cells()

    for mode, header_count in sorted(headers.items()):
        assert mode in rows, f"WATCHLIST_HEAD defines {mode} but watchlistRowHtml has no branch for it"
        assert header_count == rows[mode], (
            f"{mode}: {header_count} header columns but {rows[mode]} row cells"
        )


def test_every_mode_with_a_row_renderer_has_its_own_header():
    """The inverse: a row branch without a header entry silently renders
    its cells under the default 52w_high headings."""
    headers = _frontend_header_columns()
    rows = _frontend_row_cells()

    missing = set(rows) - set(headers)
    assert not missing, f"watchlistRowHtml renders {sorted(missing)} with no WATCHLIST_HEAD entry"
