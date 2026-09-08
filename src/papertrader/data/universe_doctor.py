"""Finds and repairs stale symbols in the tradable universe CSV.

An NSE ticker that no longer resolves is invisible in normal operation: the
fetch loop logs it at DEBUG and moves on, so a universe silently shrinks as
companies rename, merge or delist. A 2025-09 -> 2026-09 backtest over
`universe_nifty500.csv` resolved only 435 of 479 symbols -- and several of
the 44 failures were not delistings at all but *stale aliases for companies
still in the file under their current ticker* (NALCO alongside NATIONALUM,
MINDA alongside UNOMINDA, JBPHARMA alongside JBCHEPHARM), so those names
were being screened once instead of twice while looking like attrition.

The renames below are CANDIDATES, not facts. Every one is probed against
the data source before anything is written, so a wrong guess here can never
silently replace a good symbol with a dead one -- it just fails to verify
and gets reported. That matters because these mappings are the kind of
thing that goes out of date exactly as quietly as the problem they fix.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger(__name__)

# Stale symbol -> (candidate replacement, why). Verified at runtime.
KNOWN_RENAMES: dict[str, tuple[str, str]] = {
    "BHARATFORGE": ("BHARATFORG", "NSE ticker is BHARATFORG, not the full name"),
    "CEAT": ("CEATLTD", "NSE ticker is CEATLTD"),
    "LTIM": ("LTIMINDTREE", "LTI Mindtree trades as LTIMINDTREE"),
    "NALCO": ("NATIONALUM", "National Aluminium trades as NATIONALUM"),
    "MINDA": ("UNOMINDA", "Minda Industries renamed Uno Minda"),
    "JBPHARMA": ("JBCHEPHARM", "JB Chemicals & Pharmaceuticals"),
    "AMARAJABAT": ("ARE&M", "Amara Raja renamed Amara Raja Energy & Mobility"),
    "GARDENREACH": ("GRSE", "Garden Reach Shipbuilders trades as GRSE"),
    "NARAYANHRUD": ("NH", "Narayana Hrudayalaya trades as NH"),
    "RAINBOWCH": ("RAINBOW", "Rainbow Children's Medicare"),
    "REPCO": ("REPCOHOME", "Repco Home Finance"),
    "WOCKPHARDT": ("WOCKPHARMA", "Wockhardt trades as WOCKPHARMA"),
    "WELSPUNIND": ("WELSPUNLIV", "Welspun India renamed Welspun Living"),
    "GOKALDAS": ("GOKEX", "Gokaldas Exports trades as GOKEX"),
    "SIYARAM": ("SIYSIL", "Siyaram Silk Mills"),
    "VARDHMAN": ("VTL", "Vardhman Textiles trades as VTL"),
    "VRL": ("VRLLOG", "VRL Logistics"),
    "TIPS": ("TIPSMUSIC", "Tips Industries renamed Tips Music"),
    "MAXFIN": ("MFSL", "Max Financial Services"),
    "STRIDES": ("STAR", "Strides Pharma Science trades as STAR"),
    "KALPATPOWR": ("KPIL", "Kalpataru Power renamed Kalpataru Projects International"),
    "HBLPOWER": ("HBLENGINE", "HBL Power renamed HBL Engineering"),
    "CENTURYTEX": ("ABREL", "Century Textiles renamed Aditya Birla Real Estate"),
}


@dataclass
class UniverseReport:
    ok: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)      # (old, new)
    duplicates: list[tuple[str, str]] = field(default_factory=list)   # (stale, existing)
    unresolved: list[str] = field(default_factory=list)               # no working replacement

    @property
    def needs_attention(self) -> bool:
        return bool(self.renamed or self.duplicates or self.unresolved)


def probe_yfinance(symbol: str) -> bool:
    """True if `symbol` resolves to real NSE price data right now."""
    import yfinance as yf
    try:
        df = yf.Ticker(symbol + ".NS").history(period="5d")
        return not df.empty
    except Exception:  # noqa: BLE001 - any failure means "can't use this symbol"
        return False


def diagnose(
    symbols: Iterable[str],
    probe: Callable[[str], bool] = probe_yfinance,
    pause_seconds: float = 0.2,
    on_progress: Callable[[int, int], None] | None = None,
) -> UniverseReport:
    """Probe every symbol and work out what to do with the broken ones.

    `probe` is injected so this is testable without a network, and paced by
    default because firing a few hundred requests at Yahoo back to back is
    what trips its throttling -- which returns 404s indistinguishable from
    a genuinely dead ticker, i.e. the exact failure this tool exists to
    diagnose.
    """
    symbols = list(symbols)
    present = set(symbols)
    report = UniverseReport()
    verified: dict[str, bool] = {}

    def check(sym: str) -> bool:
        if sym not in verified:
            if verified:
                time.sleep(pause_seconds)
            verified[sym] = probe(sym)
        return verified[sym]

    for i, sym in enumerate(symbols):
        if check(sym):
            report.ok.append(sym)
        else:
            candidate = KNOWN_RENAMES.get(sym, (None, ""))[0]
            if candidate and candidate in present:
                # The live ticker is already in the universe -- this entry is
                # a stale alias for a company we are already screening.
                report.duplicates.append((sym, candidate))
            elif candidate and check(candidate):
                report.renamed.append((sym, candidate))
                present.add(candidate)
            else:
                report.unresolved.append(sym)
        if on_progress:
            on_progress(i + 1, len(symbols))
    return report


def apply_report(path: str, report: UniverseReport, drop_unresolved: bool = False) -> int:
    """Rewrite the universe CSV per `report`. Returns rows changed.

    Only verified replacements are written. Unresolved symbols are left in
    place unless `drop_unresolved` -- removing a symbol because one probe
    failed would quietly shrink the universe on a bad network day, which is
    the same class of silent loss this tool exists to surface.
    """
    renames = dict(report.renamed)
    drop = {stale for stale, _ in report.duplicates}
    if drop_unresolved:
        drop |= set(report.unresolved)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = ["symbol"]

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    changed = 0
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if sym in drop:
            changed += 1
            continue
        new = renames.get(sym, sym)
        if new in seen:  # a rename that collides with an entry already kept
            changed += 1
            continue
        if new != sym:
            changed += 1
        seen.add(new)
        out.append({"symbol": new})

    with open(path, "w", encoding="utf-8", newline="") as fh:
        # csv defaults to CRLF, which would rewrite every line of a
        # LF-terminated file and bury a three-symbol fix in a 480-line diff.
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    return changed
