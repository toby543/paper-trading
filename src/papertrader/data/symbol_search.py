"""Search over data/symbol_directory.csv (every NSE/BSE equity + company name)."""
from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass

from ..config import REPO_ROOT

DIRECTORY_PATH = os.path.join(REPO_ROOT, "data", "symbol_directory.csv")


@dataclass(frozen=True)
class SymbolEntry:
    symbol: str
    name: str
    exchange: str


_lock = threading.Lock()
_cache: dict[str, list[SymbolEntry]] = {}


def load_directory(path: str = DIRECTORY_PATH) -> list[SymbolEntry]:
    with _lock:
        if path not in _cache:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                _cache[path] = [
                    SymbolEntry(r["symbol"].strip().upper(), r["name"].strip(), r["exchange"].strip())
                    for r in csv.DictReader(fh) if r.get("symbol")
                ]
        return _cache[path]


def search(query: str, limit: int = 15, path: str = DIRECTORY_PATH) -> list[SymbolEntry]:
    """Case-insensitive match on symbol or company name. Ranked: exact symbol,
    symbol prefix, name prefix, word inside the name, then any substring; NSE before BSE."""
    q = query.strip().upper()
    if not q:
        return []
    ranked: list[tuple[int, int, str, SymbolEntry]] = []
    for e in load_directory(path):
        name = e.name.upper()
        if e.symbol == q:
            rank = 0
        elif e.symbol.startswith(q):
            rank = 1
        elif name.startswith(q):
            rank = 2
        elif (" " + q) in name:
            rank = 3
        elif q in e.symbol or q in name:
            rank = 4
        else:
            continue
        ranked.append((rank, 0 if e.exchange == "NSE" else 1, e.symbol, e))
    ranked.sort(key=lambda t: t[:3])
    return [t[3] for t in ranked[:limit]]


def find(symbol: str, path: str = DIRECTORY_PATH) -> SymbolEntry | None:
    sym = symbol.strip().upper()
    for e in load_directory(path):
        if e.symbol == sym:
            return e
    return None
