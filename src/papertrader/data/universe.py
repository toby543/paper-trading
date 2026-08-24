"""Loads the tradable stock universe from CSV."""
from __future__ import annotations

import csv


def load_universe(path: str) -> list[str]:
    symbols: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                symbols.append(sym)
    return symbols
