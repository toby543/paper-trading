"""Builds data/symbol_directory.csv -- every NSE and BSE equity and NSE ETF with
its company name, for the Manual Swing profile's stock search.

NSE symbols are written as-is; BSE-only stocks (ISIN not on NSE) as
"<BSE ticker>.BO", matching the universe convention. Re-run occasionally so
new listings become searchable:
    python scripts/build_symbol_directory.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ipo_universe import BSE_SERIES, NSE_SERIES, fetch_bse_bhavcopy, fetch_nse  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "symbol_directory.csv")


NSE_SECURITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"


def fetch_nse_security_list() -> pd.DataFrame:
    """Everything NSE lists for trading, including ETFs (NIFTYBEES etc.) that
    the equity list leaves out, and listings newer than that file."""
    resp = requests.get(NSE_SECURITY_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    df["Series"] = df["Series"].str.strip()
    return df


def main() -> int:
    nse = fetch_nse()
    nse_eq = nse[nse["SERIES"].isin(NSE_SERIES)]
    rows = [(r["SYMBOL"].strip().upper(), r["NAME OF COMPANY"].strip(), "NSE") for _, r in nse_eq.iterrows()]
    print(f"NSE: {len(rows)} equities")

    have = {r[0] for r in rows}
    sec = fetch_nse_security_list()
    extra = [(r["Symbol"].strip().upper(), str(r["Security Name"]).strip(), "NSE")
             for _, r in sec[sec["Series"].isin(NSE_SERIES)].iterrows()
             if r["Symbol"].strip().upper() not in have]
    rows += extra
    print(f"NSE ETFs and newer listings from the security list: {len(extra)}")

    bhav = fetch_bse_bhavcopy()
    bhav = bhav[bhav["SctySrs"].isin(BSE_SERIES | {"Z"})]
    bhav = bhav[bhav["ISIN"].astype(str).str.startswith("INE")]
    bhav = bhav[~bhav["ISIN"].isin(set(nse["ISIN NUMBER"].str.strip()))]
    bse_rows = [(f"{r['TckrSymb'].strip().upper()}.BO", str(r["FinInstrmNm"]).strip(), "BSE")
                for _, r in bhav.drop_duplicates("TckrSymb").iterrows()]
    print(f"BSE-only: {len(bse_rows)} equities")

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "name", "exchange"])
        writer.writerows(sorted(rows + bse_rows))
    print(f"Wrote {len(rows) + len(bse_rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
