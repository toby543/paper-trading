"""Builds data/universe_ipo_all.csv for the IPO Base Breakout profile.

Covers every sector on both exchanges (no sector filtering anywhere):
  * NSE: every EQ/BE-series listing whose DATE OF LISTING falls within
    --max-age-days. The strategy only accepts stocks 60-500 trading days
    old, so older listings can never qualify -- dropping them up front cuts
    ~2,500 symbols to a few hundred and keeps a scan fast.
  * BSE-only: BSE main-board stocks whose ISIN is not on the NSE list,
    written as "<BSE ticker>.BO" (Yahoo resolves the ticker name, not the numeric scrip code). BSE publishes no listing date, so these
    are pre-filtered on a minimum one-day traded value instead, and the
    strategy's own history-length check does the age filtering.

Re-run periodically (new IPOs list every week):
    python scripts/build_ipo_universe.py
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_BHAV_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}
NSE_SERIES = {"EQ", "BE"}
BSE_SERIES = {"A", "B", "X", "XT", "T"}
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "universe_ipo_all.csv")


_RIGHTS_ENTITLEMENT = re.compile(r"-RE\d*$", re.IGNORECASE)


def is_rights_entitlement(symbol: str, company_name: str) -> bool:
    """NSE lists a rights entitlement as its own ticker (e.g. CENTEXT-RE,
    company name "...Limited-RE"). It is not a stock and has no price history."""
    return bool(_RIGHTS_ENTITLEMENT.search(str(symbol).strip())
                or _RIGHTS_ENTITLEMENT.search(str(company_name).strip()))


def fetch_nse() -> pd.DataFrame:
    resp = requests.get(NSE_LIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    df["SERIES"] = df["SERIES"].str.strip()
    df["LISTED"] = pd.to_datetime(df["DATE OF LISTING"], format="%d-%b-%Y", errors="coerce")
    return df


def fetch_bse_bhavcopy() -> pd.DataFrame:
    for back in range(0, 10):
        d = date.today() - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        resp = requests.get(BSE_BHAV_URL.format(d=d.strftime("%Y%m%d")), headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.text.startswith("TradDt"):
            print(f"BSE bhavcopy date: {d}")
            return pd.read_csv(io.StringIO(resp.text))
    raise RuntimeError("could not download a recent BSE bhavcopy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-days", type=int, default=800,
                    help="NSE listings older than this many calendar days are dropped "
                         "(500 trading days is ~730 calendar days; default leaves a buffer)")
    ap.add_argument("--bse-min-turnover-inr", type=float, default=2_000_000,
                    help="drop BSE-only stocks whose latest-day traded value is below this")
    ap.add_argument("--no-bse", action="store_true", help="NSE only")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    nse = fetch_nse()
    cutoff = datetime.now() - timedelta(days=args.max_age_days)
    nse_eq = nse[nse["SERIES"].isin(NSE_SERIES)]
    recent = nse_eq[nse_eq["LISTED"] >= cutoff]
    recent = recent[~recent.apply(
        lambda r: is_rights_entitlement(r["SYMBOL"], r["NAME OF COMPANY"]), axis=1)]
    symbols = sorted(recent["SYMBOL"].str.strip().str.upper().unique())
    print(f"NSE: {len(nse_eq)} equity listings, {len(symbols)} listed within {args.max_age_days} days")

    bse_symbols: list[str] = []
    if not args.no_bse:
        nse_isins = set(nse["ISIN NUMBER"].str.strip())
        bhav = fetch_bse_bhavcopy()
        bhav = bhav[bhav["SctySrs"].isin(BSE_SERIES)]
        bhav = bhav[bhav["ISIN"].astype(str).str.startswith("INE")]
        bhav = bhav[~bhav["ISIN"].isin(nse_isins)]
        bhav = bhav[pd.to_numeric(bhav["TtlTrfVal"], errors="coerce").fillna(0) >= args.bse_min_turnover_inr]
        bse_symbols = sorted(f"{t.strip().upper()}.BO" for t in bhav["TckrSymb"].unique())
        print(f"BSE-only: {len(bse_symbols)} stocks above the turnover floor")

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol"])
        for sym in symbols + bse_symbols:
            writer.writerow([sym])
    print(f"Wrote {len(symbols) + len(bse_symbols)} symbols to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
