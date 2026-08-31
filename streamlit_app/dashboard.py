"""Read-only Streamlit companion viewer for the Momentum Desk paper
trading dashboard.

This is a SEPARATE, optional app -- it does not run the trading engine
and holds no state of its own. It logs into the real Flask dashboard
(`python main.py serve`) over HTTP using the exact same username/
password/2FA flow a browser would, using the returned session cookie
to poll its existing /api/* endpoints, and renders the results.

Why this exists: Streamlit Community Cloud cannot host the Flask app
itself (it only runs `streamlit run` scripts, sleeps after 12h with no
traffic, and gives no durable disk for the SQLite ledger) -- see the
project README's "Streamlit companion viewer" section. This script is
the supported way to get a Streamlit-hosted *view* of the account
without trying to move the actual engine there. Your Flask dashboard
still needs to be reachable at some URL this script's process can
reach (e.g. a VM's public IP:port) -- pointing this at 127.0.0.1 only
works if you also run this script on that same machine.

Run locally:
    streamlit run streamlit_app/dashboard.py

Deploy on Streamlit Community Cloud: point the app's main file at
streamlit_app/dashboard.py (Streamlit Cloud picks up
streamlit_app/requirements.txt automatically since it lives next to
the entry point).
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Momentum Desk Viewer", layout="wide", page_icon="📈")


class AuthError(RuntimeError):
    pass


def _login(base_url: str, username: str, password: str, code: str) -> requests.Session:
    session = requests.Session()

    resp = session.post(
        f"{base_url}/login",
        data={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code == 429:
        raise AuthError("Too many failed attempts on this account. Wait 15 minutes and try again.")
    if resp.status_code == 401:
        raise AuthError("Incorrect username or password.")
    resp.raise_for_status()

    # The password step re-renders one of two templates depending on
    # whether this account has already enrolled in 2FA -- distinguish
    # them by which endpoint the form on the page posts to, rather than
    # trying to parse the page's visible text (which can change).
    if 'action="/login/enroll"' in resp.text:
        raise AuthError(
            "This account hasn't set up 2FA yet. Log into the main dashboard in a "
            "browser first to scan the QR code and finish enrollment, then come back here."
        )
    if 'action="/login/verify"' not in resp.text:
        raise AuthError("Unexpected response from the dashboard while logging in.")

    if not code:
        raise AuthError("Enter the 6-digit code from your authenticator app.")

    resp2 = session.post(f"{base_url}/login/verify", data={"code": code}, timeout=10)
    if resp2.status_code == 429:
        raise AuthError("Too many failed attempts on this account. Wait 15 minutes and try again.")
    if resp2.status_code == 401:
        raise AuthError("Incorrect 2FA code.")
    resp2.raise_for_status()
    return session


def _get_json(session: requests.Session, base_url: str, path: str, **params):
    resp = session.get(f"{base_url}{path}", params=params, timeout=10)
    if resp.status_code == 401:
        raise AuthError("Session expired -- please log in again.")
    resp.raise_for_status()
    return resp.json()


def _pnl_color(value: float) -> str:
    return "normal" if value >= 0 else "inverse"


if "api_session" not in st.session_state:
    st.session_state.api_session = None
    st.session_state.base_url = ""

with st.sidebar:
    st.header("Connect")
    base_url_input = st.text_input(
        "Dashboard URL", value=st.session_state.base_url,
        placeholder="https://your-server:8000",
        help="Must be reachable from wherever this Streamlit app runs -- "
             "127.0.0.1 only works if you're also running this script locally.",
    )
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    code = st.text_input("2FA code", max_chars=6, help="From your authenticator app")

    if st.button("Log in", type="primary", use_container_width=True):
        try:
            base_url_clean = base_url_input.rstrip("/")
            st.session_state.api_session = _login(base_url_clean, username, password, code)
            st.session_state.base_url = base_url_clean
            st.success("Logged in.")
        except AuthError as exc:
            st.session_state.api_session = None
            st.error(str(exc))
        except requests.RequestException as exc:
            st.session_state.api_session = None
            st.error(f"Could not reach {base_url_input}: {exc}")

    if st.session_state.api_session is not None:
        st.success(f"Connected to {st.session_state.base_url}")
        if st.button("Log out", use_container_width=True):
            st.session_state.api_session = None
            st.rerun()

    refresh_seconds = st.slider("Auto-refresh", 0, 120, 30, step=5, format="%d sec (0 = off)")

if st.session_state.api_session is None:
    st.title("📈 Momentum Desk — Viewer")
    st.info("Log in from the sidebar to view your portfolio. This is a read-only "
            "companion to the main dashboard -- it can't place trades or change settings.")
    st.stop()

if refresh_seconds:
    st_autorefresh(interval=refresh_seconds * 1000, key="autorefresh")

sess = st.session_state.api_session
base = st.session_state.base_url

try:
    summary = _get_json(sess, base, "/api/summary")
    trades = _get_json(sess, base, "/api/trades", limit=50)
    curve = _get_json(sess, base, "/api/equity_curve")
except AuthError as exc:
    st.session_state.api_session = None
    st.error(str(exc))
    st.stop()
except requests.RequestException as exc:
    st.error(f"Could not reach the dashboard: {exc}")
    st.stop()

st.title("📈 Momentum Desk — Viewer")
status_cols = st.columns(2)
status_cols[0].markdown(f"**Market:** {'🟢 Open' if summary.get('market_open') else '⚪ Closed'}")
regime = summary.get("regime") or {}
if regime.get("enabled") and regime.get("status"):
    label = "🟢 Uptrend" if regime["status"] == "up" else "🔴 Downtrend"
    status_cols[1].markdown(f"**Regime ({regime.get('index_symbol', '').lstrip('^')}):** {label}")

st.caption(f"As of {summary.get('as_of', '—')}")

positions = summary.get("positions", [])
unrealized = sum(p.get("unrealized_pnl", 0.0) for p in positions)

cols = st.columns(7)
cols[0].metric("Total Equity", f"₹{summary.get('total_equity', 0):,.2f}",
                f"{summary.get('total_pnl', 0):+,.2f} ({summary.get('total_pnl_pct', 0):+.2f}%)")
cols[1].metric("Cash", f"₹{summary.get('cash', 0):,.2f}")
cols[2].metric("Positions Value", f"₹{summary.get('positions_value', 0):,.2f}")
cols[3].metric("Unrealized P&L", f"₹{unrealized:,.2f}", delta_color=_pnl_color(unrealized))
realized = summary.get("total_realized_pnl", 0.0)
cols[4].metric("Realized P&L", f"₹{realized:,.2f}", delta_color=_pnl_color(realized))
cols[5].metric("Open Positions", f"{summary.get('open_positions', 0)} / {summary.get('max_positions', 0)}")
cols[6].metric("Universe Scanned", f"{summary.get('universe_size', 0):,}")

st.subheader("Open Positions")
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
else:
    st.caption("No open positions.")

st.subheader("Equity Curve")
if curve:
    curve_df = pd.DataFrame(curve)
    curve_df["timestamp"] = pd.to_datetime(curve_df["timestamp"])
    st.line_chart(curve_df.set_index("timestamp")["total_equity"])
else:
    st.caption("Not enough equity history yet.")

st.subheader("Recent Trades")
if trades:
    st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
else:
    st.caption("No trades yet.")
