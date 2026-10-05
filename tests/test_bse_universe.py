"""BSE support: ".BO" symbols route to Yahoo's BSE ticker, and the IPO
Base Breakout profile scans a combined NSE + BSE universe."""
import yaml

from papertrader.data.nse_client import MarketDataClient, is_bse_symbol, yahoo_ticker
from papertrader.data.universe import load_universe


def test_yahoo_ticker_routes_by_exchange():
    assert yahoo_ticker("RELIANCE") == "RELIANCE.NS"
    assert yahoo_ticker("ABB.BO") == "ABB.BO"
    assert is_bse_symbol("abb.bo") and not is_bse_symbol("BAJAJ-AUTO")


def test_bse_symbols_skip_the_nse_quote_api(monkeypatch):
    client = MarketDataClient()

    def boom(symbol):
        raise AssertionError("NSE API must not be called for a BSE symbol")

    monkeypatch.setattr(client, "_quote_from_nse", boom)
    seen = []
    monkeypatch.setattr(client, "_quote_from_yfinance", lambda s: seen.append(s) or "quote")
    assert client.get_quote("ABB.BO") == "quote"
    assert seen == ["ABB.BO"]


def test_ipo_profile_universe_covers_both_exchanges():
    profile = yaml.safe_load(open("profiles/ipo_base_breakout.yaml"))
    symbols = load_universe(profile["universe_file"])
    assert any(s.endswith(".BO") for s in symbols), "no BSE symbols"
    assert any(not s.endswith(".BO") for s in symbols), "no NSE symbols"
    assert len(symbols) == len(set(symbols))


def _load_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_ipo_universe", "scripts/build_ipo_universe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rights_entitlements_are_not_universe_members():
    is_re = _load_builder().is_rights_entitlement
    assert is_re("CENTEXT-RE", "Century Extrusions Limited-RE")
    assert is_re("ABC-RE1", "ABC Limited")
    assert not is_re("BAJAJ-AUTO", "Bajaj Auto Limited")
    assert not is_re("KLBRENG-B", "Kilburn Engineering Limited")
    assert not is_re("RELIANCE", "Reliance Industries Limited")


def test_universe_has_no_rights_entitlement_symbols():
    symbols = load_universe("data/universe_ipo_all.csv")
    assert not [s for s in symbols if s.upper().endswith("-RE")]
