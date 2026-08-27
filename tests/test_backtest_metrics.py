from papertrader.backtest.metrics import avg_value, cagr_pct, max_drawdown_pct, win_rate_pct


def test_max_drawdown_empty_and_flat():
    assert max_drawdown_pct([]) == 0.0
    assert max_drawdown_pct([100]) == 0.0
    assert max_drawdown_pct([100, 110, 120]) == 0.0


def test_max_drawdown_computes_worst_peak_to_trough():
    dd = max_drawdown_pct([100, 120, 90, 130, 80])
    assert abs(dd - 38.4615) < 0.01


def test_cagr_doubling_in_one_year():
    c = cagr_pct(100_000, 200_000, 365)
    assert abs(c - 100.0) < 1.0


def test_cagr_doubling_in_two_years():
    c = cagr_pct(100_000, 200_000, 730)
    assert abs(c - 41.4) < 1.0


def test_cagr_degenerate_inputs_return_zero():
    assert cagr_pct(0, 100, 365) == 0.0
    assert cagr_pct(100, 0, 365) == 0.0
    assert cagr_pct(100, 200, 0) == 0.0
    assert cagr_pct(-100, 200, 365) == 0.0


def test_win_rate():
    assert win_rate_pct([]) == 0.0
    assert win_rate_pct([10, -5, 20, -1]) == 50.0
    assert win_rate_pct([10, 20, 30]) == 100.0
    assert win_rate_pct([-1, -2]) == 0.0


def test_avg_value():
    assert avg_value([]) == 0.0
    assert avg_value([10, 20, 30]) == 20.0
