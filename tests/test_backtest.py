"""
tests/test_backtest.py
======================
Unit tests for bayesian_kelly.backtest.
"""

import numpy as np
import pytest
from bayesian_kelly.backtest import (
    ema_atr,
    donchian_signals,
    simulate_trades,
    run_kelly,
    performance,
)
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ohlcv():
    """Synthetic trending price series with known trade outcomes."""
    np.random.seed(42)
    n = 500
    dates = [datetime(2000, 1, 1)]
    for i in range(1, n):
        from datetime import timedelta
        dates.append(dates[-1] + timedelta(days=1))

    closes = 1000 + np.cumsum(np.random.randn(n) * 5)
    highs = closes + np.abs(np.random.randn(n) * 3)
    lows = closes - np.abs(np.random.randn(n) * 3)
    return dates, closes, highs, lows


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

class TestEmaATR:
    def test_output_shape(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        assert len(atr) == len(closes)

    def test_all_positive(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        assert np.all(atr > 0)

    def test_custom_period(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows, period=21)
        assert len(atr) == len(closes)

    def test_atr_less_than_range(self, synthetic_ohlcv):
        """ATR should not systematically exceed the daily range."""
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        daily_range = highs - lows
        # On average ATR should be in the ballpark of daily range
        assert atr.mean() < daily_range.mean() * 3


# ---------------------------------------------------------------------------
# Donchian signal tests
# ---------------------------------------------------------------------------

class TestDonchianSignals:
    def test_output_shape(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        entries, directions = donchian_signals(closes)
        assert len(entries) == len(closes)
        assert len(directions) == len(closes)

    def test_first_bar_no_signal(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        entries, _ = donchian_signals(closes)
        assert not entries[0]

    def test_directions_only_plus_minus_one(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        _, directions = donchian_signals(closes)
        assert set(np.unique(directions)).issubset({-1, 1})

    def test_some_signals_generated(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        entries, _ = donchian_signals(closes)
        assert entries.sum() > 0


# ---------------------------------------------------------------------------
# Trade simulation tests
# ---------------------------------------------------------------------------

class TestSimulateTrades:
    def test_returns_list(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        entries, directions = donchian_signals(closes)
        trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)
        assert isinstance(trades, list)

    def test_trade_fields(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        entries, directions = donchian_signals(closes)
        trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)
        if trades:
            t = trades[0]
            assert "entry_date" in t
            assert "exit_date" in t
            assert "outcome" in t
            assert "win" in t

    def test_outcomes_binary(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        entries, directions = donchian_signals(closes)
        trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)
        outcomes = [t["outcome"] for t in trades]
        unique = set(round(o, 6) for o in outcomes)
        rr = 2.0 / 1.5
        assert unique.issubset({-1.0, round(rr, 6)})

    def test_exit_after_entry(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        entries, directions = donchian_signals(closes)
        trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)
        for t in trades:
            assert t["exit_date"] >= t["entry_date"]


# ---------------------------------------------------------------------------
# run_kelly tests
# ---------------------------------------------------------------------------

class TestRunKelly:
    @pytest.fixture
    def sample_trades(self, synthetic_ohlcv):
        dates, closes, highs, lows = synthetic_ohlcv
        atr = ema_atr(closes, highs, lows)
        entries, directions = donchian_signals(closes)
        return simulate_trades(dates, closes, highs, lows, atr, entries, directions)

    def test_capital_shape(self, sample_trades):
        cap, fracs = run_kelly(sample_trades, "bayesian")
        assert len(cap) == len(sample_trades) + 1

    def test_fracs_shape(self, sample_trades):
        cap, fracs = run_kelly(sample_trades, "bayesian")
        assert len(fracs) == len(sample_trades)

    def test_capital_positive(self, sample_trades):
        for strat in ("full", "half", "quarter", "bayesian"):
            cap, _ = run_kelly(sample_trades, strat)
            assert np.all(cap >= 0)

    def test_initial_capital(self, sample_trades):
        cap, _ = run_kelly(sample_trades, "bayesian", initial_capital=50_000)
        assert cap[0] == pytest.approx(50_000)

    def test_invalid_strategy(self, sample_trades):
        with pytest.raises(ValueError):
            run_kelly(sample_trades, "turbo_kelly")

    def test_fracs_bounded(self, sample_trades):
        for strat in ("full", "half", "quarter", "bayesian"):
            _, fracs = run_kelly(sample_trades, strat, f_max=0.25)
            assert np.all(fracs >= 0)
            assert np.all(fracs <= 0.25 + 1e-9)


# ---------------------------------------------------------------------------
# Performance metric tests
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_flat_curve(self):
        cap = np.ones(101) * 100_000
        m = performance(cap)
        assert m["terminal_x"] == pytest.approx(1.0)
        assert m["max_dd_pct"] == pytest.approx(0.0, abs=1e-6)
        assert m["total_return_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_growing_curve(self):
        cap = np.linspace(100_000, 200_000, 101)
        m = performance(cap)
        assert m["terminal_x"] == pytest.approx(2.0, rel=0.01)
        assert m["max_dd_pct"] == pytest.approx(0.0, abs=1e-3)

    def test_drawdown_captured(self):
        cap = np.array([100_000, 120_000, 80_000, 110_000], dtype=float)
        m = performance(cap)
        # Peak is 120k, trough is 80k → DD = 1 - 80/120 ≈ 33.3%
        assert m["max_dd_pct"] == pytest.approx(33.33, rel=0.01)

    def test_calmar_positive_for_growing_curve(self):
        cap = np.array([100_000, 110_000, 105_000, 130_000], dtype=float)
        m = performance(cap)
        assert m["calmar"] > 0

    def test_n_trades(self):
        cap = np.ones(51) * 100_000
        m = performance(cap)
        assert m["n_trades"] == 50
