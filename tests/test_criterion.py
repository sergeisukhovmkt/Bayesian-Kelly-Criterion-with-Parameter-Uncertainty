"""
tests/test_criterion.py
=======================
Unit tests for bayesian_kelly.criterion.BayesianKelly.
"""

import math
import pytest
from bayesian_kelly.criterion import BayesianKelly, BayesianKellyResult


class TestBayesianKellyInit:
    def test_default_params(self):
        bk = BayesianKelly()
        assert bk.alpha == 1.0
        assert bk.beta == 1.0
        assert bk.kappa == 30.0
        assert bk.f_max == 0.25

    def test_custom_params(self):
        bk = BayesianKelly(alpha0=10.0, beta0=5.0, kappa=50.0, f_max=0.15)
        assert bk.alpha == 10.0
        assert bk.beta == 5.0
        assert bk.kappa == 50.0
        assert bk.f_max == 0.15

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="α₀"):
            BayesianKelly(alpha0=0.0)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="β₀"):
            BayesianKelly(beta0=-1.0)

    def test_invalid_kappa(self):
        with pytest.raises(ValueError, match="κ"):
            BayesianKelly(kappa=0.0)

    def test_invalid_f_max(self):
        with pytest.raises(ValueError, match="f_max"):
            BayesianKelly(f_max=0.0)

    def test_invalid_f_max_above_one(self):
        with pytest.raises(ValueError, match="f_max"):
            BayesianKelly(f_max=1.5)


class TestBayesianKellyUpdate:
    def test_win_increments_alpha(self):
        bk = BayesianKelly()
        bk.update(True)
        assert bk.alpha == 2.0
        assert bk.beta == 1.0

    def test_loss_increments_beta(self):
        bk = BayesianKelly()
        bk.update(False)
        assert bk.alpha == 1.0
        assert bk.beta == 2.0

    def test_trade_counter(self):
        bk = BayesianKelly()
        for _ in range(5):
            bk.update(True)
        assert bk.n_trades == 5

    def test_batch_update_equivalent(self):
        bk1 = BayesianKelly()
        bk2 = BayesianKelly()

        outcomes = [True, False, True, True, False, True]
        for o in outcomes:
            bk1.update(o)
        bk2.update_batch(outcomes)

        assert bk1.alpha == bk2.alpha
        assert bk1.beta == bk2.beta
        assert bk1.n_trades == bk2.n_trades


class TestBayesianKellyCalculate:
    def test_returns_result_type(self):
        bk = BayesianKelly()
        result = bk.calculate(b=1.0)
        assert isinstance(result, BayesianKellyResult)

    def test_no_data_cold_start(self):
        """With uniform prior and b=1, position should be near zero."""
        bk = BayesianKelly(kappa=30.0)
        result = bk.calculate(b=1.0)
        # p̄ = 0.5, f_base = 0.5 - 0.5/1 = 0.0
        assert result.position_size == pytest.approx(0.0, abs=1e-9)

    def test_positive_edge_yields_positive_size(self):
        bk = BayesianKelly()
        bk.update_batch([True] * 60 + [False] * 40)
        result = bk.calculate(b=1.0)
        assert result.position_size > 0.0

    def test_negative_edge_yields_zero(self):
        bk = BayesianKelly()
        bk.update_batch([False] * 70 + [True] * 30)
        result = bk.calculate(b=1.0)
        assert result.position_size == 0.0

    def test_f_max_cap(self):
        bk = BayesianKelly(f_max=0.10)
        # Extreme edge: should be capped
        bk.update_batch([True] * 1000)
        result = bk.calculate(b=10.0)
        assert result.position_size <= 0.10 + 1e-12

    def test_confidence_weight_bounds(self):
        bk = BayesianKelly()
        result = bk.calculate(b=1.33)
        assert 0.0 < result.confidence_weight < 1.0

    def test_confidence_weight_increases_with_n(self):
        bk_small = BayesianKelly(kappa=30)
        bk_large = BayesianKelly(kappa=30)

        bk_small.update_batch([True, False] * 10)
        bk_large.update_batch([True, False] * 200)

        r_small = bk_small.calculate(b=1.0)
        r_large = bk_large.calculate(b=1.0)

        assert r_large.confidence_weight > r_small.confidence_weight

    def test_invalid_odds(self):
        bk = BayesianKelly()
        with pytest.raises(ValueError, match="b"):
            bk.calculate(b=0.0)

    def test_posterior_mean_convergence(self):
        """After many wins, posterior mean should approach true win rate."""
        true_p = 0.60
        bk = BayesianKelly()
        import random
        rng = random.Random(42)
        for _ in range(2000):
            bk.update(rng.random() < true_p)
        assert abs(bk.posterior_mean - true_p) < 0.03

    def test_position_size_grows_with_n_eff(self):
        """Larger n_eff should yield larger position size (same p̄)."""
        sizes = []
        for n_wins in [5, 50, 500]:
            n_losses = n_wins  # keep p̄ = 0.5 → f_base = 0
            bk = BayesianKelly(kappa=30)
            bk.update_batch([True] * n_wins + [False] * n_losses)
            # use b=2 so f_base > 0
            sizes.append(bk.calculate(b=2.0).position_size)
        assert sizes[0] < sizes[1] < sizes[2]


class TestBayesianKellyReset:
    def test_reset_restores_prior(self):
        bk = BayesianKelly(alpha0=2.0, beta0=3.0)
        bk.update_batch([True] * 100)
        bk.reset()
        assert bk.alpha == 2.0
        assert bk.beta == 3.0
        assert bk.n_trades == 0

    def test_reset_then_recalculate(self):
        bk = BayesianKelly()
        bk.update_batch([True] * 100)
        f_after_training = bk.calculate(b=1.33).position_size
        bk.reset()
        f_after_reset = bk.calculate(b=1.33).position_size
        # After reset, should be back to cold-start level
        assert f_after_reset < f_after_training


class TestBayesianKellyProperties:
    def test_credible_interval_contains_mean(self):
        bk = BayesianKelly()
        bk.update_batch([True] * 30 + [False] * 20)
        lo, hi = bk.credible_interval_95
        assert lo <= bk.posterior_mean <= hi

    def test_credible_interval_in_unit_interval(self):
        bk = BayesianKelly()
        bk.update_batch([True] * 5)
        lo, hi = bk.credible_interval_95
        assert 0.0 <= lo <= hi <= 1.0

    def test_posterior_std_decreases_with_n(self):
        bk_small = BayesianKelly()
        bk_large = BayesianKelly()
        bk_small.update_batch([True] * 5 + [False] * 5)
        bk_large.update_batch([True] * 500 + [False] * 500)
        assert bk_large.posterior_std < bk_small.posterior_std

    def test_repr(self):
        bk = BayesianKelly()
        bk.update(True)
        r = repr(bk)
        assert "BayesianKelly" in r
        assert "α=" in r
