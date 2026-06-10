"""
bayesian_kelly.criterion
========================
Core implementation of the Bayesian Kelly Criterion (Sukhov, 2026).

The classical Kelly formula assumes exact knowledge of the win probability p.
In practice, p must be estimated from finite samples, introducing estimation
risk that compounds into catastrophic drawdowns.  This module implements the
Bayesian regularisation described in Sukhov (2026), equation (13):

    f* = (p̄ − (1 − p̄)/b) × n_eff / (n_eff + κ)

where
  p̄     = posterior mean of p under Beta(α, β)
  n_eff  = α + β  (effective sample size)
  κ      = robustness parameter (calibrated via simulation; default 30)

References
----------
Sukhov, S. (2026). Bayesian Kelly Criterion with Parameter Uncertainty:
  A Robust Framework for Position Sizing Under Estimation Risk.
  Working Paper.

Kelly, J. L. (1956). A New Interpretation of Information Rate.
  Bell System Technical Journal, 35(4), 917–926.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BayesianKellyResult:
    """Output of a single position-size calculation."""
    position_size: float
    """Recommended fraction of capital to risk [0, f_max]."""
    f_base: float
    """Raw Kelly fraction before confidence weighting (may be negative)."""
    posterior_mean: float
    """Posterior mean p̄ = α / (α + β)."""
    posterior_std: float
    """Posterior standard deviation of p ~ Beta(α, β)."""
    confidence_weight: float
    """Confidence weight Φ = n_eff / (n_eff + κ) ∈ (0, 1)."""
    effective_n: float
    """Effective sample size n_eff = α + β."""
    alpha: float
    """Current posterior α parameter."""
    beta: float
    """Current posterior β parameter."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BayesianKelly:
    """
    Online Bayesian Kelly position sizer with Beta conjugate prior.

    The Beta–Binomial model is fully conjugate: after observing k wins in n
    trades, the posterior is Beta(α₀ + k, β₀ + n − k), updated in O(1).

    Parameters
    ----------
    alpha0 : float
        Prior pseudo-successes.  Use 1.0 (uniform) for most applications;
        use 10.0 for a conservative prior equivalent to 20 prior observations.
    beta0 : float
        Prior pseudo-failures.  Symmetric with alpha0 for a non-informative
        starting point.
    kappa : float
        Robustness parameter κ controlling the confidence discount.
        Recommended values:
          - High-frequency (> 100 trades / day) : κ = 10
          - Swing trading (1–10 trades / day)   : κ = 30
          - Position trading (< 1 trade / day)  : κ = 50
    f_max : float
        Hard cap on position size.  Recommended: 0.25 for single-asset
        strategies, 0.15 for correlated multi-asset portfolios.

    Examples
    --------
    >>> bk = BayesianKelly(kappa=30, f_max=0.25)
    >>> bk.update(win=True)
    >>> bk.update(win=False)
    >>> result = bk.calculate(b=1.33)
    >>> result.position_size
    0.0...
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        kappa: float = 30.0,
        f_max: float = 0.25,
    ) -> None:
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("Prior parameters α₀ and β₀ must be positive.")
        if kappa <= 0:
            raise ValueError("Robustness parameter κ must be positive.")
        if not 0 < f_max <= 1:
            raise ValueError("f_max must be in (0, 1].")

        self._alpha0 = float(alpha0)
        self._beta0 = float(beta0)
        self.kappa = float(kappa)
        self.f_max = float(f_max)

        # Mutable posterior state
        self.alpha: float = float(alpha0)
        self.beta: float = float(beta0)
        self._n_trades: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, win: bool) -> None:
        """
        Incorporate a single trade outcome into the posterior.

        Parameters
        ----------
        win : bool
            True if the trade was profitable, False otherwise.
        """
        if win:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self._n_trades += 1

    def update_batch(self, outcomes: Sequence[bool]) -> None:
        """
        Incorporate a sequence of trade outcomes.

        Parameters
        ----------
        outcomes : sequence of bool
            Ordered sequence of trade results (True = win).
        """
        wins = sum(1 for o in outcomes if o)
        losses = len(outcomes) - wins
        self.alpha += wins
        self.beta += losses
        self._n_trades += len(outcomes)

    def calculate(self, b: float) -> BayesianKellyResult:
        """
        Compute the Bayesian Kelly position size for given odds.

        Implements Sukhov (2026) equation (13):
            f* = max(0, min(f_max, f_base × Φ))

        where:
            f_base = p̄ − (1 − p̄) / b
            Φ      = n_eff / (n_eff + κ)

        Parameters
        ----------
        b : float
            Reward-to-risk ratio (e.g. 1.33 for a 2:1 target / 1.5 stop).

        Returns
        -------
        BayesianKellyResult
            Structured result containing position size and diagnostic fields.
        """
        if b <= 0:
            raise ValueError("Odds ratio b must be positive.")

        n_eff = self.alpha + self.beta
        p_bar = self.alpha / n_eff

        f_base = p_bar - (1.0 - p_bar) / b
        phi = n_eff / (n_eff + self.kappa)
        f_sized = max(0.0, min(self.f_max, f_base * phi))

        # Posterior standard deviation
        p_var = (self.alpha * self.beta) / (n_eff ** 2 * (n_eff + 1.0))
        p_std = math.sqrt(p_var)

        return BayesianKellyResult(
            position_size=f_sized,
            f_base=f_base,
            posterior_mean=p_bar,
            posterior_std=p_std,
            confidence_weight=phi,
            effective_n=n_eff,
            alpha=self.alpha,
            beta=self.beta,
        )

    def reset(self) -> None:
        """Reset the posterior to the original prior."""
        self.alpha = self._alpha0
        self.beta = self._beta0
        self._n_trades = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_trades(self) -> int:
        """Number of trade outcomes observed so far."""
        return self._n_trades

    @property
    def posterior_mean(self) -> float:
        """Current posterior mean of win probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        """Current posterior standard deviation of win probability."""
        n = self.alpha + self.beta
        return math.sqrt(self.alpha * self.beta / (n ** 2 * (n + 1.0)))

    @property
    def credible_interval_95(self) -> tuple[float, float]:
        """
        Approximate 95% credible interval for win probability.

        Uses the Normal approximation: p̄ ± 1.96 × σ_p, clipped to [0, 1].
        For small samples, prefer a Beta quantile calculation.
        """
        p = self.posterior_mean
        s = self.posterior_std
        return (max(0.0, p - 1.96 * s), min(1.0, p + 1.96 * s))

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianKelly("
            f"α={self.alpha:.2f}, β={self.beta:.2f}, "
            f"p̄={self.posterior_mean:.4f}, "
            f"n={self._n_trades}, "
            f"κ={self.kappa}, f_max={self.f_max})"
        )
