"""
bayesian_kelly
==============
Bayesian Kelly Criterion with Parameter Uncertainty.

A rigorous framework for position sizing under estimation risk,
as developed in Sukhov (2026).

Key components
--------------
BayesianKelly
    Core online position sizer with Beta conjugate prior.

MonteCarloStudy
    Monte Carlo engine for strategy comparison.

backtest
    Trade simulation and empirical validation utilities.

plotting
    Publication-quality figures.

Quick start
-----------
>>> from bayesian_kelly import BayesianKelly
>>> bk = BayesianKelly(kappa=30, f_max=0.25)
>>> for win in [True, False, True, True, False]:
...     bk.update(win)
>>> result = bk.calculate(b=1.33)
>>> print(f"Position size: {result.position_size:.4f}")
>>> print(f"Posterior mean: {result.posterior_mean:.4f}")
>>> print(f"Confidence: {result.confidence_weight:.4f}")

References
----------
Sukhov, S. (2026). Bayesian Kelly Criterion with Parameter Uncertainty:
  A Robust Framework for Position Sizing Under Estimation Risk.
  Working Paper. https://ssrn.com/abstract=6542019
"""

from bayesian_kelly.criterion import BayesianKelly, BayesianKellyResult
from bayesian_kelly.monte_carlo import MonteCarloStudy, MonteCarloResults

__version__ = "1.0.0"
__author__ = "Sergei Sukhov"
__email__ = "s.sukhov@mmrls.com"
__all__ = [
    "BayesianKelly",
    "BayesianKellyResult",
    "MonteCarloStudy",
    "MonteCarloResults",
]
