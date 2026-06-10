"""
bayesian_kelly.monte_carlo
==========================
Monte Carlo engine for comparing Kelly position-sizing variants.

Reproduces the simulation study in Section 5 of Sukhov (2026):
  - 10,000 paths per configuration
  - Strategies: Full Kelly, Half Kelly, Quarter Kelly, Bayesian Kelly
  - Performance metrics: terminal wealth, growth rate, max drawdown,
    Sharpe ratio, ruin rate

Usage
-----
    from bayesian_kelly.monte_carlo import MonteCarloStudy

    study = MonteCarloStudy(true_p=0.55, b=1.0, n_trades=1000, n_paths=10_000)
    results = study.run()
    print(results.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bayesian_kelly.criterion import BayesianKelly


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_KAPPA = 30.0
DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
DEFAULT_F_MAX = 0.25
DEFAULT_N_PATHS = 10_000
DEFAULT_N_TRADES = 1_000
DEFAULT_RUIN_THRESHOLD = 0.10  # capital below 10% of initial


# ---------------------------------------------------------------------------
# Path-level simulation
# ---------------------------------------------------------------------------

def _simulate_path(
    true_p: float,
    b: float,
    n_trades: int,
    strategy: str,
    estimation_window: int,
    kappa: float,
    alpha0: float,
    beta0: float,
    f_max: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Simulate a single capital path under a given Kelly variant.

    Returns
    -------
    terminal_wealth : float   (as multiple of initial capital)
    growth_rate     : float   (per-trade geometric growth rate)
    max_drawdown    : float   (maximum fractional drawdown)
    ruin_flag       : float   (1.0 if ruined, else 0.0)
    """
    capital = np.empty(n_trades + 1)
    capital[0] = 1.0

    outcomes = rng.random(n_trades) < true_p  # True = win
    payoffs = np.where(outcomes, b, -1.0)

    bk = BayesianKelly(alpha0=alpha0, beta0=beta0, kappa=kappa, f_max=f_max)
    wins_buf: List[int] = []

    for t in range(n_trades):
        win = bool(outcomes[t])
        c = capital[t]

        # --- position size ---
        if strategy == "bayesian":
            f = bk.calculate(b).position_size
        else:
            if len(wins_buf) < max(5, estimation_window // 2):
                # cold start: sit out
                capital[t + 1] = c
                bk.update(win)
                wins_buf.append(int(win))
                continue

            window = wins_buf[-estimation_window:]
            p_hat = float(np.mean(window))
            f_base = p_hat - (1.0 - p_hat) / b
            divisor = {"full": 1, "half": 2, "quarter": 4}[strategy]
            f = max(0.0, min(f_max, f_base / divisor))

        capital[t + 1] = c * (1.0 + f * payoffs[t])
        bk.update(win)
        wins_buf.append(int(win))

    terminal = capital[-1]
    g = (1.0 / n_trades) * np.log(max(terminal, 1e-10))
    running_max = np.maximum.accumulate(capital)
    mdd = float(np.max((running_max - capital) / (running_max + 1e-15)))
    ruin = float(terminal < DEFAULT_RUIN_THRESHOLD)

    return terminal, g, mdd, ruin


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Summary statistics for one strategy across all Monte Carlo paths."""
    name: str
    terminal_mean: float
    terminal_median: float
    terminal_5th: float
    terminal_95th: float
    growth_mean: float
    max_dd_mean: float
    max_dd_95th: float
    sharpe: float
    ruin_rate: float
    n_paths: int

    def __str__(self) -> str:
        return (
            f"{self.name:<18} "
            f"term={self.terminal_mean:6.3f}× "
            f"g={self.growth_mean*100:5.2f}% "
            f"MDD={self.max_dd_mean*100:5.1f}% "
            f"Sharpe={self.sharpe:5.3f} "
            f"Ruin={self.ruin_rate*100:5.2f}%"
        )


@dataclass
class MonteCarloResults:
    """Collected results for all strategies in a simulation study."""
    config: Dict
    stats: Dict[str, StrategyStats]
    elapsed_seconds: float

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "",
            "=" * 74,
            "  Monte Carlo Study — Bayesian Kelly Criterion",
            "=" * 74,
            f"  true p = {cfg['true_p']:.2f}  |  b = {cfg['b']:.2f}  |  "
            f"T = {cfg['n_trades']}  |  paths = {cfg['n_paths']:,}  |  "
            f"κ = {cfg['kappa']}",
            "-" * 74,
            f"  {'Strategy':<18} {'Term×':>7} {'g%':>6} {'MDD%':>6} "
            f"{'Sharpe':>7} {'Ruin%':>7}",
            "-" * 74,
        ]
        for s in self.stats.values():
            lines.append(
                f"  {s.name:<18} {s.terminal_mean:>7.3f} "
                f"{s.growth_mean*100:>6.2f} "
                f"{s.max_dd_mean*100:>6.1f} "
                f"{s.sharpe:>7.3f} "
                f"{s.ruin_rate*100:>7.2f}"
            )
        lines += [
            "-" * 74,
            f"  Elapsed: {self.elapsed_seconds:.1f}s",
            "=" * 74,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main study class
# ---------------------------------------------------------------------------

class MonteCarloStudy:
    """
    Configures and runs a Monte Carlo comparison of Kelly variants.

    Parameters
    ----------
    true_p : float
        True (unknown to the sizer) win probability.
    b : float
        Reward-to-risk ratio (odds).
    n_trades : int
        Number of trades per simulated path.
    n_paths : int
        Number of Monte Carlo paths.
    estimation_window : int
        Rolling window for the plug-in estimators.
    kappa : float
        Robustness parameter for Bayesian Kelly.
    alpha0, beta0 : float
        Prior hyperparameters for Bayesian Kelly.
    f_max : float
        Hard position cap applied to all strategies.
    seed : int, optional
        Random seed for reproducibility.
    strategies : list of str, optional
        Subset of strategies to run. Defaults to all four variants.
    """

    STRATEGY_LABELS = {
        "full": "Full Kelly",
        "half": "Half Kelly",
        "quarter": "Quarter Kelly",
        "bayesian": "Bayesian Kelly",
    }

    def __init__(
        self,
        true_p: float,
        b: float = 1.0,
        n_trades: int = DEFAULT_N_TRADES,
        n_paths: int = DEFAULT_N_PATHS,
        estimation_window: int = 50,
        kappa: float = DEFAULT_KAPPA,
        alpha0: float = DEFAULT_ALPHA0,
        beta0: float = DEFAULT_BETA0,
        f_max: float = DEFAULT_F_MAX,
        seed: Optional[int] = None,
        strategies: Optional[List[str]] = None,
    ) -> None:
        self.true_p = true_p
        self.b = b
        self.n_trades = n_trades
        self.n_paths = n_paths
        self.estimation_window = estimation_window
        self.kappa = kappa
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.f_max = f_max
        self.seed = seed
        self.strategies = strategies or list(self.STRATEGY_LABELS.keys())

    def run(self, verbose: bool = True) -> MonteCarloResults:
        """
        Execute the Monte Carlo study.

        Parameters
        ----------
        verbose : bool
            Print progress messages.

        Returns
        -------
        MonteCarloResults
        """
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        all_stats: Dict[str, StrategyStats] = {}

        for strat_key in self.strategies:
            label = self.STRATEGY_LABELS[strat_key]
            if verbose:
                print(f"  Running {label:20s} ({self.n_paths:,} paths)…", end="", flush=True)

            terminals = np.empty(self.n_paths)
            growths = np.empty(self.n_paths)
            mdds = np.empty(self.n_paths)
            ruins = np.empty(self.n_paths)

            for i in range(self.n_paths):
                terminals[i], growths[i], mdds[i], ruins[i] = _simulate_path(
                    true_p=self.true_p,
                    b=self.b,
                    n_trades=self.n_trades,
                    strategy=strat_key,
                    estimation_window=self.estimation_window,
                    kappa=self.kappa,
                    alpha0=self.alpha0,
                    beta0=self.beta0,
                    f_max=self.f_max,
                    rng=rng,
                )

            sharpe = growths.mean() / (growths.std() + 1e-10)

            all_stats[strat_key] = StrategyStats(
                name=label,
                terminal_mean=float(terminals.mean()),
                terminal_median=float(np.median(terminals)),
                terminal_5th=float(np.percentile(terminals, 5)),
                terminal_95th=float(np.percentile(terminals, 95)),
                growth_mean=float(growths.mean()),
                max_dd_mean=float(mdds.mean()),
                max_dd_95th=float(np.percentile(mdds, 95)),
                sharpe=float(sharpe),
                ruin_rate=float(ruins.mean()),
                n_paths=self.n_paths,
            )

            if verbose:
                print(f" done → term={terminals.mean():.3f}× MDD={mdds.mean()*100:.1f}%")

        elapsed = time.perf_counter() - t0

        config = dict(
            true_p=self.true_p,
            b=self.b,
            n_trades=self.n_trades,
            n_paths=self.n_paths,
            kappa=self.kappa,
            alpha0=self.alpha0,
            beta0=self.beta0,
            f_max=self.f_max,
        )

        results = MonteCarloResults(
            config=config,
            stats=all_stats,
            elapsed_seconds=elapsed,
        )

        if verbose:
            print(results.summary())

        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_full_grid(
    p_values=(0.45, 0.50, 0.55, 0.60),
    n_values=(10, 25, 50, 100, 500),
    b: float = 1.0,
    n_trades: int = DEFAULT_N_TRADES,
    n_paths: int = DEFAULT_N_PATHS,
    kappa: float = DEFAULT_KAPPA,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[tuple, MonteCarloResults]:
    """
    Run the full p × n simulation grid from Section 5 of Sukhov (2026).

    Parameters
    ----------
    p_values : tuple of float
        True win probabilities to test.
    n_values : tuple of int
        Initial estimation window sizes to test (affects Bayesian prior
        effective sample size at cold start).
    b : float
        Odds ratio.
    n_trades : int
        Trades per path.
    n_paths : int
        Paths per configuration.
    kappa : float
        Robustness parameter.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict mapping (p, n) → MonteCarloResults
    """
    results = {}
    total = len(p_values) * len(n_values)
    idx = 0

    for p in p_values:
        for n in n_values:
            idx += 1
            if verbose:
                print(f"\n[{idx}/{total}] p={p:.2f}  n={n}")
            study = MonteCarloStudy(
                true_p=p,
                b=b,
                n_trades=n_trades,
                n_paths=n_paths,
                estimation_window=n,
                kappa=kappa,
                seed=seed + idx,
                verbose=verbose,
            )
            results[(p, n)] = study.run(verbose=verbose)

    return results
