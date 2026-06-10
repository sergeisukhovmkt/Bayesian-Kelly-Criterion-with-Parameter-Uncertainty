# Changelog

All notable changes to this project are documented here.

---

## [1.0.0] — 2026-03-xx

### Added

**Core library (`bayesian_kelly/`)**
- `criterion.py` — `BayesianKelly` class with full online Bayesian update loop
  - `update(win)` and `update_batch(outcomes)` for sequential / batch incorporation of trade outcomes
  - `calculate(b)` returning a structured `BayesianKellyResult` dataclass
  - `credible_interval_95` property for diagnostic reporting
  - `reset()` to restore the prior
- `backtest.py` — trade-level backtesting engine
  - `load_ohlcv` for Investing.com CSV format with optional date filtering
  - `ema_atr` (Wilder-smoothed ATR)
  - `donchian_signals` (Donchian channel breakout, long + short)
  - `simulate_trades` with ATR-based stop and target, conservative same-bar tie-breaking
  - `run_kelly` for all four strategy variants (Full / Half / Quarter / Bayesian)
  - `performance` returning terminal wealth, max drawdown, Sharpe, and Calmar
- `monte_carlo.py` — Monte Carlo study infrastructure
  - `MonteCarloStudy` class with configurable (p, b, T, n_paths, κ) grid
  - `run_full_grid` convenience function replicating the Section 5 experiment
  - `StrategyStats` and `MonteCarloResults` dataclasses
- `plotting.py` — four publication-quality figures
  - `plot_empirical_validation` — four-panel figure matching Figure 1 of the paper
  - `plot_monte_carlo_distributions` — KDE terminal wealth overlay
  - `plot_confidence_weight` — Φ vs n_eff for multiple κ values
  - `plot_posterior_evolution` — posterior mean and credible interval over trades

**Scripts**
- `scripts/run_empirical.py` — CLI replication of Section 6 (ES futures backtest)
- `scripts/run_monte_carlo.py` — CLI replication of Section 5 (full simulation grid)

**Tests**
- `tests/test_criterion.py` — 20+ unit tests for `BayesianKelly`
- `tests/test_backtest.py` — unit tests for ATR, Donchian, simulation, and performance metrics

**Documentation**
- `README.md` with installation, quick-start, replication guide, and limitations
- `pyproject.toml` with full metadata and optional dependencies
- `CHANGELOG.md` (this file)
- `LICENSE` (MIT)
