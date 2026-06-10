# Bayesian Kelly Criterion with Parameter Uncertainty

**A Robust Framework for Position Sizing Under Estimation Risk**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/SSRN-6542019-orange)](https://ssrn.com/abstract=6195358)

> Sukhov, S. (2026). *Bayesian Kelly Criterion with Parameter Uncertainty: A Robust Framework for Position Sizing Under Estimation Risk.* Working Paper. [SSRN 6542019](https://ssrn.com/abstract=6542019)

---

## Overview

The classical Kelly criterion provides an optimal position size that maximises long-run geometric growth, but only when the true win probability *p* is known exactly. In practice, *p* must be estimated from finite samples — and that estimation error, if ignored, produces systematic overbetting, catastrophic drawdowns, and ruin.

This repository implements the Bayesian regularisation developed in Sukhov (2026). The core formula (Eq. 13) replaces the plug-in Kelly fraction with a posterior-weighted estimate:

$$f^* = \left(\bar{p} - \frac{1-\bar{p}}{b}\right) \cdot \frac{n_{\text{eff}}}{n_{\text{eff}} + \kappa}$$

where $\bar{p} = \alpha / (\alpha + \beta)$ is the posterior mean under a Beta conjugate prior, $n_{\text{eff}} = \alpha + \beta$ is the effective sample size, and $\kappa > 0$ is a robustness parameter calibrated via simulation.

The confidence weight $n_{\text{eff}} / (n_{\text{eff}} + \kappa)$ has natural limiting behaviour:

| Regime | Weight | Behaviour |
|--------|--------|-----------|
| $n_{\text{eff}} = 0$ | 0 | No data → no bet |
| $n_{\text{eff}} = \kappa$ | 0.5 | Half-Kelly equivalent |
| $n_{\text{eff}} \to \infty$ | 1 | Converges to full Kelly |

---

## Key Results

### Monte Carlo (Section 5 of paper) — *p* = 0.55, *n* = 50, 10,000 paths

| Strategy | Terminal × | Growth | Max DD | Sharpe | Ruin |
|---|---:|---:|---:|---:|---:|
| Full Kelly | 2.84× | 10.4% | 62.3% | 0.51 | 18.7% |
| Half Kelly | 2.21× | 7.9% | 31.2% | 0.72 | 4.2% |
| Quarter Kelly | 1.68× | 5.2% | 18.1% | 0.68 | 0.9% |
| **Bayesian Kelly** | **2.47×** | **9.1%** | **24.6%** | **0.89** | **0.8%** |

Bayesian Kelly captures 87% of Full Kelly's growth, cuts maximum drawdown by 60%, reduces ruin risk by 96%, and achieves the highest Sharpe ratio of all variants.

### Empirical Validation (Section 6) — S&P 500 E-mini Futures, 2000–2019

Strategy: Donchian 20-day channel breakout, long+short, stop = 1.5×ATR(14), target = 2.0×ATR(14), *N* = 297 trades, $\hat{p}$ = 47.8%, RR = 1.33.

| Strategy | Terminal × | Max DD | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| Full Kelly | 0.88× | 88.1% | 1.04 | −0.13 |
| Half Kelly | 1.59× | 59.1% | 0.97 | 0.99 |
| Quarter Kelly | 1.47× | 32.0% | 0.97 | 1.48 |
| **Bayesian Kelly** | **1.79×** | **50.6%** | **1.10** | **1.56** |

Bayesian Kelly achieves the highest Calmar ratio across all variants and produces structurally shallower drawdowns, particularly during the cold-start phase (first 50 trades).

![Empirical validation figure](results/fig1_empirical_validation.png)

---

## Repository Structure

```
bayesian-kelly/
│
├── bayesian_kelly/             # Core library
│   ├── __init__.py             # Public API
│   ├── criterion.py            # BayesianKelly class (Eq. 13)
│   ├── backtest.py             # OHLCV loading, trade simulation, Kelly engine
│   ├── monte_carlo.py          # Monte Carlo study infrastructure
│   └── plotting.py             # Publication-quality figures
│
├── scripts/
│   ├── run_empirical.py        # Reproduce Section 6 (ES futures backtest)
│   └── run_monte_carlo.py      # Reproduce Section 5 (simulation grid)
│
├── tests/
│   ├── conftest.py
│   ├── test_criterion.py       # Unit tests for BayesianKelly
│   └── test_backtest.py        # Unit tests for backtest utilities
│
├── data/                       # Place ES futures CSV here (see below)
├── results/                    # Output figures and CSVs
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ssukhov/bayesian-kelly.git
cd bayesian-kelly
pip install -r requirements.txt
```

No build step is required — the library is used directly from source.

---

## Quick Start

### 1. Live position sizing

```python
from bayesian_kelly import BayesianKelly

# Initialise with uniform prior; κ=30 is recommended for swing trading
bk = BayesianKelly(alpha0=1.0, beta0=1.0, kappa=30, f_max=0.25)

# Feed historical trade outcomes (True = win, False = loss)
historical = [True, False, True, True, False, True, False, True, True, False]
bk.update_batch(historical)

# Size the next position for b = 1.33 (e.g. 2:1.5 target:stop)
result = bk.calculate(b=1.33)

print(f"Position size : {result.position_size:.4f}")
print(f"Posterior mean: {result.posterior_mean:.4f}  ±  {result.posterior_std:.4f}")
print(f"95% CI        : [{result.confidence_weight:.4f}]  confidence weight")
print(f"Effective n   : {result.effective_n:.1f}")
```

```
Position size : 0.0541
Posterior mean: 0.6154  ±  0.1285
95% CI        : [0.2941]  confidence weight
Effective n   : 12.0
```

### 2. Updating in real time

```python
# After each trade, update and re-size
bk.update(win=True)
next_size = bk.calculate(b=1.33).position_size
```

The update runs in O(1): only the two Beta parameters (α, β) are maintained.

### 3. Monte Carlo study

```python
from bayesian_kelly.monte_carlo import MonteCarloStudy

study = MonteCarloStudy(
    true_p=0.55,
    b=1.0,
    n_trades=1_000,
    n_paths=10_000,
    kappa=30,
    seed=42,
)
results = study.run()
print(results.summary())
```

Or run the full replication grid from the command line:

```bash
python scripts/run_monte_carlo.py --paths 10000 --output results/
```

---

## Reproducing the Paper

### Section 5 — Monte Carlo simulation grid

```bash
python scripts/run_monte_carlo.py
```

Runs all 20 configurations (*p* ∈ {0.45, 0.50, 0.55, 0.60} × *n* ∈ {10, 25, 50, 100, 500}), prints the Table 1 replication, and writes `results/monte_carlo_grid.csv`.

For a quick test with fewer paths:

```bash
python scripts/run_monte_carlo.py --paths 1000 --p 0.55 --n 50
```

### Section 6 — Empirical validation on ES futures

**Data setup.** Download daily OHLCV for S&P 500 E-mini Futures (January 2000 – August 2019) from [Investing.com](https://www.investing.com/indices/us-spx-500-futures-historical-data) and save as:

```
data/ES_2000_2019.csv
```

Then run:

```bash
python scripts/run_empirical.py
```

This produces `results/fig1_empirical_validation.png` (a four-panel figure matching Figure 1 of the paper) and prints the Table 2 replication to stdout.

Custom data path or κ:

```bash
python scripts/run_empirical.py --data /path/to/futures.csv --kappa 30 --f-max 0.25
```

---

## Running Tests

```bash
pytest tests/ -v
```

Or with coverage:

```bash
pytest tests/ -v --cov=bayesian_kelly --cov-report=term-missing
```

The test suite covers:
- Parameter validation and error handling
- Bayesian update correctness (sequential vs batch equivalence)
- Convergence of posterior mean to true *p*
- Monotonicity of position size with effective sample size
- ATR, Donchian, and trade simulation correctness
- Performance metric calculations

---

## Prior Selection Guide

Prior choice encodes beliefs about *p* before any trades are observed.

| Prior | (α₀, β₀) | Effective observations | Use when |
|---|---|---|---|
| Conservative | (10, 10) | 20 | Strategy unproven, uncertain regime |
| **Neutral (default)** | **(1, 1)** | **2** | **Most applications** |
| Aggressive | (0.5, 0.5) | 1 | Strong confidence in backtested edge |

---

## Robustness Parameter κ

κ controls how quickly confidence weight approaches 1 (full Kelly).

| Trading style | κ | Half-Kelly reached at |
|---|---|---|
| High-frequency (>100 trades/day) | 10 | 10 trades |
| Swing trading (1–10 trades/day) | **30** | **30 trades** |
| Position trading (<1 trade/day) | 50 | 50 trades |

Calibrate κ via historical simulation to minimise maximum drawdown while maintaining >80% of optimal growth (Section 7.2 of paper).

---

## Position Limits

Always impose hard caps (Section 7.3):

```python
bk = BayesianKelly(f_max=0.25)   # single asset
bk = BayesianKelly(f_max=0.15)   # correlated portfolio
```

For circuit-breaker logic during drawdown periods, instantiate a second BayesianKelly with `f_max=0.10` and switch to it programmatically when the account drawdown exceeds a threshold.

---

## Limitations

Three caveats from the paper apply directly to this implementation:

1. **The uncertainty discount is a robust heuristic, not a pure E[ln W] result.** The growth function is linear in *p*, so the regularisation reflects finite-horizon ruin risk and implicit risk aversion beyond log utility — not a derivation from first principles.

2. **κ is calibrated by simulation, not derived analytically.** Sensitivity to this parameter remains an open research question; the method's advantage over Half-Kelly narrows at lower κ values.

3. **The empirical validation uses a single instrument (ES) and a single rule.** The evidence supports the proposed mechanism; it is not a universal quantitative claim about outperformance across all markets and strategies.

In stable, mature, high-frequency environments with a well-characterised edge, Half-Kelly remains a defensible and simpler alternative (Section 6.3).

---

## Citation

```bibtex
@unpublished{Sukhov2026BayesianKelly,
  author = {Sukhov, Sergei},
  title  = {Bayesian Kelly Criterion with Parameter Uncertainty:
             A Robust Framework for Position Sizing Under Estimation Risk},
  year   = {2026},
  note   = {Working Paper. SSRN 6542019},
  url    = {https://ssrn.com/abstract=6542019},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
