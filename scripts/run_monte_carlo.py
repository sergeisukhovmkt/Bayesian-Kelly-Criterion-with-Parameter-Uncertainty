"""
scripts/run_monte_carlo.py
==========================
Reproduce the Monte Carlo simulation study from Section 5 of Sukhov (2026).

Runs 10,000 paths per (p, n) configuration across the full grid:
  p ∈ {0.45, 0.50, 0.55, 0.60}
  n ∈ {10, 25, 50, 100, 500}

Results are printed to stdout and saved as a CSV table.

Usage
-----
    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --paths 1000 --output results/
    python scripts/run_monte_carlo.py --p 0.55 --n 50  # single config
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayesian_kelly.monte_carlo import MonteCarloStudy, run_full_grid, DEFAULT_KAPPA


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian Kelly — Monte Carlo simulation study"
    )
    p.add_argument("--paths", type=int, default=10_000,
                   help="Monte Carlo paths per config (default: 10,000)")
    p.add_argument("--trades", type=int, default=1_000,
                   help="Trades per path (default: 1,000)")
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                   help=f"Robustness parameter κ (default: {DEFAULT_KAPPA})")
    p.add_argument("--p", type=float, default=None,
                   help="Single true_p value (omit for full grid)")
    p.add_argument("--n", type=int, default=None,
                   help="Single estimation window n (omit for full grid)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output", default="results",
                   help="Directory for CSV output (default: results/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBayesian Kelly — Monte Carlo Study")
    print(f"  paths={args.paths:,}  trades={args.trades}  κ={args.kappa}  seed={args.seed}\n")

    # ── Single config mode ───────────────────────────────────────────────
    if args.p is not None or args.n is not None:
        true_p = args.p if args.p is not None else 0.55
        n = args.n if args.n is not None else 50
        study = MonteCarloStudy(
            true_p=true_p,
            b=1.0,
            n_trades=args.trades,
            n_paths=args.paths,
            estimation_window=n,
            kappa=args.kappa,
            seed=args.seed,
        )
        results = study.run(verbose=True)
        return

    # ── Full grid ────────────────────────────────────────────────────────
    p_values = (0.45, 0.50, 0.55, 0.60)
    n_values = (10, 25, 50, 100, 500)

    all_results = run_full_grid(
        p_values=p_values,
        n_values=n_values,
        b=1.0,
        n_trades=args.trades,
        n_paths=args.paths,
        kappa=args.kappa,
        seed=args.seed,
        verbose=True,
    )

    # ── Write CSV ────────────────────────────────────────────────────────
    csv_path = output_dir / "monte_carlo_grid.csv"
    fieldnames = [
        "true_p", "est_window", "strategy",
        "terminal_mean", "terminal_median", "terminal_5th", "terminal_95th",
        "growth_pct", "max_dd_pct", "max_dd_95th_pct",
        "sharpe", "ruin_pct",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for (p, n), res in all_results.items():
            for strat_key, stats in res.stats.items():
                writer.writerow({
                    "true_p":          p,
                    "est_window":      n,
                    "strategy":        stats.name,
                    "terminal_mean":   round(stats.terminal_mean, 4),
                    "terminal_median": round(stats.terminal_median, 4),
                    "terminal_5th":    round(stats.terminal_5th, 4),
                    "terminal_95th":   round(stats.terminal_95th, 4),
                    "growth_pct":      round(stats.growth_mean * 100, 4),
                    "max_dd_pct":      round(stats.max_dd_mean * 100, 2),
                    "max_dd_95th_pct": round(stats.max_dd_95th * 100, 2),
                    "sharpe":          round(stats.sharpe, 4),
                    "ruin_pct":        round(stats.ruin_rate * 100, 2),
                })

    print(f"\nResults saved → {csv_path}")

    # ── Key table: p=0.55, n=50 (replicates Table 1 of paper) ───────────
    key = (0.55, 50)
    if key in all_results:
        print(f"\nTable 1 replication — p = 0.55, n = 50:")
        print(f"{'Strategy':<20} {'Term×':>7} {'g%':>6} {'MDD%':>7} {'Sharpe':>8} {'Ruin%':>7}")
        print("─" * 58)
        for stats in all_results[key].stats.values():
            print(
                f"{stats.name:<20} {stats.terminal_mean:>7.3f} "
                f"{stats.growth_mean*100:>6.2f} "
                f"{stats.max_dd_mean*100:>7.1f} "
                f"{stats.sharpe:>8.3f} "
                f"{stats.ruin_rate*100:>7.2f}"
            )


if __name__ == "__main__":
    main()
