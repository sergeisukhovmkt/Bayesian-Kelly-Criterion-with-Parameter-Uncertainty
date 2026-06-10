"""
scripts/run_empirical.py
========================
Reproduce the empirical validation study from Section 6 of Sukhov (2026).

Requires
--------
- Daily OHLCV data for S&P 500 E-mini Futures, Jan 2000 – Aug 2019.
  Expected path: data/ES_2000_2019.csv  (Investing.com CSV format)

Usage
-----
    python scripts/run_empirical.py
    python scripts/run_empirical.py --data path/to/data.csv --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayesian_kelly.backtest import (
    load_ohlcv,
    ema_atr,
    donchian_signals,
    simulate_trades,
    run_kelly,
    performance,
    ATR_PERIOD,
    CHANNEL_DAYS,
    STOP_MULT,
    TARGET_MULT,
    RR,
    KAPPA,
    ALPHA_0,
    BETA_0,
    F_MAX,
    INITIAL_CAPITAL,
)
from bayesian_kelly.plotting import plot_empirical_validation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian Kelly — empirical validation on ES futures"
    )
    p.add_argument(
        "--data",
        default="data/ES_2000_2019.csv",
        help="Path to OHLCV CSV file (default: data/ES_2000_2019.csv)",
    )
    p.add_argument(
        "--output",
        default="results",
        help="Directory for output figure (default: results/)",
    )
    p.add_argument(
        "--kappa", type=float, default=KAPPA,
        help=f"Robustness parameter κ (default: {KAPPA})",
    )
    p.add_argument(
        "--f-max", type=float, default=F_MAX,
        help=f"Hard position cap (default: {F_MAX})",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip figure generation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = {
    "Full Kelly":     "full",
    "Half Kelly":     "half",
    "Quarter Kelly":  "quarter",
    "Bayesian Kelly": "bayesian",
}


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path} …")
    try:
        dates, closes, highs, lows = load_ohlcv(data_path)
    except FileNotFoundError:
        print(
            f"\n[ERROR] Data file not found: {data_path}\n"
            "  Download daily OHLCV for S&P 500 E-mini Futures from\n"
            "  https://www.investing.com/indices/us-spx-500-futures-historical-data\n"
            "  and place it at the expected path, or pass --data <path>."
        )
        sys.exit(1)

    # ── Compute indicators and signals ───────────────────────────────────
    atr = ema_atr(closes, highs, lows, ATR_PERIOD)
    entries, directions = donchian_signals(closes, CHANNEL_DAYS)
    trades = simulate_trades(
        dates, closes, highs, lows, atr, entries, directions,
        stop_mult=STOP_MULT, target_mult=TARGET_MULT,
    )

    outcomes = np.array([t["outcome"] for t in trades])
    wr = float((outcomes > 0).mean())
    ev = wr * RR - (1.0 - wr)
    kf_emp = wr - (1.0 - wr) / RR

    print(
        f"\nStrategy : Donchian {CHANNEL_DAYS}d L+S  |  "
        f"Stop {STOP_MULT}×ATR  Target {TARGET_MULT}×ATR  RR={RR:.2f}\n"
        f"Data     : {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}  "
        f"({len(dates)/252:.1f} yr)\n"
        f"Trades   : {len(trades)}  |  Win rate: {wr*100:.1f}%  |  "
        f"EV: {ev:+.4f}R  |  Kelly f*: {kf_emp:.4f}"
    )

    # ── Run all strategies ───────────────────────────────────────────────
    caps: dict = {}
    fracs: dict = {}
    perf: dict = {}

    for label, strat in VARIANTS.items():
        cap, frac = run_kelly(
            trades, strat, b=RR,
            kappa=args.kappa, alpha0=ALPHA_0, beta0=BETA_0,
            f_max=args.f_max, initial_capital=INITIAL_CAPITAL,
        )
        caps[label] = cap
        fracs[label] = frac
        perf[label] = performance(cap, initial=INITIAL_CAPITAL)

    # ── Print results table ──────────────────────────────────────────────
    col = "{:<20} {:>10} {:>9} {:>9} {:>9}"
    print(f"\n{col.format('Strategy', 'Terminal×', 'MaxDD%', 'Sharpe', 'Calmar')}")
    print("─" * 62)
    for label, m in perf.items():
        print(
            col.format(
                label,
                f"{m['terminal_x']:.3f}×",
                f"{m['max_dd_pct']:.2f}%",
                f"{m['sharpe']:.3f}",
                f"{m['calmar']:.3f}",
            )
        )

    bk_m = perf["Bayesian Kelly"]
    fk_m = perf["Full Kelly"]
    hk_m = perf["Half Kelly"]

    print(
        f"\nBayesian vs Full  — "
        f"Δterm: {(bk_m['terminal_x']-fk_m['terminal_x'])/max(abs(fk_m['terminal_x']),1e-6)*100:+.1f}%  "
        f"ΔDD: {(bk_m['max_dd_pct']-fk_m['max_dd_pct']):+.1f}pp  "
        f"ΔCalmar: {bk_m['calmar']-fk_m['calmar']:+.3f}"
    )
    print(
        f"Bayesian vs Half  — "
        f"Δterm: {(bk_m['terminal_x']-hk_m['terminal_x'])/max(abs(hk_m['terminal_x']),1e-6)*100:+.1f}%  "
        f"ΔDD: {(bk_m['max_dd_pct']-hk_m['max_dd_pct']):+.1f}pp  "
        f"ΔCalmar: {bk_m['calmar']-hk_m['calmar']:+.3f}"
    )

    # ── Generate figure ──────────────────────────────────────────────────
    if not args.no_plot:
        title_meta = (
            f"Strategy: Donchian {CHANNEL_DAYS}d breakout L+S  |  "
            f"Stop = {STOP_MULT}×ATR({ATR_PERIOD})  |  "
            f"Target = {TARGET_MULT}×ATR({ATR_PERIOD})  |  "
            f"RR = {RR:.2f}  |  N = {len(trades)} trades  |  "
            f"Win rate = {wr*100:.1f}%  |  "
            f"κ = {args.kappa:.0f}  |  f_max = {args.f_max}"
        )
        fig = plot_empirical_validation(
            caps, fracs, perf, trades,
            title_meta=title_meta,
            initial_capital=INITIAL_CAPITAL,
        )
        out_path = output_dir / "fig1_empirical_validation.png"
        fig.savefig(out_path, dpi=300)
        print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
