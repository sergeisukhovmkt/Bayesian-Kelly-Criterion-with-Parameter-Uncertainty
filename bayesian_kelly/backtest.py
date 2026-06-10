"""
bayesian_kelly.backtest
=======================
Trade-level backtesting engine for the empirical validation study
(Section 6 of Sukhov, 2026).

The module provides:
  - OHLCV data loading (Investing.com CSV format)
  - ATR indicator (Wilder smoothing)
  - Donchian channel breakout signal generation
  - Trade simulation with ATR-scaled stops and targets
  - Multi-strategy capital curve simulation
  - Performance metric computation

Usage
-----
    from bayesian_kelly.backtest import (
        load_ohlcv, ema_atr, donchian_signals,
        simulate_trades, run_kelly, performance
    )

    dates, closes, highs, lows = load_ohlcv("data/ES_2000_2019.csv")
    atr = ema_atr(closes, highs, lows)
    entries, directions = donchian_signals(closes)
    trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)
    cap, fracs = run_kelly(trades, "bayesian")
    stats = performance(cap)
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bayesian_kelly.criterion import BayesianKelly

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
CHANNEL_DAYS = 20
STOP_MULT = 1.5
TARGET_MULT = 2.0
RR = TARGET_MULT / STOP_MULT          # 1.333...
EST_WINDOW = 50                        # rolling window for plug-in estimators
KAPPA = 30.0
ALPHA_0 = 1.0
BETA_0 = 1.0
F_MAX = 0.25
INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv(
    path: str | Path,
    date_format: str = "%m/%d/%Y",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse daily OHLCV data from an Investing.com CSV export.

    Expected column order:
        Date, Close/Price, Open, High, Low, Volume, Change%

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    date_format : str
        strptime format string for the Date column.
    start, end : datetime, optional
        Filter to this date range (inclusive).

    Returns
    -------
    dates : list of datetime
    closes, highs, lows : np.ndarray of float64
    """
    rows: List[Tuple] = []

    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh.readlines()[1:]:
            parts = [p.strip().strip('"') for p in raw.strip().split('","')]
            if len(parts) < 5:
                parts = [p.strip().strip('"') for p in raw.strip().split(",")]
            if len(parts) < 5:
                continue
            try:
                dt = datetime.strptime(parts[0], date_format)
                close = float(parts[1].replace(",", ""))
                high = float(parts[3].replace(",", ""))
                low = float(parts[4].replace(",", ""))
                rows.append((dt, close, high, low))
            except (ValueError, IndexError):
                continue

    rows.sort(key=lambda x: x[0])

    if start:
        rows = [r for r in rows if r[0] >= start]
    if end:
        rows = [r for r in rows if r[0] <= end]

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    dates = [r[0] for r in rows]
    closes = np.array([r[1] for r in rows], dtype=np.float64)
    highs = np.array([r[2] for r in rows], dtype=np.float64)
    lows = np.array([r[3] for r in rows], dtype=np.float64)

    return dates, closes, highs, lows


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema_atr(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    """
    Wilder-smoothed Average True Range (equivalent to EMA with α = 1/period).

    Parameters
    ----------
    closes, highs, lows : np.ndarray
        Daily OHLCV arrays of the same length.
    period : int
        ATR lookback period (default 14).

    Returns
    -------
    atr : np.ndarray
    """
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = np.empty(n)
    atr[period - 1] = tr[:period].mean()

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    atr[:period - 1] = atr[period - 1]
    return atr


def donchian_signals(
    closes: np.ndarray,
    period: int = CHANNEL_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate long/short entry signals for a Donchian channel breakout.

    A long signal fires when today's close exceeds the prior `period`-bar
    high (breakout above resistance).  A short signal fires when today's
    close falls below the prior `period`-bar low.

    Parameters
    ----------
    closes : np.ndarray
        Daily close prices.
    period : int
        Channel lookback in bars.

    Returns
    -------
    entries : bool np.ndarray
        True on bars where an entry signal is generated.
    directions : int np.ndarray
        +1 for long, -1 for short.
    """
    n = len(closes)
    hi_ch = np.array(
        [
            closes[max(0, t - period) : t].max() if t >= period else closes[0]
            for t in range(n)
        ]
    )
    lo_ch = np.array(
        [
            closes[max(0, t - period) : t].min() if t >= period else closes[0]
            for t in range(n)
        ]
    )

    prev_c = np.roll(closes, 1)
    prev_hi = np.roll(hi_ch, 1)
    prev_lo = np.roll(lo_ch, 1)

    e_long = (closes > hi_ch) & (prev_c <= prev_hi)
    e_short = (closes < lo_ch) & (prev_c >= prev_lo)
    e_long[0] = e_short[0] = False

    entries = e_long | e_short
    directions = np.where(e_long, 1, -1)
    return entries, directions


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_trades(
    dates: List[datetime],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    entries: np.ndarray,
    directions: np.ndarray,
    stop_mult: float = STOP_MULT,
    target_mult: float = TARGET_MULT,
) -> List[Dict]:
    """
    Simulate trade-by-trade outcomes using ATR-based stop and target levels.

    Each trade is closed at whichever event fires first:
      (a) target hit  → outcome = +RR  (in R units, e.g. 1.33)
      (b) stop hit    → outcome = −1.0
      (c) both same bar → stop is assumed (conservative bias)

    Parameters
    ----------
    dates, closes, highs, lows, atr : arrays of equal length
    entries : bool array, True on signal bars
    directions : int array, +1 long / -1 short
    stop_mult, target_mult : ATR multipliers

    Returns
    -------
    trades : list of dicts with keys
        entry_date, exit_date, outcome, win, entry_idx, exit_idx
    """
    n = len(closes)
    rr = target_mult / stop_mult
    trades: List[Dict] = []

    in_trade = False
    direction = 0
    stop = target = entry_price = 0.0
    entry_idx = 0

    for t in range(1, n):
        if not in_trade:
            if entries[t]:
                in_trade = True
                direction = int(directions[t])
                entry_price = closes[t]
                entry_idx = t
                stop = entry_price - direction * stop_mult * atr[t]
                target = entry_price + direction * target_mult * atr[t]
        else:
            if direction == 1:
                stop_hit = lows[t] <= stop
                target_hit = highs[t] >= target
            else:
                stop_hit = highs[t] >= stop
                target_hit = lows[t] <= target

            if stop_hit and target_hit:
                outcome = -1.0  # conservative: assume stop triggered first
            elif target_hit:
                outcome = rr
            elif stop_hit:
                outcome = -1.0
            else:
                continue

            trades.append(
                dict(
                    entry_date=dates[entry_idx],
                    exit_date=dates[t],
                    outcome=outcome,
                    win=outcome > 0,
                    entry_idx=entry_idx,
                    exit_idx=t,
                )
            )
            in_trade = False

    return trades


# ---------------------------------------------------------------------------
# Kelly backtest engine
# ---------------------------------------------------------------------------

def run_kelly(
    trades: List[Dict],
    strategy: str,
    b: float = RR,
    kappa: float = KAPPA,
    alpha0: float = ALPHA_0,
    beta0: float = BETA_0,
    f_max: float = F_MAX,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a capital curve under a given Kelly variant.

    Parameters
    ----------
    trades : list of trade dicts from simulate_trades()
    strategy : str
        One of 'full', 'half', 'quarter', 'bayesian'.
    b : float
        Reward-to-risk ratio.
    kappa, alpha0, beta0, f_max : float
        Bayesian Kelly parameters (used for 'bayesian' strategy;
        f_max also caps plug-in strategies).
    initial_capital : float

    Returns
    -------
    capital : np.ndarray, shape (n_trades + 1,)
    fracs   : np.ndarray, shape (n_trades,)
    """
    if strategy not in ("full", "half", "quarter", "bayesian"):
        raise ValueError(f"Unknown strategy '{strategy}'.")

    capital_list: List[float] = [initial_capital]
    fracs_list: List[float] = []
    wins_buf: List[int] = []

    bk = BayesianKelly(alpha0=alpha0, beta0=beta0, kappa=kappa, f_max=f_max)

    for tr in trades:
        win = bool(tr["win"])
        c = capital_list[-1]

        if strategy == "bayesian":
            f = bk.calculate(b).position_size
        else:
            if len(wins_buf) < 5:
                # cold start: preserve capital, update state
                capital_list.append(c)
                fracs_list.append(0.0)
                bk.update(win)
                wins_buf.append(int(win))
                continue

            p_hat = float(np.mean(wins_buf[-EST_WINDOW:]))
            f_base = p_hat - (1.0 - p_hat) / b
            divisor = {"full": 1, "half": 2, "quarter": 4}[strategy]
            f = max(0.0, min(f_max, f_base / divisor))

        fracs_list.append(f)
        capital_list.append(c * (1.0 + f * tr["outcome"]))

        bk.update(win)
        wins_buf.append(int(win))

    return np.array(capital_list), np.array(fracs_list)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance(
    capital: np.ndarray,
    initial: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute standard risk-adjusted performance metrics from a capital array.

    Parameters
    ----------
    capital : np.ndarray
        Capital levels, starting from initial (length = n_trades + 1).
    initial : float, optional
        Override initial capital for return computation.

    Returns
    -------
    dict with keys:
        terminal_x, total_return_pct, max_dd_pct, sharpe, calmar, n_trades
    """
    if initial is None:
        initial = float(capital[0])

    n = len(capital) - 1
    returns = np.diff(capital) / (capital[:-1] + 1e-15)
    running_max = np.maximum.accumulate(capital)
    drawdowns = (running_max - capital) / (running_max + 1e-15)

    max_dd = float(drawdowns.max())
    terminal_x = float(capital[-1] / initial)
    total_ret = terminal_x - 1.0
    sharpe = (
        returns.mean() / (returns.std() + 1e-10) * (n ** 0.5)
        if n > 0
        else 0.0
    )
    calmar = total_ret / (max_dd + 1e-10)

    return dict(
        terminal_x=terminal_x,
        total_return_pct=total_ret * 100,
        max_dd_pct=max_dd * 100,
        sharpe=sharpe,
        calmar=calmar,
        n_trades=n,
    )
