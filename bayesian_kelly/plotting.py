"""
bayesian_kelly.plotting
=======================
Publication-quality figures for the Bayesian Kelly Criterion study.

Reproduces Figure 1 from Sukhov (2026) and provides additional
diagnostic plots for Monte Carlo analysis.

All functions return matplotlib Figure objects so callers can
save, display, or embed them as needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE = {
    "Full Kelly":     "#d62728",
    "Half Kelly":     "#ff7f0e",
    "Quarter Kelly":  "#bcbd22",
    "Bayesian Kelly": "#2ca02c",
}

DASHES = {
    "Full Kelly":     (4, 2),
    "Half Kelly":     (6, 2, 1, 2),
    "Quarter Kelly":  (2, 2),
    "Bayesian Kelly": None,
}

LINEWIDTH = {k: (1.8 if "Bayesian" in k else 1.2) for k in PALETTE}

ACADEMIC_RC = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.4,
    "grid.linestyle":    ":",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}


def _apply_style() -> None:
    plt.rcParams.update(ACADEMIC_RC)


def _plot_line(ax, x, y, label: str, **kwargs) -> None:
    ds = DASHES[label]
    lw = LINEWIDTH[label]
    color = PALETTE[label]
    zorder = 4 if "Bayesian" in label else 3
    alpha = 1.0 if "Bayesian" in label else 0.9
    if ds:
        ax.plot(x, y, color=color, lw=lw, dashes=ds,
                label=label, zorder=zorder, alpha=alpha, **kwargs)
    else:
        ax.plot(x, y, color=color, lw=lw,
                label=label, zorder=zorder, alpha=alpha, **kwargs)


# ---------------------------------------------------------------------------
# Figure 1 — Empirical validation (4-panel)
# ---------------------------------------------------------------------------

def plot_empirical_validation(
    caps: Dict[str, np.ndarray],
    fracs: Dict[str, np.ndarray],
    perf: Dict[str, Dict],
    trades: List[Dict],
    title_meta: str = "",
    initial_capital: float = 100_000.0,
) -> plt.Figure:
    """
    Reproduce the four-panel empirical validation figure (Sukhov 2026, Fig. 1).

    Parameters
    ----------
    caps : dict label → capital array (length n+1)
    fracs : dict label → position-size array (length n)
    perf : dict label → performance metrics dict
    trades : list of trade dicts (for date axis)
    title_meta : str
        Subtitle line shown below the main figure title.
    initial_capital : float

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()

    exit_dates = [t["exit_date"] for t in trades]
    t_axis = [trades[0]["entry_date"]] + exit_dates

    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.38, wspace=0.26,
        top=0.93, bottom=0.06,
        left=0.09, right=0.97,
    )

    # ── Panel I: Capital (log scale) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for label, cap in caps.items():
        x = t_axis[: len(cap)]
        _plot_line(ax1, x, cap / initial_capital, label)

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.set_ylabel("Wealth relative to initial capital", labelpad=6)
    ax1.set_title("Panel I — Capital growth (log scale)", loc="left", fontweight="bold")
    ax1.axhline(1.0, color="#555555", lw=0.6, ls="--", zorder=2)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9,
               edgecolor="#aaaaaa", ncol=2)
    ax1.set_xlim(t_axis[0], t_axis[-1])

    # ── Panel II: Drawdown ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    for label, cap in caps.items():
        rm = np.maximum.accumulate(cap)
        dd = -(rm - cap) / (rm + 1e-15) * 100
        x = t_axis[: len(cap)]
        _plot_line(ax2, x, dd, label)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.set_ylabel("Drawdown from peak (%)", labelpad=6)
    ax2.set_title("Panel II — Drawdown from peak", loc="left", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9,
               edgecolor="#aaaaaa", ncol=2)
    ax2.set_xlim(t_axis[0], t_axis[-1])
    ax2.axhline(0, color="#555555", lw=0.6, ls="--", zorder=2)

    # ── Panel III: Kelly fraction per trade ──────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    for label in ["Full Kelly", "Bayesian Kelly"]:
        frc = fracs[label]
        x = exit_dates[: len(frc)]
        _plot_line(ax3, x, frc, label)

    f_max = max(fracs["Bayesian Kelly"].max(), 0.01)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(4))
    ax3.set_ylabel("Position size (fraction)", labelpad=6)
    ax3.set_title("Panel III — Kelly fraction per trade", loc="left", fontweight="bold")
    ax3.axhline(0.25, color="#888888", lw=0.5, ls=":", label="Cap = 0.25")
    ax3.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#aaaaaa")
    ax3.set_xlim(exit_dates[0], exit_dates[-1])
    ax3.set_ylim(bottom=0)

    # ── Panel IV: Risk–return scatter ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    label_offsets = {
        "Full Kelly":     (3, -6),
        "Half Kelly":     (3,  3),
        "Quarter Kelly":  (-2, 5),
        "Bayesian Kelly": (3, -6),
    }
    for label, m in perf.items():
        ax4.scatter(
            m["max_dd_pct"], m["terminal_x"],
            s=70, color=PALETTE[label], zorder=5,
            edgecolors="#333333", linewidths=0.5,
        )
        dx, dy = label_offsets[label]
        ax4.annotate(
            label,
            xy=(m["max_dd_pct"], m["terminal_x"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, color=PALETTE[label],
        )

    # Calmar iso-lines
    for calmar_val, ls in [(0.5, "--"), (1.0, ":"), (1.5, ":")]:
        dd_range = np.linspace(5, 95, 200)
        tw_range = 1 + calmar_val * dd_range / 100
        ax4.plot(dd_range, tw_range, color="#aaaaaa", lw=0.6, ls=ls, zorder=1)
        ax4.text(dd_range[-1] - 2, tw_range[-1] + 0.02,
                 f"Calmar={calmar_val}", fontsize=7, color="#999999", ha="right")

    ax4.set_xlabel("Maximum drawdown (%)", labelpad=5)
    ax4.set_ylabel("Terminal wealth (×)", labelpad=5)
    ax4.set_title("Panel IV — Risk–return", loc="left", fontweight="bold")
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0.5)

    # ── Figure-level title ───────────────────────────────────────────────
    fig.text(0.5, 0.972,
             "Bayesian Kelly Criterion: Empirical Validation on S&P 500 E-mini Futures (2000–2019)",
             ha="center", va="top", fontsize=11, fontweight="bold")
    if title_meta:
        fig.text(0.5, 0.956, title_meta,
                 ha="center", va="top", fontsize=8.5, color="#444444")

    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Monte Carlo terminal wealth distributions
# ---------------------------------------------------------------------------

def plot_monte_carlo_distributions(
    results_by_strategy: Dict[str, np.ndarray],
    true_p: float,
    b: float,
    n_trades: int,
) -> plt.Figure:
    """
    Plot kernel-density terminal wealth distributions for all strategies.

    Parameters
    ----------
    results_by_strategy : dict label → 1-D array of terminal wealth multiples
    true_p, b, n_trades : simulation parameters for the subtitle

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()

    try:
        from scipy.stats import gaussian_kde
        use_kde = True
    except ImportError:
        use_kde = False

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, terminals in results_by_strategy.items():
        color = PALETTE.get(label, "#333333")
        if use_kde:
            lo, hi = np.percentile(terminals, [1, 99])
            xs = np.linspace(lo, hi, 500)
            kde = gaussian_kde(terminals, bw_method=0.15)
            ax.plot(xs, kde(xs), color=color, lw=1.6,
                    label=f"{label}  (med={np.median(terminals):.2f}×)")
            ax.axvline(np.median(terminals), color=color, lw=0.8, ls="--", alpha=0.6)
        else:
            ax.hist(terminals, bins=80, density=True, alpha=0.4,
                    color=color, label=label)

    ax.axvline(1.0, color="#555555", lw=0.8, ls=":", label="Break-even (1×)")
    ax.set_xlabel("Terminal wealth (× initial capital)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Terminal Wealth Distributions — p = {true_p:.2f}, b = {b:.2f}, T = {n_trades}",
        fontweight="bold",
    )
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Confidence weight as a function of n_eff
# ---------------------------------------------------------------------------

def plot_confidence_weight(
    kappa_values: Tuple[float, ...] = (10.0, 30.0, 50.0),
    n_max: int = 300,
) -> plt.Figure:
    """
    Plot the confidence weight Φ = n_eff / (n_eff + κ) for several κ values.

    Illustrates how the uncertainty discount evolves with sample size.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    n_eff = np.linspace(0, n_max, 500)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for kappa, color in zip(kappa_values, colors):
        phi = n_eff / (n_eff + kappa)
        ax.plot(n_eff, phi, color=color, lw=1.6, label=f"κ = {kappa:.0f}")
        # mark half-Kelly point
        ax.axvline(kappa, color=color, lw=0.6, ls="--", alpha=0.5)

    ax.axhline(0.5, color="#888888", lw=0.5, ls=":", label="Half-Kelly (Φ = 0.5)")
    ax.axhline(1.0, color="#aaaaaa", lw=0.5, ls=":")
    ax.set_xlabel("Effective sample size $n_{\\mathrm{eff}}$ (α + β)")
    ax.set_ylabel("Confidence weight  Φ = $n_{\\mathrm{eff}}$ / ($n_{\\mathrm{eff}}$ + κ)")
    ax.set_title(
        "Confidence Weight vs Effective Sample Size  [Sukhov (2026), Eq. 13]",
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Posterior evolution over trades
# ---------------------------------------------------------------------------

def plot_posterior_evolution(
    win_sequence: List[bool],
    alpha0: float = 1.0,
    beta0: float = 1.0,
    true_p: Optional[float] = None,
) -> plt.Figure:
    """
    Visualise how the Beta posterior mean and credible interval evolve
    as trades are observed.

    Parameters
    ----------
    win_sequence : list of bool
        Ordered trade outcomes.
    alpha0, beta0 : float
        Prior hyperparameters.
    true_p : float, optional
        If known, draw a horizontal reference line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()

    from bayesian_kelly.criterion import BayesianKelly

    bk = BayesianKelly(alpha0=alpha0, beta0=beta0)
    ns, means, lo95, hi95 = [], [], [], []

    for w in win_sequence:
        bk.update(w)
        lo, hi = bk.credible_interval_95
        ns.append(bk.n_trades)
        means.append(bk.posterior_mean)
        lo95.append(lo)
        hi95.append(hi)

    ns = np.array(ns)
    means = np.array(means)
    lo95 = np.array(lo95)
    hi95 = np.array(hi95)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(ns, lo95, hi95, alpha=0.2, color="#1f77b4", label="95% credible interval")
    ax.plot(ns, means, color="#1f77b4", lw=1.6, label="Posterior mean p̄")
    if true_p is not None:
        ax.axhline(true_p, color="#d62728", lw=1.0, ls="--", label=f"True p = {true_p:.2f}")

    ax.set_xlabel("Number of trades observed")
    ax.set_ylabel("Win probability estimate")
    ax.set_title("Bayesian Posterior Evolution — Beta Conjugate Update", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa")
    fig.tight_layout()
    return fig
