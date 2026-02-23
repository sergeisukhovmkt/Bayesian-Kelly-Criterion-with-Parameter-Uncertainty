"""
Bayesian Kelly Criterion — Empirical Validation on ES Futures
=============================================================
Replicates the experimental framework of Sukhov (2026) on real market data.

Data     : S&P 500 E-mini Futures, daily OHLCV, Jan 2000 – Aug 2019
Strategy : Donchian 20-day channel breakout, long and short
Entry    : Close above 20-day high (long) / below 20-day low (short)
Stop     : 1.5 × ATR(14) from entry
Target   : 2.0 × ATR(14) from entry   →  RR = 1.33

Position sizing methods compared:
  (1) Full Kelly    — plug-in estimator, rolling 50-trade window
  (2) Half Kelly    — f_Full / 2
  (3) Quarter Kelly — f_Full / 4
  (4) Bayesian Kelly — Sukhov (2026), eq. (13), κ = 30, α₀ = β₀ = 1

Reference: Sukhov, S. (2026). Bayesian Kelly Criterion with Parameter
           Uncertainty: A Robust Framework for Position Sizing Under
           Estimation Risk. Working Paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, MultipleLocator
import warnings
warnings.filterwarnings('ignore')


# ── Hyperparameters ──────────────────────────────────────────────────────────

ATR_PERIOD   = 14
CHANNEL_DAYS = 20
STOP_MULT    = 1.5
TARGET_MULT  = 2.0
RR           = TARGET_MULT / STOP_MULT      # 1.333...
EST_WINDOW   = 50                           # rolling estimation window
KAPPA        = 30.0                         # Bayesian robustness parameter
ALPHA_0      = 1.0                          # prior: uniform Beta(1,1)
BETA_0       = 1.0
F_MAX        = 0.25                         # hard position cap
INITIAL      = 100_000.0                    # starting capital

DATA_PATH = '/mnt/user-data/uploads/S_P_500_Futures_Historical_Data.csv'


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ohlcv(path: str):
    """Parse Investing.com CSV export (Date, Price, Open, High, Low, Vol, Chg%)."""
    from datetime import datetime
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as fh:
        for line in fh.readlines()[1:]:
            parts = [p.strip().strip('"') for p in line.strip().split('","')]
            if len(parts) < 5:
                continue
            try:
                rows.append((
                    datetime.strptime(parts[0], '%m/%d/%Y'),
                    float(parts[1].replace(',', '')),   # close
                    float(parts[3].replace(',', '')),   # high
                    float(parts[4].replace(',', '')),   # low
                ))
            except ValueError:
                continue
    rows.sort(key=lambda x: x[0])
    dates  = [r[0] for r in rows]
    closes = np.array([r[1] for r in rows], dtype=float)
    highs  = np.array([r[2] for r in rows], dtype=float)
    lows   = np.array([r[3] for r in rows], dtype=float)
    return dates, closes, highs, lows


# ── Indicators ───────────────────────────────────────────────────────────────

def ema_atr(closes: np.ndarray, highs: np.ndarray,
            lows: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR (equivalent to EMA with α = 1/period)."""
    n  = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
    atr = np.empty(n)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


def donchian_signals(closes: np.ndarray, period: int = 20):
    """
    Generate entry signals for Donchian channel breakout strategy.

    Returns
    -------
    entries   : bool array, True on entry bar
    directions: int array, +1 long / -1 short
    """
    n      = len(closes)
    hi_ch  = np.array([
        closes[max(0, t - period):t].max() if t >= period else closes[0]
        for t in range(n)
    ])
    lo_ch  = np.array([
        closes[max(0, t - period):t].min() if t >= period else closes[0]
        for t in range(n)
    ])
    prev_c  = np.roll(closes, 1)
    prev_hi = np.roll(hi_ch,  1)
    prev_lo = np.roll(lo_ch,  1)

    e_long  = (closes > hi_ch) & (prev_c <= prev_hi)
    e_short = (closes < lo_ch) & (prev_c >= prev_lo)
    e_long[0] = e_short[0] = False

    entries    = e_long | e_short
    directions = np.where(e_long, 1, -1)
    return entries, directions


# ── Trade simulation ─────────────────────────────────────────────────────────

def simulate_trades(
    dates: list,
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    atr:    np.ndarray,
    entries:    np.ndarray,
    directions: np.ndarray,
    stop_mult:   float = STOP_MULT,
    target_mult: float = TARGET_MULT,
) -> list:
    """
    Simulate trade-by-trade outcomes with ATR-based stop and target.

    Each trade is closed by whichever triggers first:
      (a) target hit  → outcome = +RR  (in R units)
      (b) stop hit    → outcome = -1.0
      (c) both same bar → stop assumed (conservative)

    Returns list of dicts with entry_date, exit_date, outcome, win.
    """
    n = len(closes)
    trades: list = []
    in_trade  = False
    direction = 0
    stop = target = entry_price = 0.0
    entry_idx = 0

    for t in range(1, n):
        if not in_trade:
            if entries[t]:
                in_trade    = True
                direction   = int(directions[t])
                entry_price = closes[t]
                entry_idx   = t
                stop   = entry_price - direction * stop_mult   * atr[t]
                target = entry_price + direction * target_mult * atr[t]
        else:
            if direction == 1:
                stop_hit   = lows[t]  <= stop
                target_hit = highs[t] >= target
            else:
                stop_hit   = highs[t] >= stop
                target_hit = lows[t]  <= target

            if stop_hit and target_hit:
                outcome = -1.0                           # conservative: stop
            elif target_hit:
                outcome = target_mult / stop_mult        # = RR
            elif stop_hit:
                outcome = -1.0
            else:
                continue

            trades.append(dict(
                entry_date = dates[entry_idx],
                exit_date  = dates[t],
                outcome    = outcome,
                win        = outcome > 0,
            ))
            in_trade = False

    return trades


# ── Bayesian Kelly (Sukhov 2026, eq. 13) ─────────────────────────────────────

class BayesianKelly:
    """
    Online Bayesian Kelly sizer with Beta(α, β) conjugate prior.

    Parameters
    ----------
    alpha0 : float  Prior successes  (default 1 → uniform)
    beta0  : float  Prior failures   (default 1 → uniform)
    kappa  : float  Robustness parameter κ in eq. (13)
    f_max  : float  Hard position cap
    """

    def __init__(
        self,
        alpha0: float = ALPHA_0,
        beta0:  float = BETA_0,
        kappa:  float = KAPPA,
        f_max:  float = F_MAX,
    ):
        self.alpha = alpha0
        self.beta  = beta0
        self.kappa = kappa
        self.f_max = f_max

    def update(self, win: bool) -> None:
        """Update posterior with a single trade outcome."""
        self.alpha += int(win)
        self.beta  += int(not win)

    def position_size(self, b: float) -> float:
        """
        Compute Bayesian Kelly fraction.

        Implements eq. (13) from Sukhov (2026):
            f* = (p̄ - (1 - p̄)/b) × n_eff / (n_eff + κ)

        where p̄ = α/(α+β)  and  n_eff = α+β.
        """
        n_eff  = self.alpha + self.beta
        p_mean = self.alpha / n_eff
        f_base = p_mean - (1.0 - p_mean) / b
        phi    = n_eff / (n_eff + self.kappa)      # confidence weight
        return max(0.0, min(self.f_max, f_base * phi))

    @property
    def posterior_std(self) -> float:
        """Posterior standard deviation of p ~ Beta(α, β)."""
        n = self.alpha + self.beta
        return ((self.alpha * self.beta) / (n * n * (n + 1))) ** 0.5


# ── Kelly backtest engine ─────────────────────────────────────────────────────

def run_kelly(
    trades:   list,
    strategy: str,           # 'full' | 'half' | 'quarter' | 'bayesian'
    b:        float = RR,
) -> tuple:
    """
    Simulate capital curve trade-by-trade under a given Kelly variant.

    Parameters
    ----------
    trades   : list of trade dicts from simulate_trades()
    strategy : sizing rule identifier
    b        : odds ratio (RR) used in Kelly formula

    Returns
    -------
    capital  : np.ndarray  length n_trades + 1
    fracs    : np.ndarray  length n_trades
    """
    capital = [INITIAL]
    fracs   = []
    buf     = []                        # rolling binary outcome buffer
    bk      = BayesianKelly()

    for tr in trades:
        win = bool(tr['win'])
        c   = capital[-1]

        # ── size the position ──
        if strategy == 'bayesian':
            f = bk.position_size(b)

        else:
            if len(buf) < 5:            # cold start: sit out first 5 trades
                capital.append(c)
                fracs.append(0.0)
                bk.update(win)
                buf.append(int(win))
                continue

            p_hat  = np.mean(buf[-EST_WINDOW:])
            f_base = p_hat - (1.0 - p_hat) / b
            denom  = {'full': 1, 'half': 2, 'quarter': 4}[strategy]
            f      = max(0.0, min(F_MAX, f_base / denom))

        fracs.append(f)
        r = tr['outcome']               # +RR or -1.0
        capital.append(c * (1.0 + f * r))

        bk.update(win)
        buf.append(int(win))

    return np.array(capital), np.array(fracs)


# ── Performance metrics ───────────────────────────────────────────────────────

def performance(capital: np.ndarray) -> dict:
    """Compute standard risk-adjusted performance metrics."""
    n      = len(capital) - 1
    ret    = np.diff(capital) / capital[:-1]
    rm     = np.maximum.accumulate(capital)
    dd     = (rm - capital) / rm
    sharpe = ret.mean() / (ret.std() + 1e-10) * n ** 0.5  # annualised by trade count
    calmar = (capital[-1] / INITIAL - 1) / (dd.max() + 1e-10)
    return dict(
        terminal_x  = capital[-1] / INITIAL,
        max_dd_pct  = dd.max() * 100,
        sharpe      = sharpe,
        calmar      = calmar,
        n_trades    = n,
    )


# ── Run all strategies ────────────────────────────────────────────────────────

print('Loading data …')
dates, closes, highs, lows = load_ohlcv(DATA_PATH)
atr = ema_atr(closes, highs, lows, ATR_PERIOD)

entries, directions = donchian_signals(closes, CHANNEL_DAYS)
trades = simulate_trades(dates, closes, highs, lows, atr, entries, directions)

outcomes = np.array([t['outcome'] for t in trades])
wr       = (outcomes > 0).mean()
ev       = wr * RR - (1.0 - wr)
kf_emp   = wr - (1.0 - wr) / RR

print(f'\nStrategy  : Donchian {CHANNEL_DAYS}d L+S  |  Stop {STOP_MULT}×ATR  Target {TARGET_MULT}×ATR  RR={RR:.2f}')
print(f'Data      : {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}  ({len(dates)/252:.1f} yr)')
print(f'Trades    : {len(trades)}  |  Win rate: {wr*100:.1f}%  |  EV: {ev:+.4f}R  |  Kelly f*: {kf_emp:.4f}')

VARIANTS = {
    'Full Kelly':    'full',
    'Half Kelly':    'half',
    'Quarter Kelly': 'quarter',
    'Bayesian Kelly':'bayesian',
}

caps, fracs, perf = {}, {}, {}
for label, strat in VARIANTS.items():
    c, f       = run_kelly(trades, strat, b=RR)
    caps[label]  = c
    fracs[label] = f
    perf[label]  = performance(c)

print(f'\n{"Strategy":<20} {"Terminal×":>10} {"MaxDD%":>9} {"Sharpe":>9} {"Calmar":>9}')
print('─' * 60)
for label, m in perf.items():
    print(f'{label:<20} {m["terminal_x"]:>9.3f}× {m["max_dd_pct"]:>8.2f}%'
          f' {m["sharpe"]:>9.3f} {m["calmar"]:>9.3f}')

bk_m = perf['Bayesian Kelly']
fk_m = perf['Full Kelly']
hk_m = perf['Half Kelly']
print(f'\nBayesian vs Full  — growth: {bk_m["terminal_x"]/max(fk_m["terminal_x"],1e-6)*100:.0f}%'
      f'  DD: −{(1-bk_m["max_dd_pct"]/max(fk_m["max_dd_pct"],1e-6))*100:.0f}%'
      f'  Calmar: {bk_m["calmar"]-fk_m["calmar"]:+.3f}')
print(f'Bayesian vs Half  — growth: {bk_m["terminal_x"]/max(hk_m["terminal_x"],1e-6)*100:.0f}%'
      f'  DD: −{(1-bk_m["max_dd_pct"]/max(hk_m["max_dd_pct"],1e-6))*100:.0f}%'
      f'  Calmar: {bk_m["calmar"]-hk_m["calmar"]:+.3f}')


# ── Figure — academic style ───────────────────────────────────────────────────

PALETTE = {
    'Full Kelly':    '#d62728',   # strong red
    'Half Kelly':    '#ff7f0e',   # orange
    'Quarter Kelly': '#bcbd22',   # olive/gold
    'Bayesian Kelly':'#2ca02c',   # green
}
DASHES = {
    'Full Kelly':    (4, 2),
    'Half Kelly':    (6, 2, 1, 2),
    'Quarter Kelly': (2, 2),
    'Bayesian Kelly': None,       # solid
}
LINEWIDTH = {k: (1.8 if 'Bayesian' in k else 1.2) for k in PALETTE}

exit_dates = [t['exit_date'] for t in trades]
t_axis     = [trades[0]['entry_date']] + exit_dates   # length = n_trades + 1

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'axes.linewidth':   0.7,
    'xtick.major.width':0.7,
    'ytick.major.width':0.7,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'axes.grid':        True,
    'grid.color':       '#cccccc',
    'grid.linewidth':   0.4,
    'grid.linestyle':   ':',
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

fig = plt.figure(figsize=(12, 14))
gs  = gridspec.GridSpec(
    3, 2,
    figure    = fig,
    hspace    = 0.38,
    wspace    = 0.26,
    top       = 0.93,
    bottom    = 0.06,
    left      = 0.09,
    right     = 0.97,
)

# ── Panel I: Capital growth (log scale) ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])

for label, cap in caps.items():
    x = t_axis[:len(cap)]
    lw = LINEWIDTH[label]
    ds = DASHES[label]
    if ds:
        ax1.plot(x, cap / INITIAL, color=PALETTE[label], lw=lw,
                 dashes=ds, label=label, zorder=3, alpha=0.9)
    else:
        ax1.plot(x, cap / INITIAL, color=PALETTE[label], lw=lw,
                 label=label, zorder=4, alpha=1.0)

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.set_ylabel('Wealth relative to initial capital', labelpad=6)
ax1.set_title('Panel I — Capital growth (log scale)', loc='left', fontweight='bold')
ax1.axhline(1.0, color='#555555', lw=0.6, ls='--', zorder=2)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9,
           edgecolor='#aaaaaa', ncol=2)
ax1.set_xlim(t_axis[0], t_axis[-1])

# ── Panel II: Drawdown ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])

for label, cap in caps.items():
    rm = np.maximum.accumulate(cap)
    dd = -(rm - cap) / rm * 100
    x  = t_axis[:len(cap)]
    lw = LINEWIDTH[label]
    ds = DASHES[label]
    if ds:
        ax2.plot(x, dd, color=PALETTE[label], lw=lw,
                 dashes=ds, label=label, zorder=3, alpha=0.9)
    else:
        ax2.plot(x, dd, color=PALETTE[label], lw=lw,
                 label=label, zorder=4, alpha=1.0)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.set_ylabel('Drawdown from peak (%)', labelpad=6)
ax2.set_title('Panel II — Drawdown from peak', loc='left', fontweight='bold')
ax2.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#aaaaaa', ncol=2)
ax2.set_xlim(t_axis[0], t_axis[-1])
ax2.axhline(0, color='#555555', lw=0.6, ls='--', zorder=2)

# ── Panel III: Kelly fraction per trade ───────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])

for label in ['Full Kelly', 'Bayesian Kelly']:
    frc = fracs[label]
    x   = exit_dates[:len(frc)]
    lw  = LINEWIDTH[label]
    ds  = DASHES[label]
    if ds:
        ax3.plot(x, frc, color=PALETTE[label], lw=lw,
                 dashes=ds, label=label, alpha=0.85)
    else:
        ax3.plot(x, frc, color=PALETTE[label], lw=lw,
                 label=label, alpha=1.0)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_major_locator(mdates.YearLocator(4))
ax3.set_ylabel('Position size (fraction)', labelpad=6)
ax3.set_title('Panel III — Kelly fraction per trade', loc='left', fontweight='bold')
ax3.axhline(F_MAX, color='#888888', lw=0.5, ls=':', label=f'Cap = {F_MAX}')
ax3.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#aaaaaa')
ax3.set_xlim(exit_dates[0], exit_dates[-1])
ax3.set_ylim(bottom=0)

# ── Panel IV: Risk–return scatter ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])

for label, m in perf.items():
    ax4.scatter(m['max_dd_pct'], m['terminal_x'],
                s     = 70,
                color = PALETTE[label],
                zorder= 5,
                edgecolors='#333333',
                linewidths=0.5,
                label = label)
    # offset labels to avoid overlap
    offsets = {
        'Full Kelly':    ( 3,  -6),
        'Half Kelly':    ( 3,   3),
        'Quarter Kelly': (-2,   5),
        'Bayesian Kelly':( 3,  -6),
    }
    dx, dy = offsets[label]
    ax4.annotate(
        label,
        xy         = (m['max_dd_pct'], m['terminal_x']),
        xytext     = (dx, dy),
        textcoords = 'offset points',
        fontsize   = 8,
        color      = PALETTE[label],
    )

ax4.set_xlabel('Maximum drawdown (%)', labelpad=5)
ax4.set_ylabel('Terminal wealth (×)', labelpad=5)
ax4.set_title('Panel IV — Risk–return', loc='left', fontweight='bold')

# iso-Calmar reference lines
for calmar_val, ls in [(0.5,'--'),(1.0,':'),(1.5,':')]:
    dd_range = np.linspace(5, 95, 200)
    tw_range = 1 + calmar_val * dd_range / 100
    ax4.plot(dd_range, tw_range, color='#aaaaaa', lw=0.6, ls=ls, zorder=1)
    ax4.text(dd_range[-1] - 2, tw_range[-1] + 0.02,
             f'Calmar={calmar_val}', fontsize=7, color='#999999', ha='right')

ax4.set_xlim(left=0)
ax4.set_ylim(bottom=0.5)

# ── Figure-level title and caption ───────────────────────────────────────────
fig.text(
    0.5, 0.972,
    'Bayesian Kelly Criterion: Empirical Validation on S&P 500 E-mini Futures (2000–2019)',
    ha='center', va='top', fontsize=11, fontweight='bold',
)
fig.text(
    0.5, 0.956,
    (f'Strategy: Donchian {CHANNEL_DAYS}d breakout L+S  |  '
     f'Stop = {STOP_MULT}×ATR({ATR_PERIOD})  |  Target = {TARGET_MULT}×ATR({ATR_PERIOD})  |  '
     f'RR = {RR:.2f}  |  '
     f'N = {len(trades)} trades  |  Win rate = {wr*100:.1f}%  |  '
     f'κ = {KAPPA:.0f}  |  f_max = {F_MAX}'),
    ha='center', va='top', fontsize=8.5, color='#444444',
)

out_path = '/mnt/user-data/outputs/bayesian_kelly_academic.png'
plt.savefig(out_path, dpi=300)
plt.close()
print(f'\nFigure saved → {out_path}')