# Bayesian-Kelly-Criterion-with-Parameter-Uncertainty

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
