"""Carver-style trend ladder research.

Adapts the EWMAC-ladder + continuous-forecasting approach from Rob Carver's
*Advanced Futures Trading Strategies* to the FX universe we have data for
(6 pairs: EUR/USD, GBP/USD, AUD/USD, AUD/JPY, USD/CHF, USD/JPY).

Frozen design (no in-sample tuning):
  * EWMAC pairs: (16,64), (32,128), (64,256) — Carver's medium-speed defaults.
  * Forecast scaling factors per Carver: 4.10, 2.65, 1.69 (target abs(mean)=10).
  * Forecast cap: ±20.
  * Forecast diversification multiplier (FDM): 1.4 across the 3 ladder variants.
  * Vol estimate: 36-day EWMA of daily price-change squared.
  * Vol target: 25% annualised (standard Carver default).
  * Trading cost: 1 bp per turn (tight FX spread + slippage proxy).

Reference: research/parabolic_short/ and research/mss_trend/ for the same
hostile-audit structure (skeptical pre-flight, IS/OOS/sanctuary split,
bootstrap CI gate).
"""
