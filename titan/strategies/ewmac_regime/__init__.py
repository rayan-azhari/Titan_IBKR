"""I1 v2 -- Multi-feature HMM regime gate + EWMAC ensemble.

Live wrapper for the audit-verdict cell I1v2 C6_smoothed
(`research/ewmac/run_i1v2_audit.py`, pre-reg in
`directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md`).

SHADOW DEPLOYMENT STATUS (2026-05-17):
- Scaffolded but NOT yet wired into the live `v37_live` STRATEGY_SET.
- Live wiring + parity test + cron-driven regime panel update are
  scheduled for the next iteration (subsequent session).
- L65 single-strategy and joint ruin gates: PASS at 5% weight.
- L67 portfolio inclusion verdict: PORTFOLIO_CONDITIONAL (unchanged).

Public:
    EwmacRegimeStrategy
    EwmacRegimeConfig
"""

from titan.strategies.ewmac_regime.strategy import (
    EwmacRegimeConfig,
    EwmacRegimeStrategy,
)

__all__ = ["EwmacRegimeConfig", "EwmacRegimeStrategy"]
