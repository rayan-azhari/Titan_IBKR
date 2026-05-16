"""VRP v2 (Vol Risk Premium) -- percentile-gate redesign of E1.

Pre-registered in directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md.
Class: titan.research.framework.typology.StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY.

Mitigations applied vs E1:
    L26 -- percentile-rolling regime gates (not bare thresholds).
    L25 -- new class with relaxed MC default (P(MaxDD>50%) < 10%).
"""
