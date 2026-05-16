"""D2 -- Commodity futures carry (BGR 2019, 22-commodity universe).

Pre-registered in directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md.
Class: titan.research.framework.typology.StrategyClass.CARRY.

Cross-sectional long-short portfolio: long top quintile of commodities
by carry (log F_M1 / F_M2), short bottom quintile. Monthly rebalance,
equal-weight within each leg, dollar-neutral between legs.
"""
