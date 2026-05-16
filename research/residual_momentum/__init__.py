"""A1 -- Residual momentum (Blitz, Huij & Martens, JEmpF 2011).

Pre-registered in directives/Pre-Reg A1 Residual Momentum 2026-05-15.md.
Class: titan.research.framework.typology.StrategyClass.CROSS_ASSET_MOMENTUM
(cross-sectional equity variant; same defaults).

Mechanism: regress each S&P 500 stock's daily excess return on the
Fama-French 3 factors (Mkt-RF, SMB, HML) over a trailing 36-month window;
take the residual; compute its skip-1 cumulative 12-month return divided
by residual std → "residual t-stat". Rank cross-sectionally; long top
quintile, short bottom quintile; equal-weighted, dollar-neutral.
"""
