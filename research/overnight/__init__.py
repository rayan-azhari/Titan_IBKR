"""G4 -- Overnight session decomposition (SPY-only v1).

Pre-registered in directives/Pre-Reg G4 Overnight Session Decomposition 2026-05-15.md.
Class: titan.research.framework.typology.StrategyClass.INTRADAY_MICROSTRUCTURE.

Tests Lou-Polk-Skouras (JFE 2019): overnight session (close[t-1]->open[t])
earns positive expected return; intraday session (open[t]->close[t]) earns
flat/negative. Audit: does an overnight-only long-SPY strategy survive the
5-axis matrix on retail-implementable SPY at ETF costs?
"""
