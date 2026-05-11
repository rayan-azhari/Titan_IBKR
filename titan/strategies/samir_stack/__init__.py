"""Samir-Stack: regime-gated 40/60 leveraged-equity + bond strategy.

Production live class. Wraps the research-validated regime classifier
(6 indicators + optional causal HMM) with NautilusTrader execution.

Research lineage: research/samir_stack/
WFO 5-fold + 12mo sanctuary: 5/5 folds positive, mean Calmar 0.655,
sanctuary Calmar 1.05.
"""

from titan.strategies.samir_stack.strategy import (
    SamirStackConfig,
    SamirStackStrategy,
)

__all__ = ["SamirStackConfig", "SamirStackStrategy"]
