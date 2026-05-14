"""GEM Dual Momentum -- live strategy package.

Selected production cell: C12 (C8 + 2x leverage via MES futures).

Pre-reg: ``directives/Pre-Reg GEM Dual Momentum 2026-05-14.md``.

This package contains:
  * ``live_logic.py`` -- pure-python state machine (testable + parity-able
    against the research function ``research/gem/gem_strategy.gem_returns``).
  * ``sizing.py``     -- target-weight to MES futures + ETF contract translation.
  * ``strategy.py``   -- thin NautilusTrader Strategy wrapping live_logic.
  * ``config.py``     -- TOML-loadable configuration.

The NautilusTrader-side (strategy.py) is a thin orchestration layer; all
trading logic lives in live_logic.py so the parity test can validate
research↔live equivalence WITHOUT booting NT.
"""
