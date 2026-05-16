"""B2 -- Carver EWMAC trend ensemble.

Pre-registered in ``directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md``.

Multi-speed EWMAC (exponentially-weighted moving average crossover) trend
following on the 24-commodity IBKR roll-stitched M1 universe. Designed to
test whether a many-speed ensemble dissolves the residual plateau
brittleness that B4c's 3-window TSMOM ensemble (33.9% mitigation) could
not eliminate.
"""

from research.ewmac.ewmac_strategy import (
    EwmacConfig,
    compute_ewmac_forecast,
    ewmac_assert_causal,
    ewmac_returns,
)

__all__ = [
    "EwmacConfig",
    "compute_ewmac_forecast",
    "ewmac_assert_causal",
    "ewmac_returns",
]
