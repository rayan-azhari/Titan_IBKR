"""
position_sizing.py

Fractional Kelly position sizing with volatility targeting and hard cap.

The meta-labeller's probability is more useful as a continuous signal than a
binary filter. Fractional Kelly maps it to a portfolio fraction while respecting:
  1. Vol-targeting: no single trade contributes more than vol_target_pct to
     daily portfolio volatility.
  2. Hard cap: maximum position as a fraction of portfolio.

Notes:
  - Full Kelly assumes sequential independent bets -- not satisfied in practice.
  - Quarter-Kelly (fraction=0.25) is the standard conservative institutional choice.
  - Kelly can be negative (output = 0, never short your own signal confidence).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class WinLossTracker:
    """Running tracker of trade returns for dynamic Kelly W/L ratio estimation."""

    window: int = 200
    prior_wl_ratio: float = 1.5   # conservative prior before enough data

    _wins: List[float] = field(default_factory=list, repr=False)
    _losses: List[float] = field(default_factory=list, repr=False)

    def record(self, return_pct: float):
        """Record a completed trade return (positive or negative)."""
        if return_pct > 0:
            self._wins.append(return_pct)
            if len(self._wins) > self.window:
                self._wins.pop(0)
        elif return_pct < 0:
            self._losses.append(abs(return_pct))
            if len(self._losses) > self.window:
                self._losses.pop(0)

    @property
    def ratio(self) -> float:
        """Mean win return / mean loss return. Falls back to prior if insufficient."""
        if len(self._wins) < 10 or len(self._losses) < 10:
            return self.prior_wl_ratio
        return float(np.mean(self._wins) / np.mean(self._losses))

    @property
    def n_trades(self) -> int:
        return len(self._wins) + len(self._losses)


class FractionalKelly:
    """
    Position sizer using fractional Kelly with volatility targeting.

    f* = fraction * (p * b - q) / b
    where:
        p = probability of success (upper barrier hit)
        q = 1 - p
        b = win/loss ratio (mean_win_return / mean_loss_return)
        fraction = Kelly fraction (default 0.25 = quarter-Kelly)
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_position_pct: float = 0.05,
        min_edge: float = 0.02,
        vol_target_pct: float = 0.01,
    ):
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.min_edge = min_edge
        self.vol_target_pct = vol_target_pct

    def size(
        self,
        prob_success: float,
        win_loss_ratio: float,
        portfolio_value: float,
        instrument_daily_vol: float,
    ) -> float:
        """
        Compute dollar position size.

        Returns 0.0 if below edge threshold or Kelly is negative.
        """
        edge = prob_success - 0.5
        if edge < self.min_edge:
            return 0.0

        p = prob_success
        q = 1.0 - p
        b = max(win_loss_ratio, 1e-6)

        kelly_f = (p * b - q) / b
        if kelly_f <= 0:
            return 0.0

        fractional_f = self.fraction * kelly_f
        kelly_size = fractional_f * portfolio_value

        if instrument_daily_vol > 1e-9:
            vol_cap = (self.vol_target_pct * portfolio_value) / instrument_daily_vol
        else:
            vol_cap = kelly_size

        hard_cap = self.max_position_pct * portfolio_value

        return float(min(kelly_size, vol_cap, hard_cap))

    def shares(
        self,
        prob_success: float,
        win_loss_ratio: float,
        portfolio_value: float,
        instrument_daily_vol: float,
        price: float,
    ) -> float:
        """Convenience: returns number of shares/contracts."""
        dollar_size = self.size(
            prob_success, win_loss_ratio, portfolio_value, instrument_daily_vol
        )
        if price <= 0:
            return 0.0
        return dollar_size / price
