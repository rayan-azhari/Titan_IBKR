"""Fractional Kelly sizing per strategy (V3.7 / L65 + L67).

Computes the Kelly-optimal fraction `f* = mu / sigma^2` for each strategy
from WFO-stitched OOS returns, applies a fractional-Kelly scaling
(default 0.25x for parameter-estimation uncertainty, per MacLean/Thorp/
Ziemba 2010), and gates strategies with too-thin edge from deployment.

Why fractional, not full Kelly:
    - Full Kelly assumes the edge estimate is exact. In practice, OOS
      Sharpe estimates have substantial uncertainty (CI width ~ 1/sqrt(n_folds)).
    - Full Kelly maximises long-run log wealth but has crippling drawdowns.
    - Fractional Kelly (typically 0.25-0.5x full Kelly) trades a small
      fraction of long-run growth for materially lower drawdowns.

Why DSR deflation:
    - Strategy was chosen from many candidates -> sample Sharpe is biased
      upward. Use DSR-deflated Sharpe as the "true" edge estimate.

Reference:
    - Kelly 1956, "A New Interpretation of Information Rate"
    - MacLean, Thorp & Ziemba 2010, "Good and Bad Properties of the
      Kelly Criterion"
    - Bailey & Lopez de Prado 2014, "Deflated Sharpe Ratio"

Usage:

    from titan.research.framework.kelly import compute_kelly_fraction

    kelly = compute_kelly_fraction(
        returns=stitched_oos_returns,
        periods_per_year=252,
        dsr_deflated_sharpe=0.85,  # from titan.research.framework.dsr
        fractional=0.25,
    )
    print(kelly.report())
    if kelly.passes_gate(min_full_kelly=0.05):
        deploy_at_weight = kelly.fractional_weight
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KellyFraction:
    """Per-strategy Kelly fraction analysis.

    Attributes:
        full_kelly: f* = mu / sigma^2 where mu and sigma are ANNUALISED.
            Interpretation: fraction of capital to bet on this strategy
            under log-utility-maximising sizing.
        fractional_weight: fractional_factor * full_kelly. Default 0.25x.
        sample_sharpe: raw OOS Sharpe (annualised).
        deflated_sharpe: DSR-adjusted Sharpe (if provided), else None.
        annual_return: estimated annual return from sample mean.
        annual_vol: annualised volatility (sample std).
        fractional_factor: scaling applied (typically 0.25 or 0.5).
        n_obs: number of return observations.
    """

    full_kelly: float
    fractional_weight: float
    sample_sharpe: float
    deflated_sharpe: float | None
    annual_return: float
    annual_vol: float
    fractional_factor: float
    n_obs: int

    def passes_gate(self, *, min_full_kelly: float = 0.05) -> bool:
        """Default L65/L67 gate: full Kelly must exceed `min_full_kelly`.

        Rationale: if Kelly-optimal sizing is < 5%, the strategy's edge
        is too thin relative to vol to overcome estimation error after
        DSR deflation. Deploying it adds noise without alpha.
        """
        return self.full_kelly >= min_full_kelly

    def report(self) -> str:
        dsr_str = f"{self.deflated_sharpe:+.4f}" if self.deflated_sharpe is not None else "n/a"
        return (
            f"KellyFraction(n_obs={self.n_obs})\n"
            f"  Sample Sharpe (annualised): {self.sample_sharpe:+.4f}\n"
            f"  DSR-deflated Sharpe:        {dsr_str}\n"
            f"  Annual return:              {self.annual_return:+.4%}\n"
            f"  Annual vol:                 {self.annual_vol:.4%}\n"
            f"  Full Kelly (f*):            {self.full_kelly:+.4%}\n"
            f"  Fractional ({self.fractional_factor:.2f}x):     {self.fractional_weight:+.4%}\n"
            f"  Passes gate (f* >= 5%):     {self.passes_gate()}"
        )


def compute_kelly_fraction(
    returns: pd.Series,
    *,
    periods_per_year: int,
    dsr_deflated_sharpe: float | None = None,
    fractional: float = 0.25,
) -> KellyFraction:
    """Compute fractional Kelly weight from a return series.

    Parameters:
        returns: per-bar net returns (OOS preferred).
        periods_per_year: annualisation factor (252 for daily, 6048 for
            FX H1, 1764 for US equity RTH H1).
        dsr_deflated_sharpe: optional DSR-adjusted Sharpe; if provided,
            replaces the sample Sharpe in the Kelly computation (more
            conservative).
        fractional: scaling factor applied to full Kelly. Default 0.25
            (industry standard for parameter-estimation uncertainty).

    Returns:
        KellyFraction dataclass.
    """
    s = returns.dropna()
    n = len(s)
    if n < 30:
        return KellyFraction(
            full_kelly=0.0,
            fractional_weight=0.0,
            sample_sharpe=0.0,
            deflated_sharpe=dsr_deflated_sharpe,
            annual_return=0.0,
            annual_vol=0.0,
            fractional_factor=fractional,
            n_obs=n,
        )

    mu_per = float(s.mean())
    sigma_per = float(s.std(ddof=1))
    annual_return = mu_per * periods_per_year
    annual_vol = sigma_per * np.sqrt(periods_per_year)
    sample_sharpe = (mu_per / sigma_per * np.sqrt(periods_per_year)) if sigma_per > 1e-12 else 0.0

    # Use DSR-deflated Sharpe if provided; otherwise use sample.
    edge_sharpe = dsr_deflated_sharpe if dsr_deflated_sharpe is not None else sample_sharpe
    # Reconstruct mu_annualised from deflated Sharpe at observed vol.
    mu_for_kelly = edge_sharpe * annual_vol

    # Kelly: f* = mu / sigma^2 (both annualised, log-utility convention).
    full_kelly = mu_for_kelly / (annual_vol**2) if annual_vol > 1e-12 else 0.0
    full_kelly = max(0.0, full_kelly)  # cap at 0 (no shorting via negative Kelly here)

    fractional_weight = fractional * full_kelly

    return KellyFraction(
        full_kelly=float(full_kelly),
        fractional_weight=float(fractional_weight),
        sample_sharpe=float(sample_sharpe),
        deflated_sharpe=dsr_deflated_sharpe,
        annual_return=float(annual_return),
        annual_vol=float(annual_vol),
        fractional_factor=fractional,
        n_obs=n,
    )


__all__ = ["KellyFraction", "compute_kelly_fraction"]
