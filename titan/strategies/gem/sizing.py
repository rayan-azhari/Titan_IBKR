"""GEM contract / share sizing.

Converts the strategy's target weights (continuous fractions in
[0, max_leverage]) into discrete contract counts and ETF share counts
for live execution.

Two execution vehicles supported:

  * **MES (CME Micro E-mini S&P 500 futures)** for the SPY leg.
    1 MES contract = $5 × ES front-month price. Margin is much smaller
    than notional, providing the leverage operationally. Costs are
    modelled as 1 tick of slippage + commission per round trip.

  * **ETFs** (SPY, EFA, IEF) for non-leveraged exposure.
    Used when ``execution_mode = "etf"`` or when the user's account
    doesn't have futures permissions. Margin / leverage is via Reg-T
    rather than the futures clearinghouse.

The strategy is agnostic to execution -- target weights output by
``GemLiveLogic.current_weights`` are the same in both cases. The
sizing layer translates the weights to specific contract / share
quantities given the current NAV and market prices.

References:
  * directives/IBKR & NautilusTrader API Reference.md -- MES instrument ID
  * directives/Cost Model Audit 2026-05-11.md -- COST_CME_FUTURES_LIQUID
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# CME Micro E-mini S&P 500 multiplier (USD per index point).
MES_MULTIPLIER_USD_PER_POINT: float = 5.0


@dataclass(frozen=True)
class SizingDecision:
    """One row of the sizing layer's output -- per-instrument target quantity."""

    symbol: str  # e.g. "MES", "SPY", "EFA", "IEF"
    target_quantity: float  # MES contracts (futures) or shares (ETFs)
    rounded_quantity: int  # integer for live execution
    target_notional_usd: float  # notional in USD
    target_weight: float  # the weight from the strategy
    execution_vehicle: Literal["MES", "SPY", "EFA", "IEF"]


def size_with_mes(
    target_weights: dict[str, float],
    nav_usd: float,
    es_price: float,
    efa_price: float,
    ief_price: float,
) -> list[SizingDecision]:
    """Translate target weights into MES contracts (SPY leg) + ETF shares (EFA / IEF).

    The strategy's SPY weight is executed via MES futures: the contract
    count is rounded to the nearest integer. EFA and IEF legs are
    executed as ETF shares at full integer share count.

    Parameters
    ----------
    target_weights:
        Dict from ``GemLiveLogic.current_weights()`` with keys SPY / EFA / IEF.
    nav_usd:
        Current account NAV in USD.
    es_price:
        Current ES (or MES) front-month index price -- used for MES sizing.
    efa_price, ief_price:
        Current ETF share prices for EFA and IEF respectively.

    Returns:
    -------
    A list of SizingDecision objects, one per non-zero target. Zero-weight
    legs are omitted.
    """
    if nav_usd <= 0:
        raise ValueError("nav_usd must be positive")
    if es_price <= 0 or efa_price <= 0 or ief_price <= 0:
        raise ValueError("prices must be positive")

    decisions: list[SizingDecision] = []

    # SPY leg -> MES contracts.
    w_spy = float(target_weights.get("SPY", 0.0))
    if w_spy != 0.0:
        target_notional = nav_usd * w_spy
        contract_notional = MES_MULTIPLIER_USD_PER_POINT * es_price
        raw_contracts = target_notional / contract_notional
        # Round half to nearest int (banker's rounding via math.floor + 0.5 fallback).
        rounded = (
            int(math.floor(raw_contracts + 0.5))
            if raw_contracts >= 0
            else int(math.ceil(raw_contracts - 0.5))
        )
        decisions.append(
            SizingDecision(
                symbol="MES",
                target_quantity=raw_contracts,
                rounded_quantity=rounded,
                target_notional_usd=target_notional,
                target_weight=w_spy,
                execution_vehicle="MES",
            )
        )

    # EFA leg -> ETF shares.
    w_efa = float(target_weights.get("EFA", 0.0))
    if w_efa != 0.0:
        target_notional = nav_usd * w_efa
        raw_shares = target_notional / efa_price
        rounded = (
            int(math.floor(raw_shares + 0.5))
            if raw_shares >= 0
            else int(math.ceil(raw_shares - 0.5))
        )
        decisions.append(
            SizingDecision(
                symbol="EFA",
                target_quantity=raw_shares,
                rounded_quantity=rounded,
                target_notional_usd=target_notional,
                target_weight=w_efa,
                execution_vehicle="EFA",
            )
        )

    # IEF leg -> ETF shares.
    w_ief = float(target_weights.get("IEF", 0.0))
    if w_ief != 0.0:
        target_notional = nav_usd * w_ief
        raw_shares = target_notional / ief_price
        rounded = (
            int(math.floor(raw_shares + 0.5))
            if raw_shares >= 0
            else int(math.ceil(raw_shares - 0.5))
        )
        decisions.append(
            SizingDecision(
                symbol="IEF",
                target_quantity=raw_shares,
                rounded_quantity=rounded,
                target_notional_usd=target_notional,
                target_weight=w_ief,
                execution_vehicle="IEF",
            )
        )

    return decisions


def size_with_etf_only(
    target_weights: dict[str, float],
    nav_usd: float,
    spy_price: float,
    efa_price: float,
    ief_price: float,
) -> list[SizingDecision]:
    """Translate target weights into pure-ETF share counts (no futures leverage).

    Used when the operator wants the 1x exposure profile (no MES). The
    strategy's SPY weight is executed via SPY ETF shares. Weights > 1
    (from max_leverage) get clipped to 1.0 with a warning since ETFs
    can't be levered without margin (which this sizing layer doesn't
    model).
    """
    if nav_usd <= 0:
        raise ValueError("nav_usd must be positive")

    decisions: list[SizingDecision] = []
    for sym, price, key in (
        ("SPY", spy_price, "SPY"),
        ("EFA", efa_price, "EFA"),
        ("IEF", ief_price, "IEF"),
    ):
        w = float(target_weights.get(key, 0.0))
        if w == 0.0:
            continue
        if w > 1.0:
            w = 1.0  # ETF-only execution can't lever above 1x
        target_notional = nav_usd * w
        raw_shares = target_notional / price
        rounded = (
            int(math.floor(raw_shares + 0.5))
            if raw_shares >= 0
            else int(math.ceil(raw_shares - 0.5))
        )
        decisions.append(
            SizingDecision(
                symbol=sym,
                target_quantity=raw_shares,
                rounded_quantity=rounded,
                target_notional_usd=target_notional,
                target_weight=w,
                execution_vehicle=sym,  # type: ignore[arg-type]
            )
        )
    return decisions
