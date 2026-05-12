"""Pluggable equity engines and bond sleeves for the Samir-Stack state machine.

Background
----------
``run_stacked_strategy`` historically hard-coded its equity sleeve as
``synthetic_3x.synthetic_leveraged_returns(spy_close, L, ...)`` and its
bond sleeve as ``ief_close.pct_change()``. The 2026-05-12 audit identified
that the various sweep scripts (``run_overlay_sweep``,
``run_futures_sweep``) bolted on alternative engines and rotation logic
via monkey-patching and synthetic-close construction — both of which made
look-ahead bias easy to introduce and hard to spot.

This module replaces that pattern with two small typed contracts:

* ``EquityEngine`` — produces the daily-return series of a constant-L
  exposure to the underlying, per $1 of sleeve equity.
* ``BondSleeve`` — produces the daily-return series of the bond sleeve,
  per $1 of sleeve equity. Static and rotating sleeves both implement it.

Both contracts are PURE: no monkey-patching, no synthetic-close
construction. The strategy state machine asks each component for its
daily return series and applies the existing tier × weight logic on top.

The Phase 1 fixes (lag in ``bond_rotation_returns``, dividend handling
in ``futures_returns_tr``) are inherited automatically — the engines
here just wrap the corrected functions.

Reference
---------
``directives/Samir-Stack Remediation Plan 2026-05-12.md`` §2 (design
decisions) and §3 (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from research.samir_stack.margin_model import (
    CSPX_TER_ANNUAL,
    IBKR_LITE_SPREAD_OVER_BENCHMARK,
    IBKR_PRO_SPREAD_OVER_BENCHMARK,
    constant_leverage_margin_returns,
    futures_returns_tr,
)
from research.samir_stack.run_samir_improvements import bond_rotation_returns
from research.samir_stack.synthetic_3x import funding_series, synthetic_leveraged_returns

# ── EquityEngine ABC + 4 concrete implementations ─────────────────────────


class EquityEngine(Protocol):
    """Daily-return generator for the equity sleeve at a given leverage.

    Implementations must produce a Series of per-$1-equity daily returns.
    The state machine handles regime tiering, sleeve weight, transaction
    costs, and drawdown gating downstream; engines are responsible only
    for the leverage / cost-of-financing maths.

    Contract:
      * Input: TR-adjusted underlying close series (e.g. yfinance SPY close)
        and a constant target leverage ``L``.
      * Output: Series of daily returns, same index as input (first bar
        NaN from pct_change). No look-ahead — return at bar t is computed
        from data through bar t only.
    """

    def daily_returns(self, underlying_tr_close: pd.Series, leverage: float) -> pd.Series: ...


@dataclass
class SyntheticETFEngine:
    """Daily-rebalanced leveraged ETF (e.g. 3USL.LSEETF) synthetic.

    Inherits volatility drag from compounding L × r daily; pays the
    fund's TER and wholesale funding. Use this engine when the
    operational target is a leveraged-ETF wrapper like WisdomTree 3USL.
    """

    ter_annual: float = 0.0075  # 3USL default TER
    trading_days_per_year: int = 252

    def daily_returns(self, underlying_tr_close: pd.Series, leverage: float) -> pd.Series:
        return synthetic_leveraged_returns(
            underlying_tr_close,
            leverage=leverage,
            ter_annual=self.ter_annual if leverage > 1.0 else 0.0,
            trading_days_per_year=self.trading_days_per_year,
        )


@dataclass
class MarginEngine:
    """Constant-L margined position (e.g. CSPX held on IBKR Pro margin).

    Same daily-rebalance vol drag as the leveraged-ETF model, but pays
    broker margin (~SOFR + 1.5%) on the borrowed portion and only the
    underlying ETF's small TER (~0.07% for CSPX).
    """

    ter_annual: float = CSPX_TER_ANNUAL
    broker: str = "ibkr_pro"
    trading_days_per_year: int = 252

    def daily_returns(self, underlying_tr_close: pd.Series, leverage: float) -> pd.Series:
        return constant_leverage_margin_returns(
            underlying_tr_close,
            leverage=leverage,
            ter_annual=self.ter_annual,
            broker=self.broker,
            trading_days_per_year=self.trading_days_per_year,
        )


@dataclass
class FuturesEngine:
    """Constant-L SPX futures position (MES / ES).

    Uses the corrected ``futures_returns_tr`` (Phase 1 fix) that strips
    the dividend from the TR input before applying basis decay. No daily
    rebalance, no TER, T-bill on cash equity, quarterly roll slippage.
    """

    dividend_yield: float = 0.015
    margin_pct: float = 0.06
    rolls_per_year: int = 4
    roll_slippage_bps: float = 5.0
    trading_days_per_year: int = 252

    def daily_returns(self, underlying_tr_close: pd.Series, leverage: float) -> pd.Series:
        return futures_returns_tr(
            underlying_tr_close,
            leverage=leverage,
            dividend_yield=self.dividend_yield,
            margin_pct=self.margin_pct,
            rolls_per_year=self.rolls_per_year,
            roll_slippage_bps=self.roll_slippage_bps,
            trading_days_per_year=self.trading_days_per_year,
        )


@dataclass
class FuturesMarginEngine:
    """Futures position with an additional broker margin borrow.

    Models the case where the operator combines two leverage mechanics:

    1. Futures (cheap basis-based financing) — the primary leverage vehicle.
    2. Broker margin borrow on top of cash equity — used to fund a LARGER
       cash collateral base than the operator's own equity provides.

    Effective leverage on equity is ``leverage * (1 + borrow_ratio)``. At
    ``borrow_ratio = 0.0`` the engine reduces EXACTLY to ``FuturesEngine``
    — no new behaviour. At positive ``borrow_ratio`` the operator pays
    the broker spread on the borrowed portion (the wholesale-funding
    component cancels out against T-bill earned on the larger collateral
    base, so the *net* cost is just ``borrow_ratio * broker_spread`` per
    year — exactly the broker's margin markup).

    Use cases
    ---------
    * Push effective leverage beyond what cash futures naturally provides
      at the operator's NAV. At MES 6% IM, $E cash supports ~16x notional
      natively, so this is only relevant for L_eff >> 10.
    * Free up some cash equity to hold higher-yielding instruments (e.g.
      bonds) while still maintaining the futures notional. Models the
      operator who would rather pay broker spread on a margin loan than
      hold low-yield cash as futures collateral.
    * Sweep-cell exploration of "what if we hybridise MES + margin" in
      Phase 5 without changing the sleeve abstraction.

    Math (per $1 of cash equity, per day)
    -------------------------------------
    Let ``L`` be the base leverage and ``b`` = ``borrow_ratio``::

        L_total = L * (1 + b)                            # eff. leverage on equity
        daily_return = L_total * spy_TR_ret              # gross levered exposure
                     - L_total * div_yield/252           # strip div (futures don't pay)
                     + (1 + b) * funding/252             # T-bill on (E+B) collateral
                     - b * (funding + spread)/252        # margin interest on B
                     - L_total * roll/252                # roll slippage on notional

    The two interest terms simplify to ``(funding - b*spread)/252``, so
    the net economic cost of the borrow is ``b * broker_spread`` per
    year, regardless of the prevailing funding regime. This is correct:
    you pay the broker their spread, while the wholesale-rate component
    cancels (the broker funds at wholesale and earns the spread; you
    earn T-bill on the larger collateral).

    Sanity check
    ------------
    At ``borrow_ratio=0`` and identical other params, this engine MUST
    produce returns equal to ``FuturesEngine`` to floating-point
    precision. Pinned by a test in ``tests/test_samir_stack_engines.py``.
    """

    borrow_ratio: float = 0.0
    """B/E — broker margin borrow as fraction of cash equity. 0 = pure
    futures (no borrow); 0.5 = borrow 50% of equity as additional
    collateral; 1.0 = borrow 100% of equity (doubles collateral base)."""

    broker: str = "ibkr_pro"
    """Broker tier for the margin-spread component. ``ibkr_pro`` = +1.5%
    above wholesale; ``ibkr_lite`` = +2.5%."""

    dividend_yield: float = 0.015
    margin_pct: float = 0.06
    rolls_per_year: int = 4
    roll_slippage_bps: float = 5.0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.borrow_ratio < 0:
            raise ValueError(f"borrow_ratio must be >= 0, got {self.borrow_ratio}")

    def daily_returns(self, underlying_tr_close: pd.Series, leverage: float) -> pd.Series:
        # Fast path: when no borrow, delegate to FuturesEngine math so
        # this engine is bit-exact identical at borrow_ratio=0.
        if self.borrow_ratio == 0.0:
            return futures_returns_tr(
                underlying_tr_close,
                leverage=leverage,
                dividend_yield=self.dividend_yield,
                margin_pct=self.margin_pct,
                rolls_per_year=self.rolls_per_year,
                roll_slippage_bps=self.roll_slippage_bps,
                trading_days_per_year=self.trading_days_per_year,
            )

        if leverage <= 0:
            raise ValueError(f"leverage must be > 0, got {leverage}")
        # Refuse if the levered notional can't be cash-collateralised even
        # WITH the margin borrow (matches FuturesEngine's existing guard).
        l_total = leverage * (1.0 + self.borrow_ratio)
        if l_total * self.margin_pct >= 1.0:
            raise ValueError(
                f"effective leverage * margin_pct = "
                f"{l_total * self.margin_pct:.2f} >= 1; insufficient "
                f"cash to support L_eff={l_total:.2f} at "
                f"{self.margin_pct:.0%} margin requirement even with "
                f"borrow_ratio={self.borrow_ratio:.2f}."
            )

        spread = (
            IBKR_PRO_SPREAD_OVER_BENCHMARK
            if self.broker == "ibkr_pro"
            else IBKR_LITE_SPREAD_OVER_BENCHMARK
        )
        spy_tr_ret = underlying_tr_close.pct_change()
        funding = funding_series(underlying_tr_close.index)
        daily_funding = funding / self.trading_days_per_year
        daily_div = self.dividend_yield / self.trading_days_per_year
        daily_roll = (
            self.rolls_per_year * self.roll_slippage_bps / 10_000.0
        ) / self.trading_days_per_year
        daily_spread = spread / self.trading_days_per_year
        # See class docstring for derivation. The (1+b)*funding and
        # -b*(funding+spread) terms combine to (funding - b*spread).
        return (
            (l_total * spy_tr_ret)
            - (l_total * daily_div)
            + daily_funding
            - (self.borrow_ratio * daily_spread)
            - (l_total * daily_roll)
        )


# ── BondSleeve ABC + 2 concrete implementations ──────────────────────────


class BondSleeve(Protocol):
    """Daily-return generator for the bond sleeve.

    Contract:
      * ``daily_returns(index)`` returns a Series indexed to ``index``
        (filled with 0.0 outside the sleeve's native coverage) giving
        per-$1-equity daily returns for the bond sleeve.
      * Rotation sleeves MUST lag their winner decision one bar
        (winner_at_(t-1) decides return_at_t). Static sleeves are
        trivially lag-free.
    """

    def daily_returns(self, index: pd.DatetimeIndex) -> pd.Series: ...


@dataclass
class StaticBondSleeve:
    """Single fixed bond instrument (e.g. IEF or IGLT).

    Daily return = ``close.pct_change()``. No look-ahead because
    pct_change uses past + current close only, and the state machine's
    ``prev_eq_w * ret_t`` pattern ensures the return is earned under
    yesterday's position.
    """

    name: str
    close: pd.Series

    def daily_returns(self, index: pd.DatetimeIndex) -> pd.Series:
        return self.close.reindex(index).pct_change().fillna(0.0)


@dataclass
class RotationBondSleeve:
    """Momentum-based rotation among two bond instruments + cash fallback.

    At bar t, the winner is determined from the LAGGED ``lookback_days``
    momentum (i.e. closes through bar t-1). If all candidates have
    non-positive lagged momentum the sleeve goes to cash (return = 0
    for that bar).

    Implementation note: delegates to the Phase-1-fixed
    ``bond_rotation_returns`` for the IEF/HYG case for back-compat; for
    the general N-instrument case it computes the lag locally with
    identical semantics.
    """

    name: str
    candidates: dict[str, pd.Series]  # symbol -> close series
    lookback_days: int = 60

    def daily_returns(self, index: pd.DatetimeIndex) -> pd.Series:
        if len(self.candidates) < 2:
            raise ValueError(
                f"RotationBondSleeve {self.name!r} needs ≥2 candidates, got {len(self.candidates)}"
            )
        if len(self.candidates) == 2 and set(self.candidates.keys()) == {"IEF", "HYG"}:
            # Use the Phase 1-corrected helper for back-compat with the
            # historical IEF/HYG rotation. The helper applies the same
            # lag semantics defined below.
            return (
                bond_rotation_returns(
                    self.candidates["IEF"],
                    self.candidates["HYG"],
                    lookback_days=self.lookback_days,
                )
                .reindex(index)
                .fillna(0.0)
            )

        # General N-instrument case — replicate the lag contract.
        common = None
        for s in self.candidates.values():
            common = s.index if common is None else common.intersection(s.index)
        if common is None or len(common) == 0:
            return pd.Series(0.0, index=index)

        closes = pd.DataFrame({k: v.reindex(common) for k, v in self.candidates.items()})
        moms = closes.pct_change(self.lookback_days)
        rets = closes.pct_change().fillna(0.0)

        # winner_today = argmax over candidates with positive momentum,
        # else "CASH". Then shift by 1 bar so today's return is earned
        # under yesterday's decision.
        winner_today = pd.Series("CASH", index=common)
        positive = moms.where(moms > 0)
        # Skip rows where all candidates are non-positive (pandas warns
        # if we call idxmax on an all-NA row).
        any_positive = positive.notna().any(axis=1)
        if any_positive.any():
            valid = positive.loc[any_positive].idxmax(axis=1)
            winner_today.loc[valid.index] = valid
        winner = winner_today.shift(1).fillna("CASH")

        out = pd.Series(0.0, index=common)
        for sym in self.candidates:
            mask = winner == sym
            if mask.any():
                out.loc[mask] = rets.loc[mask, sym]
        return out.reindex(index).fillna(0.0)


# ── Convenience: validate that an engine + sleeve pair is bit-exact
#    equivalent to a known-good legacy pair. ──────────────────────────────


def assert_engine_matches_legacy(
    engine: EquityEngine,
    legacy_fn,
    spy_close: pd.Series,
    leverage: float,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Test helper: assert engine.daily_returns ≡ legacy_fn(spy, leverage).

    Used by the Phase 2 bit-exact reproduction gate.
    """
    new_rets = engine.daily_returns(spy_close, leverage).dropna()
    old_rets = legacy_fn(spy_close, leverage=leverage).dropna()
    common = new_rets.index.intersection(old_rets.index)
    diff = float(np.abs((new_rets.loc[common] - old_rets.loc[common])).max())
    assert diff < tolerance, (
        f"Engine output differs from legacy by max(|Δ|)={diff:.3e}, tolerance={tolerance:.3e}"
    )
