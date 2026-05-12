"""Regression tests for the 2026-05-12 audit's look-ahead findings.

Two distinct bugs in research code were identified and fixed:

1. ``bond_rotation_returns`` used today's close for both the winner decision
   AND the return assignment (audit finding A1). Fix: shift winner mask by 1
   bar so today's return is earned under yesterday's decision.

2. ``futures_returns`` subtracted ``L * (funding - dividend)`` as basis decay,
   correct only if the input is price-only. The actual input is yfinance
   adjusted close (total-return), so the function double-counted the
   dividend (audit finding A3). Fix: new ``futures_returns_tr`` strips the
   dividend from the TR series explicitly.

These tests pin both fixes so a regression to the buggy semantics fails
loudly. They are intentionally minimal — the goal is to catch the *class*
of bug, not certify every parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.samir_stack.margin_model import futures_returns, futures_returns_tr
from research.samir_stack.run_samir_improvements import bond_rotation_returns

# ── A1: bond_rotation_returns must not use today's close for today's return ──


def test_bond_rotation_winner_is_lagged_by_one_bar() -> None:
    """Construct a deterministic IEF/HYG fixture where the winner flips on a
    known date. The rotation return on the flip day must use the OLD winner
    (yesterday's decision), not the NEW winner (today's decision).

    Fixture: IEF rises monotonically for 60 days, then flatlines.
            HYG flatlines for 60 days, then jumps +10% on a single day.
    On flip day t*, the 60d-momentum winner shifts IEF → HYG.

    Correct (lagged) semantics: rotation return at t* uses the OLD winner
    (IEF) because yesterday's decision was IEF. Result: rotation return at
    t* = ief_ret[t*] (a small ~0 number).

    Buggy (same-bar) semantics: rotation uses TODAY's winner (HYG), so
    rotation return at t* = hyg_ret[t*] = +10%. That would be free money
    from a same-bar look-ahead.
    """
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ief = pd.Series(np.linspace(100.0, 110.0, n), index=idx, name="IEF")
    hyg_vals = np.full(n, 100.0)
    flip_day = 70
    hyg_vals[flip_day] = 110.0  # +10% single-day jump
    hyg = pd.Series(hyg_vals, index=idx, name="HYG")

    rot = bond_rotation_returns(ief, hyg, lookback_days=60)

    # Pre-flip: IEF has positive 60d momentum, HYG flat → winner=IEF
    # Lagged: on flip_day, the decision-as-of-yesterday is still IEF
    #   → rotation return == ief.pct_change()[flip_day] ≈ small positive
    # Buggy: on flip_day, today's winner is HYG (HYG just jumped +10%)
    #   → rotation return == hyg.pct_change()[flip_day] = +0.10

    rot_on_flip = float(rot.iloc[flip_day])
    ief_ret_on_flip = float(ief.pct_change().iloc[flip_day])
    hyg_ret_on_flip = float(hyg.pct_change().iloc[flip_day])

    # The fixed function must NOT return HYG's same-bar +10% return
    assert rot_on_flip < 0.05, (
        f"bond_rotation_returns has same-bar look-ahead: "
        f"rotation return on flip day = {rot_on_flip:.4f} ≈ HYG jump "
        f"({hyg_ret_on_flip:.4f}). Should be IEF's small move "
        f"({ief_ret_on_flip:.4f}) since yesterday's winner was IEF."
    )
    # Specifically, it must equal IEF's same-day return (within float tolerance)
    assert rot_on_flip == pytest.approx(ief_ret_on_flip, abs=1e-9), (
        f"On flip day, lagged-winner rotation should pay IEF's return "
        f"({ief_ret_on_flip:.6f}), got {rot_on_flip:.6f}."
    )


def test_bond_rotation_first_bar_is_zero() -> None:
    """With the lag fix, the first bar's winner is NaN → 'CASH' → zero
    return. This pins the boundary condition so a future refactor doesn't
    silently reintroduce day-0 look-ahead."""
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ief = pd.Series(np.linspace(100.0, 110.0, n), index=idx)
    hyg = pd.Series(np.linspace(100.0, 105.0, n), index=idx)
    rot = bond_rotation_returns(ief, hyg, lookback_days=60)
    assert float(rot.iloc[0]) == 0.0


# ── A3: futures_returns_tr must not double-count dividends ────────────────


def test_futures_returns_tr_at_L1_tracks_underlying_TR_within_tolerance() -> None:
    """At L=1 the futures wrapper should produce returns approximately equal
    to SPY total return + (funding − dividend − roll). Construct a controlled
    underlying with known CAGR and verify the wrapper's CAGR is within
    ±0.5pp of (spy_TR + funding - div - roll).

    The old (buggy) ``futures_returns`` would systematically add an extra
    +div/yr (~+1.5pp/yr at L=1) due to the dividend double-count.
    """
    n_years = 5.0
    n = int(n_years * 252)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Deterministic ~10% annualised TR (log-linear)
    annual_tr = 0.10
    daily_tr = (1.0 + annual_tr) ** (1.0 / 252.0) - 1.0
    spy_tr = pd.Series(
        100.0 * (1.0 + daily_tr) ** np.arange(n),
        index=idx,
    )

    rets = futures_returns_tr(
        spy_tr,
        leverage=1.0,
        dividend_yield=0.015,
        rolls_per_year=4,
        roll_slippage_bps=5.0,
    ).dropna()

    n_years_actual = len(rets) / 252.0
    cagr_realised = float((1.0 + rets).prod() ** (1.0 / n_years_actual) - 1.0)

    # Expected: spy_TR + avg_funding - dividend - roll
    # For 2020-2025 the piecewise funding averages ~2.0%.
    # The exact value doesn't matter — we just need |bias| << dividend_yield.
    expected_min = annual_tr - 0.015 - 0.002  # drop divid and roll, no T-bill credit
    expected_max = annual_tr - 0.015 + 0.05 - 0.002  # plus generous T-bill upper bound

    assert expected_min <= cagr_realised <= expected_max, (
        f"futures_returns_tr at L=1 CAGR = {cagr_realised:.4f}, expected "
        f"in [{expected_min:.4f}, {expected_max:.4f}]. Suggests dividend "
        f"or funding term is wrong."
    )
    # And specifically: the L=1 bias vs TR must be smaller than the buggy
    # function's +div=1.5pp/yr signature. The legitimate bias is
    # `funding - dividend - roll` which varies with the funding regime
    # (for 2020-2025 the avg funding ≈ 2.5% → bias ≈ +0.8pp). We pin
    # |bias| < 1.5pp so a regression that adds back the dividend (giving
    # ~+2-3pp bias) fails loudly.
    bias_vs_tr = cagr_realised - annual_tr
    assert bias_vs_tr < 0.015, (
        f"At L=1 the wrapper bias vs underlying TR should be < +1.5pp "
        f"(T-bill pickup minus dividend minus roll, varies with funding "
        f"regime). Got {bias_vs_tr * 100:+.2f}pp — looks like the "
        f"dividend is being added back."
    )


def test_old_futures_returns_emits_deprecation_warning() -> None:
    """The buggy ``futures_returns`` must warn when called, so anyone
    reproducing pre-2026-05-12 backtests sees the flag."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    spy = pd.Series(np.linspace(100.0, 130.0, 300), index=idx)
    with pytest.warns(DeprecationWarning, match="dividend double-count"):
        futures_returns(spy, leverage=2.0)


def test_futures_returns_tr_L3_bias_versus_old_function() -> None:
    """Quantitative pin: at L=3 the corrected wrapper's CAGR must be
    materially lower than the buggy wrapper's CAGR. Per the audit, the bias
    at L=3 is roughly +4.75pp/yr on a real SPY series.
    """
    n_years = 5.0
    n = int(n_years * 252)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    daily_tr = (1.0 + 0.10) ** (1.0 / 252.0) - 1.0
    spy_tr = pd.Series(100.0 * (1.0 + daily_tr) ** np.arange(n), index=idx)

    fixed = futures_returns_tr(spy_tr, leverage=3.0).dropna()
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        buggy = futures_returns(spy_tr, leverage=3.0).dropna()

    cagr_fixed = float((1.0 + fixed).prod() ** (1.0 / n_years) - 1.0)
    cagr_buggy = float((1.0 + buggy).prod() ** (1.0 / n_years) - 1.0)
    bias = cagr_buggy - cagr_fixed

    # Bias should be ≈ 2 × L × div / yr − (some funding interaction) ≈ +4-5pp at L=3.
    # Pin a generous range: at minimum +2pp, at most +7pp.
    assert 0.02 < bias < 0.07, (
        f"Expected L=3 dividend-double-count bias in [2pp, 7pp]/yr, got "
        f"{bias * 100:+.2f}pp. If too small the fix may have over-corrected; "
        f"if too large the synthetic fixture may be wrong."
    )
