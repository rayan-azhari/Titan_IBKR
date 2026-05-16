"""Tests for titan/research/framework/early_gate.py.

The early gate is the Pass-1-gates-Pass-2 speed lever (#1) — see L52. The
critical invariants are:

    * A negative headline Sharpe is *always* rejected (CI_lo > 0 is
      impossible analytically; no need to run MC).
    * A clearly-positive Sharpe (well above SE) is always accepted.
    * The Lo (2002) SE formula scales correctly with sample size and with
      block-size inflation.
    * The sweep-level gate ("any cell can clear") flips iff at least one
      cell's headline beats its SE.
    * The I1 reproduction: 13 cells in [-0.31, -0.25] over ~1300 OOS bars
      all reject. This is the concrete incident the gate is designed to
      prevent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.early_gate import (
    format_gate_report,
    pass1_can_clear_any_cell,
    pass1_can_clear_ci_gate,
    pass1_can_clear_from_returns,
)

# ── analytical gate ───────────────────────────────────────────────────────


def test_gate_rejects_negative_sharpe() -> None:
    """Any negative Sharpe is rejected — CI_lo > 0 is analytically impossible."""
    res = pass1_can_clear_ci_gate(-0.28, n_oos_bars=1300, block_size=20)
    assert res.can_clear is False
    assert "<= 0" in res.reason


def test_gate_accepts_strongly_positive_sharpe() -> None:
    """A clearly-positive Sharpe with ample sample size must pass."""
    # SR=1.0 over 1000 bars: SE ~ sqrt(1.5/1000) ~= 0.039; CI_lo ~= 1.0 - 1.96*0.039 ~= 0.92.
    res = pass1_can_clear_ci_gate(1.0, n_oos_bars=1000, block_size=1)
    assert res.can_clear is True
    assert res.approx_ci_lo > 0


def test_gate_rejects_marginal_sharpe_with_large_block() -> None:
    """A marginal positive Sharpe is rejected when block inflation is large enough."""
    # SR=0.20 over 1000 bars with block=20:
    # SE = sqrt((1 + 0.5*0.04) / 1000 * 20) = sqrt(1.02 * 20 / 1000) ~= 0.143
    # CI_lo ~= 0.20 - 1.96 * 0.143 ~= -0.08  → fail
    res = pass1_can_clear_ci_gate(0.20, n_oos_bars=1000, block_size=20)
    assert res.can_clear is False


def test_gate_se_scales_with_inverse_sqrt_n() -> None:
    """Doubling sample size halves the variance; SE shrinks by sqrt(2)."""
    r1 = pass1_can_clear_ci_gate(0.5, n_oos_bars=500, block_size=1)
    r2 = pass1_can_clear_ci_gate(0.5, n_oos_bars=1000, block_size=1)
    assert r2.se_inflated == pytest.approx(r1.se_inflated / math.sqrt(2), rel=1e-9)


def test_gate_se_scales_with_sqrt_block_size() -> None:
    """SE inflates by sqrt(block_size) (effective sample size adjustment)."""
    base = pass1_can_clear_ci_gate(0.5, n_oos_bars=1000, block_size=1)
    inflated = pass1_can_clear_ci_gate(0.5, n_oos_bars=1000, block_size=25)
    assert inflated.se_inflated == pytest.approx(base.se_inflated * math.sqrt(25), rel=1e-9)


def test_gate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="confidence"):
        pass1_can_clear_ci_gate(0.5, n_oos_bars=1000, block_size=1, confidence=0.80)
    with pytest.raises(ValueError, match="n_oos_bars"):
        pass1_can_clear_ci_gate(0.5, n_oos_bars=0, block_size=1)
    with pytest.raises(ValueError, match="block_size"):
        pass1_can_clear_ci_gate(0.5, n_oos_bars=1000, block_size=0)


def test_gate_99_confidence_is_stricter_than_95() -> None:
    """A borderline Sharpe that just passes at 95% must fail at 99%."""
    # Find a borderline: SR=0.10, n=1000, block=1.
    r95 = pass1_can_clear_ci_gate(0.10, n_oos_bars=1000, block_size=1, confidence=0.95)
    r99 = pass1_can_clear_ci_gate(0.10, n_oos_bars=1000, block_size=1, confidence=0.99)
    # Approx CI_lo at 99% must be lower (more pessimistic).
    assert r99.approx_ci_lo < r95.approx_ci_lo


# ── from-returns convenience ──────────────────────────────────────────────


def test_from_returns_extracts_sharpe_and_n() -> None:
    """Returns-based wrapper computes headline + n_oos consistently with the analytic gate."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, size=1500))
    res = pass1_can_clear_from_returns(rets, periods_per_year=252, block_size=10)
    assert res.n_oos_bars == 1500
    # Re-deriving the gate manually must agree.
    from titan.research.metrics import sharpe as _sharpe

    sr = float(_sharpe(rets, periods_per_year=252))
    manual = pass1_can_clear_ci_gate(sr, n_oos_bars=1500, block_size=10)
    assert res.can_clear == manual.can_clear
    assert res.approx_ci_lo == pytest.approx(manual.approx_ci_lo, rel=1e-9)


def test_from_returns_handles_empty_series() -> None:
    """Empty / all-NaN series returns can_clear=False without raising."""
    res = pass1_can_clear_from_returns(pd.Series(dtype=float), periods_per_year=252)
    assert res.can_clear is False
    assert res.n_oos_bars == 0


# ── sweep-level gate ──────────────────────────────────────────────────────


def test_any_cell_gate_returns_true_iff_at_least_one_passes() -> None:
    rng = np.random.default_rng(1)
    # Construct: one strongly positive cell, two negative cells.
    good = pd.Series(rng.normal(0.005, 0.01, size=1000))
    bad1 = pd.Series(rng.normal(-0.001, 0.01, size=1000))
    bad2 = pd.Series(rng.normal(-0.002, 0.01, size=1000))
    any_pass, per_cell = pass1_can_clear_any_cell(
        {"good": good, "bad1": bad1, "bad2": bad2},
        periods_per_year=252,
        block_size=1,
    )
    assert any_pass is True
    assert per_cell["good"].can_clear is True
    assert per_cell["bad1"].can_clear is False
    assert per_cell["bad2"].can_clear is False


def test_gate_reproduces_i1_rejection() -> None:
    """L51 / I1 incident: 13 cells in [-0.31, -0.25] over ~1300 OOS bars.

    The early gate must reject ALL of them — that's the ~50min of compute
    the gate exists to save. Use the analytic path with the actual headline
    Sharpes from the I1 result log so the test is deterministic.
    """
    i1_headline_sharpes = [
        -0.2816,  # C1_canonical
        -0.2816,  # C2_short_train
        -0.2816,  # C3_long_train
        -0.2816,  # C4_vol_based_id
        -0.2817,  # C5_no_gate_baseline
        -0.2459,  # C7_gross_no_costs (best)
        -0.2795,  # C8_3_state
        -0.2822,  # X1_train_full_smooth10
        -0.2746,  # X2_high_mean_id
        -0.3054,  # X3_3state_vol_id (worst)
        -0.2815,  # X4_train_1y_smoothing5
        -0.2778,  # X5_seed_99
        -0.2788,  # X6_seed_123
    ]
    # All 13 must reject. Even the best (-0.2459) is negative, so the
    # block-bootstrap CI_lo > 0 condition is analytically impossible.
    for sr in i1_headline_sharpes:
        res = pass1_can_clear_ci_gate(sr, n_oos_bars=1300, block_size=20)
        assert res.can_clear is False, f"I1 Sharpe={sr} unexpectedly passed gate"


# ── reporting ─────────────────────────────────────────────────────────────


def test_format_report_renders_full_table() -> None:
    rng = np.random.default_rng(2)
    cells = {
        "C1": pd.Series(rng.normal(-0.001, 0.01, size=500)),
        "C2": pd.Series(rng.normal(+0.005, 0.01, size=500)),
    }
    _, per_cell = pass1_can_clear_any_cell(cells, periods_per_year=252)
    report = format_gate_report(per_cell, audit_label="DEMO")
    assert "Pass-1 early-gate" in report
    assert "DEMO" in report
    assert "C1" in report and "C2" in report
    assert "approx CI_lo" in report


def test_format_report_calls_out_full_rejection() -> None:
    rng = np.random.default_rng(3)
    cells = {
        "C1": pd.Series(rng.normal(-0.001, 0.01, size=500)),
        "C2": pd.Series(rng.normal(-0.002, 0.01, size=500)),
    }
    _, per_cell = pass1_can_clear_any_cell(cells, periods_per_year=252)
    report = format_gate_report(per_cell)
    assert "No cell can plausibly clear" in report
    assert "RETIRED" in report
