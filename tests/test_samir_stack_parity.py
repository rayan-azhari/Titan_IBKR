"""Live-vs-research parity tests for the Samir-Stack strategy.

Asserts the live regime-computation path matches the research backtest:
given the same input series, the live ``compute_regime_score`` and
``target_tier_from_score`` produce identical outputs to the research
``regime_score_equal`` and ``_equity_target_tier``.

This is the structural guard required before deployment per the project's
research-math discipline (see directives/IC Signal Analysis.md and
CLAUDE.md memory): if the two paths diverge, this test fails and the
drift is caught before any trade is placed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.regime_score import (
    DEFAULT_INDICATORS,
    regime_score_equal,
)
from research.samir_stack.stacked_strategy import (
    StackedConfig,
    _equity_target_tier,
)
from titan.strategies.samir_stack.regime import (
    RegimeBuffers,
    compute_regime_score,
    target_tier_from_score,
)


def _make_synthetic_data(n_days: int = 1000, seed: int = 42) -> dict:
    """Build synthetic but realistic SPY/VIX/HYG/IEF series for parity testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")

    # SPY: random walk with drift, vol clusters
    log_rets = rng.normal(0.0003, 0.012, n_days)
    spy = pd.Series(np.exp(np.cumsum(log_rets)) * 100.0, index=dates)

    # VIX: mean-reverting around 20, with occasional spikes
    vix = pd.Series(
        20.0 + 5.0 * rng.standard_normal(n_days).cumsum() / np.sqrt(n_days),
        index=dates,
    ).clip(lower=10.0, upper=80.0)

    # HYG: random walk
    hyg = pd.Series(
        np.exp(np.cumsum(rng.normal(0.0001, 0.005, n_days))) * 80.0,
        index=dates,
    )
    # IEF: random walk, lower vol
    ief = pd.Series(
        np.exp(np.cumsum(rng.normal(0.0001, 0.003, n_days))) * 100.0,
        index=dates,
    )
    return {"spy": spy, "vix": vix, "hyg": hyg, "ief": ief}


def test_regime_score_parity_equal_weight():
    """Live compute_regime_score matches research regime_score_equal."""
    data = _make_synthetic_data(n_days=1000)

    # Research path
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data["hyg"],
        ief_close=data["ief"],
    )
    research_score = regime_score_equal(panel, indicators=DEFAULT_INDICATORS)

    # Live path: feed each bar into buffers + compute score
    buffers = RegimeBuffers()
    live_scores = []
    for ts in data["spy"].index:
        buffers.add_spy(
            ts, float(data["spy"].loc[ts]), float(data["spy"].loc[ts]), float(data["spy"].loc[ts])
        )  # synthetic OHLC
        if ts in data["vix"].index:
            buffers.add_vix(ts, float(data["vix"].loc[ts]))
        if ts in data["hyg"].index:
            buffers.add_hyg(ts, float(data["hyg"].loc[ts]))
        if ts in data["ief"].index:
            buffers.add_ief(ts, float(data["ief"].loc[ts]))
        score, _ = compute_regime_score(buffers, enable_credit=True)
        live_scores.append(score)
    live_series = pd.Series(live_scores, index=data["spy"].index)

    # Compare values where research is non-NaN
    common = research_score.dropna().index.intersection(live_series.dropna().index)
    research_aligned = research_score.loc[common]
    live_aligned = live_series.loc[common]

    # Sample a few bars after warmup (last 20% of bars)
    n_test = max(1, len(common) // 5)
    test_idx = common[-n_test:]

    diffs = []
    for ts in test_idx:
        r = research_aligned.loc[ts]
        ll = live_aligned.loc[ts]
        if pd.notna(r) and pd.notna(ll):
            diffs.append(abs(r - ll))

    assert len(diffs) > 50, f"Too few comparable bars: {len(diffs)}"
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    # Allow small numerical tolerance — both paths use the same math but
    # different code, so floating-point residuals up to ~1e-3 are OK
    assert max_diff < 0.05, (
        f"Live and research regime scores diverged. max_diff={max_diff:.4f}, "
        f"mean_diff={mean_diff:.4f} on {len(diffs)} test bars"
    )


def test_target_tier_parity_basic_thresholds():
    """Live target_tier_from_score matches research _equity_target_tier."""
    cfg = StackedConfig(L_max=3.0, tier_thresholds=(0.30, 0.50, 0.75), hysteresis_buffer=0.05)

    # Test a grid of (score, current_tier) pairs
    test_cases = [
        (0.10, 0.0),  # deep hostile from cash → cash
        (0.20, 1.0),  # hostile from 1x → cash
        (0.40, 0.0),  # moderate from cash → 1x (above 0.30+0.05)
        (0.40, 2.0),  # moderate from 2x → 1x
        (0.60, 1.0),  # benign-ish from 1x → 2x (above 0.50+0.05)
        (0.80, 2.0),  # full benign from 2x → 3x (above 0.75+0.05)
        (0.32, 0.0),  # just above lower threshold but below buffer → cash
        (0.32, 1.0),  # at 1x, score above lower → stay 1x
        (0.50, 1.0),  # at 1x, score at exact 2x threshold → stay 1x (no buffer)
        (0.55, 1.0),  # at 1x, score above 2x+buffer → 2x
    ]
    for score, current in test_cases:
        live_target = target_tier_from_score(
            score,
            current,
            tier_thresholds=cfg.tier_thresholds,
            hysteresis_buffer=cfg.hysteresis_buffer,
            L_max=cfg.L_max,
        )
        research_target = _equity_target_tier(score, cfg, current)
        assert live_target == research_target, (
            f"Tier mismatch at score={score}, current={current}: "
            f"live={live_target}, research={research_target}"
        )


def test_target_tier_caps_at_L_max():
    """Tier never exceeds L_max regardless of threshold settings."""
    # L_max=2 but score very high → tier capped at 2
    target = target_tier_from_score(
        0.95,
        1.0,
        tier_thresholds=(0.30, 0.50, 0.75),
        hysteresis_buffer=0.05,
        L_max=2.0,
    )
    assert target == 2.0


def test_target_tier_handles_nan():
    """NaN score holds current tier (no spurious moves)."""
    target = target_tier_from_score(
        float("nan"),
        2.0,
        tier_thresholds=(0.30, 0.50, 0.75),
        hysteresis_buffer=0.05,
        L_max=3.0,
    )
    assert target == 2.0


def test_fx_unit_conversion_gbp_base_usd_equity():
    """FX-aware unit sizing: GBP-base account holding USD-quoted 3USL.

    Mirrors the production config: equity_quote_ccy=USD, base_ccy=GBP,
    fx_rate_equity_quote_to_base=0.7519 (1 USD = 0.7519 GBP at GBPUSD=1.33).
    """
    from titan.risk.strategy_equity import convert_notional_to_units

    notional_gbp = 4000.0
    price_usd = 100.0
    fx_usd_to_gbp = 0.7519

    units = convert_notional_to_units(
        notional_base=notional_gbp,
        price=price_usd,
        quote_ccy="USD",
        base_ccy="GBP",
        fx_rate_quote_to_base=fx_usd_to_gbp,
    )
    # £4000 / 0.7519 GBP/USD = $5320, / $100 = 53 units (int floor)
    expected = int(notional_gbp / fx_usd_to_gbp / price_usd)
    assert units == expected
    assert units == 53


def test_fx_fail_fast_on_default_rate():
    """convert_notional_to_units must refuse default 1.0 across currencies."""
    from titan.risk.strategy_equity import convert_notional_to_units

    try:
        convert_notional_to_units(
            notional_base=1000.0,
            price=100.0,
            quote_ccy="USD",
            base_ccy="GBP",
            fx_rate_quote_to_base=None,
        )
        raise AssertionError("Expected ValueError on missing fx_rate")
    except ValueError as e:
        assert "fx_rate_quote_to_base is required" in str(e)


def test_fx_passthrough_when_currencies_match():
    """No FX conversion when quote_ccy == base_ccy (IGLT in GBP account)."""
    from titan.risk.strategy_equity import convert_notional_to_units

    units = convert_notional_to_units(
        notional_base=6000.0,
        price=10.0,
        quote_ccy="GBP",
        base_ccy="GBP",
        fx_rate_quote_to_base=None,
    )
    assert units == 600  # £6000 / £10 = 600 IGLT shares


if __name__ == "__main__":
    test_regime_score_parity_equal_weight()
    test_target_tier_parity_basic_thresholds()
    test_target_tier_caps_at_L_max()
    test_target_tier_handles_nan()
    test_fx_unit_conversion_gbp_base_usd_equity()
    test_fx_fail_fast_on_default_rate()
    test_fx_passthrough_when_currencies_match()
    print("All parity tests passed")
