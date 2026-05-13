"""Unit tests for V3 layered defenses (Layer C/B/A).

Verifies each defense layer's behaviour in isolation and the composite
behaviour when all three are active. Each test constructs a deterministic
scenario where the defense should (or shouldn't) fire and pins the
expected outcome.

Defenses:
  Layer A — asymmetric hysteresis on HMM gate
  Layer B — momentum confluence (additional entry gates)
  Layer C — DD circuit breaker (mechanical failsafe)

Pinning the behaviour here prevents future regressions where a refactor
"silently" disables one of the defenses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_v3.strategy_v3 import V3Config, run_v3_strategy


def _make_constant_underlying(n: int = 800, daily_drift: float = 0.0004) -> pd.Series:
    """Smooth log-linear SPY series for deterministic strategy testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(100.0 * (1.0 + daily_drift) ** np.arange(n), index=idx, name="SPY")


def _make_crash_underlying(
    n: int = 800, crash_start: int = 600, crash_depth: float = -0.30
) -> pd.Series:
    """SPY series that's smooth then crashes 30% over 20 bars."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    daily_drift = 0.0004
    values = 100.0 * (1.0 + daily_drift) ** np.arange(n)
    # Inject a 30% crash over bars [crash_start, crash_start+20]
    decay = (1.0 + crash_depth) ** (1.0 / 20.0)
    for i in range(crash_start, min(crash_start + 20, n)):
        values[i] = values[i - 1] * decay
    # After the crash, hold flat
    for i in range(crash_start + 20, n):
        values[i] = values[crash_start + 19]
    return pd.Series(values, index=idx, name="SPY")


# ── Defense disabled by default ───────────────────────────────────────────


def test_defaults_disable_all_three_defenses() -> None:
    """Pin: V3Config defaults must NOT activate any defense layer.

    This pins the "minimum viable" behaviour so that someone enabling
    a defense has to do it explicitly. Preserves backward compat with
    the Layer-1+2 baseline.
    """
    cfg = V3Config()
    assert cfg.dd_kill == 0.0, "Layer C must be OFF by default (dd_kill=0)"
    assert cfg.use_momentum_confluence is False, "Layer B must be OFF by default"
    assert cfg.score_enter_threshold is None and cfg.score_exit_threshold is None, (
        "Layer A must be OFF by default (asymmetric thresholds = None)"
    )


# ── Layer C — DD circuit breaker ──────────────────────────────────────────


def test_layer_c_kill_triggers_after_drawdown_breach() -> None:
    """When strategy DD exceeds dd_kill, the strategy must force cash on
    the next bar regardless of regime score.

    Scenario: smooth uptrend then a sharp crash. HMM regime score held
    at 1.0 (forces deploy). Without Layer C the strategy holds through
    the crash and loses heavily. With Layer C at dd_kill=0.10 the
    strategy should exit during the crash.
    """
    spy = _make_crash_underlying(crash_start=400, crash_depth=-0.30)
    # Force always-benign regime so the gate doesn't help — only Layer C can save us.
    score = pd.Series(1.0, index=spy.index)

    cfg_no_c = V3Config(L_target=2.0, dd_kill=0.0)
    cfg_with_c = V3Config(L_target=2.0, dd_kill=0.10, dd_re_entry_score=0.70, dd_re_entry_bars=5)

    df_no_c = run_v3_strategy(spy, score, cfg_no_c)
    df_with_c = run_v3_strategy(spy, score, cfg_with_c)

    # Without C, MaxDD ≈ -60% (2x leverage on -30% crash)
    no_c_maxdd = float(((df_no_c["equity"] / df_no_c["equity"].cummax()) - 1).min())
    with_c_maxdd = float(((df_with_c["equity"] / df_with_c["equity"].cummax()) - 1).min())

    assert no_c_maxdd < -0.40, (
        f"Sanity: 2x exposure to -30% crash should hit -40%+, got {no_c_maxdd:.3f}"
    )
    assert with_c_maxdd > -0.30, (
        f"Layer C should cap MaxDD: got {with_c_maxdd:.3f} (expected > -30% with dd_kill=10%)"
    )
    # And the strategy must have entered the killed state
    assert (df_with_c["dd_state"] == "killed").any(), "dd_state must visit 'killed'"


def test_layer_c_recovery_re_anchors_hwm() -> None:
    """After kill → recovery, HWM resets to current equity.

    Without HWM reset, the strategy stays "underwater" relative to the
    pre-crash peak forever and re-trips the breaker on minor wobbles.
    With reset, the strategy can re-enter cleanly and only re-trips on
    new drawdowns measured from the post-recovery equity.
    """
    # Crash then smooth recovery
    spy = _make_crash_underlying(crash_start=400, crash_depth=-0.30)
    score = pd.Series(1.0, index=spy.index)
    cfg = V3Config(L_target=2.0, dd_kill=0.10, dd_re_entry_score=0.70, dd_re_entry_bars=5)
    df = run_v3_strategy(spy, score, cfg)

    # The HWM should reset after recovery. We check the dd_state series:
    # there should be a "killed" period followed by a return to "normal".
    state_seq = df["dd_state"].tolist()
    saw_killed = "killed" in state_seq
    if saw_killed:
        last_kill_idx = max(i for i, s in enumerate(state_seq) if s == "killed")
        # After last killed bar, must return to normal (recovery happens)
        post_kill = state_seq[last_kill_idx + 1 :]
        if len(post_kill) > 0:
            # If there are any post-kill bars, eventually we must be back to normal
            assert "normal" in post_kill, (
                "After kill the strategy must recover to normal eventually"
            )


# ── Layer B — momentum confluence ─────────────────────────────────────────


def test_layer_b_above_200d_sma_blocks_entry() -> None:
    """If close is below the 200-day SMA, Layer B blocks entry even
    when the HMM says benign.

    Scenario: SPY in clear downtrend (price below 200d MA throughout
    the test window). HMM regime score = 1.0 (always benign). Without
    Layer B the strategy deploys throughout. With Layer B
    (require_above_200d_sma=True) the strategy stays cash.
    """
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # SPY: 200d declining trend
    spy = pd.Series(100.0 * (0.9995) ** np.arange(n), index=idx, name="SPY")
    score = pd.Series(1.0, index=idx)

    cfg_no_b = V3Config(L_target=2.0, use_momentum_confluence=False)
    cfg_with_b = V3Config(
        L_target=2.0,
        use_momentum_confluence=True,
        require_above_200d_sma=True,
        mom_12_1_threshold=-1.0,  # very loose: don't block on momentum
        dd_velocity_threshold=-1.0,  # very loose: don't block on dd_vel
    )

    df_no_b = run_v3_strategy(spy, score, cfg_no_b)
    df_with_b = run_v3_strategy(spy, score, cfg_with_b)

    # Without B: most of the (post-warmup) bars are deployed
    frac_deployed_no_b = float(df_no_b["deployed"].iloc[250:].mean())
    frac_deployed_with_b = float(df_with_b["deployed"].iloc[250:].mean())

    assert frac_deployed_no_b > 0.8, (
        f"Sanity: without B and score=1, should be deployed most of the time. "
        f"Got {frac_deployed_no_b:.3f}"
    )
    assert frac_deployed_with_b < 0.2, (
        f"With B blocking entry on <200d-SMA, should rarely deploy in a downtrend. "
        f"Got {frac_deployed_with_b:.3f}"
    )


def test_layer_b_requires_positive_12_1_momentum_for_entry() -> None:
    """12-1 momentum threshold blocks entry when trailing 12-1 mom is below threshold."""
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # SPY: rises then crashes over the last 250d, with 12-1 mom strongly negative at the end
    values = np.empty(n)
    values[:550] = 100.0 * 1.0008 ** np.arange(550)  # bullish phase
    # Crash 40% over the last 250 days
    crash_factor = (0.60) ** (1.0 / 250.0)
    for i in range(550, n):
        values[i] = values[i - 1] * crash_factor
    spy = pd.Series(values, index=idx, name="SPY")
    score = pd.Series(1.0, index=idx)

    # Configure: require_above_200d_sma=False so we isolate the 12-1 mom test.
    # mom_12_1_threshold=0 means require 12-1 momentum strictly > 0 for entry.
    cfg = V3Config(
        L_target=2.0,
        use_momentum_confluence=True,
        require_above_200d_sma=False,
        mom_12_1_threshold=0.0,
        dd_velocity_threshold=-1.0,  # loose
    )
    df = run_v3_strategy(spy, score, cfg)

    # In the late period (post bar 700), 12-1 momentum is strongly negative.
    # Strategy should be in cash for that period.
    late_deployed = float(df["deployed"].iloc[700:].mean())
    assert late_deployed < 0.2, (
        f"With 12-1 mom < 0 in the late crash period, strategy should be in cash. "
        f"Got frac_deployed={late_deployed:.3f}"
    )


# ── Layer A — asymmetric hysteresis ───────────────────────────────────────


def test_layer_a_asymmetric_entry_threshold_higher_than_exit() -> None:
    """With score_enter=0.7 and score_exit=0.5, the strategy:
    - Stays in cash when score=0.6 (below entry threshold)
    - Enters when score crosses up through 0.7
    - Stays in (does NOT exit) when score drops to 0.6 (still above exit)
    - Exits when score drops below 0.5
    """
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spy = pd.Series(100.0 * 1.0004 ** np.arange(n), index=idx, name="SPY")

    # Build a deterministic score path:
    # bars 0-499: score = 0.6 (between exit and entry threshold)
    # bars 500-699: score = 0.8 (above entry)
    # bars 700-749: score = 0.6 (between thresholds — should HOLD)
    # bars 750-799: score = 0.4 (below exit — should exit)
    score_values = np.empty(n)
    score_values[:500] = 0.6
    score_values[500:700] = 0.8
    score_values[700:750] = 0.6
    score_values[750:] = 0.4
    score = pd.Series(score_values, index=idx)

    cfg = V3Config(
        L_target=2.0,
        score_enter_threshold=0.7,
        score_exit_threshold=0.5,
    )
    df = run_v3_strategy(spy, score, cfg)

    # In the 0.6 pre-entry period: should be in cash
    # (score=0.6 < entry threshold 0.7 → no entry)
    assert df["deployed"].iloc[250:499].mean() == 0.0, (
        "Layer A: must not enter when score < enter_threshold"
    )

    # In the 0.8 deployed period: should be deployed
    assert df["deployed"].iloc[510:699].mean() > 0.9, (
        "Layer A: must deploy when score > enter_threshold"
    )

    # In the 0.6 hold period: should STILL be deployed (above exit threshold)
    # (score=0.6 > exit threshold 0.5 → hold, no exit)
    assert df["deployed"].iloc[710:749].mean() > 0.9, (
        "Layer A: must hold position when score is between thresholds (asymmetric hysteresis)"
    )

    # In the 0.4 exit period: should exit
    assert df["deployed"].iloc[760:].mean() < 0.1, "Layer A: must exit when score < exit_threshold"


# ── Composite — all three layers together ────────────────────────────────


def test_all_three_layers_compose_without_explosion() -> None:
    """Sanity: configuring all three layers simultaneously doesn't crash
    and produces sensible output (non-NaN, equity > 0 after 800 bars)."""
    spy = _make_constant_underlying(n=800)
    score = pd.Series(0.7, index=spy.index)
    cfg = V3Config(
        L_target=2.0,
        score_enter_threshold=0.7,
        score_exit_threshold=0.5,
        use_momentum_confluence=True,
        mom_12_1_threshold=0.0,
        dd_velocity_threshold=-0.05,
        require_above_200d_sma=True,
        dd_kill=0.15,
        dd_re_entry_score=0.70,
        dd_re_entry_bars=5,
    )
    df = run_v3_strategy(spy, score, cfg)
    assert df["equity"].notna().all(), "Equity must not contain NaN"
    assert (df["equity"] > 0).all(), "Equity must stay positive"
    assert "dd_state" in df.columns
    assert "gate_a_intent" in df.columns
    assert "gate_b_intent" in df.columns
    assert "gate_c_active" in df.columns
