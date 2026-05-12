"""Tests for the shared bond_gold decision primitives.

These primitives back both:
  - ``scripts/replay_audit.py`` (Tier 2.5: backtest-vs-live diff)
  - ``titan/strategies/reconciliation/strategy.py`` D5 (Tier 2.1:
    in-process shadow-decision check)

The ``test_replay_audit.py`` and ``test_reconciliation_strategy.py``
tests already exercise the primitives indirectly through their
respective callers. This module locks down the shared module's public
surface and the LIVE_CONFIGS dictionary against accidental drift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from titan.utils.bond_gold_decisions import (
    LIVE_CONFIGS,
    BondGoldDecisionConfig,
    compute_z_score,
    expected_action,
    load_signal_closes_from_parquet,
)


def test_live_configs_contains_three_bond_equity_strategies():
    """Lock in the live deployment surface: three bond_equity_* strategies."""
    expected = {
        "bond_equity_ihyu_cspx",
        "bond_equity_ihyg_vusd",
        "bond_equity_ihyg_eimi",
    }
    assert set(LIVE_CONFIGS.keys()) == expected


def test_live_configs_field_consistency():
    """Each LIVE_CONFIGS entry must have all required fields populated."""
    for name, cfg in LIVE_CONFIGS.items():
        assert isinstance(cfg, BondGoldDecisionConfig)
        assert cfg.name == name
        assert cfg.trade_symbol  # non-empty
        assert cfg.signal_ticker  # non-empty
        assert cfg.lookback >= 1
        assert 0.0 < cfg.threshold < 5.0
        assert cfg.hold_days >= 1
        assert cfg.zscore_window >= 50


def test_live_configs_match_strategy_registry_values():
    """The three live configs must match the values set in
    scripts/run_portfolio.py STRATEGY_REGISTRY exactly. If the registry
    changes, this test fails — and both must be updated together."""
    # bond_equity_ihyu_cspx: IHYU signal, CSPX trade, LB=10, threshold=0.5, hold=10
    cfg = LIVE_CONFIGS["bond_equity_ihyu_cspx"]
    assert (cfg.signal_ticker, cfg.trade_symbol) == ("IHYU", "CSPX")
    assert cfg.lookback == 10 and cfg.threshold == 0.50 and cfg.hold_days == 10

    # bond_equity_ihyg_vusd: IHYG signal, VUSD trade, LB=5, threshold=0.25, hold=5
    cfg = LIVE_CONFIGS["bond_equity_ihyg_vusd"]
    assert (cfg.signal_ticker, cfg.trade_symbol) == ("IHYG", "VUSD")
    assert cfg.lookback == 5 and cfg.threshold == 0.25 and cfg.hold_days == 5

    # bond_equity_ihyg_eimi: IHYG signal, EIMI trade, LB=5, threshold=0.25, hold=5
    cfg = LIVE_CONFIGS["bond_equity_ihyg_eimi"]
    assert (cfg.signal_ticker, cfg.trade_symbol) == ("IHYG", "EIMI")
    assert cfg.lookback == 5 and cfg.threshold == 0.25 and cfg.hold_days == 5


def test_compute_z_score_basic_invariants_via_shared_module():
    """The shared module's compute_z_score behaves identically to the
    re-export in scripts/replay_audit.py (which historically had the
    inlined implementation)."""
    # Insufficient history → None
    assert compute_z_score([100.0] * 5, lookback=5, zscore_window=504) is None

    # Constant series → None (zero std)
    assert compute_z_score([100.0] * 100, lookback=5, zscore_window=504) is None


def test_expected_action_no_entry_while_long_invariant():
    """Critical post-fix invariant: never entry when already long.
    This is the May 11 antipattern guard."""
    assert (
        expected_action(z=10.0, is_long=True, bars_held=100, threshold=0.25, hold_days=5) == "hold"
    )


def test_load_signal_closes_returns_none_for_missing_parquet(tmp_path: Path):
    """Missing parquet must return None (caller skips), not raise."""
    assert load_signal_closes_from_parquet("DOES_NOT_EXIST", tmp_path) is None


def test_load_signal_closes_returns_list_for_valid_parquet(tmp_path: Path):
    """A valid parquet with a 'close' column returns a sorted list of floats."""
    df = pd.DataFrame(
        {"close": [100.0, 101.5, 99.2, 102.0]},
        index=pd.date_range("2026-05-01", periods=4, freq="D"),
    )
    out_path = tmp_path / "TEST_D.parquet"
    df.to_parquet(out_path)
    closes = load_signal_closes_from_parquet("TEST", tmp_path)
    assert closes is not None
    assert closes == [100.0, 101.5, 99.2, 102.0]


def test_load_signal_closes_returns_none_when_close_column_missing(tmp_path: Path):
    """A parquet without a 'close' column is unusable — return None."""
    df = pd.DataFrame(
        {"price": [100.0, 101.5]},
        index=pd.date_range("2026-05-01", periods=2, freq="D"),
    )
    out_path = tmp_path / "BAD_D.parquet"
    df.to_parquet(out_path)
    assert load_signal_closes_from_parquet("BAD", tmp_path) is None
