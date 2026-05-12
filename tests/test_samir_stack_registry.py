"""Tests for the samir_stack_paper registry entry in run_portfolio.py.

Catches regressions in the paper-validation deployment config — the
most fragile part of moving a strategy from "tested in isolation" to
"running in the docker container alongside the live portfolio."

See ``directives/Samir-Stack Paper Validation 2026-05-12.md`` for the
full deployment runbook this entry serves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# scripts.run_portfolio is heavy (subscribes ibapi, loads dotenv).
# Importing it just to read STRATEGY_REGISTRY is fine — the side
# effects don't reach the IBKR gateway unless main() is called.
from scripts.run_portfolio import (
    _STRATEGY_WARMUP_FILES,
    STRATEGY_REGISTRY,
    STRATEGY_SETS,
)


def test_samir_stack_paper_entry_exists():
    """The paper-validation entry must be registered."""
    assert "samir_stack_paper" in STRATEGY_REGISTRY


def test_samir_validation_set_exists():
    """The strategy set used to opt-in to paper validation."""
    assert "samir_validation" in STRATEGY_SETS


def test_samir_stack_paper_NOT_in_champion_portfolio():
    """Critical safety check: the paper entry must NOT auto-deploy
    in the live champion set. Promotion to champion_portfolio is the
    final step of the 4-week validation gate; doing it accidentally
    would push an unvalidated strategy into the live container."""
    assert "samir_stack_paper" not in STRATEGY_SETS["champion_portfolio"]


def test_samir_validation_set_includes_paper_entry_plus_champion():
    """The validation set is champion_portfolio + samir_stack_paper.
    Operator runs `--strategies samir_validation` to add the trial
    strategy alongside existing live strategies, all sharing the same
    NLV via auto-equity allocation."""
    samir_set = STRATEGY_SETS["samir_validation"]
    assert "samir_stack_paper" in samir_set
    # Every strategy in champion_portfolio must also be in samir_validation
    for s in STRATEGY_SETS["champion_portfolio"]:
        assert s in samir_set, f"{s} (in champion) missing from samir_validation"


def test_samir_stack_paper_uses_correct_strategy_class():
    entry = STRATEGY_REGISTRY["samir_stack_paper"]
    assert entry["module"] == "titan.strategies.samir_stack.strategy"
    assert entry["config_cls"] == "SamirStackConfig"
    assert entry["strategy_cls"] == "SamirStackStrategy"


def test_samir_stack_paper_v1_phases_2_and_3_disabled():
    """v1 paper deployment must NOT enable Phase 2 (futures) or Phase 3
    (bond rotation) — those layer in after a clean v1 baseline."""
    cfg_kwargs = STRATEGY_REGISTRY["samir_stack_paper"]["config_kwargs"]
    assert cfg_kwargs.get("equity_is_future", False) is False, (
        "Phase 2 (MES futures) must be off for v1 paper validation. "
        "Layer in only after 4 clean weeks; see directives/Samir-Stack "
        "Paper Validation 2026-05-12.md §10."
    )
    assert cfg_kwargs.get("bond_rotation_instruments", ()) == (), (
        "Phase 3 (bond rotation) must be off for v1 paper validation."
    )


def test_samir_stack_paper_uses_champion_research_parameters():
    """Pin the validation entry to the exact research-blessed config."""
    cfg_kwargs = STRATEGY_REGISTRY["samir_stack_paper"]["config_kwargs"]
    assert cfg_kwargs["L_max"] == 3.0
    assert cfg_kwargs["equity_weight"] == 0.10
    assert cfg_kwargs["bond_weight"] == 0.90
    assert cfg_kwargs["vol_target_annual"] == 0.08
    assert cfg_kwargs["vol_target_window"] == 30


def test_samir_stack_paper_signal_instruments_consistent_with_bar_types():
    """For each signal_*_id config, there must be a matching
    bar_type_*_d that NT can subscribe to. Catches cases where you
    add an instrument id but forget to wire its bar."""
    cfg_kwargs = STRATEGY_REGISTRY["samir_stack_paper"]["config_kwargs"]
    pairs = [
        ("equity_instrument_id", "bar_type_equity_d"),
        ("bond_instrument_id", "bar_type_bond_d"),
        ("signal_spy_id", "bar_type_spy_d"),
        ("signal_hyg_id", "bar_type_hyg_d"),
    ]
    for id_key, bar_key in pairs:
        instrument_id = cfg_kwargs.get(id_key)
        bar_type = cfg_kwargs.get(bar_key)
        if instrument_id is None and bar_type is None:
            continue  # both absent — fine (optional indicator)
        assert instrument_id is not None, f"{id_key} set but {bar_key} missing"
        assert bar_type is not None, f"{bar_key} set but {id_key} missing"
        # Sanity check: bar_type starts with the instrument_id
        assert bar_type.startswith(instrument_id + "-"), (
            f"bar_type {bar_type!r} doesn't match instrument_id {instrument_id!r}"
        )


def test_samir_stack_paper_warmup_files_listed():
    """The warmup-file mapping must include every parquet the strategy
    needs at startup, so the freshness checker warns the operator if
    one is stale before the first bar fires."""
    files = _STRATEGY_WARMUP_FILES.get("samir_stack_paper", [])
    expected = {"CSPX_D.parquet", "IEF_D.parquet", "SPY_D.parquet", "HYG_D.parquet"}
    assert set(files) == expected


def test_samir_stack_paper_warmup_parquets_exist_on_disk():
    """The warmup files must actually exist in data/. Running the
    container with a missing parquet would silently disable that
    indicator and degrade the regime score — failing here at test
    time is much louder."""
    project_root = Path(__file__).resolve().parents[1]
    files = _STRATEGY_WARMUP_FILES.get("samir_stack_paper", [])
    missing = [f for f in files if not (project_root / "data" / f).exists()]
    if missing:
        pytest.fail(
            f"Missing warmup parquets for samir_stack_paper: {missing}. "
            f"Refresh with: uv run python scripts/download_data_yfinance.py "
            f"--symbols {' '.join(f.replace('_D.parquet', '') for f in missing)}"
        )


def test_samir_stack_paper_contracts_have_required_fields():
    """Each IBContract entry needs at least secType + symbol +
    primaryExchange + currency for SMART routing to work."""
    contracts = STRATEGY_REGISTRY["samir_stack_paper"]["contracts"]
    assert len(contracts) >= 4  # CSPX + IEF + SPY + HYG
    for c in contracts:
        assert getattr(c, "secType", None), f"Missing secType on {c}"
        assert getattr(c, "symbol", None), f"Missing symbol on {c}"
        assert getattr(c, "currency", None), f"Missing currency on {c}"
        # SMART routing requires primaryExchange
        assert getattr(c, "primaryExchange", None) or getattr(c, "exchange", None), (
            f"Need exchange or primaryExchange on {c}"
        )
