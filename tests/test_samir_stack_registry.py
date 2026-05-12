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
    """The warmup files must actually exist in data/ on the deployment
    machine. Running the container with a missing parquet would silently
    disable that indicator and degrade the regime score.

    In CI ``data/`` is empty (parquets are gitignored / refreshed
    locally), so we SKIP rather than FAIL. The check still runs
    locally before deploy and via the operator's pre-deployment
    checklist (see directives/Samir-Stack Paper Validation 2026-05-12.md
    §3 item 3).
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    files = _STRATEGY_WARMUP_FILES.get("samir_stack_paper", [])
    if not data_dir.exists() or not any(data_dir.glob("*.parquet")):
        pytest.skip("data/ has no parquets — CI environment, deferred to local pre-deploy check")
    missing = [f for f in files if not (data_dir / f).exists()]
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


# ── Per-strategy weighted equity allocation ──────────────────────────


def test_samir_stack_paper_uses_weight_above_one():
    """Samir's 10/90 split puts only 10% of its share into CSPX. At the
    default equal-allocation, the equity sleeve would be too small to
    trade above IBKR's $4 minimum commission cost-effectively. Weighted
    above 1.0 to fix this."""
    weight = STRATEGY_REGISTRY["samir_stack_paper"].get("weight", 1.0)
    assert weight > 1.0, (
        f"samir_stack_paper has weight={weight}; expected > 1.0 to "
        f"compensate for the 10/90 split's small equity sleeve"
    )


def test_other_strategies_use_default_weight():
    """All non-Samir trading strategies should use the implicit default
    weight (1.0). If you change one, document it explicitly."""
    expected_default = [
        "mr_audjpy",
        "bond_equity_ihyu_cspx",
        "bond_equity_ihyg_vusd",
        "bond_equity_ihyg_eimi",
    ]
    for name in expected_default:
        weight = STRATEGY_REGISTRY[name].get("weight", 1.0)
        assert weight == 1.0, (
            f"{name} has explicit weight={weight}; default is 1.0. "
            f"If intentional, update this test."
        )


def weighted_allocation_pure(nlv_usd: float, weights: dict[str, float]) -> dict[str, float]:
    """Pure-function copy of the weighted-allocation math in
    ``_auto_allocate_initial_equity``. Tested here so the math is
    locked in even though the surrounding function calls into ibapi."""
    total_w = sum(weights.values())
    if total_w <= 0:
        # Fallback: equal allocation
        n = len(weights)
        return dict.fromkeys(weights, nlv_usd / n) if n > 0 else {}
    return {name: nlv_usd * (w / total_w) for name, w in weights.items()}


def test_weighted_allocation_equal_weights_matches_legacy():
    """When all weights are 1.0, output matches the old equal-divide
    behaviour exactly."""
    nlv = 30_000.0
    weights = {f"s{i}": 1.0 for i in range(4)}
    out = weighted_allocation_pure(nlv, weights)
    assert all(abs(v - 7500.0) < 0.01 for v in out.values())


def test_weighted_allocation_proportional():
    """Weights divide NLV in proportion. 4 strategies with weights
    1+1+1+3 → ratios 1/6, 1/6, 1/6, 3/6 = 50%."""
    nlv = 30_000.0
    weights = {"a": 1.0, "b": 1.0, "c": 1.0, "samir": 3.0}
    out = weighted_allocation_pure(nlv, weights)
    assert abs(out["a"] - 5000.0) < 0.01
    assert abs(out["samir"] - 15000.0) < 0.01
    assert abs(sum(out.values()) - nlv) < 0.01


def test_weighted_allocation_zero_total_falls_back_to_equal():
    """Defensive: if all weights are 0, don't divide-by-zero. Fall back
    to equal allocation."""
    nlv = 30_000.0
    weights = {"a": 0.0, "b": 0.0}
    out = weighted_allocation_pure(nlv, weights)
    assert abs(out["a"] - 15000.0) < 0.01
    assert abs(out["b"] - 15000.0) < 0.01


def test_samir_validation_set_allocation_at_30k_paper():
    """End-to-end check at a realistic paper NLV: with 4 default-weight
    + 1 weight=3 strategy, Samir gets 3/7 of NLV."""
    nlv = 30_000.0
    weights = {
        "mr_audjpy": 1.0,
        "bond_equity_ihyu_cspx": 1.0,
        "bond_equity_ihyg_vusd": 1.0,
        "bond_equity_ihyg_eimi": 1.0,
        "samir_stack_paper": 3.0,
    }
    out = weighted_allocation_pure(nlv, weights)
    # Samir gets 3/7 ≈ 42.9% → ~$12,857
    assert 12_000 < out["samir_stack_paper"] < 14_000
    # Each other strategy gets 1/7 ≈ 14.3% → ~$4,286
    for name in ["mr_audjpy", "bond_equity_ihyu_cspx"]:
        assert 4_000 < out[name] < 5_000
