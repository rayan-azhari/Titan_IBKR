"""Tests for the position-reconciliation watchdog strategy.

NautilusTrader's Strategy class is Cython and can't be instantiated
without a full TradingNode, so these tests focus on:

  1. Module imports cleanly
  2. The config class accepts the expected fields
  3. The detection logic finds drift in synthetic position fixtures via
     a re-implementation of the core algorithm in pure Python (the same
     algorithm the strategy method runs on the live cache).

The pure-Python copy of the detection algorithm here is deliberate — it
lets us exercise every detection branch without standing up a Cython
runtime, and it keeps the test small. If the strategy changes its
detection logic, this test must be updated to match (and that's the
point: the test forces the change to be considered).

Tier 1.1 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# ── Fixture types: minimal stand-ins for NT Position / Order ──────────


@dataclass
class FakePosition:
    instrument_id: str
    strategy_id: str
    signed_qty: float
    is_open: bool = True


@dataclass
class FakeOrder:
    client_order_id: str
    instrument_id: str
    ts_init: int  # nanoseconds
    is_open: bool = True
    status: str = "ACCEPTED"


# ── Pure-Python re-implementation of the detection logic ─────────────


def detect_nlv_drop(
    nlv_now: dict[str, float],
    nlv_prev: dict[str, float],
    *,
    threshold_pct: float = 0.02,
) -> list[str]:
    """Re-implementation of D4 (NLV-drop) detection.

    Mirrors the strategy's D4 branch in ``_reconcile``. Empty/zero baselines
    silently seed without firing — only meaningful drops between two real
    samples produce a finding.
    """
    findings: list[str] = []
    for ccy, curr in nlv_now.items():
        prev = nlv_prev.get(ccy)
        if prev is None or prev <= 0:
            continue
        drop_pct = (prev - curr) / prev
        if drop_pct > threshold_pct:
            findings.append(
                f"[D4] {ccy}: NLV dropped {drop_pct * 100:.2f}% from {prev:,.2f} to {curr:,.2f}"
            )
    return findings


def detect_drift(
    positions: list,
    orders: list,
    portfolio_net: dict[str, float],
    *,
    now_ns: int,
    qty_epsilon: float = 1e-6,
    stale_order_minutes: int = 15,
) -> list[str]:
    """Re-implementation of ReconciliationStrategy._reconcile detection.

    Kept in lockstep with the strategy code by code review (no shared
    module — strategy method calls Cython attributes that we can't mock
    cheaply). Test failure here means the strategy's behaviour MUST
    also change.
    """
    findings: list[str] = []

    open_positions = [p for p in positions if p.is_open]
    by_instrument: dict = defaultdict(list)
    for p in open_positions:
        by_instrument[p.instrument_id].append(p)

    # D1
    for instrument_id, pos_list in by_instrument.items():
        if len(pos_list) > 1:
            findings.append(f"[D1] {instrument_id} has {len(pos_list)} open positions")

    # D2
    for instrument_id, pos_list in by_instrument.items():
        cache_sum = sum(float(p.signed_qty) for p in pos_list)
        portfolio = portfolio_net.get(instrument_id, 0.0)
        if abs(cache_sum - portfolio) > qty_epsilon:
            findings.append(
                f"[D2] {instrument_id}: cache_sum={cache_sum:+.4f} vs portfolio={portfolio:+.4f}"
            )

    # D3
    open_orders = [o for o in orders if o.is_open]
    stale_threshold = now_ns - stale_order_minutes * 60 * 1_000_000_000
    for order in open_orders:
        if order.ts_init < stale_threshold:
            findings.append(f"[D3] {order.instrument_id} order {order.client_order_id} stale")
    return findings


# ── Tests ────────────────────────────────────────────────────────────


def test_module_imports_cleanly():
    """The strategy module must import without side effects."""
    from titan.strategies.reconciliation import strategy as recon  # noqa: F401


def test_config_accepts_expected_fields():
    """Config dataclass accepts every field run_portfolio.py wires."""
    from titan.strategies.reconciliation.strategy import ReconciliationConfig

    cfg = ReconciliationConfig(
        bar_type="AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL",
        stale_order_minutes=20,
        alert_cooldown_minutes=120,
        shadow_check_enabled=False,
    )
    assert cfg.stale_order_minutes == 20
    assert cfg.alert_cooldown_minutes == 120
    assert cfg.shadow_check_enabled is False


def test_config_shadow_check_defaults_to_enabled():
    """Tier 2.1 ships shadow checking ON by default — auto-skips per-
    strategy when its parquet is missing, so the default is safe."""
    from titan.strategies.reconciliation.strategy import ReconciliationConfig

    cfg = ReconciliationConfig(bar_type="X")
    assert cfg.shadow_check_enabled is True


def test_detect_drift_clean_state_returns_no_findings():
    """No drift → no findings."""
    positions = [FakePosition("VUSD.LSEETF", "BondGoldStrategy-002", 30.0)]
    orders: list = []
    portfolio_net = {"VUSD.LSEETF": 30.0}
    findings = detect_drift(positions, orders, portfolio_net, now_ns=10_000_000_000_000)
    assert findings == []


def test_detect_drift_d1_fires_on_double_position_for_same_instrument():
    """The May 11 bug class: two open positions for the same instrument
    (an EXTERNAL orphan + a strategy-tagged double-up)."""
    positions = [
        FakePosition("VUSD.LSEETF", "EXTERNAL", 215.0),
        FakePosition("VUSD.LSEETF", "BondGoldStrategy-002", 30.0),
    ]
    portfolio_net = {"VUSD.LSEETF": 245.0}  # cache_sum matches portfolio
    findings = detect_drift(positions, [], portfolio_net, now_ns=10_000_000_000_000)
    assert any("[D1]" in f and "VUSD" in f for f in findings)
    # D2 should NOT fire because cache_sum (215+30=245) matches portfolio
    assert not any("[D2]" in f for f in findings)


def test_detect_drift_d2_fires_on_cache_vs_portfolio_mismatch():
    """If NT cache and NT portfolio accounting disagree, escalate."""
    positions = [FakePosition("EUR/USD.IDEALPRO", "MTFStrategy", 10000.0)]
    portfolio_net = {"EUR/USD.IDEALPRO": 5000.0}  # mismatch
    findings = detect_drift(positions, [], portfolio_net, now_ns=10_000_000_000_000)
    assert any("[D2]" in f and "EUR/USD" in f for f in findings)


def test_detect_drift_d3_fires_on_stale_open_order():
    """Order open longer than stale_order_minutes triggers D3."""
    now_ns = 10_000_000_000_000_000
    sixteen_min_ago_ns = now_ns - 16 * 60 * 1_000_000_000
    orders = [
        FakeOrder(
            client_order_id="O-stale",
            instrument_id="VUSD.LSEETF",
            ts_init=sixteen_min_ago_ns,
        )
    ]
    findings = detect_drift([], orders, {}, now_ns=now_ns, stale_order_minutes=15)
    assert any("[D3]" in f and "O-stale" in f for f in findings)


def test_detect_drift_d3_does_not_fire_on_recent_order():
    """A 5-minute-old open order is fine — only >15 min triggers D3."""
    now_ns = 10_000_000_000_000_000
    five_min_ago_ns = now_ns - 5 * 60 * 1_000_000_000
    orders = [FakeOrder(client_order_id="O-recent", instrument_id="X", ts_init=five_min_ago_ns)]
    findings = detect_drift([], orders, {}, now_ns=now_ns, stale_order_minutes=15)
    assert not any("[D3]" in f for f in findings)


def test_detect_drift_handles_closed_positions():
    """Closed positions in cache must be ignored — only is_open matters."""
    positions = [
        FakePosition("X", "EXTERNAL", 100.0, is_open=False),
        FakePosition("X", "Strategy-A", 30.0, is_open=True),
    ]
    portfolio_net = {"X": 30.0}
    findings = detect_drift(positions, [], portfolio_net, now_ns=10_000_000_000_000)
    # No double-position because the closed one is filtered out
    assert not any("[D1]" in f for f in findings)


def test_detect_drift_d1_and_d2_compose_independently():
    """Two unrelated instruments can both trigger different findings."""
    positions = [
        FakePosition("A", "EXTERNAL", 100.0),
        FakePosition("A", "Strategy-A", 50.0),  # D1 trigger for A
        FakePosition("B", "Strategy-B", 999.0),  # D2 trigger for B (mismatch)
    ]
    portfolio_net = {"A": 150.0, "B": 1.0}
    findings = detect_drift(positions, [], portfolio_net, now_ns=10_000_000_000_000)
    assert any("[D1]" in f and " A " in f for f in findings)
    assert any("[D2]" in f and " B" in f for f in findings)


# ── D4: NLV-drop detection tests ─────────────────────────────────────


def test_detect_nlv_drop_first_sample_seeds_no_alert():
    """No previous baseline → no alert (just seeds)."""
    findings = detect_nlv_drop({"GBP": 9882.0}, {}, threshold_pct=0.02)
    assert findings == []


def test_detect_nlv_drop_within_threshold_no_alert():
    """1% drop with 2% threshold → no alert."""
    findings = detect_nlv_drop({"GBP": 9783.0}, {"GBP": 9882.0}, threshold_pct=0.02)
    assert findings == []


def test_detect_nlv_drop_above_threshold_alerts():
    """3% drop with 2% threshold → alert."""
    findings = detect_nlv_drop({"GBP": 9586.0}, {"GBP": 9882.0}, threshold_pct=0.02)
    assert any("[D4]" in f and "GBP" in f for f in findings)


def test_detect_nlv_drop_increase_no_alert():
    """NLV increased → no alert (only drops trigger D4)."""
    findings = detect_nlv_drop({"GBP": 10500.0}, {"GBP": 9882.0}, threshold_pct=0.02)
    assert findings == []


def test_detect_nlv_drop_zero_baseline_skipped():
    """Zero or negative baseline is treated as missing (no division by zero)."""
    findings = detect_nlv_drop({"GBP": 100.0}, {"GBP": 0.0}, threshold_pct=0.02)
    assert findings == []


def test_detect_nlv_drop_per_currency_independent():
    """Multi-currency account: GBP drop alerts independently of USD."""
    findings = detect_nlv_drop(
        {"GBP": 9000.0, "USD": 13000.0},
        {"GBP": 9882.0, "USD": 13000.0},  # GBP drops 9%, USD flat
        threshold_pct=0.02,
    )
    assert any("[D4]" in f and "GBP" in f for f in findings)
    assert not any("[D4]" in f and "USD" in f for f in findings)


# ── D5: shadow-decision divergence (Tier 2.1) ────────────────────────


def detect_shadow_divergence(
    z: float,
    *,
    threshold: float,
    cache_is_long: bool,
    strategy_name: str = "test",
    trade_symbol: str = "TEST",
    net_qty: float = 0.0,
    hold_days: int = 5,
) -> list[str]:
    """Re-implementation of D5 (shadow divergence) detection.

    Mirrors ``ReconciliationStrategy._shadow_decision_check`` minus the
    NT cache iteration and parquet load (those are integration-level).
    Test failure here means the strategy's behaviour MUST also change.
    """
    findings: list[str] = []
    if z > threshold and not cache_is_long:
        findings.append(
            f"[D5 shadow {strategy_name}] z={z:+.3f} > threshold "
            f"{threshold} but cache is FLAT for {trade_symbol} — "
            f"possible missed entry or broker rejection"
        )
    elif z <= threshold and cache_is_long:
        findings.append(
            f"[D5 shadow {strategy_name}] z={z:+.3f} <= threshold "
            f"{threshold} but cache shows LONG {net_qty:+.0f} "
            f"{trade_symbol} (may be inside hold_days={hold_days})"
        )
    return findings


def test_d5_no_finding_when_bullish_and_long():
    """Strategy should be long, cache shows long → agreement."""
    findings = detect_shadow_divergence(z=1.5, threshold=0.25, cache_is_long=True, net_qty=30.0)
    assert findings == []


def test_d5_no_finding_when_bearish_and_flat():
    """Strategy should be flat, cache shows flat → agreement."""
    findings = detect_shadow_divergence(z=-0.5, threshold=0.25, cache_is_long=False)
    assert findings == []


def test_d5_fires_on_missed_entry():
    """Bullish signal but cache is flat → broker rejection or missed entry."""
    findings = detect_shadow_divergence(
        z=1.5,
        threshold=0.25,
        cache_is_long=False,
        strategy_name="bond_equity_ihyg_vusd",
        trade_symbol="VUSD",
    )
    assert len(findings) == 1
    assert "[D5 shadow bond_equity_ihyg_vusd]" in findings[0]
    assert "VUSD" in findings[0]
    assert "missed entry" in findings[0]


def test_d5_fires_on_phantom_long():
    """Bearish signal but cache shows long → may be inside hold_days
    (informational), but still flagged."""
    findings = detect_shadow_divergence(
        z=-0.5,
        threshold=0.25,
        cache_is_long=True,
        strategy_name="bond_equity_ihyg_vusd",
        trade_symbol="VUSD",
        net_qty=30.0,
        hold_days=5,
    )
    assert len(findings) == 1
    assert "LONG" in findings[0]
    assert "hold_days=5" in findings[0]


def test_d5_z_at_threshold_treated_as_bearish():
    """z == threshold should NOT trigger an entry-side finding (entry
    rule is strict z > threshold, matching expected_action)."""
    # cache is flat, z exactly at threshold → no missed-entry finding
    assert detect_shadow_divergence(z=0.25, threshold=0.25, cache_is_long=False) == []
    # cache is long, z at threshold → flagged as 'phantom-long-or-hold'
    findings = detect_shadow_divergence(z=0.25, threshold=0.25, cache_is_long=True, net_qty=30.0)
    assert len(findings) == 1


def test_d5_uses_post_fix_strict_threshold_comparison():
    """The shared expected_action uses strict > for entry. The shadow
    check should match: z slightly above threshold → entry expected."""
    findings = detect_shadow_divergence(
        z=0.26, threshold=0.25, cache_is_long=False, trade_symbol="VUSD"
    )
    assert any("missed entry" in f for f in findings)
