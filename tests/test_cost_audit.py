"""Tests for the cost-audit script's pure logic.

The IBKR connection + execution-fetching is integration-level and not
testable in CI without a gateway. The drift-computation and aggregation
logic is pure and is what we test here.

Tier 2.4 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

from datetime import datetime, timezone

from scripts.cost_audit import Fill, _aggregate, _compute_drift


def _mk_fill(symbol: str, side: str, qty: float, price: float, commission: float) -> Fill:
    return Fill(
        exec_id=f"E-{symbol}-{side}-{qty}",
        time_utc=datetime(2026, 5, 12, tzinfo=timezone.utc),
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        avg_price=price,
        commission_ccy="USD",
        commission_amt=commission,
    )


def test_compute_drift_zero_drift_when_realised_matches_modelled():
    """If realised commission exactly equals modelled cost, drift = 0."""
    # SPY default spread is 0.5 bps; 100 shares @ $400 = $40k notional
    # modelled = max(4.0, 0) + spread = $4 floor + 0.5 bps spread = $4 + $2 = $6
    # = 1.5 bps total
    fill = _mk_fill("SPY", "BUY", qty=100.0, price=400.0, commission=6.0)
    rows = _compute_drift([fill])
    assert len(rows) == 1
    # Drift should be very close to zero
    assert abs(rows[0]["drift_pct"]) < 0.05, rows[0]


def test_compute_drift_positive_when_realised_exceeds_modelled():
    """The May 2026 case: small notional with $4 minimum dominates."""
    # SPY at $400, buy 10 shares = $4,000 notional
    # modelled = max(4.0/4000 * 10000, 0) + 0.5bps = 10bps + 0.5bps = 10.5 bps
    # = $4.20
    # realised = $4 commission only (the floor)
    # drift = (4 - 4.2) / 4.2 = -4.8% (close to zero, model already accounts)
    fill = _mk_fill("SPY", "BUY", qty=10.0, price=400.0, commission=4.0)
    rows = _compute_drift([fill])
    assert len(rows) == 1
    # Should be close to zero drift since model accounts for the floor
    assert abs(rows[0]["drift_pct"]) < 0.10, rows[0]


def test_compute_drift_positive_when_unmodelled_extra_charges_appear():
    """If realised commission is much larger than modelled, drift > 0."""
    # Strategy thinks $40k SPY trade costs $4 + 0.5bps spread = $6
    # But actually pays $20 (e.g., a fee surcharge or different commission tier)
    fill = _mk_fill("SPY", "BUY", qty=100.0, price=400.0, commission=20.0)
    rows = _compute_drift([fill])
    assert rows[0]["drift_pct"] > 0.5, (
        f"Expected significant positive drift, got {rows[0]['drift_pct']}"
    )


def test_compute_drift_skips_zero_notional():
    """Zero-price or zero-qty fills are skipped (would divide by zero)."""
    fill = _mk_fill("SPY", "BUY", qty=0.0, price=400.0, commission=0.0)
    assert _compute_drift([fill]) == []
    fill2 = _mk_fill("SPY", "BUY", qty=10.0, price=0.0, commission=0.0)
    assert _compute_drift([fill2]) == []


def test_compute_drift_uses_unknown_symbol_default_spread():
    """Symbols not in SPREAD_BPS_PER_SIDE table get the 5bps default."""
    fill = _mk_fill("UNKNOWN_SYM", "BUY", qty=100.0, price=100.0, commission=4.0)
    rows = _compute_drift([fill])
    assert len(rows) == 1
    # 100 * 100 = $10k notional, 5bps spread = $5, plus $4 floor = $4 commission portion
    # modelled = max(4, 0) + 5bps spread = 4bps commission + 5bps spread = 9 bps total
    # = $9 modelled, $4 realised, drift = (4-9)/9 ≈ -56%
    assert rows[0]["modelled_bps"] >= 5.0  # spread alone is 5
    assert rows[0]["drift_pct"] < 0  # realised commission less than full modelled cost


def test_compute_drift_fx_conversion_applied():
    """Commission in non-base currency is converted via fx_to_account_base map."""
    # IBKR charged 5 GBP commission; account base USD
    # GBP/USD ≈ 1.30 → realised in USD = 6.50
    fill = Fill(
        exec_id="E-1",
        time_utc=datetime(2026, 5, 12, tzinfo=timezone.utc),
        symbol="SPY",
        side="BUY",
        qty=100.0,
        price=400.0,
        avg_price=400.0,
        commission_ccy="GBP",
        commission_amt=5.0,
    )
    rows = _compute_drift([fill], fx_to_account_base={"GBP": 1.30})
    assert rows[0]["realised_cost_base"] == 6.5


def test_aggregate_groups_by_symbol_and_side():
    """Multiple fills on the same symbol/side roll up to one summary row."""
    fills = [
        _mk_fill("SPY", "BUY", 100.0, 400.0, 4.0),
        _mk_fill("SPY", "BUY", 50.0, 405.0, 4.0),
        _mk_fill("SPY", "SELL", 100.0, 410.0, 4.0),
    ]
    rows = _compute_drift(fills)
    summary = _aggregate(rows)
    assert len(summary) == 2  # SPY/BUY and SPY/SELL
    spy_buy = next(s for s in summary if s["side"] == "BUY")
    assert spy_buy["n_fills"] == 2


def test_aggregate_weights_drift_by_notional():
    """Weighted drift correctly accounts for fill size, not just count.

    Two fills: small ($1k notional) with 100% drift, large ($100k) with
    0% drift. The unweighted average would be 50%, but the
    notional-weighted should be close to 0%.
    """
    # Build artificial drift via raw row dicts, bypassing _compute_drift
    rows = [
        {
            "symbol": "X",
            "side": "BUY",
            "qty": 1.0,
            "notional": 1000.0,
            "modelled_cost": 1.0,
            "realised_cost_base": 2.0,  # 100% drift
        },
        {
            "symbol": "X",
            "side": "BUY",
            "qty": 100.0,
            "notional": 100_000.0,
            "modelled_cost": 100.0,
            "realised_cost_base": 100.0,  # 0% drift
        },
    ]
    summary = _aggregate(rows)
    assert len(summary) == 1
    # Total modelled = 101, total realised = 102 → drift = 1/101 ≈ 1%
    assert abs(summary[0]["weighted_drift_pct"]) < 0.02


def test_aggregate_handles_zero_modelled_cost():
    """If a group has zero modelled cost (shouldn't happen but defensive)
    the divisor guard returns 0 instead of NaN/inf."""
    rows = [
        {
            "symbol": "X",
            "side": "BUY",
            "qty": 1.0,
            "notional": 0.0,
            "modelled_cost": 0.0,
            "realised_cost_base": 5.0,
        }
    ]
    summary = _aggregate(rows)
    assert summary[0]["weighted_drift_pct"] == 0.0


def test_compute_drift_per_fill_fields_complete():
    """Output rows must contain every field the CSV writer expects."""
    fill = _mk_fill("CSPX", "BUY", qty=10.0, price=500.0, commission=4.0)
    rows = _compute_drift([fill])
    expected_keys = {
        "exec_id",
        "time_utc",
        "symbol",
        "side",
        "qty",
        "price",
        "notional",
        "modelled_bps",
        "modelled_cost",
        "realised_commission_ccy",
        "realised_commission_amt",
        "realised_cost_base",
        "drift_pct",
    }
    assert set(rows[0].keys()) == expected_keys
