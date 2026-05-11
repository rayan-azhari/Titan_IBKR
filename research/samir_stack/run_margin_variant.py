"""Margin variant of Samir-Stack vs the original 3x leveraged-ETF version.

Runs the same regime-gated 40/60 stack three ways and prints a
side-by-side stats comparison:

  A. Original — daily-rebalanced 3x ETF (3USL synthetic) at L_max=3
  B. Margin-constant-L — IBKR margin on CSPX (1x SPY UCITS) at L_max=2,
     constant target leverage with daily rebalancing
  C. Margin-drift — same but borrow stays fixed; leverage drifts. Slips
     into a margin call simulation if equity falls below maintenance.

The bond sleeve is unchanged across all three (60% IEF).

Usage:
    uv run python research/samir_stack/run_margin_variant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.margin_model import (  # noqa: E402
    constant_leverage_margin_returns,
    drift_margin_returns,
)
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
    summarize_stacked,
)
from research.samir_stack.synthetic_3x import synthetic_leveraged_returns  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _bond_60_40_with_engine(
    spy: pd.Series,
    ief: pd.Series,
    score: pd.Series,
    *,
    L_max: float,
    equity_returns_engine,
) -> pd.DataFrame:
    """Run ``run_stacked_strategy`` but inject custom equity-tier returns.

    The stacked strategy's equity sleeve is replaced with
    ``equity_returns_engine(L)`` for each tier L. The bond sleeve stays
    on IEF. Cleanest way to compare leveraged-ETF vs margin while keeping
    the regime gating, capitulation overlay, and DD circuit breaker
    identical.
    """
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=L_max,
        # Use only as many thresholds as the L_max ceiling allows.
        tier_thresholds=tuple([0.30, 0.50, 0.75][: int(L_max)]),
        leverage_ter_annual=0.0,  # ignored — engine handles its own TER
    )

    # Monkey-patch the per-tier equity-return precompute path inside
    # ``run_stacked_strategy``. That function reads
    # ``synthetic_leveraged_returns`` directly, so we wrap the public
    # call instead by passing a SPY proxy whose per-day return matches
    # what our engine would have produced. Simpler: re-implement the
    # equity-tier loop here using the engine, then call the stacked
    # strategy with a synthetic SPY series that yields the same per-tier
    # returns when fed back through ``synthetic_leveraged_returns(L=1)``.
    #
    # Actually the most robust path: monkey-patch the import. We do that
    # by replacing the symbol in the module's namespace just for this
    # call, then restoring it.
    import research.samir_stack.stacked_strategy as ss_mod

    saved = ss_mod.synthetic_leveraged_returns

    def patched(spy_series, leverage, **_kwargs):
        return equity_returns_engine(spy_series, leverage)

    ss_mod.synthetic_leveraged_returns = patched
    try:
        out = run_stacked_strategy(spy, ief, score, cfg)
    finally:
        ss_mod.synthetic_leveraged_returns = saved
    return out


def _engine_leveraged_etf(spy: pd.Series, leverage: float) -> pd.Series:
    """Original Samir engine: synthetic daily-rebalanced leveraged ETF.

    Uses 3USL-style funding (wholesale Fed Funds approx) and 0.75% TER
    when L > 1. Identical to the original `run_pipeline` behaviour.
    """
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        synthetic_leveraged_returns(
            spy,
            leverage=leverage,
            ter_annual=0.0075 if leverage > 1.0 else 0.0,
        )
        .reindex(spy.index)
        .fillna(0.0)
    )


def _engine_margin_constant_l(spy: pd.Series, leverage: float) -> pd.Series:
    """Margin engine, constant L, daily rebalancing.

    IBKR Pro funding (Fed Funds + 1.5%) and CSPX TER (0.07%) — much lower
    than 3USL but borrowing is more expensive per dollar.
    """
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        constant_leverage_margin_returns(
            spy,
            leverage=leverage,
            broker="ibkr_pro",
        )
        .reindex(spy.index)
        .fillna(0.0)
    )


def _engine_margin_drift(spy: pd.Series, leverage: float) -> pd.Series:
    """Drift engine: borrow stays fixed; leverage drifts; margin calls
    auto-deleverage to 1/maintenance_margin.

    Important caveat: this engine doesn't truly "drift" inside the
    stacked-strategy harness because the harness rebalances tier
    weights every regime change, which cancels the drift on each
    regime move. The engine still differs from constant-L because it
    skips the daily rebalance WITHIN a single tier hold. The regime
    transitions still trigger forced rebalances.
    """
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    df = drift_margin_returns(
        spy,
        initial_leverage=leverage,
        broker="ibkr_pro",
    )
    return df["equity_ret"].reindex(spy.index).fillna(0.0)


# ── Headline comparison ───────────────────────────────────────────────────


def main() -> int:
    print("Loading data...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data["hyg"],
        ief_close=data["ief"],
        tlt_close=data["tlt"],
    )
    score = regime_score_equal(panel)

    print("Running 3 variants of Samir-Stack...", flush=True)

    variants = [
        ("A: Original 3x leveraged ETF", _engine_leveraged_etf, 3.0),
        ("B: Margin constant-L=2 on CSPX", _engine_margin_constant_l, 2.0),
        ("C: Margin drift L=2 on CSPX", _engine_margin_drift, 2.0),
        # Bonus: margin at L=3 to control for leverage difference.
        ("D: Margin constant-L=3 on CSPX (control)", _engine_margin_constant_l, 3.0),
    ]

    rows = []
    margin_call_counts: dict[str, int] = {}
    for name, engine, L_max in variants:
        df = _bond_60_40_with_engine(
            data["spy"], data["ief"], score, L_max=L_max, equity_returns_engine=engine
        )
        s = summarize_stacked(df)
        s["variant"] = name
        s["L_max"] = L_max
        rows.append(s)

        # Track margin calls if the drift engine was used at L_max
        if engine is _engine_margin_drift:
            drift_df = drift_margin_returns(data["spy"], initial_leverage=L_max, broker="ibkr_pro")
            margin_call_counts[name] = int(drift_df["margin_call"].sum())

    summary = pd.DataFrame(rows).set_index("variant")
    cols = [
        "n_years",
        "L_max",
        "cagr",
        "vol",
        "sharpe",
        "max_dd",
        "calmar",
        "trades_per_year",
        "frac_in_cash",
        "avg_eq_pos_when_in",
    ]
    print("\n" + "=" * 110)
    print("Side-by-side: Samir-Stack with leveraged-ETF vs IBKR margin equity sleeve")
    print("Bond sleeve identical (60% IEF) across all variants.")
    print("=" * 110)
    print(summary[cols].to_string())

    if margin_call_counts:
        print("\nMargin-call events (drift engine only, on the underlying SPY series):")
        for k, v in margin_call_counts.items():
            print(f"  {k}: {v} margin-call days over {summary.loc[k, 'n_years']:.1f} years")

    # ── Standalone underlying-leverage comparison (no regime gating) ──
    # Useful sanity check: shows the pure cost-structure difference
    # between leveraged ETF and margin at the same L, holding underlying
    # = SPY for the entire period.
    print("\n" + "=" * 110)
    print("Standalone holding (no regime gate, full period, 100% leveraged equity):")
    print("=" * 110)
    spy = data["spy"]
    standalone_rows = []
    for label, engine, L in [
        ("3USL synthetic L=3", _engine_leveraged_etf, 3.0),
        ("Margin const-L L=2", _engine_margin_constant_l, 2.0),
        ("Margin const-L L=3", _engine_margin_constant_l, 3.0),
        ("Margin drift   L=2", _engine_margin_drift, 2.0),
        ("Margin drift   L=3", _engine_margin_drift, 3.0),
        ("Buy-hold SPY (control)", _engine_leveraged_etf, 1.0),
    ]:
        rets = engine(spy, L)
        eq = (1.0 + rets.fillna(0.0)).cumprod()
        n_years = len(eq) / 252.0
        cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)
        vol = float(rets.std() * np.sqrt(252))
        peak = eq.cummax()
        maxdd = float(((eq - peak) / peak).min())
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
        standalone_rows.append(
            {
                "instrument": label,
                "cagr": round(cagr, 4),
                "vol": round(vol, 4),
                "sharpe": round(sharpe, 3),
                "max_dd": round(maxdd, 4),
                "calmar": round(cagr / abs(maxdd) if maxdd < -1e-9 else 0.0, 3),
            }
        )
    print(pd.DataFrame(standalone_rows).set_index("instrument").to_string())

    # Save the headline comparison
    out_path = REPORTS_DIR / "margin_variant_comparison.csv"
    summary[cols].to_csv(out_path)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
