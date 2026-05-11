"""run_bond_equity_wfo_realistic.py — re-run bond_equity WFO with realistic IBKR cost model.

The original `run_bond_equity_wfo.py` uses a flat 5 bps cost per position
transition, which is reasonable for liquid US-listed ETFs at retail tiered
pricing on positions >= $15k. But the live champion strategies are running
at $3,427 USD per-strategy equity (auto-allocated from the $10,065 GBP NLV
across 4 strategies), where IBKR's $4 USD minimum commission alone is 117
bps per fill — 23x the modelled 5 bps.

This script re-runs the same WFO with the realistic cost model:

    cost_per_fill_bps = max(
        min_commission_usd / notional_usd * 10_000,
        per_share_bps,
    ) + spread_bps_per_side

and sweeps `notional_usd` to show the break-even assumption.

Run:
    uv run python research/cross_asset/run_bond_equity_wfo_realistic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cross_asset.run_bond_equity_wfo import (  # noqa: E402
    _position_with_hold,
    load_daily,
)
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

# Per-instrument typical bid-ask spread on IBKR LSE listings, in bps PER SIDE.
# Values derived from IBKR market-data observation; confirm against your
# own tick stream before relying on these for sizing decisions.
SPREAD_BPS_PER_SIDE = {
    "CSPX": 2.5,  # Liquid S&P UCITS, 5 bps round-trip
    "VUSD": 2.5,  # Vanguard S&P USD line, similar to CSPX
    "VUSA": 2.5,
    "EIMI": 6.0,  # iShares EM IMI, less liquid → 12 bps round-trip
    "EMIM": 6.0,
    "GLD": 1.0,
    "IGLT": 5.0,
    "SPY": 0.5,
    "QQQ": 0.5,
}


def realistic_cost_bps(
    notional_usd: float,
    *,
    spread_bps_per_side: float,
    min_commission_usd: float = 4.0,
    per_share_bps: float = 0.0,
) -> float:
    """Per-fill cost in bps for an order of given USD notional.

    IBKR cost = max(commission_floor, per_share_rate * shares).
    The floor dominates for small notionals; per_share dominates above
    ~$10k notional at the 0.35 bps tier.

    Spread is added on top — it's the half-spread paid implicitly via
    the bid/ask, which the commission report never shows.
    """
    if notional_usd <= 0:
        return 0.0
    floor_bps = min_commission_usd / notional_usd * 10_000
    commission_bps = max(floor_bps, per_share_bps)
    return commission_bps + spread_bps_per_side


def run_wfo_realistic(
    bond_close: pd.Series,
    target_close: pd.Series,
    *,
    target_symbol: str,
    notional_usd: float,
    min_commission_usd: float = 4.0,
    per_share_bps: float = 0.0,
    lookback: int = 20,
    hold_days: int = 20,
    threshold: float = 0.50,
    is_days: int = 504,
    oos_days: int = 126,
) -> dict:
    """WFO with realistic cost. Mirrors `run_bond_wfo` but uses
    `realistic_cost_bps` per transition instead of a flat constant.
    """
    spread_bps = SPREAD_BPS_PER_SIDE.get(target_symbol, 5.0)
    cost_bps_per_side = realistic_cost_bps(
        notional_usd,
        spread_bps_per_side=spread_bps,
        min_commission_usd=min_commission_usd,
        per_share_bps=per_share_bps,
    )

    bond_mom = np.log(bond_close / bond_close.shift(lookback)).dropna()
    common = bond_mom.index.intersection(target_close.index)
    bond_mom = bond_mom.reindex(common)
    target = target_close.reindex(common)
    n = len(common)
    if n < is_days + oos_days:
        return {"error": f"Insufficient data: {n} bars"}

    mom_vals = bond_mom.values
    target_vals = target.values
    target_ret = np.empty(n)
    target_ret[0] = 0.0
    target_ret[1:] = (target_vals[1:] - target_vals[:-1]) / target_vals[:-1]

    folds = []
    stitched: list[np.ndarray] = []
    fold_idx = 0
    oos_start = is_days

    while oos_start + oos_days <= n:
        is_start = oos_start - is_days
        is_mom = mom_vals[is_start:oos_start]
        is_mean = float(np.nanmean(is_mom))
        is_std = float(np.nanstd(is_mom)) or 1.0

        oos_end = oos_start + oos_days
        oos_mom = mom_vals[oos_start:oos_end]
        oos_z = (oos_mom - is_mean) / is_std
        oos_pos = _position_with_hold(oos_z, threshold, hold_days)
        oos_ret = target_ret[oos_start:oos_end]
        oos_pos_lagged = np.concatenate(([0.0], oos_pos[:-1]))
        oos_trans = np.zeros(oos_days)
        oos_trans[1:] = np.abs(oos_pos_lagged[1:] - oos_pos_lagged[:-1])
        oos_strat = oos_ret * oos_pos_lagged - oos_trans * cost_bps_per_side / 10_000

        eq = np.cumprod(1 + oos_strat)
        hwm = np.maximum.accumulate(eq)
        dd = float(((eq - hwm) / hwm).min()) if len(eq) > 0 else 0.0

        folds.append(
            {
                "fold": fold_idx,
                "oos_start": common[oos_start].strftime("%Y-%m-%d"),
                "oos_end": common[min(oos_end - 1, n - 1)].strftime("%Y-%m-%d"),
                "oos_sharpe": round(sharpe(oos_strat, periods_per_year=BARS_PER_YEAR["D"]), 3),
                "oos_trades": int(np.sum(oos_trans > 0)),
                "oos_dd_pct": round(dd * 100, 2),
            }
        )
        stitched.append(oos_strat)
        fold_idx += 1
        oos_start += oos_days

    all_oos = np.concatenate(stitched) if stitched else np.array([])
    if len(all_oos) >= 20:
        st_sh = sharpe(all_oos, periods_per_year=BARS_PER_YEAR["D"])
        ci_lo, ci_hi = bootstrap_sharpe_ci(
            all_oos, periods_per_year=BARS_PER_YEAR["D"], n_resamples=1000, seed=42
        )
        eq = np.cumprod(1 + all_oos)
        hwm = np.maximum.accumulate(eq)
        st_dd = float(((eq - hwm) / hwm).min())
        st_ret = float(np.mean(all_oos) * 252)
    else:
        st_sh = ci_lo = ci_hi = st_dd = st_ret = 0.0

    fold_df = pd.DataFrame(folds)
    return {
        "cost_bps_per_side": round(cost_bps_per_side, 2),
        "spread_bps_per_side": spread_bps,
        "commission_floor_bps": round(min_commission_usd / notional_usd * 10_000, 2),
        "stitched_sharpe": round(st_sh, 3),
        "sharpe_ci_95_lo": round(ci_lo, 3),
        "sharpe_ci_95_hi": round(ci_hi, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "stitched_ret_pct": round(st_ret * 100, 2),
        "n_folds": len(folds),
        "pct_positive": round((fold_df["oos_sharpe"] > 0).mean(), 3) if len(fold_df) > 0 else 0.0,
        "total_trades": int(fold_df["oos_trades"].sum()) if len(fold_df) > 0 else 0,
    }


# Live deployed strategies (signal_symbol, target_symbol, lookback, hold, threshold)
LIVE_DEPLOYMENTS = [
    ("IHYU", "CSPX", 20, 20, 0.50),
    ("IHYG", "VUSD", 5, 5, 0.25),
    ("IHYG", "EIMI", 5, 5, 0.25),
]

# Notional sweep — covers current live ($3.4k auto-alloc), original design
# ($10k), and what's needed to bring per-fill cost back to ~5 bps ($80k+).
NOTIONAL_SWEEP_USD = [3_427, 10_000, 25_000, 50_000, 100_000]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--min-commission",
        type=float,
        default=4.0,
        help="IBKR per-fill minimum (USD). Paper=4.0; Pro tier=1.0; tiered Lite often 4.0+",
    )
    args = p.parse_args()

    print(
        f"\nRealistic-cost WFO for live bond_equity strategies\n"
        f"min_commission_usd = ${args.min_commission:.2f}\n"
        f"{'=' * 110}"
    )

    rows = []
    for signal_sym, target_sym, lb, hd, thr in LIVE_DEPLOYMENTS:
        bond = load_daily(signal_sym)
        target = load_daily(target_sym)
        for notional in NOTIONAL_SWEEP_USD:
            res = run_wfo_realistic(
                bond,
                target,
                target_symbol=target_sym,
                notional_usd=notional,
                min_commission_usd=args.min_commission,
                lookback=lb,
                hold_days=hd,
                threshold=thr,
            )
            if "error" in res:
                continue
            rows.append(
                {
                    "strategy": f"{signal_sym}->{target_sym}",
                    "notional_usd": notional,
                    "cost_bps/side": res["cost_bps_per_side"],
                    "comm_floor_bps": res["commission_floor_bps"],
                    "spread_bps": res["spread_bps_per_side"],
                    "Sharpe": res["stitched_sharpe"],
                    "CI95_lo": res["sharpe_ci_95_lo"],
                    "CI95_hi": res["sharpe_ci_95_hi"],
                    "DD_pct": res["stitched_dd_pct"],
                    "ret_pct_yr": res["stitched_ret_pct"],
                    "%pos_folds": res["pct_positive"],
                    "trades": res["total_trades"],
                }
            )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Bring in the documented numbers from the legacy 5 bps run for direct comparison.
    print(f"\n{'=' * 110}")
    print("Documented Sharpe under modelled 5 bps (from project memory):")
    documented = {
        "IHYU->CSPX": "+0.68 (CI lo +0.026)",
        "IHYG->VUSD": "+1.16 (CI lo +0.47)",
        "IHYG->EIMI": "+0.97 (CI lo +0.23)",
    }
    for k, v in documented.items():
        print(f"  {k:<15} Sharpe = {v}")

    # Deployment gate: which (strategy, notional) combinations still pass
    # CI lower bound > 0?
    print(f"\n{'=' * 110}")
    print("Deployment gate (CI lower bound > 0 = passes bootstrap test):")
    df["passes_gate"] = df["CI95_lo"] > 0.0
    pivot = df.pivot_table(
        index="strategy", columns="notional_usd", values="passes_gate", aggfunc="first"
    )
    print(pivot.to_string())

    # Save full result for the report
    out = PROJECT_ROOT / ".tmp" / "reports" / "bond_equity_realistic_costs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
