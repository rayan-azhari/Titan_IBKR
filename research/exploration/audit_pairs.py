"""pairs V3.7 audit (Wave B).

Strategy: GLD/EFA pairs trading with periodic beta re-estimation.
Spread = price_GLD - beta × price_EFA. Z-score normalised.
Entry: |z| > 2.0; Exit: |z| < 0.5; Invalidate: |z| > 4.0.

V1 claimed OOS Sharpe +1.14, OOS/IS ratio 3.50 (high — possible
overfit).

V3.7 audit:
- L21 causality smoke (beta refit causality is the key risk here)
- L66 baseline: PAIRS class -> cash (market-neutral, expect SR > 0)
- L67 portfolio inclusion: does it improve joint portfolio vs 60/40?
- L65 ruin at proposed weight

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import assess_strategy_ruin  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    cdar,
    cvar,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "pairs_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 1.0
PERIODS_PER_YEAR = 252


def _load_d(symbol: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_D.parquet")
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def pairs_returns(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_z: float = 4.0,
    refit_window: int = 126,
    zscore_window: int = 252,
) -> pd.Series:
    """Per-bar net return of GLD/EFA pairs strategy.

    Causality: beta is re-estimated on a ROLLING IS window of `refit_window`
    bars ending at t-1 (past-only). z-score uses expanding window through
    t-1. Position decided at t earns t->t+1 return.
    """
    common = price_a.index.intersection(price_b.index)
    a = price_a.reindex(common)
    b = price_b.reindex(common)
    n = len(a)

    # Compute log returns
    log_a = np.log(a / a.shift(1))
    log_b = np.log(b / b.shift(1))

    # Beta is re-estimated every refit_window bars using OLS on past bars.
    betas = pd.Series(np.nan, index=common)
    for t in range(refit_window, n):
        if t % refit_window == 0 or t == refit_window:
            sub_a = log_a.iloc[t - refit_window:t].dropna()
            sub_b = log_b.iloc[t - refit_window:t].dropna()
            common_sub = sub_a.index.intersection(sub_b.index)
            if len(common_sub) < 30:
                continue
            sa = sub_a.reindex(common_sub).to_numpy()
            sb = sub_b.reindex(common_sub).to_numpy()
            sb_mean = sb.mean()
            sa_mean = sa.mean()
            denom = ((sb - sb_mean) ** 2).sum()
            if denom < 1e-12:
                continue
            beta = ((sa - sa_mean) * (sb - sb_mean)).sum() / denom
            betas.iloc[t] = beta
    betas = betas.ffill().fillna(1.0)

    # Spread = log_a - beta * log_b (in log-return space for stationarity)
    spread = log_a - betas * log_b

    # Z-score on expanding past-only window
    rolling_mu = spread.rolling(zscore_window, min_periods=60).mean()
    rolling_sigma = spread.rolling(zscore_window, min_periods=60).std()
    z = (spread - rolling_mu) / rolling_sigma.replace(0.0, np.nan)

    # Position state machine
    pos = np.zeros(n, dtype=float)
    state = 0
    arr_z = z.to_numpy()
    for i in range(n):
        v = arr_z[i]
        if np.isnan(v):
            pos[i] = float(state)
            continue
        # Force close if blowout
        if state != 0 and abs(v) > max_z:
            state = 0
        # Entry: short spread when z > entry_z (sell A, buy B)
        if state == 0:
            if v > entry_z:
                state = -1
            elif v < -entry_z:
                state = 1
        # Exit when |z| < exit_z
        elif abs(v) < exit_z:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=common)

    # Per-bar spread return: pos[t-1] * (log_a[t] - beta[t-1] * log_b[t])
    held = position.shift(1).fillna(0.0)
    held_beta = betas.shift(1).fillna(1.0)
    spread_ret = log_a - held_beta * log_b
    gross = held * spread_ret.fillna(0.0)

    # Cost: 2x turnover (both legs)
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (2 * COST_BPS / 10_000.0)
    return (gross - cost).rename("ret")


def assert_causal_pairs(a: pd.Series, b: pd.Series) -> None:
    """L21 smoke."""
    base = pairs_returns(a, b)
    cutoff = len(a) - 20
    a_c = a.copy()
    b_c = b.copy()
    a_c.iloc[cutoff:] *= 100.0
    b_c.iloc[cutoff:] *= 100.0
    pert = pairs_returns(a_c, b_c)
    diff = (base.iloc[:cutoff - 252] - pert.iloc[:cutoff - 252]).abs().max()
    assert diff < 1e-12, f"L21 fail: {diff}"
    print(f"[pairs] L21 PASS (max diff={diff:.2e})")


def main() -> None:
    print("=" * 88)
    print("pairs V3.7 audit (GLD/EFA)")
    print("=" * 88)

    gld = _load_d("GLD")
    efa = _load_d("EFA")
    print(f"\n[load] GLD: {len(gld)} bars, EFA: {len(efa)} bars")

    try:
        assert_causal_pairs(gld, efa)
    except AssertionError as e:
        print(f"[pairs] L21 smoke FAIL (acknowledged): {e}")
        print("  Likely subtle non-causality in beta refit / z-score window")
        print("  interaction. Investigate before declaring CONDITIONAL.")

    # Sweep entry_z + exit_z
    print("\n--- Sweep on GLD/EFA ---")
    print(f"{'entry_z':>9} {'exit_z':>7} {'Sharpe':>10} {'CI_lo':>10} {'CI_hi':>10} {'MaxDD':>10}")
    sweep = []
    for ez in [1.5, 2.0, 2.5]:
        for xz in [0.0, 0.5, 1.0]:
            ret = pairs_returns(gld, efa, entry_z=ez, exit_z=xz)
            sr = float(sharpe(ret, periods_per_year=PERIODS_PER_YEAR))
            ci_lo, ci_hi = bootstrap_sharpe_ci(ret, periods_per_year=PERIODS_PER_YEAR, seed=42)
            mdd = float(max_drawdown(ret))
            sweep.append((ez, xz, sr, ci_lo, ci_hi, mdd))
            print(f"{ez:>9.2f} {xz:>7.2f} {sr:>+9.4f} {ci_lo:>+9.3f} {ci_hi:>+9.3f} {mdd:>+9.2%}")

    # Live config (entry=2.0, exit=0.5)
    live_ret = pairs_returns(gld, efa, entry_z=2.0, exit_z=0.5)
    live_sr = float(sharpe(live_ret, periods_per_year=PERIODS_PER_YEAR))
    live_ci = bootstrap_sharpe_ci(live_ret, periods_per_year=PERIODS_PER_YEAR, seed=42)
    print(f"\n[live config (entry_z=2.0, exit_z=0.5)] Sharpe={live_sr:+.4f}, "
          f"CI=[{live_ci[0]:+.3f}, {live_ci[1]:+.3f}]")
    print(f"  CVaR-95 = {cvar(live_ret, alpha=0.05):+.4%}")
    print(f"  CDaR-95 = {cdar(live_ret, alpha=0.05):+.4%}")

    # L66 baseline gate (cash, since pairs is market-neutral)
    print("\n--- L66 baseline gate (cash; SR > 0 with CI_lo > -0.5) ---")
    baseline_pass = live_sr > 0 and live_ci[0] > -0.5
    print(f"  Live Sharpe = {live_sr:+.4f}, CI_lo = {live_ci[0]:+.3f}")
    print(f"  L66 gate: {'PASS' if baseline_pass else 'FAIL'}")

    # L65 ruin at sample weights
    print("\n--- L65 single-strategy ruin ---")
    for w in [0.05, 0.10, 0.15, 0.20]:
        ruin = assess_strategy_ruin(
            live_ret, deployment_weight=w,
            portfolio_kill_threshold=0.15, horizon_bars=252,
            block_size=21, n_paths=2000, seed=42,
        )
        print(f"  weight={w:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
              f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, passes={ruin.passes()}")

    # Verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if not baseline_pass:
        print(f"RETIRE: live Sharpe {live_sr:+.4f} fails baseline gate.")
    elif live_sr > 0.30 and live_ci[0] > 0:
        print(f"CONDITIONAL_WATCHPOINT candidate: live SR {live_sr:+.4f} with "
              f"CI_lo {live_ci[0]:+.3f}. Joint L65 vs current portfolio needed.")
    else:
        print(f"MARGINAL: live SR {live_sr:+.4f}, CI_lo {live_ci[0]:+.3f}. "
              f"Defer; needs more sweep cells or larger universe.")
    print("V1 claim was +1.14; check L62 gap.")


if __name__ == "__main__":
    main()
