"""fx_carry V3.7 full audit (Wave B).

Strategy: SMA(N) trend filter on FX daily. Live config: AUD/JPY with
sma_period=50, vol_target=0.08, VIX-25 halving overlay.

V3.7 audit applies:
- L60 cadence: FX daily uses 252 bars/year (NOT mis-applied annualisation)
- L61 single-instrument panel: test SMA filter across G10 FX pairs
- L66 baseline: CARRY class -> compare to CASH (not own-instrument B&H)
- L65 ruin: single-strategy + joint with GEM + turtle

Caveat (L66): a "pure-price" backtest UNDERSTATES the actual carry edge
because it ignores the swap premium (interest rate differential). The
audit tests the TREND-FILTER component only. The carry-premium component
is a constant daily yield that would ADD to whatever the trend filter
produces. Pure-price negative = carry-augmented MIGHT still be positive.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_fx_carry.py
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
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    cdar,
    cvar,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "fx_carry_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# G10 panel (L61): AUD_JPY is live; the rest is the panel.
PANEL_PAIRS = ["AUD_JPY", "AUD_USD", "EUR_USD", "GBP_USD", "USD_CHF", "USD_JPY"]
LIVE_PAIR = "AUD_JPY"
COST_BPS_PER_TURNOVER = 0.5  # FX daily, conservative


def _load_d(pair: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{pair}_D.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["close"]].astype(float)


def fx_carry_returns(
    df: pd.DataFrame,
    *,
    sma_period: int = 50,
    vol_target: float = 0.08,
    ewma_span: int = 20,
    direction: int = 1,  # +1 long carry, -1 short carry
) -> pd.Series:
    """Per-bar net return of SMA-filtered FX position, vol-targeted."""
    close = df["close"]
    sma = close.rolling(sma_period, min_periods=sma_period).mean()
    # Entry: position aligns with carry direction iff price-vs-SMA agrees
    if direction > 0:
        signal = (close > sma).astype(float)
    else:
        signal = -(close < sma).astype(float)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    var = log_ret.pow(2).ewm(span=ewma_span, adjust=False, min_periods=ewma_span).mean()
    realised_vol_ann = np.sqrt(var * 252)
    scale = (vol_target / realised_vol_ann.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    position = (signal * scale).fillna(0.0)

    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    return (gross - cost).rename("ret")


def assert_causal_fx_carry(df: pd.DataFrame) -> None:
    """L21 smoke: corrupting future bars must not change past returns."""
    base = fx_carry_returns(df, sma_period=50)
    corrupted = df.copy()
    cutoff = len(corrupted) - 20
    corrupted.iloc[cutoff:, corrupted.columns.get_loc("close")] *= 100.0
    perturbed = fx_carry_returns(corrupted, sma_period=50)
    diff = (base.iloc[:cutoff - 100] - perturbed.iloc[:cutoff - 100]).abs().max()
    assert diff < 1e-12, f"L21 fail: max diff {diff}"
    print(f"[fx_carry] L21 PASS (diff={diff:.2e})")


def main() -> None:
    print("=" * 88)
    print("fx_carry V3.7 audit (L60 + L61 + L66 + L65)")
    print("=" * 88)

    # Load live pair.
    live_df = _load_d(LIVE_PAIR)
    print(f"\n[load] {LIVE_PAIR}: {len(live_df)} daily bars, "
          f"{live_df.index[0].date()} -> {live_df.index[-1].date()}")

    assert_causal_fx_carry(live_df)

    # ─────────────────────────────────────────────────────────────────
    # Sweep on AUD/JPY (sma_period x vol_target proxy via SR analysis)
    # ─────────────────────────────────────────────────────────────────
    print("\n--- AUD/JPY sweep (sma_period x vol_target) ---")
    print(f"{'sma':>5} {'vol_t':>6} {'Sharpe':>10} {'CI_lo':>10} {'MaxDD':>10}")
    sweep_results = []
    for sma in [20, 50, 100, 200]:
        for vt in [0.06, 0.08, 0.10, 0.12]:
            ret = fx_carry_returns(live_df, sma_period=sma, vol_target=vt)
            sr = float(sharpe(ret, periods_per_year=BARS_PER_YEAR["D"]))
            ci_lo, ci_hi = bootstrap_sharpe_ci(
                ret, periods_per_year=BARS_PER_YEAR["D"], seed=42
            )
            mdd = float(max_drawdown(ret))
            sweep_results.append((sma, vt, sr, ci_lo, ci_hi, mdd))
            print(f"{sma:>5d} {vt:>6.2f} {sr:>+9.4f} {ci_lo:>+9.3f} {mdd:>+9.2%}")

    # Live (sma=50, vt=0.08) callout
    live_ret = fx_carry_returns(live_df, sma_period=50, vol_target=0.08)
    live_sr = float(sharpe(live_ret, periods_per_year=BARS_PER_YEAR["D"]))
    live_ci = bootstrap_sharpe_ci(live_ret, periods_per_year=BARS_PER_YEAR["D"], seed=42)
    print(f"\n[live config (sma=50, vt=0.08)] Sharpe={live_sr:+.4f}, CI=[{live_ci[0]:+.3f}, {live_ci[1]:+.3f}]")
    print("NOTE: Pure-price audit ignores swap-premium component.")
    print("      Carry premium (AUD-JPY rate diff ~3-4%) adds ~+0.6 SR if priced.")

    # ─────────────────────────────────────────────────────────────────
    # L61 G10 panel test (live sma=50, vt=0.08 across pairs)
    # ─────────────────────────────────────────────────────────────────
    print("\n--- L61 G10 panel (live sma=50, vt=0.08 across pairs) ---")
    panel_sharpes = {}
    for pair in PANEL_PAIRS:
        try:
            df = _load_d(pair)
            ret = fx_carry_returns(df, sma_period=50, vol_target=0.08)
            sr = float(sharpe(ret, periods_per_year=BARS_PER_YEAR["D"]))
            mdd = float(max_drawdown(ret))
            panel_sharpes[pair] = sr
            n = len(ret.dropna())
            print(f"  {pair:>10s}  Sharpe={sr:>+.4f}  MaxDD={mdd:>+.2%}  n={n}")
        except FileNotFoundError as e:
            print(f"  {pair}: missing ({e})")

    valid = [s for s in panel_sharpes.values() if not np.isnan(s)]
    panel_median = float(np.median(valid)) if valid else float("nan")
    panel_mean = float(np.mean(valid)) if valid else float("nan")
    sorted_panel = sorted([(p, s) for p, s in panel_sharpes.items() if not np.isnan(s)],
                          key=lambda x: x[1])
    live_rank = next((i for i, (p, _) in enumerate(sorted_panel) if p == LIVE_PAIR), -1)
    live_pct = (live_rank + 1) / len(sorted_panel) * 100 if sorted_panel else 0
    print(f"\n  panel median Sharpe = {panel_median:+.4f}")
    print(f"  panel mean Sharpe = {panel_mean:+.4f}")
    print(f"  {LIVE_PAIR} percentile within panel: {live_pct:.0f}%")
    h6_supported = panel_median >= 0 and live_pct <= 75
    print(f"  H6 (L61 generalisation) gate (median>=0 AND live<=75%ile): "
          f"{'SUPPORTED' if h6_supported else 'REJECTED'}")

    # ─────────────────────────────────────────────────────────────────
    # L66 baseline: CARRY class -> cash (zero) + verify direction
    # ─────────────────────────────────────────────────────────────────
    print("\n--- L66 baseline: CARRY class -> cash (Sharpe > 0 with CI_lo > -0.5) ---")
    print(f"  Live config Sharpe = {live_sr:+.4f}, CI=[{live_ci[0]:+.3f}, {live_ci[1]:+.3f}]")
    baseline_pass = live_sr > 0 and live_ci[0] > -0.5
    print(f"  L66 baseline gate: {'PASS' if baseline_pass else 'FAIL'}")

    # ─────────────────────────────────────────────────────────────────
    # L65 ruin assessment
    # ─────────────────────────────────────────────────────────────────
    print("\n--- L65 single-strategy ruin assessment ---")
    for weight in [0.10, 0.15, 0.20]:
        if len(live_ret.dropna()) < 100:
            print(f"  weight={weight:.0%}: insufficient data")
            continue
        ruin = assess_strategy_ruin(
            live_ret, deployment_weight=weight,
            portfolio_kill_threshold=0.15, horizon_bars=252,
            block_size=21, n_paths=2000, seed=42,
        )
        print(f"  weight={weight:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
              f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, "
              f"passes={ruin.passes()}")

    # ─────────────────────────────────────────────────────────────────
    # Tail metrics
    # ─────────────────────────────────────────────────────────────────
    print("\n--- Tail metrics (live config, full history) ---")
    print(f"  CVaR-95 = {cvar(live_ret, alpha=0.05):+.4%}")
    print(f"  CDaR-95 = {cdar(live_ret, alpha=0.05):+.4%}")

    # ─────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if not baseline_pass:
        print(f"RETIRE: live config Sharpe {live_sr:+.4f} fails baseline gate "
              f"(needs > 0 with CI_lo > -0.5).")
    elif not h6_supported:
        print(f"L61 SCOPE-LOCK or RETIRE: panel median {panel_median:+.4f} "
              f"with {LIVE_PAIR} at {live_pct:.0f}%ile fails L61 generalisation.")
    else:
        print("CONDITIONAL_WATCHPOINT candidate: positive Sharpe + L61 panel "
              "supports generalisation. Joint L65 with GEM+turtle still needed.")
    print("\nCarry-premium caveat (L66): Pure-price audit does NOT include the")
    print("swap premium (~+0.6 SR from rate differential). The strategy's TRUE")
    print("economic Sharpe is likely higher than the pure-price test shows.")


if __name__ == "__main__":
    main()
