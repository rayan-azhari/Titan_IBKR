"""Top-5 Diversified Portfolio with 1%-per-trade risk sizing.

Combines the 4 best strategies from autoresearch v2 (420+ experiments):
  1. AUD/JPY MR vwap46+donchian sp0.5 is32k oos8k  (40%)
  2. IWB ML Stacking cbars=5                         (25%)
  3. HYG->IWB Cross-Asset (doubled allocation)       (30%)
  4. AUD/USD MR vwap36+donchian sp0.5                (5%)

  QQQ ML removed: -21% DD, 10.5% RoR — weight redistributed to HYG->IWB.

For each strategy the raw OOS return series is extracted, then scaled
so that each strategy's per-trade risk equals 1% of its allocated sub-equity.
The scaling factor is derived from the strategy's historical per-trade
return volatility relative to a 1% ATR stop.

Portfolio equity curves are combined with target weights, then metrics
and pairwise correlation are reported across four weight scenarios.

Usage:
    uv run python research/auto/phase_portfolio.py
"""

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Strategy configurations ────────────────────────────────────────────────

ML_BASE = dict(
    strategy="stacking",
    timeframe="D",
    xgb_params=dict(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, random_state=42, verbosity=0,
    ),
    lstm_hidden=32, lookback=20, lstm_epochs=30, n_nested_folds=3,
    label_params=[
        dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
        dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
        dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
    ],
    signal_threshold=0.6, cost_bps=2.0, is_years=2, oos_months=2,
)

STRATEGIES = [
    {
        "name": "AUD/JPY MR",
        "instrument": "AUD_JPY",
        "runner": "mr",
        "cfg": dict(
            strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="H1",
            vwap_anchor=46, regime_filter="conf_donchian_pos_20",
            tier_grid="conservative", spread_bps=0.5, slippage_bps=0.2,
            is_bars=32000, oos_bars=8000,
        ),
        "stop_mult": 1.5,
    },
    {
        "name": "IWB ML",
        "instrument": "IWB",
        "runner": "ml",
        "cfg": dict(**ML_BASE, instruments=["IWB"]),
        "stop_mult": 2.0,
    },
    {
        "name": "HYG->IWB",
        "instrument": "IWB",
        "runner": "xa",
        "cfg": dict(
            strategy="cross_asset", instruments=["IWB"], bond="HYG",
            lookback=10, hold_days=10, threshold=0.50,
            is_days=504, oos_days=126, spread_bps=5.0,
        ),
        "stop_mult": 1.5,
    },
    {
        "name": "AUD/USD MR",
        "instrument": "AUD_USD",
        "runner": "mr",
        "cfg": dict(
            strategy="mean_reversion", instruments=["AUD_USD"], timeframe="H1",
            vwap_anchor=36, regime_filter="conf_donchian_pos_20",
            tier_grid="conservative", spread_bps=0.5, slippage_bps=0.2,
            is_bars=30000, oos_bars=7500,
        ),
        "stop_mult": 1.5,
    },
]

WEIGHT_SCENARIOS = {
    "Target  (40/25/30/5)":   [0.40, 0.25, 0.30, 0.05],
    "HYG-heavy(35/20/40/5)":  [0.35, 0.20, 0.40, 0.05],
    "MR-heavy (50/20/25/5)":  [0.50, 0.20, 0.25, 0.05],
    "Equal    (25ea)":        [0.25, 0.25, 0.25, 0.25],
}

RISK_PCT = 0.01        # target risk per trade as fraction of sub-equity
TOTAL_CAPITAL = 100_000


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(ret: pd.Series, ann: int = 252) -> float:
    s = ret.std()
    return float(ret.mean() / s * sqrt(ann)) if s > 1e-10 else 0.0


def _max_dd(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def _calmar(ret: pd.Series) -> float:
    dd = _max_dd(ret)
    ann_ret = float(ret.mean() * 252)
    return ann_ret / abs(dd) if abs(dd) > 1e-6 else 0.0


def _score(ret: pd.Series, parity: float = 1.0) -> float:
    sh = _sharpe(ret)
    dd = _max_dd(ret)
    return sh + 0.3 * min(parity, 1.5) - 0.5 * max(0, -dd - 0.15)


def scale_to_risk(ret: pd.Series, stop_mult: float) -> pd.Series:
    """Scale a return series so each trade risks ~1% of equity.

    The raw series has arbitrary units (MR uses tier sizes 1/2/4/8;
    ML uses 1-unit positions).  We estimate the "typical stop loss hit"
    as stop_mult standard deviations of daily returns, then solve for
    the position fraction f that makes f * stop_dist = RISK_PCT.

    This is a proportional scaling — it preserves the Sharpe ratio but
    sets the volatility target so that a 1-ATR adverse move costs ~1%.
    """
    if len(ret) < 20:
        return ret
    # Proxy: std of non-zero daily returns ≈ average |bar return|
    nz = ret[ret != 0.0]
    if len(nz) < 10:
        return ret
    typical_bar_vol = float(nz.std())
    # stop_dist in return-space = stop_mult × typical daily vol
    stop_dist = stop_mult * typical_bar_vol
    if stop_dist < 1e-9:
        return ret
    # Current "average position size" ≈ mean absolute bar return / typical vol
    # We want: position_frac × stop_dist = RISK_PCT
    # Current effective position ≈ 1 unit (raw), so scale = RISK_PCT / stop_dist
    scale = RISK_PCT / stop_dist
    return ret * scale


# ── Strategy runners ───────────────────────────────────────────────────────

def get_returns(strat: dict) -> pd.Series | None:
    """Run the strategy WFO with return_raw=True and return daily OOS returns."""
    from research.auto.evaluate import (
        run_cross_asset_wfo, run_mean_reversion_wfo, run_ml_wfo,
    )
    instrument = strat["instrument"]
    cfg = strat["cfg"]
    runner = strat["runner"]
    name = strat["name"]

    print(f"  Running {name} ({instrument}) ...", end=" ", flush=True)
    try:
        if runner == "ml":
            r = run_ml_wfo(instrument, cfg, return_raw=True)
        elif runner == "mr":
            r = run_mean_reversion_wfo(instrument, cfg, return_raw=True)
        elif runner == "xa":
            r = run_cross_asset_wfo(instrument, cfg, return_raw=True)
        else:
            print("UNKNOWN RUNNER")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    raw = r.get("stitched_returns")
    if raw is None or len(raw) < 20:
        print(f"no raw returns (sharpe={r.get('sharpe', '?')})")
        return None

    # Ensure daily frequency
    if not isinstance(raw.index, pd.DatetimeIndex):
        print("bad index type")
        return None

    raw = raw.sort_index()
    # Normalise to tz-naive UTC dates so all series can be aligned
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    raw.index = raw.index.normalize()         # strip intraday time component
    raw = raw.resample("D").sum()             # aggregate to daily
    # Expand to full business-day calendar; non-trade days become 0.
    # This is required so that inner-join across strategies finds a common
    # calendar rather than the sparse intersection of trade days only.
    full_idx = pd.bdate_range(raw.index.min(), raw.index.max())
    raw = raw.reindex(full_idx, fill_value=0.0)
    print(f"OK  sharpe={r.get('sharpe','?'):.3f}  trade_days={int((raw!=0).sum())}")
    return raw


# ── Portfolio combination ──────────────────────────────────────────────────

def combine(curves: list[pd.Series], weights: list[float]) -> pd.Series:
    """Combine return series on their common date range."""
    df = pd.concat(curves, axis=1, join="inner")
    df.columns = range(len(curves))
    df = df.fillna(0.0)
    w = np.array(weights) / sum(weights)  # normalise
    return (df * w).sum(axis=1)


def report_scenario(name: str, port_ret: pd.Series, curves: list[pd.Series],
                    strategy_names: list[str], weights: list[float]) -> float:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    sh = _sharpe(port_ret)
    dd = _max_dd(port_ret)
    cal = _calmar(port_ret)
    ann = float(port_ret.mean() * 252)
    sc = _score(port_ret)
    print(f"  Portfolio Sharpe  : {sh:.3f}")
    print(f"  Portfolio Max DD  : {dd*100:.1f}%")
    print(f"  Calmar Ratio      : {cal:.3f}")
    print(f"  Annual Return     : {ann*100:.1f}%")
    print(f"  Composite SCORE   : {sc:.4f}")
    print(f"  Date range        : {port_ret.index[0].date()} -> {port_ret.index[-1].date()}")
    print(f"  Trading days      : {len(port_ret)}")

    # Correlation matrix on common dates
    df = pd.concat(curves, axis=1, join="inner").fillna(0.0)
    df.columns = strategy_names
    corr = df.corr()
    print(f"\n  Correlation matrix:")
    header = " " * 14 + "".join(f"{n:>10}" for n in strategy_names)
    print(f"  {header}")
    for row in strategy_names:
        vals = "".join(f"{corr.loc[row, col]:>10.2f}" for col in strategy_names)
        print(f"  {row:<14}{vals}")

    # Per-strategy standalone stats
    print(f"\n  Per-strategy OOS (standalone, before weighting):")
    print(f"  {'Strategy':<14} {'Weight':>7} {'Sharpe':>7} {'MaxDD':>7} {'Bars':>6}")
    for s_ret, sname, w in zip(curves, strategy_names, weights):
        print(f"  {sname:<14} {w:>7.0%} {_sharpe(s_ret):>7.3f} "
              f"{_max_dd(s_ret)*100:>6.1f}% {len(s_ret):>6}")
    return sc


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Top-5 Portfolio Backtest  (1% per-trade risk model)")
    print("=" * 60)

    # Collect raw return series
    print("\nStep 1: Running all 5 strategies...")
    raw_curves = []
    valid_strategies = []
    for strat in STRATEGIES:
        ret = get_returns(strat)
        if ret is not None:
            scaled = scale_to_risk(ret, strat["stop_mult"])
            raw_curves.append(scaled)
            valid_strategies.append(strat)

    if len(raw_curves) < 2:
        print("ERROR: fewer than 2 valid strategies — cannot build portfolio")
        sys.exit(1)

    # ── Find the largest subset of strategies with a positive common overlap ──
    # With 5 strategies try all combinations (31 subsets) and pick the
    # largest set that has positive overlap, breaking ties by most overlap days.
    from itertools import combinations as _combos

    best_subset: tuple | None = None
    best_overlap = -1
    n_total = len(raw_curves)
    for size in range(n_total, 1, -1):
        for combo in _combos(range(n_total), size):
            cs = max(raw_curves[i].index[0] for i in combo)
            ce = min(raw_curves[i].index[-1] for i in combo)
            days = (ce - cs).days
            if days > best_overlap:
                best_overlap = days
                best_subset = combo
        if best_subset is not None and best_overlap > 0:
            break  # largest size with positive overlap found

    if best_subset is None or best_overlap <= 0:
        print("ERROR: no overlapping subset found")
        sys.exit(1)

    skipped = set(range(n_total)) - set(best_subset)
    for i in skipped:
        c = raw_curves[i]
        print(f"  SKIP (insufficient overlap): {valid_strategies[i]['name']} "
              f"({c.index[0].date()} -> {c.index[-1].date()})")

    curves_in = [raw_curves[i] for i in best_subset]
    strats_in = [valid_strategies[i] for i in best_subset]

    strategy_names = [s["name"] for s in strats_in]
    n = len(curves_in)
    print(f"\nPortfolio strategies: {n}")

    common_start = max(c.index[0] for c in curves_in)
    common_end = min(c.index[-1] for c in curves_in)
    print(f"Common date range: {common_start.date()} -> {common_end.date()}")

    clipped = [c[common_start:common_end] for c in curves_in]
    valid_strategies = strats_in
    raw_curves = curves_in

    # Run weight scenarios
    print("\nStep 2: Portfolio weight scenarios...")
    best_score = -99.0
    best_scenario = None

    for scenario_name, full_weights in WEIGHT_SCENARIOS.items():
        w = full_weights[:n]  # trim if fewer strategies available
        port_ret = combine(clipped, w)
        sc = report_scenario(scenario_name, port_ret, clipped,
                             strategy_names, w)
        if sc > best_score:
            best_score = sc
            best_scenario = scenario_name

    print(f"\n{'='*60}")
    print(f"  Best scenario : {best_scenario}")
    print(f"  Best SCORE    : {best_score:.4f}")
    print(f"  Baseline BEST : 5.1368  (AUD/JPY standalone)")
    delta = best_score - 5.1368
    print(f"  Delta vs solo : {delta:+.4f}")
    if best_score > 5.1368:
        print("  *** PORTFOLIO BEATS STANDALONE CHAMPION ***")
    else:
        print("  Standalone AUD/JPY still wins (expected: diversification "
              "reduces SCORE but also reduces DD and correlation risk)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
