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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from titan.research.metrics import BARS_PER_YEAR  # noqa: E402
from titan.research.metrics import sharpe as _sh_metric

# ── Strategy configurations ────────────────────────────────────────────────

ML_BASE = dict(
    strategy="stacking",
    timeframe="D",
    xgb_params=dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        verbosity=0,
    ),
    lstm_hidden=32,
    lookback=20,
    lstm_epochs=30,
    n_nested_folds=3,
    label_params=[
        dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
        dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
        dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
    ],
    signal_threshold=0.6,
    cost_bps=2.0,
    is_years=2,
    oos_months=2,
)

STRATEGIES = [
    {
        "name": "AUD/JPY MR",
        "instrument": "AUD_JPY",
        "runner": "mr",
        "cfg": dict(
            strategy="mean_reversion",
            instruments=["AUD_JPY"],
            timeframe="H1",
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            tier_grid="conservative",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
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
            strategy="cross_asset",
            instruments=["IWB"],
            bond="HYG",
            lookback=10,
            hold_days=10,
            threshold=0.50,
            is_days=504,
            oos_days=126,
            spread_bps=5.0,
        ),
        "stop_mult": 1.5,
    },
    {
        "name": "AUD/USD MR",
        "instrument": "AUD_USD",
        "runner": "mr",
        "cfg": dict(
            strategy="mean_reversion",
            instruments=["AUD_USD"],
            timeframe="H1",
            vwap_anchor=36,
            regime_filter="conf_donchian_pos_20",
            tier_grid="conservative",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=30000,
            oos_bars=7500,
        ),
        "stop_mult": 1.5,
    },
]

# NOTE: hand-picked weight scenarios were removed -- they were cherry-picked
# *after* observing OOS results which biased the reported Sharpe/Calmar
# upward. The replacement flow (see ``main``) computes inverse-vol weights on
# the IS half of each strategy's return series and reports them OOS, plus a
# band of 100 randomised-weight portfolios to show sensitivity.
_LEGACY_WEIGHT_SCENARIOS_INFO = (
    "Hand-picked weights removed (look-ahead). See main() for IS-derived "
    "inverse-vol weights + random-sensitivity band."
)

RISK_PCT = 0.01  # target risk per trade as fraction of sub-equity
TOTAL_CAPITAL = 100_000


# ── Helpers ────────────────────────────────────────────────────────────────


def _sharpe(ret: pd.Series, ann: int = 252) -> float:
    """Sharpe via the shared ``titan.research.metrics`` helper.

    The old implementation dropped zero-return days before annualising,
    which ``sqrt(1/active_ratio)`` over-stated Sharpe for sparse / low-
    frequency strategies. The shared helper includes zero days in the
    denominator — non-trade days are information about selectivity.
    """
    return _sh_metric(ret, periods_per_year=int(ann))


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


def scale_to_risk(
    ret: pd.Series,
    stop_mult: float,
    *,
    bars_per_year: int = BARS_PER_YEAR["D"],
    trades: list[float] | None = None,
) -> pd.Series:
    """Scale a return series so each trade's P&L approximates RISK_PCT.

    When ``trades`` is provided (list of per-trade returns from the WFO
    runner's ``stitched_trades`` output), scaling uses the *per-trade*
    return vol directly: this is the exact "1% per trade" semantics the
    remediation plan calls for. When ``trades`` is absent, falls back to
    the per-active-day approximation using the non-zero bars in ``ret``.

    Old behavior (pre-April-2026 remediation):

    * filtered ``nz = ret[ret != 0.0]`` before measuring vol, over-counting
      vol for low-frequency strategies;
    * inserted a hard-coded ``* 24`` multiplier regardless of bar timeframe
      — an H1-like factor that made no sense for the daily-aggregated
      series every caller actually passes in.

    Current behavior — trade-level when available:

        scale = RISK_PCT / (stop_mult * trade_level_vol)

    Fallback (no trade list supplied):

        scale = RISK_PCT / (stop_mult * per_active_period_vol)

    ``bars_per_year`` is kept in the signature so callsites declare
    series frequency explicitly even when the fallback path is used.
    """
    if len(ret) < 20:
        return ret

    # Preferred path: per-trade risk vol.
    if trades is not None and len(trades) >= 10:
        import numpy as _np

        trade_vol = float(_np.std(_np.asarray(trades, dtype=float), ddof=1))
        stop_dist = stop_mult * trade_vol
        if stop_dist >= 1e-9:
            return ret * (RISK_PCT / stop_dist)

    # Fallback: per-active-day vol (pre-trade-plumbing behavior).
    nz = ret[ret != 0.0]
    if len(nz) < 10:
        return ret
    per_period_vol = float(nz.std())
    stop_dist = stop_mult * per_period_vol
    if stop_dist < 1e-9:
        return ret
    del bars_per_year  # caller-declared frequency; fallback doesn't use it
    return ret * (RISK_PCT / stop_dist)


# ── Strategy runners ───────────────────────────────────────────────────────


def get_returns(strat: dict) -> tuple[pd.Series, list[float]] | None:
    """Run the strategy WFO with return_raw=True and return (daily OOS
    returns, list of per-trade returns). The trade list is used by
    ``scale_to_risk`` for exact per-trade risk sizing; the daily series
    drives the portfolio combination and Sharpe reporting.
    """
    from research.auto.evaluate import (
        run_cross_asset_wfo,
        run_mean_reversion_wfo,
        run_ml_wfo,
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
    trades = list(r.get("stitched_trades", []))
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
    raw.index = raw.index.normalize()  # strip intraday time component
    raw = raw.resample("D").sum()  # aggregate to daily
    # Expand to full business-day calendar; non-trade days become 0.
    full_idx = pd.bdate_range(raw.index.min(), raw.index.max())
    raw = raw.reindex(full_idx, fill_value=0.0)
    print(
        f"OK  sharpe={r.get('sharpe', '?'):.3f}  "
        f"trade_days={int((raw != 0).sum())}  trades={len(trades)}"
    )
    return raw, trades


# ── Portfolio combination ──────────────────────────────────────────────────


def combine(curves: list[pd.Series], weights: list[float]) -> pd.Series:
    """Combine return series on their common date range.

    ``join="inner"`` already drops any index rows that aren't present in all
    curves, so the previous ``df.fillna(0.0)`` was dead code -- removed.

    Slippage / rebalancing cost caveat
    ----------------------------------
    This function assumes weights are applied *continuously* with zero
    friction. In live trading each rebalance implies real trades to hit the
    new target weights -- that friction is not modelled here. See the TODO
    in ``scale_to_risk`` for the related trade-level costing gap. If you
    need a conservative estimate, subtract ~1-5 bps per rebalance day from
    the reported portfolio returns.
    """
    df = pd.concat(curves, axis=1, join="inner")
    df.columns = range(len(curves))
    w = np.array(weights) / sum(weights)  # normalise
    return (df * w).sum(axis=1)


def report_scenario(
    name: str,
    port_ret: pd.Series,
    curves: list[pd.Series],
    strategy_names: list[str],
    weights: list[float],
) -> float:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    sh = _sharpe(port_ret)
    dd = _max_dd(port_ret)
    cal = _calmar(port_ret)
    ann = float(port_ret.mean() * 252)
    sc = _score(port_ret)
    print(f"  Portfolio Sharpe  : {sh:.3f}")
    print(f"  Portfolio Max DD  : {dd * 100:.1f}%")
    print(f"  Calmar Ratio      : {cal:.3f}")
    print(f"  Annual Return     : {ann * 100:.1f}%")
    print(f"  Composite SCORE   : {sc:.4f}")
    print(f"  Date range        : {port_ret.index[0].date()} -> {port_ret.index[-1].date()}")
    print(f"  Trading days      : {len(port_ret)}")

    # Correlation matrix on common dates
    df = pd.concat(curves, axis=1, join="inner").fillna(0.0)
    df.columns = strategy_names
    corr = df.corr()
    print("\n  Correlation matrix:")
    header = " " * 14 + "".join(f"{n:>10}" for n in strategy_names)
    print(f"  {header}")
    for row in strategy_names:
        vals = "".join(f"{corr.loc[row, col]:>10.2f}" for col in strategy_names)
        print(f"  {row:<14}{vals}")

    # Per-strategy standalone stats
    print("\n  Per-strategy OOS (standalone, before weighting):")
    print(f"  {'Strategy':<14} {'Weight':>7} {'Sharpe':>7} {'MaxDD':>7} {'Bars':>6}")
    for s_ret, sname, w in zip(curves, strategy_names, weights):
        print(
            f"  {sname:<14} {w:>7.0%} {_sharpe(s_ret):>7.3f} "
            f"{_max_dd(s_ret) * 100:>6.1f}% {len(s_ret):>6}"
        )
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
        result = get_returns(strat)
        if result is not None:
            ret, trades = result
            scaled = scale_to_risk(
                ret,
                strat["stop_mult"],
                trades=trades if trades else None,
            )
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
        print(
            f"  SKIP (insufficient overlap): {valid_strategies[i]['name']} "
            f"({c.index[0].date()} -> {c.index[-1].date()})"
        )

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

    # ── Step 2: IS-derived inverse-vol weights, reported OOS ─────────
    #
    # The old code evaluated hand-picked weight scenarios in-sample on the
    # same data that produced the stitched OOS returns, biasing everything
    # upward. Instead we:
    #   a) Split each strategy's stitched returns into an IS half and an
    #      OOS half (50/50 by date).
    #   b) Compute inverse-vol weights on IS only.
    #   c) Report the realised OOS portfolio Sharpe/DD under those weights.
    #   d) Report the OOS distribution of 100 random weight portfolios so
    #      the reader can see how sensitive the answer is to the choice.
    print("\nStep 2: IS-derived inverse-vol weights, OOS evaluation.")
    print(_LEGACY_WEIGHT_SCENARIOS_INFO)

    split_idx = len(clipped[0]) // 2
    is_slice = [c.iloc[:split_idx] for c in clipped]
    oos_slice = [c.iloc[split_idx:] for c in clipped]

    # IS inverse-vol weights.
    is_vols = []
    for s in is_slice:
        active = s[s != 0.0]
        v = float(active.std()) if len(active) >= 10 else float(s.std())
        is_vols.append(v if v > 1e-9 else 1e-9)
    inv_vols = [1.0 / v for v in is_vols]
    total_inv = sum(inv_vols)
    iv_weights = [iv / total_inv for iv in inv_vols]

    oos_port = combine(oos_slice, iv_weights)
    score = report_scenario(
        f"IS-inv-vol weights OOS ({'/'.join(f'{w:.0%}' for w in iv_weights)})",
        oos_port,
        oos_slice,
        strategy_names,
        iv_weights,
    )

    # Random-weight sensitivity band (OOS).
    import numpy as _np

    _np.random.seed(42)
    random_scores = []
    for _ in range(100):
        w = _np.random.dirichlet(_np.ones(n))
        pr = combine(oos_slice, list(w))
        random_scores.append(_sharpe(pr))
    rs = _np.array(random_scores)
    print("\n  Random-weight OOS Sharpe band (n=100 draws):")
    print(
        f"    p05={_np.percentile(rs, 5):.2f}  p50={_np.percentile(rs, 50):.2f}  "
        f"p95={_np.percentile(rs, 95):.2f}"
    )
    print(f"    IS-inv-vol picks Sharpe={_sharpe(oos_port):.2f}")

    print(f"\n{'=' * 60}")
    print(f"  Reported score (OOS, IS-inv-vol) : {score:.4f}")
    print("  Compared to AUD/JPY standalone  : 5.1368")
    delta = score - 5.1368
    print(f"  Delta (CAN be negative)         : {delta:+.4f}")
    print("  Note: standalone number is not directly comparable -- this is")
    print("  a true OOS estimate; prior runs mixed IS weight-selection with")
    print("  OOS metrics which biased upward.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
