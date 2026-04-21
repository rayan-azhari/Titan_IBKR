"""portfolio_eval.py -- Comprehensive portfolio statistics for the top-5 strategy set.

Computes per-strategy AND combined portfolio metrics:
  - Sharpe, Sortino, Calmar, CAGR, Max DD, Avg DD duration
  - Win rate, number of wins/losses, avg win, avg loss, profit factor
  - % capital deployed (time in market)
  - Risk per trade (1% ATR stop model)
  - Risk of ruin (Monte Carlo, 5000 paths)
  - Pairwise correlation matrix
  - Annual return breakdown by year

Strategies:
  1. AUD/JPY MR vwap46+donchian sp0.5 is32k oos8k  -- 40%
  2. IWB ML Stacking cbars=5                         -- 25%
  3. HYG->IWB Cross-Asset (doubled from 15%)         -- 30%
  4. AUD/USD MR vwap36+donchian sp0.5                -- 5%

  QQQ ML removed: -21% DD, 10.5% RoR, 46% time in market — unacceptable risk.
  HYG->IWB allocation doubled to absorb the freed weight.

Usage:
    uv run python research/auto/portfolio_eval.py
"""

import sys
from itertools import combinations
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
        "target_weight": 0.40,
    },
    {
        "name": "IWB ML",
        "instrument": "IWB",
        "runner": "ml",
        "cfg": dict(**ML_BASE, instruments=["IWB"], is_years=2, oos_months=2),
        "stop_mult": 2.0,
        "target_weight": 0.25,
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
        "target_weight": 0.30,  # increased from 0.15 — replaces QQQ ML slot
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
        "target_weight": 0.05,
    },
]

RISK_PCT = 0.01  # 1% of sub-equity per trade
TOTAL_CAPITAL = 100_000
MC_PATHS = 5000  # Monte Carlo paths for risk of ruin
RUIN_THRESHOLD = 0.50  # 50% loss = ruin


# ── Per-trade extraction helpers ───────────────────────────────────────────


def trades_from_daily_returns(ret: pd.Series, signal: pd.Series | None = None) -> list[float]:
    """Extract per-trade returns from a daily return series.

    A 'trade' is a contiguous block of non-zero returns (position held).
    Trade return = cumulative product of bar returns within the block.
    """
    if len(ret) == 0:
        return []
    nz = (ret != 0.0).astype(int)
    # Detect entry (0->1) and exit (1->0) transitions
    entries = nz.diff().fillna(nz)
    block_id = (entries == 1).cumsum()
    block_id = block_id.where(nz == 1, other=0)
    trades = []
    for bid, group in ret.groupby(block_id):
        if bid == 0:
            continue
        trade_ret = float((1 + group).prod() - 1)
        trades.append(trade_ret)
    return trades


def mr_trades_from_results(r: dict) -> list[float]:
    """Extract individual trade returns from run_mr_wfo result.

    _fold_backtest accumulates trade_returns per fold but doesn't expose
    them through the public interface. We approximate using trades_from_daily_returns
    on the stitched_returns series.
    """
    raw = r.get("stitched_returns", pd.Series(dtype=float))
    return trades_from_daily_returns(raw)


def xa_trades_from_results(r: dict) -> list[float]:
    """Extract trade returns from cross-asset WFO result."""
    raw = r.get("stitched_returns", pd.Series(dtype=float))
    return trades_from_daily_returns(raw)


def ml_trades_from_results(r: dict) -> list[float]:
    """Extract trade returns from ML WFO result."""
    raw = r.get("stitched_returns", pd.Series(dtype=float))
    return trades_from_daily_returns(raw)


# ── Statistics ─────────────────────────────────────────────────────────────


def _sharpe(ret: pd.Series, ann: int = 252) -> float:
    from titan.research.metrics import sharpe as _sh

    return float(_sh(ret, periods_per_year=int(ann)))


def _sortino(ret: pd.Series, ann: int = 252) -> float:
    # Downside-only semi-std; Sortino is not in the shared metrics module yet,
    # so we keep the inline math but annualise via BARS_PER_YEAR['D'] proxy.
    import numpy as _np

    down = ret[ret < 0]
    s = down.std()
    return float(ret.mean() / s * _np.sqrt(ann)) if s > 1e-10 else 0.0


def _max_dd(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def _calmar(ret: pd.Series) -> float:
    dd = abs(_max_dd(ret))
    ann_ret = float(ret.mean() * 252)
    return ann_ret / dd if dd > 1e-6 else 0.0


def _cagr(ret: pd.Series) -> float:
    if len(ret) < 2:
        return 0.0
    eq = (1 + ret).cumprod()
    n_years = len(ret) / 252
    return float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0


def _avg_dd_duration(ret: pd.Series) -> float:
    """Average number of trading days in a drawdown period."""
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    in_dd = eq < peak
    runs, current = [], 0
    for v in in_dd:
        if v:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def _pct_deployed(ret: pd.Series) -> float:
    """% of trading days with a non-zero position (in market)."""
    return float((ret != 0.0).mean()) * 100


def trade_stats(trades: list[float]) -> dict:
    """Compute win rate, counts, avg W/L, profit factor from per-trade returns."""
    if not trades:
        return dict(
            n_trades=0,
            win_rate=0.0,
            n_wins=0,
            n_losses=0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_trade=0.0,
        )
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    return dict(
        n_trades=n,
        win_rate=len(wins) / n * 100 if n > 0 else 0.0,
        n_wins=len(wins),
        n_losses=len(losses),
        avg_win=float(np.mean(wins)) * 100 if wins else 0.0,
        avg_loss=float(np.mean(losses)) * 100 if losses else 0.0,
        profit_factor=profit_factor,
        best_trade=float(max(trades)) * 100,
        worst_trade=float(min(trades)) * 100,
        avg_trade=float(np.mean(trades)) * 100,
    )


def risk_of_ruin(trades: list[float], n_sims: int = 5000, ruin_threshold: float = 0.50) -> float:
    """Monte Carlo risk of ruin.

    Simulates n_sims equity paths by sampling with replacement from trade_returns.
    Returns % of paths that breached the ruin_threshold drawdown.
    """
    if not trades or len(trades) < 5:
        return float("nan")
    arr = np.array(trades)
    rng = np.random.default_rng(42)
    # Simulate 1000 future trades per path
    n_trades_fwd = 1000
    samples = rng.choice(arr, size=(n_sims, n_trades_fwd), replace=True)
    equity_paths = np.cumprod(1 + samples, axis=1)  # shape (n_sims, n_trades_fwd)
    # Ruin = any point where equity < (1 - ruin_threshold) starting from 1.0
    ruin_level = 1 - ruin_threshold
    ruined = (equity_paths < ruin_level).any(axis=1)
    return float(ruined.mean() * 100)


def annual_returns(ret: pd.Series) -> dict[int, float]:
    if not isinstance(ret.index, pd.DatetimeIndex):
        return {}
    return {yr: float((1 + g).prod() - 1) for yr, g in ret.groupby(ret.index.year)}


def scale_to_risk(ret: pd.Series, stop_mult: float) -> pd.Series:
    """Scale return series to 1% risk per trade via ATR-proxy stop."""
    nz = ret[ret != 0.0]
    if len(nz) < 10:
        return ret
    typical_bar_vol = float(nz.std())
    stop_dist = stop_mult * typical_bar_vol
    if stop_dist < 1e-9:
        return ret
    return ret * (RISK_PCT / stop_dist)


# ── Strategy runner ─────────────────────────────────────────────────────────


def get_raw_results(strat: dict) -> dict | None:
    from research.auto.evaluate import (
        run_cross_asset_wfo,
        run_mean_reversion_wfo,
        run_ml_wfo,
    )

    instrument = strat["instrument"]
    cfg = strat["cfg"]
    runner = strat["runner"]
    name = strat["name"]
    print(f"  [{name}] Running {runner.upper()} on {instrument} ...", end=" ", flush=True)
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
        print(f"no data (sharpe={r.get('sharpe', '?')})")
        return None

    raw = raw.sort_index()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    raw.index = raw.index.normalize()
    raw = raw.resample("D").sum()

    # Expand to business-day calendar (non-trade days = 0)
    full_idx = pd.bdate_range(raw.index.min(), raw.index.max())
    raw = raw.reindex(full_idx, fill_value=0.0)

    trade_days = int((raw != 0).sum())
    print(
        f"OK  sharpe={r.get('sharpe', '?'):.3f}  "
        f"trade_days={trade_days}  "
        f"range={raw.index[0].date()} -> {raw.index[-1].date()}"
    )
    r["stitched_returns"] = raw
    return r


# ── Reporting ───────────────────────────────────────────────────────────────

SEP = "=" * 70
SEP2 = "-" * 70


def report_strategy(
    name: str, ret: pd.Series, trades: list[float], weight: float, stop_mult: float
) -> None:
    print(f"\n{SEP}")
    print(f"  STRATEGY: {name}  (target alloc {weight:.0%})")
    print(SEP)

    scaled = scale_to_risk(ret, stop_mult)
    ts = trade_stats(trades)
    ror = risk_of_ruin(trades)

    # Metrics on 1%-risk-scaled returns
    sh = _sharpe(scaled)
    so = _sortino(scaled)
    cal = _calmar(scaled)
    cg = _cagr(scaled)
    dd = _max_dd(scaled)
    avg_dd = _avg_dd_duration(scaled)
    dep = _pct_deployed(ret)

    print(f"  Date range      : {ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"  Trading days    : {len(ret):,}  ({dep:.1f}% in market)")
    print()
    print("  --- Return Metrics (1% ATR risk-scaled) ---")
    print(f"  Sharpe Ratio    : {sh:.3f}")
    print(f"  Sortino Ratio   : {so:.3f}")
    print(f"  Calmar Ratio    : {cal:.3f}")
    print(f"  CAGR            : {cg * 100:.2f}%")
    print(f"  Max Drawdown    : {dd * 100:.2f}%")
    print(f"  Avg DD Duration : {avg_dd:.1f} days")
    print()
    print("  --- Trade Statistics ---")
    print(f"  Total Trades    : {ts['n_trades']}")
    print(f"  Win Rate        : {ts['win_rate']:.1f}%  ({ts['n_wins']} W / {ts['n_losses']} L)")
    print(f"  Avg Win         : +{ts['avg_win']:.2f}%")
    print(f"  Avg Loss        : {ts['avg_loss']:.2f}%")
    print(f"  Profit Factor   : {ts['profit_factor']:.2f}")
    print(f"  Best Trade      : +{ts['best_trade']:.2f}%")
    print(f"  Worst Trade     : {ts['worst_trade']:.2f}%")
    print(f"  Avg Trade       : {ts['avg_trade']:+.3f}%")
    print()
    print("  --- Risk ---")
    print(f"  Risk per trade  : ~{RISK_PCT * 100:.0f}% of sub-equity (ATR stop {stop_mult}x)")
    print(f"  Risk of Ruin    : {ror:.2f}%  (50% loss, 1000-trade MC, {MC_PATHS:,} paths)")
    print()
    ann = annual_returns(scaled)
    if ann:
        print("  --- Annual Returns (risk-scaled) ---")
        for yr, r_ann in sorted(ann.items()):
            bar = "#" * max(0, int(r_ann * 200)) if r_ann > 0 else "v" * max(0, int(-r_ann * 200))
            bar = bar[:30]
            print(f"  {yr}: {r_ann * 100:+7.2f}%  {bar}")


def report_portfolio(
    port_ret: pd.Series, strategy_names: list[str], clipped: list[pd.Series], weights: list[float]
) -> float:
    ts = trade_stats(trades_from_daily_returns(port_ret))
    ror = risk_of_ruin(trades_from_daily_returns(port_ret))
    sh = _sharpe(port_ret)
    so = _sortino(port_ret)
    cal = _calmar(port_ret)
    cg = _cagr(port_ret)
    dd = _max_dd(port_ret)
    avg_dd = _avg_dd_duration(port_ret)
    dep = _pct_deployed(port_ret)
    score = sh + 0.3 * 1.0 - 0.5 * max(0, -dd - 0.15)

    print(f"\n{SEP}")
    print(
        f"  COMBINED PORTFOLIO  "
        f"({' / '.join(f'{n} {w:.0%}' for n, w in zip(strategy_names, weights))})"
    )
    print(SEP)
    print(f"  Date range      : {port_ret.index[0].date()} -> {port_ret.index[-1].date()}")
    print(f"  Trading days    : {len(port_ret):,}  ({dep:.1f}% in market)")
    print()
    print("  --- Return Metrics ---")
    print(f"  Sharpe Ratio    : {sh:.3f}")
    print(f"  Sortino Ratio   : {so:.3f}")
    print(f"  Calmar Ratio    : {cal:.3f}")
    print(f"  CAGR            : {cg * 100:.2f}%")
    print(f"  Max Drawdown    : {dd * 100:.2f}%")
    print(f"  Avg DD Duration : {avg_dd:.1f} days")
    print(f"  Composite SCORE : {score:.4f}")
    print()
    print("  --- Trade Statistics (portfolio level) ---")
    print(f"  Total Trades    : {ts['n_trades']}")
    print(f"  Win Rate        : {ts['win_rate']:.1f}%  ({ts['n_wins']} W / {ts['n_losses']} L)")
    print(f"  Avg Win         : +{ts['avg_win']:.2f}%")
    print(f"  Avg Loss        : {ts['avg_loss']:.2f}%")
    print(f"  Profit Factor   : {ts['profit_factor']:.2f}")
    print()
    print("  --- Risk ---")
    print(f"  Risk of Ruin    : {ror:.2f}%  (50% loss, 1000-trade MC, {MC_PATHS:,} paths)")
    max_concurrent_loss = sum(w * RISK_PCT for w in weights)
    print(
        f"  Max simultaneous loss (all positions hit stop): {max_concurrent_loss * 100:.2f}% of total capital"
    )
    print()

    # Correlation matrix
    df = pd.concat(clipped, axis=1, join="inner").fillna(0.0)
    df.columns = strategy_names
    corr = df.corr()
    print("  --- Pairwise Correlation ---")
    hdr = " " * 14 + "".join(f"{n:>12}" for n in strategy_names)
    print(f"  {hdr}")
    for row in strategy_names:
        vals = "".join(f"{corr.loc[row, col]:>12.3f}" for col in strategy_names)
        print(f"  {row:<14}{vals}")

    # Annual breakdown
    ann = annual_returns(port_ret)
    if ann:
        print("\n  --- Annual Returns ---")
        for yr, r_ann in sorted(ann.items()):
            bar = "#" * max(0, int(r_ann * 300)) if r_ann > 0 else "v" * max(0, int(-r_ann * 300))
            bar = bar[:35]
            print(f"  {yr}: {r_ann * 100:+7.2f}%  {bar}")

    return score


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    print(SEP)
    print("  COMPREHENSIVE PORTFOLIO EVALUATION")
    print("  Top-5 Strategies | 1% ATR Risk Model | Monte Carlo RoR")
    print(SEP)
    print()

    # ── Step 1: Run all strategies ─────────────────────────────────────────
    print("Step 1: Running all 5 strategies...")
    raw_results = []
    valid_strats = []

    for strat in STRATEGIES:
        r = get_raw_results(strat)
        if r is not None:
            raw_results.append(r)
            valid_strats.append(strat)

    if len(raw_results) < 2:
        print("ERROR: fewer than 2 valid strategies")
        sys.exit(1)

    print(f"\nValid strategies: {len(raw_results)}/{len(STRATEGIES)}")

    # ── Step 2: Scale to 1% risk ───────────────────────────────────────────
    raw_rets = [r["stitched_returns"] for r in raw_results]
    scaled_rets = [scale_to_risk(ret, s["stop_mult"]) for ret, s in zip(raw_rets, valid_strats)]

    # ── Step 3: Extract per-trade data ─────────────────────────────────────
    all_trades = []
    for r, strat, runner_type in zip(
        raw_results, valid_strats, [s["runner"] for s in valid_strats]
    ):
        if runner_type == "mr":
            t = mr_trades_from_results(r)
        elif runner_type == "ml":
            t = ml_trades_from_results(r)
        else:
            t = xa_trades_from_results(r)
        all_trades.append(t)

    # ── Step 4: Per-strategy reports ───────────────────────────────────────
    print("\nStep 2: Per-strategy detailed statistics...")

    for i, (strat, ret, trades) in enumerate(zip(valid_strats, raw_rets, all_trades)):
        report_strategy(strat["name"], ret, trades, strat["target_weight"], strat["stop_mult"])

    # ── Step 5: Find overlapping date range ────────────────────────────────
    print(f"\n{SEP}")
    print("  PORTFOLIO COMBINATION")
    print(SEP)

    # Find best overlapping subset
    best_subset = None
    best_overlap = -1
    n = len(raw_rets)
    for size in range(n, 1, -1):
        for combo in combinations(range(n), size):
            cs = max(raw_rets[i].index[0] for i in combo)
            ce = min(raw_rets[i].index[-1] for i in combo)
            days = (ce - cs).days
            if days > best_overlap:
                best_overlap = days
                best_subset = combo
        if best_subset is not None and best_overlap > 0:
            break

    if best_subset is None or best_overlap <= 0:
        print("ERROR: no overlapping date range found")
        sys.exit(1)

    skipped = set(range(n)) - set(best_subset)
    for i in skipped:
        print(
            f"  EXCLUDED (no overlap): {valid_strats[i]['name']} "
            f"({raw_rets[i].index[0].date()} -> {raw_rets[i].index[-1].date()})"
        )

    port_strats = [valid_strats[i] for i in best_subset]
    port_scaled = [scaled_rets[i] for i in best_subset]
    port_names = [s["name"] for s in port_strats]
    port_weights_raw = [s["target_weight"] for s in port_strats]
    # Renormalise weights
    w_sum = sum(port_weights_raw)
    port_weights = [w / w_sum for w in port_weights_raw]

    common_start = max(c.index[0] for c in port_scaled)
    common_end = min(c.index[-1] for c in port_scaled)
    print(f"\n  Common period: {common_start.date()} -> {common_end.date()} ({best_overlap} days)")
    print(f"  Strategies in portfolio: {len(port_strats)}")

    clipped = [c[common_start:common_end] for c in port_scaled]

    # Combine on inner join
    df = pd.concat(clipped, axis=1, join="inner").fillna(0.0)
    df.columns = port_names
    w = np.array(port_weights)
    port_ret = (df * w).sum(axis=1)

    score = report_portfolio(port_ret, port_names, clipped, port_weights)

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Strategies in portfolio : {len(port_strats)}")
    print(f"  Common evaluation period: {common_start.date()} -> {common_end.date()}")
    print(f"  Portfolio Sharpe        : {_sharpe(port_ret):.3f}")
    print(f"  Portfolio CAGR          : {_cagr(port_ret) * 100:.2f}%")
    print(f"  Portfolio Max DD        : {_max_dd(port_ret) * 100:.2f}%")
    print(f"  Composite SCORE         : {score:.4f}")
    print("  Baseline (AUD/JPY solo) : 5.1368")
    print(SEP)


if __name__ == "__main__":
    main()
