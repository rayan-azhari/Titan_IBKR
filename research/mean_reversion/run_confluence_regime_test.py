"""run_confluence_regime_test.py -- Test multi-scale confluence as MR regime filter.

Tests whether AND-gated confluence DISAGREEMENT (scales don't agree on direction)
can serve as a regime filter for the VWAP mean-reversion strategy.

Hypothesis: when all 4 scales agree on trend direction, mean reversion fails.
When scales disagree (no clear trend), mean reversion works.

Also sweeps percentile thresholds to find the optimal entry aggressiveness.

Usage:
    uv run python research/mean_reversion/run_confluence_regime_test.py
    uv run python research/mean_reversion/run_confluence_regime_test.py --pair EUR_USD
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    SCALE_MAP,
    build_all_signals,
)

# ── M5-specific scale multipliers ────────────────────────────────────────────
# M5 has ~288 bars/day (FX, 24h), ~1440 bars/week
M5_SCALE_MAP = {
    "M5": 1,  # native
    "H1": 12,  # 12 M5 bars per H1
    "H4": 48,  # 48 M5 bars per H4
    "D": 288,  # 288 M5 bars per day (FX 24h)
}

# Alternative: use H1 data with standard scales (faster, same concept)
H1_SCALE_MAP = SCALE_MAP  # {"H1": 1, "H4": 4, "D": 24, "W": 120}

WINDOW_1Y_H1 = 252 * 22

# Confluence signals to test as regime filters
FILTER_SIGNALS = ["trend_mom", "ma_spread_5_20", "rsi_14_dev", "donchian_pos_20", "ema_slope_10"]

# Percentile threshold grids to sweep
TIER_GRIDS = {
    "aggressive": [0.80, 0.85, 0.90, 0.95],
    "standard": [0.90, 0.95, 0.98, 0.99],
    "conservative": [0.95, 0.98, 0.99, 0.999],
}
TIER_SIZES = [1, 2, 4, 8]


# ── Data loading ─────────────────────────────────────────────────────────────


def load_h1(pair: str) -> pd.DataFrame:
    """Load H1 data (faster than M5, same confluence concept)."""
    path = DATA_DIR / f"{pair}_H1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


# ── VWAP computation on H1 ──────────────────────────────────────────────────


def compute_vwap_deviation(close: pd.Series, anchor_period: int = 24) -> pd.Series:
    """Simple rolling VWAP deviation (no volume, price-only proxy).

    Uses rolling mean over anchor_period as VWAP proxy.
    Deviation = (close - vwap) / vwap.
    """
    vwap = close.rolling(anchor_period).mean()
    deviation = (close - vwap) / vwap.clip(lower=1e-8)
    return deviation


def compute_percentile_levels(deviation: pd.Series, window: int, pcts: list[float]) -> pd.DataFrame:
    """Rolling percentile levels for the deviation distribution."""
    levels = pd.DataFrame(index=deviation.index)
    for p in pcts:
        levels[f"p{int(p * 100)}"] = deviation.abs().rolling(window).quantile(p)
    # Shift by 1 to prevent look-ahead
    return levels.shift(1)


# ── Confluence regime filter ─────────────────────────────────────────────────


def build_confluence_disagreement_mask(
    df: pd.DataFrame, signal_name: str, scales: dict[str, int] | None = None
) -> pd.Series:
    """True when scales DISAGREE on direction = ranging regime = allow MR entries.

    False when all scales agree (trending) = block MR entries.
    """
    if scales is None:
        scales = H1_SCALE_MAP

    per_scale_sig = {}
    for label, mult in scales.items():
        prefix = f"{label}_" if mult > 1 else ""
        sigs = build_all_signals(df, WINDOW_1Y_H1, period_scale=mult, name_prefix=prefix)
        col = f"{prefix}{signal_name}"
        if col in sigs.columns:
            per_scale_sig[label] = sigs[col]

    if len(per_scale_sig) < 2:
        return pd.Series(True, index=df.index)  # fallback: allow all

    aligned = pd.concat(per_scale_sig, axis=1).dropna()
    signs = np.sign(aligned.values)

    # All agree = trending (block MR). Any disagree = ranging (allow MR).
    all_positive = (signs > 0).all(axis=1)
    all_negative = (signs < 0).all(axis=1)
    trending = all_positive | all_negative

    # Return True where scales DISAGREE (= ranging = allow MR)
    mask = pd.Series(~trending, index=aligned.index)

    # Reindex to full df index (NaN periods = warmup, default to False/blocked)
    return mask.reindex(df.index, fill_value=False)


def build_atr_regime_mask(
    df: pd.DataFrame, atr_period: int = 14, window: int = 500, threshold_pct: float = 0.30
) -> pd.Series:
    """Original ATR regime gate: True when ATR < threshold percentile."""
    from titan.strategies.ml.features import atr as compute_atr

    atr_val = compute_atr(df, atr_period)
    atr_pctile = atr_val.rolling(window).rank(pct=True)
    return atr_pctile < threshold_pct


# ── Vectorized MR backtest ───────────────────────────────────────────────────


def backtest_mr(
    close: pd.Series,
    deviation: pd.Series,
    levels: pd.DataFrame,
    regime_mask: pd.Series,
    tier_sizes: list[float],
    reversion_pct: float = 0.5,
    spread_bps: float = 3.0,
    slippage_bps: float = 1.0,
    ny_close_hour: int = 21,
    is_ratio: float = 0.70,
) -> dict:
    """Simplified vectorized MR backtest with tiered grid entries.

    Returns IS/OOS Sharpe, trade count, win rate, avg hold bars, and daily returns.
    """
    n = len(close)

    # Session filter: only allow entries 07:00-12:00 UTC
    hours = close.index.hour
    session_mask = (hours >= 7) & (hours < 12)

    # Combined entry gate
    entry_allowed = regime_mask.reindex(close.index, fill_value=False) & session_mask

    # Simulate trades
    close_arr = close.values
    dev_arr = deviation.values
    gate_arr = entry_allowed.values
    hour_arr = hours.values

    # Track state
    position = np.zeros(n)  # net position size
    entry_price = 0.0
    entry_bar = 0
    tiers_hit = set()
    trade_returns = []
    trade_durations = []
    bar_pnl = np.zeros(n)

    cost_per_unit = (spread_bps + slippage_bps) / 10_000

    for i in range(1, n):
        px = close_arr[i]
        dev = abs(dev_arr[i]) if not np.isnan(dev_arr[i]) else 0.0
        prev_pos = position[i - 1]

        # Exit checks first
        if prev_pos != 0:
            # 1. NY close hard exit
            if hour_arr[i] >= ny_close_hour:
                ret = (px - entry_price) / entry_price * np.sign(prev_pos) - cost_per_unit
                trade_returns.append(ret)
                trade_durations.append(i - entry_bar)
                bar_pnl[i] = ret * abs(prev_pos)
                position[i] = 0
                entry_price = 0.0
                tiers_hit = set()
                continue

            # 2. Reversion TP (deviation moved reversion_pct back toward 0)
            if (
                dev < levels.iloc[i, 0] * (1 - reversion_pct)
                if len(levels.columns) > 0 and not np.isnan(levels.iloc[i, 0])
                else False
            ):
                ret = (px - entry_price) / entry_price * np.sign(prev_pos) - cost_per_unit
                trade_returns.append(ret)
                trade_durations.append(i - entry_bar)
                bar_pnl[i] = ret * abs(prev_pos)
                position[i] = 0
                entry_price = 0.0
                tiers_hit = set()
                continue

        # Entry checks (only when flat or adding to position)
        if gate_arr[i]:
            for tier_idx, tier_col in enumerate(levels.columns):
                if tier_idx in tiers_hit:
                    continue
                lvl = levels.iloc[i, tier_idx]
                if np.isnan(lvl):
                    continue
                if dev > lvl:
                    size = tier_sizes[tier_idx] if tier_idx < len(tier_sizes) else 1
                    # Direction: short if price above VWAP, long if below
                    direction = -1.0 if dev_arr[i] > 0 else 1.0
                    if position[i] == 0:
                        entry_price = px
                        entry_bar = i
                    else:
                        # Weighted average entry
                        old_size = abs(position[i])
                        entry_price = (entry_price * old_size + px * size) / (old_size + size)
                    position[i] += direction * size
                    tiers_hit.add(tier_idx)
                    bar_pnl[i] -= cost_per_unit * size  # entry cost

        # Carry position forward if no exit
        if position[i] == 0 and prev_pos != 0 and bar_pnl[i] == 0:
            position[i] = prev_pos
            # Mark-to-market PnL
            bar_pnl[i] = (px - close_arr[i - 1]) / close_arr[i - 1] * prev_pos
        elif position[i] == 0 and prev_pos == 0:
            position[i] = 0

    # Daily aggregation
    daily_pnl = pd.Series(bar_pnl, index=close.index).resample("D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0.0]

    is_daily = daily_pnl.iloc[: int(len(daily_pnl) * is_ratio)]
    oos_daily = daily_pnl.iloc[int(len(daily_pnl) * is_ratio) :]

    def _sharpe(d):
        if len(d) < 10:
            return 0.0
        return float(d.mean() / d.std() * sqrt(252)) if d.std() > 1e-9 else 0.0

    def _dd(d):
        if len(d) < 5:
            return 0.0
        eq = (1 + d).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    n_trades = len(trade_returns)
    win_rate = sum(1 for r in trade_returns if r > 0) / n_trades if n_trades > 0 else 0
    avg_duration = np.mean(trade_durations) if trade_durations else 0

    return {
        "is_sharpe": round(_sharpe(is_daily), 3),
        "oos_sharpe": round(_sharpe(oos_daily), 3),
        "n_trades": n_trades,
        "trades_per_week": round(n_trades / max((n / (252 * 22)) * 52, 1), 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_hold_bars": round(avg_duration, 1),
        "avg_hold_hours": round(avg_duration, 1),  # H1 bars = hours
        "oos_dd_pct": round(_dd(oos_daily) * 100, 2),
        "oos_daily": oos_daily,
    }


# ── Main sweep ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MR FX + Confluence Regime Filter Test")
    parser.add_argument("--pair", default="EUR_USD")
    args = parser.parse_args()

    print("=" * 80)
    print(f"  MR FX + CONFLUENCE REGIME FILTER -- {args.pair} H1")
    print("=" * 80)

    df = load_h1(args.pair)
    close = df["close"]
    n = len(close)
    print(f"\n  Loaded {n} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    # VWAP deviation (24-bar rolling = 1 day on H1)
    deviation = compute_vwap_deviation(close, anchor_period=24)

    # Build regime masks
    print("\n  Building regime filters...")

    # 1. ATR gate (original)
    atr_mask = build_atr_regime_mask(df, threshold_pct=0.30)
    atr_pct_bars = atr_mask.sum() / n * 100

    # 2. Confluence disagreement gates (one per signal)
    confluence_masks = {}
    for sig_name in FILTER_SIGNALS:
        print(f"    Computing confluence for {sig_name}...")
        mask = build_confluence_disagreement_mask(df, sig_name)
        pct_bars = mask.sum() / n * 100
        confluence_masks[sig_name] = mask
        print(f"      {sig_name}: {pct_bars:.0f}% of bars allowed")

    # 3. No filter (baseline)
    no_filter = pd.Series(True, index=df.index)

    print(f"\n    ATR gate: {atr_pct_bars:.0f}% of bars allowed")

    # Combine all filters
    all_filters = {
        "no_filter": no_filter,
        "atr_only": atr_mask,
    }
    for sig_name, mask in confluence_masks.items():
        all_filters[f"conf_{sig_name}"] = mask
        # Also test ATR + confluence combined
        all_filters[f"atr+conf_{sig_name}"] = atr_mask & mask

    # Sweep: filters × percentile grids
    print(f"\n  Sweeping {len(all_filters)} filters × {len(TIER_GRIDS)} tier grids...")
    print(
        f"\n  {'Filter':<30} {'Tiers':>12} | {'IS Sh':>6} {'OOS Sh':>7} | {'Trd':>5} {'T/wk':>5}"
        f" {'WR%':>5} {'AvgH':>5} | {'DD%':>6}"
    )
    print("  " + "-" * 90)

    rows = []
    for filter_name, mask in all_filters.items():
        for grid_name, tiers_pct in TIER_GRIDS.items():
            levels = compute_percentile_levels(deviation, window=500, pcts=tiers_pct)

            result = backtest_mr(
                close,
                deviation,
                levels,
                mask,
                TIER_SIZES,
                reversion_pct=0.5,
                spread_bps=2.0,
                slippage_bps=1.0,
            )

            flag = "+" if result["oos_sharpe"] > 0.5 and result["n_trades"] >= 20 else " "
            tier_str = "/".join(str(int(p * 100)) for p in tiers_pct)
            print(
                f" {flag}{filter_name:<29} {tier_str:>12}"
                f" | {result['is_sharpe']:>+6.3f} {result['oos_sharpe']:>+7.3f}"
                f" | {result['n_trades']:>5} {result['trades_per_week']:>5.1f}"
                f" {result['win_rate']:>4.0f}% {result['avg_hold_hours']:>4.0f}h"
                f" | {result['oos_dd_pct']:>+5.1f}%"
            )

            rows.append(
                {
                    "filter": filter_name,
                    "tiers": grid_name,
                    "tiers_pct": tier_str,
                    **{k: v for k, v in result.items() if k != "oos_daily"},
                }
            )

    results_df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("  TOP 10 BY OOS SHARPE")
    print(f"{'=' * 60}")
    top = results_df.head(10)
    for _, r in top.iterrows():
        flag = "+" if r["oos_sharpe"] > 0.5 and r["n_trades"] >= 20 else " "
        print(
            f" {flag}{r['filter']:<30} {r['tiers']:>12}"
            f" | OOS Sh={r['oos_sharpe']:>+.3f} Trd={r['n_trades']:>4}"
            f" WR={r['win_rate']:.0f}% Hold={r['avg_hold_hours']:.0f}h"
        )

    # Compare confluence vs ATR
    print(f"\n{'=' * 60}")
    print("  FILTER COMPARISON (standard tiers: 90/95/98/99)")
    print(f"{'=' * 60}")
    std = results_df[results_df["tiers"] == "standard"]
    for _, r in std.sort_values("oos_sharpe", ascending=False).iterrows():
        flag = "+" if r["oos_sharpe"] > 0.5 and r["n_trades"] >= 20 else " "
        print(
            f" {flag}{r['filter']:<30} | OOS Sh={r['oos_sharpe']:>+.3f}"
            f" Trd={r['n_trades']:>4} WR={r['win_rate']:.0f}%"
            f" T/wk={r['trades_per_week']:.1f}"
        )

    # Save
    save_path = REPORTS_DIR / f"mr_confluence_regime_{args.pair.lower()}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
