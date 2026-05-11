"""Hold-period audit for the Samir-Stack and live champion-portfolio strategies.

For each strategy, runs the canonical backtest and extracts the hold period
(measured in bars from entry to exit) for every distinct position. Reports
mean / min / max / median / 95th-percentile across all trades.

Bar units differ by strategy:
  - Samir-Stack          → daily bars
  - bond_equity_*        → daily bars
  - mr_audjpy            → H1 bars (1 hour)

Hold-period definitions:

  Samir-Stack: a "trade" is a contiguous block where the equity sleeve
  weight is > 0 (regardless of which tier 1/2/3 within the block). Tier
  transitions inside a held position do NOT split the trade.

  bond_equity_*: a "trade" is a contiguous block where position == 1
  (long) on the lagged-position series.

  mr_audjpy: a "trade" is a contiguous block where the strategy holds
  ANY tier of the grid. Sub-tier additions don't split the trade.

Usage:
    uv run python research/analysis/hold_period_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _bar_blocks(in_position: np.ndarray) -> list[int]:
    """Return list of block lengths where ``in_position`` is True."""
    if not in_position.any():
        return []
    # Detect rising edges (False → True) and falling edges (True → False)
    padded = np.concatenate(([False], in_position, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [int(e - s) for s, e in zip(starts, ends)]


def _summarise(blocks: list[int], unit: str = "bars") -> dict:
    if not blocks:
        return {"n_trades": 0, "unit": unit}
    arr = np.array(blocks, dtype=float)
    return {
        "n_trades": len(blocks),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p95": float(np.percentile(arr, 95)),
        "unit": unit,
    }


def _fmt_bars(stats: dict, *, bars_per_day: float = 1.0) -> str:
    """Convert bar-stats to a human-readable line. ``bars_per_day=1`` for
    daily bars, ``24`` for H1 (so we can show days too)."""
    if stats["n_trades"] == 0:
        return "  (no trades)"
    unit = stats["unit"]
    line = (
        f"  n={stats['n_trades']:>4}  mean={stats['mean']:>6.1f} {unit}  "
        f"median={stats['median']:>5.1f}  min={stats['min']:>4}  "
        f"max={stats['max']:>5}  p95={stats['p95']:>6.1f}"
    )
    if bars_per_day != 1.0:
        line += (
            f"   |  ~ mean {stats['mean'] / bars_per_day:.1f}d  "
            f"max {stats['max'] / bars_per_day:.1f}d"
        )
    return line


# ── Samir-Stack ────────────────────────────────────────────────────────────


def samir_stack_hold_periods() -> dict:
    """Run the canonical Samir-Stack and extract hold periods of the equity
    sleeve. Defined as: contiguous bars where ``equity_pos > 0``. Tier
    transitions (1/2/3) within a held position don't split the trade.
    """
    from research.samir_stack.data_loader import load_panel
    from research.samir_stack.indicators import build_indicator_panel
    from research.samir_stack.regime_score import regime_score_equal
    from research.samir_stack.stacked_strategy import (
        StackedConfig,
        run_stacked_strategy,
    )

    data = load_panel(start="2003-04-01", end="2026-04-02")
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data["hyg"],
        ief_close=data["ief"],
        tlt_close=data["tlt"],
    )
    score = regime_score_equal(panel)

    cfg = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=3.0)
    out = run_stacked_strategy(data["spy"], data["ief"], score, cfg)

    in_pos = (out["equity_pos"] > 0).to_numpy()
    blocks = _bar_blocks(in_pos)
    return _summarise(blocks, unit="days")


# ── bond_equity_* live deployments ─────────────────────────────────────────

# Mirror the live config from scripts/run_portfolio.py.
LIVE_BOND_EQUITY = [
    ("IHYU", "CSPX", 20, 20, 0.50),
    ("IHYG", "VUSD", 5, 5, 0.25),
    ("IHYG", "EIMI", 5, 5, 0.25),
]


def bond_equity_hold_periods() -> dict[str, dict]:
    """For each live `bond_equity_*` strategy, run the WFO and extract
    hold periods from the stitched OOS positions.
    """
    from research.cross_asset.run_bond_equity_wfo import (
        _position_with_hold,
        load_daily,
    )

    results: dict[str, dict] = {}
    for signal_sym, target_sym, lb, hd, thr in LIVE_BOND_EQUITY:
        bond_close = load_daily(signal_sym)
        target_close = load_daily(target_sym)
        common = bond_close.index.intersection(target_close.index)
        bond_mom = np.log(
            bond_close.reindex(common) / bond_close.reindex(common).shift(lb)
        ).dropna()
        if len(bond_mom) < 100:
            continue
        # Z-score on the full series (proxy for the rolling-fold IS calibration).
        mu = float(bond_mom.mean())
        sigma = float(bond_mom.std()) or 1.0
        z = ((bond_mom - mu) / sigma).to_numpy()
        pos = _position_with_hold(z, threshold=thr, hold_days=hd)
        in_pos = pos > 0
        blocks = _bar_blocks(in_pos)
        results[f"{signal_sym}->{target_sym}"] = _summarise(blocks, unit="days")
    return results


# ── mr_audjpy ──────────────────────────────────────────────────────────────


def mr_audjpy_hold_periods() -> dict:
    """Replay the AUD/JPY MR strategy on H1 bars and extract hold periods.

    Hold = contiguous H1 bars where ANY tier of the grid is open. Reported
    in H1 bars and converted to hours.
    """
    DATA = PROJECT_ROOT / "data" / "AUD_JPY_H1.parquet"
    if not DATA.exists():
        return {"n_trades": 0, "unit": "hours", "note": f"missing {DATA}"}

    df = pd.read_parquet(DATA).sort_index()
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))

    closes = df["close"].astype(float).to_numpy()
    highs = df.get("high", df["close"]).astype(float).to_numpy()
    lows = df.get("low", df["close"]).astype(float).to_numpy()
    ts = df.index

    # Mirror the live config.
    VWAP_ANCHOR = 24
    PCT_WINDOW = 500
    REVERSION = 0.50
    NY_CLOSE = 21
    SESSION_START = 7
    SESSION_END = 12
    DON_PERIOD = 20
    DON_SCALES = [1, 4, 24, 120]
    TIERS_PCT = [0.95, 0.98, 0.99, 0.999]

    n = len(closes)
    min_bars = DON_PERIOD * DON_SCALES[-1] + PCT_WINDOW
    in_position = np.zeros(n, dtype=bool)
    pos_open = False
    rolling_dev = np.zeros(n)

    # Pre-compute rolling VWAP deviation
    for i in range(VWAP_ANCHOR, n):
        vwap = closes[i - VWAP_ANCHOR : i].mean()
        rolling_dev[i] = abs((closes[i] - vwap) / max(abs(vwap), 1e-8))

    for i in range(min_bars, n):
        # NY close hard-flat
        if pos_open and ts[i].hour >= NY_CLOSE:
            pos_open = False
        # Compute current percentile threshold
        window_dev = rolling_dev[i - PCT_WINDOW : i]
        level = np.percentile(window_dev, TIERS_PCT[0] * 100)
        # Reversion TP
        if pos_open and rolling_dev[i] < level * (1 - REVERSION):
            pos_open = False
        # Entry: session + regime gate + threshold
        if not pos_open and SESSION_START <= ts[i].hour < SESSION_END:
            # Donchian regime check: scales must DISAGREE
            signs = []
            for mult in DON_SCALES:
                w = DON_PERIOD * mult
                if i < w:
                    continue
                lo = lows[i - w : i].min()
                hi = highs[i - w : i].max()
                if hi - lo < 1e-8:
                    continue
                don_pos = (closes[i] - lo) / (hi - lo) - 0.5
                signs.append(np.sign(don_pos))
            ranging = not (all(s > 0 for s in signs) or all(s < 0 for s in signs))
            if ranging and rolling_dev[i] > level:
                pos_open = True
        in_position[i] = pos_open

    blocks = _bar_blocks(in_position)
    return _summarise(blocks, unit="H1 bars")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 90)
    print("HOLD-PERIOD AUDIT")
    print("=" * 90)

    print("\nSamir-Stack (full pipeline, equity sleeve, daily bars):")
    s = samir_stack_hold_periods()
    print(_fmt_bars(s))

    print("\nLive champion: bond_equity_* strategies (daily bars):")
    for k, v in bond_equity_hold_periods().items():
        print(f"  {k}:")
        print(_fmt_bars(v))

    print("\nLive champion: mr_audjpy (H1 bars):")
    a = mr_audjpy_hold_periods()
    if "note" in a:
        print(f"  ({a['note']})")
    else:
        print(_fmt_bars(a, bars_per_day=24.0))

    print("\n" + "=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
