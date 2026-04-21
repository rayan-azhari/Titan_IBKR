"""run_retracement_validation.py -- Phase 0A: Trend Retracement Concept Validation.

Tests whether entering pullbacks within an AND-gated trend improves Sharpe
over entering on trend alone (baseline: GLD confluence +1.46).

Pullback detection uses existing signals from the 52-signal IC library:
  - keltner_pos: position within Keltner channel (negative = pulled back)
  - rsi_14_dev: RSI deviation from 50 (negative = oversold)
  - accel_rsi14: RSI momentum change (positive = turning up)

Entry rule: confluence confirms trend AND pullback detected AND momentum turning.

Usage:
    uv run python research/retracement/run_retracement_validation.py
    uv run python research/retracement/run_retracement_validation.py --pair GLD
    uv run python research/retracement/run_retracement_validation.py --pair SPY --sweep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    SCALE_MAP,
    _load_ohlcv,
    build_all_signals,
)

WINDOW_1Y_H1 = 252 * 7  # ~7 H1 bars/day for equities
TF_WEIGHTS = {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30}


def _vectorized_position(sig: np.ndarray, threshold: float, exit_buf: float) -> np.ndarray:
    n = len(sig)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        s = sig[i]
        if cur == 0.0:
            if s >= threshold:
                cur = 1.0
            elif s <= -threshold:
                cur = -1.0
        elif cur > 0 and s < -exit_buf:
            cur = 0.0
        elif cur < 0 and s > exit_buf:
            cur = 0.0
        pos[i] = cur
    return pos


def compute_confluence_and_pullback(
    df: pd.DataFrame,
    window_1y: int,
) -> dict[str, pd.Series]:
    """Compute confluence score AND pullback indicators on single H1 stream."""
    close = df["close"]

    # Compute signals at each scale
    per_scale_trend_mom = {}
    per_scale_keltner = {}

    for label, mult in SCALE_MAP.items():
        prefix = f"{label}_" if mult > 1 else ""
        sigs = build_all_signals(df, window_1y, period_scale=mult, name_prefix=prefix)
        # trend_mom for confluence
        tm_col = f"{prefix}trend_mom"
        if tm_col in sigs.columns:
            per_scale_trend_mom[label] = sigs[tm_col]

        # pullback indicators (only at H1 scale for timing)
        if mult == 1:
            for col in ["keltner_pos", "rsi_14_dev", "accel_rsi14", "donchian_pos_20", "rsi_7_dev"]:
                if col in sigs.columns:
                    per_scale_keltner[col] = sigs[col]

    # AND-gate confluence
    aligned = pd.concat(per_scale_trend_mom, axis=1).dropna()
    signs = np.sign(aligned.values)
    agreement = (signs > 0).all(axis=1) | (signs < 0).all(axis=1)

    weighted = pd.Series(0.0, index=aligned.index)
    for label in SCALE_MAP:
        if label in aligned.columns:
            weighted += aligned[label] * TF_WEIGHTS.get(label, 0.25)
    confluence = weighted.where(agreement, 0.0)

    result = {"confluence": confluence, "close": close}
    result.update(per_scale_keltner)
    return result


def backtest_retracement(
    close: pd.Series,
    confluence: pd.Series,
    keltner_pos: pd.Series,
    rsi_dev: pd.Series,
    rsi_accel: pd.Series,
    mode: str = "confluence_only",
    threshold: float = 0.75,
    exit_buffer: float = 0.10,
    keltner_entry_max: float = -0.1,
    keltner_entry_min: float = -0.5,
    rsi_entry_max: float = -5.0,
    rsi_accel_min: float = 0.0,
    spread_bps: float = 3.0,
    is_ratio: float = 0.70,
) -> dict:
    """Backtest with different entry modes.

    Modes:
      - confluence_only: enter on confluence z-score > threshold (baseline)
      - retracement: enter on confluence + pullback conditions
      - aggressive_retrace: enter on confluence + shallow pullback
    """
    # Align all series
    common = confluence.dropna().index
    for s in [keltner_pos, rsi_dev, rsi_accel]:
        common = common.intersection(s.dropna().index)
    common = common.intersection(close.dropna().index)

    confluence = confluence.reindex(common)
    close = close.reindex(common)
    keltner = keltner_pos.reindex(common).fillna(0.0)
    rsi = rsi_dev.reindex(common).fillna(0.0)
    accel = rsi_accel.reindex(common).fillna(0.0)

    n = len(common)
    is_n = int(n * is_ratio)

    # Z-score confluence on IS
    is_mean = confluence.iloc[:is_n].mean()
    is_std = confluence.iloc[:is_n].std()
    if is_std < 1e-8:
        is_std = 1.0
    score_z = (confluence - is_mean) / is_std

    # Build entry signal based on mode
    sig_shifted = score_z.shift(1).fillna(0.0).values
    kelt_shifted = keltner.shift(1).fillna(0.0).values
    rsi_shifted = rsi.shift(1).fillna(0.0).values
    accel_shifted = accel.shift(1).fillna(0.0).values

    if mode == "confluence_only":
        # Baseline: just confluence z-score
        entry_signal = sig_shifted
    elif mode == "retracement":
        # Confluence + pullback: only enter when pulled back
        entry_signal = np.where(
            (kelt_shifted < keltner_entry_max)
            & (kelt_shifted > keltner_entry_min)
            & (rsi_shifted < rsi_entry_max)
            & (accel_shifted > rsi_accel_min),
            sig_shifted,  # use confluence z as signal strength
            0.0,  # zero out when no pullback
        )
    elif mode == "aggressive_retrace":
        # Lighter pullback requirement
        entry_signal = np.where(
            (kelt_shifted < 0.0) & (rsi_shifted < 0.0),
            sig_shifted,
            0.0,
        )
    else:
        entry_signal = sig_shifted

    pos = _vectorized_position(entry_signal, threshold, exit_buffer)

    # Returns
    close_arr = close.values
    bar_rets = np.empty(n)
    bar_rets[0] = 0.0
    bar_rets[1:] = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]

    transitions = np.abs(np.diff(pos, prepend=0))
    strat_rets = bar_rets * pos - transitions * spread_bps / 10_000

    # Daily aggregation
    is_daily = pd.Series(strat_rets[:is_n], index=common[:is_n]).resample("D").sum()
    is_daily = is_daily[is_daily != 0.0]
    oos_daily = pd.Series(strat_rets[is_n:], index=common[is_n:]).resample("D").sum()
    oos_daily = oos_daily[oos_daily != 0.0]

    def _sharpe(d):
        from titan.research.metrics import BARS_PER_YEAR as _BPY
        from titan.research.metrics import sharpe as _sh

        if len(d) < 20:
            return 0.0
        return float(_sh(d, periods_per_year=_BPY["D"]))

    def _dd(d):
        if len(d) < 5:
            return 0.0
        eq = (1 + d).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    is_sh = _sharpe(is_daily)
    oos_sh = _sharpe(oos_daily)
    oos_trades = int(np.sum(transitions[is_n:] > 0))
    tim = np.mean(np.abs(pos[is_n:])) * 100

    return {
        "mode": mode,
        "is_sharpe": round(is_sh, 3),
        "oos_sharpe": round(oos_sh, 3),
        "parity": round(oos_sh / is_sh, 3) if abs(is_sh) > 0.01 else 0.0,
        "oos_dd_pct": round(_dd(oos_daily) * 100, 2),
        "oos_trades": oos_trades,
        "time_in_market_pct": round(tim, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend Retracement Validation")
    parser.add_argument("--pair", default="GLD")
    parser.add_argument("--sweep", action="store_true", help="Sweep pullback parameters")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  TREND RETRACEMENT VALIDATION -- {args.pair} H1")
    print("=" * 70)

    df = _load_ohlcv(args.pair, "H1")
    print(f"\n  Loaded {len(df)} H1 bars")

    print("\n  Computing confluence + pullback signals...")
    signals = compute_confluence_and_pullback(df, WINDOW_1Y_H1)

    confluence = signals["confluence"]
    close = signals["close"]
    keltner = signals.get("keltner_pos", pd.Series(0.0, index=close.index))
    rsi = signals.get("rsi_14_dev", pd.Series(0.0, index=close.index))
    accel = signals.get("accel_rsi14", pd.Series(0.0, index=close.index))

    print(f"  Confluence signal: {len(confluence.dropna())} bars")

    # Test modes
    modes = ["confluence_only", "aggressive_retrace", "retracement"]

    if args.sweep:
        # Sweep pullback parameters
        keltner_maxes = [-0.05, -0.1, -0.2, -0.3]
        rsi_maxes = [0.0, -5.0, -10.0]
        thresholds = [0.50, 0.75, 1.0]

        rows = []
        for th in thresholds:
            # Baseline
            r = backtest_retracement(
                close,
                confluence,
                keltner,
                rsi,
                accel,
                mode="confluence_only",
                threshold=th,
            )
            rows.append({"threshold": th, "keltner_max": "N/A", "rsi_max": "N/A", **r})

            for km in keltner_maxes:
                for rm in rsi_maxes:
                    r = backtest_retracement(
                        close,
                        confluence,
                        keltner,
                        rsi,
                        accel,
                        mode="retracement",
                        threshold=th,
                        keltner_entry_max=km,
                        rsi_entry_max=rm,
                    )
                    rows.append({"threshold": th, "keltner_max": km, "rsi_max": rm, **r})

        df_results = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
        print(
            f"\n  {'Mode':>16} {'Thr':>5} {'Kelt':>6} {'RSI':>5}"
            f" | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6} | {'DD%':>6} {'Trd':>4} {'TIM%':>5}"
        )
        print("  " + "-" * 75)
        for _, r in df_results.head(20).iterrows():
            flag = "+" if r["oos_sharpe"] > 0.5 and r["oos_trades"] >= 10 else " "
            print(
                f" {flag}{r['mode']:>15} {r['threshold']:>5.2f}"
                f" {str(r['keltner_max']):>6} {str(r['rsi_max']):>5}"
                f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
                f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4} {r['time_in_market_pct']:>4.0f}%"
            )

        save_path = REPORTS_DIR / f"retracement_sweep_{args.pair.lower()}.csv"
        df_results.to_csv(save_path, index=False)
        print(f"\n  Saved to: {save_path}")
    else:
        print(
            f"\n  {'Mode':>20} | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
            f" | {'DD%':>6} {'Trd':>4} {'TIM%':>5}"
        )
        print("  " + "-" * 65)
        for mode in modes:
            r = backtest_retracement(
                close,
                confluence,
                keltner,
                rsi,
                accel,
                mode=mode,
            )
            flag = "+" if r["oos_sharpe"] > 0.5 else " "
            print(
                f" {flag}{mode:>19}"
                f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
                f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4} {r['time_in_market_pct']:>4.0f}%"
            )

        # Compare
        baseline = backtest_retracement(
            close,
            confluence,
            keltner,
            rsi,
            accel,
            mode="confluence_only",
        )
        retrace = backtest_retracement(
            close,
            confluence,
            keltner,
            rsi,
            accel,
            mode="retracement",
        )
        delta = retrace["oos_sharpe"] - baseline["oos_sharpe"]
        print(f"\n  Retracement vs Baseline: delta Sharpe = {delta:+.3f}")
        print(f"  Pass (delta >= 0.2): {'PASS' if delta >= 0.2 else 'FAIL'}")


if __name__ == "__main__":
    main()
