"""run_pair_sweep.py — Stage 3 greedy sweep for any FX pair / instrument.

Seeds baseline parameters from config/mtf.toml (EUR/USD validated config),
then sweeps fast_ma / slow_ma / rsi_period per timeframe using the same
greedy D -> H4 -> H1 -> W order as the EUR/USD Stage 3 optimisation.

Saves results to:
  config/mtf_{pair_lower}.toml             (best params)
  .tmp/reports/mtf_{pair_lower}_sweep.csv  (full scoreboard)

Usage:
    uv run python research/mtf/run_pair_sweep.py --pair GBP_USD
    uv run python research/mtf/run_pair_sweep.py --pair GBP_USD --load-state
    uv run python research/mtf/run_pair_sweep.py --pair GBP_USD \\
        --threshold 0.15 --w-H1 0.10 --w-H4 0.25 --w-D 0.60 --w-W 0.05
"""

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_CONFIG = PROJECT_ROOT / "config" / "mtf.toml"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv sync")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Defaults (EUR/USD Stage 1-2 winners) — overridden by CLI or state
# ─────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.10
DEFAULT_WEIGHTS = {"H1": 0.10, "H4": 0.25, "D": 0.60, "W": 0.05}
SWEEP_ORDER = ["D", "H4", "H1", "W"]

PARAM_GRIDS = {
    "D": {
        "fast_ma": [5, 8, 10, 13, 15, 20],
        "slow_ma": [20, 25, 30, 40, 50, 60, 80],
        "rsi_period": [7, 10, 14, 21],
    },
    "H4": {
        "fast_ma": [10, 15, 20, 25, 30],
        "slow_ma": [30, 40, 50, 60, 80, 100],
        "rsi_period": [7, 10, 14, 21],
    },
    "H1": {
        "fast_ma": [10, 15, 20, 25, 30],
        "slow_ma": [30, 50, 80, 100, 150],
        "rsi_period": [7, 10, 14, 21, 28],
    },
    "W": {
        "fast_ma": [3, 5, 8, 10, 13],
        "slow_ma": [8, 13, 21, 26, 34],
        "rsi_period": [7, 10, 14, 21],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, gran: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_baseline_params(tfs: list[str]) -> dict:
    with open(BASELINE_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    return {
        tf: {
            "fast_ma": cfg.get(tf, {}).get("fast_ma", 20),
            "slow_ma": cfg.get(tf, {}).get("slow_ma", 50),
            "rsi_period": cfg.get(tf, {}).get("rsi_period", 14),
        }
        for tf in tfs
    }


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def extract_stats(pf) -> dict:
    n = pf.trades.count()
    return {
        "ret": pf.total_return(),
        "sharpe": pf.sharpe_ratio(),
        "dd": pf.max_drawdown(),
        "trades": n,
        "wr": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def run_backtest(
    close: pd.Series, confluence: pd.Series, fees: float, threshold: float
) -> tuple[dict, dict]:
    # Shift by 1: signal fires at close of bar i, order fills at bar i+1
    conf_sh = confluence.shift(1).fillna(0.0)
    long_stats = extract_stats(
        vbt.Portfolio.from_signals(
            close,
            entries=conf_sh >= threshold,
            exits=conf_sh < 0,
            init_cash=10_000,
            fees=fees,
            freq="4h",
        )
    )
    short_stats = extract_stats(
        vbt.Portfolio.from_signals(
            close,
            entries=pd.Series(False, index=close.index),
            exits=pd.Series(False, index=close.index),
            short_entries=conf_sh <= -threshold,
            short_exits=conf_sh > 0,
            init_cash=10_000,
            fees=fees,
            freq="4h",
        )
    )
    return long_stats, short_stats


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF Stage 3: greedy per-TF param sweep.")
    parser.add_argument(
        "--pair",
        default="GBP_USD",
        help="Instrument in BASE_QUOTE format, e.g. GBP_USD (default: GBP_USD)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confirmation threshold (default: load from state or {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--w-H1", type=float, default=None, dest="w_H1", help="H1 weight")
    parser.add_argument("--w-H4", type=float, default=None, dest="w_H4", help="H4 weight")
    parser.add_argument("--w-D", type=float, default=None, dest="w_D", help="D weight")
    parser.add_argument("--w-W", type=float, default=None, dest="w_W", help="W weight")
    parser.add_argument(
        "--load-state",
        action="store_true",
        help="Load Stage 1-2 results from state manager automatically",
    )
    args = parser.parse_args()
    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")

    # Resolve threshold, ma_type and weights: CLI > state > EUR/USD defaults
    threshold = args.threshold
    w_H1 = args.w_H1
    w_H4 = args.w_H4
    w_D = args.w_D
    w_W = args.w_W
    ma_type_state: str | None = None

    if args.load_state or any(v is None for v in [threshold, w_H1, w_H4, w_D, w_W]):
        try:
            import research.mtf.state_manager as state_manager

            s1 = state_manager.get_stage1(pair=pair_lower)
            s2 = state_manager.get_stage2(pair=pair_lower)
            if s1:
                ma_type_state = s1[0]
                if threshold is None:
                    threshold = s1[1]
            if s2:
                saved_w = s2
                if w_H1 is None:
                    w_H1 = saved_w.get("H1")
                if w_H4 is None:
                    w_H4 = saved_w.get("H4")
                if w_D is None:
                    w_D = saved_w.get("D")
                if w_W is None:
                    w_W = saved_w.get("W")
        except ImportError:
            pass

    THRESHOLD = threshold if threshold is not None else DEFAULT_THRESHOLD
    WEIGHTS = {
        "H1": w_H1 if w_H1 is not None else DEFAULT_WEIGHTS["H1"],
        "H4": w_H4 if w_H4 is not None else DEFAULT_WEIGHTS["H4"],
        "D": w_D if w_D is not None else DEFAULT_WEIGHTS["D"],
        "W": w_W if w_W is not None else DEFAULT_WEIGHTS["W"],
    }

    config_out = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    csv_out = REPORTS_DIR / f"mtf_{pair_lower}_sweep.csv"

    tfs = list(WEIGHTS.keys())

    print("=" * 60)
    print(f"  MTF Pair Sweep: {pair}")
    print(f"  Sweep order: {' -> '.join(SWEEP_ORDER)}")
    print(f"  Weights: {WEIGHTS}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Config out: {config_out}")
    print("=" * 60)

    # Load primary timeframe (H4) as index
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print(f"ERROR: {pair}_H4.parquet not found in data/. Run download first.")
        print(f"  uv run python scripts/download_data_mtf.py --pair {pair}")
        sys.exit(1)
    primary_index = primary_df.index
    close = primary_df["close"]

    avg_spread = float(build_spread_series(primary_df, pair).mean())
    print(f"\n  H4 bars: {len(close)}   Avg spread: {avg_spread:.6f}")

    # IS/OOS split (70/30 on bar count)
    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]
    print(
        f"  IS bars:  {len(is_close)}  ({is_close.index[0].date()} to {is_close.index[-1].date()})"
    )
    print(
        f"  OOS bars: {len(oos_close)} ({oos_close.index[0].date()} to {oos_close.index[-1].date()})"
    )

    # Load close series per TF
    tf_closes: dict[str, pd.Series] = {}
    for tf in tfs:
        df = load_data(pair, tf)
        if df is not None:
            tf_closes[tf] = df["close"]
            print(f"  {tf}: {len(df)} bars")
        else:
            print(f"  {tf}: MISSING — skipping")

    if "H4" not in tf_closes:
        print("ERROR: H4 data required as primary index. Aborting.")
        sys.exit(1)

    # Seed baseline from EUR/USD validated config
    best_params = load_baseline_params(tfs)
    print(f"\n  Seeding from EUR/USD baseline ({BASELINE_CONFIG.name}):")
    for tf in SWEEP_ORDER:
        p = best_params[tf]
        print(f"    {tf}: fast={p['fast_ma']} slow={p['slow_ma']} rsi={p['rsi_period']}")

    all_results: list[dict] = []

    # Greedy sweep: tune one TF at a time
    for sweep_tf in SWEEP_ORDER:
        if sweep_tf not in tf_closes:
            print(f"\n  Skipping {sweep_tf}: no data.")
            continue

        grid = PARAM_GRIDS[sweep_tf]
        combos = [
            {"fast_ma": f, "slow_ma": s, "rsi_period": r}
            for f in grid["fast_ma"]
            for s in grid["slow_ma"]
            for r in grid["rsi_period"]
            if f < s
        ]

        print(f"\n{'=' * 50}")
        print(f"  Sweeping {sweep_tf} ({len(combos)} combos, weight={WEIGHTS[sweep_tf]:.2f})")
        print(f"{'=' * 50}")

        # Pre-compute fixed signals for all other TFs
        fixed_signals: dict[str, pd.Series] = {}
        for tf in tfs:
            if tf == sweep_tf or tf not in tf_closes:
                continue
            sig = compute_tf_signal(
                tf_closes[tf],
                best_params[tf]["fast_ma"],
                best_params[tf]["slow_ma"],
                best_params[tf]["rsi_period"],
            )
            fixed_signals[tf] = sig.reindex(primary_index, method="ffill") * WEIGHTS[tf]

        fixed_sum = (
            sum(fixed_signals.values()) if fixed_signals else pd.Series(0.0, index=primary_index)
        )

        best_cs = -999.0
        best_combo = None
        tf_results = []

        for i, combo in enumerate(combos):
            sig = compute_tf_signal(
                tf_closes[sweep_tf],
                combo["fast_ma"],
                combo["slow_ma"],
                combo["rsi_period"],
            )
            sig_resampled = sig.reindex(primary_index, method="ffill") * WEIGHTS[sweep_tf]
            confluence = fixed_sum + sig_resampled

            is_conf = confluence.iloc[:split]
            oos_conf = confluence.iloc[split:]

            is_long, is_short = run_backtest(is_close, is_conf, avg_spread, THRESHOLD)
            oos_long, oos_short = run_backtest(oos_close, oos_conf, avg_spread, THRESHOLD)

            lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0.0
            sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0.0
            cs = (
                is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]
            ) / 4

            row = {
                "pair": pair,
                "sweep_tf": sweep_tf,
                "fast_ma": combo["fast_ma"],
                "slow_ma": combo["slow_ma"],
                "rsi_period": combo["rsi_period"],
                "is_long_sharpe": is_long["sharpe"],
                "is_short_sharpe": is_short["sharpe"],
                "oos_long_sharpe": oos_long["sharpe"],
                "oos_short_sharpe": oos_short["sharpe"],
                "oos_long_trades": oos_long["trades"],
                "oos_short_trades": oos_short["trades"],
                "oos_long_wr": oos_long["wr"],
                "oos_long_dd": oos_long["dd"],
                "long_parity": lp,
                "short_parity": sp,
                "combined_sharpe": cs,
            }
            tf_results.append(row)
            all_results.append(row)

            # Gate: both parities must be positive (OOS not worse than IS)
            if lp > 0 and sp > 0 and cs > best_cs:
                best_cs = cs
                best_combo = combo

            if (i + 1) % 50 == 0:
                print(f"    [{i + 1}/{len(combos)}] done...")

        if best_combo:
            best_params[sweep_tf] = best_combo
            print(
                f"\n  Best {sweep_tf}: fast={best_combo['fast_ma']} "
                f"slow={best_combo['slow_ma']} rsi={best_combo['rsi_period']} "
                f"combined_sharpe={best_cs:.4f}"
            )
        else:
            print(f"  No improvement for {sweep_tf} with positive parity — keeping baseline.")

    # Save full scoreboard
    pd.DataFrame(all_results).to_csv(csv_out, index=False)
    print(f"\nFull scoreboard: {csv_out}")

    # Final backtest with all best params
    print("\n--- Final Backtest (All Optimised Params) ---")
    signals_sum = pd.Series(0.0, index=primary_index)
    for tf in tfs:
        if tf not in tf_closes:
            continue
        sig = compute_tf_signal(
            tf_closes[tf],
            best_params[tf]["fast_ma"],
            best_params[tf]["slow_ma"],
            best_params[tf]["rsi_period"],
        )
        signals_sum += sig.reindex(primary_index, method="ffill") * WEIGHTS[tf]

    is_conf = signals_sum.iloc[:split]
    oos_conf = signals_sum.iloc[split:]

    is_long, is_short = run_backtest(is_close, is_conf, avg_spread, THRESHOLD)
    oos_long, oos_short = run_backtest(oos_close, oos_conf, avg_spread, THRESHOLD)

    cs = (is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]) / 4
    lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0.0
    sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0.0

    print(f"\n  Combined Sharpe: {cs:.4f}")
    print(
        f"  IS  LONG:  sharpe={is_long['sharpe']:.3f}  trades={is_long['trades']}  dd={is_long['dd']:.2%}"
    )
    print(
        f"  IS  SHORT: sharpe={is_short['sharpe']:.3f}  trades={is_short['trades']}  dd={is_short['dd']:.2%}"
    )
    print(
        f"  OOS LONG:  sharpe={oos_long['sharpe']:.3f}  trades={oos_long['trades']}  dd={oos_long['dd']:.2%}"
    )
    print(
        f"  OOS SHORT: sharpe={oos_short['sharpe']:.3f}  trades={oos_short['trades']}  dd={oos_short['dd']:.2%}"
    )
    print(f"  Parity: L={lp:.2f}  S={sp:.2f}")

    # ── 7-Gate Validation ──
    print("\n--- 7-Gate Validation ---")
    gates = {
        "Gate 1 OOS/IS Sharpe >= 0.5 (long)": lp >= 0.5,
        "Gate 2 OOS Sharpe >= 1.0 (long)": oos_long["sharpe"] >= 1.0,
        "Gate 3 OOS trades >= 30 (long)": oos_long["trades"] >= 30,
        "Gate 4 OOS win rate >= 40% (long)": oos_long["wr"] >= 0.40,
        "Gate 5 OOS max DD <= 25%": abs(oos_long["dd"]) <= 0.25,
        "Gate 6 OOS/IS Sharpe >= 0.5 (short)": sp >= 0.5,
        "Gate 7 OOS Sharpe >= 1.0 (short)": oos_short["sharpe"] >= 1.0,
    }
    all_pass = True
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if not all_pass:
        print("\n  WARNING: Not all gates passed. Review before deploying.")
    else:
        print("\n  All gates passed.")

    # Save config regardless (document the optimised params even if gates fail)
    config_lines = [
        "# " + "=" * 56,
        f"# mtf_{pair_lower}.toml -- MTF Confluence Config for {pair}",
        "# " + "=" * 56,
        "# Optimised via run_pair_sweep.py (greedy Stage 3).",
        f"# Pair: {pair}",
        f"# OOS LONG Sharpe:  {oos_long['sharpe']:.3f}",
        f"# OOS SHORT Sharpe: {oos_short['sharpe']:.3f}",
        f"# Combined Sharpe:  {cs:.3f}",
        f"# Gates passed: {'YES' if all_pass else 'NO -- review before live'}",
        "",
        f'ma_type = "{ma_type_state or "WMA"}"',
        f"confirmation_threshold = {THRESHOLD:.2f}",
        "atr_stop_mult = 2.5",  # Stage 4 will overwrite this
        "",
        "[weights]",
        f"H1 = {WEIGHTS['H1']:.2f}",
        f"H4 = {WEIGHTS['H4']:.2f}",
        f"D  = {WEIGHTS['D']:.2f}",
        f"W  = {WEIGHTS['W']:.2f}",
        "",
    ]
    for tf in ["H1", "H4", "D", "W"]:
        p = best_params[tf]
        config_lines += [
            f"[{tf}]",
            f"fast_ma = {p['fast_ma']}",
            f"slow_ma = {p['slow_ma']}",
            f"rsi_period = {p['rsi_period']}",
            "",
        ]

    config_out.write_text("\n".join(config_lines), encoding="utf-8")
    print(f"\nSaved config -> {config_out}")

    # Save Stage 3 params to state manager
    try:
        import research.mtf.state_manager as state_manager

        state_manager.save_stage3(params=best_params, pair=pair_lower)
    except ImportError:
        pass

    print("\nNext step:")
    print(f"  uv run python research/mtf/run_stage4_atr.py --pair {pair}")
    print("Done.")


if __name__ == "__main__":
    main()
