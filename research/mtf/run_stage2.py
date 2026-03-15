"""run_mtf_stage2.py — Stage 2: Timeframe Weight Sweep.

Sweeps H1/H4/D/W weight combinations while keeping MA type and threshold
fixed from Stage 1.  Uses IS/OOS validation with parity check.

Usage:
    uv run python research/mtf/run_stage2.py                          # EUR/USD defaults
    uv run python research/mtf/run_stage2.py --pair GBP_USD --load-state
    uv run python research/mtf/run_stage2.py --pair GBP_USD --ma-type SMA --threshold 0.10
"""

import argparse
import sys
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt  # noqa: E402
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Weight grid — step 0.05, all 4 must sum to 1.0
# Each weight in [0.05, 0.60]
# ─────────────────────────────────────────────────────────────────────
STEP = 0.05
MIN_W, MAX_W = 0.05, 0.60


def generate_weight_combos() -> list[dict]:
    """Generate all (H1, H4, D, W) weight combos that sum to 1.0."""
    vals = [round(x, 2) for x in np.arange(MIN_W, MAX_W + STEP / 2, STEP)]
    combos = []
    for h1, h4, d, w in product(vals, repeat=4):
        if abs(h1 + h4 + d + w - 1.0) < 0.001:
            combos.append({"H1": h1, "H4": h4, "D": d, "W": w})
    return combos


# ─────────────────────────────────────────────────────────────────────
# Data & signal helpers (same as Stage 1, but with fixed MA type)
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, gran: str) -> pd.DataFrame | None:
    """Load parquet data."""
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_mtf_config(config_path: Path) -> dict:
    """Load a TOML config file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    """Directional signal for one TF using SMA."""
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def extract_stats(pf) -> dict:
    """Pull metrics from a VBT Portfolio."""
    n = pf.trades.count()
    return {
        "ret": pf.total_return(),
        "sharpe": pf.sharpe_ratio(),
        "dd": pf.max_drawdown(),
        "trades": n,
        "wr": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Sweep weight combos."""
    parser = argparse.ArgumentParser(description="MTF Stage 2: Timeframe Weight Sweep.")
    parser.add_argument("--pair", default="EUR_USD", help="Instrument (default: EUR_USD)")
    parser.add_argument(
        "--ma-type",
        default=None,
        help="MA type from Stage 1 (default: load from state or SMA)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold from Stage 1 (default: load from state or 0.10)",
    )
    parser.add_argument(
        "--load-state",
        action="store_true",
        help="Load Stage 1 results from state manager automatically",
    )
    args = parser.parse_args()
    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")

    # Resolve MA type and threshold: CLI args > state file > defaults
    ma_type = args.ma_type
    threshold = args.threshold
    if args.load_state or ma_type is None or threshold is None:
        try:
            import research.mtf.state_manager as state_manager

            s1 = state_manager.get_stage1(pair=pair_lower)
            if s1:
                if ma_type is None:
                    ma_type = s1[0]
                if threshold is None:
                    threshold = s1[1]
        except ImportError:
            pass
    ma_type = ma_type or "SMA"
    threshold = threshold if threshold is not None else 0.10

    # Load config: prefer pair-specific, fall back to base EUR/USD config
    pair_config = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    base_config = PROJECT_ROOT / "config" / "mtf.toml"
    config_path = pair_config if pair_config.exists() else base_config
    mtf_config = load_mtf_config(config_path)
    tfs = ["H1", "H4", "D", "W"]

    scoreboard_path = REPORTS_DIR / f"mtf_stage2_{pair_lower}_scoreboard.csv"
    heatmap_path = REPORTS_DIR / f"mtf_stage2_{pair_lower}_heatmap.html"

    print("=" * 60)
    print("  Stage 2: Timeframe Weight Sweep")
    print("=" * 60)
    print(f"  Pair:       {pair}")
    print(f"  MA Type:    {ma_type}")
    print(f"  Threshold:  {threshold}")

    # Load H4 as primary
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print("ERROR: H4 data required.")
        sys.exit(1)
    primary_index = primary_df.index
    close = primary_df["close"]

    # Spread
    spread_series = build_spread_series(primary_df, pair)
    avg_spread = float(spread_series.mean())

    # IS/OOS split
    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]

    # Pre-compute resampled signals for each TF (uses MA type from Stage 1)
    tf_signals: dict[str, pd.Series] = {}
    for tf in tfs:
        tf_config = mtf_config.get(tf, {})
        df = load_data(pair, tf)
        if df is None:
            continue
        fast = tf_config.get("fast_ma", 20)
        slow = tf_config.get("slow_ma", 50)
        rsi_p = tf_config.get("rsi_period", 14)
        # Honour the MA type from Stage 1 (not just SMA)
        if ma_type == "EMA":
            fast_s = df["close"].ewm(span=fast, adjust=False).mean()
            slow_s = df["close"].ewm(span=slow, adjust=False).mean()
        elif ma_type == "WMA":
            import numpy as np

            wf = np.arange(1, fast + 1, dtype=float)
            ws = np.arange(1, slow + 1, dtype=float)
            fast_s = (
                df["close"].rolling(fast).apply(lambda x: float(np.dot(x, wf) / wf.sum()), raw=True)
            )
            slow_s = (
                df["close"].rolling(slow).apply(lambda x: float(np.dot(x, ws) / ws.sum()), raw=True)
            )
        else:  # SMA
            fast_s = df["close"].rolling(fast).mean()
            slow_s = df["close"].rolling(slow).mean()
        rsi = compute_rsi(df["close"], rsi_p)
        import numpy as np

        ma_sig = pd.Series(np.where(fast_s > slow_s, 0.5, -0.5), index=df.index)
        rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=df.index)
        signal = ma_sig + rsi_sig
        tf_signals[tf] = signal.reindex(primary_index, method="ffill")
        bullish = (signal > 0).mean() * 100
        print(f"  {tf:3s}: {len(df)} bars, {bullish:.0f}% bullish")

    weight_combos = generate_weight_combos()
    print(f"\n  Weight combos: {len(weight_combos)}")
    print(f"  Weight range:  [{MIN_W}, {MAX_W}] step {STEP}")

    results: list[dict] = []

    for i, weights in enumerate(weight_combos):
        # Build confluence from pre-computed signals
        confluence = sum(tf_signals[tf] * weights[tf] for tf in tfs if tf in tf_signals)

        is_conf = confluence.iloc[:split]
        oos_conf = confluence.iloc[split:]

        # Shift by 1: signal fires at close of bar i, order fills at bar i+1
        is_conf_sh = is_conf.shift(1).fillna(0.0)
        oos_conf_sh = oos_conf.shift(1).fillna(0.0)

        # IS backtest
        is_long = extract_stats(
            vbt.Portfolio.from_signals(
                is_close,
                entries=is_conf_sh >= threshold,
                exits=is_conf_sh < 0,
                init_cash=10_000,
                fees=avg_spread,
                freq="4h",
            )
        )
        is_short = extract_stats(
            vbt.Portfolio.from_signals(
                is_close,
                entries=pd.Series(False, index=is_close.index),
                exits=pd.Series(False, index=is_close.index),
                short_entries=is_conf_sh <= -threshold,
                short_exits=is_conf_sh > 0,
                init_cash=10_000,
                fees=avg_spread,
                freq="4h",
            )
        )

        # OOS backtest
        oos_long = extract_stats(
            vbt.Portfolio.from_signals(
                oos_close,
                entries=oos_conf_sh >= threshold,
                exits=oos_conf_sh < 0,
                init_cash=10_000,
                fees=avg_spread,
                freq="4h",
            )
        )
        oos_short = extract_stats(
            vbt.Portfolio.from_signals(
                oos_close,
                entries=pd.Series(False, index=oos_close.index),
                exits=pd.Series(False, index=oos_close.index),
                short_entries=oos_conf_sh <= -threshold,
                short_exits=oos_conf_sh > 0,
                init_cash=10_000,
                fees=avg_spread,
                freq="4h",
            )
        )

        # Parity
        lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0
        sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0
        cs = (is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]) / 4

        results.append(
            {
                "w_H1": weights["H1"],
                "w_H4": weights["H4"],
                "w_D": weights["D"],
                "w_W": weights["W"],
                "is_long_ret": is_long["ret"],
                "is_long_sharpe": is_long["sharpe"],
                "is_long_dd": is_long["dd"],
                "is_long_trades": is_long["trades"],
                "is_short_ret": is_short["ret"],
                "is_short_sharpe": is_short["sharpe"],
                "is_short_dd": is_short["dd"],
                "is_short_trades": is_short["trades"],
                "oos_long_ret": oos_long["ret"],
                "oos_long_sharpe": oos_long["sharpe"],
                "oos_long_dd": oos_long["dd"],
                "oos_long_trades": oos_long["trades"],
                "oos_short_ret": oos_short["ret"],
                "oos_short_sharpe": oos_short["sharpe"],
                "oos_short_dd": oos_short["dd"],
                "oos_short_trades": oos_short["trades"],
                "long_parity": lp,
                "short_parity": sp,
                "combined_sharpe": cs,
            }
        )

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(weight_combos)}] combos done...")

    # ── Results ──
    df = pd.DataFrame(results)
    df.to_csv(scoreboard_path, index=False)
    print(f"\nFull scoreboard: {scoreboard_path}")

    # Filter: both parities > 0
    valid = df[(df.long_parity > 0) & (df.short_parity > 0)]
    if valid.empty:
        print("WARNING: No combos pass parity.")
        valid = df

    best = valid.loc[valid.combined_sharpe.idxmax()]

    print("\n" + "=" * 60)
    print("  BEST WEIGHTS (Stage 2)")
    print("=" * 60)
    print(f"  H1={best.w_H1:.2f}  H4={best.w_H4:.2f}  D={best.w_D:.2f}  W={best.w_W:.2f}")
    print(f"  Combined Sharpe: {best.combined_sharpe:.4f}")
    print()
    print(
        f"  IS  LONG:   ret={best.is_long_ret:.2%}  "
        f"sharpe={best.is_long_sharpe:.3f}  "
        f"dd={best.is_long_dd:.2%}  "
        f"trades={int(best.is_long_trades)}"
    )
    print(
        f"  IS  SHORT:  ret={best.is_short_ret:.2%}  "
        f"sharpe={best.is_short_sharpe:.3f}  "
        f"dd={best.is_short_dd:.2%}  "
        f"trades={int(best.is_short_trades)}"
    )
    print(
        f"  OOS LONG:   ret={best.oos_long_ret:.2%}  "
        f"sharpe={best.oos_long_sharpe:.3f}  "
        f"dd={best.oos_long_dd:.2%}  "
        f"trades={int(best.oos_long_trades)}"
    )
    print(
        f"  OOS SHORT:  ret={best.oos_short_ret:.2%}  "
        f"sharpe={best.oos_short_sharpe:.3f}  "
        f"dd={best.oos_short_dd:.2%}  "
        f"trades={int(best.oos_short_trades)}"
    )
    print(f"  Parity: L={best.long_parity:.2f}  S={best.short_parity:.2f}")

    # Compare to current weights
    curr = mtf_config.get("weights", {})
    print(
        f"\n  Current weights: "
        f"H1={curr.get('H1', 0):.2f}  "
        f"H4={curr.get('H4', 0):.2f}  "
        f"D={curr.get('D', 0):.2f}  "
        f"W={curr.get('W', 0):.2f}"
    )

    # Top 10
    print("\n--- TOP 10 ---")
    top = valid.nlargest(10, "combined_sharpe")
    for _, r in top.iterrows():
        print(
            f"  H1={r.w_H1:.2f} H4={r.w_H4:.2f} D={r.w_D:.2f} W={r.w_W:.2f}  "
            f"comb_sharpe={r.combined_sharpe:.4f}  "
            f"IS_Lret={r.is_long_ret:.2%}  OOS_Lret={r.oos_long_ret:.2%}  "
            f"IS_Sret={r.is_short_ret:.2%}  OOS_Sret={r.oos_short_ret:.2%}"
        )

    # ── Save to State Manager ──
    try:
        import research.mtf.state_manager as state_manager

        state_manager.save_stage2(
            weights={
                "H1": float(best.w_H1),
                "H4": float(best.w_H4),
                "D": float(best.w_D),
                "W": float(best.w_W),
            },
            pair=pair_lower,
        )
    except ImportError:
        print("WARNING: Could not import state_manager. State not saved.")

    # Heatmap (H4 vs D, aggregated across H1/W)
    try:
        import plotly.express as px

        agg = valid.groupby(["w_H4", "w_D"])["combined_sharpe"].max().reset_index()
        pivot = agg.pivot(index="w_D", columns="w_H4", values="combined_sharpe")
        fig = px.imshow(
            pivot,
            labels=dict(x="H4 Weight", y="D Weight", color="Combined Sharpe"),
            title=f"Stage 2: Combined Sharpe — H4 vs D Weight ({pair})",
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig.write_html(str(heatmap_path))
        print(f"\nHeatmap: {heatmap_path}")
    except ImportError:
        print("\nPlotly not installed — skipping heatmap.")

    print("\nNext step:")
    print(f"  uv run python research/mtf/run_pair_sweep.py --pair {pair} --load-state")
    print("\nDone.")


if __name__ == "__main__":
    main()
