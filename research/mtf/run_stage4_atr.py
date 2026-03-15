"""run_stage4_atr.py — Stage 4: ATR Stop Multiplier Sweep.

Sweeps atr_stop_mult over [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] on the OOS window
using the Stage 3 optimised parameters from config/mtf_{pair}.toml.

Selects the multiplier with the best OOS combined Sharpe (long + short) subject
to parity check (OOS Sharpe >= 0.5 * IS Sharpe).

Updates atr_stop_mult in config/mtf_{pair}.toml in-place and saves the scoreboard
to .tmp/reports/mtf_{pair}_stage4_atr.csv.

Usage:
    uv run python research/mtf/run_stage4_atr.py --pair GBP_USD
    uv run python research/mtf/run_stage4_atr.py --pair EUR_USD
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

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv sync")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

ATR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
ATR_PERIOD = 14


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


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


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


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_confluence(pair: str, cfg: dict, primary_index: pd.DatetimeIndex) -> pd.Series:
    """Compute weighted MTF confluence score aligned to H4 index."""
    weights = cfg.get("weights", {})
    tfs = ["H1", "H4", "D", "W"]
    signals_sum = pd.Series(0.0, index=primary_index)
    for tf in tfs:
        w = weights.get(tf, 0.0)
        if w == 0:
            continue
        df = load_data(pair, tf)
        if df is None:
            print(f"  WARNING: {pair}_{tf}.parquet missing — skipping.")
            continue
        tf_cfg = cfg.get(tf, {})
        sig = compute_tf_signal(
            df["close"],
            tf_cfg.get("fast_ma", 20),
            tf_cfg.get("slow_ma", 50),
            tf_cfg.get("rsi_period", 14),
        )
        signals_sum += sig.reindex(primary_index, method="ffill") * w
    return signals_sum


def extract_stats(pf) -> dict:
    n = pf.trades.count()
    return {
        "sharpe": pf.sharpe_ratio(),
        "dd": pf.max_drawdown(),
        "trades": n,
        "wr": float(pf.trades.win_rate()) if n > 0 else 0.0,
        "ret": pf.total_return(),
    }


def run_vbt(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    confluence: pd.Series,
    threshold: float,
    atr: pd.Series,
    atr_mult: float,
    fees: float,
) -> tuple[dict, dict]:
    """Run long and short VBT portfolios with ATR trailing stop.

    Signals are shifted by 1 bar: signal at bar i close executes at bar i+1.
    """
    sl_stop = (atr_mult * atr / close).clip(lower=0.001).fillna(0.001)
    # Shift by 1: signal fires at close of bar i, order fills at bar i+1
    conf_sh = confluence.shift(1).fillna(0.0)

    long_pf = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=conf_sh >= threshold,
        exits=conf_sh < 0,
        init_cash=10_000,
        fees=fees,
        freq="4h",
        sl_stop=sl_stop,
        sl_trail=True,
    )
    short_pf = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=conf_sh <= -threshold,
        short_exits=conf_sh > 0,
        init_cash=10_000,
        fees=fees,
        freq="4h",
        sl_stop=sl_stop,
        sl_trail=True,
    )
    return extract_stats(long_pf), extract_stats(short_pf)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF Stage 4: ATR Stop Multiplier Sweep.")
    parser.add_argument("--pair", default="GBP_USD", help="Instrument (default: GBP_USD)")
    args = parser.parse_args()
    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")

    pair_cfg_path = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    base_cfg_path = PROJECT_ROOT / "config" / "mtf.toml"
    config_path = pair_cfg_path if pair_cfg_path.exists() else base_cfg_path

    print("=" * 60)
    print(f"  MTF Stage 4: ATR Stop Multiplier Sweep — {pair}")
    print("=" * 60)
    print(f"  Config: {config_path.name}")
    print(f"  ATR period: {ATR_PERIOD}")
    print(f"  Multipliers: {ATR_MULTS}")

    cfg = load_config(config_path)
    threshold = cfg.get("confirmation_threshold", 0.10)
    print(f"  Threshold: {threshold}")

    # Load H4 primary data
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print(f"ERROR: {pair}_H4.parquet not found. Run download first.")
        sys.exit(1)

    primary_index = primary_df.index
    close = primary_df["close"]
    high = primary_df["high"]
    low = primary_df["low"]

    avg_spread = float(build_spread_series(primary_df, pair).mean())
    atr = compute_atr(primary_df, ATR_PERIOD)

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    is_high = high.iloc[:split]
    is_low = low.iloc[:split]
    is_atr = atr.iloc[:split]
    oos_close = close.iloc[split:]
    oos_high = high.iloc[split:]
    oos_low = low.iloc[split:]
    oos_atr = atr.iloc[split:]

    print(
        f"\n  IS:  {is_close.index[0].date()} to {is_close.index[-1].date()} ({len(is_close)} bars)"
    )
    print(
        f"  OOS: {oos_close.index[0].date()} to {oos_close.index[-1].date()} ({len(oos_close)} bars)"
    )

    # Build confluence once
    confluence = build_confluence(pair, cfg, primary_index)
    is_conf = confluence.iloc[:split]
    oos_conf = confluence.iloc[split:]

    results = []
    best_cs = -999.0
    best_mult = 2.5  # safe default

    print(
        f"\n  {'Mult':>5}  {'IS_L':>7} {'IS_S':>7} {'OOS_L':>7} {'OOS_S':>7} {'CS':>7} {'Parity_L':>9} {'Parity_S':>9}"
    )
    print("  " + "-" * 70)

    for mult in ATR_MULTS:
        is_long, is_short = run_vbt(
            is_close, is_high, is_low, is_conf, threshold, is_atr, mult, avg_spread
        )
        oos_long, oos_short = run_vbt(
            oos_close, oos_high, oos_low, oos_conf, threshold, oos_atr, mult, avg_spread
        )

        lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0.0
        sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0.0
        cs = (is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]) / 4

        print(
            f"  {mult:>5.1f}  {is_long['sharpe']:>7.3f} {is_short['sharpe']:>7.3f}"
            f" {oos_long['sharpe']:>7.3f} {oos_short['sharpe']:>7.3f}"
            f" {cs:>7.3f} {lp:>9.2f} {sp:>9.2f}"
        )

        results.append(
            {
                "atr_mult": mult,
                "is_long_sharpe": is_long["sharpe"],
                "is_short_sharpe": is_short["sharpe"],
                "oos_long_sharpe": oos_long["sharpe"],
                "oos_short_sharpe": oos_short["sharpe"],
                "oos_long_trades": oos_long["trades"],
                "oos_short_trades": oos_short["trades"],
                "oos_long_dd": oos_long["dd"],
                "oos_long_wr": oos_long["wr"],
                "long_parity": lp,
                "short_parity": sp,
                "combined_sharpe": cs,
            }
        )

        if lp > 0 and sp > 0 and cs > best_cs:
            best_cs = cs
            best_mult = mult

    # Save scoreboard
    scoreboard_path = REPORTS_DIR / f"mtf_{pair_lower}_stage4_atr.csv"
    pd.DataFrame(results).to_csv(scoreboard_path, index=False)
    print(f"\nFull scoreboard: {scoreboard_path}")

    print(f"\n  Best ATR multiplier: {best_mult}x  (combined Sharpe: {best_cs:.4f})")

    # Update the config TOML in-place
    raw = config_path.read_text(encoding="utf-8")
    updated_lines = []
    replaced = False
    for line in raw.splitlines():
        if line.strip().startswith("atr_stop_mult"):
            updated_lines.append(f"atr_stop_mult = {best_mult}")
            replaced = True
        else:
            updated_lines.append(line)
    if not replaced:
        # append before first [section] or at end
        updated_lines.append(f"atr_stop_mult = {best_mult}")
    config_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"  Updated atr_stop_mult = {best_mult} in {config_path.name}")

    # Save to state manager
    try:
        import research.mtf.state_manager as state_manager

        state_manager.save_stage4(atr_mult=best_mult, pair=pair_lower)
    except ImportError:
        pass

    print("\nNext step:")
    print(f"  uv run python research/mtf/run_portfolio.py --pair {pair}")
    print("Done.")


if __name__ == "__main__":
    main()
