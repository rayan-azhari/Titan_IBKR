"""run_stage4_sizing.py — Stage 4: Volatility Sizing + ATR Stop Sweep.

Sweeps two sizing modes head-to-head:
  binary        — full vol-target size or flat (decel used only as exit filter)
  dynamic_decel — position scales continuously with decel composite

Also sweeps:
  vol_target    — annualised volatility contribution per position
  vol_window    — realised vol lookback window
  atr_stop_mult — catastrophic hard stop multiplier (backup insurance only)

Writes config/etf_trend_{instrument_lower}.toml on completion.
Auto-loads Stages 1-3 via --load-state.

Usage:
    uv run python research/etf_trend/run_stage4_sizing.py --instrument SPY --load-state
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = PROJECT_ROOT / "config"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from research.etf_trend.run_stage2_decel import (  # noqa: E402
    compute_ma,
)
from research.etf_trend.run_stage3_exits import compute_decel_composite  # noqa: E402
from research.etf_trend.state_manager import (  # noqa: E402
    get_stage1,
    get_stage2,
    get_stage3,
    save_stage4,
)
from titan.strategies.ml.features import atr as compute_atr  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005

SIZING_MODES = ["binary", "dynamic_decel"]
VOL_TARGETS = [0.08, 0.10, 0.12, 0.15]
VOL_WINDOWS = [10, 20, 30]
ATR_STOP_MULTS = [3.0, 4.0, 5.0]

INIT_CASH = 10_000.0


def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
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


def realised_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility.

    Args:
        close: Close price series.
        window: Rolling window in days.

    Returns:
        Annualised vol series.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_position_sizes(
    close: pd.Series,
    decel: pd.Series,
    sizing_mode: str,
    vol_target: float,
    vol_window: int,
    atr_ser: pd.Series,
    atr_stop_mult: float,
) -> pd.Series:
    """Compute target position fraction (0 to 1) based on sizing mode.

    For binary: 1.0 if in_regime, 0.0 otherwise.
    For dynamic_decel: scale by clip((decel+1)/2, 0, 1) × vol_target/rv.
    ATR stop provides an independent size cap: max_size = risk% / (ATR×mult / price).

    Args:
        close: Close price series.
        decel: Composite decel signal [-1, 1].
        sizing_mode: 'binary' or 'dynamic_decel'.
        vol_target: Target annualised vol contribution (e.g. 0.10).
        vol_window: Realised vol lookback.
        atr_ser: ATR series.
        atr_stop_mult: ATR hard stop multiplier.

    Returns:
        Position fraction series [0, 1].
    """
    rv = realised_vol(close, vol_window).replace(0, np.nan).ffill()
    vol_scalar = (vol_target / rv).clip(0, 2)  # cap at 2× leverage

    if sizing_mode == "binary":
        size = vol_scalar
    else:  # dynamic_decel
        decel_scalar = ((decel + 1) / 2).clip(0, 1)
        size = vol_scalar * decel_scalar

    # ATR cap: catastrophic loss insurance
    atr_pct = (atr_ser * atr_stop_mult / close).replace(0, np.nan).ffill()
    atr_cap = 0.01 / atr_pct  # 1% equity risk / (ATR×mult / price)
    size = size.clip(0, atr_cap.clip(upper=2))

    return size.clip(0, 1)


def run_sized_backtest(
    close: pd.Series,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_mode: str,
    exit_decel_thresh: float,
    sizing_mode: str,
    vol_target: float,
    vol_window: int,
    atr_ser: pd.Series,
    atr_stop_mult: float,
) -> "vbt.Portfolio":
    """Run backtest with volatility-scaled position sizing.

    Uses size_type='targetpercent' to replicate fractional position sizing.

    Args:
        close: Close series.
        fast_ma: Fast MA.
        slow_ma: Slow MA.
        decel: Decel composite (already shifted).
        exit_mode: 'A', 'B', 'C', or 'D'.
        exit_decel_thresh: Threshold for C/D exit.
        sizing_mode: 'binary' or 'dynamic_decel'.
        vol_target: Vol contribution target.
        vol_window: Lookback window.
        atr_ser: ATR series.
        atr_stop_mult: ATR multiplier.

    Returns:
        VBT Portfolio object.
    """
    in_regime = (close > fast_ma) & (close > slow_ma)
    entry_signal = in_regime & (decel >= 0)

    if exit_mode == "A":
        exit_signal = ~in_regime
    elif exit_mode == "B":
        exit_signal = close < fast_ma
    elif exit_mode == "C":
        exit_signal = decel < exit_decel_thresh
    else:
        exit_signal = (~in_regime) | (decel < exit_decel_thresh)

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    sizes = (
        compute_position_sizes(
            close, decel, sizing_mode, vol_target, vol_window, atr_ser, atr_stop_mult
        )
        .shift(1)
        .fillna(0)
    )

    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="targetpercent",
        init_cash=INIT_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def extract_stats(pf: "vbt.Portfolio") -> dict:
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "calmar": float(pf.calmar_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "total_return": float(pf.total_return()),
        "n_trades": int(n),
        "win_rate": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def write_config(instrument: str, params: dict) -> Path:
    """Write locked parameters to config/etf_trend_{instrument_lower}.toml.

    Args:
        instrument: Instrument symbol (lowercase).
        params: Dict of all locked parameters.

    Returns:
        Path to the written config file.
    """
    lines = [
        f"# ETF Trend Strategy — {instrument.upper()} locked config",
        "# Generated by run_stage4_sizing.py",
        "",
        f'instrument        = "{params["instrument_id"]}"',
        f'ma_type           = "{params["ma_type"]}"',
        f"fast_ma           = {params['fast_ma']}",
        f"slow_ma           = {params['slow_ma']}",
        f"decel_signals     = {params['decel_signals']}",
        f'exit_mode         = "{params["exit_mode"]}"',
        f"exit_decel_thresh = {params['exit_decel_thresh']}",
        f'sizing_mode       = "{params["sizing_mode"]}"',
        f"vol_target        = {params['vol_target']}",
        f"vol_window        = {params['vol_window']}",
        f"atr_stop_mult     = {params['atr_stop_mult']}",
        'eod_eval_time     = "20:30"  # 15:30 ET in UTC',
        "warmup_bars       = 300",
    ]
    config_path = CONFIG_DIR / f"etf_trend_{instrument.lower()}.toml"
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 4: Sizing Mode + ATR Sweep.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument("--load-state", action="store_true", help="Load Stages 1-3 from state")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend — Stage 4: Volatility Sizing + ATR Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    # ── Load previous stages ──────────────────────────────────────────────
    if args.load_state:
        s1 = get_stage1(inst_lower)
        s2 = get_stage2(inst_lower)
        s3 = get_stage3(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found.")
            sys.exit(1)
        ma_type, fast_ma_p, slow_ma_p = s1
        decel_signals, decel_weights = s2 if s2 else ([], {})
        exit_mode, exit_decel_thresh = s3 if s3 else ("A", -0.3)
    else:
        ma_type, fast_ma_p, slow_ma_p = "SMA", 50, 200
        decel_signals, _decel_weights = [], {}
        exit_mode, exit_decel_thresh = "A", -0.3

    print(f"  MA: {ma_type} fast={fast_ma_p} slow={slow_ma_p}")
    print(f"  Decel: {decel_signals or '(none)'}")
    print(f"  Exit mode: {exit_mode} thresh={exit_decel_thresh}")
    total = len(SIZING_MODES) * len(VOL_TARGETS) * len(VOL_WINDOWS) * len(ATR_STOP_MULTS)
    print(f"  Total combos: {total}")

    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]
    _is_df, _oos_df = df.iloc[:split], df.iloc[split:]

    fast_full = compute_ma(close, fast_ma_p, ma_type)
    slow_full = compute_ma(close, slow_ma_p, ma_type)
    is_fast, is_slow = fast_full.iloc[:split], slow_full.iloc[:split]
    oos_fast, oos_slow = fast_full.iloc[split:], slow_full.iloc[split:]

    atr_full = compute_atr(df)
    is_atr, oos_atr = atr_full.iloc[:split], atr_full.iloc[split:]

    decel_full = compute_decel_composite(close, df, slow_full, decel_signals)
    is_decel = decel_full.iloc[:split].shift(1).fillna(0)
    oos_decel = decel_full.iloc[split:].shift(1).fillna(0)

    results: list[dict] = []
    print("\n  Running sweep ...")

    for sizing_mode in SIZING_MODES:
        for vol_target in VOL_TARGETS:
            for vol_window in VOL_WINDOWS:
                for atr_mult in ATR_STOP_MULTS:
                    is_pf = run_sized_backtest(
                        is_close,
                        is_fast,
                        is_slow,
                        is_decel,
                        exit_mode,
                        exit_decel_thresh,
                        sizing_mode,
                        vol_target,
                        vol_window,
                        is_atr,
                        atr_mult,
                    )
                    oos_pf = run_sized_backtest(
                        oos_close,
                        oos_fast,
                        oos_slow,
                        oos_decel,
                        exit_mode,
                        exit_decel_thresh,
                        sizing_mode,
                        vol_target,
                        vol_window,
                        oos_atr,
                        atr_mult,
                    )
                    is_stats = extract_stats(is_pf)
                    oos_stats = extract_stats(oos_pf)
                    ratio = (
                        oos_stats["sharpe"] / is_stats["sharpe"]
                        if is_stats["sharpe"] > 0.01
                        else 0.0
                    )
                    results.append(
                        {
                            "sizing_mode": sizing_mode,
                            "vol_target": vol_target,
                            "vol_window": vol_window,
                            "atr_stop_mult": atr_mult,
                            "is_sharpe": round(is_stats["sharpe"], 3),
                            "oos_sharpe": round(oos_stats["sharpe"], 3),
                            "oos_is_ratio": round(ratio, 3),
                            "oos_calmar": round(oos_stats["calmar"], 3),
                            "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                            "oos_n_trades": oos_stats["n_trades"],
                        }
                    )

    scoreboard = pd.DataFrame(results).sort_values("oos_calmar", ascending=False)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage4_{inst_lower}_sizing.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved → {scoreboard_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]
    print("\n  ── Best sizing configuration ──────────────────────")
    for col in [
        "sizing_mode",
        "vol_target",
        "vol_window",
        "atr_stop_mult",
        "is_sharpe",
        "oos_sharpe",
        "oos_calmar",
        "oos_max_dd",
    ]:
        print(f"  {col:20s}: {best[col]}")

    # ── Write config ──────────────────────────────────────────────────────
    params = {
        "instrument_id": f"{instrument}.ARCA",
        "ma_type": ma_type,
        "fast_ma": fast_ma_p,
        "slow_ma": slow_ma_p,
        "decel_signals": decel_signals,
        "exit_mode": exit_mode,
        "exit_decel_thresh": exit_decel_thresh,
        "sizing_mode": str(best["sizing_mode"]),
        "vol_target": float(best["vol_target"]),
        "vol_window": int(best["vol_window"]),
        "atr_stop_mult": float(best["atr_stop_mult"]),
    }
    config_path = write_config(inst_lower, params)
    print(f"\n  Config written → {config_path.relative_to(PROJECT_ROOT)}")

    save_stage4(
        sizing_mode=params["sizing_mode"],
        vol_target=params["vol_target"],
        vol_window=params["vol_window"],
        atr_stop_mult=params["atr_stop_mult"],
        instrument=inst_lower,
    )

    print("\n  [PASS] Stage 4 complete.")
    print(f"  Next: uv run python research/etf_trend/run_portfolio.py --instrument {instrument}")
    print("=" * 60)


if __name__ == "__main__":
    main()
