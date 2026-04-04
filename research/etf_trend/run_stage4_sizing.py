"""run_stage4_sizing.py -- Stage 4: Sizing Mode + ATR Stop Sweep.

Sweeps three sizing modes head-to-head:
  binary        -- fully invested (100%) whenever a trade is open
  dynamic_decel -- position scales continuously with decel composite:
                   decel=+1 -> 100%, decel=0 -> 50%, decel=-1 -> 0%
  vol_target    -- volatility-targeted leverage: size = clip(target_vol/realized_vol,
                   0.5, max_leverage). Boosts returns in calm markets, de-risks in
                   volatile markets. Allows leverage > 1x.

Also sweeps:
  atr_stop_mult -- ATR multiplier for hard stop order placement only

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

from research.etf_trend.run_stage3_exits import (  # noqa: E402
    apply_sma_confirmation,
    build_entry_signal,
    compute_decel_composite,
    compute_ma,
)
from research.etf_trend.state_manager import (  # noqa: E402
    get_stage1,
    get_stage2,
    get_stage3,
    save_stage4,
)

FEES = 0.001
SLIPPAGE = 0.0005

SIZING_MODES = ["binary", "dynamic_decel"]
SIZING_MODES_LEVERAGED = ["binary"]  # no margin leverage on top of already-leveraged ETFs
ATR_STOP_MULTS = [3.0, 4.0, 5.0]

# Exchange suffix per symbol (used when writing locked config)
INSTRUMENT_EXCHANGE: dict[str, str] = {
    "TQQQ": "NASDAQ",
    "IWB": "ARCA",
    "IWM": "ARCA",
    "QQQ": "ARCA",
    "SPY": "ARCA",
}

# Vol-target leverage sweep params
VOL_TARGETS = [0.10, 0.15, 0.20, 0.25]
MAX_LEVERAGES = [1.0, 1.25, 1.5, 2.0]

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


def compute_position_sizes(
    decel: pd.Series,
    sizing_mode: str,
) -> pd.Series:
    """Compute target position fraction (0 to 1) based on sizing mode.

    Binary:        fully invested (1.0) whenever in a trade.
    dynamic_decel: scales continuously with decel composite.
                   decel=+1 -> 100%, decel=0 -> 50%, decel=-1 -> 0%.

    Args:
        decel: Composite decel signal [-1, 1].
        sizing_mode: 'binary' or 'dynamic_decel'.

    Returns:
        Position fraction series [0, 1].
    """
    if sizing_mode == "binary":
        return pd.Series(1.0, index=decel.index)
    else:  # dynamic_decel
        return ((decel + 1) / 2).clip(0, 1)


def compute_vol_target_sizes(
    close: pd.Series,
    in_regime: pd.Series,
    target_vol: float,
    max_leverage: float,
    vol_window: int = 20,
) -> pd.Series:
    """Return per-bar leverage fraction (0 when flat, up to max_leverage when in regime).

    Uses shift(1) -- no look-ahead. When regime is False, size = 0 (flat).
    In calm markets (realized_vol < target_vol), leverage > 1.0.
    In volatile markets, leverage < 1.0 (de-risked).
    """
    rets = close.pct_change()
    realized_vol = rets.rolling(vol_window).std().bfill() * np.sqrt(252)
    leverage = (target_vol / realized_vol).clip(lower=0.5, upper=max_leverage)
    return leverage.where(in_regime, other=0.0).shift(1).fillna(0.0)


def run_vol_target_stats(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_mode: str,
    exit_decel_thresh: float,
    target_vol: float,
    max_leverage: float,
    entry_mode: str = "decel_positive",
    fast_ma: pd.Series | None = None,
    confirm_days: int = 1,
    exec_close: pd.Series | None = None,
) -> dict:
    """Simulate vol-targeted leverage strategy and return stats dict directly.

    Returns all stats needed for the scoreboard without a VBT Portfolio object.
    Transaction costs charged at entry and exit bars only.

    Args:
        close: Signal close (used for regime gate and vol computation).
        exec_close: Execution close for P&L (defaults to close when None).
    """
    in_regime = close > slow_ma

    size_arr = compute_vol_target_sizes(close, in_regime, target_vol, max_leverage)

    # Detect actual regime transitions (not raw entry signal) for cost accounting.
    # size_arr[t] > 0 = in trade; size_arr[t] == 0 = flat.
    in_pos = size_arr > 0
    entering = in_pos & (~in_pos.shift(1).fillna(True))  # flat → long
    exiting = (~in_pos) & in_pos.shift(1).fillna(False)  # long → flat

    exec_c = exec_close if exec_close is not None else close
    daily_rets = exec_c.pct_change().fillna(0)
    strat_rets = daily_rets * size_arr
    strat_rets -= entering.astype(float) * (FEES + SLIPPAGE)
    strat_rets -= exiting.astype(float) * (FEES + SLIPPAGE)

    equity = INIT_CASH * (1 + strat_rets).cumprod()
    total_ret = float(equity.iloc[-1] / INIT_CASH - 1)

    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = float(dd.min())

    std = strat_rets.std()
    sharpe = float(strat_rets.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
    # Match VectorBT calmar_ratio: annualized_return (365-basis) / abs(max_dd)
    n_bars = len(close)
    ann_ret = (1 + total_ret) ** (365 / n_bars) - 1
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-9 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd, 3),
        "total_return": round(total_ret, 3),
        "n_trades": int(entering.sum()),
        "win_rate": 0.0,
    }


def run_sized_backtest(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_mode: str,
    exit_decel_thresh: float,
    sizing_mode: str,
    atr_stop_mult: float,
    entry_mode: str = "decel_positive",
    fast_ma: pd.Series | None = None,
    confirm_days: int = 1,
    decel_confirm_days: int = 1,
    exec_close: pd.Series | None = None,
) -> "vbt.Portfolio":
    """Run backtest with decel-scaled position sizing.

    Args:
        close: Signal close (for regime gate, entry/exit signals).
        slow_ma: Slow MA (sole trend boundary -- fast MA removed).
        decel: Decel composite (already shifted).
        exit_mode: 'A', 'C', or 'D'.
        exit_decel_thresh: Threshold for C/D exit.
        sizing_mode: 'binary' or 'dynamic_decel'.
        atr_stop_mult: ATR multiplier (used for stop order placement only).
        entry_mode: 'decel_positive' or 'asymmetric'.
        fast_ma: Fast re-entry MA for asymmetric mode (optional).
        confirm_days: Consecutive closes below slow_ma required to fire SMA-break exit.
        decel_confirm_days: Consecutive bars decel must stay below threshold before exit.
        exec_close: Execution close for P&L (defaults to close when None).

    Returns:
        VBT Portfolio object.
    """
    entry_signal = build_entry_signal(close, slow_ma, decel, entry_mode, fast_ma)

    below_slow = apply_sma_confirmation(~(close > slow_ma), confirm_days)
    if exit_mode == "A":
        exit_signal = below_slow
    elif exit_mode == "C":
        exit_signal = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
    else:  # D -- confirmed SMA break OR sustained decel collapse
        decel_exit = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
        exit_signal = below_slow | decel_exit

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)
    exec_c = exec_close if exec_close is not None else close

    if sizing_mode == "binary":
        # Fully invested with compounding -- matches Stage 3 behavior (VBT default)
        return vbt.Portfolio.from_signals(
            exec_c,
            entries=entries,
            exits=exits,
            init_cash=INIT_CASH,
            fees=FEES,
            slippage=SLIPPAGE,
            freq="1D",
        )
    else:  # dynamic_decel -- fractional allocation on initial cash
        sizes = compute_position_sizes(decel, sizing_mode).shift(1).fillna(0)
        return vbt.Portfolio.from_signals(
            exec_c,
            entries=entries,
            exits=exits,
            size=sizes * INIT_CASH,
            size_type="value",
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


def write_config(instrument: str, params: dict, signal_instrument: str | None = None) -> Path:
    """Write locked parameters to config/etf_trend_{instrument_lower}.toml.

    Args:
        instrument: Instrument symbol (lowercase).
        params: Dict of all locked parameters.
        signal_instrument: Signal source symbol when different from instrument (e.g. QQQ for TQQQ).

    Returns:
        Path to the written config file.
    """
    inst_upper = instrument.upper()
    exchange = INSTRUMENT_EXCHANGE.get(inst_upper, "ARCA")
    lines = [
        f"# ETF Trend Strategy -- {inst_upper} locked config",
        "# Generated by run_stage4_sizing.py",
        "",
        f'instrument        = "{inst_upper}.{exchange}"',
    ]
    if signal_instrument and signal_instrument.upper() != inst_upper:
        sig_upper = signal_instrument.upper()
        sig_exchange = INSTRUMENT_EXCHANGE.get(sig_upper, "ARCA")
        lines.append(f'signal_instrument = "{sig_upper}.{sig_exchange}"')
    lines += [
        f'ma_type           = "{params["ma_type"]}"',
        f"slow_ma           = {params['slow_ma']}",
        f"decel_signals     = {params['decel_signals']}",
        *(
            [
                f"d_pct_smooth      = {params['decel_params']['d_pct_smooth']}",
                f"rv_window         = {params['decel_params']['rv_window']}",
                f"macd_fast         = {params['decel_params']['macd_fast']}",
            ]
            if params.get("decel_params")
            else []
        ),
        f'entry_mode        = "{params["entry_mode"]}"',
        *(
            [f"fast_reentry_ma   = {params['fast_reentry_ma']}"]
            if params["fast_reentry_ma"] is not None
            else []
        ),
        f'exit_mode         = "{params["exit_mode"]}"',
        f"exit_decel_thresh = {params['exit_decel_thresh']}",
        f"exit_confirm_days = {params['exit_confirm_days']}",
        f"decel_confirm_days = {params['decel_confirm_days']}",
        f'sizing_mode       = "{params["sizing_mode"]}"',
        *(
            [
                f"vol_target        = {params['vol_target']}",
                f"max_leverage      = {params['max_leverage']}",
            ]
            if params["sizing_mode"] == "vol_target"
            else []
        ),
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
    parser.add_argument(
        "--signal-instrument",
        default=None,
        help="Signal source instrument (e.g. QQQ for TQQQ). Defaults to --instrument.",
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()
    signal_inst = (args.signal_instrument or instrument).upper()
    is_dual = signal_inst != instrument

    print("=" * 60)
    print("  ETF Trend -- Stage 4: Volatility Sizing + ATR Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")
    if is_dual:
        print(f"  Signal src: {signal_inst}  (signals on {signal_inst}, P&L on {instrument})")

    # ── Load previous stages ──────────────────────────────────────────────
    if args.load_state:
        s1 = get_stage1(inst_lower)
        s2 = get_stage2(inst_lower)
        s3 = get_stage3(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found.")
            sys.exit(1)
        ma_type, slow_ma_p, _s1_fast_ma = s1
        decel_signals, _decel_weights, decel_params = s2 if s2 else ([], {}, {})
        s3_result = s3 if s3 else ("A", -0.3, "decel_positive", None, 1, 1)
        (
            exit_mode,
            exit_decel_thresh,
            entry_mode,
            fast_reentry_ma_p,
            exit_confirm_days_p,
            decel_confirm_days_p,
        ) = s3_result
    else:
        ma_type, slow_ma_p = "SMA", 200
        decel_signals, _decel_weights, decel_params = [], {}, {}
        (
            exit_mode,
            exit_decel_thresh,
            entry_mode,
            fast_reentry_ma_p,
            exit_confirm_days_p,
            decel_confirm_days_p,
        ) = ("A", -0.3, "decel_positive", None, 1, 1)

    # Binary-only sizing for leveraged ETFs (already 3× leveraged; no additional margin)
    active_sizing_modes = SIZING_MODES_LEVERAGED if is_dual else SIZING_MODES
    vol_combos = 0 if is_dual else len(VOL_TARGETS) * len(MAX_LEVERAGES) * len(ATR_STOP_MULTS)

    print(f"  MA: {ma_type} slow={slow_ma_p}")
    print(f"  Decel: {decel_signals or '(none)'}")
    print(
        f"  Exit mode: {exit_mode} thresh={exit_decel_thresh} "
        f"confirm={exit_confirm_days_p}d decel_confirm={decel_confirm_days_p}d"
    )
    print(f"  Entry mode: {entry_mode}")
    if fast_reentry_ma_p:
        print(f"  Fast re-entry MA: {fast_reentry_ma_p}")
    if is_dual:
        print("  [Leveraged ETF mode] -- binary sizing only, no vol_target")
    total = len(active_sizing_modes) * len(ATR_STOP_MULTS) + vol_combos
    print(f"  Total combos: {total}")

    exec_df = load_data(instrument)
    if is_dual:
        sig_df = load_data(signal_inst)
        common = sig_df.index.intersection(exec_df.index)
        sig_df = sig_df.loc[common]
        exec_df = exec_df.loc[common]
        close = sig_df["close"]
        exec_close_full: pd.Series | None = exec_df["close"]
    else:
        sig_df = exec_df
        close = exec_df["close"]
        exec_close_full = None

    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]
    is_exec = exec_close_full.iloc[:split] if exec_close_full is not None else None
    oos_exec = exec_close_full.iloc[split:] if exec_close_full is not None else None

    slow_full = compute_ma(close, slow_ma_p, ma_type)
    is_slow = slow_full.iloc[:split]
    oos_slow = slow_full.iloc[split:]

    decel_full = compute_decel_composite(close, sig_df, slow_full, decel_signals, decel_params)
    is_decel = decel_full.iloc[:split].fillna(0)
    oos_decel = decel_full.iloc[split:].fillna(0)

    # Pre-compute fast re-entry MA for asymmetric mode
    if fast_reentry_ma_p is not None:
        fast_full = compute_ma(close, fast_reentry_ma_p, ma_type)
        is_fast_ma = fast_full.iloc[:split]
        oos_fast_ma = fast_full.iloc[split:]
    else:
        is_fast_ma = None
        oos_fast_ma = None

    results: list[dict] = []
    print("\n  Running sweep ...")

    for sizing_mode in active_sizing_modes:
        for atr_mult in ATR_STOP_MULTS:
            is_pf = run_sized_backtest(
                is_close,
                is_slow,
                is_decel,
                exit_mode,
                exit_decel_thresh,
                sizing_mode,
                atr_mult,
                entry_mode,
                is_fast_ma,
                exit_confirm_days_p,
                decel_confirm_days_p,
                exec_close=is_exec,
            )
            oos_pf = run_sized_backtest(
                oos_close,
                oos_slow,
                oos_decel,
                exit_mode,
                exit_decel_thresh,
                sizing_mode,
                atr_mult,
                entry_mode,
                oos_fast_ma,
                exit_confirm_days_p,
                decel_confirm_days_p,
                exec_close=oos_exec,
            )
            is_stats = extract_stats(is_pf)
            oos_stats = extract_stats(oos_pf)
            ratio = oos_stats["sharpe"] / is_stats["sharpe"] if is_stats["sharpe"] > 0.01 else 0.0
            results.append(
                {
                    "sizing_mode": sizing_mode,
                    "vol_target": None,
                    "max_leverage": None,
                    "atr_stop_mult": atr_mult,
                    "is_sharpe": round(is_stats["sharpe"], 3),
                    "oos_sharpe": round(oos_stats["sharpe"], 3),
                    "oos_is_ratio": round(ratio, 3),
                    "oos_calmar": round(oos_stats["calmar"], 3),
                    "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                    "oos_n_trades": oos_stats["n_trades"],
                    "oos_total_return": round(oos_stats["total_return"], 3),
                }
            )

    # Vol-target leverage sweep (skipped for leveraged ETFs)
    if not is_dual:
        for target_vol in VOL_TARGETS:
            for max_lev in MAX_LEVERAGES:
                for atr_mult in ATR_STOP_MULTS:
                    is_stats = run_vol_target_stats(
                        is_close,
                        is_slow,
                        is_decel,
                        exit_mode,
                        exit_decel_thresh,
                        target_vol,
                        max_lev,
                        entry_mode,
                        is_fast_ma,
                        exit_confirm_days_p,
                    )
                    oos_stats = run_vol_target_stats(
                        oos_close,
                        oos_slow,
                        oos_decel,
                        exit_mode,
                        exit_decel_thresh,
                        target_vol,
                        max_lev,
                        entry_mode,
                        oos_fast_ma,
                        exit_confirm_days_p,
                    )
                    ratio = (
                        oos_stats["sharpe"] / is_stats["sharpe"]
                        if is_stats["sharpe"] > 0.01
                        else 0.0
                    )
                    results.append(
                        {
                            "sizing_mode": "vol_target",
                            "vol_target": target_vol,
                            "max_leverage": max_lev,
                            "atr_stop_mult": atr_mult,
                            "is_sharpe": round(is_stats["sharpe"], 3),
                            "oos_sharpe": round(oos_stats["sharpe"], 3),
                            "oos_is_ratio": round(ratio, 3),
                            "oos_calmar": round(oos_stats["calmar"], 3),
                            "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                            "oos_n_trades": oos_stats["n_trades"],
                            "oos_total_return": round(oos_stats["total_return"], 3),
                        }
                    )

    scoreboard = pd.DataFrame(results).sort_values("oos_calmar", ascending=False)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage4_{inst_lower}_sizing.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved: {scoreboard_path.relative_to(PROJECT_ROOT)}")

    # B&H OOS reference -- instrument-specific, used for goal-aware selection.
    # Prefer configs that beat B&H return while keeping drawdown within 5% of B&H.
    # If none qualify, fall back to best Calmar.
    _bah_targets: dict[str, dict[str, float]] = {
        "spy": {"return": 2.808, "max_dd": -0.337},
        "qqq": {"return": 2.867, "max_dd": -0.351},
    }
    _bah = _bah_targets.get(inst_lower, {"return": 1.0, "max_dd": -0.5})
    BAH_RETURN = _bah["return"]
    BAH_MAX_DD = _bah["max_dd"]
    print(f"\n  B&H target: return > {BAH_RETURN:.1%}, MaxDD > {BAH_MAX_DD:.1%}")
    beats_bah = scoreboard[
        (scoreboard["oos_total_return"] > BAH_RETURN)
        & (scoreboard["oos_max_dd"] > BAH_MAX_DD * 1.05)
    ]
    if not beats_bah.empty:
        best = beats_bah.sort_values("oos_calmar", ascending=False).iloc[0]
        print("\n  [INFO] B&H-beating config found -- using return+drawdown selection.")
    else:
        best = scoreboard.iloc[0]
    print("\n  -- Best sizing configuration ----------------------")
    for col in [
        "sizing_mode",
        "vol_target",
        "max_leverage",
        "atr_stop_mult",
        "is_sharpe",
        "oos_sharpe",
        "oos_calmar",
        "oos_max_dd",
        "oos_total_return",
    ]:
        print(f"  {col:20s}: {best[col]}")

    best_vol_target = float(best["vol_target"]) if best["vol_target"] is not None else 0.15
    best_max_lev = float(best["max_leverage"]) if best["max_leverage"] is not None else 1.0

    # ── Write config ──────────────────────────────────────────────────────
    params = {
        "ma_type": ma_type,
        "slow_ma": slow_ma_p,
        "decel_signals": decel_signals,
        "decel_params": decel_params,
        "entry_mode": entry_mode,
        "fast_reentry_ma": fast_reentry_ma_p,
        "exit_mode": exit_mode,
        "exit_decel_thresh": exit_decel_thresh,
        "exit_confirm_days": exit_confirm_days_p,
        "decel_confirm_days": decel_confirm_days_p,
        "sizing_mode": str(best["sizing_mode"]),
        "vol_target": best_vol_target,
        "max_leverage": best_max_lev,
        "atr_stop_mult": float(best["atr_stop_mult"]),
    }
    config_path = write_config(
        inst_lower,
        params,
        signal_instrument=signal_inst if is_dual else None,
    )
    print(f"\n  Config written: {config_path.relative_to(PROJECT_ROOT)}")

    save_stage4(
        sizing_mode=params["sizing_mode"],
        atr_stop_mult=params["atr_stop_mult"],
        vol_target=best_vol_target,
        max_leverage=best_max_lev,
        instrument=inst_lower,
    )

    print("\n  [PASS] Stage 4 complete.")
    print(f"  Next: uv run python research/etf_trend/run_portfolio.py --instrument {instrument}")
    print("=" * 60)


if __name__ == "__main__":
    main()
