"""run_stage3_exits.py -- Stage 3: Exit Mode + Entry Mode Sweep.

Sweeps exit strategies and entry modes head-to-head.

Exit modes (slow MA is sole trend-reversal gate):
  A -- Close below slow_ma (trend reversal)
  C -- Decel composite < exit_thresh (early exit before price breaks)
  D -- A OR C, whichever fires first

Entry modes (all use crossover -- fire only when condition becomes True):
  decel_positive -- enter when (close > slow_ma AND decel >= 0) transitions False→True
  decel_cross    -- enter ONLY when decel crosses from negative to positive
  asymmetric     -- enter on crossover into (close > slow_ma OR close > fast_ma)
                    (fast_ma period swept from FAST_REENTRY_MAS)
                    Re-entry blocked if price is >10% below slow_ma (deep bear).
  dual_regime    -- enter on crossover into (close > fast_ma AND close > slow_ma)
                    (fast_ma period swept from FAST_REENTRY_MAS)

Usage:
    uv run python research/etf_trend/run_stage3_exits.py --instrument SPY --load-state
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

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from research.etf_trend.run_stage2_decel import (  # noqa: E402
    adx_signal,
    build_composite,
    compute_ma,
    d_pct,
    macd_signal,
    rv_signal,
)
from research.etf_trend.state_manager import get_stage1, get_stage2, save_stage3  # noqa: E402
from titan.strategies.ml.features import adx as compute_adx  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005

EXIT_MODES = ["A", "C", "D"]
EXIT_DECEL_THRESHOLDS = [-0.5, -0.3, -0.1, 0.0]
# Consecutive closes below slow_ma required before exit fires; 1 = no filter (current behavior)
EXIT_CONFIRM_DAYS = [1, 2, 3, 5]
# Consecutive bars decel must stay below threshold before decel-exit fires (C and D modes only)
DECEL_CONFIRM_DAYS = [1, 3, 5]
ENTRY_MODES = ["decel_positive", "asymmetric", "dual_regime"]
# Fast MA periods swept when entry_mode == "asymmetric" or "dual_regime"
FAST_REENTRY_MAS = [30, 50, 75, 100]
# Max distance below slow_ma to allow asymmetric re-entry (10% buffer)
ASYMMETRIC_BEAR_BUFFER = 0.90


def apply_sma_confirmation(below_slow: pd.Series, n_days: int) -> pd.Series:
    """Exit only after N consecutive closes below slow_ma.

    n_days=1 reproduces single-bar behavior (no filter).
    rolling(n).min() on a 0/1 series returns 1 only when ALL n values are 1.
    No look-ahead: only past and current bars used.

    Args:
        below_slow: Boolean series, True when close < slow_ma.
        n_days: Minimum consecutive days below slow_ma to trigger exit.

    Returns:
        Confirmed exit boolean series.
    """
    if n_days <= 1:
        return below_slow
    return below_slow.rolling(n_days, min_periods=n_days).min().fillna(0).astype(bool)


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


def compute_decel_composite(
    close: pd.Series,
    df: pd.DataFrame,
    slow_ma_ser: pd.Series,
    signals: list[str],
    params: dict | None = None,
) -> pd.Series:
    """Build composite decel signal from selected signal set.

    Args:
        params: Optional signal hyperparameters from Stage 2 sweep.
                Keys: d_pct_smooth, rv_window, macd_fast. Defaults used if absent.
    """
    if not signals:
        return pd.Series(1.0, index=close.index)

    p = params or {}
    parts: dict[str, pd.Series] = {}
    if "d_pct" in signals:
        parts["d_pct"] = d_pct(close, slow_ma_ser, smooth=int(p.get("d_pct_smooth", 10)))
    if "rv_20" in signals:
        parts["rv_20"] = rv_signal(close, window=int(p.get("rv_window", 20)))
    if "adx_14" in signals:
        parts["adx_14"] = adx_signal(df)
    if "macd_hist" in signals:
        parts["macd_hist"] = macd_signal(close, fast=int(p.get("macd_fast", 12)))

    n = len(parts)
    weights = {k: 1.0 / n for k in parts}
    return build_composite(parts, weights)


def _crossover(condition: pd.Series) -> pd.Series:
    """Return True only on bars where condition transitions False → True.

    Prevents entries when the condition was already True (e.g. price already
    above MA at warmup end, or immediate re-entry after decel exit while price
    stays above the MA).
    """
    return condition & ~condition.shift(1).fillna(False)


def build_entry_signal(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    entry_mode: str,
    fast_ma: pd.Series | None = None,
) -> pd.Series:
    """Build entry signal based on entry mode.

    All modes return a crossover signal (True only when the entry condition
    BECOMES True, not every bar it stays True). This ensures:
      - The first trade waits for price to actually cross the MA from below,
        not just to be above it when the warmup period ends.
      - After a decel exit while price stays above the MA, re-entry waits for
        a genuine regime change (e.g. decel recovers while price dips and
        recrosses, or price dips through the MA and recovers).

    decel_positive: crossover of (close > slow_ma AND decel >= 0).
    decel_cross:    close > slow_ma AND decel crosses 0 from below.
    asymmetric:     crossover into (close > slow_ma) OR (close > fast_ma AND
                    not deep bear). Re-enters faster than slow-MA exit gate.
    dual_regime:    crossover into (close > fast_ma AND close > slow_ma).
                    Stricter Malik gate; re-entry only on confirmed dual-MA cross.

    Args:
        close: Close price series (unshifted).
        slow_ma: Slow MA series (sole exit/trend boundary).
        decel: Composite decel signal (unshifted).
        entry_mode: 'decel_positive', 'decel_cross', 'asymmetric', or 'dual_regime'.
        fast_ma: Fast MA for asymmetric re-entry or dual_regime gate.

    Returns:
        Boolean entry signal series (unshifted -- caller applies shift).
    """
    above_slow = close > slow_ma

    if entry_mode == "asymmetric" and fast_ma is not None:
        above_fast = close > fast_ma
        not_deep_bear = close > slow_ma * ASYMMETRIC_BEAR_BUFFER
        in_regime = above_slow | (above_fast & not_deep_bear)
        return _crossover(in_regime)

    if entry_mode == "dual_regime" and fast_ma is not None:
        # Crossover into both-MA regime (Malik dual-MA gate)
        above_both = above_slow & (close > fast_ma)
        return _crossover(above_both)

    decel_pos = decel >= 0
    if entry_mode == "decel_cross":
        # Decel crosses 0 from below while above MA (unchanged -- already a crossover)
        decel_was_neg = ~decel_pos.shift(1).fillna(True)
        return above_slow & decel_pos & decel_was_neg

    # decel_positive: crossover into (above slow MA AND positive decel)
    return _crossover(above_slow & decel_pos)


def run_exit_backtest(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_mode: str,
    exit_decel_thresh: float = -0.3,
    entry_mode: str = "decel_positive",
    fast_ma: pd.Series | None = None,
    confirm_days: int = 1,
    decel_confirm_days: int = 1,
    exec_close: pd.Series | None = None,
) -> "vbt.Portfolio":
    """Run long-only backtest with specified exit and entry modes.

    Args:
        close: Signal close (used for regime gate, entry/exit logic).
        confirm_days: Consecutive closes below slow_ma required to fire SMA-break exit.
                      1 = original single-bar behavior. Applies to modes A and D only.
        decel_confirm_days: Consecutive bars decel must stay below threshold before
                            decel-exit fires. 1 = immediate (original). Applies to C and D.
        exec_close: Execution close for P&L (defaults to close when None).
    """
    entry_signal = build_entry_signal(close, slow_ma, decel, entry_mode, fast_ma)

    below_slow = apply_sma_confirmation(~(close > slow_ma), confirm_days)
    if exit_mode == "A":
        exit_signal = below_slow
    elif exit_mode == "C":
        exit_signal = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
    else:  # D: confirmed SMA break OR sustained decel collapse
        decel_exit = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
        exit_signal = below_slow | decel_exit

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)
    exec_c = exec_close if exec_close is not None else close

    return vbt.Portfolio.from_signals(
        exec_c,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def regime_split(close: pd.Series, df: pd.DataFrame, n_sma: int = 250) -> dict[str, pd.Series]:
    sma250 = close.rolling(n_sma, min_periods=1).mean()
    adx_vals = compute_adx(df)
    bull = (close > sma250) & (adx_vals > 25)
    bear = close < sma250
    choppy = ~(bull | bear)
    return {"bull": bull, "bear": bear, "choppy": choppy}


def sharpe_in_regime(pf: "vbt.Portfolio", mask: pd.Series) -> float:
    ret = pf.returns()
    ret_masked = ret[mask.reindex(ret.index, fill_value=False)]
    if len(ret_masked) < 10:
        return float("nan")
    mean_r = ret_masked.mean()
    std_r = ret_masked.std()
    if std_r < 1e-8:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(252))


def extract_stats(pf: "vbt.Portfolio") -> dict:
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "calmar": float(pf.calmar_ratio()),
        "total_return": float(pf.total_return()),
        "n_trades": int(n),
        "win_rate": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 3: Exit Mode Sweep.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument("--load-state", action="store_true", help="Load Stage 1+2 from state")
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
    print("  ETF Trend -- Stage 3: Exit Mode + Entry Mode Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")
    if is_dual:
        print(f"  Signal src: {signal_inst}  (signals on {signal_inst}, P&L on {instrument})")

    if args.load_state:
        s1 = get_stage1(inst_lower)
        s2 = get_stage2(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found.")
            sys.exit(1)
        ma_type, slow_ma_p, _stage1_fast_ma = s1
        decel_signals, _, decel_params = s2 if s2 else ([], {}, {})
    else:
        ma_type, slow_ma_p = "SMA", 200
        decel_signals = []
        decel_params: dict = {}
        print("  [No state] Using defaults.")

    print(f"  MA: {ma_type} slow={slow_ma_p}")
    print(f"  Decel signals: {decel_signals or '(none)'}")
    print(f"  Exit modes:    {EXIT_MODES}")
    print(f"  Entry modes:   {ENTRY_MODES}")
    print(f"  Fast re-entry MAs (asymmetric): {FAST_REENTRY_MAS}")
    print(f"  Bear buffer:   {ASYMMETRIC_BEAR_BUFFER:.0%} of slow_ma")

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
    _is_df, oos_df = sig_df.iloc[:split], sig_df.iloc[split:]

    slow_full = compute_ma(close, slow_ma_p, ma_type)
    is_slow = slow_full.iloc[:split]
    oos_slow = slow_full.iloc[split:]

    decel_full = compute_decel_composite(close, sig_df, slow_full, decel_signals, decel_params)
    is_decel_raw = decel_full.iloc[:split].fillna(0)
    oos_decel_raw = decel_full.iloc[split:].fillna(0)

    regimes = regime_split(oos_close, oos_df)

    results: list[dict] = []
    print("\n  Running combinations ...\n")

    for entry_mode in ENTRY_MODES:
        uses_fast = entry_mode in ("asymmetric", "dual_regime")
        fast_ma_options: list[int | None] = FAST_REENTRY_MAS if uses_fast else [None]  # type: ignore[assignment]

        for fast_p in fast_ma_options:
            # Pre-compute fast MA if needed
            if fast_p is not None:
                fast_full = compute_ma(close, fast_p, ma_type)
                is_fast_ma = fast_full.iloc[:split]
                oos_fast_ma = fast_full.iloc[split:]
            else:
                is_fast_ma = None
                oos_fast_ma = None

            for exit_mode in EXIT_MODES:
                for thresh in EXIT_DECEL_THRESHOLDS:
                    if exit_mode == "A" and thresh != EXIT_DECEL_THRESHOLDS[0]:
                        continue  # thresh irrelevant for A

                    # confirm_days only applies to SMA-break component (modes A and D)
                    confirm_day_options = EXIT_CONFIRM_DAYS if exit_mode != "C" else [1]
                    # decel_confirm_days applies to decel exit component (modes C and D)
                    decel_confirm_options = DECEL_CONFIRM_DAYS if exit_mode in ("C", "D") else [1]

                    for confirm_days in confirm_day_options:
                        for decel_confirm in decel_confirm_options:
                            is_pf = run_exit_backtest(
                                is_close,
                                is_slow,
                                is_decel_raw,
                                exit_mode,
                                thresh,
                                entry_mode,
                                is_fast_ma,
                                confirm_days,
                                decel_confirm,
                                exec_close=is_exec,
                            )
                            oos_pf = run_exit_backtest(
                                oos_close,
                                oos_slow,
                                oos_decel_raw,
                                exit_mode,
                                thresh,
                                entry_mode,
                                oos_fast_ma,
                                confirm_days,
                                decel_confirm,
                                exec_close=oos_exec,
                            )

                            oos_stats = extract_stats(oos_pf)
                            is_stats = extract_stats(is_pf)
                            ratio = (
                                oos_stats["sharpe"] / is_stats["sharpe"]
                                if is_stats["sharpe"] > 0.01
                                else 0.0
                            )

                            bull_sharpe = sharpe_in_regime(oos_pf, regimes["bull"])
                            bear_sharpe = sharpe_in_regime(oos_pf, regimes["bear"])
                            choppy_sharpe = sharpe_in_regime(oos_pf, regimes["choppy"])

                            fast_label = f"reentry={fast_p}" if fast_p else ""
                            label = (
                                f"[{entry_mode[:10]} {fast_label}] Mode {exit_mode}"
                                + (f" t={thresh}" if exit_mode in ("C", "D") else "")
                                + (f" c={confirm_days}" if confirm_days > 1 else "")
                                + (f" dc={decel_confirm}" if decel_confirm > 1 else "")
                            )
                            results.append(
                                {
                                    "entry_mode": entry_mode,
                                    "fast_reentry_ma": fast_p if fast_p else "n/a",
                                    "exit_mode": exit_mode,
                                    "exit_decel_thresh": thresh
                                    if exit_mode in ("C", "D")
                                    else "n/a",
                                    "exit_confirm_days": confirm_days,
                                    "decel_confirm_days": decel_confirm,
                                    "is_sharpe": round(is_stats["sharpe"], 3),
                                    "oos_sharpe": round(oos_stats["sharpe"], 3),
                                    "oos_is_ratio": round(ratio, 3),
                                    "oos_calmar": round(oos_stats["calmar"], 3),
                                    "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                                    "oos_total_return": round(oos_stats["total_return"], 3),
                                    "oos_n_trades": oos_stats["n_trades"],
                                    "bull_sharpe": round(bull_sharpe, 3)
                                    if not np.isnan(bull_sharpe)
                                    else "n/a",
                                    "bear_sharpe": round(bear_sharpe, 3)
                                    if not np.isnan(bear_sharpe)
                                    else "n/a",
                                    "choppy_sharpe": round(choppy_sharpe, 3)
                                    if not np.isnan(choppy_sharpe)
                                    else "n/a",
                                }
                            )
                            print(
                                f"  {label:52s}  Sharpe={oos_stats['sharpe']:.3f}  "
                                f"Ret={oos_stats['total_return']:.1%}  "
                                f"DD={oos_stats['max_drawdown']:.1%}  "
                                f"Cal={oos_stats['calmar']:.3f}  "
                                f"N={oos_stats['n_trades']}"
                            )

    # Sort by Calmar -- penalises high-DD configs that just hold forever
    scoreboard = pd.DataFrame(results).sort_values("oos_calmar", ascending=False)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage3_{inst_lower}_exits.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved: {scoreboard_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]
    best_entry = str(best["entry_mode"])
    best_mode = str(best["exit_mode"])
    best_thresh_raw = best["exit_decel_thresh"]
    best_thresh = float(best_thresh_raw) if best_thresh_raw != "n/a" else -0.3
    best_fast_raw = best["fast_reentry_ma"]
    best_fast = int(best_fast_raw) if best_fast_raw != "n/a" else None
    best_confirm = int(best["exit_confirm_days"])
    best_decel_confirm = int(best["decel_confirm_days"])

    print("\n  -- Best configuration (by OOS Calmar) ------")
    print(f"  Entry mode:         {best_entry}")
    if best_fast:
        print(f"  Fast reentry MA:    {best_fast}")
    print(f"  Exit mode:          {best_mode}")
    if best_mode in ("C", "D"):
        print(f"  Decel thresh:       {best_thresh}")
        print(f"  Decel confirm days: {best_decel_confirm}")
    print(f"  SMA confirm days:   {best_confirm}")
    print(f"  IS Sharpe:          {best['is_sharpe']}")
    print(f"  OOS Sharpe:         {best['oos_sharpe']}")
    print(f"  OOS Calmar:         {best['oos_calmar']}")
    print(f"  OOS MaxDD:          {best['oos_max_dd']:.1%}")
    print(f"  OOS Total Return:   {best['oos_total_return']:.1%}")
    print(f"  OOS N trades:       {best['oos_n_trades']}")

    print("\n  -- Regime split (best config) --------------------")
    for regime in ("bull", "bear", "choppy"):
        print(f"  {regime:10s}: Sharpe = {best[f'{regime}_sharpe']}")

    save_stage3(
        exit_mode=best_mode,
        exit_decel_thresh=best_thresh,
        entry_mode=best_entry,
        fast_reentry_ma=best_fast,
        exit_confirm_days=best_confirm,
        decel_confirm_days=best_decel_confirm,
        instrument=inst_lower,
    )

    print("\n  [PASS] Stage 3 complete.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage4_sizing.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
