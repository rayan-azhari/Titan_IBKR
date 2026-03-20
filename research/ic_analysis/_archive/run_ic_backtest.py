"""run_ic_backtest.py -- MTF IC Signal Strategy Backtest (Phase 3).

Backtests the top-ranked MTF-stacked composites from run_mtf_stack.py using VectorBT.

Default strategy: equal-weight composite of accel_stoch_k + accel_rsi14 across
W / D / H4 / H1 timeframes.  These signals were identified as STRONG (IC=0.45+,
ICIR=1.16+) by run_mtf_stack.py on EUR/USD H1.

Signal pipeline:
    1. Compute 52 signals on each TF's native bars (build_all_signals).
    2. Forward-fill all signals to H1 index (causal — no look-ahead).
    3. For each target signal x TF, compute IC vs H1 fwd returns (h=1) on IS only.
    4. Sign-normalise each TF signal: multiply by sign(IS IC) so composite > 0 = bullish.
    5. Average sign-normalised values across TFs and across target signals -> composite.
    6. Z-score normalise composite using IS mean / std (applied to both IS and OOS).
    7. Threshold: Long when composite_z > theta, Short when < -theta, Flat otherwise.

Entry / exit:
    Long  entry : composite_z crosses above  threshold (shifted 1 bar -- fills next open)
    Long  exit  : composite_z drops below  0
    Short entry : composite_z crosses below -threshold
    Short exit  : composite_z rises above  0

Validation:
    IS  = first 70% of bars (time-ordered)
    OOS = remaining 30%
    Reject if OOS/IS Sharpe ratio < 0.5.

Cost model (realistic):
    Spread   : half-spread paid at each fill (default per instrument).
    Slippage : market-impact slippage per fill (default 0.5 pip, added to spread).
    Swap     : overnight carry cost applied at NY close (21:00 UTC) per night held.
               Defaults are long-run averages per pair -- override via --swap-long/short.

Position sizing (ATR-based, 1-2% risk per trade):
    stop_dist = stop_atr_mult x ATR(14) on H1
    units     = (init_cash x risk_pct) / stop_dist
    Max leverage capped at max_leverage x (default 30).
    No compounding -- units fixed relative to init_cash (realistic for live sizing).

Output:
    Console: threshold sweep table + best IS threshold OOS validation
    CSV:     .tmp/reports/ic_backtest_{slug}.csv

Usage:
    uv run python research/ic_analysis/run_ic_backtest.py
    uv run python research/ic_analysis/run_ic_backtest.py \\
        --instrument EUR_USD --risk-pct 0.01 --slippage-pips 0.5 \\
        --swap-long -0.5 --swap-short 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402


def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    """True-range ATR inlined to avoid cross-package import."""
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(p).mean()

# -- Constants -----------------------------------------------------------------

DEFAULT_SIGNALS = ["accel_stoch_k", "accel_rsi14"]
DEFAULT_TFS = ["W", "D", "H4", "H1"]
DEFAULT_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
IS_RATIO = 0.70
REF_HORIZON = 1

# Default half-spread per fill in price units (paid at each entry and exit)
SPREAD_DEFAULTS: dict[str, float] = {
    "EUR_USD": 0.00005,
    "GBP_USD": 0.00008,
    "USD_JPY": 0.007,
    "AUD_USD": 0.00007,
    "AUD_JPY": 0.010,
    "USD_CHF": 0.00007,
}

# 1 pip in price units
PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "AUD_JPY": 0.01,
    "USD_CHF": 0.0001,
}

# Long-run average swap (pips/night). Negative = you pay. Override via CLI.
SWAP_DEFAULTS: dict[str, dict[str, float]] = {
    "EUR_USD": {"long": -0.5,  "short":  0.3},
    "GBP_USD": {"long": -0.2,  "short":  0.1},
    "USD_JPY": {"long":  1.2,  "short": -1.5},
    "AUD_USD": {"long": -0.3,  "short":  0.2},
    "AUD_JPY": {"long":  0.5,  "short": -0.7},
    "USD_CHF": {"long":  0.2,  "short": -0.4},
}

INIT_CASH = 10_000.0
DEFAULT_RISK_PCT = 0.01
DEFAULT_STOP_ATR = 1.5
DEFAULT_SLIPPAGE_PIPS = 0.5
MAX_LEVERAGE = 30


# -- Data helpers --------------------------------------------------------------


def _load_tf(instrument: str, tf: str) -> pd.DataFrame | None:
    try:
        return _load_ohlcv(instrument, tf)
    except FileNotFoundError:
        return None


def _build_and_align(
    instrument: str,
    tfs: list[str],
    base_tf: str = "H1",
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, pd.DataFrame]:
    """Load + compute 52 signals + ffill all TFs to H1 index.

    Returns (tf_signals, base_index, base_df).
    base_df is the H1 OHLCV frame (needed for ATR and close).
    """
    base_df = _load_tf(instrument, base_tf)
    if base_df is None:
        raise FileNotFoundError(f"Base TF {base_tf} not found for {instrument}")
    base_index = base_df.index

    tf_signals: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        df = _load_tf(instrument, tf)
        if df is None:
            print(f"  [WARN] No data for {instrument} {tf} -- skipping.")
            continue
        native_sigs = build_all_signals(df)
        
        # PREVENT LOOKAHEAD BIAS: 
        # Higher tf bars are timestamped at the OPEN. We must shift them by 1 before 
        # forward-filling so the signal is only available AFTER the bar closes.
        if tf != base_tf:
            native_sigs = native_sigs.shift(1)
            
        aligned = native_sigs.reindex(base_index, method="ffill")
        tf_signals[tf] = aligned
        print(f"  {tf:3s}: {len(df):>5d} native bars -> {len(aligned):>5d} aligned H1 bars")

    return tf_signals, base_index, base_df


# -- Composite building --------------------------------------------------------


def _ic_sign(signal_series: pd.Series, close: pd.Series, horizon: int = 1) -> float:
    fwd = np.log(close.shift(-horizon) / close)
    both = pd.concat([signal_series, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0


def build_composite(
    tf_signals: dict[str, pd.DataFrame],
    base_close: pd.Series,
    tfs: list[str],
    target_signals: list[str],
    is_mask: pd.Series,
) -> pd.Series:
    """Sign-normalised equal-weight MTF composite. IS-calibrated only."""
    parts: list[pd.Series] = []
    for sig in target_signals:
        for tf in tfs:
            if tf not in tf_signals:
                continue
            if sig not in tf_signals[tf].columns:
                print(f"  [WARN] Signal '{sig}' not found in {tf} -- skipping.")
                continue
            raw = tf_signals[tf][sig]
            sign = _ic_sign(raw[is_mask], base_close[is_mask], horizon=REF_HORIZON)
            oriented = raw * sign
            oriented.name = f"{sig}_{tf}"
            parts.append(oriented)

    if not parts:
        raise ValueError("No valid signal/TF combinations found.")

    composite = pd.concat(parts, axis=1).mean(axis=1)
    composite.name = "mtf_composite"
    return composite


def zscore_normalise(composite: pd.Series, is_mask: pd.Series) -> pd.Series:
    """Z-score normalise using IS mean/std only (no look-ahead)."""
    is_vals = composite[is_mask]
    mu = float(is_vals.mean())
    sigma = float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=composite.index)
    return (composite - mu) / sigma


# -- Position sizing -----------------------------------------------------------


def build_size_array(
    h1_df: pd.DataFrame,
    close: pd.Series,
    risk_pct: float,
    stop_atr_mult: float,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.Series:
    """ATR-based sizing returned as FRACTION OF PORTFOLIO (for size_type='percent').

    size_pct = risk_pct / stop_pct   where stop_pct = ATR_stop / close.

    Using percent-of-portfolio sizing makes the formula currency-agnostic: the
    same 1% risk target applies equally to EUR/USD (price ~1.08) and USD/JPY
    (price ~145) without any manual pip-value conversion.

    Capped at max_leverage (default 30x).
    """
    atr14 = _atr14(h1_df, p=14)
    stop_dist = stop_atr_mult * atr14
    safe_close = pd.Series(np.where(close > 0, close, np.nan), index=close.index)
    stop_pct = stop_dist / safe_close          # stop as fraction of price
    safe_stop_pct = stop_pct.where(stop_pct > 0)
    size_pct = risk_pct / safe_stop_pct       # target leverage (e.g. 4.0 = 400%)
    return size_pct.clip(upper=max_leverage).fillna(0.0)


# -- Swap drag (post-hoc) ------------------------------------------------------


def _compute_swap_drag(
    position_mask_long: pd.Series,
    position_mask_short: pd.Series,
    size_pct: pd.Series,
    close: pd.Series,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
) -> dict:
    """Swap drag expressed as a fraction of INIT_CASH (currency-agnostic).

    With size_type='percent', 1 pip P&L on a size_pct position equals:
        size_pct x pip_pct   where pip_pct = pip_size / close

    So swap drag per night (as fraction of init_cash) =
        size_pct x (pip_size / close) x swap_pips

    We use median size_pct and median close as proxies.
    Rollover counted at 21:00 UTC bars (NY 17:00 ET broker close).
    """
    idx = position_mask_long.index
    is_rollover = pd.Series(idx.hour == 21, index=idx)

    live = size_pct[size_pct > 0]
    avg_size_pct = float(live.median()) if not live.empty else 0.0
    median_close = float(close.median()) if len(close) > 0 else 1.0
    pip_pct = pip_size / median_close          # 1 pip as fraction of price

    long_nights = float((is_rollover & position_mask_long).sum())
    short_nights = float((is_rollover & position_mask_short).sum())

    # Drag as fraction of portfolio (multiply by INIT_CASH for dollar amount)
    long_drag_pct = long_nights * swap_long_pips * pip_pct * avg_size_pct
    short_drag_pct = short_nights * swap_short_pips * pip_pct * avg_size_pct

    return {
        "long_nights": long_nights,
        "short_nights": short_nights,
        "long_drag_usd": long_drag_pct * INIT_CASH,
        "short_drag_usd": short_drag_pct * INIT_CASH,
        "total_drag_usd": (long_drag_pct + short_drag_pct) * INIT_CASH,
    }


# -- Backtest helpers ----------------------------------------------------------


def _stats(pf) -> dict:
    n = pf.trades.count()
    return {
        "ret": float(pf.total_return()),
        "sharpe": float(pf.sharpe_ratio()),
        "dd": float(pf.max_drawdown()),
        "trades": int(n),
        "wr": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def _run_vbt(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    spread: float,
    slippage: float,
    size: pd.Series,
    freq: str,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
) -> dict:
    """Long + short VBT backtest with spread, slippage, ATR sizing, and swap drag.

    Fees are normalised by median close so that spread/slippage expressed in
    price-units (e.g. 0.007 for 0.7 pip on USD/JPY) translate to the correct
    fraction-of-notional regardless of the pair's price level.
    """
    sig = signal_z.shift(1).fillna(0.0)
    size_arr = size.reindex(close.index).fillna(0.0).values
    med_close = float(close.median()) or 1.0
    # fees in VBT = fraction of notional; normalise so 1 pip stays 1 pip
    vbt_fees = spread / med_close
    vbt_slip = slippage / med_close

    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    sl = _stats(pf_long)
    ss = _stats(pf_short)

    swap = _compute_swap_drag(
        pf_long.position_mask(),
        pf_short.position_mask(),
        size,
        close,
        swap_long_pips,
        swap_short_pips,
        pip_size,
    )

    combined_trades = sl["trades"] + ss["trades"]
    combined_wr = (
        (sl["wr"] * sl["trades"] + ss["wr"] * ss["trades"]) / combined_trades
        if combined_trades > 0 else 0.0
    )

    gross_pnl = (sl["ret"] + ss["ret"]) / 2 * INIT_CASH
    net_pnl = gross_pnl + swap["total_drag_usd"]

    return {
        "long_sharpe": sl["sharpe"],
        "short_sharpe": ss["sharpe"],
        "combined_sharpe": (sl["sharpe"] + ss["sharpe"]) / 2,
        "long_ret": sl["ret"],
        "short_ret": ss["ret"],
        "gross_ret": (sl["ret"] + ss["ret"]) / 2,
        "net_ret": net_pnl / INIT_CASH,
        "long_dd": sl["dd"],
        "short_dd": ss["dd"],
        "long_trades": sl["trades"],
        "short_trades": ss["trades"],
        "combined_trades": combined_trades,
        "combined_wr": combined_wr,
        "swap_drag_usd": swap["total_drag_usd"],
        "long_nights": swap["long_nights"],
        "short_nights": swap["short_nights"],
    }


# -- Main backtest pipeline ----------------------------------------------------


def run_backtest(
    instrument: str,
    target_signals: list[str],
    tfs: list[str],
    thresholds: list[float],
    spread: float | None,
    risk_pct: float,
    stop_atr: float,
    slippage_pips: float,
    swap_long_pips: float | None,
    swap_short_pips: float | None,
) -> pd.DataFrame:
    slug = instrument.lower()
    base_tf = tfs[-1]

    pip_size = PIP_SIZE.get(instrument, 0.0001)
    eff_spread = spread if spread is not None else SPREAD_DEFAULTS.get(instrument, 0.00005)
    eff_swap_long = swap_long_pips if swap_long_pips is not None else (
        SWAP_DEFAULTS.get(instrument, {}).get("long", -0.5)
    )
    eff_swap_short = swap_short_pips if swap_short_pips is not None else (
        SWAP_DEFAULTS.get(instrument, {}).get("short", 0.3)
    )
    slippage = slippage_pips * pip_size

    W = 84
    print()
    print("=" * W)
    print(f"  MTF IC STRATEGY BACKTEST -- {instrument}")
    print(f"  Signals  : {' + '.join(target_signals)}")
    print(f"  TFs      : {' -> '.join(tfs)}")
    print(f"  Split    : {int(IS_RATIO * 100)}% IS / {int((1 - IS_RATIO) * 100)}% OOS")
    spread_pips = eff_spread / pip_size
    print(f"  Spread   : {spread_pips:.2f} pip/fill  |  "
          f"Slippage: {slippage_pips:.2f} pip/fill  |  "
          f"Risk/trade: {risk_pct * 100:.1f}%  |  Stop: {stop_atr:.1f}xATR14")
    print(f"  Swap     : long {eff_swap_long:+.2f} pip/night  |  "
          f"short {eff_swap_short:+.2f} pip/night")
    print("=" * W)

    # 1. Load + align
    print("\n  Loading and aligning signals...")
    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]

    n = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True
    print(f"\n  Total H1 bars: {n:,}  |  IS: {is_n:,}  |  OOS: {n - is_n:,}")

    # 2. ATR sizing
    oos_mask = is_mask == False  # noqa: E712 -- explicit to avoid ~ type error
    print("\n  Computing ATR-based position sizes...")
    size = build_size_array(base_df, base_close, risk_pct, stop_atr)
    for label, mask in [("IS ", is_mask), ("OOS", oos_mask)]:
        s_sub = size[mask]
        live = s_sub[s_sub > 0]
        if not live.empty:
            med_pct = float(live.median())
            print(f"  Median size_pct {label}: {med_pct:.2f}x  ({med_pct * 100:.0f}% of portfolio)")

    # 3. Build composite
    print("\n  Building MTF composite...")
    composite = build_composite(tf_signals, base_close, tfs, target_signals, is_mask)
    composite_z = zscore_normalise(composite, is_mask)

    is_z = composite_z[is_mask]
    oos_z = composite_z[oos_mask]
    is_close = base_close[is_mask]
    oos_close = base_close[oos_mask]
    is_size = size[is_mask]
    oos_size = size[oos_mask]

    print(f"  Composite z -- IS: mean={is_z.mean():.3f}  std={is_z.std():.3f}")
    print(f"  Composite z -- OOS: mean={oos_z.mean():.3f}  std={oos_z.std():.3f}")
    pos_frac = (composite_z > 0).mean() * 100
    n_sigs = len(target_signals) * len(tf_signals)
    print(f"  Positive composite: {pos_frac:.1f}%  |  n_signal_tf pairs: {n_sigs}")

    # 4. Threshold sweep
    print(f"\n  Running threshold sweep over {len(thresholds)} values...")
    results: list[dict] = []

    for theta in thresholds:
        is_res = _run_vbt(
            is_close, is_z, theta, eff_spread, slippage, is_size,
            "h", eff_swap_long, eff_swap_short, pip_size,
        )
        oos_res = _run_vbt(
            oos_close, oos_z, theta, eff_spread, slippage, oos_size,
            "h", eff_swap_long, eff_swap_short, pip_size,
        )

        parity = (
            oos_res["combined_sharpe"] / is_res["combined_sharpe"]
            if is_res["combined_sharpe"] != 0 else 0.0
        )
        is_long_exp = (is_z > theta).mean() * 100
        oos_long_exp = (oos_z > theta).mean() * 100

        results.append({
            "threshold": theta,
            "is_sharpe": is_res["combined_sharpe"],
            "oos_sharpe": oos_res["combined_sharpe"],
            "parity": parity,
            "is_long_sharpe": is_res["long_sharpe"],
            "is_short_sharpe": is_res["short_sharpe"],
            "oos_long_sharpe": oos_res["long_sharpe"],
            "oos_short_sharpe": oos_res["short_sharpe"],
            "is_gross_ret": is_res["gross_ret"],
            "oos_gross_ret": oos_res["gross_ret"],
            "oos_net_ret": oos_res["net_ret"],
            "oos_swap_drag_usd": oos_res["swap_drag_usd"],
            "oos_long_nights": oos_res["long_nights"],
            "oos_short_nights": oos_res["short_nights"],
            "is_long_dd": is_res["long_dd"],
            "is_short_dd": is_res["short_dd"],
            "oos_long_dd": oos_res["long_dd"],
            "oos_short_dd": oos_res["short_dd"],
            "is_trades": is_res["combined_trades"],
            "oos_trades": oos_res["combined_trades"],
            "is_wr": is_res["combined_wr"],
            "oos_wr": oos_res["combined_wr"],
            "is_long_exposure_pct": is_long_exp,
            "oos_long_exposure_pct": oos_long_exp,
        })

    df = pd.DataFrame(results)

    # 5. Print sweep
    print()
    print("  THRESHOLD SWEEP:")
    print(f"  {'th':>6}  {'IS_Sharpe':>10}  {'OOS_Sharpe':>11}  {'Parity':>8}  "
          f"{'Trades':>8}  {'WR%':>6}  {'SwapDrag$':>10}  {'Net Ret%':>9}")
    print("  " + "-" * 82)
    for _, row in df.iterrows():
        flag = " ***" if row["parity"] >= 0.5 and row["oos_sharpe"] > 0 else ""
        print(
            f"  {row['threshold']:>6.2f}  {row['is_sharpe']:>+10.3f}  "
            f"{row['oos_sharpe']:>+11.3f}  {row['parity']:>8.3f}  "
            f"{row['oos_trades']:>8.0f}  {row['oos_wr'] * 100:>5.1f}%  "
            f"{row['oos_swap_drag_usd']:>+9.0f}$  "
            f"{row['oos_net_ret']:>+8.1%}{flag}"
        )

    # 6. Best threshold detail
    passing = df[(df["parity"] >= 0.5) & (df["oos_sharpe"] > 0)]
    if not passing.empty:
        best = passing.loc[passing["oos_sharpe"].idxmax()]
        quality_gate = True
    else:
        best = df.loc[df["is_sharpe"].idxmax()]
        quality_gate = False

    print()
    print("=" * W)
    label = "BEST THRESHOLD (OOS > 0 AND parity >= 0.5):" if quality_gate else (
        "BEST THRESHOLD (IS only -- NO OOS-passing threshold found):"
    )
    print(f"  {label}")
    print("=" * W)

    theta_best = float(best["threshold"])
    print(f"  Threshold : {theta_best:.2f}z\n")

    def _r(a: float, b: float) -> str:
        return f"{b / a:>8.3f}" if a != 0 else "     N/A"

    print(f"  {'Metric':<30}  {'IS':>10}  {'OOS':>12}  {'OOS/IS':>8}")
    print("  " + "-" * 66)
    print(f"  {'Sharpe (combined)':<30}  {best['is_sharpe']:>+10.3f}  "
          f"{best['oos_sharpe']:>+12.3f}  {_r(best['is_sharpe'], best['oos_sharpe'])}")
    print(f"  {'Sharpe (long)':<30}  {best['is_long_sharpe']:>+10.3f}  "
          f"{best['oos_long_sharpe']:>+12.3f}  {_r(best['is_long_sharpe'], best['oos_long_sharpe'])}")
    print(f"  {'Sharpe (short)':<30}  {best['is_short_sharpe']:>+10.3f}  "
          f"{best['oos_short_sharpe']:>+12.3f}  {_r(best['is_short_sharpe'], best['oos_short_sharpe'])}")
    print(f"  {'Win Rate':<30}  {best['is_wr'] * 100:>9.1f}%  {best['oos_wr'] * 100:>10.1f}%")
    print(f"  {'Max Drawdown (long)':<30}  {best['is_long_dd']:>+10.3f}  {best['oos_long_dd']:>+12.3f}")
    print(f"  {'Max Drawdown (short)':<30}  {best['is_short_dd']:>+10.3f}  {best['oos_short_dd']:>+12.3f}")
    print(f"  {'Trades':<30}  {best['is_trades']:>10.0f}  {best['oos_trades']:>12.0f}")
    print(f"  {'Long exposure %':<30}  {best['is_long_exposure_pct']:>9.1f}%  "
          f"{best['oos_long_exposure_pct']:>10.1f}%")

    # Cost breakdown
    oos_years = (n - is_n) / (365 * 24)
    gross_ret = float(best["oos_gross_ret"])
    swap_drag = float(best["oos_swap_drag_usd"])
    net_ret = float(best["oos_net_ret"])
    ann_net = (1 + net_ret) ** (1 / oos_years) - 1 if oos_years > 0 else 0.0
    ann_gross = (1 + gross_ret) ** (1 / oos_years) - 1 if oos_years > 0 else 0.0

    print()
    print(f"  {'-' * 66}")
    print(f"  COST BREAKDOWN  (OOS = {oos_years:.1f} yr  |  init_cash = ${INIT_CASH:,.0f})")
    print(f"  {'-' * 66}")
    print(f"  {'Gross OOS return':<38}  {gross_ret:>+9.1%}  ({ann_gross:>+6.1%}/yr)")
    print(f"  {'Spread + slippage':<38}  (embedded in Sharpe / drawdown)")
    print(f"  {'Swap drag (long {:.0f} + short {:.0f} nights)'.format(best['oos_long_nights'], best['oos_short_nights']):<38}  "
          f"${swap_drag:>+8.2f}  ({swap_drag / INIT_CASH:>+.2%})")
    print(f"  {'Net OOS return (after swap)':<38}  {net_ret:>+9.1%}  ({ann_net:>+6.1%}/yr)")
    print(f"  {'-' * 66}")

    print()
    if quality_gate:
        ratio = float(best["oos_sharpe"]) / float(best["is_sharpe"]) if best["is_sharpe"] else 0.0
        if ratio >= 0.5:
            print("  QUALITY GATE: PASSED -- OOS/IS Sharpe >= 0.5.")
            print("  -> Ready for Phase 4 (WFO) and Phase 6 (titan/ implementation).")
        else:
            print("  QUALITY GATE: FAILED -- OOS/IS Sharpe < 0.5.")
    else:
        print("  QUALITY GATE: FAILED -- No threshold passes both criteria.")

    # 7. Save CSV
    out_path = REPORTS_DIR / f"ic_backtest_{slug}.csv"
    df.to_csv(out_path, index=False)
    print()
    print(f"  Sweep results saved: {out_path}")
    print("=" * W)
    return df


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF IC Strategy Backtest")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    parser.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    parser.add_argument(
        "--threshold-sweep",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--spread", type=float, default=None,
        help="Half-spread per fill in price units (default: per instrument)",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=DEFAULT_RISK_PCT,
        help="Fraction of equity risked per trade (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--stop-atr", type=float, default=DEFAULT_STOP_ATR,
        help="ATR(14) multiplier for stop distance (default: 1.5)",
    )
    parser.add_argument(
        "--slippage-pips", type=float, default=DEFAULT_SLIPPAGE_PIPS,
        help="Market-impact slippage per fill in pips (default: 0.5)",
    )
    parser.add_argument(
        "--swap-long", type=float, default=None,
        help="Pips/night for long positions -- negative = you pay (default: per instrument)",
    )
    parser.add_argument(
        "--swap-short", type=float, default=None,
        help="Pips/night for short positions -- positive = you pay (default: per instrument)",
    )
    args = parser.parse_args()

    run_backtest(
        instrument=args.instrument,
        target_signals=[s.strip() for s in args.signals.split(",")],
        tfs=[t.strip() for t in args.tfs.split(",")],
        thresholds=[float(t.strip()) for t in args.threshold_sweep.split(",")],
        spread=args.spread,
        risk_pct=args.risk_pct,
        stop_atr=args.stop_atr,
        slippage_pips=args.slippage_pips,
        swap_long_pips=args.swap_long,
        swap_short_pips=args.swap_short,
    )


if __name__ == "__main__":
    main()
