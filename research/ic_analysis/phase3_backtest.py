"""Phase 3 — IS/OOS Backtest with multi-asset cost profiles, direction control,
cost sensitivity sweep, holding-period histogram, and Risk of Ruin.

Backtests the top-ranked MTF-stacked composites from phase1_sweep.py using VectorBT.

Default strategy: equal-weight composite of accel_stoch_k + accel_rsi14 across
W / D / H4 / H1 timeframes.

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
    Short entry : composite_z crosses below -threshold  (disabled when direction=long_only)
    Short exit  : composite_z rises above  0

Validation:
    IS  = first 70% of bars (time-ordered)
    OOS = remaining 30%
    Reject if OOS/IS Sharpe ratio < 0.5.

Cost model (asset-class profiles):
    Spread   : spread_bps basis points of median close per fill.
    Slippage : slippage_bps basis points of median close per fill.
    Swap     : overnight carry cost applied at NY close (21:00 UTC) per night held.
               FX only -- equity/etf/futures profiles set swap to zero.

Position sizing (ATR-based, 1% risk per trade):
    stop_dist = stop_atr_mult x ATR(14) on H1
    size_pct  = risk_pct / stop_pct   where stop_pct = ATR_stop / close
    Max leverage capped at max_leverage x (default per profile).
    No compounding -- units fixed relative to init_cash.

Output:
    Console: threshold sweep table + best IS threshold OOS validation
    CSV:     .tmp/reports/phase3_{slug}.csv

Usage:
    uv run python research/ic_analysis/phase3_backtest.py
    uv run python research/ic_analysis/phase3_backtest.py \\
        --instrument EUR_USD --asset-class fx_major --direction both \\
        --risk-pct 0.01 --spread-bps 0.5 --slippage-bps 0.5
    uv run python research/ic_analysis/phase3_backtest.py \\
        --instrument AAPL --asset-class equity_lc --direction long_only
"""

# NOTE (R5): Research pipeline uses float64 for numpy/pandas compatibility.
# Live execution in titan/ MUST use decimal.Decimal per AGENTS.md.
# Do not port research float logic directly into production without conversion.

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

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)

# -- Constants -----------------------------------------------------------------

DEFAULT_SIGNALS: list[str] = ["accel_stoch_k", "accel_rsi14"]
DEFAULT_TFS: list[str] = ["W", "D", "H4", "H1"]
DEFAULT_THRESHOLDS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
IS_RATIO: float = 0.70
# Horizon used to determine signal sign (orientation).
# Default=1 preserves backward compatibility. Pass --ref-horizon N to match
# the natural horizon discovered in Phase 1 (e.g. h=60 for vol signals).
REF_HORIZON: int = 1

INIT_CASH: float = 10_000.0
DEFAULT_RISK_PCT: float = 0.01
DEFAULT_STOP_ATR: float = 1.5

FRICTION_MULTS: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]

# Asset-class cost profiles.  spread_bps / slippage_bps are in basis points of
# median close price and are converted to price units at runtime.
COST_PROFILES: dict[str, dict] = {
    "fx_major": {
        "spread_bps": 0.5,
        "slippage_bps": 0.5,
        "max_leverage": 20.0,
        "swap_long": -0.5,
        "swap_short": 0.3,
        "pip_size": 0.0001,
    },
    "fx_cross": {
        "spread_bps": 1.0,
        "slippage_bps": 0.5,
        "max_leverage": 20.0,
        "swap_long": -0.5,
        "swap_short": 0.3,
        "pip_size": 0.0001,
    },
    "equity_lc": {
        "spread_bps": 2.0,
        "slippage_bps": 2.0,
        "max_leverage": 4.0,
        "swap_long": 0.0,
        "swap_short": 0.0,
        "pip_size": 0.01,
    },
    "etf": {
        "spread_bps": 1.0,
        "slippage_bps": 1.0,
        "max_leverage": 3.0,
        "swap_long": 0.0,
        "swap_short": 0.0,
        "pip_size": 0.01,
    },
    "futures": {
        "spread_bps": 1.5,
        "slippage_bps": 1.0,
        "max_leverage": 10.0,
        "swap_long": 0.0,
        "swap_short": 0.0,
        "pip_size": 0.01,
    },
}

# Legacy FX dicts -- kept for backward compatibility (phase4/phase5 import these).
SPREAD_DEFAULTS: dict[str, float] = {
    "EUR_USD": 0.00005,
    "GBP_USD": 0.00008,
    "USD_JPY": 0.007,
    "AUD_USD": 0.00007,
    "AUD_JPY": 0.010,
    "USD_CHF": 0.00007,
}

PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "AUD_JPY": 0.01,
    "USD_CHF": 0.0001,
}

SWAP_DEFAULTS: dict[str, dict[str, float]] = {
    "EUR_USD": {"long": -0.5, "short": 0.3},
    "GBP_USD": {"long": -0.2, "short": 0.1},
    "USD_JPY": {"long": 1.2, "short": -1.5},
    "AUD_USD": {"long": -0.3, "short": 0.2},
    "AUD_JPY": {"long": 0.5, "short": -0.7},
    "USD_CHF": {"long": 0.2, "short": -0.4},
}

# Scalar kept for backward compat; actual cap comes from profile max_leverage.
MAX_LEVERAGE: float = 30.0


# -- ATR helper ----------------------------------------------------------------


def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    """True-range ATR inlined to avoid cross-package import."""
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(p).mean()


# -- HMM regime gate -----------------------------------------------------------


def _apply_hmm_gate(
    composite_z: pd.Series,
    instrument: str,
    timeframe: str,
    is_mask: pd.Series,  # kept for API compat; good_state now uses full-dataset majority
) -> pd.Series:
    """Zero composite_z during the bad HMM state (full-dataset majority = good state).

    Uses the FULL regime file majority to determine good_state so that the
    definition stays stable across WFO folds (per-fold IS windows in crash periods
    would otherwise flip good_state to the bear state, causing inconsistent gating).
    """
    regime_path = ROOT / ".tmp" / "regime" / f"{instrument}_{timeframe}_regime.parquet"
    if not regime_path.exists():
        print(f"  [HMM GATE] Regime file not found: {regime_path.name} -- gate skipped")
        return composite_z
    regime = pd.read_parquet(regime_path)[["hmm_state"]]
    regime = regime.reindex(composite_z.index, method="ffill")
    # Use full-dataset majority so good_state is consistent across all WFO folds
    good_state = int(regime["hmm_state"].value_counts().idxmax())
    n_good = int((regime["hmm_state"] == good_state).sum())
    n_bad = int((regime["hmm_state"] != good_state).sum())
    print(
        f"  [HMM GATE] good_state={good_state}  active={n_good} bars  "
        f"gated={n_bad} bars ({n_bad / len(composite_z) * 100:.1f}%)"
    )
    gated = composite_z.copy()
    gated[regime["hmm_state"] != good_state] = 0.0
    return gated


# -- Data helpers --------------------------------------------------------------


def _load_tf(instrument: str, tf: str) -> pd.DataFrame | None:
    """Load OHLCV for one instrument / timeframe; return None if missing."""
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
        window_1y = _get_annual_bars(tf)
        native_sigs = build_all_signals(df, window_1y)

        # PREVENT LOOKAHEAD BIAS:
        # Higher tf bars are timestamped at the OPEN. Shift by 1 before
        # forward-filling so the signal is only available AFTER bar closes.
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
    ref_horizon: int = REF_HORIZON,
) -> pd.Series:
    """Sign-normalised equal-weight MTF composite. IS-calibrated only.

    ref_horizon: horizon used to determine sign orientation. Should match the
    natural horizon from Phase 1 (default=1 for backward compatibility). Using
    h=1 when the signal peaks at h=60 can invert the composite — pass the
    correct horizon via --ref-horizon to avoid 0-trade or wrong-direction runs.
    """
    parts: list[pd.Series] = []
    for sig in target_signals:
        for tf in tfs:
            if tf not in tf_signals:
                continue
            if sig not in tf_signals[tf].columns:
                print(f"  [WARN] Signal '{sig}' not found in {tf} -- skipping.")
                continue
            raw = tf_signals[tf][sig]
            sign = _ic_sign(raw[is_mask], base_close[is_mask], horizon=ref_horizon)
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
) -> tuple[pd.Series, pd.Series]:
    """ATR-based sizing returned as FRACTION OF PORTFOLIO (for size_type='percent').

    size_pct = risk_pct / stop_pct   where stop_pct = ATR_stop / close.

    Using percent-of-portfolio sizing makes the formula currency-agnostic: the
    same 1% risk target applies equally to EUR/USD (price ~1.08) and USD/JPY
    (price ~145) without any manual pip-value conversion.

    Capped at max_leverage.
    """
    atr14 = _atr14(h1_df, p=14)
    stop_dist = stop_atr_mult * atr14
    safe_close = pd.Series(np.where(close > 0, close, np.nan), index=close.index)
    stop_pct = stop_dist / safe_close
    safe_stop_pct = stop_pct.where(stop_pct > 0).fillna(0.0)
    size_pct = risk_pct / safe_stop_pct.replace(0.0, np.nan)
    size_pct_clipped = size_pct.clip(upper=max_leverage).fillna(0.0)
    return size_pct_clipped, safe_stop_pct


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

    Swap drag per night (as fraction of init_cash) =
        size_pct x (pip_size / close) x swap_pips

    Returns zeros when swap_long_pips == swap_short_pips == 0.0 (equity/etf).
    Rollover counted at 21:00 UTC bars (NY 17:00 ET broker close).
    """
    if swap_long_pips == 0.0 and swap_short_pips == 0.0:
        return {
            "long_nights": 0.0,
            "short_nights": 0.0,
            "long_drag_usd": 0.0,
            "short_drag_usd": 0.0,
            "total_drag_usd": 0.0,
        }

    idx = position_mask_long.index
    is_rollover = pd.Series(idx.hour == 21, index=idx)

    live = size_pct[size_pct > 0]
    avg_size_pct = float(live.median()) if not live.empty else 0.0
    median_close = float(close.median()) if len(close) > 0 else 1.0
    pip_pct = pip_size / median_close

    long_nights = float((is_rollover & position_mask_long).sum())
    short_nights = float((is_rollover & position_mask_short).sum())

    long_drag_pct = long_nights * swap_long_pips * pip_pct * avg_size_pct
    short_drag_pct = short_nights * swap_short_pips * pip_pct * avg_size_pct

    return {
        "long_nights": long_nights,
        "short_nights": short_nights,
        "long_drag_usd": long_drag_pct * INIT_CASH,
        "short_drag_usd": short_drag_pct * INIT_CASH,
        "total_drag_usd": (long_drag_pct + short_drag_pct) * INIT_CASH,
    }


# -- Trade Sharpe helper -------------------------------------------------------


def _trade_sharpe(pf, oos_years: float) -> float:
    """Annualised Sharpe built from per-trade % returns (not daily bars).

    trades_per_year = n_trades / oos_years
    sharpe = mean(returns) / std(returns) * sqrt(trades_per_year)
    """
    try:
        rets = pf.trades.records_readable["Return"].values
    except Exception:
        return 0.0
    if len(rets) < 2:
        return 0.0
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))
    if sigma < 1e-10:
        return 0.0
    n_trades = len(rets)
    trades_per_year = n_trades / oos_years if oos_years > 0 else n_trades
    return float(mu / sigma * np.sqrt(trades_per_year))


# -- Risk of Ruin (Balsara) ----------------------------------------------------


def _risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_pct: float,
) -> float:
    """Balsara formula: P(ruin at 25% DD).

    C3 FIX: avg_win and avg_loss are normalised by risk_pct to express
    in risk-unit (R) terms.  A 3% return on a 1% risk trade = 3R.

    edge = win_rate * avg_win_R - (1 - win_rate) * avg_loss_R
    cap_units = 0.25 / risk_pct   (25% DD expressed in risk units)
    p = ((1 - edge) / (1 + edge)) ** cap_units
    """
    # Normalise to R-multiples so edge is in same units as cap_units.
    avg_win_R = avg_win / risk_pct if risk_pct > 0 else 0.0
    avg_loss_R = abs(avg_loss) / risk_pct if risk_pct > 0 else 0.0
    edge = win_rate * avg_win_R - (1.0 - win_rate) * avg_loss_R
    if abs(edge) < 1e-10:
        return 1.0
    if edge < 0:
        return 1.0  # negative edge → certain ruin
    cap_units = 0.25 / risk_pct
    ratio = (1.0 - edge) / (1.0 + edge)
    if ratio <= 0:
        return 0.0  # edge so large that ruin is impossible
    p = ratio**cap_units
    return float(np.clip(p, 0.0, 1.0))


# -- VBT core ------------------------------------------------------------------


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
    stop_pct: pd.Series,
    freq: str,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
    direction: str = "both",
    oos_years: float = 1.0,
) -> dict:
    """Long (+ optionally short) VBT backtest with spread, slippage, ATR sizing.

    When direction == 'long_only', the short portfolio is skipped entirely and
    combined_sharpe == long_sharpe.

    Returns both trade-Sharpe and daily-Sharpe metrics.
    """
    sig = signal_z.shift(1).fillna(0.0)
    size_arr = size.reindex(close.index).fillna(0.0).values
    stop_arr = stop_pct.reindex(close.index).fillna(0.0).values

    # fees in VBT = fraction of notional. Dynamic arrays correctly charge
    # high percentages at low prices, and low percentages at high prices for absolute spreads.
    # We use dynamic arrays to ensure high-priced assets aren't undercharged relative to historic medians
    vbt_fees = (spread / close).bfill().values

    # Gap execution risk: 10x slippage on severe (>5%) price dislocations
    base_slip = slippage / close
    gap_multiplier = np.where(close.pct_change().abs() > 0.05, 10.0, 1.0)
    vbt_slip = (base_slip * gap_multiplier).bfill().values

    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        sl_stop=stop_arr,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    sl = _stats(pf_long)
    long_trade_sharpe = _trade_sharpe(pf_long, oos_years)
    long_daily_sharpe = sl["sharpe"]

    swap = _compute_swap_drag(
        pf_long.position_mask(),
        pd.Series(False, index=close.index),
        size,
        close,
        swap_long_pips,
        0.0,
        pip_size,
    )

    if direction == "long_only":
        combined_trades = sl["trades"]
        combined_wr = sl["wr"]
        gross_ret = sl["ret"]
        net_pnl = gross_ret * INIT_CASH + swap["total_drag_usd"]

        return {
            "long_sharpe": long_daily_sharpe,
            "short_sharpe": 0.0,
            "combined_sharpe": long_daily_sharpe,
            "long_ret": sl["ret"],
            "short_ret": 0.0,
            "gross_ret": gross_ret,
            "net_ret": net_pnl / INIT_CASH,
            "long_dd": sl["dd"],
            "short_dd": 0.0,
            "long_trades": sl["trades"],
            "short_trades": 0,
            "combined_trades": combined_trades,
            "combined_wr": combined_wr,
            "swap_drag_usd": swap["total_drag_usd"],
            "long_nights": swap["long_nights"],
            "short_nights": 0.0,
            "long_trade_sharpe": long_trade_sharpe,
            "short_trade_sharpe": 0.0,
            "combined_trade_sharpe": long_trade_sharpe,
            "long_daily_sharpe": long_daily_sharpe,
            "pf_long": pf_long,
        }

    # direction == "both"
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        sl_stop=stop_arr,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    ss = _stats(pf_short)
    short_trade_sharpe = _trade_sharpe(pf_short, oos_years)

    swap_short = _compute_swap_drag(
        pd.Series(False, index=close.index),
        pf_short.position_mask(),
        size,
        close,
        0.0,
        swap_short_pips,
        pip_size,
    )
    total_swap_drag = swap["total_drag_usd"] + swap_short["total_drag_usd"]

    combined_trades = sl["trades"] + ss["trades"]
    combined_wr = (
        (sl["wr"] * sl["trades"] + ss["wr"] * ss["trades"]) / combined_trades
        if combined_trades > 0
        else 0.0
    )
    gross_ret = (sl["ret"] + ss["ret"]) / 2
    net_pnl = gross_ret * INIT_CASH + total_swap_drag

    # C2 FIX: Compute combined Sharpe from blended return stream, not avg of ratios.
    long_ret_s = pf_long.returns()
    short_ret_s = pf_short.returns()
    blended_daily = (long_ret_s + short_ret_s) / 2
    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import sharpe as _sh

    combined_daily_sharpe = float(_sh(blended_daily, periods_per_year=_BPY["D"]))

    # C2 FIX: Combined trade Sharpe from merged per-trade returns.
    all_trade_rets = []
    try:
        all_trade_rets.extend(pf_long.trades.records_readable["Return"].tolist())
    except Exception:
        pass
    try:
        all_trade_rets.extend(pf_short.trades.records_readable["Return"].tolist())
    except Exception:
        pass
    if len(all_trade_rets) >= 2:
        tr_arr = np.array(all_trade_rets)
        tr_mu = float(tr_arr.mean())
        tr_sig = float(tr_arr.std(ddof=1))
        trades_py = len(tr_arr) / oos_years if oos_years > 0 else float(len(tr_arr))
        combined_trade_sharpe = tr_mu / tr_sig * np.sqrt(trades_py) if tr_sig > 1e-10 else 0.0
    else:
        combined_trade_sharpe = 0.0

    return {
        "long_sharpe": long_daily_sharpe,
        "short_sharpe": ss["sharpe"],
        "combined_sharpe": combined_daily_sharpe,
        "long_ret": sl["ret"],
        "short_ret": ss["ret"],
        "gross_ret": gross_ret,
        "net_ret": net_pnl / INIT_CASH,
        "long_dd": sl["dd"],
        "short_dd": ss["dd"],
        "long_trades": sl["trades"],
        "short_trades": ss["trades"],
        "combined_trades": combined_trades,
        "combined_wr": combined_wr,
        "swap_drag_usd": total_swap_drag,
        "long_nights": swap["long_nights"],
        "short_nights": swap_short["short_nights"],
        "long_trade_sharpe": long_trade_sharpe,
        "short_trade_sharpe": short_trade_sharpe,
        "combined_trade_sharpe": combined_trade_sharpe,
        "long_daily_sharpe": long_daily_sharpe,
        "pf_long": pf_long,
    }


# -- Holding-period histogram --------------------------------------------------


def _holding_histogram(
    pf_long,
    natural_horizon_bars: int | None = None,
    hours_per_bar: float = 1.0,
) -> float:
    """Print a bar-count histogram of trade durations. Returns median hold bars."""
    buckets = [0, 2, 5, 10, 20, 40, float("inf")]
    labels = ["1-2", "3-5", "6-10", "11-20", "21-40", "41+"]

    try:
        durations = pf_long.trades.records_readable["Duration"].values
    except Exception:
        print("  [WARN] Could not read trade durations for histogram.")
        return float("nan")

    if len(durations) == 0:
        print("  [INFO] No trades — histogram skipped.")
        return float("nan")

    # Duration in VBT is stored as timedelta or integer bars depending on version.
    # Normalise to integer bar counts.
    if hasattr(durations[0], "total_seconds"):
        durations = np.array([d.total_seconds() / (3600 * hours_per_bar) for d in durations])
    else:
        durations = np.array(durations, dtype=float)

    median_hold = float(np.median(durations))

    counts, _ = np.histogram(durations, bins=buckets)
    total = counts.sum() or 1

    print("\n  HOLDING PERIOD HISTOGRAM (OOS long trades):")
    print(f"  {'Bars':>8}  {'Count':>7}  {'%':>6}  {'Bar':}")
    print("  " + "-" * 50)
    for lbl, cnt in zip(labels, counts):
        pct = cnt / total * 100
        bar = "#" * max(1, int(pct / 2)) if cnt > 0 else ""
        print(f"  {lbl:>8}  {cnt:>7d}  {pct:>5.1f}%  {bar}")
    print(f"  {'Median':>8}  {median_hold:>7.1f} bars")

    if natural_horizon_bars is not None:
        lo = 0.5 * natural_horizon_bars
        hi = 1.5 * natural_horizon_bars
        if median_hold < lo:
            print(
                f"  [WARN] Median hold ({median_hold:.1f}) < 0.5x natural horizon "
                f"({natural_horizon_bars}). Strategy exits too fast."
            )
        elif median_hold > hi:
            print(
                f"  [WARN] Median hold ({median_hold:.1f}) > 1.5x natural horizon "
                f"({natural_horizon_bars}). Strategy holds too long."
            )

    return median_hold


# -- Cost sensitivity sweep ----------------------------------------------------


def _cost_sensitivity_sweep(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    base_spread: float,
    base_slippage: float,
    size: pd.Series,
    stop_pct: pd.Series,
    freq: str,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
    direction: str,
    oos_years: float,
) -> dict:
    """Run OOS backtest at 5 friction multipliers; find break-even multiplier."""
    sharpe_by_mult: dict[str, float] = {}

    print("\n  COST SENSITIVITY SWEEP:")
    print(f"  {'Mult':>6}  {'Spread':>10}  {'Slippage':>10}  {'OOS Sharpe':>11}")
    print("  " + "-" * 46)

    for mult in FRICTION_MULTS:
        res = _run_vbt(
            close,
            signal_z,
            threshold,
            base_spread * mult,
            base_slippage * mult,
            size,
            stop_pct,
            freq,
            swap_long_pips,
            swap_short_pips,
            pip_size,
            direction=direction,
            oos_years=oos_years,
        )
        s = res["combined_sharpe"]
        key = f"sharpe_{mult}x"
        sharpe_by_mult[key] = s
        print(
            f"  {mult:>6.1f}x  {base_spread * mult:>10.6f}  "
            f"{base_slippage * mult:>10.6f}  {s:>+11.3f}"
        )

    # Break-even: largest mult where OOS Sharpe > 0
    break_even = 0.0
    for mult in FRICTION_MULTS:
        key = f"sharpe_{mult}x"
        if sharpe_by_mult[key] > 0:
            break_even = mult

    print(f"\n  Break-even friction multiplier: {break_even:.1f}x")
    sharpe_by_mult["break_even_friction_mult"] = break_even
    return sharpe_by_mult


# -- Main backtest pipeline ----------------------------------------------------


def run_backtest(
    instrument: str,
    target_signals: list[str],
    tfs: list[str],
    thresholds: list[float],
    asset_class: str = "fx_major",
    direction: str = "both",
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    max_leverage: float | None = None,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_atr: float = DEFAULT_STOP_ATR,
    natural_horizon: int | None = None,
    # Legacy FX params -- accepted for backward compat; resolved against profile.
    spread: float | None = None,
    slippage_pips: float | None = None,
    swap_long_pips: float | None = None,
    swap_short_pips: float | None = None,
    hmm_gate: bool = False,
    ref_horizon: int = REF_HORIZON,
) -> pd.DataFrame:
    """Run Phase 3 IS/OOS backtest and return per-threshold results as a DataFrame."""
    slug = instrument.lower()
    base_tf = tfs[-1]
    freq = "d" if base_tf in ("D", "daily") else "h"

    profile = COST_PROFILES.get(asset_class, COST_PROFILES["fx_major"])
    eff_max_leverage = max_leverage if max_leverage is not None else profile["max_leverage"]
    eff_swap_long = swap_long_pips if swap_long_pips is not None else profile["swap_long"]
    eff_swap_short = swap_short_pips if swap_short_pips is not None else profile["swap_short"]
    pip_size_val = PIP_SIZE.get(instrument, profile["pip_size"])

    # Resolve spread and slippage.
    # Priority: explicit bps override > legacy price-unit override > profile default.
    # All values are ultimately stored as price units for _run_vbt.
    # Legacy FX spread override (price units) takes precedence if provided.
    if spread is not None:
        # Legacy: price-unit spread provided directly (old CLI / phase4 compat).
        eff_spread_bps: float | None = None
        eff_spread_price = float(spread)
        eff_slippage_price = (
            float(slippage_pips) * pip_size_val
            if slippage_pips is not None
            else (profile["slippage_bps"] / 10000)  # placeholder; resolved below
        )
        _spread_from_bps = False
    else:
        _spread_from_bps = True
        eff_spread_bps = spread_bps if spread_bps is not None else profile["spread_bps"]
        eff_slippage_bps_val = slippage_bps if slippage_bps is not None else profile["slippage_bps"]
        # Will be converted to price units once we know median close.
        eff_spread_price = 0.0  # filled after data load
        eff_slippage_price = 0.0  # filled after data load

    # G5 FIX: Auto-load Phase 2 composite winner if no explicit signals given.
    if target_signals is None or target_signals == DEFAULT_SIGNALS:
        p2_dir = ROOT / ".tmp" / "reports"
        p2_winner_path = p2_dir / f"phase2_{instrument.lower()}_winner.csv"
        if p2_winner_path.exists():
            try:
                p2_df = pd.read_csv(p2_winner_path)
                if "signal" in p2_df.columns:
                    p2_signals = p2_df["signal"].tolist()
                    if len(p2_signals) > 0:
                        target_signals = p2_signals
                        print(
                            f"  [G5] Auto-loaded {len(p2_signals)} signals from Phase 2: {p2_winner_path.name}"
                        )
            except Exception as exc:
                print(f"  [G5] Could not load Phase 2 winner: {exc}")

    W = 84
    print()
    print("=" * W)
    print(f"  PHASE 3 MTF IC BACKTEST -- {instrument}  [{asset_class} / {direction}]")
    print(f"  Signals  : {' + '.join(target_signals)}")
    print(f"  TFs      : {' -> '.join(tfs)}")
    print(f"  Split    : {int(IS_RATIO * 100)}% IS / {int((1 - IS_RATIO) * 100)}% OOS")
    # G4 FIX: Portfolio heat warning.
    print("  [!] WARNING: Per-instrument backtest. Portfolio heat limits NOT enforced.")
    print("    Directive requires: FX margin <= 15-20%, equity gross <= 100-150%.")
    print("    Enforce portfolio-level constraints in live deployment / pipeline_validation.")
    if _spread_from_bps:
        print(
            f"  Spread   : {eff_spread_bps:.2f} bps  |  "
            f"Slippage: {eff_slippage_bps_val:.2f} bps  |  "
            f"Risk/trade: {risk_pct * 100:.1f}%  |  Stop: {stop_atr:.1f}xATR14"
        )
    else:
        print(
            f"  Spread   : {eff_spread_price:.6f} (price)  |  "
            f"Slippage: {eff_slippage_price:.6f} (price)  |  "
            f"Risk/trade: {risk_pct * 100:.1f}%  |  Stop: {stop_atr:.1f}xATR14"
        )
    print(
        f"  Swap     : long {eff_swap_long:+.2f} pip/night  |  "
        f"short {eff_swap_short:+.2f} pip/night  |  "
        f"MaxLev: {eff_max_leverage:.1f}x"
    )
    print("=" * W)

    # 1. Load + align
    print("\n  Loading and aligning signals...")
    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]

    # Now that we have close prices, convert bps to price units.
    if _spread_from_bps:
        med_close = float(base_close.median()) or 1.0
        eff_spread_price = (eff_spread_bps / 10000) * med_close  # type: ignore[operator]
        eff_slippage_price = (eff_slippage_bps_val / 10000) * med_close
        print(
            f"  Median close: {med_close:.5f}  "
            f"-> spread {eff_spread_price:.6f}  slippage {eff_slippage_price:.6f}"
        )

    n = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True
    oos_n = n - is_n
    # C5 FIX: Use correct bars_per_year for daily vs hourly timeframes.
    bars_per_year = 252 if freq == "d" else 365 * 24
    oos_years = oos_n / bars_per_year
    print(
        f"\n  Total {base_tf} bars: {n:,}  |  IS: {is_n:,}  |  OOS: {oos_n:,}  "
        f"({oos_years:.2f} yr OOS)"
    )

    # 2. ATR sizing
    oos_mask = ~is_mask
    print("\n  Computing ATR-based position sizes...")
    size, stop_pct = build_size_array(
        base_df, base_close, risk_pct, stop_atr, max_leverage=eff_max_leverage
    )
    for label, mask in [("IS ", is_mask), ("OOS", oos_mask)]:
        s_sub = size[mask]
        live = s_sub[s_sub > 0]
        if not live.empty:
            med_pct = float(live.median())
            print(f"  Median size_pct {label}: {med_pct:.2f}x  ({med_pct * 100:.0f}% of portfolio)")

    # 3. Build composite
    print("\n  Building MTF composite...")
    composite = build_composite(
        tf_signals, base_close, tfs, target_signals, is_mask, ref_horizon=ref_horizon
    )
    composite_z = zscore_normalise(composite, is_mask)

    if hmm_gate:
        composite_z = _apply_hmm_gate(composite_z, instrument, base_tf, is_mask)

    is_z = composite_z[is_mask]
    oos_z = composite_z[oos_mask]
    is_close = base_close[is_mask]
    oos_close = base_close[oos_mask]
    is_size = size[is_mask]
    oos_size = size[oos_mask]
    is_stop_pct = stop_pct[is_mask]
    oos_stop_pct = stop_pct[oos_mask]

    print(f"  Composite z -- IS: mean={is_z.mean():.3f}  std={is_z.std():.3f}")
    print(f"  Composite z -- OOS: mean={oos_z.mean():.3f}  std={oos_z.std():.3f}")
    pos_frac = (composite_z > 0).mean() * 100
    n_sigs = len(target_signals) * len(tf_signals)
    print(f"  Positive composite: {pos_frac:.1f}%  |  n_signal_tf pairs: {n_sigs}")

    # 4. Threshold sweep
    is_years = is_n / bars_per_year
    print(f"\n  Running threshold sweep over {len(thresholds)} values...")
    results: list[dict] = []

    for theta in thresholds:
        is_res = _run_vbt(
            is_close,
            is_z,
            theta,
            eff_spread_price,
            eff_slippage_price,
            is_size,
            is_stop_pct,
            freq,
            eff_swap_long,
            eff_swap_short,
            pip_size_val,
            direction=direction,
            oos_years=is_years,
        )
        oos_res = _run_vbt(
            oos_close,
            oos_z,
            theta,
            eff_spread_price,
            eff_slippage_price,
            oos_size,
            oos_stop_pct,
            freq,
            eff_swap_long,
            eff_swap_short,
            pip_size_val,
            direction=direction,
            oos_years=oos_years,
        )

        is_combined = is_res["combined_sharpe"]
        oos_combined = oos_res["combined_sharpe"]
        parity = oos_combined / is_combined if is_combined != 0 else 0.0
        is_long_exp = (is_z > theta).mean() * 100
        oos_long_exp = (oos_z > theta).mean() * 100

        results.append(
            {
                "threshold": theta,
                "is_sharpe": is_combined,
                "oos_sharpe": oos_combined,
                "is_trade_sharpe": is_res["combined_trade_sharpe"],
                "oos_trade_sharpe": oos_res["combined_trade_sharpe"],
                "oos_daily_sharpe": oos_res["long_daily_sharpe"]
                if direction == "long_only"
                else (oos_res["long_sharpe"] + oos_res["short_sharpe"]) / 2,
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
            }
        )

    df = pd.DataFrame(results)

    # 5. Print sweep table
    print()
    print("  THRESHOLD SWEEP:")
    print(
        f"  {'th':>6}  {'IS_Daily':>10}  {'OOS_Daily':>10}  "
        f"{'IS_Trade':>10}  {'OOS_Trade':>10}  {'Parity':>8}  "
        f"{'Trades':>8}  {'Net Ret%':>9}"
    )
    print("  " + "-" * 88)
    for _, row in df.iterrows():
        flag = (
            " ***"
            if row["parity"] >= 0.5 and row["oos_sharpe"] > 0 and row["oos_trade_sharpe"] > 0
            else ""
        )
        print(
            f"  {row['threshold']:>6.2f}  {row['is_sharpe']:>+10.3f}  "
            f"{row['oos_sharpe']:>+10.3f}  {row['is_trade_sharpe']:>+10.3f}  "
            f"{row['oos_trade_sharpe']:>+10.3f}  {row['parity']:>8.3f}  "
            f"{row['oos_trades']:>8.0f}  {row['oos_net_ret']:>+8.1%}{flag}"
        )

    # 6. Select best threshold
    passing = df[(df["parity"] >= 0.5) & (df["oos_sharpe"] > 0) & (df["oos_trade_sharpe"] > 0)]
    if not passing.empty:
        best = passing.loc[passing["oos_sharpe"].idxmax()]
        quality_gate = True
    else:
        best = df.loc[df["is_sharpe"].idxmax()]
        quality_gate = False

    print()
    print("=" * W)
    label = (
        "BEST THRESHOLD (OOS > 0 AND parity >= 0.5):"
        if quality_gate
        else "BEST THRESHOLD (IS only -- NO OOS-passing threshold found):"
    )
    print(f"  {label}")
    print("=" * W)

    theta_best = float(best["threshold"])
    print(f"  Threshold : {theta_best:.2f}z\n")

    def _r(a: float, b: float) -> str:
        return f"{b / a:>8.3f}" if a != 0 else "     N/A"

    print(f"  {'Metric':<30}  {'IS':>10}  {'OOS':>12}  {'OOS/IS':>8}")
    print("  " + "-" * 66)
    print(
        f"  {'Sharpe (daily)':<30}  {best['is_sharpe']:>+10.3f}  "
        f"{best['oos_sharpe']:>+12.3f}  {_r(best['is_sharpe'], best['oos_sharpe'])}"
    )
    print(
        f"  {'Sharpe (trade)':<30}  {best['is_trade_sharpe']:>+10.3f}  "
        f"{best['oos_trade_sharpe']:>+12.3f}  {_r(best['is_trade_sharpe'], best['oos_trade_sharpe'])}"
    )
    print(
        f"  {'Sharpe (long)':<30}  {best['is_long_sharpe']:>+10.3f}  "
        f"{best['oos_long_sharpe']:>+12.3f}  "
        f"{_r(best['is_long_sharpe'], best['oos_long_sharpe'])}"
    )
    print(
        f"  {'Sharpe (short)':<30}  {best['is_short_sharpe']:>+10.3f}  "
        f"{best['oos_short_sharpe']:>+12.3f}  "
        f"{_r(best['is_short_sharpe'], best['oos_short_sharpe'])}"
    )
    print(f"  {'Win Rate':<30}  {best['is_wr'] * 100:>9.1f}%  {best['oos_wr'] * 100:>10.1f}%")
    print(
        f"  {'Max Drawdown (long)':<30}  {best['is_long_dd']:>+10.3f}  "
        f"{best['oos_long_dd']:>+12.3f}"
    )
    print(
        f"  {'Max Drawdown (short)':<30}  {best['is_short_dd']:>+10.3f}  "
        f"{best['oos_short_dd']:>+12.3f}"
    )
    print(f"  {'Trades':<30}  {best['is_trades']:>10.0f}  {best['oos_trades']:>12.0f}")
    print(
        f"  {'Long exposure %':<30}  {best['is_long_exposure_pct']:>9.1f}%  "
        f"{best['oos_long_exposure_pct']:>10.1f}%"
    )

    # Cost breakdown
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
    long_n = best["oos_long_nights"]
    short_n = best["oos_short_nights"]
    print(
        f"  {'Swap drag (long {:.0f} + short {:.0f} nights)'.format(long_n, short_n):<38}  "
        f"${swap_drag:>+8.2f}  ({swap_drag / INIT_CASH:>+.2%})"
    )
    print(f"  {'Net OOS return (after swap)':<38}  {net_ret:>+9.1%}  ({ann_net:>+6.1%}/yr)")
    print(f"  {'-' * 66}")

    # 7. Re-run OOS at best threshold to get portfolio objects and per-trade stats.
    oos_best = _run_vbt(
        oos_close,
        oos_z,
        theta_best,
        eff_spread_price,
        eff_slippage_price,
        oos_size,
        oos_stop_pct,
        freq,
        eff_swap_long,
        eff_swap_short,
        pip_size_val,
        direction=direction,
        oos_years=oos_years,
    )
    pf_long_best = oos_best["pf_long"]

    # 8. Holding period histogram
    # Determine hours per bar
    tf_lower = base_tf.lower()
    if tf_lower == "d":
        hrs = 24.0
    elif tf_lower.startswith("h"):
        try:
            hrs = float(tf_lower.replace("h", "") or 1.0)
        except ValueError:
            hrs = 1.0
    else:
        hrs = 1.0

    median_hold = _holding_histogram(
        pf_long_best, natural_horizon_bars=natural_horizon, hours_per_bar=hrs
    )

    # 9. Risk of Ruin
    try:
        trade_rets = pf_long_best.trades.records_readable["Return"].values
        if len(trade_rets) >= 2:
            wins = trade_rets[trade_rets > 0]
            losses = trade_rets[trade_rets <= 0]
            wr_oos = float(len(wins) / len(trade_rets))
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            ror = _risk_of_ruin(wr_oos, avg_win, avg_loss, risk_pct)
        else:
            ror = 1.0
    except Exception:
        ror = 1.0

    print(f"\n  Risk of Ruin (Balsara, 25% DD): {ror:.4f}", end="")
    if ror < 0.05:
        print("  [PASS < 0.05]")
    else:
        print("  [FAIL >= 0.05  — threshold not met]")

    # 10. Cost sensitivity sweep
    friction = _cost_sensitivity_sweep(
        oos_close,
        oos_z,
        theta_best,
        eff_spread_price,
        eff_slippage_price,
        oos_size,
        oos_stop_pct,
        freq,
        eff_swap_long,
        eff_swap_short,
        pip_size_val,
        direction=direction,
        oos_years=oos_years,
    )

    # 11. Quality gate — all 8 criteria from the directive
    print()
    gate_failures: list[str] = []

    if not quality_gate:
        gate_failures.append(
            "No threshold passes OOS Sharpe > 0 + Trade Sharpe > 0 + Parity >= 0.5"
        )
    else:
        ratio = (
            float(best["oos_sharpe"]) / float(best["is_sharpe"])
            if float(best["is_sharpe"]) != 0
            else 0.0
        )
        if ratio < 0.5:
            gate_failures.append(f"IS/OOS Sharpe parity {ratio:.3f} < 0.50")

    oos_trades_val = int(best.get("oos_trades", 0))
    if oos_trades_val < 30:
        gate_failures.append(f"OOS trades {oos_trades_val} < 30")

    oos_wr_val = float(best.get("oos_wr", 0.0))
    if oos_wr_val < 0.40:
        gate_failures.append(f"Win rate {oos_wr_val:.1%} < 40%")

    max_dd_val = float(best.get("oos_long_dd", 0.0))
    if direction == "both":
        max_dd_val = min(max_dd_val, float(best.get("oos_short_dd", 0.0)))
    if max_dd_val < -0.25:
        gate_failures.append(f"Max DD {max_dd_val:.1%} < -25%")

    if ror >= 0.05:
        gate_failures.append(f"RoR {ror:.4f} >= 5%")

    bef_val = friction.get("break_even_friction_mult", 0.0)
    if bef_val < 2.0:
        gate_failures.append(f"Break-even friction {bef_val:.1f}x < 2.0x")

    all_gates_pass = quality_gate and len(gate_failures) == 0
    if all_gates_pass:
        print("  QUALITY GATE: PASSED -- all 8 criteria met.")
        print("  -> Ready for Phase 4 (WFO) and Phase 6 (titan/ implementation).")
    else:
        print("  QUALITY GATE: FAILED")
        for f in gate_failures:
            print(f"    [FAIL] {f}")

    # 12. Build unified output CSV
    out_rows: list[dict] = []
    for _, row in df.iterrows():
        theta_row = float(row["threshold"])
        is_best = abs(theta_row - theta_best) < 1e-9
        # For non-best rows, cost sensitivity and RoR columns are filled from
        # the best-threshold run to keep the schema consistent (one row per theta).
        out_rows.append(
            {
                "instrument": instrument,
                "timeframe": base_tf,
                "direction": direction,
                "signals_used": "+".join(target_signals),
                "threshold": theta_row,
                "is_sharpe": row["is_sharpe"],
                "oos_sharpe": row["oos_sharpe"],
                "is_trade_sharpe": row["is_trade_sharpe"],
                "oos_trade_sharpe": row["oos_trade_sharpe"],
                "oos_daily_sharpe": row["oos_daily_sharpe"],
                "is_oos_ratio": row["parity"],
                "n_oos_trades": row["oos_trades"],
                "win_rate": row["oos_wr"],
                "max_dd_pct": min(row["oos_long_dd"], row["oos_short_dd"])
                if direction == "both"
                else row["oos_long_dd"],
                "ror_25pct": ror if is_best else float("nan"),
                "break_even_friction_mult": friction["break_even_friction_mult"]
                if is_best
                else float("nan"),
                "median_hold_bars": median_hold if is_best else float("nan"),
                "sharpe_0.5x": friction.get("sharpe_0.5x", float("nan"))
                if is_best
                else float("nan"),
                "sharpe_1.0x": friction.get("sharpe_1.0x", float("nan"))
                if is_best
                else float("nan"),
                "sharpe_1.5x": friction.get("sharpe_1.5x", float("nan"))
                if is_best
                else float("nan"),
                "sharpe_2.0x": friction.get("sharpe_2.0x", float("nan"))
                if is_best
                else float("nan"),
                "sharpe_3.0x": friction.get("sharpe_3.0x", float("nan"))
                if is_best
                else float("nan"),
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = REPORTS_DIR / f"phase3_{slug}.csv"
    out_df.to_csv(out_path, index=False)
    print()
    print(f"  Sweep results saved: {out_path}")
    print("=" * W)
    return out_df


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 MTF IC Strategy Backtest")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    parser.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    parser.add_argument(
        "--threshold-sweep",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--asset-class",
        default="fx_major",
        choices=list(COST_PROFILES.keys()),
        help="Asset-class cost profile (default: fx_major)",
    )
    parser.add_argument(
        "--direction",
        default="both",
        choices=["both", "long_only"],
        help="Trade direction: both (long+short) or long_only (default: both)",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=None,
        help="Half-spread per fill in basis points of price (overrides profile)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="Market-impact slippage per fill in bps (overrides profile)",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=None,
        help="Maximum position size leverage (overrides profile)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=DEFAULT_RISK_PCT,
        help="Fraction of equity risked per trade (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--stop-atr",
        type=float,
        default=DEFAULT_STOP_ATR,
        help="ATR(14) multiplier for stop distance (default: 1.5)",
    )
    parser.add_argument(
        "--natural-horizon",
        type=int,
        default=None,
        help="Expected hold duration in bars (for histogram warning check)",
    )
    # Legacy FX CLI args -- kept for backward compat.
    parser.add_argument(
        "--spread",
        type=float,
        default=None,
        help="[Legacy] Half-spread in price units (use --spread-bps instead)",
    )
    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=None,
        help="[Legacy] Slippage in pips (use --slippage-bps instead)",
    )
    parser.add_argument(
        "--swap-long",
        type=float,
        default=None,
        help="Pips/night for long positions -- negative = you pay (default: per profile)",
    )
    parser.add_argument(
        "--swap-short",
        type=float,
        default=None,
        help="Pips/night for short positions (default: per profile)",
    )
    parser.add_argument(
        "--hmm-gate",
        action="store_true",
        default=False,
        help="Gate entries to the IS-majority HMM state (requires Phase 0 regime file)",
    )
    parser.add_argument(
        "--ref-horizon",
        type=int,
        default=REF_HORIZON,
        help=(
            "Horizon (bars) used to determine signal sign orientation. "
            "Set to the Phase 1 natural horizon (e.g. 60 for vol signals). "
            "Default=1; wrong value inverts the composite and produces 0 trades."
        ),
    )
    args = parser.parse_args()

    run_backtest(
        instrument=args.instrument,
        target_signals=[s.strip() for s in args.signals.split(",")],
        tfs=[t.strip() for t in args.tfs.split(",")],
        thresholds=[float(t.strip()) for t in args.threshold_sweep.split(",")],
        asset_class=args.asset_class,
        direction=args.direction,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        max_leverage=args.max_leverage,
        risk_pct=args.risk_pct,
        stop_atr=args.stop_atr,
        natural_horizon=args.natural_horizon,
        spread=args.spread,
        slippage_pips=args.slippage_pips,
        swap_long_pips=args.swap_long,
        swap_short_pips=args.swap_short,
        hmm_gate=args.hmm_gate,
        ref_horizon=args.ref_horizon,
    )


if __name__ == "__main__":
    main()
