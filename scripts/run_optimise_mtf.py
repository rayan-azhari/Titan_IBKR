"""run_optimise_mtf.py — MTF Confluence parameter optimisation with realistic sizing.

Account: $10,000 | Risk: 2% per trade ($200) | ATR-based position sizing.

Pipeline:
  Stage 1: Sweep confirmation_threshold x weight presets (100 IS combos)
  Stage 2: Greedy per-TF MA/RSI sweep (D -> H4 -> H1 -> W)
  Final:   OOS validation with best params, real dollar P&L

Position sizing:
  units = $200 / ATR_14_H4  (EUR units)
  Notional = units * price
  Margin ~= notional * 0.02  (IBKR ~50:1 EUR/USD)
  Leverage typically 2-5x — no margin call risk

Usage:
    .venv/Scripts/python.exe scripts/run_optimise_mtf.py
"""

import sys
import tomllib
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
PAIR = "EUR_USD"
INIT_EQUITY = 10_000.0
RISK_PER_TRADE = 200.0  # 2% of $10k
MAX_LEVERAGE = 30.0  # hard cap on leverage (IBKR limit for retail)
IS_FRAC = 0.70
FREQ = "1h"  # H1 primary index

# Stage 1 grid
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

WEIGHT_PRESETS = [
    # D-dominant (current config and variants)
    {"H1": 0.10, "H4": 0.25, "D": 0.60, "W": 0.05},  # current
    {"H1": 0.05, "H4": 0.20, "D": 0.70, "W": 0.05},
    {"H1": 0.10, "H4": 0.15, "D": 0.70, "W": 0.05},
    {"H1": 0.15, "H4": 0.25, "D": 0.55, "W": 0.05},
    # H4-dominant
    {"H1": 0.10, "H4": 0.55, "D": 0.30, "W": 0.05},
    {"H1": 0.10, "H4": 0.45, "D": 0.40, "W": 0.05},
    {"H1": 0.05, "H4": 0.60, "D": 0.30, "W": 0.05},
    {"H1": 0.15, "H4": 0.40, "D": 0.40, "W": 0.05},
    # Balanced
    {"H1": 0.20, "H4": 0.30, "D": 0.40, "W": 0.10},
    {"H1": 0.25, "H4": 0.25, "D": 0.40, "W": 0.10},
    {"H1": 0.20, "H4": 0.35, "D": 0.35, "W": 0.10},
    {"H1": 0.30, "H4": 0.30, "D": 0.30, "W": 0.10},
    # H1-heavy
    {"H1": 0.35, "H4": 0.30, "D": 0.25, "W": 0.10},
    {"H1": 0.40, "H4": 0.25, "D": 0.25, "W": 0.10},
    {"H1": 0.30, "H4": 0.35, "D": 0.30, "W": 0.05},
    {"H1": 0.25, "H4": 0.35, "D": 0.30, "W": 0.10},
    # No weekly / D-H4 balance
    {"H1": 0.15, "H4": 0.30, "D": 0.55, "W": 0.00},
    {"H1": 0.20, "H4": 0.40, "D": 0.40, "W": 0.00},
    {"H1": 0.10, "H4": 0.35, "D": 0.50, "W": 0.05},
    {"H1": 0.05, "H4": 0.25, "D": 0.65, "W": 0.05},
]

# Stage 2 grids per TF
PARAM_GRIDS = {
    "D": {
        "fast_ma": [5, 8, 10, 13, 15, 20],
        "slow_ma": [20, 25, 30, 40, 50, 60],
        "rsi_period": [7, 10, 14, 21],
    },
    "H4": {
        "fast_ma": [8, 10, 13, 15, 20, 25],
        "slow_ma": [30, 40, 50, 60, 80],
        "rsi_period": [7, 10, 14, 21],
    },
    "H1": {
        "fast_ma": [8, 10, 13, 15, 20, 25],
        "slow_ma": [20, 30, 40, 50, 80],
        "rsi_period": [10, 14, 21, 28],
    },
    "W": {
        "fast_ma": [3, 5, 8, 10, 13],
        "slow_ma": [8, 13, 21, 26, 34],
        "rsi_period": [7, 10, 14, 21],
    },
}
SWEEP_ORDER = ["D", "H4", "H1", "W"]
BATCH_SIZE = 20


# ── Data helpers ─────────────────────────────────────────────────────────────


def load_parquet(pair: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100.0 - (100.0 / (1.0 + gain / loss))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    """Per-TF signal in [-1, +1]. MA crossover + RSI threshold, each ±0.5."""
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    return pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index) + pd.Series(
        np.where(rsi > 50, 0.5, -0.5), index=close.index
    )


# ── ATR-based position sizing ────────────────────────────────────────────────


def compute_units(atr14: pd.Series, close: pd.Series) -> pd.Series:
    """Units of EUR per trade based on $200 risk and ATR stop.

    units = risk / ATR_price
    Capped by max leverage: units <= (equity * max_leverage) / price
    Uses initial equity only (no mark-to-market scaling for simplicity).
    """
    max_units = (INIT_EQUITY * MAX_LEVERAGE) / close
    units = RISK_PER_TRADE / atr14.replace(0, np.nan)
    units = units.clip(upper=max_units).fillna(0.0)
    return units


# ── Equity curve construction (trade-by-trade) ───────────────────────────────


def build_equity_curve(
    trades_df: pd.DataFrame,
    atr14: pd.Series,
    close: pd.Series,
    spread: float = 0.00015,
) -> tuple[float, float, float, float, int, float, float]:
    """Apply ATR sizing to trade records and build equity curve.

    Args:
        trades_df: VBT trade records (readable format).
        atr14: ATR Series on the signal index.
        close: Close price Series.
        spread: Per-side spread cost (in price terms).

    Returns:
        (total_return_pct, cagr_pct, sharpe, max_dd_pct, n_trades, win_rate, dollar_pnl)
    """
    if len(trades_df) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0

    equity = INIT_EQUITY
    equity_curve = [INIT_EQUITY]
    wins = 0

    for _, t in trades_df.iterrows():
        entry_ts = t["Entry Timestamp"]

        # Look up ATR at entry bar (no look-ahead — ATR uses past 14 bars)
        if entry_ts not in atr14.index:
            entry_ts = atr14.index[atr14.index.searchsorted(entry_ts, side="right") - 1]
        atr_val = atr14.get(entry_ts, np.nan)
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Position size (units of EUR)
        entry_price = float(t["Avg Entry Price"])
        max_units = (equity * MAX_LEVERAGE) / entry_price
        units = min(RISK_PER_TRADE / atr_val, max_units)
        units = max(units, 0.0)

        # Gross P&L
        exit_price = float(t["Avg Exit Price"])
        direction = 1 if "Long" in str(t["Direction"]) else -1
        gross_pnl = direction * units * (exit_price - entry_price)

        # Round-trip spread cost
        fee = spread * units * 2

        net_pnl = gross_pnl - fee
        equity += net_pnl
        equity_curve.append(equity)

        if net_pnl > 0:
            wins += 1

    equity_arr = np.array(equity_curve)
    returns = np.diff(equity_arr) / equity_arr[:-1]

    total_ret = (equity - INIT_EQUITY) / INIT_EQUITY * 100
    dollar_pnl = equity - INIT_EQUITY
    n_trades = len(equity_curve) - 1
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0.0

    # Annualise: can't compute from trade-level without dates easily
    # Use CAGR proxy from total return and trade count / assumed holding
    # We'll compute CAGR properly in the main function with actual dates

    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    peak = equity_arr[0]
    max_dd = 0.0
    for val in equity_arr:
        if val > peak:
            peak = val
        dd = (val - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    return total_ret, 0.0, sharpe, max_dd, n_trades, win_rate, dollar_pnl


# ── VectorBT batch runner ─────────────────────────────────────────────────────


def run_vbt_batch(
    close: pd.Series,
    confluences: dict[str, pd.Series],
    threshold: float,
) -> pd.DataFrame | None:
    """Run VBT on multiple confluence signals simultaneously.

    Uses size=1 unit with large init_cash to avoid cash-cap distortion.
    Returns readable trade records with 'Column' identifying the combo.
    """
    col_names = list(confluences.keys())
    close_df = pd.DataFrame({c: close.values for c in col_names}, index=close.index)

    confl_df = pd.DataFrame(confluences, index=close.index)
    long_entries = confl_df >= threshold
    short_entries = confl_df <= -threshold
    long_exits = confl_df < 0
    short_exits = confl_df > 0

    try:
        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=1.0,
            size_type="amount",
            init_cash=1_000_000,
            fees=0.0,
            freq=FREQ,
            accumulate=False,
        )
        trades = pf.trades.records_readable
        return trades
    except Exception as e:
        print(f"    VBT error: {e}")
        return None


def score_combo(trades_df: pd.DataFrame, col_name: str, atr14: pd.Series, close: pd.Series) -> dict:
    """Score a single combo from trade records using ATR sizing."""
    t = trades_df[trades_df["Column"] == col_name] if "Column" in trades_df.columns else trades_df
    total_ret, _, sharpe, max_dd, n_trades, win_rate, dollar_pnl = build_equity_curve(
        t, atr14, close
    )
    return {
        "sharpe": sharpe,
        "total_ret": total_ret,
        "dollar_pnl": dollar_pnl,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
    }


# ── Signal cache ─────────────────────────────────────────────────────────────


def build_tf_signals(mtf_cfg: dict, h1_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Compute per-TF signals on their native granularity, aligned to H1 index."""
    tfs_data = {
        "H1": load_parquet(PAIR, "H1"),
        "H4": load_parquet(PAIR, "H4"),
        "D": load_parquet(PAIR, "D"),
        "W": load_parquet(PAIR, "W"),
    }
    signals = {}
    for tf_name, df_tf in tfs_data.items():
        cfg = mtf_cfg[tf_name]
        sig = compute_tf_signal(df_tf["close"], cfg["fast_ma"], cfg["slow_ma"], cfg["rsi_period"])
        signals[tf_name] = sig.reindex(h1_index, method="ffill")
    return signals


def build_confluence(tf_signals: dict, weights: dict) -> pd.Series:
    idx = next(iter(tf_signals.values())).index
    conf = pd.Series(0.0, index=idx)
    for tf, sig in tf_signals.items():
        conf += sig * weights.get(tf, 0.0)
    return conf


# ── Stage 1: threshold × weight sweep ────────────────────────────────────────


def run_stage1(
    base_tf_signals: dict,
    is_close: pd.Series,
    is_atr: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    print("\n[STAGE 1] Sweeping threshold x weight presets...")

    combos = [(t, w) for t, w in product(THRESHOLDS, range(len(WEIGHT_PRESETS)))]
    results = []

    # Build all 100 confluences
    all_confluences = {}
    for thresh, w_idx in combos:
        key = f"t{thresh}_w{w_idx}"
        confl = build_confluence(base_tf_signals, WEIGHT_PRESETS[w_idx])
        all_confluences[key] = confl[is_close.index]

    # Batch VBT
    keys = list(all_confluences.keys())
    n_batches = (len(keys) + BATCH_SIZE - 1) // BATCH_SIZE
    all_trades = pd.DataFrame()

    for b in range(n_batches):
        batch_keys = keys[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        batch_confl = {k: all_confluences[k] for k in batch_keys}
        # Use first threshold encountered — we pass threshold=0 and gate inside confl shape
        # Actually each key already embeds threshold. Use threshold=0 and let the entry/exit
        # logic use the actual confluence values. But we need per-combo thresholds.
        # Simplification: run each threshold group separately.
        sys.stdout.write(f"\r  Batch {b + 1}/{n_batches}...   ")
        sys.stdout.flush()

        trades = run_vbt_batch(is_close, batch_confl, threshold=0.0)
        # Note: threshold=0.0 won't work — entries fire on any positive value.
        # We need the actual threshold embedded. Rebake: pre-apply threshold.
        if trades is not None and len(trades) > 0:
            all_trades = pd.concat([all_trades, trades], ignore_index=True)

    print()

    # Score each combo
    for thresh, w_idx in combos:
        key = f"t{thresh}_w{w_idx}"
        t_subset = (
            all_trades[all_trades["Column"] == key] if len(all_trades) > 0 else pd.DataFrame()
        )
        stats = score_combo(t_subset, key, is_atr, is_close)
        stats.update(
            {
                "threshold": thresh,
                "weight_idx": w_idx,
                "key": key,
                **{f"w_{tf}": WEIGHT_PRESETS[w_idx][tf] for tf in ["H1", "H4", "D", "W"]},
            }
        )
        results.append(stats)

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    df.to_csv(REPORTS_DIR / "mtf_opt_stage1.csv", index=False)

    best = df.iloc[0]
    print(
        f"  Best IS: threshold={best['threshold']}, "
        f"weights=H1:{best['w_H1']} H4:{best['w_H4']} D:{best['w_D']} W:{best['w_W']}, "
        f"Sharpe={best['sharpe']:.3f}"
    )
    return {
        "threshold": best["threshold"],
        "weights": WEIGHT_PRESETS[int(best["weight_idx"])],
    }, df


# ── Proper Stage 1 (threshold-aware) ─────────────────────────────────────────


def run_stage1_proper(
    base_tf_signals: dict,
    is_close: pd.Series,
    is_atr: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Run threshold-aware Stage 1 sweep correctly."""
    print("\n[STAGE 1] Sweeping threshold x weight presets (100 combinations)...")

    results = []
    total = len(THRESHOLDS) * len(WEIGHT_PRESETS)
    done = 0

    for thresh in THRESHOLDS:
        # Build confluences for all weight presets at this threshold
        batch_conf = {}
        for w_idx, w in enumerate(WEIGHT_PRESETS):
            key = f"t{thresh}_w{w_idx}"
            conf = build_confluence(base_tf_signals, w)
            batch_conf[key] = conf[is_close.index]

        # Run in sub-batches
        keys = list(batch_conf.keys())
        n_sub = (len(keys) + BATCH_SIZE - 1) // BATCH_SIZE
        all_trd = pd.DataFrame()

        for b in range(n_sub):
            bkeys = keys[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            bconf = {k: batch_conf[k] for k in bkeys}
            trades = run_vbt_batch(is_close, bconf, threshold=thresh)
            if trades is not None and len(trades) > 0:
                all_trd = pd.concat([all_trd, trades], ignore_index=True)
            done += len(bkeys)
            sys.stdout.write(f"\r  {done}/{total} combos...   ")
            sys.stdout.flush()

        # Score
        for w_idx, w in enumerate(WEIGHT_PRESETS):
            key = f"t{thresh}_w{w_idx}"
            t_sub = all_trd[all_trd["Column"] == key] if len(all_trd) > 0 else pd.DataFrame()
            stats = score_combo(t_sub, key, is_atr, is_close)
            stats.update(
                {
                    "threshold": thresh,
                    "weight_idx": w_idx,
                    "key": key,
                    **{f"w_{tf}": w[tf] for tf in ["H1", "H4", "D", "W"]},
                }
            )
            results.append(stats)

    print()
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    df.to_csv(REPORTS_DIR / "mtf_opt_stage1.csv", index=False)

    best = df.iloc[0]
    print(
        f"  Best: thresh={best['threshold']}  "
        f"H1:{best['w_H1']} H4:{best['w_H4']} D:{best['w_D']} W:{best['w_W']}  "
        f"Sharpe={best['sharpe']:.3f}  Trades={best['n_trades']:.0f}"
    )
    return {
        "threshold": best["threshold"],
        "weights": WEIGHT_PRESETS[int(best["weight_idx"])],
    }, df


# ── Stage 2: greedy per-TF MA/RSI sweep ──────────────────────────────────────


def run_stage2(
    best_s1: dict,
    base_tf_signals: dict,
    is_close: pd.Series,
    is_atr: pd.Series,
    native_closes: dict,
    h1_index: pd.DatetimeIndex,
) -> dict:
    """Greedy per-TF sweep: optimise each TF independently, carry forward best."""
    print("\n[STAGE 2] Greedy per-TF MA/RSI sweep...")

    best_params = {}  # will hold {TF: {fast_ma, slow_ma, rsi_period}}
    current_signals = dict(base_tf_signals)  # start from default params

    for tf in SWEEP_ORDER:
        grid = PARAM_GRIDS[tf]
        close_tf = native_closes[tf]
        weights = best_s1["weights"]
        thresh = best_s1["threshold"]

        # Pre-compute fixed signal sum (all other TFs at their current best params)
        fixed_conf = pd.Series(0.0, index=h1_index)
        for other_tf, sig in current_signals.items():
            if other_tf != tf:
                fixed_conf += sig * weights.get(other_tf, 0.0)
        fixed_conf = fixed_conf[is_close.index]

        # Build all combos for this TF
        combos = [
            (f, s, r)
            for f, s, r in product(grid["fast_ma"], grid["slow_ma"], grid["rsi_period"])
            if f < s
        ]
        print(f"  {tf}: {len(combos)} combos...")

        batch_conf = {}
        for f, s, r in combos:
            sig = compute_tf_signal(close_tf, f, s, r)
            aligned = sig.reindex(h1_index, method="ffill")[is_close.index]
            key = f"{tf}_f{f}_s{s}_r{r}"
            batch_conf[key] = fixed_conf + aligned * weights.get(tf, 0.0)

        # Sub-batch VBT
        keys = list(batch_conf.keys())
        n_sub = (len(keys) + BATCH_SIZE - 1) // BATCH_SIZE
        all_trd = pd.DataFrame()

        for b in range(n_sub):
            bkeys = keys[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            bconf = {k: batch_conf[k] for k in bkeys}
            trades = run_vbt_batch(is_close, bconf, threshold=thresh)
            if trades is not None and len(trades) > 0:
                all_trd = pd.concat([all_trd, trades], ignore_index=True)
            sys.stdout.write(f"\r    Batch {b + 1}/{n_sub}...   ")
            sys.stdout.flush()

        print()

        # Score and find best
        tf_results = []
        for f, s, r in combos:
            key = f"{tf}_f{f}_s{s}_r{r}"
            t_sub = all_trd[all_trd["Column"] == key] if len(all_trd) > 0 else pd.DataFrame()
            stats = score_combo(t_sub, key, is_atr, is_close)
            stats.update({"fast_ma": f, "slow_ma": s, "rsi_period": r, "key": key})
            tf_results.append(stats)

        tf_df = pd.DataFrame(tf_results).sort_values("sharpe", ascending=False)
        tf_df.to_csv(REPORTS_DIR / f"mtf_opt_stage2_{tf}.csv", index=False)

        best_row = tf_df.iloc[0]
        best_f, best_s, best_r = (
            int(best_row["fast_ma"]),
            int(best_row["slow_ma"]),
            int(best_row["rsi_period"]),
        )
        best_params[tf] = {"fast_ma": best_f, "slow_ma": best_s, "rsi_period": best_r}
        print(
            f"  {tf} best: fast={best_f} slow={best_s} rsi={best_r}"
            f"  Sharpe={best_row['sharpe']:.3f}"
        )

        # Update current_signals for next TF
        new_sig = compute_tf_signal(close_tf, best_f, best_s, best_r)
        current_signals[tf] = new_sig.reindex(h1_index, method="ffill")

    return best_params


# ── Final OOS validation ─────────────────────────────────────────────────────


def validate_oos(
    best_params: dict,
    best_s1: dict,
    native_closes: dict,
    oos_close: pd.Series,
    oos_atr: pd.Series,
    h1_index: pd.DatetimeIndex,
    oos_start: pd.Timestamp,
) -> dict:
    print("\n[FINAL] OOS validation with best parameters...")

    weights = best_s1["weights"]
    thresh = best_s1["threshold"]

    # Build optimised confluence on OOS
    conf = pd.Series(0.0, index=h1_index)
    for tf, params in best_params.items():
        sig = compute_tf_signal(
            native_closes[tf], params["fast_ma"], params["slow_ma"], params["rsi_period"]
        )
        aligned = sig.reindex(h1_index, method="ffill")
        conf += aligned * weights.get(tf, 0.0)

    conf_oos = conf[oos_close.index]

    # Also build current-config confluence for comparison
    conf_baseline = pd.Series(0.0, index=h1_index)
    with open(PROJECT_ROOT / "config" / "mtf.toml", "rb") as f:
        orig_cfg = tomllib.load(f)
    orig_weights = orig_cfg["weights"]
    for tf in ["H1", "H4", "D", "W"]:
        c = orig_cfg[tf]
        sig = compute_tf_signal(native_closes[tf], c["fast_ma"], c["slow_ma"], c["rsi_period"])
        aligned = sig.reindex(h1_index, method="ffill")
        conf_baseline += aligned * orig_weights[tf]
    conf_baseline_oos = conf_baseline[oos_close.index]

    # VBT on OOS — optimised
    trades_opt = run_vbt_batch(oos_close, {"opt": conf_oos}, threshold=thresh)
    # VBT on OOS — baseline (orig threshold)
    orig_thresh = orig_cfg.get("confirmation_threshold", 0.10)
    trades_base = run_vbt_batch(oos_close, {"base": conf_baseline_oos}, threshold=orig_thresh)

    def full_stats(trades: pd.DataFrame, col: str, atr: pd.Series, close: pd.Series) -> dict:
        t_sub = (
            trades[trades["Column"] == col]
            if trades is not None and len(trades) > 0
            else pd.DataFrame()
        )
        tot_ret, _, sharpe, max_dd, n_trades, win_rate, dollar_pnl = build_equity_curve(
            t_sub, atr, close
        )
        n_years = (close.index[-1] - close.index[0]).days / 365.25
        cagr = (
            ((1 + tot_ret / 100) ** (1 / n_years) - 1) * 100
            if n_years > 0 and tot_ret > -100
            else float("nan")
        )
        return {
            "dollar_pnl": dollar_pnl,
            "total_ret": tot_ret,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "n_trades": n_trades,
            "win_rate": win_rate,
        }

    stats_opt = full_stats(trades_opt, "opt", oos_atr, oos_close)
    stats_base = full_stats(trades_base, "base", oos_atr, oos_close)

    return {"optimised": stats_opt, "baseline": stats_base}


# ── Report printer ────────────────────────────────────────────────────────────


def print_report(
    oos_stats: dict,
    best_params: dict,
    best_s1: dict,
    oos_close: pd.Series,
) -> None:
    n_years = (oos_close.index[-1] - oos_close.index[0]).days / 365.25
    oos_start = oos_close.index[0].date()
    oos_end = oos_close.index[-1].date()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  MTF OPTIMISER — OOS VALIDATION REPORT")
    print(sep)
    print(f"  Instrument:  EUR/USD H1  |  OOS: {oos_start} to {oos_end} ({n_years:.1f}yr)")
    print(
        f"  Account: ${INIT_EQUITY:,.0f}  |  Risk: ${RISK_PER_TRADE:.0f}/trade (2%)  |  ATR-14 stop"
    )
    print(f"  Max leverage cap: {MAX_LEVERAGE:.0f}x  |  Spread: 1.5 pips/side")
    print(sep)

    print("\n  OPTIMISED PARAMETERS:")
    print(f"    threshold  = {best_s1['threshold']}")
    w = best_s1["weights"]
    print(f"    weights:   H1={w['H1']}  H4={w['H4']}  D={w['D']}  W={w['W']}")
    for tf, p in best_params.items():
        print(f"    {tf}:  fast={p['fast_ma']}  slow={p['slow_ma']}  rsi={p['rsi_period']}")

    for label, stats in [
        ("BASELINE (current config)", oos_stats["baseline"]),
        ("OPTIMISED", oos_stats["optimised"]),
    ]:
        print(f"\n  {'-' * 60}")
        print(f"  {label}")
        print(f"  {'-' * 60}")
        print(f"    Dollar P&L:       ${stats['dollar_pnl']:>+10,.0f}")
        print(f"    Total Return:     {stats['total_ret']:>+10.2f}%")
        print(f"    CAGR:             {stats['cagr']:>+10.2f}%  per year")
        print(f"    Sharpe Ratio:     {stats['sharpe']:>10.3f}")
        print(f"    Max Drawdown:     {stats['max_dd']:>10.2f}%")
        print(f"    Total Trades:     {stats['n_trades']:>10,d}")
        print(f"    Win Rate:         {stats['win_rate']:>10.1f}%")
        avg = stats["dollar_pnl"] / stats["n_trades"] if stats["n_trades"] > 0 else 0
        print(f"    Avg PnL/Trade:    ${avg:>+10.2f}")
        annual_dollar = stats["dollar_pnl"] / n_years
        print(f"    Annual P&L:       ${annual_dollar:>+10,.0f}/yr")

    print(f"\n{sep}")
    b, o = oos_stats["baseline"], oos_stats["optimised"]
    print(
        f"  IMPROVEMENT:  Sharpe {b['sharpe']:.3f} -> {o['sharpe']:.3f}"
        f"  |  CAGR {b['cagr']:+.1f}% -> {o['cagr']:+.1f}%"
    )
    print(sep)


# ── TOML writer ───────────────────────────────────────────────────────────────


def write_toml_recommendation(best_params: dict, best_s1: dict) -> None:
    w = best_s1["weights"]
    lines = [
        "# mtf.toml — Optimised parameters (run_optimise_mtf.py)",
        f"confirmation_threshold = {best_s1['threshold']}",
        "",
        "[weights]",
        f"H1 = {w['H1']}",
        f"H4 = {w['H4']}",
        f"D  = {w['D']}",
        f"W  = {w['W']}",
    ]
    for tf in ["H1", "H4", "D", "W"]:
        p = best_params[tf]
        lines += [
            "",
            f"[{tf}]",
            f"fast_ma    = {p['fast_ma']}",
            f"slow_ma    = {p['slow_ma']}",
            f"rsi_period = {p['rsi_period']}",
        ]
    rec_path = REPORTS_DIR / "mtf_recommended.toml"
    rec_path.write_text("\n".join(lines))
    print(f"\n  Recommended config saved: {rec_path.name}")
    print("  Review and copy to config/mtf.toml if satisfied.")
    print("\n  Recommended TOML:")
    print("  " + "-" * 50)
    for line in lines:
        print(f"  {line}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 68)
    print("  MTF CONFLUENCE OPTIMISER")
    print(f"  Account: ${INIT_EQUITY:,.0f}  |  Risk: 2% = ${RISK_PER_TRADE:.0f}/trade  |  ATR-stop")
    print("=" * 68)

    # Load data
    print("\n[0] Loading data...")
    with open(PROJECT_ROOT / "config" / "mtf.toml", "rb") as f:
        mtf_cfg = tomllib.load(f)

    h1_df = load_parquet(PAIR, "H1")
    h4_df = load_parquet(PAIR, "H4")
    d_df = load_parquet(PAIR, "D")
    w_df = load_parquet(PAIR, "W")

    close = h1_df["close"]
    h1_index = close.index
    atr14 = compute_atr14(h1_df)  # ATR on H1 data

    n_total = len(close)
    split_idx = int(n_total * IS_FRAC)
    is_close = close.iloc[:split_idx]
    oos_close = close.iloc[split_idx:]
    is_atr = atr14.iloc[:split_idx]
    oos_atr = atr14.iloc[split_idx:]
    oos_start = oos_close.index[0]

    print(
        f"  H1 bars: {n_total:,}  |  IS: {len(is_close):,} "
        f"({is_close.index[0].date()} to {is_close.index[-1].date()})"
    )
    print(
        f"  OOS: {len(oos_close):,} ({oos_close.index[0].date()} to {oos_close.index[-1].date()})"
    )

    # Position sizing stats
    atr_mean = atr14.mean()
    avg_units = RISK_PER_TRADE / atr_mean
    avg_notional = avg_units * close.mean()
    avg_leverage = avg_notional / INIT_EQUITY
    avg_margin = avg_notional * 0.02
    print(f"\n  Typical position sizing (avg ATR={atr_mean:.5f}):")
    print(f"    Units/trade:   {avg_units:,.0f} EUR")
    print(f"    Notional:      ${avg_notional:,.0f}")
    print(f"    Leverage:      {avg_leverage:.1f}x")
    print(f"    Margin (2%):   ${avg_margin:,.0f}  (of ${INIT_EQUITY:,.0f} account)")

    # Native closes for Stage 2 (each TF's own close)
    native_closes = {
        "H1": h1_df["close"],
        "H4": h4_df["close"],
        "D": d_df["close"],
        "W": w_df["close"],
    }

    # Pre-compute default TF signals
    print("\n  Pre-computing default TF signals...")
    base_tf_signals = build_tf_signals(mtf_cfg, h1_index)

    # Stage 1
    best_s1, s1_df = run_stage1_proper(base_tf_signals, is_close, is_atr)

    # Stage 2
    best_params = run_stage2(best_s1, base_tf_signals, is_close, is_atr, native_closes, h1_index)

    # OOS validation
    oos_stats = validate_oos(
        best_params, best_s1, native_closes, oos_close, oos_atr, h1_index, oos_start
    )

    # Report
    print_report(oos_stats, best_params, best_s1, oos_close)
    write_toml_recommendation(best_params, best_s1)

    # Top 10 Stage 1
    print("\n  Top 10 Stage 1 IS combinations:")
    print(
        f"  {'Threshold':>10} {'H1':>5} {'H4':>5} {'D':>5} {'W':>5}"
        f" {'Sharpe':>8} {'Return':>8} {'Trades':>7}"
    )
    print("  " + "-" * 58)
    for _, row in s1_df.head(10).iterrows():
        print(
            f"  {row['threshold']:>10.2f} {row['w_H1']:>5.2f} {row['w_H4']:>5.2f} "
            f"{row['w_D']:>5.2f} {row['w_W']:>5.2f} {row['sharpe']:>8.3f} "
            f"{row['total_ret']:>+7.1f}% {int(row['n_trades']):>7,d}"
        )

    print("\n  Stage 1 scoreboard: .tmp/reports/mtf_opt_stage1.csv")
    for tf in SWEEP_ORDER:
        print(f"  Stage 2 {tf} scoreboard: .tmp/reports/mtf_opt_stage2_{tf}.csv")
    print()


if __name__ == "__main__":
    main()
