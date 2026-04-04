"""phase1_param_sweep.py -- Phase 1 Extension -- Parameter Grid Search.

Sweeps parameter grids for 10 signal families and ranks each combination
by Spearman IC at each horizon. Computes IC on IS (first 80%) and OOS
(last 20%) separately to detect parameter overfitting.

Signal families and grids:
    RSI             -- period: [5, 7, 9, 12, 14, 21, 28]
    MA Spread       -- fast in [3,5,8,10], slow in [15,20,30,50] (fast < slow)
    ROC             -- period: [2, 3, 5, 10, 15, 20, 30, 60]
    Bollinger/Z     -- window: [10, 15, 20, 30, 50, 100]
    ATR (norm)      -- period: [7, 10, 14, 21]
    Stochastic      -- k in [5,9,14,21], d in [2,3,5]
    Donchian        -- window: [5, 10, 20, 40, 55, 100]
    ADX             -- period: [7, 10, 14, 21, 28]
    MACD hist       -- fast in [8,10,12], slow in [20,26,30], signal in [7,9,12]
    Realized Vol    -- window: [5, 10, 20, 40, 60]

IS/OOS split:
    80% of bars (time-ordered) = IS.  Last 20% = OOS.
    Best parameters selected on IS IC. OOS IC reported for same parameters.
    Flags OOS drop > 30% relative to IS IC as a potential overfit warning.

Usage:
    uv run python research/ic_analysis/phase1_param_sweep.py
    uv run python research/ic_analysis/phase1_param_sweep.py \\
        --instrument EUR_USD --timeframe H4

Look-ahead safety:
    Signal computation uses only .rolling() / .ewm() / .shift(+n) (causal).
    Forward returns use close.shift(-h) -- intentional target, not feature.
    The IS/OOS split divides (signal, fwd_return) aligned pairs -- the signal
    itself is computed over the full dataset (backward-looking only), so the
    split is on the evaluation pairs, not the signal computation window.
"""

import argparse
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _load_ohlcv,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
)
from titan.strategies.ml.features import (  # noqa: E402
    adx,
    atr,
    ema,
    macd_hist,
    rsi,
    sma,
    stochastic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [1, 5, 10, 20, 60]
ICIR_WINDOW = 60
IS_FRACTION = 0.80
OOS_DROP_WARN = 0.30  # warn if OOS IC drops > 30% relative to IS IC


# ── Parameter grids ────────────────────────────────────────────────────────────


def _build_param_grid() -> dict[str, list[dict]]:
    """Return parameter grids for all 10 signal families."""
    grid: dict[str, list[dict]] = {}

    grid["RSI"] = [{"period": p} for p in [5, 7, 9, 12, 14, 21, 28]]

    ma_grid = []
    for fast in [3, 5, 8, 10]:
        for slow in [15, 20, 30, 50]:
            if fast < slow:
                ma_grid.append({"fast": fast, "slow": slow})
    grid["MA_Spread"] = ma_grid

    grid["ROC"] = [{"period": p} for p in [2, 3, 5, 10, 15, 20, 30, 60]]

    grid["Bollinger"] = [{"window": w} for w in [10, 15, 20, 30, 50, 100]]

    grid["ATR"] = [{"period": p} for p in [7, 10, 14, 21]]

    stoch_grid = []
    for k in [5, 9, 14, 21]:
        for d in [2, 3, 5]:
            stoch_grid.append({"k": k, "d": d})
    grid["Stochastic"] = stoch_grid

    grid["Donchian"] = [{"window": w} for w in [5, 10, 20, 40, 55, 100]]

    grid["ADX"] = [{"period": p} for p in [7, 10, 14, 21, 28]]

    macd_grid = []
    for fast, slow, sig in product([8, 10, 12], [20, 26, 30], [7, 9, 12]):
        if fast < slow:
            macd_grid.append({"fast": fast, "slow": slow, "signal": sig})
    grid["MACD"] = macd_grid

    grid["RealizedVol"] = [{"window": w} for w in [5, 10, 20, 40, 60]]

    return grid


# ── Signal computation per family ─────────────────────────────────────────────


def _compute_family_signal(
    family: str,
    params: dict,
    df: pd.DataFrame,
    close: pd.Series,
) -> pd.Series | None:
    """Compute a single signal for a given family and parameter set.

    Returns None if parameters are invalid (e.g., stochastic with k < d period).
    Signal is always normalised to be dimensionless:
        - RSI: RSI(p) - 50  (centred at 0)
        - MA Spread: (EMA_fast - EMA_slow) / EMA_slow
        - ROC: log(close / close.shift(p))
        - Bollinger: (close - SMA_w) / std_w  (z-score)
        - ATR: ATR(p) / close  (normalised)
        - Stochastic: stoch_K(k,d) - 50  (centred at 0)
        - Donchian: (close - rolling_min) / range - 0.5  (centred at 0)
        - ADX: ADX(p)  (0-100 scale, not centred)
        - MACD: macd_hist(fast, slow, signal)  (absolute scale)
        - RealizedVol: rolling std of log returns * sqrt(252)
    """
    try:
        if family == "RSI":
            p = params["period"]
            return rsi(close, p) - 50.0

        elif family == "MA_Spread":
            fast, slow = params["fast"], params["slow"]
            e_fast = ema(close, fast)
            e_slow = ema(close, slow)
            return (e_fast - e_slow) / e_slow.replace(0, np.nan)

        elif family == "ROC":
            p = params["period"]
            log_c = np.log(close)
            return log_c - log_c.shift(p)

        elif family == "Bollinger":
            w = params["window"]
            s = sma(close, w)
            std = close.rolling(w).std().replace(0, np.nan)
            return (close - s) / std

        elif family == "ATR":
            p = params["period"]
            return atr(df, p) / close.replace(0, np.nan)

        elif family == "Stochastic":
            k_p, d_p = params["k"], params["d"]
            stoch_k, _ = stochastic(df, k_period=k_p, d_period=d_p)
            return stoch_k - 50.0

        elif family == "Donchian":
            w = params["window"]
            lo = df["low"].rolling(w).min()
            hi = df["high"].rolling(w).max()
            rng = (hi - lo).replace(0, np.nan)
            return (close - lo) / rng - 0.5

        elif family == "ADX":
            p = params["period"]
            return adx(df, p)

        elif family == "MACD":
            fast, slow, sig = params["fast"], params["slow"], params["signal"]
            return macd_hist(close, fast, slow, sig)

        elif family == "RealizedVol":
            w = params["window"]
            log_r = np.log(close).diff()
            return log_r.rolling(w).std() * np.sqrt(252)

    except Exception as exc:
        logger.debug("Family %s params %s error: %s", family, params, exc)
        return None

    return None


# ── IC computation for a single signal on IS and OOS ──────────────────────────


def _compute_family_ic(
    sig: pd.Series,
    fwd_is: pd.DataFrame,
    fwd_oos: pd.DataFrame,
    horizons: list[int],
) -> dict:
    """Compute IC and ICIR on IS and OOS for a single signal.

    Returns dict: {horizon -> {is_ic, oos_ic, is_icir}}.
    """
    result: dict[int, dict] = {}
    for h in horizons:
        fwd_col = f"fwd_{h}"
        # IS
        sig_is = sig.reindex(fwd_is.index)
        sig_df_is = sig_is.to_frame(name="sig")
        ic_is_df = compute_ic_table(sig_df_is, fwd_is[[fwd_col]])
        is_ic = float(ic_is_df.iloc[0, 0]) if not ic_is_df.empty else np.nan

        # ICIR on IS
        icir_s = compute_icir(
            sig_df_is,
            fwd_is[[fwd_col]],
            horizons=[h],
            window=ICIR_WINDOW,
        )
        is_icir = float(icir_s.iloc[0]) if not icir_s.empty else np.nan

        # OOS
        sig_oos = sig.reindex(fwd_oos.index)
        sig_df_oos = sig_oos.to_frame(name="sig")
        ic_oos_df = compute_ic_table(sig_df_oos, fwd_oos[[fwd_col]])
        oos_ic = float(ic_oos_df.iloc[0, 0]) if not ic_oos_df.empty else np.nan

        result[h] = {"is_ic": is_ic, "is_icir": is_icir, "oos_ic": oos_ic}
    return result


# ── Single-family sweep ────────────────────────────────────────────────────────


def _sweep_family(
    family: str,
    grid: list[dict],
    df: pd.DataFrame,
    close: pd.Series,
    fwd_is: pd.DataFrame,
    fwd_oos: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Sweep all parameter combinations for one family. Returns results DataFrame."""
    rows = []
    for params in grid:
        sig = _compute_family_signal(family, params, df, close)
        if sig is None:
            continue

        ic_data = _compute_family_ic(sig, fwd_is, fwd_oos, horizons)
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        for h, vals in ic_data.items():
            is_ic = vals["is_ic"]
            oos_ic = vals["oos_ic"]
            is_icir = vals["is_icir"]

            # Overfit flag: OOS |IC| drops > OOS_DROP_WARN relative to IS |IC|
            if not np.isnan(is_ic) and not np.isnan(oos_ic) and abs(is_ic) > 1e-6:
                drop_pct = (abs(is_ic) - abs(oos_ic)) / abs(is_ic)
                overfit = drop_pct > OOS_DROP_WARN
            else:
                overfit = False

            # Verdict based on IS IC + IS ICIR
            abs_ic = abs(is_ic) if not np.isnan(is_ic) else 0.0
            abs_ir = abs(is_icir) if not np.isnan(is_icir) else 0.0
            if abs_ic >= 0.05 and abs_ir >= 0.5:
                verdict = "STRONG"
            elif abs_ic >= 0.05:
                verdict = "USABLE"
            elif abs_ic >= 0.03:
                verdict = "WEAK"
            else:
                verdict = "NOISE"

            rows.append(
                {
                    "family": family,
                    "params": params_str,
                    "horizon": h,
                    "is_ic": round(is_ic, 5) if not np.isnan(is_ic) else np.nan,
                    "is_icir": round(is_icir, 3) if not np.isnan(is_icir) else np.nan,
                    "oos_ic": round(oos_ic, 5) if not np.isnan(oos_ic) else np.nan,
                    "verdict": verdict,
                    "overfit_warn": overfit,
                }
            )

    return pd.DataFrame(rows)


# ── Display ────────────────────────────────────────────────────────────────────


def _verdict_str(ic: float, icir: float) -> str:
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    if abs_ic >= 0.05 and abs_ir >= 0.5:
        return "STRONG"
    elif abs_ic >= 0.05:
        return "USABLE"
    elif abs_ic >= 0.03:
        return "WEAK"
    return "NOISE"


def _default_ic_for_family(family: str) -> dict[int, float]:
    """Approximate IC of the default parameter for each family (placeholder labels)."""
    # These are shown as reference in the print output
    return {}


def _print_family_results(family: str, results: pd.DataFrame, top_n: int = 5) -> None:
    """Print top-N parameter combinations per horizon for one family."""
    W = 80
    print(f"\n  {family}")
    print("  " + "-" * (W - 2))
    print(f"  {'h':>4}  {'Params':<30}  {'IS_IC':>8}  {'OOS_IC':>8}  {'IS_ICIR':>7}  Verdict  Warn")
    print("  " + "-" * (W - 2))

    for h in sorted(results["horizon"].unique()):
        sub = results[results["horizon"] == h].copy()
        sub = (
            sub.assign(abs_is_ic=sub["is_ic"].abs())
            .sort_values("abs_is_ic", ascending=False)
            .head(top_n)
        )

        for _, row in sub.iterrows():
            is_ic_s = f"{row['is_ic']:>+8.4f}" if not np.isnan(row["is_ic"]) else "     NaN"
            oos_ic_s = f"{row['oos_ic']:>+8.4f}" if not np.isnan(row["oos_ic"]) else "     NaN"
            icir_s = f"{row['is_icir']:>+7.3f}" if not np.isnan(row["is_icir"]) else "    NaN"
            warn_s = " [OVERFIT]" if row["overfit_warn"] else ""
            print(
                f"  h={h:<3}  {row['params']:<30}  {is_ic_s}  {oos_ic_s}  "
                f"{icir_s}  {row['verdict']:<7}{warn_s}"
            )


def _print_summary(all_results: pd.DataFrame) -> None:
    """Print best parameter per family x horizon."""
    W = 80
    print("\n" + "=" * W)
    print("  BEST PARAMETERS SUMMARY (ranked by IS |IC|, showing OOS IC)")
    print("=" * W)
    print(f"  {'Family':<14}  {'h':>4}  {'Params':<30}  {'IS_IC':>8}  {'OOS_IC':>8}  Verdict")
    print("  " + "-" * (W - 2))

    for family in sorted(all_results["family"].unique()):
        for h in sorted(all_results["horizon"].unique()):
            sub = all_results[(all_results["family"] == family) & (all_results["horizon"] == h)]
            if sub.empty:
                continue
            best = (
                sub.assign(abs_is_ic=sub["is_ic"].abs())
                .sort_values("abs_is_ic", ascending=False)
                .iloc[0]
            )
            is_ic_s = f"{best['is_ic']:>+8.4f}" if not np.isnan(best["is_ic"]) else "     NaN"
            oos_ic_s = f"{best['oos_ic']:>+8.4f}" if not np.isnan(best["oos_ic"]) else "     NaN"
            warn = " [!]" if best["overfit_warn"] else ""
            print(
                f"  {family:<14}  h={h:<3}  {best['params']:<30}  "
                f"{is_ic_s}  {oos_ic_s}  {best['verdict']}{warn}"
            )
    print("=" * W)
    print("  [!] = OOS IC dropped > 30% vs IS IC -- likely overfit on this parameter")
    print("=" * W)


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_param_sweep(
    instrument: str,
    timeframe: str,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = HORIZONS

    # 1. Load data
    df = _load_ohlcv(instrument, timeframe)
    close = df["close"]

    # 2. Forward returns over full dataset
    fwd_all = compute_forward_returns(close, horizons)

    # 3. IS/OOS split by bar index (time-ordered, not random)
    n = len(close)
    split = int(n * IS_FRACTION)
    is_idx = close.index[:split]
    oos_idx = close.index[split:]

    fwd_is = fwd_all.reindex(is_idx)
    fwd_oos = fwd_all.reindex(oos_idx)
    logger.info(
        "IS: %d bars (%s - %s), OOS: %d bars (%s - %s)",
        len(is_idx),
        is_idx[0].date(),
        is_idx[-1].date(),
        len(oos_idx),
        oos_idx[0].date(),
        oos_idx[-1].date(),
    )

    # 4. Sweep all families
    param_grid = _build_param_grid()
    all_results: list[pd.DataFrame] = []

    W = 74
    print("\n" + "=" * W)
    print(f"  INDICATOR PARAMETER SWEEP -- {instrument} {timeframe}")
    print(f"  Horizons: {horizons}  |  IS: {len(is_idx):,} bars  |  OOS: {len(oos_idx):,} bars")
    print(f"  IS fraction: {IS_FRACTION:.0%}  |  OOS warn threshold: >{OOS_DROP_WARN:.0%} drop")
    print("=" * W)

    for family, grid in param_grid.items():
        logger.info("Sweeping %s (%d combinations)...", family, len(grid))
        family_results = _sweep_family(family, grid, df, close, fwd_is, fwd_oos, horizons)
        if not family_results.empty:
            all_results.append(family_results)
            _print_family_results(family, family_results, top_n=3)

    if not all_results:
        logger.warning("No results produced -- check data and parameter grids")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # 5. Summary
    _print_summary(combined)

    # 6. Save
    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    full_path = report_dir / f"phase1_params_{instrument}_{timeframe}.csv"
    combined.to_csv(full_path, index=False)
    logger.info("Full sweep saved: %s", full_path)

    # Best params per family per horizon
    best_rows = []
    for family in combined["family"].unique():
        for h in combined["horizon"].unique():
            sub = combined[(combined["family"] == family) & (combined["horizon"] == h)]
            if sub.empty:
                continue
            best = (
                sub.assign(abs_is_ic=sub["is_ic"].abs())
                .sort_values("abs_is_ic", ascending=False)
                .iloc[0]
            )
            best_rows.append(best)
    best_df = pd.DataFrame(best_rows).drop(columns=["abs_is_ic"], errors="ignore")
    best_path = report_dir / f"phase1_params_{instrument}_{timeframe}_best.csv"
    best_df.to_csv(best_path, index=False)
    logger.info("Best params saved: %s", best_path)

    return combined


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Indicator Parameter IC Sweep")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="Comma-separated forward horizons, e.g. 1,5,10,20,60",
    )
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    run_param_sweep(
        instrument=args.instrument,
        timeframe=args.timeframe,
        horizons=horizons,
    )


if __name__ == "__main__":
    main()
