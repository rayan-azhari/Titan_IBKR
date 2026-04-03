"""run_ftse_dax_expansion.py — FTSE / DAX IC Signal Expansion.

Phase 1-3 research script. Determines whether FTSE and DAX add diversifying
alpha to the existing portfolio by running the IC signal sweep on available
daily index data.

Execution instruments (user confirmed):
    FTSE -> EWU (iShares MSCI UK ETF)
    DAX  -> EWG (iShares MSCI Germany ETF)

Index data (^FTSE_D.parquet, ^GDAXI_D.parquet) is used for signal generation.
P&L is simulated on the same index returns as a proxy until EWU/EWG data is
downloaded. Download proxy ETFs before live deployment:

    uv run python scripts/download_data_yfinance.py --symbols EWU EWG --tf D

Methodology
-----------
Uses the IC Equity Daily pipeline (not IC MTF FX), because only daily data
is available for FTSE/DAX. This mirrors ic_equity_daily and ic_generic
configurations for daily equities.

Signals tested: rsi_21_dev, ma_spread_10_50 (top broad cross-asset performers
from the 513-symbol sweep in FINDINGS.md, confirming STRONG in 63-70% of
daily symbols).

Direction: long_only initially (indices have structural long bias).
Cost profile: "etf" (approximates EWU/EWG; switch to "futures" for Z/FDAX).
Risk: 0.5% per trade (conservative; consistent with ic_equity_daily config).

Usage
-----
    # Full pipeline (IC sweep + Phase 3 backtest + correlation analysis):
    uv run python research/portfolio/run_ftse_dax_expansion.py

    # IC sweep only (fast):
    uv run python research/portfolio/run_ftse_dax_expansion.py --sweep-only

    # Check correlation vs existing portfolio:
    uv run python research/portfolio/run_ftse_dax_expansion.py --corr-only

Output
------
    Console: IC leaderboard + Phase 3 stats (if Phase 3 passes gates)
    CSV:     .tmp/reports/ftse_dax_ic_sweep.csv
             .tmp/reports/ftse_dax_phase3_{instrument}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Instrument configuration ───────────────────────────────────────────────────

EXPANSION_INSTRUMENTS = {
    "^FTSE": {
        "proxy_etf": "EWU",
        "label": "FTSE 100",
        "tf": "D",
        "parquet_stem": "^FTSE",
    },
    "^GDAXI": {
        "proxy_etf": "EWG",
        "label": "DAX",
        "tf": "D",
        "parquet_stem": "^GDAXI",
    },
}

# Signals to sweep: top performers from 513-symbol cross-asset sweep (FINDINGS.md)
# rsi_21_dev:     STRONG in 63% of daily symbols (mean-reversion)
# ma_spread_10_50: STRONG in 70% (trend-following)
TARGET_SIGNALS = ["rsi_21_dev", "ma_spread_10_50"]

# IC sweep: accept signals with |IC| >= 0.04 and ICIR >= 0.40 as USABLE
IC_THRESHOLD = 0.04
ICIR_THRESHOLD = 0.40

# Phase 3 thresholds to test
PHASE3_THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50]

# Quality gates for Phase 3 pass
PHASE3_OOS_SHARPE_GATE = 1.0
PHASE3_OOS_IS_RATIO_GATE = 0.5
PHASE3_MAX_DD_GATE = 0.25


# ── Data loader ────────────────────────────────────────────────────────────────


def _load_index_daily(parquet_stem: str) -> pd.DataFrame:
    """Load daily OHLCV from data/{stem}_D.parquet."""
    p = DATA_DIR / f"{parquet_stem}_D.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Index data not found: {p}\n"
            f"Download via: uv run python scripts/download_data_yfinance.py "
            f"--symbols {parquet_stem} --tf D"
        )
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


# ── Phase 1: IC sweep ──────────────────────────────────────────────────────────


def run_ic_sweep(
    df: pd.DataFrame,
    instrument_label: str,
    is_ratio: float = 0.70,
) -> pd.DataFrame:
    """Run IC sweep on daily OHLCV data for specified signals.

    Computes IC and ICIR for each signal at horizons h=5, 10, 20, 60.
    Returns a DataFrame ranked by |ICIR|.

    Mirrors the Phase 1 pipeline from research/ic_analysis/phase1_sweep.py
    but restricted to daily data and the cross-asset signal subset.
    """
    from scipy.stats import spearmanr

    close = df["close"]
    n = len(close)
    is_n = int(n * is_ratio)

    horizons = [5, 10, 20, 60]
    rows: list[dict] = []

    for sig_name in TARGET_SIGNALS:
        signal = _compute_signal(df, sig_name)
        if signal is None:
            continue

        for h in horizons:
            fwd = np.log(close.shift(-h) / close)
            both = pd.concat([signal, fwd], axis=1).dropna()
            is_both = both.iloc[:is_n]

            if len(is_both) < 30:
                continue

            ic, _ = spearmanr(is_both.iloc[:, 0], is_both.iloc[:, 1])
            if np.isnan(ic):
                continue

            # ICIR via rolling 63-day IC (≈ 1 quarter windows)
            window = 63
            rolling_ics = []
            for start in range(0, len(is_both) - window, window):
                sub = is_both.iloc[start : start + window]
                r, _ = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
                if not np.isnan(r):
                    rolling_ics.append(r)

            icir = (
                float(np.mean(rolling_ics) / np.std(rolling_ics))
                if len(rolling_ics) >= 5 and np.std(rolling_ics) > 1e-9
                else 0.0
            )

            verdict = "FLAT"
            if abs(ic) >= 0.07 and abs(icir) >= 0.60:
                verdict = "STRONG"
            elif abs(ic) >= IC_THRESHOLD and abs(icir) >= ICIR_THRESHOLD:
                verdict = "USABLE"

            rows.append(
                {
                    "instrument": instrument_label,
                    "signal": sig_name,
                    "horizon": h,
                    "IC": round(ic, 4),
                    "ICIR": round(icir, 4),
                    "verdict": verdict,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("ICIR", key=lambda x: x.abs(), ascending=False)
    return result


def _compute_signal(df: pd.DataFrame, sig_name: str) -> pd.Series | None:
    """Compute a named signal on daily OHLCV data."""
    close = df["close"]

    if sig_name == "rsi_21_dev":
        # RSI(21) - 50 (deviation from neutral)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(21).mean()
        loss = (-delta.clip(upper=0)).rolling(21).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        signal = rsi - 50.0

    elif sig_name == "ma_spread_10_50":
        # (SMA10 - SMA50) / SMA50  (normalised MA spread)
        sma10 = close.rolling(10).mean()
        sma50 = close.rolling(50).mean()
        signal = (sma10 - sma50) / (sma50 + 1e-9)

    else:
        return None

    signal = signal.shift(1)  # PREVENT LOOK-AHEAD: signal available after bar closes
    signal.name = sig_name
    return signal.dropna()


# ── Phase 3: IS/OOS backtest ───────────────────────────────────────────────────


def run_phase3_backtest(
    df: pd.DataFrame,
    instrument: str,
    best_signal: str,
    ic_sign: float,
    thresholds: list[float] = PHASE3_THRESHOLDS,
    is_ratio: float = 0.70,
) -> pd.DataFrame:
    """Simplified Phase 3 backtest for daily index data.

    Uses IC sign-normalised composite z-score thresholding,
    consistent with phase3_backtest.py methodology.

    Args:
        df:           daily OHLCV data.
        instrument:   label for display / output filenames.
        best_signal:  signal name to use (from IC sweep winner).
        ic_sign:      +1 or -1 from IC sweep (sign normalisation).
        thresholds:   z-score entry thresholds to test.
        is_ratio:     IS/OOS split ratio.

    Returns:
        pd.DataFrame with per-threshold IS and OOS stats.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        print("  ERROR: vectorbt required. Run: uv add vectorbt")
        return pd.DataFrame()

    close = df["close"]
    n = len(close)
    is_n = int(n * is_ratio)

    signal = _compute_signal(df, best_signal)
    if signal is None:
        return pd.DataFrame()

    # Align signal to close index
    signal_aligned = signal.reindex(close.index, method="ffill")

    # Sign-normalise using IS IC direction
    composite = signal_aligned * ic_sign

    # Z-score normalise using IS mean/std (no look-ahead)
    is_vals = composite.iloc[:is_n]
    mu = float(is_vals.mean())
    sigma = float(is_vals.std())
    if sigma < 1e-10:
        print(f"  [WARN] Zero variance in IS composite for {instrument}. Skipping Phase 3.")
        return pd.DataFrame()
    composite_z = (composite - mu) / sigma

    # ATR-based position sizing (1% risk per trade)
    atr14 = _atr14(df, 14)
    risk_pct = 0.005  # 0.5% per ic_equity_daily convention
    stop_atr = 1.5
    stop_dist = stop_atr * atr14
    stop_pct = (stop_dist / close).clip(upper=0.5).fillna(0.0)
    size_pct = (risk_pct / stop_pct.replace(0.0, np.nan)).clip(upper=3.0).fillna(0.0)

    # Cost model (ETF profile)
    med_close = float(close.median()) or 1.0
    spread_price = (1.0 / 10000) * med_close  # 1 bps
    slip_price = (1.0 / 10000) * med_close

    vbt_fees = (spread_price / close).bfill().values
    vbt_slip = (slip_price / close).bfill().values

    rows: list[dict] = []
    for theta in thresholds:
        sig = composite_z.shift(1).fillna(0.0)

        pf = vbt.Portfolio.from_signals(
            close,
            entries=sig > theta,
            exits=sig <= 0.0,
            sl_stop=stop_pct.values,
            size=size_pct.values,
            size_type="percent",
            init_cash=10_000.0,
            fees=vbt_fees,
            slippage=vbt_slip,
            freq="d",
        )

        # IS stats
        is_val = pf.value().iloc[:is_n]
        is_ret = is_val.pct_change().dropna()
        is_std = float(is_ret.std())
        is_sharpe = float(is_ret.mean() / is_std * np.sqrt(252)) if is_std > 1e-9 else 0.0
        # OOS stats
        oos_val = pf.value().iloc[is_n:]
        oos_ret = oos_val.pct_change().dropna()
        oos_std = float(oos_ret.std())
        oos_sharpe = float(oos_ret.mean() / oos_std * np.sqrt(252)) if oos_std > 1e-9 else 0.0
        oos_dd = float((oos_val / oos_val.cummax() - 1).min())
        n_trades = int(pf.trades.count())

        pass_gates = (
            oos_sharpe >= PHASE3_OOS_SHARPE_GATE
            and (oos_sharpe / is_sharpe >= PHASE3_OOS_IS_RATIO_GATE if is_sharpe > 1e-9 else False)
            and abs(oos_dd) <= PHASE3_MAX_DD_GATE
            and n_trades >= 30
        )

        rows.append(
            {
                "instrument": instrument,
                "signal": best_signal,
                "threshold": theta,
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "oos_is_ratio": round(oos_sharpe / is_sharpe, 3) if is_sharpe > 1e-9 else 0.0,
                "oos_max_dd": round(oos_dd, 4),
                "n_trades": n_trades,
                "pass": pass_gates,
            }
        )

    return pd.DataFrame(rows)


def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()


# ── Correlation vs existing portfolio ─────────────────────────────────────────


def check_portfolio_correlation(
    index_rets: dict[str, pd.Series],
    n_existing_strategies: int = 3,
) -> None:
    """Load a sample of existing strategy OOS returns and compute correlation."""
    print("\n  Checking correlation vs existing portfolio strategies...")

    try:
        from research.portfolio.loaders.oos_returns import load_all_strategies

        sample_strategies = ["ic_mtf_eur_usd", "ic_mtf_gbp_usd", "etf_trend_spy"]
        existing = load_all_strategies(sample_strategies[:n_existing_strategies], skip_errors=True)
    except Exception as exc:
        print(f"  [WARN] Could not load existing strategy returns: {exc}")
        return

    all_rets = {**existing, **index_rets}
    if len(all_rets) < 2:
        return

    # Align to common index
    from research.portfolio.loaders.oos_returns import align_to_common_window

    aligned = align_to_common_window(all_rets)

    df = pd.DataFrame(aligned).dropna()
    corr = df.corr()

    print(f"\n  Correlation matrix (common period: {len(df)} days):")
    print("  " + "-" * 60)
    labels = list(aligned.keys())
    print("  " + "  ".join(f"{lbl[:10]:>10}" for lbl in labels))
    for row_label in labels:
        vals = "  ".join(f"{corr.loc[row_label, col]:>+10.3f}" for col in labels)
        print(f"  {row_label:<12} {vals}")

    # Flag high correlations
    for inst_label in index_rets:
        if inst_label not in corr.index:
            continue
        for strat_label in existing:
            if strat_label not in corr.columns:
                continue
            r = corr.loc[inst_label, strat_label]
            if abs(r) > 0.60:
                print(
                    f"\n  [!] High correlation: {inst_label} ↔ {strat_label}: "
                    f"r={r:+.3f} — check if this adds diversification."
                )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FTSE/DAX IC signal expansion.")
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run IC sweep only (skip Phase 3 backtest).",
    )
    parser.add_argument(
        "--corr-only",
        action="store_true",
        help="Run correlation analysis only.",
    )
    args = parser.parse_args()

    W = 72
    print()
    print("=" * W)
    print("  FTSE / DAX IC SIGNAL EXPANSION")
    print("  Execution instruments: EWU (FTSE) and EWG (DAX) — see module docstring")
    print("=" * W)

    all_sweep_rows: list[pd.DataFrame] = []
    index_rets: dict[str, pd.Series] = {}

    for inst_key, meta in EXPANSION_INSTRUMENTS.items():
        label = meta["label"]
        stem = meta["parquet_stem"]
        proxy = meta["proxy_etf"]

        print(f"\n  -- {label} ({inst_key}) " + "-" * 30)

        try:
            df = _load_index_daily(stem)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        n = len(df)
        print(f"  Loaded {n} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
        if n < 400:
            print(f"  [WARN] Only {n} bars — results may not be statistically robust.")

        if not args.corr_only:
            # ── Phase 1: IC Sweep ──────────────────────────────────────────
            print(f"\n  Running IC sweep (signals: {TARGET_SIGNALS})...")
            sweep_df = run_ic_sweep(df, label)
            all_sweep_rows.append(sweep_df)

            if sweep_df.empty:
                print("  [WARN] No signals computed — check data quality.")
                continue

            print(f"\n  IC Sweep results for {label}:")
            print(f"  {'Signal':<20} {'h':>4} {'IC':>7} {'ICIR':>7} {'Verdict':<10}")
            print("  " + "-" * 55)
            for _, row in sweep_df.iterrows():
                print(
                    f"  {row['signal']:<20} {int(row['horizon']):>4} "
                    f"{row['IC']:>+7.4f} {row['ICIR']:>+7.4f} "
                    f"{row['verdict']:<10}"
                )

            # ── Phase 3: Backtest (only if at least one USABLE signal) ────
            usable = sweep_df[sweep_df["verdict"].isin(["STRONG", "USABLE"])]
            if usable.empty:
                print(
                    f"\n  [SKIP Phase 3] No USABLE signals found for {label}. "
                    f"Does not qualify for portfolio inclusion at this time."
                )
                continue

            if not args.sweep_only:
                # Use best signal at h=20 for Phase 3
                best = usable[usable["horizon"] == 20].head(1)
                if best.empty:
                    best = usable.head(1)
                best_sig = best.iloc[0]["signal"]
                ic_sign = float(np.sign(best.iloc[0]["IC"]))
                print(
                    f"\n  Running Phase 3 backtest with signal '{best_sig}' "
                    f"(IC sign={ic_sign:+.0f})..."
                )

                phase3_df = run_phase3_backtest(df, label, best_sig, ic_sign)
                if not phase3_df.empty:
                    print(f"\n  Phase 3 results for {label}:")
                    print(
                        f"  {'Threshold':>10} {'IS Sharpe':>10} "
                        f"{'OOS Sharpe':>10} {'OOS/IS':>7} "
                        f"{'OOS MaxDD':>10} {'Trades':>7} {'Pass':<6}"
                    )
                    print("  " + "-" * 65)
                    for _, r in phase3_df.iterrows():
                        flag = "PASS" if r["pass"] else "FAIL"
                        print(
                            f"  {r['threshold']:>10.2f} {r['is_sharpe']:>+10.3f} "
                            f"{r['oos_sharpe']:>+10.3f} {r['oos_is_ratio']:>+7.3f} "
                            f"{r['oos_max_dd']:>+10.2%} {int(r['n_trades']):>7} "
                            f"[{flag}]"
                        )

                    # Save Phase 3 results
                    out_path = REPORTS_DIR / f"ftse_dax_phase3_{stem.replace('^', '')}.csv"
                    phase3_df.to_csv(out_path, index=False)
                    print(f"  Phase 3 saved -> {out_path}")

                    # Build OOS return series for correlation check
                    passing = phase3_df[phase3_df["pass"]]
                    if not passing.empty:
                        best_theta = float(passing.iloc[0]["threshold"])
                        n_is = int(len(df) * 0.70)
                        sig_all = _compute_signal(df, best_sig)
                        if sig_all is not None:
                            sig_aligned = sig_all.reindex(df["close"].index, method="ffill")
                            is_comp = (sig_aligned * ic_sign).iloc[:n_is]
                            mu = float(is_comp.mean())
                            sigma = float(is_comp.std())
                            comp_z = (sig_aligned * ic_sign - mu) / (sigma + 1e-9)
                            entries = comp_z.shift(1).fillna(0.0) > best_theta
                            exits = comp_z.shift(1).fillna(0.0) <= 0.0
                            pos = pd.Series(0.0, index=df.index)
                            in_pos = False
                            for i in range(len(entries)):
                                if not in_pos and entries.iloc[i]:
                                    in_pos = True
                                elif in_pos and exits.iloc[i]:
                                    in_pos = False
                                pos.iloc[i] = float(in_pos)
                            daily_rets = df["close"].pct_change().fillna(0.0) * pos
                            oos_rets = daily_rets.iloc[n_is:]
                            oos_rets.name = stem.replace("^", "").lower()
                            index_rets[oos_rets.name] = oos_rets
                            print(
                                f"\n  {label} qualifies for portfolio inclusion. "
                                f"OOS Sharpe: {passing.iloc[0]['oos_sharpe']:+.3f}"
                            )
                            print(
                                f"  NOTE: Add '{oos_rets.name}' to ic_generic.toml and "
                                f"download {proxy} for live deployment."
                            )
                    else:
                        print(
                            f"\n  {label} does NOT pass Phase 3 gates. "
                            f"Not ready for portfolio inclusion."
                        )

    # ── Save IC sweep results ──────────────────────────────────────────────
    if all_sweep_rows:
        combined_sweep = pd.concat(all_sweep_rows, ignore_index=True)
        out_path = REPORTS_DIR / "ftse_dax_ic_sweep.csv"
        combined_sweep.to_csv(out_path, index=False)
        print(f"\n  IC sweep saved -> {out_path}")

    # ── Correlation vs existing portfolio ──────────────────────────────────
    if index_rets or args.corr_only:
        # For corr-only mode, build a proxy daily return series for each index
        if args.corr_only and not index_rets:
            for inst_key, meta in EXPANSION_INSTRUMENTS.items():
                try:
                    df_corr = _load_index_daily(meta["parquet_stem"])
                    n_is = int(len(df_corr) * 0.70)
                    rets = df_corr["close"].pct_change().fillna(0.0).iloc[n_is:]
                    rets.name = meta["parquet_stem"].replace("^", "").lower()
                    index_rets[rets.name] = rets
                except FileNotFoundError:
                    pass
        check_portfolio_correlation(index_rets)

    print("\n  Done.")
    print("\n  Next steps:")
    print("  1. If signals are USABLE/STRONG -> update ic_generic.toml with FTSE/DAX sections")
    print(
        "  2. Download EWU and EWG: uv run python scripts/download_data_yfinance.py "
        "--symbols EWU EWG --tf D"
    )
    print("  3. Re-run with EWU/EWG data for accurate P&L simulation")
    print("  4. Run run_combination_sweep.py to include FTSE/DAX in portfolio sweep")


if __name__ == "__main__":
    main()
