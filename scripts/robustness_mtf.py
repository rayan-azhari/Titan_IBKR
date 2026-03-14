"""robustness_mtf.py — Robustness validation for the MTF Confluence strategy.

Two tests from professional backtesting practice:

  A. Monte Carlo Trade Shuffle (N=1000)
     Shuffles the sequence of OOS trade P&Ls to reveal path-dependency.
     If the strategy relies on a lucky streak, the 5th-pct equity will
     show a very different picture from the median.

  B. Rolling Walk-Forward Validation (2yr train / 6mo test)
     Evaluates the fixed config/mtf.toml parameters on rolling 6-month
     test windows across the full dataset — no parameter re-fitting per
     window. Reveals regime-specific weaknesses.

Usage:
    .venv/Scripts/python.exe scripts/robustness_mtf.py
"""

import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse helpers from run_backtest_meta (which guards the vectorbt import)
from scripts.run_backtest_meta import (
    FEES,
    INIT_CASH,
    PAIR,
    SLIPPAGE,
    _load_parquet,
    compute_mtf_signals,
    run_vbt_portfolio,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

N_MONTE_CARLO = 1000
TRAIN_YEARS = 2
TEST_MONTHS = 6


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def monte_carlo_shuffle(pf, n_sims: int = N_MONTE_CARLO) -> None:
    """Shuffle OOS trade P&L order N times and report equity distribution.

    Tests whether the strategy's equity growth is robust to different trade
    orderings or if it depends on a specific lucky sequence.

    Sharpe annualization uses the actual trade frequency (trades/year) derived
    from the OOS period — one equity point per trade, not per H1 bar.
    Using np.sqrt(252*24) (H1 bars/year) on per-trade returns inflates Sharpe ~9x.
    """
    print("\n" + "=" * 70)
    print(f"MONTE CARLO TRADE SHUFFLE  (N={n_sims:,})")
    print("  Question: does Sharpe hold if trades arrive in a different order?")
    print("=" * 70)

    if pf.trades.count() == 0:
        print("  No trades in portfolio — skipping.")
        return

    pnl = pf.trades.records_readable["PnL"].values
    n_trades = len(pnl)
    actual_final = INIT_CASH + pnl.sum()
    print(f"  OOS trades: {n_trades:,}  |  Actual final equity: ${actual_final:,.0f}")

    # Derive trade frequency for correct Sharpe annualization
    oos_days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
    oos_years = float(oos_days) / 365.25 if oos_days > 0 else 1.0
    trades_per_year = n_trades / oos_years if oos_years > 0 else 252.0
    print(f"  OOS span: {oos_years:.1f} yr  |  Trade freq: {trades_per_year:.0f}/yr")
    print(f"  Annualization factor: sqrt({trades_per_year:.0f}) = {trades_per_year**0.5:.2f}")

    rng = np.random.default_rng(42)
    finals: list[float] = []
    sharpes: list[float] = []
    min_equities: list[float] = []

    for _ in range(n_sims):
        shuffled = rng.permutation(pnl)
        equity = np.empty(n_trades + 1)
        equity[0] = INIT_CASH
        for i, p in enumerate(shuffled):
            equity[i + 1] = equity[i] + p

        finals.append(float(equity[-1]))
        min_equities.append(float(equity.min()))

        ret = equity[1:] / equity[:-1] - 1.0
        std = float(ret.std())
        # Annualize at trade frequency, not bar frequency
        sharpe = float(ret.mean()) / std * np.sqrt(trades_per_year) if std > 0 else 0.0
        sharpes.append(sharpe)

    finals_arr = np.array(finals)
    sharpes_arr = np.array(sharpes)
    min_eq_arr = np.array(min_equities)

    pct_profitable = (finals_arr > INIT_CASH).mean() * 100
    pct_neg_sharpe = (sharpes_arr < 0).mean() * 100

    print(f"\n  Final Equity Distribution (across {n_sims:,} shuffled orderings):")
    print(f"    5th  pct:  ${np.percentile(finals_arr, 5):>12,.0f}")
    print(f"    25th pct:  ${np.percentile(finals_arr, 25):>12,.0f}")
    print(f"    Median:    ${np.median(finals_arr):>12,.0f}")
    print(f"    75th pct:  ${np.percentile(finals_arr, 75):>12,.0f}")
    print(f"    95th pct:  ${np.percentile(finals_arr, 95):>12,.0f}")
    print(f"    Actual:    ${actual_final:>12,.0f}")
    print(f"\n  % sims ending above start ($100k): {pct_profitable:.0f}%")
    print(f"  % sims with negative Sharpe:        {pct_neg_sharpe:.0f}%")
    print("\n  Sharpe Distribution:")
    print(f"    5th  pct:  {np.percentile(sharpes_arr, 5):>8.3f}")
    print(f"    Median:    {np.median(sharpes_arr):>8.3f}")
    print(f"    95th pct:  {np.percentile(sharpes_arr, 95):>8.3f}")
    print("\n  Worst-case intra-simulation drawdown (5th pct min equity):")
    worst_equity = float(np.percentile(min_eq_arr, 5))
    worst_pct = (worst_equity / INIT_CASH - 1) * 100
    print(f"    ${worst_equity:>12,.0f}  ({worst_pct:+.1f}%)")

    # Verdict
    p5_sharpe = float(np.percentile(sharpes_arr, 5))
    robust = p5_sharpe > 0.5 and pct_profitable >= 80.0
    verdict = "ROBUST" if robust else "FRAGILE"
    print("\n  Pass criteria: 5th-pct Sharpe > 0.5 AND >80% sims profitable")
    print(f"  5th-pct Sharpe: {p5_sharpe:.3f}  |  % profitable: {pct_profitable:.0f}%")
    print(f"  Verdict: {verdict}")


# ── Rolling Walk-Forward ─────────────────────────────────────────────────────


def rolling_walk_forward(
    close: pd.Series,
    open_prices: pd.Series,
    primary: pd.Series,
    atr14: pd.Series,
) -> None:
    """Rolling walk-forward with fixed MTF parameters (no re-fitting).

    Evaluates the current config/mtf.toml signals on successive 6-month
    OOS windows after 2-year anchored training periods.
    This tests regime robustness without overfitting risk.
    """
    print("\n" + "=" * 70)
    print(f"ROLLING WALK-FORWARD  ({TRAIN_YEARS}yr anchor / {TEST_MONTHS}mo test windows)")
    print("  Fixed parameters from config/mtf.toml — no re-fitting per window.")
    print("  Tests regime robustness across multiple time periods.")
    print("=" * 70)

    start = close.index[0]
    end = close.index[-1]
    anchor_start = start + pd.DateOffset(years=TRAIN_YEARS)

    results = []
    t = anchor_start
    window_num = 0

    while t + pd.DateOffset(months=TEST_MONTHS) <= end:
        test_start = t
        test_end = t + pd.DateOffset(months=TEST_MONTHS)
        window_num += 1

        mask = (close.index >= test_start) & (close.index < test_end)
        if mask.sum() < 50:
            t += pd.DateOffset(months=TEST_MONTHS)
            continue

        c_win = close[mask]
        o_win = open_prices[mask]
        sl_win = (atr14[mask] / c_win).ffill().bfill()

        le_win: pd.Series = primary[mask].eq(1)
        lx_win: pd.Series = primary[mask].ne(1)
        se_win: pd.Series = primary[mask].eq(-1)
        sx_win: pd.Series = primary[mask].ne(-1)

        if le_win.sum() == 0 and se_win.sum() == 0:
            t += pd.DateOffset(months=TEST_MONTHS)
            continue

        pf = run_vbt_portfolio(
            c_win,
            le_win,
            se_win,
            lx_win,
            sx_win,
            f"WFO_{window_num}",
            sl_stop=sl_win,
            open_prices=o_win,
        )

        n = pf.trades.count()
        if n == 0:
            t += pd.DateOffset(months=TEST_MONTHS)
            continue

        try:
            sharpe = float(pf.sharpe_ratio())
        except Exception:
            sharpe = float("nan")
        try:
            cagr = float(pf.annualized_return()) * 100
        except Exception:
            cagr = float("nan")
        max_dd = float(pf.max_drawdown()) * 100
        wr = float(pf.trades.win_rate()) * 100

        results.append(
            {
                "window": window_num,
                "period": f"{test_start.date()} to {test_end.date()}",
                "sharpe": sharpe,
                "cagr_pct": cagr,
                "max_dd_pct": max_dd,
                "win_rate_pct": wr,
                "trades": n,
            }
        )

        t += pd.DateOffset(months=TEST_MONTHS)

    if not results:
        print("  No windows with sufficient data.")
        return

    df = pd.DataFrame(results)

    print(
        f"\n  {'#':>3}  {'Period':<28}"
        f" {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>6} {'Trades':>6}"
    )
    print("  " + "-" * 68)
    for _, r in df.iterrows():
        sign = "+" if r["sharpe"] > 0 else " "
        print(
            f"  {r['window']:>3}  {r['period']:<28} {sign}{r['sharpe']:>6.2f} "
            f"{r['cagr_pct']:>+7.1f} {r['max_dd_pct']:>7.2f} "
            f"{r['win_rate_pct']:>6.1f} {r['trades']:>6,d}"
        )

    n_positive = (df["sharpe"] > 0).sum()
    n_total = len(df)

    # Check for consecutive negative windows
    max_consec_neg = 0
    cur = 0
    for s in df["sharpe"]:
        if s <= 0:
            cur += 1
            max_consec_neg = max(max_consec_neg, cur)
        else:
            cur = 0

    print(
        f"\n  Positive Sharpe windows: {n_positive}/{n_total} ({n_positive / n_total * 100:.0f}%)"
    )
    print(f"  Median Sharpe: {df['sharpe'].median():.3f}")
    print(f"  Max consecutive negative windows: {max_consec_neg}")
    print("\n  Pass criteria: >70% positive AND max consecutive negative <= 2")
    pass_rate = n_positive / n_total >= 0.70
    pass_consec = max_consec_neg <= 2
    verdict = "ROBUST" if (pass_rate and pass_consec) else "NEEDS REVIEW"
    print(f"  Verdict: {verdict}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("MTF CONFLUENCE ROBUSTNESS VALIDATION")
    print(f"  Monte Carlo (N={N_MONTE_CARLO:,}) + Rolling WFO ({TRAIN_YEARS}yr/{TEST_MONTHS}mo)")
    print(f"  Fees: {FEES * 10000:.1f} pips/side  |  Slippage: {SLIPPAGE * 10000:.1f} pip/side")
    print("=" * 70)

    # 1. Load config and data
    with open(PROJECT_ROOT / "config" / "mtf.toml", "rb") as f:
        mtf_cfg = tomllib.load(f)

    h1_df = _load_parquet(PAIR, "H1")
    close = h1_df["close"]
    open_prices = h1_df["open"]
    print(
        f"\n  Data: {len(h1_df):,} H1 bars  |  {h1_df.index[0].date()} to {h1_df.index[-1].date()}"
    )

    # 2. ATR14
    tr = pd.concat(
        [
            h1_df["high"] - h1_df["low"],
            (h1_df["high"] - h1_df["close"].shift(1)).abs(),
            (h1_df["low"] - h1_df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # 3. MTF signals
    print("\n  Computing MTF primary signals...")
    primary, _ = compute_mtf_signals(PAIR, mtf_cfg)
    print(
        f"  Long: {primary.eq(1).sum():,}  "
        f"Short: {primary.eq(-1).sum():,}  "
        f"Flat: {primary.eq(0).sum():,}"
    )

    # 4. OOS window for Monte Carlo (last 30% of data — same split as run_backtest_meta)
    IS_SPLIT = 0.70
    active_idx = primary[primary != 0].index
    split_at = active_idx[int(len(active_idx) * IS_SPLIT)]
    oos_mask = close.index >= split_at

    close_oos = close[oos_mask]
    open_oos = open_prices[oos_mask]
    sl_oos = (atr14[oos_mask] / close_oos).ffill().bfill()

    le_oos: pd.Series = primary[oos_mask].eq(1)
    lx_oos: pd.Series = primary[oos_mask].ne(1)
    se_oos: pd.Series = primary[oos_mask].eq(-1)
    sx_oos: pd.Series = primary[oos_mask].ne(-1)

    print(f"\n  OOS period: {close_oos.index[0].date()} to {close_oos.index[-1].date()}")

    pf_oos = run_vbt_portfolio(
        close_oos,
        le_oos,
        se_oos,
        lx_oos,
        sx_oos,
        "OOS Raw",
        sl_stop=sl_oos,
        open_prices=open_oos,
    )

    n_oos = pf_oos.trades.count()
    oos_sharpe = pf_oos.sharpe_ratio() if n_oos > 0 else float("nan")
    oos_cagr = pf_oos.annualized_return() * 100 if n_oos > 0 else float("nan")
    print(f"  OOS reference: Sharpe={oos_sharpe:.3f}  CAGR={oos_cagr:+.1f}%  Trades={n_oos:,}")

    # 5. Monte Carlo
    monte_carlo_shuffle(pf_oos)

    # 6. Rolling WFO across full dataset
    rolling_walk_forward(close, open_prices, primary, atr14)

    print("\n" + "=" * 70)
    print("Robustness validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
