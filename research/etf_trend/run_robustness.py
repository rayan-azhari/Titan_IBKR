"""run_robustness.py -- Stage 6: Monte Carlo + Rolling WFO Validation.

Two tests:
  Monte Carlo (N=1,000):  Shuffle trade order randomly. Gate: 5th-pct Sharpe > 0.5
                          AND > 80% simulations profitable.
  Rolling WFO (2yr/6mo): Slide anchor forward, re-optimise Stage 1 on each IS window,
                          validate on OOS. Gate: >70% windows positive AND max consecutive
                          negative <= 2.

Additional stress tests:
  - Remove 10 best trades: strategy still profitable?
  - 3x slippage: OOS Sharpe > 0.5?

Usage:
    uv run python research/etf_trend/run_robustness.py --instrument SPY
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

from research.etf_trend.run_portfolio import (  # noqa: E402
    build_signals,
    load_config,
    load_data,
    run_pf,
)

MONTE_CARLO_N = 1_000
WFO_ANCHOR_BARS = 252 * 2  # 2 years
WFO_STEP_BARS = 126  # ~6 months

# Quality gates
MC_MIN_5PCT_SHARPE = 0.5
MC_MIN_PROFITABLE_PCT = 0.80
WFO_MIN_POSITIVE_PCT = 0.70
WFO_MAX_CONSEC_NEG = 2


def run_oos_portfolio(instrument: str, config: dict, fees: float = 0.001) -> "vbt.Portfolio":
    """Re-run the OOS portfolio from config.

    Args:
        instrument: Symbol.
        config: Loaded TOML config dict.
        fees: Fee per side.

    Returns:
        VBT Portfolio (OOS only).
    """
    df = load_data(instrument)
    close = df["close"]
    split = int(len(close) * 0.70)
    oos_close = close.iloc[split:]

    all_entries, all_exits, _, all_sizes = build_signals(close, df, config)
    oos_entries = all_entries.iloc[split:]
    oos_exits = all_exits.iloc[split:]
    oos_sizes = all_sizes.iloc[split:]

    return run_pf(oos_close, oos_entries, oos_exits, oos_sizes, fees)


def monte_carlo_shuffle(pf: "vbt.Portfolio", n: int = MONTE_CARLO_N) -> dict:
    """Shuffle trade P&L (N times), compute distribution of Sharpe ratios.

    Args:
        pf: VBT Portfolio with trades.
        n: Number of shuffle iterations.

    Returns:
        Dict with sharpe distribution stats and pass/fail.
    """
    trades = pf.trades.records_readable
    if len(trades) == 0:
        return {"error": "No trades"}

    # Extract per-trade P&L (absolute)
    pnl = trades["PnL"].values
    init_cash = 10_000.0
    rng = np.random.default_rng(42)

    sharpes: list[float] = []
    profitable_count = 0

    for _ in range(n):
        shuffled_pnl = rng.permutation(pnl)
        cum_eq = np.cumsum(np.insert(shuffled_pnl, 0, init_cash))
        daily_ret = np.diff(cum_eq) / cum_eq[:-1]
        if daily_ret.std() < 1e-8:
            sharpes.append(0.0)
            continue
        sh = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        sharpes.append(sh)
        if cum_eq[-1] > init_cash:
            profitable_count += 1

    arr = np.array(sharpes)
    pct5 = float(np.percentile(arr, 5))
    profitable_pct = profitable_count / n
    gate_pass = pct5 > MC_MIN_5PCT_SHARPE and profitable_pct > MC_MIN_PROFITABLE_PCT

    return {
        "n": n,
        "mean_sharpe": round(float(arr.mean()), 3),
        "pct5_sharpe": round(pct5, 3),
        "pct95_sharpe": round(float(np.percentile(arr, 95)), 3),
        "profitable_pct": round(profitable_pct, 3),
        "gate_pass": gate_pass,
    }


def rolling_wfo(instrument: str, config: dict, df: pd.DataFrame) -> dict:
    """Rolling Walk-Forward Optimisation (2yr anchor / 6mo step).

    At each window, re-optimises Stage 1 fast/slow MA on IS, tests on OOS.
    Records whether each OOS window has positive Sharpe.

    Args:
        instrument: Symbol.
        config: Base config (used for all params except fast/slow MA in IS re-opt).
        df: Full OHLCV DataFrame.

    Returns:
        Dict with window results and pass/fail gate.
    """
    from research.etf_trend.run_stage2_decel import compute_ma  # local import
    from research.etf_trend.run_stage3_exits import compute_decel_composite  # local import

    close = df["close"]
    n = len(close)
    ma_type = config["ma_type"]
    decel_signals = config.get("decel_signals", [])
    exit_mode = config["exit_mode"]
    exit_decel_thresh = config["exit_decel_thresh"]

    # Simple candidate fast/slow grid for WFO re-opt
    fast_candidates = [20, 40, 50]
    slow_candidates = [150, 200, 250]

    window_results: list[dict] = []
    start = 0

    while start + WFO_ANCHOR_BARS + WFO_STEP_BARS <= n:
        is_end = start + WFO_ANCHOR_BARS
        oos_end = min(is_end + WFO_STEP_BARS, n)

        is_close = close.iloc[start:is_end]
        oos_close = close.iloc[is_end:oos_end]
        is_df = df.iloc[start:is_end]
        oos_df = df.iloc[is_end:oos_end]

        if len(oos_close) < 20:
            break

        # Mini re-opt on IS window
        best_is_sh, best_fast, best_slow = -np.inf, 50, 200
        for fast in fast_candidates:
            for slow in slow_candidates:
                if fast >= slow:
                    continue
                fa = compute_ma(is_close, fast, ma_type)
                sl = compute_ma(is_close, slow, ma_type)
                slow_full_is = compute_ma(close, slow, ma_type).iloc[start:is_end]
                decel_is = compute_decel_composite(is_close, is_df, slow_full_is, decel_signals)
                in_reg = (is_close > fa) & (is_close > sl)
                entry_s = (in_reg & (decel_is >= 0)).shift(1).fillna(False)
                exit_s = (~in_reg).shift(1).fillna(False)
                pf = vbt.Portfolio.from_signals(
                    is_close,
                    entries=entry_s,
                    exits=exit_s,
                    init_cash=10_000,
                    fees=0.001,
                    freq="1D",
                )
                sh = float(pf.sharpe_ratio())
                if sh > best_is_sh:
                    best_is_sh, best_fast, best_slow = sh, fast, slow

        # OOS with best IS params
        fa_oos = compute_ma(oos_close, best_fast, ma_type)
        sl_oos = compute_ma(oos_close, best_slow, ma_type)
        slow_full_oos = compute_ma(close, best_slow, ma_type).iloc[is_end:oos_end]
        decel_oos = compute_decel_composite(oos_close, oos_df, slow_full_oos, decel_signals)
        in_reg_oos = (oos_close > fa_oos) & (oos_close > sl_oos)
        entry_oos = (in_reg_oos & (decel_oos >= 0)).shift(1).fillna(False)

        if exit_mode == "A":
            exit_oos = (~in_reg_oos).shift(1).fillna(False)
        elif exit_mode == "B":
            exit_oos = (oos_close < fa_oos).shift(1).fillna(False)
        elif exit_mode == "C":
            exit_oos = (decel_oos < exit_decel_thresh).shift(1).fillna(False)
        else:
            exit_oos = ((~in_reg_oos) | (decel_oos < exit_decel_thresh)).shift(1).fillna(False)

        oos_pf = vbt.Portfolio.from_signals(
            oos_close,
            entries=entry_oos,
            exits=exit_oos,
            init_cash=10_000,
            fees=0.001,
            freq="1D",
        )
        oos_sh = float(oos_pf.sharpe_ratio())

        window_results.append(
            {
                "is_start": str(is_close.index[0].date()),
                "oos_start": str(oos_close.index[0].date()),
                "oos_end": str(oos_close.index[-1].date()),
                "best_fast": best_fast,
                "best_slow": best_slow,
                "is_sharpe": round(best_is_sh, 3),
                "oos_sharpe": round(oos_sh, 3),
                "oos_positive": oos_sh > 0,
            }
        )

        start += WFO_STEP_BARS

    if not window_results:
        return {"error": "Not enough data for WFO"}

    pos_count = sum(r["oos_positive"] for r in window_results)
    total_w = len(window_results)
    pos_pct = pos_count / total_w

    # Max consecutive negative windows
    consec = max_consec = 0
    for r in window_results:
        if not r["oos_positive"]:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    gate_pass = pos_pct >= WFO_MIN_POSITIVE_PCT and max_consec <= WFO_MAX_CONSEC_NEG

    return {
        "n_windows": total_w,
        "positive_windows": pos_count,
        "positive_pct": round(pos_pct, 3),
        "max_consec_negative": max_consec,
        "gate_pass": gate_pass,
        "windows": window_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 6: Monte Carlo + WFO.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend -- Stage 6: Robustness Validation")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    config = load_config(inst_lower)
    df = load_data(instrument)

    pf_oos = run_oos_portfolio(instrument, config)
    pf_oos_3x = run_oos_portfolio(instrument, config, fees=0.003)

    # -- Monte Carlo -------------------------------------------------------
    print(f"\n  Running Monte Carlo (N={MONTE_CARLO_N}) ...")
    mc = monte_carlo_shuffle(pf_oos, n=MONTE_CARLO_N)
    mc_pass = mc.get("gate_pass", False)
    print(f"  Mean Sharpe:       {mc.get('mean_sharpe', 'n/a')}")
    print(f"  5th-pct Sharpe:    {mc.get('pct5_sharpe', 'n/a')} (gate: > {MC_MIN_5PCT_SHARPE})")
    print(
        f"  Profitable sims:   {mc.get('profitable_pct', 'n/a'):.0%} "
        f"(gate: > {MC_MIN_PROFITABLE_PCT:.0%})"
    )
    print(f"  Monte Carlo gate:  {'PASS' if mc_pass else 'FAIL'}")

    # -- Remove top 10 trades ----------------------------------------------
    trades = pf_oos.trades.records_readable
    if len(trades) >= 10:
        top10_sum = trades.nlargest(10, "PnL")["PnL"].sum()
        remaining_pnl = pf_oos.total_profit() - top10_sum
        still_profitable = remaining_pnl > 0
        print(
            f"\n  Remove top 10 trades: P&L={remaining_pnl:.0f}  "
            f"{'Profitable' if still_profitable else 'Unprofitable'}"
        )
    else:
        print("\n  [Skip] Not enough trades for top-10 stress test.")

    # -- 3x slippage stress ------------------------------------------------
    sh_3x = float(pf_oos_3x.sharpe_ratio())
    print(f"\n  3x fees stress Sharpe: {sh_3x:.3f} (gate: > 0.5)")
    cost_stress_pass = sh_3x > 0.5

    # -- Rolling WFO -------------------------------------------------------
    print(f"\n  Running Rolling WFO (anchor={WFO_ANCHOR_BARS}bars / step={WFO_STEP_BARS}bars) ...")
    wfo = rolling_wfo(instrument, config, df)
    wfo_pass = wfo.get("gate_pass", False)

    if "error" not in wfo:
        print(f"  Windows:           {wfo['n_windows']}")
        print(
            f"  Positive:          {wfo['positive_windows']} "
            f"({wfo['positive_pct']:.0%}) (gate: > {WFO_MIN_POSITIVE_PCT:.0%})"
        )
        print(f"  Max consec. neg.:  {wfo['max_consec_negative']} (gate: <= {WFO_MAX_CONSEC_NEG})")
        print(f"  WFO gate:          {'PASS' if wfo_pass else 'FAIL'}")

        wfo_df = pd.DataFrame(wfo["windows"])
        wfo_path = REPORTS_DIR / f"etf_trend_stage6_{inst_lower}_wfo.csv"
        wfo_df.to_csv(wfo_path, index=False)
        print(f"  WFO windows saved: {wfo_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  WFO error: {wfo['error']}")
        wfo_pass = False

    # -- Final gate summary ------------------------------------------------
    print("\n  -- Quality Gate Summary --------------------------")
    gates = {
        "Monte Carlo (5th-pct > 0.5, >80% profitable)": mc_pass,
        "3x fees stress (Sharpe > 0.5)": cost_stress_pass,
        "Rolling WFO (>70% positive, max consec neg <= 2)": wfo_pass,
    }
    all_pass = True
    for gate_name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {gate_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  [PASS] All robustness gates passed.")
        print("  Strategy is validated. Proceed to paper trading (60 days).")
        print("  Runner: uv run python scripts/run_live_etf_trend.py  # port 4002")
    else:
        print("\n  [FAIL] One or more robustness gates failed.")
        print("  Review Stage 3 (exit mode) or Stage 4 (sizing parameters).")
    print("=" * 60)


if __name__ == "__main__":
    main()
