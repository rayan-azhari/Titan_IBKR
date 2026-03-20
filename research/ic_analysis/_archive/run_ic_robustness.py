"""run_ic_robustness.py -- Stage 5: Robustness Validation for IC MTF Strategy.

Mirrors ETF Trend Stage 6 (run_robustness.py) adapted for forex IC MTF signals.

Four quality gates (all must PASS before Phase 6 live implementation):
  1. Monte Carlo (N=1,000)  : Shuffle combined long+short trade P&L 1,000 times.
                              Gate: 5th-pct Sharpe > 0.5 AND > 80% of sims profitable.
  2. Remove top-N trades    : Drop 10 largest winning trades from combined P&L.
                              Gate: remaining cumulative P&L still positive.
  3. 3x slippage stress     : Re-run OOS with slippage = 1.5 pip (3x default 0.5 pip).
                              Gate: OOS combined Sharpe > 0.5.
  4. WFO consecutive folds  : Read existing wfo_{slug}.csv from Phase 4.
                              Gate: max consecutive negative OOS folds <= 2.

Runs on all 6 instruments by default; use --instrument to target one pair.

Usage:
    uv run python research/ic_analysis/run_ic_robustness.py
    uv run python research/ic_analysis/run_ic_robustness.py --instrument EUR_USD
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_ic_backtest import (  # noqa: E402
    DEFAULT_RISK_PCT,
    DEFAULT_SIGNALS,
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_STOP_ATR,
    DEFAULT_TFS,
    INIT_CASH,
    IS_RATIO,
    PIP_SIZE,
    SPREAD_DEFAULTS,
    SWAP_DEFAULTS,
    _build_and_align,
    build_composite,
    build_size_array,
    zscore_normalise,
)

# ---------------------------------------------------------------------------
# Gate thresholds (mirrors ETF Trend Stage 6)
# ---------------------------------------------------------------------------
MC_N = 1_000
MC_MIN_5PCT_SHARPE = 0.5
MC_MIN_PROFITABLE_PCT = 0.80
TOP_N_REMOVE = 10
STRESS_SLIPPAGE_MULT = 3.0
STRESS_MIN_SHARPE = 0.5
WFO_MAX_CONSEC_NEG = 2
DEFAULT_THRESHOLD = 0.75

ALL_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "AUD_JPY", "USD_CHF"]


# ---------------------------------------------------------------------------
# Portfolio builder — returns raw VBT objects for downstream tests
# ---------------------------------------------------------------------------


def _build_oos_portfolios(
    instrument: str,
    slippage_mult: float = 1.0,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple:
    """Build OOS long + short VBT portfolios for IC MTF strategy.

    Returns (pf_long, pf_short, oos_close, oos_size, swap_long, swap_short, pip_size).
    """
    pip_size = PIP_SIZE.get(instrument, 0.0001)
    spread = SPREAD_DEFAULTS.get(instrument, 0.00005)
    swap_long = SWAP_DEFAULTS.get(instrument, {}).get("long", -0.5)
    swap_short = SWAP_DEFAULTS.get(instrument, {}).get("short", 0.3)
    slippage = DEFAULT_SLIPPAGE_PIPS * slippage_mult * pip_size

    tfs = DEFAULT_TFS
    base_tf = tfs[-1]

    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]

    n = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True
    oos_mask = is_mask == False  # noqa: E712

    composite = build_composite(tf_signals, base_close, tfs, DEFAULT_SIGNALS, is_mask)
    composite_z = zscore_normalise(composite, is_mask)
    size = build_size_array(base_df, base_close, DEFAULT_RISK_PCT, DEFAULT_STOP_ATR)

    oos_close = base_close[oos_mask]
    oos_z = composite_z[oos_mask]
    oos_size = size[oos_mask]

    sig = oos_z.shift(1).fillna(0.0)
    size_arr = oos_size.reindex(oos_close.index).fillna(0.0).values
    med_close = float(oos_close.median()) or 1.0
    vbt_fees = spread / med_close
    vbt_slip = slippage / med_close

    false_s = pd.Series(False, index=oos_close.index)
    pf_long = vbt.Portfolio.from_signals(
        oos_close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )
    pf_short = vbt.Portfolio.from_signals(
        oos_close,
        entries=false_s,
        exits=false_s,
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )

    return pf_long, pf_short, oos_close, oos_size, swap_long, swap_short, pip_size


def _combined_trade_returns(pf_long, pf_short) -> np.ndarray | None:
    """Extract per-trade % returns from both portfolios using VBT's Return column.

    VBT records_readable["Return"] is PnL / initial position value — a normalised
    per-trade return that is comparable regardless of when in the equity curve the
    trade occurs. This avoids the compounding distortion that makes absolute P&L
    unusable for Monte Carlo Sharpe calculations.
    """
    parts = []
    for pf in (pf_long, pf_short):
        if pf.trades.count() > 0:
            parts.append(pf.trades.records_readable["Return"].values)
    if not parts:
        return None
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Test 1: Monte Carlo (shuffled trade % returns)
# ---------------------------------------------------------------------------


def monte_carlo_shuffle(pf_long, pf_short, oos_bars: int, n: int = MC_N) -> dict:
    """Shuffle combined trade % returns N times, compute Sharpe distribution.

    Uses per-trade % returns (VBT's Return column) so Sharpe is not distorted by
    compounding. Annualises by trades-per-year derived from the OOS window length.
    Gate: 5th-pct Sharpe > 0.5 AND > 80% of simulations profitable (sum > 0).
    """
    trade_ret = _combined_trade_returns(pf_long, pf_short)
    if trade_ret is None or len(trade_ret) < 10:
        return {"error": "Insufficient trades for Monte Carlo"}

    oos_years = oos_bars / (252 * 24)       # forex H1: 252 trading days * 24h
    trades_per_year = len(trade_ret) / max(oos_years, 0.1)
    rng = np.random.default_rng(42)
    sharpes: list[float] = []
    profitable: int = 0

    for _ in range(n):
        shuffled = rng.permutation(trade_ret)
        mu, sigma = float(shuffled.mean()), float(shuffled.std())
        sh = mu / sigma * np.sqrt(trades_per_year) if sigma > 1e-10 else 0.0
        sharpes.append(float(sh))
        if float(shuffled.sum()) > 0:       # arithmetic profitability check
            profitable = profitable + 1

    arr = np.array(sharpes)
    pct5 = float(np.percentile(arr, 5))
    profitable_pct = float(profitable) / n
    gate_pass = pct5 > MC_MIN_5PCT_SHARPE and profitable_pct > MC_MIN_PROFITABLE_PCT

    return {
        "n_trades": int(len(trade_ret)),
        "mean_sharpe": round(float(arr.mean()), 3),
        "pct5_sharpe": round(pct5, 3),
        "pct95_sharpe": round(float(np.percentile(arr, 95)), 3),
        "profitable_pct": round(profitable_pct, 3),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Test 2: Remove top-N winning trades
# ---------------------------------------------------------------------------


def remove_top_n(pf_long, pf_short, n: int = TOP_N_REMOVE) -> dict:
    """Drop the N largest winning trades by % return; check arithmetic sum still positive.

    Uses per-trade % returns (same as Monte Carlo) so the test is not distorted by
    compounding — removing early small trades vs. late large ones is equivalent.
    Gate: sum of remaining trade % returns > 0.
    """
    trade_ret = _combined_trade_returns(pf_long, pf_short)
    if trade_ret is None:
        return {"error": "No trades"}
    if len(trade_ret) < n + 1:
        return {"error": f"Only {len(trade_ret)} trades — need > {n}"}

    total_ret = float(trade_ret.sum())
    top_n_ret = float(np.sort(trade_ret)[-n:].sum())
    remaining_ret = total_ret - top_n_ret
    gate_pass = remaining_ret > 0

    return {
        "total_trades": int(len(trade_ret)),
        "total_ret_pct": round(total_ret * 100, 2),
        "top_n_ret_pct": round(top_n_ret * 100, 2),
        "remaining_ret_pct": round(remaining_ret * 100, 2),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Test 3: 3x slippage stress
# ---------------------------------------------------------------------------


def stress_triple_slippage(instrument: str) -> dict:
    """Re-run OOS with 3x slippage (1.5 pip instead of 0.5 pip).

    Gate: combined OOS Sharpe > 0.5.
    """
    pf_long_3x, pf_short_3x, _, _, swap_long, swap_short, pip_size = _build_oos_portfolios(
        instrument, slippage_mult=STRESS_SLIPPAGE_MULT
    )
    long_sh = float(pf_long_3x.sharpe_ratio())
    short_sh = float(pf_short_3x.sharpe_ratio())
    combined_sh = (long_sh + short_sh) / 2
    gate_pass = combined_sh > STRESS_MIN_SHARPE

    return {
        "long_sharpe": round(long_sh, 3),
        "short_sharpe": round(short_sh, 3),
        "combined_sharpe": round(combined_sh, 3),
        "slippage_pips": round(DEFAULT_SLIPPAGE_PIPS * STRESS_SLIPPAGE_MULT, 2),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Test 4: WFO consecutive negative folds (reads Phase 4 output CSV)
# ---------------------------------------------------------------------------


def check_wfo_consecutive(instrument: str) -> dict:
    """Read Phase 4 WFO CSV and check max consecutive negative OOS folds.

    Gate: max consecutive negative folds <= 2.
    """
    slug = instrument.lower()
    wfo_path = REPORTS_DIR / f"wfo_{slug}.csv"
    if not wfo_path.exists():
        return {"error": f"WFO CSV not found: {wfo_path.name}. Run run_wfo.py first."}

    df = pd.read_csv(wfo_path)
    if "oos_sharpe" not in df.columns:
        return {"error": f"oos_sharpe column missing in {wfo_path.name}"}

    sharpes = df["oos_sharpe"].tolist()
    n_folds = len(sharpes)
    n_pos = sum(s > 0 for s in sharpes)

    consec = max_consec = 0
    for s in sharpes:
        if s <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    gate_pass = max_consec <= WFO_MAX_CONSEC_NEG

    return {
        "n_folds": n_folds,
        "positive_folds": n_pos,
        "positive_pct": round(n_pos / n_folds, 3) if n_folds > 0 else 0.0,
        "max_consec_negative": max_consec,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Per-instrument runner
# ---------------------------------------------------------------------------


def run_robustness(instrument: str) -> dict:
    """Run all 4 robustness tests for one instrument. Returns gate summary."""
    W = 70
    print(f"\n{'=' * W}")
    print(f"  IC MTF ROBUSTNESS -- {instrument}")
    print(f"{'=' * W}")

    # Build base OOS portfolios
    print("  Building OOS portfolios (standard costs)...")
    try:
        pf_long, pf_short, oos_close, oos_size, sl_pip, ss_pip, pip_sz = (
            _build_oos_portfolios(instrument)
        )
    except Exception as e:
        print(f"  ERROR building portfolios: {e}")
        return {"instrument": instrument, "error": str(e), "all_pass": False}

    long_sh = float(pf_long.sharpe_ratio())
    short_sh = float(pf_short.sharpe_ratio())
    print(
        f"  Baseline OOS Sharpe: long={long_sh:.2f}  short={short_sh:.2f}"
        f"  combined={(long_sh + short_sh) / 2:.2f}"
    )

    # -- Test 1: Monte Carlo -----------------------------------------------
    print(f"\n  [1/4] Monte Carlo (N={MC_N:,} shuffle iterations)...")
    mc = monte_carlo_shuffle(pf_long, pf_short, oos_bars=len(oos_close), n=MC_N)
    if "error" in mc:
        print(f"  SKIP: {mc['error']}")
        mc["gate_pass"] = False
    else:
        mc_label = "PASS" if mc["gate_pass"] else "FAIL"
        print(f"  Trades: {mc['n_trades']}  |  Mean Sharpe: {mc['mean_sharpe']:.3f}")
        print(
            f"  5th-pct Sharpe: {mc['pct5_sharpe']:.3f} (gate: > {MC_MIN_5PCT_SHARPE})"
            f"  95th-pct: {mc['pct95_sharpe']:.3f}"
        )
        print(
            f"  Profitable sims: {mc['profitable_pct']:.1%}"
            f" (gate: > {MC_MIN_PROFITABLE_PCT:.0%})  --> [{mc_label}]"
        )

    # -- Test 2: Remove top-N trades ---------------------------------------
    print(f"\n  [2/4] Remove top {TOP_N_REMOVE} winning trades...")
    top_n = remove_top_n(pf_long, pf_short, n=TOP_N_REMOVE)
    if "error" in top_n:
        print(f"  SKIP: {top_n['error']}")
        top_n["gate_pass"] = False
    else:
        topn_label = "PASS" if top_n["gate_pass"] else "FAIL"
        print(
            f"  Total return: {top_n['total_ret_pct']:.1f}%  |  "
            f"Top-{TOP_N_REMOVE} contribution: {top_n['top_n_ret_pct']:.1f}%"
        )
        print(
            f"  Remaining return: {top_n['remaining_ret_pct']:.1f}%"
            f" ({'positive' if top_n['gate_pass'] else 'negative'})  --> [{topn_label}]"
        )

    # -- Test 3: 3x slippage -----------------------------------------------
    print(f"\n  [3/4] 3x slippage stress ({DEFAULT_SLIPPAGE_PIPS * STRESS_SLIPPAGE_MULT:.1f} pip)...")
    stress = stress_triple_slippage(instrument)
    stress_label = "PASS" if stress["gate_pass"] else "FAIL"
    print(
        f"  Sharpe: long={stress['long_sharpe']:.3f}  short={stress['short_sharpe']:.3f}"
        f"  combined={stress['combined_sharpe']:.3f} (gate: > {STRESS_MIN_SHARPE})"
        f"  --> [{stress_label}]"
    )

    # -- Test 4: WFO consecutive negatives ----------------------------------
    print("\n  [4/4] WFO consecutive negative folds (from Phase 4 CSV)...")
    wfo = check_wfo_consecutive(instrument)
    if "error" in wfo:
        print(f"  SKIP: {wfo['error']}")
        wfo["gate_pass"] = False
    else:
        wfo_label = "PASS" if wfo["gate_pass"] else "FAIL"
        print(
            f"  Folds: {wfo['n_folds']}  |  Positive: {wfo['positive_folds']}"
            f" ({wfo['positive_pct']:.0%})"
        )
        print(
            f"  Max consecutive negative: {wfo['max_consec_negative']}"
            f" (gate: <= {WFO_MAX_CONSEC_NEG})  --> [{wfo_label}]"
        )

    # -- Gate summary -------------------------------------------------------
    all_pass = all([mc["gate_pass"], top_n["gate_pass"], stress["gate_pass"], wfo["gate_pass"]])

    print(f"\n  {'- ' * 35}")
    print(f"  GATE SUMMARY -- {instrument}")
    print(f"  {'- ' * 35}")
    gates = {
        f"Monte Carlo 5th-pct > {MC_MIN_5PCT_SHARPE}, >{MC_MIN_PROFITABLE_PCT:.0%} profitable":
            mc["gate_pass"],
        f"Remove top {TOP_N_REMOVE} trades: remaining P&L > 0": top_n["gate_pass"],
        f"3x slippage: Sharpe > {STRESS_MIN_SHARPE}": stress["gate_pass"],
        f"WFO max consec. negative <= {WFO_MAX_CONSEC_NEG}": wfo["gate_pass"],
    }
    for gate_name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {gate_name}")

    verdict = "ALL GATES PASS -- cleared for Phase 6" if all_pass else "GATES FAILED -- do not proceed"
    print(f"\n  Verdict: {verdict}")

    return {
        "instrument": instrument,
        "baseline_long_sharpe": round(long_sh, 3),
        "baseline_short_sharpe": round(short_sh, 3),
        "baseline_combined_sharpe": round((long_sh + short_sh) / 2, 3),
        "mc_pct5_sharpe": mc.get("pct5_sharpe"),
        "mc_profitable_pct": mc.get("profitable_pct"),
        "mc_pass": mc["gate_pass"],
        "topn_remaining_pnl": top_n.get("remaining_pnl"),
        "topn_pass": top_n["gate_pass"],
        "stress_combined_sharpe": stress.get("combined_sharpe"),
        "stress_pass": stress["gate_pass"],
        "wfo_max_consec_neg": wfo.get("max_consec_negative"),
        "wfo_pass": wfo["gate_pass"],
        "all_pass": all_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="IC MTF Robustness Validation (Stage 5)")
    parser.add_argument(
        "--instrument",
        default=None,
        help="Single instrument (e.g. EUR_USD). Omit to run all 6 pairs.",
    )
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ALL_INSTRUMENTS

    W = 70
    print("=" * W)
    print("  IC MTF STRATEGY -- ROBUSTNESS VALIDATION")
    print("  Four gates: Monte Carlo | Top-N removal | 3x slippage | WFO folds")
    print("=" * W)

    all_results: list[dict] = []
    for inst in instruments:
        result = run_robustness(inst)
        all_results.append(result)

    # -- Final multi-instrument summary -------------------------------------
    if len(all_results) > 1:
        print(f"\n\n{'=' * W}")
        print("  FINAL SUMMARY -- ALL INSTRUMENTS")
        print("=" * W)
        print(
            f"  {'Instrument':<12} {'CombSh':>7} {'MC':>5} {'TopN':>5}"
            f" {'3xSlip':>7} {'WFO':>5} {'ALL':>6}"
        )
        print(f"  {'-' * 50}")
        for r in all_results:
            if "error" in r:
                print(f"  {r['instrument']:<12}  ERROR: {r['error']}")
                continue
            mc_s = "PASS" if r["mc_pass"] else "FAIL"
            tn_s = "PASS" if r["topn_pass"] else "FAIL"
            st_s = "PASS" if r["stress_pass"] else "FAIL"
            wf_s = "PASS" if r["wfo_pass"] else "FAIL"
            al_s = "PASS" if r["all_pass"] else "FAIL"
            cs = r["baseline_combined_sharpe"]
            print(
                f"  {r['instrument']:<12} {cs:>7.2f} {mc_s:>5} {tn_s:>5}"
                f" {st_s:>7} {wf_s:>5} {al_s:>6}"
            )

        total_pass = sum(r.get("all_pass", False) for r in all_results)
        print(f"\n  {total_pass}/{len(all_results)} instruments cleared all gates.")

        if total_pass == len(all_results):
            print("\n  ALL INSTRUMENTS PASSED -- proceed to Phase 6 (live implementation).")
        else:
            failed = [r["instrument"] for r in all_results if not r.get("all_pass")]
            print(f"\n  FAILED: {', '.join(failed)} -- review before going live.")

    # Save results CSV
    df_out = pd.DataFrame(all_results)
    out_path = REPORTS_DIR / "ic_robustness_summary.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
