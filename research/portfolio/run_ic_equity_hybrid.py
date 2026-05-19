"""ic_equity_daily V3.7 hybrid (L52 plateau + DSR + L53 early gate).

Critical-review follow-on to PR #9. Yesterday's L65+L67 closure shipped a
"matrix-improving CONDITIONAL" verdict without running the L52 plateau
sweep or DSR deflation -- the canonical Wave B audit used a single fixed
(threshold, RSI_period) cell per ticker. The +0.06 portfolio Sharpe lift
could be a single-peak artefact OR a top-3-of-7 selection-bias inflation.

This script runs the V3.7 hybrid properly:

    1. Per-ticker 3x3 sweep over (threshold, RSI_period). 9 cells x 7
       tickers = 63 sweep cells total.
    2. L52 plateau pre-flight per ticker: canonical cell + its 4
       (or fewer at corners) immediate neighbours must have Sharpe
       spread <= 30% (L27 strict) or <= 50% (H1).
    3. DSR per ticker: deflate canonical Sharpe using sweep variance
       (Bailey + Lopez de Prado 2014, skew + kurt from actual returns).
    4. Cross-ticker DSR: the top-3 selection across 7 tickers is itself
       an N=7 trial; deflate the top-3 BASKET Sharpe accordingly.
    5. Survivors get the L65+L67 re-run on the plateau-validated signal.

Output: `.tmp/reports/ic_equity_hybrid/result_log.md`

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/run_ic_equity_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.exploration.audit_ic_equity_daily import (  # noqa: E402
    PERIODS_PER_YEAR,
    THRESHOLDS,
    TICKERS_LIVE,
    _load_d,
)
from research.portfolio.joint_evaluation import (  # noqa: E402
    benchmark_6040_returns,
    build_portfolio,
    evaluate_portfolio_vs_benchmark,
    gem_j5_returns,
    turtle_cat_returns_daily,
)
from titan.research.framework import (  # noqa: E402
    assess_joint_ruin,
)
from titan.research.framework.dsr import (  # noqa: E402
    deflated_sharpe,
    sr_var_from_sweep,
)
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ic_equity_hybrid"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Hybrid sweep grid (per ticker): threshold steps and RSI-period steps.
# Threshold neighbours = canonical +/- 0.25; RSI neighbours = {14, 21, 28}.
# 3 x 3 = 9 cells per ticker.
RSI_PERIODS = [14, 21, 28]
THRESHOLD_DELTA = 0.25

# Plateau gates (V3.6/V3.7 framework standard).
PLATEAU_GATE_H1 = 0.50
PLATEAU_GATE_STRICT = 0.30

# DSR deployment gate: deflated probability must exceed this.
DSR_DEPLOY_GATE = 0.95


def _modified_ic_equity_returns(
    df: pd.DataFrame,
    *,
    threshold: float,
    rsi_period: int,
    is_until_idx: int | None = None,
) -> pd.Series:
    """Inline reimpl of ic_equity_returns with a swappable RSI period.

    The original audit hardcodes RSI(21). For the hybrid sweep we also
    vary the RSI period, so we cannot reuse the audit function directly.
    Equivalent logic otherwise: IS-fit ic_sign + IS mu/sigma when
    is_until_idx is set; full-series otherwise.
    """
    from scipy.stats import spearmanr

    close = df["close"]

    # Wilder RSI.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)
    rsi_dev = (rsi - 50.0).fillna(0.0)

    fwd_ret = np.log(close.shift(-1) / close).fillna(0.0)
    sub_signal = rsi_dev if is_until_idx is None else rsi_dev.iloc[:is_until_idx]
    sub_fwd = fwd_ret if is_until_idx is None else fwd_ret.iloc[:is_until_idx]
    both = pd.concat([sub_signal, sub_fwd], axis=1).dropna()
    if len(both) < 30:
        ic_sign = 1.0
    else:
        r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        ic_sign = float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0

    composite = rsi_dev * ic_sign
    sub_comp = composite if is_until_idx is None else composite.iloc[:is_until_idx]
    mu = float(sub_comp.mean())
    sigma = float(sub_comp.std()) or 1.0
    z = (composite - mu) / sigma

    pos = np.zeros(len(z), dtype=float)
    state = 0
    arr = z.to_numpy()
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos[i] = float(state)
            continue
        if state == 0 and arr[i] > threshold:
            state = 1
        elif state == 1 and arr[i] <= 0:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=z.index)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (1.0 / 10_000.0)  # 1 bps per turnover, same as audit COST_BPS=1.0
    return (gross - cost).rename("ret")


def _strict_oos_for_cell(df: pd.DataFrame, threshold: float, rsi_period: int) -> pd.Series:
    n = len(df)
    is_split = n // 2
    rets = _modified_ic_equity_returns(
        df, threshold=threshold, rsi_period=rsi_period, is_until_idx=is_split
    )
    oos = rets.iloc[is_split:]
    oos.index = pd.to_datetime(oos.index).tz_localize(None).normalize()
    return oos.groupby(oos.index).last().sort_index()


def sweep_ticker(ticker: str) -> dict:
    """3x3 sweep over (threshold, RSI_period) for one ticker.

    Returns a dict with per-cell Sharpes, canonical/neighbours, plateau
    spread, DSR, and the canonical strict-OOS return series.
    """
    df = _load_d(ticker)
    thr_canon = THRESHOLDS[ticker]
    rsi_canon = 21
    threshold_grid = [
        round(thr_canon - THRESHOLD_DELTA, 4),
        thr_canon,
        round(thr_canon + THRESHOLD_DELTA, 4),
    ]
    threshold_grid = [t for t in threshold_grid if t > 0]  # drop non-positive

    sweep_cells: list[dict] = []
    canonical_returns: pd.Series | None = None
    for thr in threshold_grid:
        for rsi_p in RSI_PERIODS:
            try:
                oos = _strict_oos_for_cell(df, threshold=thr, rsi_period=rsi_p)
                sr = float(sharpe(oos, periods_per_year=PERIODS_PER_YEAR))
                ci_lo, ci_hi = bootstrap_sharpe_ci(
                    oos, periods_per_year=PERIODS_PER_YEAR, n_resamples=1000, seed=42
                )
                is_canonical = thr == thr_canon and rsi_p == rsi_canon
                sweep_cells.append(
                    {
                        "threshold": thr,
                        "rsi_period": rsi_p,
                        "sharpe": sr,
                        "ci_lo": float(ci_lo),
                        "ci_hi": float(ci_hi),
                        "n": int(len(oos)),
                        "is_canonical": is_canonical,
                        "returns": oos,
                    }
                )
                if is_canonical:
                    canonical_returns = oos
            except Exception as e:  # noqa: BLE001
                sweep_cells.append(
                    {
                        "threshold": thr,
                        "rsi_period": rsi_p,
                        "sharpe": float("nan"),
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                        "n": 0,
                        "is_canonical": thr == thr_canon and rsi_p == rsi_canon,
                        "returns": None,
                        "error": str(e),
                    }
                )

    # ── L52 plateau pre-flight ──
    # Plateau set = canonical + its 4-connected (threshold +/-1 step OR
    # RSI +/-1 step). Up to 5 cells (canonical + 4 neighbours).
    plateau_cells = [
        c
        for c in sweep_cells
        if c.get("returns") is not None
        and (
            (c["threshold"] == thr_canon and c["rsi_period"] in RSI_PERIODS)
            or (c["rsi_period"] == rsi_canon and c["threshold"] in threshold_grid)
        )
    ]
    plateau_sharpes = [c["sharpe"] for c in plateau_cells if not np.isnan(c["sharpe"])]
    if len(plateau_sharpes) >= 3:
        plateau_mean = float(np.mean(plateau_sharpes))
        plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(
            abs(plateau_mean), 1e-9
        )
    else:
        plateau_mean = float("nan")
        plateau_spread = float("nan")

    # ── DSR per ticker (within-ticker sweep variance) ──
    sweep_sharpes_valid = [c["sharpe"] for c in sweep_cells if not np.isnan(c["sharpe"])]
    sr_var = sr_var_from_sweep(sweep_sharpes_valid)
    canonical_cell = next((c for c in sweep_cells if c["is_canonical"]), None)
    if canonical_cell is not None and canonical_returns is not None and sr_var > 0:
        dsr = deflated_sharpe(
            sr_hat=canonical_cell["sharpe"],
            sr_var_across_trials=sr_var,
            returns=canonical_returns,
            n_trials=len(sweep_sharpes_valid),
            survivors_only=False,
        )
    else:
        dsr = None

    return {
        "ticker": ticker,
        "canonical_threshold": thr_canon,
        "canonical_rsi_period": rsi_canon,
        "sweep_cells": sweep_cells,
        "plateau_mean": plateau_mean,
        "plateau_spread": plateau_spread,
        "plateau_h1_pass": (
            plateau_spread <= PLATEAU_GATE_H1 if not np.isnan(plateau_spread) else False
        ),
        "plateau_strict_pass": (
            plateau_spread <= PLATEAU_GATE_STRICT if not np.isnan(plateau_spread) else False
        ),
        "dsr": dsr,
        "canonical_returns": canonical_returns,
    }


def equal_weight_basket(returns_by_ticker: dict[str, pd.Series]) -> pd.Series:
    df = pd.concat(returns_by_ticker, axis=1).fillna(0.0)
    n = df.shape[1]
    return (df.sum(axis=1) / n).rename("basket")


def main() -> None:
    print("=" * 88)
    print("ic_equity_daily V3.7 HYBRID -- L52 plateau + DSR + L53 early gate")
    print("=" * 88)

    # ── [1/5] Per-ticker sweep ──
    print("\n[1/5] Per-ticker 3x3 sweep (threshold +/- 0.25; RSI in {14,21,28})")
    results_by_ticker: dict[str, dict] = {}
    for tkr in TICKERS_LIVE:
        print(f"\n  --- {tkr} (canonical thr={THRESHOLDS[tkr]}, RSI=21) ---")
        try:
            r = sweep_ticker(tkr)
            results_by_ticker[tkr] = r
            for c in r["sweep_cells"]:
                star = " *" if c["is_canonical"] else ""
                if np.isnan(c["sharpe"]):
                    print(
                        f"    thr={c['threshold']:>5.2f} RSI={c['rsi_period']:>2d}  "
                        f"FAILED ({c.get('error', 'n/a')})"
                    )
                else:
                    print(
                        f"    thr={c['threshold']:>5.2f} RSI={c['rsi_period']:>2d}  "
                        f"Sharpe={c['sharpe']:>+7.4f}  CI=[{c['ci_lo']:>+6.3f},"
                        f"{c['ci_hi']:>+6.3f}]{star}"
                    )
            print(
                f"    plateau spread={r['plateau_spread'] * 100:.1f}%  "
                f"(H1<={PLATEAU_GATE_H1 * 100:.0f}%: "
                f"{'PASS' if r['plateau_h1_pass'] else 'FAIL'}; "
                f"strict<={PLATEAU_GATE_STRICT * 100:.0f}%: "
                f"{'PASS' if r['plateau_strict_pass'] else 'FAIL'})"
            )
            if r["dsr"] is not None:
                print(
                    f"    DSR: canonical Sharpe={r['dsr'].sharpe:+.4f}, "
                    f"E[max SR]={r['dsr'].e_max_sr:+.4f}, "
                    f"z={r['dsr'].z:+.3f}, p={r['dsr'].dsr_prob:.4f} "
                    f"({'PASS' if r['dsr'].dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'} "
                    f"@ {DSR_DEPLOY_GATE:.0%})"
                )
        except FileNotFoundError as e:
            print(f"    {tkr}: missing parquet ({e})")

    # ── [2/5] Plateau + DSR survivors ──
    print("\n[2/5] Plateau + DSR survivors (L52 H1 plateau AND DSR p >= 0.95)")
    survivors: list[str] = []
    summary_rows: list[dict] = []
    for tkr, r in results_by_ticker.items():
        plateau_ok = r["plateau_h1_pass"]
        dsr_ok = r["dsr"] is not None and r["dsr"].dsr_prob >= DSR_DEPLOY_GATE
        is_survivor = plateau_ok and dsr_ok
        if is_survivor:
            survivors.append(tkr)
        canonical = next((c for c in r["sweep_cells"] if c["is_canonical"]), None)
        summary_rows.append(
            {
                "ticker": tkr,
                "canonical_sharpe": canonical["sharpe"] if canonical else float("nan"),
                "canonical_ci_lo": canonical["ci_lo"] if canonical else float("nan"),
                "plateau_spread_pct": (
                    r["plateau_spread"] * 100 if not np.isnan(r["plateau_spread"]) else float("nan")
                ),
                "plateau_h1": plateau_ok,
                "dsr_prob": r["dsr"].dsr_prob if r["dsr"] else float("nan"),
                "dsr_pass": dsr_ok,
                "survivor": is_survivor,
            }
        )
        print(
            f"  {tkr:>7s}: plateau={'PASS' if plateau_ok else 'FAIL'} "
            f"({r['plateau_spread'] * 100:>5.1f}%)  DSR p="
            f"{(r['dsr'].dsr_prob if r['dsr'] else float('nan')):>5.3f} "
            f"({'PASS' if dsr_ok else 'FAIL'})  "
            f"-> {'SURVIVOR' if is_survivor else 'CULL'}"
        )

    if not survivors:
        print("\n*** NO SURVIVORS *** -- top-3 selection would be entirely from")
        print("    plateau- or DSR-failing cells. Prior PR #9 CONDITIONAL verdict")
        print("    is DOWNGRADED. ic_equity_daily strict-OOS Sharpe is an artefact")
        print("    of point selection.")
        _write_log(REPORTS_DIR, results_by_ticker, summary_rows, survivors, None, None, None)
        return

    # ── [3/5] Cross-ticker DSR on basket selection ──
    print("\n[3/5] Cross-ticker DSR on top-3 selection (treats 7-ticker pool as N=7 trials)")
    pool_sharpes = [r["canonical_sharpe"] for r in summary_rows]
    sr_var_pool = sr_var_from_sweep(pool_sharpes)
    print(f"  pool sharpes: {[round(s, 3) for s in pool_sharpes]}")
    print(f"  sr_var across pool = {sr_var_pool:.5f}")

    # Re-build basket from survivors only (could be fewer than 3).
    basket_tickers = (
        sorted(survivors, key=lambda t: results_by_ticker[t]["dsr"].sharpe, reverse=True)[:3]
        if len(survivors) >= 3
        else survivors
    )
    print(f"  basket tickers (rank-3 by canonical Sharpe among survivors): {basket_tickers}")
    basket_returns_by_ticker = {
        t: results_by_ticker[t]["canonical_returns"] for t in basket_tickers
    }
    basket = equal_weight_basket(basket_returns_by_ticker)
    basket_sr = float(sharpe(basket, periods_per_year=PERIODS_PER_YEAR))
    basket_ci_lo, basket_ci_hi = bootstrap_sharpe_ci(
        basket, periods_per_year=PERIODS_PER_YEAR, n_resamples=2000, seed=42
    )
    print(f"  basket Sharpe={basket_sr:+.4f}  CI=[{basket_ci_lo:+.3f},{basket_ci_hi:+.3f}]")
    # Cross-ticker DSR on basket: n_trials = number of distinct ticker candidates (7).
    if sr_var_pool > 0:
        basket_dsr = deflated_sharpe(
            sr_hat=basket_sr,
            sr_var_across_trials=sr_var_pool,
            returns=basket,
            n_trials=len(pool_sharpes),
            survivors_only=False,
        )
        print(
            f"  basket DSR: Sharpe={basket_dsr.sharpe:+.4f}, "
            f"E[max SR]={basket_dsr.e_max_sr:+.4f}, "
            f"z={basket_dsr.z:+.3f}, p={basket_dsr.dsr_prob:.4f} "
            f"({'PASS' if basket_dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'})"
        )
    else:
        basket_dsr = None

    # ── [4/5] L65 + L67 re-run on plateau+DSR-validated basket ──
    print("\n[4/5] L65 + L67 re-run on validated basket")
    gem = gem_j5_returns().dropna()
    gem.index = pd.to_datetime(gem.index).tz_localize(None).normalize()
    gem = gem.groupby(gem.index).last().sort_index()
    turtle = turtle_cat_returns_daily().dropna()
    turtle.index = pd.to_datetime(turtle.index).tz_localize(None).normalize()
    turtle = turtle.groupby(turtle.index).last().sort_index()

    bench = benchmark_6040_returns().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench = bench.groupby(bench.index).last().sort_index()

    candidates = [
        {"gem": 0.80, "turtle": 0.20, "ic_basket": 0.00},
        {"gem": 0.80, "turtle": 0.15, "ic_basket": 0.05},
        {"gem": 0.75, "turtle": 0.15, "ic_basket": 0.10},
    ]
    joint_results: list[dict] = []
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={"gem": gem, "turtle": turtle, "ic_basket": basket},
            deployment_weights=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        joint_results.append(
            {
                "weights": w,
                "p_kill": r.p_kill_trip,
                "p95_dd": r.p95_maxdd_at_size,
                "passes": r.passes(),
            }
        )
        w_str = f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/IC{int(w['ic_basket'] * 100)}"
        print(
            f"  {w_str}: P_kill={r.p_kill_trip * 100:.3f}%, "
            f"95%DD={r.p95_maxdd_at_size * 100:.2f}%, "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    print("\n  L67 vs 60/40:")
    current = build_portfolio(
        {"gem": gem, "turtle": turtle}, {"gem": 0.80, "turtle": 0.20}
    ).dropna()
    eval_current = evaluate_portfolio_vs_benchmark(
        current, bench, portfolio_label="CURRENT", benchmark_label="60/40"
    )
    print(eval_current.report())

    proposed_5 = build_portfolio(
        {"gem": gem, "turtle": turtle, "ic_basket": basket},
        {"gem": 0.75, "turtle": 0.20, "ic_basket": 0.05},
    ).dropna()
    eval_5 = evaluate_portfolio_vs_benchmark(
        proposed_5, bench, portfolio_label="G75/T20/IC5", benchmark_label="60/40"
    )
    print(eval_5.report())

    # ── [5/5] Result log ──
    _write_log(
        REPORTS_DIR,
        results_by_ticker,
        summary_rows,
        survivors,
        basket_dsr,
        joint_results,
        (eval_current, eval_5),
    )
    print(f"\nResult log: {REPORTS_DIR / 'result_log.md'}")


def _write_log(
    reports_dir: Path,
    results_by_ticker: dict,
    summary_rows: list[dict],
    survivors: list[str],
    basket_dsr,
    joint_results,
    l67_pair,
) -> None:
    fp = reports_dir / "result_log.md"
    with fp.open("w", encoding="utf-8") as fh:
        fh.write("# ic_equity_daily V3.7 HYBRID -- L52 plateau + DSR\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Critical-review follow-on to PR #9.**\n\n")

        fh.write("## [1/5] Per-ticker sweep\n\n")
        for tkr, r in results_by_ticker.items():
            fh.write(f"### {tkr} (canonical thr={r['canonical_threshold']}, RSI=21)\n\n")
            fh.write("| Threshold | RSI | Sharpe | CI_lo | CI_hi | Canonical? |\n")
            fh.write("|---:|---:|---:|---:|---:|:---:|\n")
            for c in r["sweep_cells"]:
                star = " *" if c["is_canonical"] else ""
                sr_str = f"{c['sharpe']:+.4f}" if not np.isnan(c["sharpe"]) else "fail"
                ci_lo_str = f"{c['ci_lo']:+.3f}" if not np.isnan(c["ci_lo"]) else "fail"
                ci_hi_str = f"{c['ci_hi']:+.3f}" if not np.isnan(c["ci_hi"]) else "fail"
                fh.write(
                    f"| {c['threshold']:.2f} | {c['rsi_period']} | "
                    f"{sr_str} | {ci_lo_str} | {ci_hi_str} | {star.strip() or ''} |\n"
                )
            fh.write(
                f"\n**Plateau spread:** {r['plateau_spread'] * 100:.1f}% "
                f"(H1 50%: {'PASS' if r['plateau_h1_pass'] else 'FAIL'}; "
                f"strict 30%: {'PASS' if r['plateau_strict_pass'] else 'FAIL'})  \n"
            )
            if r["dsr"]:
                fh.write(
                    f"**DSR:** Sharpe {r['dsr'].sharpe:+.4f}, "
                    f"E[max SR] {r['dsr'].e_max_sr:+.4f}, "
                    f"z={r['dsr'].z:+.3f}, p={r['dsr'].dsr_prob:.4f} "
                    f"({'PASS' if r['dsr'].dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'} @ {DSR_DEPLOY_GATE:.0%})\n\n"
                )
            else:
                fh.write("**DSR:** insufficient sweep variance (N/A)\n\n")

        fh.write("## [2/5] Plateau + DSR survivors\n\n")
        fh.write(
            "| Ticker | Canonical Sharpe | Plateau spread | Plateau H1 | DSR p | Survivor? |\n"
        )
        fh.write("|---|---:|---:|:---:|---:|:---:|\n")
        for s in summary_rows:
            fh.write(
                f"| {s['ticker']} | {s['canonical_sharpe']:+.4f} | "
                f"{s['plateau_spread_pct']:.1f}% | "
                f"{'PASS' if s['plateau_h1'] else 'FAIL'} | "
                f"{s['dsr_prob']:.4f} | "
                f"{'YES' if s['survivor'] else 'NO'} |\n"
            )
        fh.write(f"\n**Survivors:** {', '.join(survivors) if survivors else 'NONE'}\n\n")

        if not survivors:
            fh.write(
                "## Verdict\n\n"
                "**DOWNGRADE PR #9.** No ticker survives the L52 plateau + DSR "
                "gates. The matrix-improving CONDITIONAL verdict was a point-"
                "selection artefact. ic_equity_daily strict-OOS Sharpe lift is "
                "not robust under V3.7 hybrid discipline.\n"
            )
            return

        if basket_dsr is not None:
            fh.write("## [3/5] Cross-ticker DSR on basket\n\n")
            fh.write(
                f"- Basket: {len(survivors)} survivors, top-rank by canonical Sharpe\n"
                f"- Basket Sharpe = {basket_dsr.sharpe:+.4f}\n"
                f"- E[max SR] across N=7 ticker pool = {basket_dsr.e_max_sr:+.4f}\n"
                f"- DSR z = {basket_dsr.z:+.3f}, p = {basket_dsr.dsr_prob:.4f}\n"
                f"- Gate ({DSR_DEPLOY_GATE:.0%}): "
                f"{'PASS' if basket_dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'}\n\n"
            )

        if joint_results:
            fh.write("## [4/5] L65 joint ruin (validated basket)\n\n")
            fh.write("| GEM | Turtle | ic_basket | P_kill | 95%DD | Verdict |\n")
            fh.write("|---:|---:|---:|---:|---:|:---:|\n")
            for r in joint_results:
                w = r["weights"]
                fh.write(
                    f"| {w['gem']:.0%} | {w['turtle']:.0%} | {w['ic_basket']:.0%} | "
                    f"{r['p_kill'] * 100:.3f}% | {r['p95_dd'] * 100:.2f}% | "
                    f"{'PASS' if r['passes'] else 'FAIL'} |\n"
                )

        if l67_pair:
            ev_current, ev_5 = l67_pair
            fh.write("\n## [5/5] L67 inclusion (validated basket)\n\n")
            fh.write("### CURRENT (no ic_basket)\n\n```\n" + ev_current.report() + "\n```\n\n")
            fh.write("### G75/T20/IC5\n\n```\n" + ev_5.report() + "\n```\n\n")

        fh.write("## Verdict\n\n")
        fh.write(
            "If the basket DSR p exceeds 0.95 AND the L65 + L67 results from "
            "section 4-5 hold, the PR #9 verdict is CONFIRMED with stronger "
            "evidence. Otherwise the verdict is downgraded -- operator must "
            "read the basket DSR row above.\n"
        )


if __name__ == "__main__":
    main()
