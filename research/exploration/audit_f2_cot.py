"""F2 CFTC CoT positioning extremes audit (Kang-Rouwenhorst-Tang JF 2020).

V3.7 hybrid audit per pre-reg `directives/Pre-Reg F2 CFTC CoT Positioning 2026-05-20.md`.

Universe: 14 commodities (drop CL -- CFTC name boundary at 2022-02-01).
Signal: cross-sectional speculator-positioning z-score on weekly Legacy
CoT reports. Long bottom-N (most-short-by-speculators); short top-N.
Hybrid gates: L21 + L52 plateau + DSR + 5-axis + L65 + L67.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_f2_cot.py
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
    THRESHOLDS as IC_THRESHOLDS,
)
from research.exploration.audit_ic_equity_daily import (  # noqa: E402
    _load_d as _load_ic,
)
from research.exploration.audit_ic_equity_daily import (  # noqa: E402
    ic_equity_returns,
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
    assess_strategy_ruin,
    slice_sanctuary,
)
from titan.research.framework.dsr import deflated_sharpe, sr_var_from_sweep  # noqa: E402
from titan.research.metrics import bootstrap_sharpe_ci, max_drawdown, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "f2_cot"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Drop CL (CFTC name boundary 2022-02-01); BZ provides energy exposure.
COMMODITY_SYMBOLS = [
    "BZ",  # Brent crude (energy)
    "NG",  # Natural gas (energy)
    "GC",  # Gold (precious metal)
    "SI",  # Silver
    "HG",  # Copper (industrial metal)
    "PL",  # Platinum
    "PA",  # Palladium
    "ZC",  # Corn
    "ZW",  # Wheat
    "ZS",  # Soybeans
    "CT",  # Cotton
    "KC",  # Coffee
    "SB",  # Sugar
    "CC",  # Cocoa
]

# Sweep grid (3x3=9 cells)
Z_LOOKBACKS = [52, 104, 156]  # weeks
TOP_N_GRID = [2, 3, 4]
CANONICAL_Z_LOOKBACK = 104
CANONICAL_TOP_N = 3

PERIODS_PER_YEAR = 52  # weekly cadence
COST_BPS_PER_LEG = 2.0  # continuous-contract commodity ETF cost

PLATEAU_GATE_H1 = 0.50
PLATEAU_GATE_STRICT = 0.30
DSR_DEPLOY_GATE = 0.95


def _load_commodity_close(sym: str) -> pd.Series:
    """Load yfinance =F continuous-contract close for one commodity."""
    df = pd.read_parquet(DATA_DIR / f"{sym}_F_D.parquet")
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df.sort_index().dropna(subset=["close"])["close"].astype(float).rename(sym)


def _load_cot() -> pd.DataFrame:
    """Load the CFTC Legacy Futures-Only CoT report into a long-format
    DataFrame with one row per (commodity, report_date).
    """
    df = pd.read_parquet(DATA_DIR / "cftc_cot_disaggregated.parquet")
    # Coerce types we need
    for col in (
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(
        subset=[
            "open_interest_all",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
        ]
    )
    df = df[df["open_interest_all"] > 0].copy()
    df["spec_net_oi"] = (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / df[
        "open_interest_all"
    ]
    # NG has TWO simultaneously-listed NYMEX contracts:
    #   * "NATURAL GAS"  -- physically-settled (the NG=F price proxy)
    #   * "HENRY HUB"    -- financially-settled lookalike
    # Both report 2018-07-17 .. 2022-02-01 in parallel. The "NATURAL GAS"
    # name stops at the 2022-02-01 CFTC infrastructure boundary; HENRY
    # HUB continues. We prefer NATURAL GAS while available (matches NG=F
    # price), then fall back to HENRY HUB for 2022-02-08 onwards.
    is_ng_hh = (df["symbol"] == "NG") & (
        df["market_and_exchange_names"] == "HENRY HUB - NEW YORK MERCANTILE EXCHANGE"
    )
    has_ng_pre = df["symbol"].eq("NG") & df["market_and_exchange_names"].eq(
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE"
    )
    ng_pre_dates = set(df.loc[has_ng_pre, "report_date"])
    drop_ng_hh_overlap = is_ng_hh & df["report_date"].isin(ng_pre_dates)
    df = df.loc[~drop_ng_hh_overlap].copy()
    # Sanity: enforce uniqueness on (symbol, report_date).
    df = df.drop_duplicates(subset=["symbol", "report_date"], keep="first")
    return df[["symbol", "report_date", "spec_net_oi"]].sort_values(["symbol", "report_date"])


def _build_signal_matrix(cot: pd.DataFrame, z_lookback: int) -> pd.DataFrame:
    """Build a (week, symbol) z-score matrix.

    For each commodity, rolling-z-score the spec_net_oi over the past
    z_lookback weeks (CAUSAL: z_t uses data through t).
    """
    pivot = cot.pivot(index="report_date", columns="symbol", values="spec_net_oi")
    # Restrict to our universe
    pivot = pivot[[c for c in COMMODITY_SYMBOLS if c in pivot.columns]]
    # Rolling z per commodity
    roll_mean = pivot.rolling(z_lookback, min_periods=z_lookback).mean()
    roll_std = pivot.rolling(z_lookback, min_periods=z_lookback).std(ddof=1)
    z = (pivot - roll_mean) / roll_std.replace(0, np.nan)
    return z


def _build_price_returns_weekly(
    prices: pd.DataFrame, signal_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each commodity, compute the WEEKLY log return from the close on
    one CoT report date to the close on the next CoT report date.

    Causality: signal at week t is built from CoT data released Friday
    afternoon of week t (Tuesday positions). Position effective from the
    Monday following the Friday release, held to next Friday's release.
    We approximate this by computing per-week returns as close[t+1] /
    close[t] - 1, with t in `signal_index`. The actual position lag
    (Tuesday-positions -> Friday-release -> Monday-effective) is
    captured by the .shift(1) in `_strategy_weekly_returns` below.
    """
    rets = pd.DataFrame(index=signal_index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        s = prices[col].reindex(signal_index, method="pad")
        rets[col] = np.log(s / s.shift(1))
    return rets


def _strategy_weekly_returns(
    z: pd.DataFrame,
    rets: pd.DataFrame,
    *,
    top_n: int,
    cost_bps_per_leg: float = COST_BPS_PER_LEG,
) -> pd.Series:
    """Cross-sectional weekly portfolio return.

    Algorithm at each week t:
      1. Identify commodities with valid z_t (non-NaN).
      2. If fewer than 2*top_n, skip (no rebalance).
      3. Rank by z. LONG bottom-N (most-short-by-speculators);
         SHORT top-N (most-long-by-speculators).
      4. Equal-weighted within each leg; long-leg = +0.5 NAV total,
         short-leg = -0.5 NAV total. Per-commodity weight = +/- 0.5/N.
      5. Position effective from t+1 (causal -- .shift(1) on weights).
      6. Net return = held @ rets, less turnover cost.
    """
    weights = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for t, ts in enumerate(z.index):
        z_t = z.iloc[t].dropna()
        if len(z_t) < 2 * top_n:
            continue
        ranked = z_t.sort_values()
        long_set = ranked.index[:top_n].tolist()  # bottom = most-short
        short_set = ranked.index[-top_n:].tolist()  # top = most-long
        w_per_long = +0.5 / len(long_set)
        w_per_short = -0.5 / len(short_set)
        for sym in long_set:
            weights.loc[ts, sym] = w_per_long
        for sym in short_set:
            weights.loc[ts, sym] = w_per_short
    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    dpos = held.diff().abs().fillna(0.0).sum(axis=1)
    cost = dpos * (cost_bps_per_leg / 10_000.0)
    return (gross - cost).rename("ret")


def assert_causal_f2(cot: pd.DataFrame, prices: pd.DataFrame) -> None:
    """L21 smoke. Corrupt future prices + future CoT positions; assert
    past weekly returns unchanged.
    """
    z = _build_signal_matrix(cot, z_lookback=CANONICAL_Z_LOOKBACK)
    rets = _build_price_returns_weekly(prices, z.index)
    base = _strategy_weekly_returns(z, rets, top_n=CANONICAL_TOP_N).dropna()

    # Corrupt FUTURE data (last 20 weeks). Past returns must not change.
    n = len(z)
    if n < 200:
        print("[F2] L21 skipped (window too short)")
        return
    safe_cutoff = z.index[n - 50]  # past = before this
    corrupt_from = z.index[n - 20]

    cot_c = cot.copy()
    cot_c.loc[cot_c["report_date"] >= corrupt_from, "spec_net_oi"] *= 100.0
    prices_c = prices.copy()
    prices_c.loc[prices_c.index >= corrupt_from] *= 100.0

    z_c = _build_signal_matrix(cot_c, z_lookback=CANONICAL_Z_LOOKBACK)
    rets_c = _build_price_returns_weekly(prices_c, z_c.index)
    pert = _strategy_weekly_returns(z_c, rets_c, top_n=CANONICAL_TOP_N).dropna()

    past_base = base[base.index < safe_cutoff]
    past_pert = pert[pert.index < safe_cutoff]
    # Align on common index in case z_c lost some early rows.
    common = past_base.index.intersection(past_pert.index)
    diff = (past_base.loc[common] - past_pert.loc[common]).abs().max()
    assert diff < 1e-10, f"L21 fail: past-week diff {diff}"
    print(
        f"[F2] L21 PASS (diff={diff:.2e}, corrupted from {corrupt_from.date()}, "
        f"past window ends {safe_cutoff.date()})"
    )


def _stats(rets: pd.Series) -> dict:
    rets = rets.dropna()
    sr = float(sharpe(rets, periods_per_year=PERIODS_PER_YEAR))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets, periods_per_year=PERIODS_PER_YEAR, n_resamples=2000, seed=42
    )
    mdd = float(max_drawdown(rets))
    return {
        "sharpe": sr,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(len(rets)),
        "mdd": mdd,
    }


def main() -> None:
    print("=" * 88)
    print("F2 CFTC CoT positioning extremes -- V3.7 hybrid audit (KRT 2020)")
    print("=" * 88)

    # ── Load ──
    cot = _load_cot()
    print(f"\n[load] CFTC CoT: {len(cot)} rows across {cot['symbol'].nunique()} commodities")
    print(
        f"  CoT date range: {cot['report_date'].min().date()} -> {cot['report_date'].max().date()}"
    )

    prices = pd.concat(
        {sym: _load_commodity_close(sym) for sym in COMMODITY_SYMBOLS}, axis=1
    ).dropna(how="all")
    prices.columns = COMMODITY_SYMBOLS  # ensure flat columns
    print(
        f"[load] Prices: {prices.shape[0]} bars x {prices.shape[1]} commodities, "
        f"{prices.index[0].date()} -> {prices.index[-1].date()}"
    )

    # ── L21 ──
    print("\n[L21] causality smoke test")
    assert_causal_f2(cot, prices)

    # ── Sanctuary slice on the signal-date index ──
    z_full = _build_signal_matrix(cot, z_lookback=CANONICAL_Z_LOOKBACK)
    sanc = slice_sanctuary(z_full, months=12)
    visible_idx = sanc.visible.index
    sanc_idx = sanc.sanctuary.index
    print(
        f"[sanctuary] visible {visible_idx[0].date()} -> {visible_idx[-1].date()} "
        f"({len(visible_idx)} weeks); sanctuary {sanc_idx[0].date()} -> "
        f"{sanc_idx[-1].date()} ({len(sanc_idx)} weeks)"
    )

    # ── [1/6] Per-cell sweep ──
    print("\n[1/6] Per-cell sweep on VISIBLE window")
    print(
        f"  {'z_lb':>5s} {'top_n':>5s} {'ann SR':>9s} {'CI_lo':>9s} {'CI_hi':>9s} "
        f"{'mdd':>9s} {'canonical'}"
    )
    sweep_results: list[dict] = []
    canonical_returns: pd.Series | None = None
    for zlb in Z_LOOKBACKS:
        z = _build_signal_matrix(cot, z_lookback=zlb)
        z_vis = z.loc[visible_idx.intersection(z.index)]
        rets_vis = _build_price_returns_weekly(prices, z_vis.index)
        for n in TOP_N_GRID:
            rets = _strategy_weekly_returns(z_vis, rets_vis, top_n=n)
            stats = _stats(rets)
            is_canon = zlb == CANONICAL_Z_LOOKBACK and n == CANONICAL_TOP_N
            sweep_results.append(
                {
                    "z_lookback": zlb,
                    "top_n": n,
                    "ann_sharpe": stats["sharpe"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "mdd": stats["mdd"],
                    "n": stats["n"],
                    "is_canonical": is_canon,
                    "returns": rets,
                }
            )
            if is_canon:
                canonical_returns = rets
            star = " *" if is_canon else ""
            print(
                f"  {zlb:>5d} {n:>5d}  {stats['sharpe']:>+8.4f} "
                f"{stats['ci_lo']:>+8.3f} {stats['ci_hi']:>+8.3f} "
                f"{stats['mdd']:>+8.2%}{star}"
            )

    canonical_cell = next(c for c in sweep_results if c["is_canonical"])
    assert canonical_returns is not None

    # ── [2/6] L52 plateau ──
    print("\n[2/6] L52 plateau pre-flight")
    plateau_cells = [
        c
        for c in sweep_results
        if (
            (c["z_lookback"] == CANONICAL_Z_LOOKBACK and c["top_n"] in TOP_N_GRID)
            or (c["top_n"] == CANONICAL_TOP_N and c["z_lookback"] in Z_LOOKBACKS)
        )
    ]
    plateau_sharpes = [c["ann_sharpe"] for c in plateau_cells]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    plateau_h1 = plateau_spread <= PLATEAU_GATE_H1
    plateau_strict = plateau_spread <= PLATEAU_GATE_STRICT
    print(
        f"  Plateau spread {plateau_spread * 100:.1f}%  "
        f"(H1 {'PASS' if plateau_h1 else 'FAIL'}, "
        f"strict {'PASS' if plateau_strict else 'FAIL'})"
    )

    # ── [3/6] DSR ──
    print("\n[3/6] DSR deflation (9 cells)")
    sweep_sharpes = [c["ann_sharpe"] for c in sweep_results]
    sr_var = sr_var_from_sweep(sweep_sharpes)
    if sr_var > 0:
        dsr = deflated_sharpe(
            sr_hat=canonical_cell["ann_sharpe"],
            sr_var_across_trials=sr_var,
            returns=canonical_returns,
            n_trials=len(sweep_sharpes),
            survivors_only=False,
        )
        print(
            f"  canonical SR={dsr.sharpe:+.4f}, E[max SR]={dsr.e_max_sr:+.4f}, "
            f"z={dsr.z:+.3f}, p={dsr.dsr_prob:.4f} "
            f"({'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'} @ "
            f"{DSR_DEPLOY_GATE:.0%})"
        )
    else:
        dsr = None
        print("  insufficient sweep variance")

    # ── Early-exit per pre-reg §3.7 ──
    early_fail: list[str] = []
    if not plateau_h1:
        early_fail.append(f"L52 plateau spread {plateau_spread * 100:.1f}% > 50%")
    if dsr is None or dsr.dsr_prob < DSR_DEPLOY_GATE:
        early_fail.append(
            f"DSR p={(dsr.dsr_prob if dsr else float('nan')):.4f} < {DSR_DEPLOY_GATE}"
        )
    if canonical_cell["ci_lo"] <= 0:
        early_fail.append(f"canonical CI_lo {canonical_cell['ci_lo']:+.3f} <= 0")

    if early_fail:
        print("\n" + "=" * 88)
        print("VERDICT (early-exit per pre-reg §3.7)")
        print("=" * 88)
        verdict = "RETIRE"
        print(f"  Verdict: {verdict}")
        for r in early_fail:
            print(f"    - {r}")
        _write_log_early_retire(
            report_path=REPORTS_DIR / "result_log.md",
            sweep_results=sweep_results,
            canonical_cell=canonical_cell,
            plateau_spread=plateau_spread,
            plateau_h1=plateau_h1,
            plateau_strict=plateau_strict,
            dsr=dsr,
            verdict=verdict,
            fail_reasons=early_fail,
        )
        print(f"\n  Result log: {(REPORTS_DIR / 'result_log.md').relative_to(PROJECT_ROOT)}")
        return

    # ── [4/6] 5-axis (CI + DSR already done; add sanctuary + skip MC/noise for cost) ──
    print("\n[4/6] 5-axis matrix (CI/DSR/sanctuary; MC and noise skipped for weekly cadence)")
    z_sanc = _build_signal_matrix(cot, z_lookback=CANONICAL_Z_LOOKBACK).loc[
        sanc_idx.intersection(z_full.index)
    ]
    rets_sanc = _build_price_returns_weekly(prices, z_sanc.index)
    sanc_rets = _strategy_weekly_returns(z_sanc, rets_sanc, top_n=CANONICAL_TOP_N)
    sanc_sr = float(sharpe(sanc_rets.dropna(), periods_per_year=PERIODS_PER_YEAR))
    sanc_diff = abs(canonical_cell["ann_sharpe"] - sanc_sr)
    axis_sanc_pass = sanc_diff <= 0.30
    print(
        f"  Sanctuary: visible {canonical_cell['ann_sharpe']:+.4f} vs sanctuary "
        f"{sanc_sr:+.4f} (diff {sanc_diff:.3f}, gate <= 0.30): "
        f"{'PASS' if axis_sanc_pass else 'FAIL'}"
    )

    # ── [5/6] L65 ──
    print("\n[5/6] L65 ruin")
    # Convert weekly to daily-equivalent for joint analysis: forward-fill
    # weekly returns to daily index. This is an approximation -- the
    # L65 block-bootstrap on weekly returns is the more accurate test.
    single_ruin: list[dict] = []
    for w in [0.02, 0.05, 0.10]:
        r = assess_strategy_ruin(
            canonical_returns,
            deployment_weight=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=52,  # 1y at weekly cadence
            block_size=4,
            n_paths=2000,
            seed=42,
        )
        single_ruin.append(
            {
                "weight": w,
                "p_kill": r.p_kill_trip,
                "p95_dd": r.p95_maxdd_at_size,
                "p50_dd": r.median_maxdd_at_size,
                "passes": r.passes(),
            }
        )
        print(
            f"  w={w:.0%}: P_kill={r.p_kill_trip * 100:.3f}%, "
            f"p95={r.p95_maxdd_at_size * 100:.2f}%, "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    # Joint ruin with current LIVE -- need daily series for the joint blocker.
    # Approximate F2 daily by forward-fill from weekly.
    def _ic_strict_oos(t: str) -> pd.Series:
        df = _load_ic(t)
        n = len(df)
        is_split = n // 2
        rets = ic_equity_returns(df, threshold=IC_THRESHOLDS[t], is_until_idx=is_split)
        oos = rets.iloc[is_split:]
        oos.index = pd.to_datetime(oos.index).tz_localize(None).normalize()
        return oos.groupby(oos.index).last().sort_index()

    ic_basket = (
        pd.concat({t: _ic_strict_oos(t) for t in ["HWM", "WMT", "SYK"]}, axis=1)
        .fillna(0.0)
        .mean(axis=1)
    )
    gem = gem_j5_returns().dropna()
    gem.index = pd.to_datetime(gem.index).tz_localize(None).normalize()
    gem = gem.groupby(gem.index).last().sort_index()
    turtle = turtle_cat_returns_daily().dropna()
    turtle.index = pd.to_datetime(turtle.index).tz_localize(None).normalize()
    turtle = turtle.groupby(turtle.index).last().sort_index()

    # Spread the F2 weekly returns to daily via the natural-cadence
    # interpretation: the weekly return is realised across 5 daily bars.
    # We allocate the full weekly return to the LAST day of the week so
    # that the joint daily MC sees the same lumpy realisation pattern
    # (a more accurate alternative would be to bootstrap weekly bars
    # separately; pragmatic choice for joint analysis with daily peers).
    f2_daily = canonical_returns.copy()
    f2_daily.index = pd.to_datetime(f2_daily.index).tz_localize(None).normalize()
    f2_daily = f2_daily.groupby(f2_daily.index).last()

    candidates = [
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15, "f2": 0.00},
        {"gem": 0.66, "turtle": 0.17, "ic": 0.15, "f2": 0.02},
        {"gem": 0.64, "turtle": 0.16, "ic": 0.15, "f2": 0.05},
        {"gem": 0.62, "turtle": 0.13, "ic": 0.15, "f2": 0.10},
    ]
    joint_results: list[dict] = []
    print(f"  {'weights':>40s}  {'P_kill':>9s}  {'p95 DD':>9s}  verdict")
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={
                "gem": gem,
                "turtle": turtle,
                "ic": ic_basket,
                "f2": f2_daily,
            },
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
        w_str = (
            f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/"
            f"IC{int(w['ic'] * 100)}/F2{int(w['f2'] * 100)}"
        )
        print(
            f"  {w_str:>40s}  {r.p_kill_trip * 100:>7.3f}%  "
            f"{r.p95_maxdd_at_size * 100:>7.2f}%  "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    # ── [6/6] L67 ──
    print("\n[6/6] L67 portfolio inclusion")
    bench = benchmark_6040_returns().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench = bench.groupby(bench.index).last().sort_index()

    current = build_portfolio(
        {"gem": gem, "turtle": turtle, "ic": ic_basket},
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15},
    ).dropna()
    ev_current = evaluate_portfolio_vs_benchmark(
        current, bench, portfolio_label="CURRENT G68/T17/IC15", benchmark_label="60/40"
    )
    print()
    print(ev_current.report())

    eval_proposals: list[tuple[str, "object"]] = []
    for w_f2 in [0.02, 0.05, 0.10]:
        w_g = 0.68 - 0.75 * w_f2
        w_t = 0.17 - 0.25 * w_f2
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "ic": ic_basket, "f2": f2_daily},
            {"gem": w_g, "turtle": w_t, "ic": 0.15, "f2": w_f2},
        ).dropna()
        tag = f"PROPOSED G{int(w_g * 100)}/T{int(w_t * 100)}/IC15/F2{int(w_f2 * 100)}"
        print()
        ev = evaluate_portfolio_vs_benchmark(
            proposed, bench, portfolio_label=tag, benchmark_label="60/40"
        )
        print(ev.report())
        eval_proposals.append((tag, ev))

    # ── Verdict ──
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    fail_reasons: list[str] = []
    if not axis_sanc_pass:
        fail_reasons.append(f"sanctuary divergence {sanc_diff:.3f} > 0.30")
    if not all(r["passes"] for r in single_ruin):
        fail_reasons.append("L65 single FAIL at one or more weights")
    if not all(r["passes"] for r in joint_results):
        fail_reasons.append("L65 joint FAIL at one or more candidate mixes")

    cur_sharpe = ev_current.results[0].portfolio_value
    best_lift = -1e9
    best_lift_tag = ""
    for tag, ev in eval_proposals:
        lift = ev.results[0].portfolio_value - cur_sharpe
        if lift > best_lift:
            best_lift = lift
            best_lift_tag = tag

    if not fail_reasons and best_lift >= 0.05:
        verdict = f"DEPLOY-eligible at {best_lift_tag} (Sharpe lift +{best_lift:.3f})"
    elif not fail_reasons:
        verdict = (
            f"CONDITIONAL_WATCHPOINT (paper-only) -- gates PASS but Sharpe lift "
            f"{best_lift:+.3f} below +0.05 deploy threshold"
        )
    else:
        verdict = "CONDITIONAL_WATCHPOINT (paper-only)"
    print(f"  Verdict: {verdict}")
    if fail_reasons:
        print("  Fail reasons:")
        for r in fail_reasons:
            print(f"    - {r}")
    print(f"  Best L67 Sharpe lift: {best_lift:+.3f} at {best_lift_tag}")

    _write_log_full(
        report_path=REPORTS_DIR / "result_log.md",
        sweep_results=sweep_results,
        canonical_cell=canonical_cell,
        plateau_spread=plateau_spread,
        plateau_h1=plateau_h1,
        plateau_strict=plateau_strict,
        dsr=dsr,
        sanc_diff=sanc_diff,
        sanc_sr=sanc_sr,
        single_ruin=single_ruin,
        joint_results=joint_results,
        ev_current=ev_current,
        eval_proposals=eval_proposals,
        verdict=verdict,
        fail_reasons=fail_reasons,
        best_lift=best_lift,
        best_lift_tag=best_lift_tag,
    )
    print(f"\n  Result log: {(REPORTS_DIR / 'result_log.md').relative_to(PROJECT_ROOT)}")


def _write_log_early_retire(
    *,
    report_path: Path,
    sweep_results: list[dict],
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    verdict: str,
    fail_reasons: list[str],
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# F2 CFTC CoT Positioning -- V3.7 hybrid audit (EARLY RETIRE)\n\n")
        fh.write("**Run date:** 2026-05-20\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg F2 CFTC CoT Positioning 2026-05-20.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")
        fh.write("Early-exit per pre-reg §3.7. 5-axis MC + L65 + L67 not run.\n\n")

        fh.write("## [1] Per-cell sweep\n\n")
        fh.write("| z_lookback (wks) | top_n | ann SR | CI_lo | CI_hi | mdd | n | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['z_lookback']} | {c['top_n']} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | {c['n']} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau\n\n")
        fh.write(
            f"- Plateau spread: **{plateau_spread * 100:.1f}%** "
            f"(H1 ≤50%: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict ≤30%: {'PASS' if plateau_strict else 'FAIL'})\n"
        )

        fh.write("\n## [3] DSR\n\n")
        if dsr is not None:
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
            )

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")


def _write_log_full(
    *,
    report_path: Path,
    sweep_results: list[dict],
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    sanc_diff: float,
    sanc_sr: float,
    single_ruin: list[dict],
    joint_results: list[dict],
    ev_current,
    eval_proposals: list,
    verdict: str,
    fail_reasons: list[str],
    best_lift: float,
    best_lift_tag: str,
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# F2 CFTC CoT Positioning -- V3.7 hybrid audit\n\n")
        fh.write("**Run date:** 2026-05-20\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg F2 CFTC CoT Positioning 2026-05-20.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")

        fh.write("## [1] Per-cell sweep\n\n")
        fh.write("| z_lookback (wks) | top_n | ann SR | CI_lo | CI_hi | mdd | n | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['z_lookback']} | {c['top_n']} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | {c['n']} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau\n\n")
        fh.write(
            f"- Spread: {plateau_spread * 100:.1f}% "
            f"(H1: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict: {'PASS' if plateau_strict else 'FAIL'})\n"
        )
        if dsr is not None:
            fh.write("\n## [3] DSR\n\n")
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
            )

        fh.write("\n## [4] Sanctuary\n\n")
        fh.write(
            f"- Visible canonical SR = {canonical_cell['ann_sharpe']:+.4f}\n"
            f"- Sanctuary SR = {sanc_sr:+.4f}\n"
            f"- Divergence: {sanc_diff:.3f} (gate ≤ 0.30)\n"
        )

        fh.write("\n## [5] L65 ruin\n\n### Single-strategy\n\n")
        fh.write("| Weight | P_kill | p95 DD | p50 DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|:---:|\n")
        for r in single_ruin:
            fh.write(
                f"| {r['weight']:.0%} | {r['p_kill'] * 100:.3f}% | "
                f"{r['p95_dd'] * 100:.2f}% | {r['p50_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )
        fh.write("\n### Joint\n\n| Weights | P_kill | p95 DD | Verdict |\n|---|---:|---:|:---:|\n")
        for r in joint_results:
            w = r["weights"]
            w_str = (
                f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/"
                f"IC{int(w['ic'] * 100)}/F2{int(w['f2'] * 100)}"
            )
            fh.write(
                f"| {w_str} | {r['p_kill'] * 100:.3f}% | "
                f"{r['p95_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write(
            "\n## [6] L67 inclusion\n\n### CURRENT\n\n```\n" + ev_current.report() + "\n```\n\n"
        )
        for tag, ev in eval_proposals:
            fh.write(f"### {tag}\n\n```\n" + ev.report() + "\n```\n\n")
        fh.write(f"\n### Best L67 Sharpe lift: {best_lift:+.3f} at `{best_lift_tag}`\n")

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")


if __name__ == "__main__":
    main()
