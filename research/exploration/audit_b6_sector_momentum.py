"""B6 sector-ETF cross-sectional momentum + crash-hedge overlay audit.

V3.7 hybrid audit per pre-reg `directives/Pre-Reg B6 Sector-Momentum Crash-Hedge 2026-05-19.md`.
Inspired by Daniel-Moskowitz (JFE 2016) "Momentum Crashes" -- NOT a
replication (no CRSP access; A1 already blocked on that data). The
audit tests the design principle on a survivorship-clean ETF universe.

Strategy
========

Cross-sectional momentum on 9 SPDR Select Sector ETFs (XLRE excluded
for window length). Monthly rebalance: long top-N by 12m-skip-1
momentum, short bottom-N. Crash-hedge overlay: scale exposure by
``crash_scale`` when in PANIC STATE (SPY 24m return < -10% AND SPY
21d realised vol > 2x its 252d median).

V3.7 hybrid gates: L21 + L52 plateau + DSR + 5-axis + L65 + L67.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_b6_sector_momentum.py
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
    run_block_mc,
    run_noise_robustness,
    slice_sanctuary,
)
from titan.research.framework.dsr import deflated_sharpe, sr_var_from_sweep  # noqa: E402
from titan.research.framework.mc import McConfig  # noqa: E402
from titan.research.framework.robustness import NoiseConfig  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b6_sector_momentum"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PERIODS_PER_YEAR = BARS_PER_YEAR["D"]
COST_BPS_PER_LEG = 1.0

# 9 SPDR Select Sector ETFs (XLRE excluded; launched 2015, too short)
SECTOR_TICKERS = ["XLF", "XLK", "XLE", "XLU", "XLY", "XLP", "XLI", "XLB", "XLV"]

# Sweep grid
TOP_N_GRID = [2, 3, 4]
CRASH_SCALE_GRID = [0.0, 0.25, 0.50]
CANONICAL_TOP_N = 3
CANONICAL_CRASH_SCALE = 0.0

PLATEAU_GATE_H1 = 0.50
PLATEAU_GATE_STRICT = 0.30
DSR_DEPLOY_GATE = 0.95


def _load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{ticker}_D.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()  # date-only, no hour-level mismatches
    return df.sort_index().dropna(subset=["close"])[["close"]].astype(float)


def _verify_total_return_close(closes: pd.DataFrame) -> dict:
    """Heuristic: passive sector ETFs paid ~2% avg dividend yield. If
    realised ann return < 2% for the aggregate, prices may be unadjusted.
    """
    ann_rets = {}
    for col in closes.columns:
        s = closes[col].dropna()
        if len(s) < 252:
            continue
        ann_rets[col] = float((s.iloc[-1] / s.iloc[0]) ** (PERIODS_PER_YEAR / len(s)) - 1.0)
    median_ann = float(np.median(list(ann_rets.values()))) if ann_rets else 0.0
    return {
        "per_ticker_ann_returns": ann_rets,
        "median_ann_return": median_ann,
        "looks_adjusted": median_ann > 0.02,
    }


def b6_sector_momentum_returns(
    closes: pd.DataFrame,
    spy_close: pd.Series,
    *,
    top_n: int = CANONICAL_TOP_N,
    crash_scale: float = CANONICAL_CRASH_SCALE,
    lookback_days: int = 252,
    skip_days: int = 21,
    bear_threshold: float = -0.10,
    bear_lookback_days: int = 504,
    vol_lookback_short: int = 21,
    vol_lookback_long: int = 252,
    vol_multiplier: float = 2.0,
    cost_bps_per_leg: float = COST_BPS_PER_LEG,
) -> tuple[pd.Series, pd.Series]:
    """Returns (per_bar_net_returns, panic_state_indicator).

    Algorithm:
      1. Compute 12m-skip-1 momentum per ETF at each bar (vectorised).
      2. Rank ETFs by momentum at month-end; long top-N, short bottom-N.
      3. Compute SPY-based panic state (2y log return < -0.10 AND
         21d vol > 2x 252d median).
      4. In panic state, multiply position by crash_scale (0=fully off,
         1=fully on). Causal via .shift(1).
      5. Net return = long-leg avg - short-leg avg (each leg eq-weight).
      6. Cost = (turnover) * 2 * cost_bps_per_leg / 10000.
    """
    # Log returns per ETF
    log_ret = np.log(closes / closes.shift(1)).fillna(0.0)

    # 12m skip-1 momentum: log_close at t-skip_days minus log_close at t-lookback_days
    log_close = np.log(closes)
    mom = log_close.shift(skip_days) - log_close.shift(lookback_days)
    # mom_t uses prices from [t-lookback, t-skip], so it's causal at t.

    # Rebalance dates: last business day of each month within the index
    rebal_dates = (
        closes.index.to_series().groupby([closes.index.year, closes.index.month]).last().values
    )
    rebal_set = set(pd.DatetimeIndex(rebal_dates).normalize())

    # Position matrix: per-ETF weight at each bar, starting at 0
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    last_long: list[str] = []
    last_short: list[str] = []
    in_warmup = True
    for t, ts in enumerate(closes.index):
        if ts.normalize() in rebal_set:
            mom_t = mom.iloc[t].dropna()
            if len(mom_t) < 2 * top_n:
                # Warmup: insufficient ETF count with valid momentum
                last_long, last_short = [], []
            else:
                ranked = mom_t.sort_values(ascending=False)
                last_long = list(ranked.index[:top_n])
                last_short = list(ranked.index[-top_n:])
                in_warmup = False
        # Apply last rebalance weights forward
        if not in_warmup and last_long and last_short:
            w_per_long = +0.5 / len(last_long)
            w_per_short = -0.5 / len(last_short)
            for sym in last_long:
                weights.loc[ts, sym] = w_per_long
            for sym in last_short:
                weights.loc[ts, sym] = w_per_short

    # Panic state (causal): SPY 24m return < bear_threshold AND
    # SPY 21d vol > vol_multiplier x 252d median vol.
    spy_log = np.log(spy_close / spy_close.shift(1)).fillna(0.0)
    spy_2y_ret = (spy_close / spy_close.shift(bear_lookback_days)).apply(np.log).fillna(0.0)
    bear = (spy_2y_ret < bear_threshold).astype(float)
    spy_vol_short = spy_log.rolling(vol_lookback_short).std(ddof=1)
    spy_vol_long = spy_log.rolling(vol_lookback_long).median()
    high_vol = (spy_vol_short > vol_multiplier * spy_vol_long).astype(float)
    panic_t = (bear * high_vol).fillna(0.0)

    # Align panic to closes index, causal shift
    panic_aligned = panic_t.reindex(closes.index).fillna(0.0).shift(1).fillna(0.0)
    # Crash-hedge multiplier
    multiplier = pd.Series(np.where(panic_aligned > 0.5, crash_scale, 1.0), index=closes.index)

    # Held position = weights.shift(1) * multiplier.shift(0) [multiplier is
    # already causal because panic uses .shift(1)]
    held = weights.shift(1).fillna(0.0).multiply(multiplier, axis=0)

    # Per-bar gross return
    gross = (held * log_ret).sum(axis=1)

    # Turnover (sum of |delta weight| across ETFs)
    dpos = held.diff().abs().fillna(0.0).sum(axis=1)
    cost = dpos * (cost_bps_per_leg / 10_000.0)
    net = gross - cost

    return net.rename("ret"), panic_aligned.rename("panic")


def assert_causal_b6(closes: pd.DataFrame, spy_close: pd.Series) -> None:
    """L21: corrupting future bars must not change past returns."""
    base, _ = b6_sector_momentum_returns(closes, spy_close)
    closes_c = closes.copy()
    spy_c = spy_close.copy()
    n = len(closes)
    cutoff_date = closes.index[n - 50]
    safe_cutoff_date = closes.index[n - 50 - 350]  # pad for 252+21 lookback
    closes_c.loc[closes_c.index >= cutoff_date] *= 100.0
    spy_c.loc[spy_c.index >= cutoff_date] *= 100.0
    pert, _ = b6_sector_momentum_returns(closes_c, spy_c)
    past_base = base[base.index < safe_cutoff_date]
    past_pert = pert[pert.index < safe_cutoff_date]
    diff = (past_base - past_pert).abs().max()
    assert diff < 1e-10, f"L21 fail: max past-bar diff {diff}"
    print(
        f"[B6] L21 PASS (diff={diff:.2e}, "
        f"corrupted from {cutoff_date.date()}, past window ends "
        f"{safe_cutoff_date.date()})"
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
    print("B6 sector momentum + crash-hedge overlay -- V3.7 hybrid audit")
    print("=" * 88)

    # ── Load ──
    sector_closes = pd.concat(
        {tkr: _load_etf(tkr)["close"] for tkr in SECTOR_TICKERS}, axis=1
    ).dropna(how="all")
    print(
        f"\n[load] sector ETFs: {sector_closes.shape[0]} bars × {sector_closes.shape[1]} tickers"
        f"  {sector_closes.index[0].date()} -> {sector_closes.index[-1].date()}"
    )

    spy = _load_etf("SPY")
    spy_close = spy["close"]
    print(f"[load] SPY: {len(spy_close)} bars")

    # Align on common index. The 9 sector ETFs have slightly staggered
    # first trading days at the late-Dec 1998 launch; use the LATEST
    # sector start as the window start, intersected with SPY's available
    # range. Forward-fill any single-ticker stale-bar gaps within the
    # window (sector data is high-quality, real gaps are rare).
    sector_first_all = sector_closes.dropna(how="any").index.min()
    common = sector_closes.dropna(how="any").index.intersection(spy_close.dropna().index)
    if len(common) == 0:
        # No bar where all 9 sectors AND SPY have non-NaN simultaneously
        # at the same trading day. Fall back: take SPY's range, ffill
        # sectors within it (sectors launched before SPY so missing data
        # in the SPY range is just NaN at the very start of partial-history
        # tickers, which ffill handles).
        common = spy_close.dropna().index
        common = common[common >= sector_first_all]
    sector_closes = sector_closes.loc[common].ffill().dropna(how="any")
    common = sector_closes.index
    spy_close = spy_close.loc[common]
    print(f"[align] common window: {common[0].date()} -> {common[-1].date()} ({len(common)} bars)")

    # Pre-flight L74 check
    tr_check = _verify_total_return_close(sector_closes)
    print(
        f"\n[pre-flight] median annualised price return across 9 sectors: "
        f"{tr_check['median_ann_return'] * 100:+.2f}%"
    )
    print(
        f"[pre-flight] looks total-return-adjusted: {tr_check['looks_adjusted']} (threshold > 2%)"
    )

    # ── L21 ──
    print("\n[L21] causality smoke test")
    assert_causal_b6(sector_closes, spy_close)

    # ── Sanctuary slice ──
    sanc = slice_sanctuary(sector_closes, months=12)
    visible_idx = sanc.visible.index
    sanc_idx = sanc.sanctuary.index
    sect_vis = sector_closes.loc[visible_idx]
    sect_sanc = sector_closes.loc[sanc_idx]
    spy_vis = spy_close.loc[visible_idx]
    spy_sanc = spy_close.loc[sanc_idx]
    print(
        f"[sanctuary] visible {visible_idx[0].date()} -> {visible_idx[-1].date()} "
        f"({len(visible_idx)}); sanctuary {sanc_idx[0].date()} -> "
        f"{sanc_idx[-1].date()} ({len(sanc_idx)})"
    )

    # ── [1/6] Per-cell sweep ──
    print("\n[1/6] Per-cell sweep on VISIBLE window")
    print(
        f"  {'top_n':>5s} {'crash':>6s} {'ann SR':>9s} {'CI_lo':>9s} {'CI_hi':>9s} "
        f"{'mdd':>9s} {'panic%':>8s} {'canonical'}"
    )
    sweep_results: list[dict] = []
    canonical_returns: pd.Series | None = None
    for n in TOP_N_GRID:
        for cs in CRASH_SCALE_GRID:
            rets, panic = b6_sector_momentum_returns(sect_vis, spy_vis, top_n=n, crash_scale=cs)
            stats = _stats(rets)
            panic_pct = float(panic.mean())
            is_canonical = n == CANONICAL_TOP_N and cs == CANONICAL_CRASH_SCALE
            sweep_results.append(
                {
                    "top_n": n,
                    "crash_scale": cs,
                    "ann_sharpe": stats["sharpe"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "mdd": stats["mdd"],
                    "n": stats["n"],
                    "panic_pct": panic_pct,
                    "is_canonical": is_canonical,
                    "returns": rets,
                }
            )
            if is_canonical:
                canonical_returns = rets
            star = " *" if is_canonical else ""
            print(
                f"  {n:>5d} {cs:>5.2f}  {stats['sharpe']:>+8.4f} "
                f"{stats['ci_lo']:>+8.3f} {stats['ci_hi']:>+8.3f} "
                f"{stats['mdd']:>+8.2%} {panic_pct * 100:>7.2f}%{star}"
            )

    # Extra cell: overlay-OFF baseline (crash_scale=1.0) for overlay-test
    overlay_off_rets, _ = b6_sector_momentum_returns(
        sect_vis, spy_vis, top_n=CANONICAL_TOP_N, crash_scale=1.0
    )
    overlay_off_stats = _stats(overlay_off_rets)
    print(f"\n  [overlay-OFF reference at top_n={CANONICAL_TOP_N}, crash_scale=1.0]")
    print(
        f"    SR={overlay_off_stats['sharpe']:+.4f}, "
        f"CI_lo={overlay_off_stats['ci_lo']:+.3f}, MaxDD={overlay_off_stats['mdd']:+.2%}"
    )

    canonical_cell = next(c for c in sweep_results if c["is_canonical"])
    assert canonical_returns is not None

    # ── [2/6] L52 plateau ──
    print("\n[2/6] L52 plateau pre-flight")
    plateau_cells = [
        c
        for c in sweep_results
        if (
            (c["top_n"] == CANONICAL_TOP_N and c["crash_scale"] in CRASH_SCALE_GRID)
            or (c["crash_scale"] == CANONICAL_CRASH_SCALE and c["top_n"] in TOP_N_GRID)
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
        print("  insufficient sweep variance for DSR")

    # ── Pre-reg §3.7 early-exit ──
    early_fail_reasons: list[str] = []
    if not plateau_h1:
        early_fail_reasons.append(f"L52 plateau spread {plateau_spread * 100:.1f}% > 50% H1")
    if dsr is None or dsr.dsr_prob < DSR_DEPLOY_GATE:
        early_fail_reasons.append(
            f"DSR p={(dsr.dsr_prob if dsr else float('nan')):.4f} < {DSR_DEPLOY_GATE}"
        )
    if canonical_cell["ci_lo"] <= 0:
        early_fail_reasons.append(f"canonical CI_lo {canonical_cell['ci_lo']:+.3f} <= 0")

    if early_fail_reasons:
        print("\n" + "=" * 88)
        print("VERDICT (early-exit per pre-reg §3.7)")
        print("=" * 88)
        verdict = "RETIRE"
        print(f"  Verdict: {verdict}")
        for r in early_fail_reasons:
            print(f"    - {r}")
        _write_log_early_retire(
            report_path=REPORTS_DIR / "result_log.md",
            tr_check=tr_check,
            sweep_results=sweep_results,
            overlay_off_stats=overlay_off_stats,
            canonical_cell=canonical_cell,
            plateau_spread=plateau_spread,
            plateau_h1=plateau_h1,
            plateau_strict=plateau_strict,
            dsr=dsr,
            verdict=verdict,
            fail_reasons=early_fail_reasons,
        )
        print(f"\n  Result log: {(REPORTS_DIR / 'result_log.md').relative_to(PROJECT_ROOT)}")
        return

    # ── [4/6] 5-axis ──
    print("\n[4/6] 5-axis decision matrix")
    axis_ci_pass = canonical_cell["ci_lo"] > 0
    axis_dsr_pass = dsr is not None and dsr.dsr_prob >= DSR_DEPLOY_GATE

    # MC abs-DD on canonical bar returns
    mc_cfg = McConfig(
        block_size_bars=63,
        n_paths=200,
        bootstrap_method="iid",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )

    # MC on the per-bar canonical return series directly (more reliable
    # than re-simulating the cross-sectional rebalance under bootstrapped
    # prices — the latter would mangle the cross-sectional dispersion).
    def _strategy_fn_passthrough(df: pd.DataFrame) -> pd.Series:
        return df["close"]

    # Build a fake "close" from canonical returns -> cumulative product.
    canonical_eq = (1.0 + canonical_returns).cumprod().rename("canonical_eq")
    mc_res = run_block_mc(
        canonical_eq,
        mc_cfg,
        lambda df: df["close"].pct_change().fillna(0.0),
        periods_per_year=PERIODS_PER_YEAR,
    )
    axis_mc_pass = bool(mc_res.passes)
    print(
        f"  MC abs-DD: median {mc_res.median_max_dd * 100:+.2f}%, "
        f"95th {mc_res.percentile_95_max_dd * 100:+.2f}%, "
        f"P(>35%)={mc_res.prob_exceed_threshold:.2%} "
        f"({'PASS' if axis_mc_pass else 'FAIL'})"
    )

    # Sanctuary divergence
    sanc_rets, _ = b6_sector_momentum_returns(
        sect_sanc, spy_sanc, top_n=CANONICAL_TOP_N, crash_scale=CANONICAL_CRASH_SCALE
    )
    sanc_sr = float(sharpe(sanc_rets.dropna(), periods_per_year=PERIODS_PER_YEAR))
    sanc_diff = abs(canonical_cell["ann_sharpe"] - sanc_sr)
    axis_sanc_pass = sanc_diff <= 0.30
    print(
        f"  Sanctuary: visible {canonical_cell['ann_sharpe']:+.4f} vs sanctuary "
        f"{sanc_sr:+.4f} (diff {sanc_diff:.3f}, gate <= 0.30): "
        f"{'PASS' if axis_sanc_pass else 'FAIL'}"
    )

    # Noise on canonical bar returns (perturbing equity curve)
    noise_cfg = NoiseConfig(noise_levels=(0.10, 0.30, 0.50), n_trials=10, seed=42)
    noise_res = run_noise_robustness(
        canonical_eq.to_frame(name="close"),
        lambda df: df["close"].pct_change().fillna(0.0),
        periods_per_year=PERIODS_PER_YEAR,
        cfg=noise_cfg,
    )
    axis_noise_pass = bool(noise_res.passes)
    print(
        f"  Noise (sigma=0.5): base SR {noise_res.base_sharpe:+.4f}, "
        f"mean SR {noise_res.results[-1].mean_sharpe:+.4f}, "
        f"degradation {noise_res.results[-1].mean_degradation:.3f}, "
        f"p5 {noise_res.results[-1].p5_sharpe:+.4f} "
        f"({'PASS' if axis_noise_pass else 'FAIL'})"
    )

    axes_pass_count = sum(
        [axis_ci_pass, axis_dsr_pass, axis_mc_pass, axis_sanc_pass, axis_noise_pass]
    )
    print(f"\n  5-axis matrix: {axes_pass_count}/5 PASS")

    # Overlay-effectiveness test (pre-reg §3.7)
    overlay_mdd_ratio = (
        abs(canonical_cell["mdd"]) / abs(overlay_off_stats["mdd"])
        if abs(overlay_off_stats["mdd"]) > 1e-9
        else float("nan")
    )
    overlay_sr_ratio = (
        canonical_cell["ann_sharpe"] / overlay_off_stats["sharpe"]
        if abs(overlay_off_stats["sharpe"]) > 1e-9
        else float("nan")
    )
    print(
        f"  Overlay effectiveness: MaxDD ratio {overlay_mdd_ratio:.3f} (target<0.80 for tail reduction); "
        f"Sharpe ratio {overlay_sr_ratio:.3f} (target>=0.90)"
    )

    # ── [5/6] L65 ──
    print("\n[5/6] L65 ruin")
    single_ruin: list[dict] = []
    for w in [0.02, 0.05, 0.10]:
        r = assess_strategy_ruin(
            canonical_returns,
            deployment_weight=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
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

    # Joint ruin
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

    b6_rets = canonical_returns.dropna()
    b6_rets.index = pd.to_datetime(b6_rets.index).tz_localize(None).normalize()
    b6_rets = b6_rets.groupby(b6_rets.index).last().sort_index()

    candidates = [
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15, "b6": 0.00},
        {"gem": 0.66, "turtle": 0.17, "ic": 0.15, "b6": 0.02},
        {"gem": 0.64, "turtle": 0.16, "ic": 0.15, "b6": 0.05},
        {"gem": 0.62, "turtle": 0.13, "ic": 0.15, "b6": 0.10},
    ]
    joint_results: list[dict] = []
    print(f"  {'weights':>40s}  {'P_kill':>9s}  {'p95 DD':>9s}  verdict")
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={
                "gem": gem,
                "turtle": turtle,
                "ic": ic_basket,
                "b6": b6_rets,
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
            f"IC{int(w['ic'] * 100)}/B6{int(w['b6'] * 100)}"
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
    for w_b6 in [0.02, 0.05, 0.10]:
        w_g = 0.68 - 0.75 * w_b6
        w_t = 0.17 - 0.25 * w_b6
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "ic": ic_basket, "b6": b6_rets},
            {"gem": w_g, "turtle": w_t, "ic": 0.15, "b6": w_b6},
        ).dropna()
        tag = f"PROPOSED G{int(w_g * 100)}/T{int(w_t * 100)}/IC15/B6{int(w_b6 * 100)}"
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
    if not axis_mc_pass:
        fail_reasons.append("MC abs-DD gate FAIL")
    if not axis_sanc_pass:
        fail_reasons.append(f"sanctuary divergence {sanc_diff:.3f} > 0.30")
    if not axis_noise_pass:
        fail_reasons.append("noise robustness FAIL")
    if not all(r["passes"] for r in single_ruin):
        fail_reasons.append("L65 single FAIL at one or more weights")
    if not all(r["passes"] for r in joint_results):
        fail_reasons.append("L65 joint FAIL at one or more candidate mixes")
    if overlay_mdd_ratio >= 0.80:
        fail_reasons.append(f"overlay ineffective: MaxDD ratio {overlay_mdd_ratio:.3f} >= 0.80")
    if overlay_sr_ratio < 0.90:
        fail_reasons.append(
            f"overlay over-conservative: Sharpe ratio {overlay_sr_ratio:.3f} < 0.90"
        )

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
        if axes_pass_count >= 3:
            verdict = "CONDITIONAL_WATCHPOINT (paper-only)"
        else:
            verdict = "RETIRE"
    print(f"  Verdict: {verdict}")
    if fail_reasons:
        print("  Fail reasons:")
        for r in fail_reasons:
            print(f"    - {r}")
    print(f"  Best L67 Sharpe lift: {best_lift:+.3f} at {best_lift_tag}")

    _write_log_full(
        report_path=REPORTS_DIR / "result_log.md",
        tr_check=tr_check,
        sweep_results=sweep_results,
        overlay_off_stats=overlay_off_stats,
        canonical_cell=canonical_cell,
        plateau_spread=plateau_spread,
        plateau_h1=plateau_h1,
        plateau_strict=plateau_strict,
        dsr=dsr,
        mc_res=mc_res,
        sanc_diff=sanc_diff,
        sanc_sr=sanc_sr,
        noise_res=noise_res,
        overlay_mdd_ratio=overlay_mdd_ratio,
        overlay_sr_ratio=overlay_sr_ratio,
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
    tr_check: dict,
    sweep_results: list[dict],
    overlay_off_stats: dict,
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    verdict: str,
    fail_reasons: list[str],
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# B6 Sector Momentum + Crash Hedge -- V3.7 hybrid (EARLY RETIRE)\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg B6 Sector-Momentum Crash-Hedge 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")
        fh.write(
            "Early-exit per pre-reg §3.7. 5-axis MC + L65 + L67 sections intentionally not run.\n\n"
        )

        fh.write("## Pre-flight: total-return-adjusted close check\n\n")
        for tkr, ann in tr_check["per_ticker_ann_returns"].items():
            fh.write(f"- {tkr}: {ann * 100:+.2f}%\n")
        fh.write(
            f"\nMedian ann return: {tr_check['median_ann_return'] * 100:+.2f}%\n"
            f"Looks total-return-adjusted: {tr_check['looks_adjusted']}\n\n"
        )

        fh.write("## [1] Per-cell sweep\n\n")
        fh.write("| top_n | crash_scale | ann SR | CI_lo | CI_hi | mdd | panic% | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['top_n']} | {c['crash_scale']:.2f} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | "
                f"{c['panic_pct'] * 100:.2f}% | {star} |\n"
            )
        fh.write(
            f"\n**Overlay-OFF reference** (top_n={CANONICAL_TOP_N}, crash_scale=1.0): "
            f"SR={overlay_off_stats['sharpe']:+.4f}, "
            f"CI_lo={overlay_off_stats['ci_lo']:+.3f}, "
            f"MaxDD={overlay_off_stats['mdd']:+.2%}\n"
        )

        fh.write("\n## [2] L52 plateau\n\n")
        fh.write(
            f"- Plateau spread: **{plateau_spread * 100:.1f}%** "
            f"(H1 ≤50%: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict ≤30%: {'PASS' if plateau_strict else 'FAIL'})\n"
        )

        fh.write("\n## [3] DSR deflation\n\n")
        if dsr is not None:
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] (9 cells) = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
                f"- Gate p ≥ 0.95: "
                f"{'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'}\n"
            )

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")


def _write_log_full(
    *,
    report_path: Path,
    tr_check: dict,
    sweep_results: list[dict],
    overlay_off_stats: dict,
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    mc_res,
    sanc_diff: float,
    sanc_sr: float,
    noise_res,
    overlay_mdd_ratio: float,
    overlay_sr_ratio: float,
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
        fh.write("# B6 Sector Momentum + Crash Hedge -- V3.7 hybrid audit\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg B6 Sector-Momentum Crash-Hedge 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")

        fh.write("## Pre-flight: total-return-adjusted close check\n\n")
        for tkr, ann in tr_check["per_ticker_ann_returns"].items():
            fh.write(f"- {tkr}: {ann * 100:+.2f}%\n")
        fh.write(
            f"\nMedian ann return: {tr_check['median_ann_return'] * 100:+.2f}%\n"
            f"Looks total-return-adjusted: {tr_check['looks_adjusted']}\n\n"
        )

        fh.write("## [1] Per-cell sweep\n\n")
        fh.write("| top_n | crash_scale | ann SR | CI_lo | CI_hi | mdd | panic% | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['top_n']} | {c['crash_scale']:.2f} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | "
                f"{c['panic_pct'] * 100:.2f}% | {star} |\n"
            )
        fh.write(
            f"\n**Overlay-OFF reference**: SR={overlay_off_stats['sharpe']:+.4f}, "
            f"CI_lo={overlay_off_stats['ci_lo']:+.3f}, "
            f"MaxDD={overlay_off_stats['mdd']:+.2%}\n"
        )

        fh.write("\n## [2] L52 plateau\n\n")
        fh.write(
            f"- Spread: {plateau_spread * 100:.1f}% "
            f"(H1: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict: {'PASS' if plateau_strict else 'FAIL'})\n"
        )

        fh.write("\n## [3] DSR deflation\n\n")
        if dsr is not None:
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
            )

        fh.write("\n## [4] 5-axis matrix\n\n")
        fh.write(
            f"- CI_lo: {canonical_cell['ci_lo']:+.3f} "
            f"({'PASS' if canonical_cell['ci_lo'] > 0 else 'FAIL'})\n"
        )
        if dsr is not None:
            fh.write(
                f"- DSR p: {dsr.dsr_prob:.4f} "
                f"({'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'})\n"
            )
        fh.write(
            f"- MC abs-DD: median {mc_res.median_max_dd * 100:+.2f}%, "
            f"P(>35%)={mc_res.prob_exceed_threshold:.2%} "
            f"({'PASS' if mc_res.passes else 'FAIL'})\n"
            f"- Sanctuary: visible {canonical_cell['ann_sharpe']:+.4f} vs "
            f"sanctuary {sanc_sr:+.4f} (diff {sanc_diff:.3f}) "
            f"({'PASS' if sanc_diff <= 0.30 else 'FAIL'})\n"
            f"- Noise: mean deg {noise_res.results[-1].mean_degradation:.3f}, "
            f"p5 {noise_res.results[-1].p5_sharpe:+.4f} "
            f"({'PASS' if noise_res.passes else 'FAIL'})\n"
        )

        fh.write("\n## Overlay-effectiveness test\n\n")
        fh.write(
            f"- MaxDD ratio (canonical / overlay-off): {overlay_mdd_ratio:.3f} "
            f"(target < 0.80 for tail reduction)\n"
            f"- Sharpe ratio (canonical / overlay-off): {overlay_sr_ratio:.3f} "
            f"(target ≥ 0.90)\n"
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
                f"IC{int(w['ic'] * 100)}/B6{int(w['b6'] * 100)}"
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
