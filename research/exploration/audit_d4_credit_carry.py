"""D4 HY/IG credit carry audit (Israel-Palhares-Richardson JoIM 2018).

V3.7 hybrid audit per pre-reg `directives/Pre-Reg D4 HY-IG Credit Carry 2026-05-19.md`:

    1. L21 causality smoke
    2. L52 plateau pre-flight (3x3 grid)
    3. DSR deflation across 9 cells
    4. 5-axis decision matrix
    5. L65 single + joint ruin
    6. L67 10-metric portfolio inclusion

Strategy: long HYG / short LQD, vol-targeted; some cells gate on the
HYG/LQD ratio's SMA. Per §1 + §5: HYG/LQD ETF total-return prices
include the carry via dividends, so L74 sensitivity is N/A IFF our
parquets are total-return adjusted (verified in pre-flight).

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_d4_credit_carry.py
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
from research.exploration.audit_ic_equity_daily import (
    _load_d as _load_ic,
)
from research.exploration.audit_ic_equity_daily import (
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "d4_credit_carry"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PERIODS_PER_YEAR = BARS_PER_YEAR["D"]
COST_BPS_PER_LEG = 1.0  # HYG/LQD at retail IBKR; 2 legs per turnover

# ── Sweep grid (pre-registered §2) ──
SMA_PERIODS = [0, 50, 100]  # 0 = filter off (strict IPR)
VOL_TARGETS = [0.06, 0.08, 0.10]
CANONICAL_SMA = 100
CANONICAL_VOL_TARGET = 0.08

PLATEAU_GATE_H1 = 0.50
PLATEAU_GATE_STRICT = 0.30
DSR_DEPLOY_GATE = 0.95


def _load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{ticker}_D.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["close"])[["close"]].astype(float)


def _verify_total_return_close(hyg: pd.DataFrame, lqd: pd.DataFrame) -> dict:
    """Pre-flight: confirm parquets are total-return (adjusted) close.

    Heuristic test: HYG paid ~5-6%/yr in dividends historically.
    Compute realised annualised price return over the full window:
    if it's well below the typical 5-7% credit return of HY (e.g., < 2%),
    the parquet is likely UNADJUSTED close and L74 sensitivity is required.
    """
    hyg_ret_ann = (hyg["close"].iloc[-1] / hyg["close"].iloc[0]) ** (
        PERIODS_PER_YEAR / len(hyg)
    ) - 1.0
    lqd_ret_ann = (lqd["close"].iloc[-1] / lqd["close"].iloc[0]) ** (
        PERIODS_PER_YEAR / len(lqd)
    ) - 1.0
    return {
        "hyg_ann_price_return": float(hyg_ret_ann),
        "lqd_ann_price_return": float(lqd_ret_ann),
        # Heuristic: if HYG annualised price return < 2%, almost certainly
        # NOT adjusted (true HY total return historically ~5-7%/yr).
        "looks_adjusted": bool(hyg_ret_ann > 0.02),
    }


def d4_credit_carry_returns(
    hyg: pd.DataFrame,
    lqd: pd.DataFrame,
    *,
    sma_period: int = CANONICAL_SMA,
    vol_target: float = CANONICAL_VOL_TARGET,
    ewma_span: int = 20,
    cost_bps_per_leg: float = COST_BPS_PER_LEG,
) -> pd.Series:
    """Per-bar net return of long-HYG / short-LQD vol-targeted carry.

    sma_period=0 disables the trend filter (strict IPR carry).
    Otherwise, the position only engages when HYG/LQD ratio > SMA.
    """
    # Align on common index (HYG starts later than LQD).
    df = pd.concat({"hyg": hyg["close"], "lqd": lqd["close"]}, axis=1).dropna()
    r_hy = np.log(df["hyg"] / df["hyg"].shift(1)).fillna(0.0)
    r_ig = np.log(df["lqd"] / df["lqd"].shift(1)).fillna(0.0)
    carry_ret = r_hy - r_ig

    # Realised vol of the net carry, EWMA span 20.
    var = carry_ret.pow(2).ewm(span=ewma_span, adjust=False, min_periods=ewma_span).mean()
    realised_vol_ann = np.sqrt(var * PERIODS_PER_YEAR)
    scale = (vol_target / realised_vol_ann.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)

    # Trend filter on HYG/LQD ratio (per L21 — uses past data only via .shift(1))
    if sma_period > 0:
        ratio = df["hyg"] / df["lqd"]
        sma = ratio.rolling(sma_period, min_periods=sma_period).mean()
        signal = (ratio > sma).astype(float)
    else:
        signal = pd.Series(1.0, index=df.index)

    position = (signal * scale).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * carry_ret
    dpos = position.diff().abs().fillna(0.0)
    # Cost: 2 legs per turnover unit
    cost = dpos * (2.0 * cost_bps_per_leg / 10_000.0)
    return (gross - cost).rename("ret")


def assert_causal_d4(hyg: pd.DataFrame, lqd: pd.DataFrame) -> None:
    """L21 smoke: corrupting future bars (by ALIGNED DATE, not raw position)
    must not change past return outputs.

    Critical fix: HYG starts 2010-04-04, LQD starts 2002-07-22. Corrupting
    by raw integer position into either input would corrupt different
    calendar dates -- LQD's iloc[N:] for the same N reaches back ~5 years
    further into the aligned (HYG-bounded) frame. Use the aligned-frame
    cutoff DATE to corrupt the same trading-day window in both inputs.
    """
    base = d4_credit_carry_returns(hyg, lqd)
    aligned_index = base.index
    # Corrupt the last 50 aligned trading days. Safe window for past-bar
    # comparison: pad by 250 bars to cover the SMA-100 + EWMA-20 lookbacks.
    n_aligned = len(aligned_index)
    if n_aligned < 400:
        print("[D4] L21 skipped (window too short)")
        return
    cutoff_date = aligned_index[n_aligned - 50]
    safe_cutoff_date = aligned_index[n_aligned - 50 - 250]

    hyg_c = hyg.copy()
    lqd_c = lqd.copy()
    hyg_c.loc[hyg_c.index >= cutoff_date] *= 100.0
    lqd_c.loc[lqd_c.index >= cutoff_date] *= 100.0
    pert = d4_credit_carry_returns(hyg_c, lqd_c)

    past_base = base[base.index < safe_cutoff_date]
    past_pert = pert[pert.index < safe_cutoff_date]
    diff = (past_base - past_pert).abs().max()
    assert diff < 1e-12, f"L21 fail: max past-bar diff {diff}"
    print(
        f"[D4] L21 PASS (diff={diff:.2e}, "
        f"corrupted from {cutoff_date.date()}, "
        f"past window ends {safe_cutoff_date.date()})"
    )


def _stats(rets: pd.Series, label: str) -> dict:
    rets = rets.dropna()
    sr = float(sharpe(rets, periods_per_year=PERIODS_PER_YEAR))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets, periods_per_year=PERIODS_PER_YEAR, n_resamples=2000, seed=42
    )
    mdd = float(max_drawdown(rets))
    return {
        "label": label,
        "sharpe": sr,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(len(rets)),
        "mdd": mdd,
    }


def main() -> None:
    print("=" * 88)
    print("D4 HY/IG credit carry -- V3.7 hybrid audit (IPR JoIM 2018)")
    print("=" * 88)

    # ── Load + pre-flight ──
    hyg = _load_etf("HYG")
    lqd = _load_etf("LQD")
    print(f"\n[load] HYG: {len(hyg)} bars, {hyg.index[0].date()} -> {hyg.index[-1].date()}")
    print(f"[load] LQD: {len(lqd)} bars, {lqd.index[0].date()} -> {lqd.index[-1].date()}")

    # Adjusted-close pre-flight (pre-reg §1, §5 caveat 3)
    tr_check = _verify_total_return_close(hyg, lqd)
    print(f"\n[pre-flight] HYG ann price return = {tr_check['hyg_ann_price_return'] * 100:+.2f}%")
    print(f"[pre-flight] LQD ann price return = {tr_check['lqd_ann_price_return'] * 100:+.2f}%")
    print(
        f"[pre-flight] looks total-return-adjusted: {tr_check['looks_adjusted']} "
        f"(threshold: HYG ann return > 2%)"
    )
    if not tr_check["looks_adjusted"]:
        print(
            "  *** WARNING: parquets may use UNADJUSTED close. Carry premium would be "
            "understated. L74 sensitivity check needed -- pre-reg §1 contingency."
        )

    # ── L21 causality smoke ──
    print("\n[L21] causality smoke test")
    assert_causal_d4(hyg, lqd)

    # ── Build aligned-window inputs ──
    # Align HYG and LQD on common dates first.
    aligned = pd.concat({"hyg": hyg["close"], "lqd": lqd["close"]}, axis=1).dropna()
    print(
        f"\n[align] common window: {aligned.index[0].date()} -> "
        f"{aligned.index[-1].date()} ({len(aligned)} bars)"
    )

    # Sanctuary: last 12 months
    hyg_aligned = aligned[["hyg"]].rename(columns={"hyg": "close"})
    lqd_aligned = aligned[["lqd"]].rename(columns={"lqd": "close"})
    sanc = slice_sanctuary(hyg_aligned, months=12)
    visible_idx = sanc.visible.index
    sanc_idx = sanc.sanctuary.index
    hyg_visible = hyg_aligned.loc[visible_idx]
    lqd_visible = lqd_aligned.loc[visible_idx]
    hyg_sanc = hyg_aligned.loc[sanc_idx]
    lqd_sanc = lqd_aligned.loc[sanc_idx]
    print(
        f"[sanctuary] visible {visible_idx[0].date()} -> {visible_idx[-1].date()} "
        f"({len(visible_idx)} bars); sanctuary {sanc_idx[0].date()} -> "
        f"{sanc_idx[-1].date()} ({len(sanc_idx)} bars)"
    )

    # ── [1/6] Per-cell sweep on visible ──
    print("\n[1/6] Per-cell sweep on VISIBLE window")
    print(
        f"  {'sma':>5s} {'vol_t':>6s} {'ann SR':>9s} {'CI_lo':>9s} {'CI_hi':>9s} "
        f"{'mdd':>9s} {'canonical'}"
    )
    sweep_results: list[dict] = []
    canonical_returns: pd.Series | None = None
    for sma in SMA_PERIODS:
        for vt in VOL_TARGETS:
            rets = d4_credit_carry_returns(hyg_visible, lqd_visible, sma_period=sma, vol_target=vt)
            stats = _stats(rets, f"sma{sma}_vt{vt}")
            is_canonical = sma == CANONICAL_SMA and vt == CANONICAL_VOL_TARGET
            sweep_results.append(
                {
                    "sma_period": sma,
                    "vol_target": vt,
                    "ann_sharpe": stats["sharpe"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "mdd": stats["mdd"],
                    "n": stats["n"],
                    "is_canonical": is_canonical,
                    "returns": rets,
                }
            )
            if is_canonical:
                canonical_returns = rets
            star = " *" if is_canonical else ""
            print(
                f"  {sma:>5d} {vt:>5.2f}  {stats['sharpe']:>+8.4f} "
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
            (c["sma_period"] == CANONICAL_SMA and c["vol_target"] in VOL_TARGETS)
            or (c["vol_target"] == CANONICAL_VOL_TARGET and c["sma_period"] in SMA_PERIODS)
        )
    ]
    plateau_sharpes = [c["ann_sharpe"] for c in plateau_cells]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    plateau_h1 = plateau_spread <= PLATEAU_GATE_H1
    plateau_strict = plateau_spread <= PLATEAU_GATE_STRICT
    print(
        f"  Plateau set ({len(plateau_cells)} cells): mean SR {plateau_mean:+.4f}, "
        f"spread {plateau_spread * 100:.1f}%  "
        f"(H1 {'PASS' if plateau_h1 else 'FAIL'}, "
        f"strict {'PASS' if plateau_strict else 'FAIL'})"
    )

    # ── [3/6] DSR ──
    print("\n[3/6] DSR deflation on canonical (9 cells)")
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

    # ── Pre-reg §3.7 early-exit on plateau/DSR/CI_lo failure ──
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

    # ── [4/6] 5-axis decision matrix ──
    print("\n[4/6] 5-axis decision matrix (canonical cell)")
    axis_ci_pass = canonical_cell["ci_lo"] > 0
    axis_dsr_pass = dsr is not None and dsr.dsr_prob >= DSR_DEPLOY_GATE

    # MC abs-DD on canonical -- shared-block MC with HYG primary + LQD extra
    mc_cfg = McConfig(
        block_size_bars=21,
        n_paths=200,
        bootstrap_method="shared_block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )

    def _strategy_fn_mc(df: pd.DataFrame) -> pd.Series:
        # df has 'close' (HYG primary) + 'lqd_close' (extra_series).
        hyg_p = df[["close"]].copy()
        lqd_p = df[["lqd_close"]].rename(columns={"lqd_close": "close"})
        return d4_credit_carry_returns(
            hyg_p, lqd_p, sma_period=CANONICAL_SMA, vol_target=CANONICAL_VOL_TARGET
        )

    mc_res = run_block_mc(
        hyg_visible["close"],
        mc_cfg,
        _strategy_fn_mc,
        periods_per_year=PERIODS_PER_YEAR,
        extra_series={"lqd_close": lqd_visible["close"]},
    )
    axis_mc_pass = bool(mc_res.passes)
    print(
        f"  MC abs-DD: median {mc_res.median_max_dd * 100:+.2f}%, "
        f"95th {mc_res.percentile_95_max_dd * 100:+.2f}%, "
        f"P(>35%)={mc_res.prob_exceed_threshold:.2%} "
        f"({'PASS' if axis_mc_pass else 'FAIL'})"
    )

    # Sanctuary divergence
    sanc_rets = d4_credit_carry_returns(
        hyg_sanc, lqd_sanc, sma_period=CANONICAL_SMA, vol_target=CANONICAL_VOL_TARGET
    )
    sanc_sr = float(sharpe(sanc_rets.dropna(), periods_per_year=PERIODS_PER_YEAR))
    sanc_diff = abs(canonical_cell["ann_sharpe"] - sanc_sr)
    axis_sanc_pass = sanc_diff <= 0.30
    print(
        f"  Sanctuary: visible SR {canonical_cell['ann_sharpe']:+.4f} vs "
        f"sanctuary SR {sanc_sr:+.4f} (diff {sanc_diff:.3f}, gate <= 0.30): "
        f"{'PASS' if axis_sanc_pass else 'FAIL'}"
    )

    # Noise robustness on HYG (LQD held constant for this gate)
    noise_cfg = NoiseConfig(noise_levels=(0.10, 0.30, 0.50), n_trials=10, seed=42)

    def _strategy_fn_noise(df: pd.DataFrame) -> pd.Series:
        return d4_credit_carry_returns(
            df, lqd_visible, sma_period=CANONICAL_SMA, vol_target=CANONICAL_VOL_TARGET
        )

    noise_res = run_noise_robustness(
        hyg_visible,
        _strategy_fn_noise,
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

    # ── [5/6] L65 ruin ──
    print("\n[5/6] L65 ruin gates")
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
            f"p95 DD={r.p95_maxdd_at_size * 100:.2f}%, "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    # Joint ruin with current LIVE (GEM + Turtle + IC top-3 basket)
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

    # Normalise canonical returns index for joint analysis
    d4_rets = canonical_returns.dropna()
    d4_rets.index = pd.to_datetime(d4_rets.index).tz_localize(None).normalize()
    d4_rets = d4_rets.groupby(d4_rets.index).last().sort_index()

    candidates = [
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15, "d4": 0.00},  # current LIVE
        {"gem": 0.66, "turtle": 0.17, "ic": 0.15, "d4": 0.02},
        {"gem": 0.64, "turtle": 0.16, "ic": 0.15, "d4": 0.05},
        {"gem": 0.62, "turtle": 0.13, "ic": 0.15, "d4": 0.10},
    ]
    joint_results: list[dict] = []
    print(f"  {'weights':>40s}  {'P_kill':>9s}  {'p95 DD':>9s}  verdict")
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={
                "gem": gem,
                "turtle": turtle,
                "ic": ic_basket,
                "d4": d4_rets,
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
            f"IC{int(w['ic'] * 100)}/D4{int(w['d4'] * 100)}"
        )
        print(
            f"  {w_str:>40s}  {r.p_kill_trip * 100:>7.3f}%  "
            f"{r.p95_maxdd_at_size * 100:>7.2f}%  "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    # ── [6/6] L67 portfolio inclusion ──
    print("\n[6/6] L67 portfolio inclusion vs 60/40")
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
    for w_d4 in [0.02, 0.05, 0.10]:
        # Carve from GEM (75%) and turtle (25%) proportionally.
        w_g = 0.68 - 0.75 * w_d4
        w_t = 0.17 - 0.25 * w_d4
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "ic": ic_basket, "d4": d4_rets},
            {"gem": w_g, "turtle": w_t, "ic": 0.15, "d4": w_d4},
        ).dropna()
        tag = f"PROPOSED G{int(w_g * 100)}/T{int(w_t * 100)}/IC15/D4{int(w_d4 * 100)}"
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
        fail_reasons.append("L65 single ruin FAIL at one or more weights")
    if not all(r["passes"] for r in joint_results):
        fail_reasons.append("L65 joint ruin FAIL at one or more candidate mixes")

    # L67 lift check: compare current vs best proposed
    cur_sharpe = ev_current.results[0].portfolio_value  # axis 1 = Sharpe
    best_lift = -1e9
    best_lift_tag = ""
    for tag, ev in eval_proposals:
        prop_sharpe = ev.results[0].portfolio_value
        lift = prop_sharpe - cur_sharpe
        if lift > best_lift:
            best_lift = lift
            best_lift_tag = tag

    if not fail_reasons and best_lift >= 0.05:
        verdict = f"DEPLOY-eligible at {best_lift_tag} (Sharpe lift +{best_lift:.3f})"
    elif not fail_reasons:
        verdict = (
            f"CONDITIONAL_WATCHPOINT (paper-only) -- gates PASS but Sharpe "
            f"lift {best_lift:+.3f} below +0.05 deploy threshold"
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

    # ── Write result log ──
    _write_log_full(
        report_path=REPORTS_DIR / "result_log.md",
        tr_check=tr_check,
        sweep_results=sweep_results,
        canonical_cell=canonical_cell,
        plateau_spread=plateau_spread,
        plateau_h1=plateau_h1,
        plateau_strict=plateau_strict,
        dsr=dsr,
        mc_res=mc_res,
        sanc_diff=sanc_diff,
        sanc_sr=sanc_sr,
        noise_res=noise_res,
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
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    verdict: str,
    fail_reasons: list[str],
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# D4 HY/IG Credit Carry -- V3.7 hybrid audit (EARLY RETIRE)\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg D4 HY-IG Credit Carry 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")
        fh.write(
            "Early-exit triggered per pre-reg §3.7. The 5-axis MC + L65 + L67 "
            "sections were intentionally not run; the verdict is sealed by "
            "the L52 plateau and/or DSR gates.\n\n"
        )

        fh.write("## Pre-flight: total-return-adjusted close check\n\n")
        fh.write(
            f"- HYG annualised price return: {tr_check['hyg_ann_price_return'] * 100:+.2f}%\n"
            f"- LQD annualised price return: {tr_check['lqd_ann_price_return'] * 100:+.2f}%\n"
            f"- Looks total-return-adjusted: {tr_check['looks_adjusted']}\n\n"
        )

        fh.write("## [1] Per-cell sweep (visible window)\n\n")
        fh.write("| sma_period | vol_target | ann SR | CI_lo | CI_hi | mdd | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['sma_period']} | {c['vol_target']:.2f} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau pre-flight\n\n")
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
        else:
            fh.write("- DSR not computed (insufficient sweep variance)\n")

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")


def _write_log_full(
    *,
    report_path: Path,
    tr_check: dict,
    sweep_results: list[dict],
    canonical_cell: dict,
    plateau_spread: float,
    plateau_h1: bool,
    plateau_strict: bool,
    dsr,
    mc_res,
    sanc_diff: float,
    sanc_sr: float,
    noise_res,
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
        fh.write("# D4 HY/IG Credit Carry -- V3.7 hybrid audit\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg D4 HY-IG Credit Carry 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")

        fh.write("## Pre-flight: total-return-adjusted close check\n\n")
        fh.write(
            f"- HYG annualised price return: {tr_check['hyg_ann_price_return'] * 100:+.2f}%\n"
            f"- LQD annualised price return: {tr_check['lqd_ann_price_return'] * 100:+.2f}%\n"
            f"- Looks total-return-adjusted: {tr_check['looks_adjusted']}\n\n"
        )

        fh.write("## [1] Per-cell sweep (visible window)\n\n")
        fh.write("| sma_period | vol_target | ann SR | CI_lo | CI_hi | mdd | canonical |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['sma_period']} | {c['vol_target']:.2f} | "
                f"{c['ann_sharpe']:+.4f} | {c['ci_lo']:+.3f} | "
                f"{c['ci_hi']:+.3f} | {c['mdd']:+.2%} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau pre-flight\n\n")
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

        fh.write("\n## [4] 5-axis decision matrix\n\n")
        fh.write(
            f"- CI_lo > 0: {canonical_cell['ci_lo']:+.3f} "
            f"({'PASS' if canonical_cell['ci_lo'] > 0 else 'FAIL'})\n"
        )
        if dsr is not None:
            fh.write(
                f"- DSR p ≥ 0.95: {dsr.dsr_prob:.4f} "
                f"({'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'})\n"
            )
        fh.write(
            f"- MC abs-DD: median {mc_res.median_max_dd * 100:+.2f}%, "
            f"95th {mc_res.percentile_95_max_dd * 100:+.2f}%, "
            f"P(>35%)={mc_res.prob_exceed_threshold:.2%} "
            f"({'PASS' if mc_res.passes else 'FAIL'})\n"
            f"- Sanctuary divergence: visible {canonical_cell['ann_sharpe']:+.4f} vs "
            f"sanctuary {sanc_sr:+.4f} (diff {sanc_diff:.3f}) "
            f"({'PASS' if sanc_diff <= 0.30 else 'FAIL'})\n"
            f"- Noise (sigma=0.5): mean degradation "
            f"{noise_res.results[-1].mean_degradation:.3f}, "
            f"p5 {noise_res.results[-1].p5_sharpe:+.4f} "
            f"({'PASS' if noise_res.passes else 'FAIL'})\n"
        )

        fh.write("\n## [5] L65 ruin\n\n")
        fh.write("### Single-strategy\n\n")
        fh.write("| Weight | P_kill | p95 DD | p50 DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|:---:|\n")
        for r in single_ruin:
            fh.write(
                f"| {r['weight']:.0%} | {r['p_kill'] * 100:.3f}% | "
                f"{r['p95_dd'] * 100:.2f}% | {r['p50_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )
        fh.write("\n### Joint (with current LIVE strategies)\n\n")
        fh.write("| Weights | P_kill | p95 DD | Verdict |\n")
        fh.write("|---|---:|---:|:---:|\n")
        for r in joint_results:
            w = r["weights"]
            w_str = (
                f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/"
                f"IC{int(w['ic'] * 100)}/D4{int(w['d4'] * 100)}"
            )
            fh.write(
                f"| {w_str} | {r['p_kill'] * 100:.3f}% | "
                f"{r['p95_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [6] L67 10-metric portfolio inclusion (vs 60/40)\n\n")
        fh.write("### CURRENT (no D4)\n\n```\n" + ev_current.report() + "\n```\n\n")
        for tag, ev in eval_proposals:
            fh.write(f"### {tag}\n\n```\n" + ev.report() + "\n```\n\n")

        fh.write(f"\n### Best L67 Sharpe lift: {best_lift:+.3f} at `{best_lift_tag}`\n")

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")


if __name__ == "__main__":
    main()
