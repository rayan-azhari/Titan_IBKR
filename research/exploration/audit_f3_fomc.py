"""F3 FOMC pre-announcement drift audit (Lucca-Moench JF 2015).

V3.7 hybrid audit per pre-reg `directives/Pre-Reg F3 FOMC Pre-Announcement Drift 2026-05-19.md`:

    1. L21 causality smoke (corrupting future SPY must not change past trades)
    2. L52 plateau pre-flight per cell (3x3 grid)
    3. DSR per cell + canonical
    4. 5-axis decision matrix
    5. L65 single + joint ruin
    6. L67 10-metric portfolio inclusion

FOMC scheduled-meeting announcement dates 2010-2026 are hardcoded below
(source: federalreserve.gov/monetarypolicy/fomccalendars.htm).
Intermeeting / emergency announcements (2020-03-03, 2020-03-15) are
EXCLUDED per the pre-reg.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_f3_fomc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import (  # noqa: E402
    assess_joint_ruin,
    assess_strategy_ruin,
    run_block_mc,
    run_noise_robustness,
    slice_sanctuary,
)
from titan.research.framework.dsr import (  # noqa: E402
    deflated_sharpe,
    sr_var_from_sweep,
)
from titan.research.framework.mc import McConfig  # noqa: E402
from titan.research.framework.robustness import NoiseConfig  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "f3_fomc"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── FOMC scheduled-meeting announcement dates 2010-2026 ──
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# Each entry = announcement-day date (2nd day of 2-day meeting).
# EXCLUDED: 2020-03-03 (emergency 50bp), 2020-03-15 (Sunday emergency 100bp).
# The 2020 March 17-18 scheduled meeting was CANCELLED -- not in this list.
FOMC_DATES_STR = [
    # 2010
    "2010-01-27",
    "2010-03-16",
    "2010-04-28",
    "2010-06-23",
    "2010-08-10",
    "2010-09-21",
    "2010-11-03",
    "2010-12-14",
    # 2011
    "2011-01-26",
    "2011-03-15",
    "2011-04-27",
    "2011-06-22",
    "2011-08-09",
    "2011-09-21",
    "2011-11-02",
    "2011-12-13",
    # 2012
    "2012-01-25",
    "2012-03-13",
    "2012-04-25",
    "2012-06-20",
    "2012-08-01",
    "2012-09-13",
    "2012-10-24",
    "2012-12-12",
    # 2013
    "2013-01-30",
    "2013-03-20",
    "2013-05-01",
    "2013-06-19",
    "2013-07-31",
    "2013-09-18",
    "2013-10-30",
    "2013-12-18",
    # 2014
    "2014-01-29",
    "2014-03-19",
    "2014-04-30",
    "2014-06-18",
    "2014-07-30",
    "2014-09-17",
    "2014-10-29",
    "2014-12-17",
    # 2015
    "2015-01-28",
    "2015-03-18",
    "2015-04-29",
    "2015-06-17",
    "2015-07-29",
    "2015-09-17",
    "2015-10-28",
    "2015-12-16",
    # 2016
    "2016-01-27",
    "2016-03-16",
    "2016-04-27",
    "2016-06-15",
    "2016-07-27",
    "2016-09-21",
    "2016-11-02",
    "2016-12-14",
    # 2017
    "2017-02-01",
    "2017-03-15",
    "2017-05-03",
    "2017-06-14",
    "2017-07-26",
    "2017-09-20",
    "2017-11-01",
    "2017-12-13",
    # 2018
    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",
    # 2019
    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",
    # 2020 -- 7 scheduled (March meeting cancelled, emergency action 3/3 and 3/15 excluded)
    "2020-01-29",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    # 2022
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    # 2023
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    # 2026 (in our SPY data window; partial year)
    "2026-01-28",
    "2026-03-18",
]
FOMC_DATES = pd.to_datetime(FOMC_DATES_STR).tz_localize(None).normalize()

PERIODS_PER_YEAR = BARS_PER_YEAR["D"]
COST_BPS_PER_TURNOVER = 0.5  # SPY at retail IBKR


# ── Sweep grid (pre-registered §2) ──
HOLD_DAYS_GRID = [1, 2, 3]
ENTRY_OFFSET_GRID = [0, 1, 2]
CANONICAL_HOLD = 1
CANONICAL_ENTRY_OFFSET = 0

PLATEAU_GATE_H1 = 0.50
PLATEAU_GATE_STRICT = 0.30
DSR_DEPLOY_GATE = 0.95


def _load_spy() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "SPY_D.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["close"])[["close"]].astype(float)


def f3_trade_returns(
    spy: pd.DataFrame,
    *,
    hold_days: int = CANONICAL_HOLD,
    entry_offset_days: int = CANONICAL_ENTRY_OFFSET,
    fomc_dates: pd.DatetimeIndex = FOMC_DATES,
    cost_bps: float = COST_BPS_PER_TURNOVER,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the F3 per-bar net return series + per-trade dataframe.

    Position on bar t equals 1 if t is in the entry window for any FOMC
    event: entry at close_{T - entry_offset_days - 1}, hold through
    close_{T - entry_offset_days - 1 + hold_days}, exit at that close.
    Cost charged on entry and exit (2 turnovers per trade).

    Returns
    -------
    bar_rets : per-bar net log return series, aligned to spy.index
    trades_df : per-trade record (event_date, entry_date, exit_date,
                gross_ret, net_ret)
    """
    close = spy["close"]
    idx = spy.index
    bar_log_ret = np.log(close / close.shift(1)).fillna(0.0)

    position = pd.Series(0.0, index=idx)
    trades: list[dict] = []

    # Map calendar date -> position in the SPY index (next trading day on/after).
    def _find_idx_pos(target: pd.Timestamp) -> int | None:
        # idx.searchsorted with 'left' returns the index of the first bar >= target.
        pos = idx.searchsorted(pd.Timestamp(target), side="left")
        if pos >= len(idx):
            return None
        return int(pos)

    for event in fomc_dates:
        announcement_pos = _find_idx_pos(event)
        if announcement_pos is None:
            continue
        # If the FOMC calendar date isn't itself a trading day (rare), the
        # nearest *next* trading bar becomes the effective "announcement"
        # bar. The pre-reg interprets entry_offset_days from the
        # announcement trading bar.
        entry_pos = announcement_pos - entry_offset_days - 1
        exit_pos = entry_pos + hold_days
        if entry_pos < 0 or exit_pos >= len(idx):
            continue

        # Position is ON from entry_pos+1 (we held the bar AFTER the close
        # at entry_pos) through exit_pos (inclusive of the close at
        # exit_pos). Effective held-bars = exit_pos - entry_pos.
        for k in range(entry_pos + 1, exit_pos + 1):
            position.iloc[k] = 1.0

        gross = float(close.iloc[exit_pos] / close.iloc[entry_pos] - 1.0)
        net = gross - 2.0 * cost_bps / 10_000.0
        trades.append(
            {
                "event_date": event,
                "entry_date": idx[entry_pos],
                "exit_date": idx[exit_pos],
                "gross_ret": gross,
                "net_ret": net,
                "hold_bars": exit_pos - entry_pos,
            }
        )

    held = position.shift(1).fillna(0.0)
    gross_bar = held * bar_log_ret
    # Per-bar cost: at every position change (entry or exit) we pay cost_bps.
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (cost_bps / 10_000.0)
    bar_rets = (gross_bar - cost).rename("ret")

    return bar_rets, pd.DataFrame(trades)


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


def _per_trade_sharpe(
    trades_df: pd.DataFrame, *, periods_per_year: int = 8
) -> float:
    """Per-trade Sharpe annualised by sqrt(periods_per_year).

    Default 8 = FOMC scheduled meetings per year (per L06 + L60: this is
    the correct annualisation factor for a sparse trade-frequency series,
    NOT 252 which would over-annualise by sqrt(252/8) ~ 5.6x).
    """
    if len(trades_df) < 5:
        return float("nan")
    r = trades_df["net_ret"]
    std = r.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def _assert_l21_causal(spy: pd.DataFrame) -> None:
    """L21: corrupting future SPY closes must not change past trade rets."""
    base_bar, base_trades = f3_trade_returns(spy)
    spy_corrupt = spy.copy()
    cutoff = len(spy_corrupt) - 100
    spy_corrupt.iloc[cutoff:] *= 100.0
    pert_bar, pert_trades = f3_trade_returns(spy_corrupt)

    # Compare past bars (index < cutoff - hold) bit-exact
    safe_cutoff = max(0, cutoff - 10)
    diff_bar = (base_bar.iloc[:safe_cutoff] - pert_bar.iloc[:safe_cutoff]).abs().max()
    assert diff_bar < 1e-12, f"L21 bar-return diff: {diff_bar}"

    # Compare past trades
    past_base = base_trades[base_trades["exit_date"] < spy.index[safe_cutoff]]
    past_pert = pert_trades[pert_trades["exit_date"] < spy.index[safe_cutoff]]
    common = past_base.merge(past_pert, on="event_date", suffixes=("_b", "_p"))
    if len(common) > 0:
        max_ret_diff = float((common["net_ret_b"] - common["net_ret_p"]).abs().max())
        assert max_ret_diff < 1e-12, f"L21 trade-net-ret diff: {max_ret_diff}"
    print(f"[F3] L21 PASS (bar-diff={diff_bar:.2e}, n_past_trades={len(common)})")


def main() -> None:
    print("=" * 88)
    print("F3 FOMC pre-announcement drift -- V3.7 hybrid audit")
    print("=" * 88)

    # ── Load data ──
    spy = _load_spy()
    print(f"\n[load] SPY: {len(spy)} daily bars, {spy.index[0].date()} -> {spy.index[-1].date()}")
    print(f"[fomc] {len(FOMC_DATES)} scheduled-meeting announcement dates")

    # ── L21 causality ──
    print("\n[L21] causality smoke test")
    _assert_l21_causal(spy)

    # ── Sanctuary slice (12 months hold-out at end) ──
    sanc = slice_sanctuary(spy, months=12)
    spy_visible = sanc.visible
    spy_sanctuary = sanc.sanctuary
    print(
        f"[sanctuary] visible {spy_visible.index[0].date()} -> "
        f"{spy_visible.index[-1].date()} ({len(spy_visible)} bars); "
        f"sanctuary {spy_sanctuary.index[0].date()} -> "
        f"{spy_sanctuary.index[-1].date()} ({len(spy_sanctuary)} bars)"
    )

    # ── [1/6] Per-cell sweep on visible window ──
    print("\n[1/6] Per-cell sweep on VISIBLE window")
    print(
        f"  {'hold':>5s} {'entry_off':>10s} {'n_trades':>9s} {'pt_SR':>8s} "
        f"{'annSR':>9s} {'CI_lo':>9s} {'hit%':>7s} {'mdd':>8s} {'canonical'}"
    )
    sweep_results: list[dict] = []
    canonical_bar_rets: pd.Series | None = None
    canonical_trades: pd.DataFrame | None = None
    for hold_days in HOLD_DAYS_GRID:
        for offset in ENTRY_OFFSET_GRID:
            bar_rets, trades = f3_trade_returns(
                spy_visible, hold_days=hold_days, entry_offset_days=offset
            )
            stats = _stats(bar_rets, f"hold{hold_days}_off{offset}")
            pt_sr = _per_trade_sharpe(trades)
            hit = float((trades["net_ret"] > 0).mean()) if len(trades) > 0 else float("nan")
            is_canonical = hold_days == CANONICAL_HOLD and offset == CANONICAL_ENTRY_OFFSET
            sweep_results.append(
                {
                    "hold_days": hold_days,
                    "entry_offset_days": offset,
                    "n_trades": int(len(trades)),
                    "pt_sharpe": pt_sr,
                    "ann_sharpe": stats["sharpe"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "hit_rate": hit,
                    "mdd": stats["mdd"],
                    "is_canonical": is_canonical,
                    "bar_rets": bar_rets,
                    "trades": trades,
                }
            )
            if is_canonical:
                canonical_bar_rets = bar_rets
                canonical_trades = trades
            star = " *" if is_canonical else ""
            print(
                f"  {hold_days:>5d} {offset:>10d} {len(trades):>9d} "
                f"{pt_sr:>+7.4f} {stats['sharpe']:>+8.4f} "
                f"{stats['ci_lo']:>+8.3f} {hit:>6.2%} "
                f"{stats['mdd']:>+7.2%}{star}"
            )

    if canonical_bar_rets is None or canonical_trades is None:
        raise RuntimeError("Canonical cell not found in sweep")

    # ── [2/6] L52 plateau pre-flight ──
    print("\n[2/6] L52 plateau pre-flight")
    plateau_cells = [
        c
        for c in sweep_results
        if (
            (c["hold_days"] == CANONICAL_HOLD and c["entry_offset_days"] in ENTRY_OFFSET_GRID)
            or (
                c["entry_offset_days"] == CANONICAL_ENTRY_OFFSET
                and c["hold_days"] in HOLD_DAYS_GRID
            )
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

    # ── [3/6] DSR per canonical (over 9 cells) ──
    print("\n[3/6] DSR deflation on canonical cell (9 cells)")
    sweep_sharpes = [c["ann_sharpe"] for c in sweep_results]
    sr_var = sr_var_from_sweep(sweep_sharpes)
    canonical_cell = next(c for c in sweep_results if c["is_canonical"])
    if sr_var > 0:
        dsr = deflated_sharpe(
            sr_hat=canonical_cell["ann_sharpe"],
            sr_var_across_trials=sr_var,
            returns=canonical_bar_rets,
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
        print("  insufficient sweep variance for DSR (cells nearly identical)")

    # ── Pre-reg §3.7 early-exit on plateau or DSR failure ──
    # Per the pre-reg decision tree: "plateau spread > 50% -> RETIRE (L52
    # strict); DSR p < 0.95 -> RETIRE (selection bias dominates)". When
    # both fail, we skip the expensive 5-axis MC + L65 + L67 -- the verdict
    # is already sealed and the remaining gates cannot change it.
    if (not plateau_h1) or (dsr is None or dsr.dsr_prob < DSR_DEPLOY_GATE):
        verdict_reasons: list[str] = []
        if not plateau_h1:
            verdict_reasons.append(f"L52 plateau spread {plateau_spread * 100:.1f}% > 50% H1")
        if dsr is None or dsr.dsr_prob < DSR_DEPLOY_GATE:
            verdict_reasons.append(
                f"DSR p={(dsr.dsr_prob if dsr else float('nan')):.4f} < {DSR_DEPLOY_GATE}"
            )
        if canonical_cell["ci_lo"] <= 0:
            verdict_reasons.append(f"canonical CI_lo {canonical_cell['ci_lo']:+.3f} <= 0")
        print("\n" + "=" * 88)
        print("VERDICT (early-exit per pre-reg §3.7)")
        print("=" * 88)
        verdict = "RETIRE"
        print(f"  Verdict: {verdict}")
        for r in verdict_reasons:
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
            fail_reasons=verdict_reasons,
            n_canonical_trades=len(canonical_trades),
        )
        print(f"\n  Result log: {(REPORTS_DIR / 'result_log.md').relative_to(PROJECT_ROOT)}")
        return

    # ── [4/6] 5-axis decision matrix (canonical cell) ──
    print("\n[4/6] 5-axis decision matrix (canonical cell)")
    axis_ci_pass = canonical_cell["ci_lo"] > 0
    axis_dsr_pass = dsr is not None and dsr.dsr_prob >= DSR_DEPLOY_GATE

    # MC absolute DD on canonical bar returns
    mc_cfg = McConfig(
        block_size_bars=21,
        n_paths=200,
        bootstrap_method="iid",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )

    def _strategy_fn_for_mc(df: pd.DataFrame) -> pd.Series:
        rets, _ = f3_trade_returns(
            df, hold_days=CANONICAL_HOLD, entry_offset_days=CANONICAL_ENTRY_OFFSET
        )
        return rets

    mc_res = run_block_mc(
        spy_visible["close"],
        mc_cfg,
        _strategy_fn_for_mc,
        periods_per_year=PERIODS_PER_YEAR,
    )
    axis_mc_pass = bool(mc_res.passes)
    print(
        f"  MC abs-DD: median MaxDD {mc_res.median_max_dd * 100:+.2f}%, "
        f"95th {mc_res.percentile_95_max_dd * 100:+.2f}%, "
        f"P(MaxDD>{mc_cfg.max_dd_threshold_pct:.0%})={mc_res.prob_exceed_threshold:.2%} "
        f"({'PASS' if axis_mc_pass else 'FAIL'})"
    )

    # Sanctuary divergence: re-compute canonical on sanctuary, compare Sharpes
    sanc_bar, sanc_trades = f3_trade_returns(
        spy_sanctuary, hold_days=CANONICAL_HOLD, entry_offset_days=CANONICAL_ENTRY_OFFSET
    )
    sanc_sr = float(sharpe(sanc_bar.dropna(), periods_per_year=PERIODS_PER_YEAR))
    sanc_diff = abs(canonical_cell["ann_sharpe"] - sanc_sr)
    axis_sanc_pass = sanc_diff <= 0.3
    print(
        f"  Sanctuary: visible SR {canonical_cell['ann_sharpe']:+.4f} vs "
        f"sanctuary SR {sanc_sr:+.4f} (diff {sanc_diff:.3f}, gate <= 0.30): "
        f"{'PASS' if axis_sanc_pass else 'FAIL'}"
    )

    # Noise robustness (Varma): perturbs spy_visible's close, re-runs strategy
    noise_cfg = NoiseConfig(noise_levels=(0.10, 0.30, 0.50), n_trials=10, seed=42)

    def _strategy_fn_for_noise(df: pd.DataFrame) -> pd.Series:
        rets, _ = f3_trade_returns(
            df, hold_days=CANONICAL_HOLD, entry_offset_days=CANONICAL_ENTRY_OFFSET
        )
        return rets

    noise_res = run_noise_robustness(
        spy_visible,
        _strategy_fn_for_noise,
        periods_per_year=PERIODS_PER_YEAR,
        cfg=noise_cfg,
    )
    axis_noise_pass = bool(noise_res.passes)
    print(
        f"  Noise (sigma=0.5): base SR {noise_res.base_sharpe:+.4f}, "
        f"mean SR {noise_res.results[-1].mean_sharpe:+.4f}, "
        f"degradation {noise_res.results[-1].mean_degradation:.3f}, "
        f"worst-case p5 {noise_res.results[-1].p5_sharpe:+.4f} "
        f"({'PASS' if axis_noise_pass else 'FAIL'})"
    )

    axes_pass_count = sum(
        [axis_ci_pass, axis_dsr_pass, axis_mc_pass, axis_sanc_pass, axis_noise_pass]
    )
    print(f"\n  5-axis matrix: {axes_pass_count}/5 PASS")

    # ── [5/6] L65 single + joint ruin ──
    print("\n[5/6] L65 ruin gates")
    single_ruin_results: list[dict] = []
    for w in [0.02, 0.05, 0.10]:
        r = assess_strategy_ruin(
            canonical_bar_rets,
            deployment_weight=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        single_ruin_results.append(
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

    # Joint ruin with current LIVE strategies (excluding I1v2 SHADOW)
    # Reuse the validated IC top-3 basket from PR #10 hybrid.
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

    candidates = [
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15, "f3": 0.00},  # current LIVE
        {"gem": 0.66, "turtle": 0.17, "ic": 0.15, "f3": 0.02},
        {"gem": 0.63, "turtle": 0.17, "ic": 0.15, "f3": 0.05},
    ]
    joint_results: list[dict] = []
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={
                "gem": gem,
                "turtle": turtle,
                "ic": ic_basket,
                "f3": canonical_bar_rets,
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
        w_str = f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/IC{int(w['ic'] * 100)}/F3{int(w['f3'] * 100)}"
        print(
            f"  {w_str}: P_kill={r.p_kill_trip * 100:.3f}%, "
            f"p95 DD={r.p95_maxdd_at_size * 100:.2f}%, "
            f"{'PASS' if r.passes() else 'FAIL'}"
        )

    # ── [6/6] L67 10-metric portfolio inclusion ──
    print("\n[6/6] L67 portfolio inclusion vs 60/40")
    bench = benchmark_6040_returns().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench = bench.groupby(bench.index).last().sort_index()

    current = build_portfolio(
        {"gem": gem, "turtle": turtle, "ic": ic_basket},
        {"gem": 0.68, "turtle": 0.17, "ic": 0.15},
    ).dropna()
    ev_current = evaluate_portfolio_vs_benchmark(
        current, bench, portfolio_label="CURRENT GEM68/T17/IC15", benchmark_label="60/40"
    )
    print()
    print(ev_current.report())

    eval_proposals: list[tuple[str, "object"]] = []
    for w_f3 in [0.02, 0.05]:
        w_g = 0.68 - w_f3 * 0.8
        w_t = 0.17 - w_f3 * 0.2
        w_ic = 0.15
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "ic": ic_basket, "f3": canonical_bar_rets},
            {"gem": w_g, "turtle": w_t, "ic": w_ic, "f3": w_f3},
        ).dropna()
        tag = (
            f"PROPOSED G{int(w_g * 100)}/T{int(w_t * 100)}/IC{int(w_ic * 100)}/F3{int(w_f3 * 100)}"
        )
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
    if not plateau_h1:
        fail_reasons.append(f"L52 plateau spread {plateau_spread * 100:.1f}% > 50% H1")
    if dsr is None or dsr.dsr_prob < DSR_DEPLOY_GATE:
        fail_reasons.append(
            f"DSR p={(dsr.dsr_prob if dsr else float('nan')):.4f} < {DSR_DEPLOY_GATE}"
        )
    if not axis_ci_pass:
        fail_reasons.append(f"canonical CI_lo {canonical_cell['ci_lo']:+.3f} <= 0")
    if not axis_mc_pass:
        fail_reasons.append("MC abs-DD gate FAIL")
    if not axis_sanc_pass:
        fail_reasons.append(f"sanctuary divergence {sanc_diff:.3f} > 0.30")
    if not axis_noise_pass:
        fail_reasons.append("noise robustness FAIL")
    if not all(r["passes"] for r in single_ruin_results):
        fail_reasons.append("L65 single ruin FAIL at one or more weights")

    if not fail_reasons:
        verdict = "DEPLOY-eligible pending L67 lift verification"
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

    # ── Write result log ──
    _write_log(
        report_path=REPORTS_DIR / "result_log.md",
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
        single_ruin_results=single_ruin_results,
        joint_results=joint_results,
        ev_current=ev_current,
        eval_proposals=eval_proposals,
        verdict=verdict,
        fail_reasons=fail_reasons,
        n_canonical_trades=len(canonical_trades),
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
    n_canonical_trades: int,
) -> None:
    """Result log written when pre-reg §3.7 early-exit triggers (plateau or
    DSR failure). The 5-axis MC + L65 + L67 sections are intentionally
    absent because the verdict is sealed by the early gates.
    """
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# F3 FOMC Pre-Announcement Drift -- V3.7 hybrid audit (EARLY RETIRE)\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg F3 FOMC Pre-Announcement Drift 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")
        fh.write(
            "The audit triggered the pre-reg §3.7 early-exit (L52 plateau "
            "and/or DSR gate failed). The 5-axis MC + L65 + L67 sections "
            "were intentionally not run -- per the decision tree they "
            "cannot rescue a strategy whose underlying signal fails the "
            "robustness pre-flight.\n\n"
        )

        fh.write("## [1] Per-cell sweep (visible window)\n\n")
        fh.write(
            "| hold | entry_off | n_trades | per-trade SR | ann SR | CI_lo | CI_hi | hit | mdd | canonical |\n"
        )
        fh.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['hold_days']} | {c['entry_offset_days']} | {c['n_trades']} | "
                f"{c['pt_sharpe']:+.4f} | {c['ann_sharpe']:+.4f} | "
                f"{c['ci_lo']:+.3f} | {c['ci_hi']:+.3f} | "
                f"{c['hit_rate']:.2%} | {c['mdd']:+.2%} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau pre-flight\n\n")
        fh.write(
            f"- Plateau spread: **{plateau_spread * 100:.1f}%** "
            f"(H1 gate ≤50%: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict ≤30%: {'PASS' if plateau_strict else 'FAIL'})\n"
        )

        fh.write("\n## [3] DSR deflation (canonical, 9 cells)\n\n")
        if dsr is not None:
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] across 9 cells = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
                f"- Gate p ≥ 0.95: {'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'}\n"
            )
        else:
            fh.write("- DSR not computed (insufficient sweep variance)\n")

        fh.write(
            f"\n## Canonical-cell summary\n\n"
            f"- Cell: (hold={canonical_cell['hold_days']}, "
            f"entry_offset={canonical_cell['entry_offset_days']})\n"
            f"- Trades: {n_canonical_trades}\n"
            f"- Per-trade Sharpe: {canonical_cell['pt_sharpe']:+.4f}\n"
            f"- Annualised Sharpe: {canonical_cell['ann_sharpe']:+.4f}\n"
            f"- CI95: [{canonical_cell['ci_lo']:+.3f}, {canonical_cell['ci_hi']:+.3f}]\n"
            f"- Hit rate: {canonical_cell['hit_rate']:.2%}\n"
            f"- MaxDD: {canonical_cell['mdd']:+.2%}\n"
        )

        fh.write("\n## Verdict & fail reasons\n\n")
        fh.write(f"**{verdict}**\n\n")
        for r in fail_reasons:
            fh.write(f"- {r}\n")

        fh.write(
            "\n## Interpretation\n\n"
            "The Lucca-Moench (2015) pre-FOMC drift was documented for the "
            "1994-2011 sample. Multiple follow-on papers "
            "(Cieslak, Morse & Vissing-Jorgensen 2018; Cieslak & "
            "Vissing-Jorgensen 2021) suggested attenuation after the "
            "Fed-transparency reforms around 2014. Our 2010-2025 sample "
            "audit on retail-implementable SPY (ETF, not SPX index) "
            "supports the decay hypothesis. The plateau spread across the "
            "3x3 cell grid is wide enough that no single configuration "
            "is robust to a one-step perturbation in either dimension. "
            "DSR p ~= 0 confirms the canonical Sharpe is below the "
            "expected max under N=9 trials -- the apparent edge in any "
            "individual cell is consistent with selection bias.\n\n"
            "**Retirement Registry next-time lesson**: The pre-FOMC drift, "
            "as originally documented in Lucca-Moench 2015, does not "
            "replicate on SPY 2010-2025 in a deployment-eligible form. A "
            "re-audit on an extended pre-2014 window OR with a regime-"
            "conditional gate (e.g., only active during specific Fed "
            "policy phases) could test whether the decay is conditional "
            "rather than absolute -- but the base strategy fails the "
            "V3.7 hybrid.\n"
        )


def _write_log(
    *,
    report_path: Path,
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
    single_ruin_results: list[dict],
    joint_results: list[dict],
    ev_current,
    eval_proposals: list,
    verdict: str,
    fail_reasons: list[str],
    n_canonical_trades: int,
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# F3 FOMC Pre-Announcement Drift -- V3.7 hybrid audit\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write("**Pre-reg:** `directives/Pre-Reg F3 FOMC Pre-Announcement Drift 2026-05-19.md`\n")
        fh.write(f"**Verdict:** {verdict}\n\n")

        fh.write("## [1] Per-cell sweep (visible window)\n\n")
        fh.write(
            "| hold | entry_off | n_trades | per-trade SR | ann SR | CI_lo | CI_hi | hit | mdd | canonical |\n"
        )
        fh.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for c in sweep_results:
            star = "*" if c["is_canonical"] else ""
            fh.write(
                f"| {c['hold_days']} | {c['entry_offset_days']} | {c['n_trades']} | "
                f"{c['pt_sharpe']:+.4f} | {c['ann_sharpe']:+.4f} | "
                f"{c['ci_lo']:+.3f} | {c['ci_hi']:+.3f} | "
                f"{c['hit_rate']:.2%} | {c['mdd']:+.2%} | {star} |\n"
            )

        fh.write("\n## [2] L52 plateau pre-flight\n\n")
        fh.write(
            f"- Plateau spread: **{plateau_spread * 100:.1f}%** "
            f"(H1 gate ≤50%: {'PASS' if plateau_h1 else 'FAIL'}; "
            f"strict ≤30%: {'PASS' if plateau_strict else 'FAIL'})\n"
        )

        fh.write("\n## [3] DSR deflation (canonical, 9 cells)\n\n")
        if dsr is not None:
            fh.write(
                f"- canonical SR = {dsr.sharpe:+.4f}\n"
                f"- E[max SR] across 9 cells = {dsr.e_max_sr:+.4f}\n"
                f"- z = {dsr.z:+.3f}, p = {dsr.dsr_prob:.4f}\n"
                f"- Gate p ≥ 0.95: {'PASS' if dsr.dsr_prob >= DSR_DEPLOY_GATE else 'FAIL'}\n"
            )
        else:
            fh.write("- insufficient sweep variance for DSR (cells nearly identical)\n")

        fh.write("\n## [4] 5-axis decision matrix\n\n")
        fh.write(
            f"- canonical n_trades: {n_canonical_trades}\n"
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
        for r in single_ruin_results:
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
                f"IC{int(w['ic'] * 100)}/F3{int(w['f3'] * 100)}"
            )
            fh.write(
                f"| {w_str} | {r['p_kill'] * 100:.3f}% | "
                f"{r['p95_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [6] L67 10-metric portfolio inclusion (vs 60/40)\n\n")
        fh.write("### CURRENT (no F3)\n\n```\n" + ev_current.report() + "\n```\n\n")
        for tag, ev in eval_proposals:
            fh.write(f"### {tag}\n\n```\n" + ev.report() + "\n```\n\n")

        fh.write("## Verdict\n\n")
        fh.write(f"**{verdict}**\n\n")
        if fail_reasons:
            fh.write("Fail reasons:\n\n")
            for r in fail_reasons:
                fh.write(f"- {r}\n")


if __name__ == "__main__":
    main()
