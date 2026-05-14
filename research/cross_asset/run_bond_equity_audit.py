"""run_bond_equity_audit.py -- Track 1 audit re-run for bond-equity champions.

Pre-registered in:
    directives/Bond-Equity Audit DSR-Sanctuary 2026-05-14.md

Adds three new gates on top of the existing audit-corrected
`run_bond_wfo` (which already uses shared sharpe + bootstrap CI):

    1. DSR-adjusted prob across the cell sweep (Bailey & López de Prado 2014)
    2. Underlying-resampled MC with 50-bar shared block indices (audit A6)
    3. Sanctuary hold-out: last 12 months trimmed before WFO, separate
       one-shot sanctuary pass on the trimmed window

Usage::

    python research/cross_asset/run_bond_equity_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo  # noqa: E402
from research.samir_stack.run_phase5_joint_sweep import deflated_sharpe_prob  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, sharpe  # noqa: E402

SANCTUARY_MONTHS = 12
MC_PATHS = 200
MC_BLOCK_SIZE = 50


@dataclass(frozen=True)
class Cell:
    bond: str
    target: str
    lookback: int
    hold: int
    threshold: float
    note: str

    @property
    def name(self) -> str:
        return f"{self.bond}->{self.target}_LB{self.lookback}_H{self.hold}_T{self.threshold:.2f}"


CELLS: list[Cell] = [
    # Champions under audit (§1.1 of directive)
    Cell("IHYU", "CSPX", 10, 20, 0.50, "Champion: +0.68 OOS"),
    Cell("IHYG", "VUSD", 10, 20, 0.50, "Champion: +1.16 / CI_lo +0.47"),
    Cell("IHYG", "EMIM", 10, 20, 0.50, "Champion: +0.97 / CI_lo +0.23"),
    # Reference baselines (already audit-corrected; sanity check that gates
    # don't artificially fail audited cells)
    Cell("TLT", "QQQ", 10, 10, 0.50, "Reference: +0.895 corrected"),
    Cell("IEF", "GLD", 60, 20, 0.00, "Reference: +0.721 corrected"),
    Cell("HYG", "IWB", 10, 10, 0.50, "Reference: +0.895 revalidated"),
]


def _slice_sanctuary(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Return (visible, sanctuary, sanctuary_start)."""
    if len(close) == 0:
        return close, close, pd.Timestamp("1970-01-01", tz="UTC")
    end = close.index[-1]
    sanc_start = end - pd.DateOffset(months=SANCTUARY_MONTHS)
    visible = close[close.index < sanc_start]
    sanctuary = close[close.index >= sanc_start]
    return visible, sanctuary, sanc_start


def _run_wfo_visible(cell: Cell) -> dict:
    """Run the existing run_bond_wfo on pre-sanctuary data."""
    bond = load_daily(cell.bond)
    target = load_daily(cell.target)
    bond_v, _, sanc_start = _slice_sanctuary(bond)
    target_v, _, _ = _slice_sanctuary(target)
    return run_bond_wfo(
        bond_close=bond_v,
        target_close=target_v,
        lookback=cell.lookback,
        hold_days=cell.hold,
        threshold=cell.threshold,
    ), sanc_start


def _run_sanctuary_pass(cell: Cell) -> dict:
    """Run a separate WFO on the trailing 12 months only.

    Sanctuary window is small (~252 daily bars) so the WFO degenerates
    to essentially a single one-shot test. We compute per-bar strategy
    returns directly to avoid the IS/OOS fold split which doesn't fit
    in 252 bars.
    """
    bond_full = load_daily(cell.bond)
    target_full = load_daily(cell.target)
    _, bond_s, sanc_start = _slice_sanctuary(bond_full)
    _, target_s, _ = _slice_sanctuary(target_full)
    common = bond_s.index.intersection(target_s.index)
    if len(common) < 60:
        return {
            "stage": "sanctuary",
            "n_bars": int(len(common)),
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "reason": "insufficient_sanctuary_data",
        }
    target_s = target_s.reindex(common)
    # Bond momentum needs `lookback` bars BEFORE the sanctuary window for
    # the diff to evaluate inside the sanctuary -- compute on the full
    # bond series (with pre-sanctuary history), then restrict to the
    # sanctuary index.
    mom_full = np.log(bond_full / bond_full.shift(cell.lookback)).dropna()
    mom_s = mom_full.reindex(common)
    # Use a z-score over the prior 252 (causal) for the sanctuary
    z = (mom_s - mom_s.rolling(252, min_periods=20).mean()) / mom_s.rolling(252, min_periods=20).std()
    pos = (z > cell.threshold).astype(float)
    # Hold for hold_days minimum (simple)
    pos_held = pos.rolling(cell.hold, min_periods=1).max()
    target_ret = target_s.pct_change().fillna(0.0)
    strat = pos_held.shift(1).fillna(0.0) * target_ret
    # Apply cost on transitions
    transitions = (pos_held != pos_held.shift(1).fillna(0.0)).astype(float)
    cost_per_transition = 5.0 / 10_000  # 5 bps spread
    strat = strat - transitions * cost_per_transition
    strat = strat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if len(strat) < 20:
        return {
            "stage": "sanctuary",
            "n_bars": int(len(strat)),
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "reason": "insufficient_after_filter",
        }
    sh = sharpe(strat, periods_per_year=BARS_PER_YEAR["D"])
    eq = (1.0 + strat).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {
        "stage": "sanctuary",
        "n_bars": int(len(strat)),
        "sharpe": round(sh, 4),
        "max_drawdown": round(mdd, 4),
        "sanctuary_start": str(sanc_start.date()),
        "reason": "",
    }


def _underlying_resampled_mc(
    cell: Cell, *, n_paths: int = MC_PATHS, block_size: int = MC_BLOCK_SIZE, seed: int = 42,
) -> dict:
    """Bootstrap (bond, target) log returns with SHARED block indices —
    preserves cross-asset correlation. cumprod to rebuild synthetic prices.
    Run the strategy on each path; record MaxDD + Sharpe distribution."""
    bond = load_daily(cell.bond)
    target = load_daily(cell.target)
    bond_v, _, _ = _slice_sanctuary(bond)
    target_v, _, _ = _slice_sanctuary(target)
    common = bond_v.index.intersection(target_v.index)
    bond_v = bond_v.reindex(common)
    target_v = target_v.reindex(common)
    bond_logret = np.log(bond_v).diff().dropna()
    target_logret = np.log(target_v).diff().dropna()
    common_ret = bond_logret.index.intersection(target_logret.index)
    bond_logret = bond_logret.reindex(common_ret).values
    target_logret = target_logret.reindex(common_ret).values
    n = len(bond_logret)
    if n < 200:
        return {
            "stage": "mc", "n_paths": 0, "reason": "insufficient_underlying_data",
            "median_sharpe": 0.0, "median_maxdd": 0.0,
            "p_maxdd_gt_25pct": 1.0,
        }
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    mc_sharpes: list[float] = []
    mc_maxdds: list[float] = []
    for _ in range(n_paths):
        block_starts = rng.integers(0, n - block_size, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        idx = idx[:n]
        # SHARED indices for both legs -- preserves correlation
        synth_bond_ret = bond_logret[idx]
        synth_target_ret = target_logret[idx]
        synth_bond_px = bond_v.iloc[0] * np.exp(np.cumsum(synth_bond_ret))
        synth_target_px = target_v.iloc[0] * np.exp(np.cumsum(synth_target_ret))
        # Recompute strategy on synthetic paths
        synth_bond = pd.Series(synth_bond_px, index=common_ret[: len(synth_bond_px)])
        synth_target = pd.Series(synth_target_px, index=common_ret[: len(synth_target_px)])
        try:
            r = run_bond_wfo(
                bond_close=synth_bond,
                target_close=synth_target,
                lookback=cell.lookback,
                hold_days=cell.hold,
                threshold=cell.threshold,
            )
            if "error" in r:
                continue
            mc_sharpes.append(float(r["stitched_sharpe"]))
            mc_maxdds.append(float(r["stitched_dd_pct"]) / 100.0)
        except Exception:
            continue
    if not mc_sharpes:
        return {"stage": "mc", "n_paths": 0, "reason": "no_paths_succeeded",
                "median_sharpe": 0.0, "median_maxdd": 0.0, "p_maxdd_gt_25pct": 1.0}
    n_ok = len(mc_sharpes)
    p_mdd25 = sum(1 for d in mc_maxdds if d < -0.25) / n_ok
    return {
        "stage": "mc", "n_paths": n_ok,
        "median_sharpe": round(float(np.median(mc_sharpes)), 4),
        "median_maxdd": round(float(np.median(mc_maxdds)), 4),
        "p5_sharpe": round(float(np.quantile(mc_sharpes, 0.05)), 4),
        "p95_sharpe": round(float(np.quantile(mc_sharpes, 0.95)), 4),
        "p_maxdd_gt_25pct": round(p_mdd25, 4),
        "reason": "",
    }


def _apply_dsr(rows: list[dict]) -> list[dict]:
    """DSR across the per-cell sweep. Skew/kurt default to 0/3 (normal)."""
    sharpes = np.array([r.get("sharpe", 0.0) for r in rows], dtype=float)
    if len(sharpes) < 2:
        for r in rows:
            r["dsr_prob"] = 1.0
        return rows
    sr_var = float(np.var(sharpes, ddof=1))
    for r in rows:
        T = max(int(r.get("n_oos_bars", 252)), 30)
        skew, kurt = 0.0, 3.0
        r["dsr_prob"] = round(float(deflated_sharpe_prob(r["sharpe"], sr_var, skew, kurt, T, len(rows))), 4)
    return rows


def _decision(row: dict) -> tuple[str, str]:
    """Apply directive §1.3 decision matrix for one cell."""
    dsr_ok = row.get("dsr_prob", 0.0) >= 0.95
    mc_ok = row.get("p_maxdd_gt_25pct", 1.0) < 0.05
    sanc_ok = row.get("sanctuary_sharpe", 0.0) >= 0.0
    ci_lo_ok = row.get("sharpe_ci_95_lo", -1.0) > 0.0
    new_gates_ok = dsr_ok and mc_ok and sanc_ok
    if new_gates_ok and ci_lo_ok:
        return "DEPLOY_ELIGIBLE", "All gates pass. Live config audit-confirmed."
    if new_gates_ok and not ci_lo_ok:
        return "CONDITIONAL", "Gates OK but CI_lo <= 0. Status quo + watchpoint."
    if not dsr_ok and ci_lo_ok:
        return "SUSPECT_SWEEP", f"DSR prob {row.get('dsr_prob', 0):.3f} < 0.95 -- Sharpe may be sweep-cherry-picked."
    if not mc_ok and ci_lo_ok:
        return "RISK_UPGRADE", f"MC P(MaxDD>25%) = {row.get('p_maxdd_gt_25pct', 1):.3f} >= 5%. Position-size cap required."
    if not sanc_ok and ci_lo_ok:
        return "REGIME_WATCH", f"Sanctuary Sharpe {row.get('sanctuary_sharpe', 0):.3f} < 0. Recent-period underperformance."
    return "RETIRE", "Multiple gate failures + CI_lo <= 0. Open config-change PR to retire."


def main() -> None:
    print("=" * 100)
    print("  BOND-EQUITY AUDIT -- DSR + SANCTUARY + UNDERLYING-RESAMPLED MC")
    print(f"  Sanctuary: last {SANCTUARY_MONTHS} months held out")
    print(f"  MC: {MC_PATHS} paths × {MC_BLOCK_SIZE}-bar shared blocks")
    print(f"  N = {len(CELLS)} cells; DSR null-max ~ sqrt(2 ln {len(CELLS)}) ~ "
          f"{np.sqrt(2 * np.log(len(CELLS))):.2f}")
    print("=" * 100)

    rows: list[dict] = []
    for cell in CELLS:
        print(f"\n  {cell.name} -- {cell.note}")
        try:
            wfo, sanc_start = _run_wfo_visible(cell)
        except Exception as exc:
            print(f"    SKIP -- {exc}")
            continue
        if "error" in wfo:
            print(f"    SKIP -- {wfo['error']}")
            continue
        print(
            f"    WFO (pre-sanctuary): Sharpe={wfo['stitched_sharpe']}, "
            f"CI=[{wfo['sharpe_ci_95_lo']}, {wfo['sharpe_ci_95_hi']}], "
            f"DD={wfo['stitched_dd_pct']}%, "
            f"folds={wfo['n_folds']}, pos%={wfo['pct_positive']*100:.0f}%, "
            f"trades={wfo['total_trades']}"
        )
        print(f"    Sanctuary pass ({cell.bond}/{cell.target} last 12mo)...")
        sanc = _run_sanctuary_pass(cell)
        print(f"      Sharpe={sanc['sharpe']}, DD={sanc.get('max_drawdown', 0)*100:.2f}%, n_bars={sanc['n_bars']}")
        print(f"    Underlying-resampled MC (200 paths, 50-bar shared blocks)...")
        mc = _underlying_resampled_mc(cell)
        print(f"      median Sharpe={mc.get('median_sharpe', 0)}, P(MaxDD>25%)={mc.get('p_maxdd_gt_25pct', 1)}, paths={mc.get('n_paths', 0)}")
        rows.append({
            "cell": cell.name,
            "bond": cell.bond, "target": cell.target,
            "lookback": cell.lookback, "hold": cell.hold, "threshold": cell.threshold,
            "note": cell.note,
            "sanctuary_start": str(sanc_start.date()),
            "sharpe": wfo["stitched_sharpe"],
            "sharpe_ci_95_lo": wfo["sharpe_ci_95_lo"],
            "sharpe_ci_95_hi": wfo["sharpe_ci_95_hi"],
            "stitched_dd_pct": wfo["stitched_dd_pct"],
            "n_folds": wfo["n_folds"],
            "pct_positive": wfo["pct_positive"],
            "total_trades": wfo["total_trades"],
            "n_oos_bars": int(len(wfo.get("stitched_returns", []))),
            "sanctuary_sharpe": sanc["sharpe"],
            "sanctuary_dd": sanc.get("max_drawdown", 0.0),
            "sanctuary_n_bars": sanc["n_bars"],
            "mc_median_sharpe": mc.get("median_sharpe", 0.0),
            "mc_median_maxdd": mc.get("median_maxdd", 0.0),
            "mc_p_maxdd_gt_25pct": mc.get("p_maxdd_gt_25pct", 1.0),
            "mc_n_paths": mc.get("n_paths", 0),
        })

    if not rows:
        print("\n  No cells produced results.")
        return
    rows = _apply_dsr(rows)
    for r in rows:
        verdict, rationale = _decision(r)
        r["verdict"] = verdict
        r["rationale"] = rationale

    print()
    print("=" * 100)
    print("  AUDIT SUMMARY")
    print("=" * 100)
    df = pd.DataFrame(rows)
    cols = ["cell", "sharpe", "sharpe_ci_95_lo", "stitched_dd_pct",
            "sanctuary_sharpe", "mc_p_maxdd_gt_25pct", "dsr_prob", "verdict"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print(df[cols].to_string(index=False))
    print()
    print("=" * 100)
    print("  PER-CELL RATIONALE")
    print("=" * 100)
    for r in rows:
        print(f"\n  {r['cell']} -- {r['verdict']}")
        print(f"    {r['rationale']}")

    # Save
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / ".tmp" / "reports" / "bond_equity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{stamp}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
