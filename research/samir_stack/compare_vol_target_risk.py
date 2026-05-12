"""Compare risk distributions of two vol-target levels via bootstrap.

Walks through 6% vs 12% target_vol on the L=3 10/90 champion. Computes
the bootstrap drawdown distribution and worst-path tail to make the
single-MaxDD-observation comparison honest.

Usage:
    uv run python research/samir_stack/compare_vol_target_risk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_overlay_sweep import run_with_overlays  # noqa: E402
from research.samir_stack.run_risk_of_ruin import (  # noqa: E402
    _ruin_probability,
)


def _max_drawdown(rets: np.ndarray) -> float:
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def _wfo_stitched(df) -> np.ndarray:
    n = len(df)
    is_days, oos_days, step = 504, 252, 252
    if n < is_days + oos_days:
        return np.array([])
    rets = df["ret_strategy"].to_numpy()
    chunks = []
    oos_start = is_days
    while oos_start + oos_days <= n:
        chunks.append(rets[oos_start : oos_start + oos_days])
        oos_start += step
    return np.concatenate(chunks)


def main() -> int:
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")
    common = (
        data["spy"]
        .index.intersection(efa.index)
        .intersection(data["tlt"].index)
        .intersection(data["hyg"].index)
        .intersection(data["ief"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa_a = efa.reindex(common)
    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    print("Building strategies...", flush=True)
    targets = [0.06, 0.08, 0.12]
    stitched_by_target: dict = {}
    for tv in targets:
        df = run_with_overlays(
            spy,
            efa_a,
            ief,
            hyg,
            tlt,
            samir_score,
            panel,
            use_capitulation=False,
            use_vol_target=True,
            vol_target_annual=tv,
        )
        stitched_by_target[tv] = _wfo_stitched(df)
        print(f"  target_vol={tv:.0%}: {len(stitched_by_target[tv])} OOS bars", flush=True)

    n_yrs = len(next(iter(stitched_by_target.values()))) / 252.0

    def _stats(s):
        eq = np.cumprod(1.0 + s)
        return {
            "stitched_cagr": float(eq[-1] ** (1.0 / n_yrs) - 1.0),
            "stitched_vol": float(np.std(s) * np.sqrt(252)),
            "sharpe": float(np.mean(s) / np.std(s) * np.sqrt(252)),
            "max_dd": _max_drawdown(s),
            "worst_day": float(s.min()),
            "worst_21d": float(
                np.array([(1.0 + s[i : i + 21]).prod() - 1.0 for i in range(len(s) - 21)]).min()
            ),
        }

    stats_by_target = {tv: _stats(s) for tv, s in stitched_by_target.items()}

    print()
    print("=" * 100)
    print("EMPIRICAL (single observed path through 16 OOS years)")
    print("=" * 100)
    header = f"{'metric':<32}  " + "  ".join(f"{tv:.0%} target".rjust(12) for tv in targets)
    print(header)
    print("-" * len(header))
    rows = [
        ("Annualised return", "stitched_cagr", True),
        ("Realised vol", "stitched_vol", True),
        ("Sharpe", "sharpe", False),
        ("Worst single observed DD", "max_dd", True),
        ("Worst single day", "worst_day", True),
        ("Worst 21-day", "worst_21d", True),
    ]
    for label, key, as_pct in rows:
        cells = []
        for tv in targets:
            v = stats_by_target[tv][key]
            cells.append(f"{v * 100:>11.3f}%" if as_pct else f"{v:>12.3f}")
        print(f"{label:<32}  " + "  ".join(cells))

    print()
    print("=" * 100)
    print("BOOTSTRAP DRAWDOWN PROJECTIONS - distribution of worst MaxDD over horizon")
    print("(5-day block bootstrap, 5,000 paths)")
    print("=" * 100)
    horizons = [1.0, 5.0, 10.0, 20.0]
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    for tv in targets:
        s = stitched_by_target[tv]
        print()
        print(f"--- target_vol = {tv:.0%} ---")
        print(f"{'horizon_yrs':>12}  " + "  ".join(f"P(DD>{int(t * 100):>2}%)" for t in thresholds))
        for h in horizons:
            row_parts = [f"{h:>12.0f}"]
            for thr in thresholds:
                res = _ruin_probability(s, horizon_years=h, drawdown_threshold=thr)
                row_parts.append(f"{res['p_breach']:>11.4f}")
            print("  ".join(row_parts))

    print()
    print("=" * 100)
    print("MAX-DD percentiles (10-year horizon, 5,000 bootstrap paths)")
    print("=" * 100)
    for tv in targets:
        s = stitched_by_target[tv]
        res = _ruin_probability(s, horizon_years=10.0, drawdown_threshold=0.05)
        print(
            f"  target_vol={tv:.0%}:  median={res['median_max_dd'] * 100:>6.2f}%  "
            f"5th-pct={res['p95_max_dd'] * 100:>6.2f}%  "
            f"1st-pct={res['p99_max_dd'] * 100:>6.2f}%  "
            f"worst-of-5000={res['worst_max_dd'] * 100:>6.2f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
