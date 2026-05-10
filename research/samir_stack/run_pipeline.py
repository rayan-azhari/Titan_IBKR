"""End-to-end Samir-Stack pipeline orchestrator.

Runs every phase and writes a comprehensive report to .tmp/reports/.
This is the canonical "reproduce all results" script.

Phases:
    1. Load data + build indicator panel
    2. IC validation
    3. Build regime score
    4. Backtest across L_max ∈ {1, 2, 3}
    5. Run benchmarks (SPY, 60/40, Faber, HFEA, Samir-pure)
    6. Stress decomposition
    7. Monte Carlo RoR
    8. Walk-forward + sanctuary

Usage:
    uv run python research/samir_stack/run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.samir_stack.benchmarks import _summarize, all_benchmarks
from research.samir_stack.data_loader import load_panel
from research.samir_stack.ic_validation import validate_indicators
from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.monte_carlo import monte_carlo_ror
from research.samir_stack.regime_score import (
    correlation_matrix,
    regime_score_equal,
)
from research.samir_stack.strategy import StrategyConfig, run_strategy, summarize
from research.samir_stack.stress import crisis_pivot, run_stress_table
from research.samir_stack.wfo import run_wfo

REPORT_DIR = Path(".tmp/reports/samir_stack")


def main(
    *,
    start: str = "2003-04-01",
    end: str = "2026-04-02",
    leverage_grid: tuple[float, ...] = (1.0, 2.0, 3.0),
    n_mc_paths: int = 5_000,
    mc_block_len: int = 63,
    n_wfo_folds: int = 5,
    sanctuary_years: float = 1.0,
) -> dict[str, pd.DataFrame]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {REPORT_DIR}")

    # ── Phase 1: data + panel ─────────────────────────────────────────
    print("\n[1/8] Loading data...")
    data = load_panel(start=start, end=end)
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data.get("hyg"),
        ief_close=data.get("ief"),
        tlt_close=data.get("tlt"),
    )
    print(
        f"     SPY bars: {len(data['spy'])}, range: {data['spy'].index.min().date()} → {data['spy'].index.max().date()}"
    )
    panel.to_parquet(REPORT_DIR / "indicator_panel.parquet")

    # ── Phase 2: IC validation ────────────────────────────────────────
    print("\n[2/8] IC validation on regime indicators...")
    ic_df = validate_indicators(panel, data["spy"], horizon=21)
    ic_df.to_csv(REPORT_DIR / "ic_validation.csv", index=False)
    print(
        ic_df[["indicator", "ic", "vol_ratio_h_b", "crash_hit_rate", "verdict"]].to_string(
            index=False
        )
    )

    # ── Phase 3: regime score ─────────────────────────────────────────
    print("\n[3/8] Building regime score (equal-weight)...")
    score = regime_score_equal(panel)
    pd.DataFrame({"regime_score": score}).to_parquet(REPORT_DIR / "regime_score.parquet")
    print(
        f"     Score: mean={score.mean():.3f}, frac<0.3={float((score < 0.3).mean()):.3f}, frac>0.7={float((score > 0.7).mean()):.3f}"
    )
    corr = correlation_matrix(panel)
    corr.to_csv(REPORT_DIR / "indicator_correlation.csv")

    # ── Phase 4: backtest grid ────────────────────────────────────────
    print("\n[4/8] Strategy backtest across L_max grid...")
    strategy_results = {}
    strategy_summaries = []
    for L in leverage_grid:
        cfg = StrategyConfig(L_max=L)
        res = run_strategy(data["spy"], score, cfg)
        strategy_results[L] = res
        s = summarize(res)
        s["L_max"] = L
        strategy_summaries.append(s)
        res.to_parquet(REPORT_DIR / f"strategy_L_max_{int(L)}.parquet")
    strat_summary_df = pd.DataFrame(strategy_summaries)
    strat_summary_df.to_csv(REPORT_DIR / "strategy_summary.csv", index=False)
    print(strat_summary_df.to_string(index=False))

    # ── Phase 5: benchmarks ───────────────────────────────────────────
    print("\n[5/8] Benchmarks...")
    bench_rets, bench_df = all_benchmarks(data["spy"], data["ief"], data["tlt"], score)
    # Add Samir-Stack rows to bench_df for unified comparison
    for L, res in strategy_results.items():
        s = _summarize(res["ret_strategy"], f"Samir-Stack L_max={int(L)}x")
        bench_df = pd.concat([bench_df, pd.DataFrame([s])], ignore_index=True)
    bench_df.to_csv(REPORT_DIR / "all_strategies_summary.csv", index=False)
    print(bench_df.to_string(index=False))

    # ── Phase 6: stress decomposition ─────────────────────────────────
    print("\n[6/8] Stress decomposition...")
    named_for_stress = {
        "spy": bench_rets["spy"],
        "60_40": bench_rets["60_40"],
        "faber": bench_rets["faber_200"],
        "hfea": bench_rets["hfea"],
        "samir_pure": bench_rets["samir_pure"],
        "samir_stack_3x": strategy_results[3.0]["ret_strategy"]
        if 3.0 in strategy_results
        else None,
    }
    named_for_stress = {k: v for k, v in named_for_stress.items() if v is not None}
    stress_df = run_stress_table(named_for_stress)
    stress_df.to_csv(REPORT_DIR / "stress_scenarios.csv", index=False)
    pv_dd = crisis_pivot(stress_df, "max_dd")
    print((pv_dd * 100).round(2).to_string())

    # ── Phase 7: Monte Carlo RoR ──────────────────────────────────────
    print(f"\n[7/8] Monte Carlo RoR ({n_mc_paths} paths, block={mc_block_len}d)...")
    mc_rows = []
    candidates = {
        "spy": bench_rets["spy"],
        "60_40": bench_rets["60_40"],
        "faber": bench_rets["faber_200"],
        "hfea": bench_rets["hfea"],
        "samir_pure": bench_rets["samir_pure"],
    }
    for L, res in strategy_results.items():
        candidates[f"samir_lmax_{int(L)}"] = res["ret_strategy"]

    for name, rets in candidates.items():
        mc = monte_carlo_ror(rets, n_paths=n_mc_paths, mean_block_len=mc_block_len, seed=42)
        mc["strategy"] = name
        mc_rows.append(mc)
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(REPORT_DIR / "monte_carlo_ror.csv", index=False)
    cols_to_show = [
        "strategy",
        "cagr_median",
        "cagr_p05",
        "maxdd_median",
        "maxdd_p05",
        "prob_maxdd_gt_25pct",
        "prob_maxdd_gt_35pct",
        "prob_maxdd_gt_50pct",
    ]
    print(mc_df[cols_to_show].to_string(index=False))

    # ── Phase 8: WFO + sanctuary ──────────────────────────────────────
    print(f"\n[8/8] Walk-forward ({n_wfo_folds} folds + {sanctuary_years}-yr sanctuary)...")
    wfo_results = {}
    for L in leverage_grid:
        cfg = StrategyConfig(L_max=L)
        folds, sanctuary = run_wfo(
            data,
            cfg,
            n_folds=n_wfo_folds,
            sanctuary_years=sanctuary_years,
        )
        wfo_results[L] = (folds, sanctuary)
        folds.to_csv(REPORT_DIR / f"wfo_folds_L_max_{int(L)}.csv", index=False)
        with open(REPORT_DIR / f"wfo_sanctuary_L_max_{int(L)}.txt", "w") as f:
            for k, v in sanctuary.items():
                f.write(f"{k}: {v}\n")
        print(
            f"  L_max={int(L)}x: mean fold CAGR={folds['cagr'].mean() * 100:.2f}%, "
            f"sanctuary CAGR={sanctuary['cagr'] * 100:.2f}%, "
            f"sanctuary MaxDD={sanctuary['max_dd'] * 100:.2f}%, "
            f"folds positive: {int((folds['cagr'] > 0).sum())}/{len(folds)}"
        )

    print(f"\nAll outputs → {REPORT_DIR}")
    return {
        "ic_validation": ic_df,
        "strategy_summary": strat_summary_df,
        "all_strategies": bench_df,
        "stress": stress_df,
        "monte_carlo": mc_df,
    }


if __name__ == "__main__":
    main()
