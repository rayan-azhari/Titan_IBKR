"""Sensitivity analysis: how do regime / DD thresholds affect RoR?

Sweeps over four conservatism "presets" — Aggressive, Moderate,
Conservative, Very Conservative — and reports the impact on:
    - CAGR (single-path realisation 2003-2026)
    - MaxDD (single path)
    - Monte Carlo P(MaxDD > 25%, 35%, 50%)
    - Annual trade count
    - Frac in cash

Each preset varies four parameters jointly:
    * tier_thresholds (entry to 1x / 2x / 3x)
    * dd_throttle (DD level for leverage cap)
    * dd_kill (DD level for full cash)
    * dd_re_entry_score (regime score required to exit DD-state)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.samir_stack.data_loader import load_panel
from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.monte_carlo import monte_carlo_ror
from research.samir_stack.regime_score import regime_score_equal
from research.samir_stack.strategy import StrategyConfig, run_strategy, summarize


@dataclass
class Preset:
    """Named threshold preset."""

    name: str
    tier_thresholds: tuple[float, float, float]
    dd_throttle: float
    dd_kill: float
    dd_re_entry_score: float
    re_entry_quiet_bars: int = 20


# Increasing conservatism left → right
PRESETS = [
    Preset("aggressive_orig", (0.30, 0.50, 0.75), 0.12, 0.18, 0.70, 20),
    Preset("moderate", (0.35, 0.55, 0.80), 0.10, 0.15, 0.72, 25),
    Preset("conservative", (0.40, 0.60, 0.85), 0.08, 0.12, 0.75, 30),
    Preset("very_conservative", (0.50, 0.65, 0.90), 0.06, 0.10, 0.80, 40),
]


def run_threshold_sweep(
    L_max: float = 3.0,
    *,
    n_mc_paths: int = 3_000,
    mc_block_len: int = 63,
) -> pd.DataFrame:
    """Run every preset for a given L_max and return summary DataFrame."""
    data = load_panel(start="2003-04-01", end="2026-04-02")
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data["hyg"],
        ief_close=data["ief"],
        tlt_close=data["tlt"],
    )
    score = regime_score_equal(panel)

    rows = []
    for preset in PRESETS:
        cfg = StrategyConfig(
            L_max=L_max,
            tier_thresholds=preset.tier_thresholds,
            dd_throttle=preset.dd_throttle,
            dd_kill=preset.dd_kill,
            dd_re_entry_score=preset.dd_re_entry_score,
            re_entry_quiet_bars=preset.re_entry_quiet_bars,
        )
        res = run_strategy(data["spy"], score, cfg)
        s = summarize(res)
        mc = monte_carlo_ror(
            res["ret_strategy"], n_paths=n_mc_paths, mean_block_len=mc_block_len, seed=42
        )
        rows.append(
            {
                "preset": preset.name,
                "L_max": L_max,
                "tier_thresholds": str(preset.tier_thresholds),
                "dd_throttle": preset.dd_throttle,
                "dd_kill": preset.dd_kill,
                "dd_re_entry": preset.dd_re_entry_score,
                "single_cagr": s["cagr"],
                "single_maxdd": s["max_dd"],
                "single_calmar": s["calmar"],
                "trades_per_year": s["trades_per_year"],
                "frac_cash": s["frac_in_cash"],
                "mc_cagr_median": mc["cagr_median"],
                "mc_cagr_p05": mc["cagr_p05"],
                "mc_maxdd_median": mc["maxdd_median"],
                "mc_maxdd_p05": mc["maxdd_p05"],
                "p_dd_gt_25": mc["prob_maxdd_gt_25pct"],
                "p_dd_gt_35": mc["prob_maxdd_gt_35pct"],
                "p_dd_gt_50": mc["prob_maxdd_gt_50pct"],
                "p_cagr_neg": mc["prob_cagr_negative"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Threshold sweep across L_max in {1, 2, 3}...\n")
    for L in (1.0, 2.0, 3.0):
        df = run_threshold_sweep(L_max=L)
        print(f"\n=== L_max = {int(L)}x ===")
        cols = [
            "preset",
            "single_cagr",
            "single_maxdd",
            "single_calmar",
            "trades_per_year",
            "frac_cash",
            "mc_cagr_median",
            "mc_cagr_p05",
            "mc_maxdd_median",
            "mc_maxdd_p05",
            "p_dd_gt_25",
            "p_dd_gt_35",
            "p_dd_gt_50",
        ]
        print(df[cols].to_string(index=False))
