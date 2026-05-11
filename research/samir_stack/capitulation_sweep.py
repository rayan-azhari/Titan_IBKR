"""Capitulation-overlay parameter sweep with robustness gates.

WHY THIS IS HARD: there are only 2 capitulation events in our 2003-2026
sample (2008-10 and 2020-03). Naive grid search will overfit to those
dates — any parameter set that *happens* to fire on Oct 17, 2008 instead
of Oct 24 will look better in-sample but isn't generalisable.

ROBUSTNESS GATES — a parameter set is acceptable only if:
  1. It produces ≥1 activation per crisis (BOTH 2008 and 2020), not just one.
  2. Per-crisis stress impact is positive in BOTH crises individually.
  3. Monte Carlo P(DD>50%) increase is < 0.3pp vs v1.
  4. WFO mean fold CAGR is ≥ v1.

Sweep approach:
  * 1D sensitivity: vary each key param individually around the current
    default. Look for plateaus of stable improvement.
  * 2D coarse grid: vary the two most-impactful params jointly. Pick a
    cell that's surrounded by other passing cells (= robust region).
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from research.samir_stack.capitulation import CapitulationConfig
from research.samir_stack.data_loader import load_panel
from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.monte_carlo import monte_carlo_ror
from research.samir_stack.regime_score import regime_score_equal
from research.samir_stack.stacked_strategy import (
    StackedConfig,
    run_stacked_strategy,
    summarize_stacked,
)

CRISIS_WINDOWS = [
    ("GFC_recovery", "2008-10-01", "2010-04-30"),
    ("COVID_recovery", "2020-03-01", "2020-12-31"),
]


def _evaluate(
    cap_cfg: CapitulationConfig,
    spy: pd.Series,
    ief: pd.Series,
    score: pd.Series,
    panel: pd.DataFrame,
    baseline_summary: dict,
    baseline_rets: pd.Series,
    baseline_mc_p50: float,
    baseline_per_crisis: dict,
) -> dict:
    """Run one parameter set and return a result dict with robustness flags."""
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=3.0,
        capitulation=cap_cfg,
    )
    res = run_stacked_strategy(spy, ief, score, cfg, indicator_panel=panel)
    s = summarize_stacked(res)

    # Activations
    activations = (res["opportunistic"].astype(int).diff() == 1).sum()
    activation_dates = res.loc[res["opportunistic"].astype(int).diff() == 1].index.tolist()

    # Per-crisis return uplift vs baseline
    per_crisis = {}
    fired_in_crisis = {}
    for label, start, end in CRISIS_WINDOWS:
        rets_v2 = res.loc[start:end, "ret_strategy"]
        if len(rets_v2) == 0:
            per_crisis[label] = 0.0
            fired_in_crisis[label] = False
            continue
        v2_ret = float((1 + rets_v2).cumprod().iloc[-1] - 1)
        v1_ret = baseline_per_crisis[label]
        per_crisis[label] = v2_ret - v1_ret
        # Did the overlay fire within this window?
        fired_dates_in_window = [
            d for d in activation_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)
        ]
        fired_in_crisis[label] = len(fired_dates_in_window) > 0

    # Monte Carlo (small for speed; user can re-run definitive at end)
    mc = monte_carlo_ror(res["ret_strategy"], n_paths=1500, mean_block_len=63, seed=42)

    # Robustness flags
    fires_both = fired_in_crisis.get("GFC_recovery", False) and fired_in_crisis.get(
        "COVID_recovery", False
    )
    helps_both = (
        per_crisis.get("GFC_recovery", -1.0) > 0
        and per_crisis.get("COVID_recovery", -1.0) > 0
    )
    ror_acceptable = mc["prob_maxdd_gt_50pct"] - baseline_mc_p50 < 0.003  # < 0.3pp
    cagr_uplift = s["cagr"] - baseline_summary["cagr"]
    cagr_acceptable = cagr_uplift >= 0

    return {
        "cagr": s["cagr"],
        "max_dd": s["max_dd"],
        "calmar": s["calmar"],
        "cagr_uplift_bp": int(cagr_uplift * 10000),
        "activations": int(activations),
        "fires_GFC": fired_in_crisis.get("GFC_recovery", False),
        "fires_COVID": fired_in_crisis.get("COVID_recovery", False),
        "GFC_uplift": round(per_crisis.get("GFC_recovery", 0.0), 4),
        "COVID_uplift": round(per_crisis.get("COVID_recovery", 0.0), 4),
        "p_dd50": mc["prob_maxdd_gt_50pct"],
        "p_dd35": mc["prob_maxdd_gt_35pct"],
        "fires_both": fires_both,
        "helps_both": helps_both,
        "ror_ok": ror_acceptable,
        "cagr_ok": cagr_acceptable,
        "robust_pass": fires_both and helps_both and ror_acceptable and cagr_acceptable,
    }


def setup() -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, dict, pd.Series, float, dict]:
    """Load data and compute the v1 baseline."""
    data = load_panel(start="2003-04-01", end="2026-04-02")
    panel = build_indicator_panel(
        data["spy"],
        vix_close=data["vix"],
        hyg_close=data["hyg"],
        ief_close=data["ief"],
        tlt_close=data["tlt"],
    )
    score = regime_score_equal(panel)

    cfg_v1 = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=3.0)
    res_v1 = run_stacked_strategy(data["spy"], data["ief"], score, cfg_v1)
    s_v1 = summarize_stacked(res_v1)
    mc_v1 = monte_carlo_ror(res_v1["ret_strategy"], n_paths=1500, mean_block_len=63, seed=42)
    p50_v1 = mc_v1["prob_maxdd_gt_50pct"]

    baseline_per_crisis = {}
    for label, start, end in CRISIS_WINDOWS:
        rets = res_v1.loc[start:end, "ret_strategy"]
        baseline_per_crisis[label] = float((1 + rets).cumprod().iloc[-1] - 1) if len(rets) else 0.0

    return data["spy"], data["ief"], score, panel, s_v1, res_v1["ret_strategy"], p50_v1, baseline_per_crisis


def sensitivity_1d(
    spy, ief, score, panel, base_summary, base_rets, base_p50, base_crisis,
    *,
    param: str,
    values: list,
    base_overrides: dict | None = None,
) -> pd.DataFrame:
    """Sweep one parameter while keeping others at their (current default) values."""
    rows = []
    for v in values:
        kwargs = {"enabled": True}
        if base_overrides:
            kwargs.update(base_overrides)
        kwargs[param] = v
        cfg = CapitulationConfig(**kwargs)
        result = _evaluate(
            cfg, spy, ief, score, panel, base_summary, base_rets, base_p50, base_crisis
        )
        result[param] = v
        rows.append(result)
    return pd.DataFrame(rows)


def sweep_2d(
    spy, ief, score, panel, base_summary, base_rets, base_p50, base_crisis,
    *,
    param_a: str,
    values_a: list,
    param_b: str,
    values_b: list,
    base_overrides: dict | None = None,
) -> pd.DataFrame:
    """Sweep two parameters jointly. Returns long-format DataFrame."""
    rows = []
    for va, vb in product(values_a, values_b):
        kwargs = {"enabled": True}
        if base_overrides:
            kwargs.update(base_overrides)
        kwargs[param_a] = va
        kwargs[param_b] = vb
        cfg = CapitulationConfig(**kwargs)
        result = _evaluate(
            cfg, spy, ief, score, panel, base_summary, base_rets, base_p50, base_crisis
        )
        result[param_a] = va
        result[param_b] = vb
        rows.append(result)
    return pd.DataFrame(rows)
