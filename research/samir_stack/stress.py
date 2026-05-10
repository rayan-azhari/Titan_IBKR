"""Stress-scenario analysis: how did the strategy do in each named crisis?

Slices the backtest equity curve to specific crisis windows and reports
the per-scenario MaxDD + return + days underwater. Helps localize WHERE
the global MaxDD came from.
"""

from __future__ import annotations

import pandas as pd

# Crisis windows, conservatively wide on both ends
CRISIS_WINDOWS = [
    ("GFC_2008", "2007-10-09", "2009-03-31"),
    ("GFC_recovery", "2009-03-09", "2010-04-30"),
    ("EU_debt_2011", "2011-04-29", "2011-10-31"),
    ("China_2015", "2015-05-20", "2016-02-29"),
    ("Brexit_2016", "2016-06-15", "2016-07-30"),
    ("Q4_2018", "2018-09-20", "2019-01-31"),
    ("COVID_crash", "2020-02-19", "2020-04-30"),
    ("COVID_recovery", "2020-03-23", "2020-12-31"),
    ("2022_rate_shock", "2022-01-03", "2022-12-31"),
    ("2025_2026_holdout", "2025-04-01", "2026-04-02"),
]


def scenario_metrics(rets: pd.Series, start: str, end: str) -> dict:
    """Compute scenario metrics on a return slice."""
    s = rets.loc[start:end]
    if len(s) < 2:
        return {"start": start, "end": end, "n_bars": 0, "error": "no data"}
    eq = (1.0 + s).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return {
        "start": start,
        "end": end,
        "n_bars": len(s),
        "total_return": round(float(eq.iloc[-1] - 1.0), 4),
        "max_dd": round(float(dd.min()), 4),
        "vol": round(float(s.std() * (252**0.5)), 4),
        "days_underwater": int((dd < -0.001).sum()),
    }


def run_stress_table(named_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a wide table: rows = (strategy, scenario), cols = metrics.

    Compare same scenario across strategies side-by-side.
    """
    rows = []
    for strat_name, rets in named_returns.items():
        for crisis, start, end in CRISIS_WINDOWS:
            m = scenario_metrics(rets, start, end)
            m["strategy"] = strat_name
            m["scenario"] = crisis
            rows.append(m)
    df = pd.DataFrame(rows)
    return df[
        [
            "strategy",
            "scenario",
            "start",
            "end",
            "total_return",
            "max_dd",
            "vol",
            "days_underwater",
            "n_bars",
        ]
    ]


def crisis_pivot(stress_df: pd.DataFrame, metric: str = "max_dd") -> pd.DataFrame:
    """Pivot to scenario-x-strategy matrix on a single metric."""
    pv = stress_df.pivot(index="scenario", columns="strategy", values=metric)
    # Order rows by chronological window
    crisis_order = [c[0] for c in CRISIS_WINDOWS]
    return pv.reindex(crisis_order)
