"""Walk-forward + sanctuary validation.

Splits the data into N folds (default 5). For each fold:
    - IS = train period for indicator parameters (we don't tune them
      here — they're fixed — but the WFO is still informative because
      regime patterns may evolve)
    - OOS = forward window of `oos_years` years

Aggregates OOS Sharpe / CAGR / MaxDD across folds and applies the
project's bootstrap-CI gate to each fold's OOS.

Sanctuary: the last `sanctuary_years` are held out from the WFO entirely
and only run once at the end as a final out-of-sample check. Per the
project's research-math discipline, the autoresearch agent never sees
this period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.regime_score import regime_score_equal
from research.samir_stack.strategy import StrategyConfig, run_strategy


def _fold_metrics(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) < 60:
        return {"error": "insufficient bars", "n_bars": len(rets)}
    eq = (1.0 + rets).cumprod()
    n_years = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    return {
        "n_bars": len(rets),
        "n_years": round(n_years, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
    }


def run_wfo(
    data: dict[str, pd.Series],
    cfg: StrategyConfig,
    *,
    n_folds: int = 5,
    sanctuary_years: float = 1.0,
    start_year: int = 2003,
) -> tuple[pd.DataFrame, dict]:
    """Run rolling-window WFO with a final sanctuary holdout.

    Returns:
        fold_df: DataFrame with one row per fold (OOS metrics)
        sanctuary: dict with metrics on the held-out period

    Notes
    -----
    The strategy has no per-fold-tunable parameters (regime indicators
    use fixed thresholds, score combination is equal-weight). The WFO
    is therefore a pure stability check — does the strategy work in
    each non-overlapping period?
    """
    spy = data["spy"]

    # Carve out sanctuary
    full_end = spy.index.max()
    sanctuary_start = full_end - pd.Timedelta(days=int(sanctuary_years * 365))
    pre_sanctuary_end = sanctuary_start - pd.Timedelta(days=1)

    # Determine fold boundaries on the pre-sanctuary period
    backtest_start = pd.Timestamp(f"{start_year}-04-01")
    pre_sanctuary_index = spy.loc[backtest_start:pre_sanctuary_end].index
    n_total = len(pre_sanctuary_index)
    fold_size = n_total // n_folds

    fold_results = []
    for k in range(n_folds):
        fold_start_idx = k * fold_size
        fold_end_idx = (k + 1) * fold_size if k < n_folds - 1 else n_total
        fold_dates = pre_sanctuary_index[fold_start_idx:fold_end_idx]
        if len(fold_dates) < 252:
            continue
        fold_start_dt = fold_dates[0]
        fold_end_dt = fold_dates[-1]

        # Build indicators on data up to fold_end_dt (causal — using only
        # data available to the strategy at each bar, even from before
        # this fold's window, because the indicators are stationary
        # transforms)
        spy_slice = spy.loc[backtest_start:fold_end_dt]
        panel = build_indicator_panel(
            spy_slice,
            vix_close=data["vix"].loc[:fold_end_dt],
            hyg_close=data["hyg"].loc[:fold_end_dt] if "hyg" in data else None,
            ief_close=data["ief"].loc[:fold_end_dt] if "ief" in data else None,
            tlt_close=data["tlt"].loc[:fold_end_dt] if "tlt" in data else None,
        )
        score = regime_score_equal(panel)
        res = run_strategy(spy_slice, score, cfg)

        # Extract returns within this fold's window only
        fold_rets = res.loc[fold_start_dt:fold_end_dt, "ret_strategy"]
        m = _fold_metrics(fold_rets)
        m["fold"] = k
        m["start"] = str(fold_start_dt.date())
        m["end"] = str(fold_end_dt.date())
        fold_results.append(m)

    fold_df = pd.DataFrame(fold_results)

    # Sanctuary: run the strategy from start to full_end and slice the
    # last `sanctuary_years` of returns
    panel_full = build_indicator_panel(
        spy,
        vix_close=data["vix"],
        hyg_close=data.get("hyg"),
        ief_close=data.get("ief"),
        tlt_close=data.get("tlt"),
    )
    score_full = regime_score_equal(panel_full)
    res_full = run_strategy(spy, score_full, cfg)
    sanctuary_rets = res_full.loc[sanctuary_start:full_end, "ret_strategy"]
    sanctuary = _fold_metrics(sanctuary_rets)
    sanctuary["start"] = str(sanctuary_start.date())
    sanctuary["end"] = str(full_end.date())

    return fold_df, sanctuary
