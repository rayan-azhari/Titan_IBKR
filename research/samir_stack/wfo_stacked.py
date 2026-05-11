"""WFO + sanctuary for the STACKED strategy with HMM-enhanced score.

Replaces the v1 equity-only WFO. This is the production validator for
the deployable strategy: 40/60 stack, L_max=3, 7-indicator regime score
(6 explicit indicators + causal pure-risk HMM).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.regime_score import (
    PRODUCTION_INDICATORS,
    regime_score_equal,
)
from research.samir_stack.stacked_strategy import (
    StackedConfig,
    run_stacked_strategy,
)


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


def run_wfo_stacked(
    data: dict[str, pd.Series],
    spy_ohlc: pd.DataFrame,
    cfg: StackedConfig,
    *,
    n_folds: int = 5,
    sanctuary_years: float = 1.0,
    start_year: int = 2003,
    enable_hmm: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run rolling-window WFO on the stacked strategy with sanctuary holdout.

    The HMM is fit causally inside ``build_indicator_panel`` (504-bar
    warmup, annual re-fit) so each fold uses only data available before
    that fold's evaluation period.
    """
    spy = data["spy"]
    full_end = spy.index.max()
    sanctuary_start = full_end - pd.Timedelta(days=int(sanctuary_years * 365))
    pre_sanctuary_end = sanctuary_start - pd.Timedelta(days=1)

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

        # Build panel using only data up to fold_end_dt
        spy_slice = spy.loc[backtest_start:fold_end_dt]
        ohlc_slice = spy_ohlc.loc[backtest_start:fold_end_dt] if enable_hmm else None
        panel = build_indicator_panel(
            spy_slice,
            vix_close=data["vix"].loc[:fold_end_dt],
            hyg_close=data["hyg"].loc[:fold_end_dt] if "hyg" in data else None,
            ief_close=data["ief"].loc[:fold_end_dt] if "ief" in data else None,
            tlt_close=data["tlt"].loc[:fold_end_dt] if "tlt" in data else None,
            spy_ohlc=ohlc_slice,
            enable_hmm=enable_hmm,
        )
        indicators = PRODUCTION_INDICATORS if enable_hmm else None
        score = (
            regime_score_equal(panel, indicators=indicators)
            if indicators
            else regime_score_equal(panel)
        )
        ief_slice = data["ief"].loc[backtest_start:fold_end_dt]
        res = run_stacked_strategy(spy_slice, ief_slice, score, cfg, indicator_panel=panel)

        fold_rets = res.loc[fold_start_dt:fold_end_dt, "ret_strategy"]
        m = _fold_metrics(fold_rets)
        m["fold"] = k
        m["start"] = str(fold_start_dt.date())
        m["end"] = str(fold_end_dt.date())
        fold_results.append(m)

    fold_df = pd.DataFrame(fold_results)

    # Sanctuary: build full panel, run strategy, slice last sanctuary_years
    panel_full = build_indicator_panel(
        spy,
        vix_close=data["vix"],
        hyg_close=data.get("hyg"),
        ief_close=data.get("ief"),
        tlt_close=data.get("tlt"),
        spy_ohlc=spy_ohlc if enable_hmm else None,
        enable_hmm=enable_hmm,
    )
    indicators = PRODUCTION_INDICATORS if enable_hmm else None
    score_full = (
        regime_score_equal(panel_full, indicators=indicators)
        if indicators
        else regime_score_equal(panel_full)
    )
    res_full = run_stacked_strategy(spy, data["ief"], score_full, cfg, indicator_panel=panel_full)
    sanctuary_rets = res_full.loc[sanctuary_start:full_end, "ret_strategy"]
    sanctuary = _fold_metrics(sanctuary_rets)
    sanctuary["start"] = str(sanctuary_start.date())
    sanctuary["end"] = str(full_end.date())
    return fold_df, sanctuary
