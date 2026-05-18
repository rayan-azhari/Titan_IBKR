"""Benchmark strategies for the Samir-Stack comparison.

The bar to clear is not "beat SPY buy-hold" — leverage will do that on
mean returns alone. The bar is to beat **Faber GTAA** (Samir's framework
without leverage) on RoR-adjusted return. If we can't, the leverage
adds risk without earning its keep.

All benchmarks operate on the same date range as the strategy, with
identical cost / TER assumptions where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.synthetic_3x import synthetic_leveraged_returns
from titan.research.metrics import BARS_PER_YEAR, annualize_vol

_DAILY = BARS_PER_YEAR["D"]


def _summarize(rets: pd.Series, name: str) -> dict:
    """Standard summary of a daily return series."""
    rets = rets.dropna()
    if len(rets) < 20:
        return {"name": name, "error": "insufficient bars"}
    eq = (1.0 + rets).cumprod()
    n_years = len(rets) / _DAILY
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)
    vol = annualize_vol(float(rets.std()), periods_per_year=_DAILY)
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(_DAILY))
        if rets.std() > 1e-12 else 0.0
    )
    sortino = (
        float(rets.mean() / rets[rets < 0].std() * np.sqrt(_DAILY))
        if (rets < 0).sum() >= 5 and rets[rets < 0].std() > 1e-12
        else 0.0
    )
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    # Ulcer index — sqrt(mean(DD^2)) over the curve
    dd_series = (eq - peak) / peak
    ulcer = float(np.sqrt((dd_series**2).mean()))
    return {
        "name": name,
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "ulcer": round(ulcer, 4),
    }


def benchmark_spy_buy_hold(spy_close: pd.Series) -> tuple[pd.Series, dict]:
    """Plain SPY buy and hold (no expenses)."""
    rets = spy_close.pct_change()
    return rets, _summarize(rets, "SPY buy-hold")


def benchmark_60_40(
    spy_close: pd.Series, ief_close: pd.Series, *, rebalance_freq: str = "M"
) -> tuple[pd.Series, dict]:
    """Classic 60/40 with monthly rebalancing.

    Each month, target weights are reset to 60% SPY / 40% IEF. Within the
    month, weights drift with returns — no explicit transaction cost.
    """
    common = spy_close.index.intersection(ief_close.index)
    spy = spy_close.loc[common]
    ief = ief_close.loc[common]
    spy_r = spy.pct_change().fillna(0.0)
    ief_r = ief.pct_change().fillna(0.0)

    # Build month-end rebalance dates
    rebal_dates = pd.Series(common, index=common).resample(rebalance_freq).last().dropna().index
    w_spy = pd.Series(0.6, index=common)
    w_ief = pd.Series(0.4, index=common)
    # Drift weights between rebalances
    cur_w_spy, cur_w_ief = 0.6, 0.4
    for i, ts in enumerate(common):
        # Drift weights with returns
        s_grow = (1.0 + spy_r.iloc[i]) * cur_w_spy
        i_grow = (1.0 + ief_r.iloc[i]) * cur_w_ief
        total = s_grow + i_grow
        cur_w_spy = s_grow / total
        cur_w_ief = i_grow / total
        # Rebalance if at month-end
        if ts in rebal_dates:
            cur_w_spy, cur_w_ief = 0.6, 0.4
        w_spy.iloc[i] = cur_w_spy
        w_ief.iloc[i] = cur_w_ief

    # Portfolio return = w_spy * spy_r + w_ief * ief_r (using prior weights)
    rets = w_spy.shift(1).fillna(0.6) * spy_r + w_ief.shift(1).fillna(0.4) * ief_r
    return rets, _summarize(rets, "60/40 SPY/IEF")


def benchmark_faber_gtaa(spy_close: pd.Series, *, ma_window: int = 200) -> tuple[pd.Series, dict]:
    """Faber's Global Tactical Asset Allocation lite — single asset variant.

    Hold SPY when close > 200d SMA, else cash (zero return). Decision is
    end-of-day, applied next bar (shift discipline). This is the natural
    benchmark for any regime-gated strategy: the simplest possible
    risk-on/risk-off rule.
    """
    spy_r = spy_close.pct_change()
    sma = spy_close.rolling(ma_window).mean()
    in_market = (spy_close > sma).astype(float)
    # Apply position from previous bar (shift discipline)
    rets = spy_r * in_market.shift(1).fillna(0.0)
    return rets, _summarize(rets, f"Faber GTAA ({ma_window}d)")


def benchmark_hfea(
    spy_close: pd.Series,
    tlt_close: pd.Series,
    *,
    spy_weight: float = 0.55,
    tlt_weight: float = 0.45,
    spy_lev: float = 3.0,
    tlt_lev: float = 3.0,
    rebalance_freq: str = "Q",
) -> tuple[pd.Series, dict]:
    """HFEA (Hedgefundie's Excellent Adventure): 55% UPRO / 45% TMF.

    Approximates UPRO (3x SPY) and TMF (3x TLT) via the synthetic
    leveraged-ETF generator. Quarterly rebalance.
    """
    common = spy_close.index.intersection(tlt_close.index)
    spy = spy_close.loc[common]
    tlt = tlt_close.loc[common]
    upro_r = synthetic_leveraged_returns(spy, leverage=spy_lev).fillna(0.0)
    tmf_r = synthetic_leveraged_returns(tlt, leverage=tlt_lev).fillna(0.0)

    rebal_dates = pd.Series(common, index=common).resample(rebalance_freq).last().dropna().index
    cur_w_spy, cur_w_tlt = spy_weight, tlt_weight
    weights_spy = pd.Series(0.0, index=common)
    weights_tlt = pd.Series(0.0, index=common)
    for i, ts in enumerate(common):
        s_grow = (1.0 + upro_r.iloc[i]) * cur_w_spy
        t_grow = (1.0 + tmf_r.iloc[i]) * cur_w_tlt
        total = s_grow + t_grow
        cur_w_spy = s_grow / total if total > 0 else 0
        cur_w_tlt = t_grow / total if total > 0 else 0
        if ts in rebal_dates:
            cur_w_spy, cur_w_tlt = spy_weight, tlt_weight
        weights_spy.iloc[i] = cur_w_spy
        weights_tlt.iloc[i] = cur_w_tlt

    rets = (
        weights_spy.shift(1).fillna(spy_weight) * upro_r
        + weights_tlt.shift(1).fillna(tlt_weight) * tmf_r
    )
    return rets, _summarize(rets, "HFEA 55/45 UPRO/TMF (synth)")


def benchmark_samir_pure(
    spy_close: pd.Series,
    regime_score: pd.Series,
    *,
    threshold: float = 0.30,
    re_entry_threshold: float = 0.70,
    re_entry_bars: int = 5,
) -> tuple[pd.Series, dict]:
    """Pure Samir binary: 1x SPY when regime benign, cash when hostile.

    No leverage, no tiers. Uses the same regime classifier as Samir-Stack
    so the comparison isolates the leverage contribution.
    """
    common = spy_close.index.intersection(regime_score.index)
    spy_r = spy_close.loc[common].pct_change().fillna(0.0)
    score = regime_score.loc[common]

    in_market = pd.Series(0.0, index=common)
    state = 0.0
    quiet = 9999
    for i, ts in enumerate(common):
        s = score.iloc[i]
        if state == 1.0:
            if s < threshold:
                state = 0.0
                quiet = 0
        else:
            if s >= re_entry_threshold:
                quiet += 1
                if quiet >= re_entry_bars:
                    state = 1.0
            elif s < threshold:
                quiet = 0
        in_market.iloc[i] = state

    rets = spy_r * in_market.shift(1).fillna(0.0)
    return rets, _summarize(rets, "Samir-pure (1x + regime gate)")


def all_benchmarks(
    spy_close: pd.Series,
    ief_close: pd.Series,
    tlt_close: pd.Series,
    regime_score: pd.Series,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Run every benchmark and return (returns_dict, summary_df)."""
    out = {}
    summaries = []
    for name, runner in [
        ("spy", lambda: benchmark_spy_buy_hold(spy_close)),
        ("60_40", lambda: benchmark_60_40(spy_close, ief_close)),
        ("faber_200", lambda: benchmark_faber_gtaa(spy_close, ma_window=200)),
        ("hfea", lambda: benchmark_hfea(spy_close, tlt_close)),
        ("samir_pure", lambda: benchmark_samir_pure(spy_close, regime_score)),
    ]:
        try:
            rets, summary = runner()
            out[name] = rets
            summaries.append(summary)
        except Exception as e:
            summaries.append({"name": name, "error": str(e)})
    return out, pd.DataFrame(summaries)
