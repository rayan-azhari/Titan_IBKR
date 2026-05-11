"""Equity/bond allocation sweep on the improved Samir-Stack.

Tests whether the 40/60 default split is optimal or if shifting to
50/50 / 60/40 / 70/30 would improve the strategy.

Runs the sweep on TWO engines so we don't draw a conclusion from a
single equity-engine quirk:
  - Margin drift L=2 + I1+I2+I3 (the new champion)
  - 3x leveraged ETF + I1+I2+I3 (the original engine)

Quick smoke test — single full-period run per allocation, no WFO.

Usage:
    uv run python research/samir_stack/run_allocation_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_improved_with_margin import (  # noqa: E402
    _engine_3x_leveraged,
    _engine_margin_drift,
)
from research.samir_stack.run_samir_improvements import (  # noqa: E402
    bond_rotation_returns,
    compose_with_rate_shock,
)
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)


def run_with_allocation(
    spy: pd.Series,
    efa: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    tlt: pd.Series,
    samir_score: pd.Series,
    *,
    equity_weight: float,
    bond_weight: float,
    L_max: float,
    equity_engine,
) -> pd.DataFrame:
    """Same as run_improved_stack but exposes equity/bond weight knobs."""
    common = (
        spy.index.intersection(efa.index)
        .intersection(ief.index)
        .intersection(hyg.index)
        .intersection(tlt.index)
        .intersection(samir_score.index)
    )
    spy_a = spy.reindex(common)
    efa_a = efa.reindex(common)
    ief_a = ief.reindex(common)
    hyg_a = hyg.reindex(common)
    tlt_a = tlt.reindex(common)
    samir_a = samir_score.reindex(common)

    # I3: opt-in EFA
    spy_ret = spy_a.pct_change().fillna(0.0)
    efa_ret = efa_a.pct_change().fillna(0.0)
    spy_12m = spy_a.pct_change(252)
    efa_12m = efa_a.pct_change(252)
    use_efa_mask = (efa_12m - spy_12m) > 0.05
    chosen_ret = pd.Series(
        np.where(use_efa_mask.fillna(False), efa_ret.values, spy_ret.values),
        index=common,
    )
    equity_underlying = (1.0 + chosen_ret).cumprod() * float(spy_a.iloc[0])

    # I2: bond rotation
    rot_rets = bond_rotation_returns(ief_a, hyg_a)
    bond_underlying = (1.0 + rot_rets.reindex(common).fillna(0.0)).cumprod() * float(ief_a.iloc[0])

    # I1: rate-shock score
    score = compose_with_rate_shock(samir_a, tlt_a)

    # Monkey-patch equity engine
    import research.samir_stack.stacked_strategy as ss_mod

    saved = ss_mod.synthetic_leveraged_returns

    def patched(spy_series, leverage, **_kwargs):
        return equity_engine(spy_series, leverage)

    ss_mod.synthetic_leveraged_returns = patched

    cfg = StackedConfig(
        equity_weight=equity_weight,
        bond_weight=bond_weight,
        L_max=L_max,
        tier_thresholds=tuple([0.30, 0.50, 0.75][: int(L_max)]),
    )
    try:
        return run_stacked_strategy(equity_underlying, bond_underlying, score, cfg)
    finally:
        ss_mod.synthetic_leveraged_returns = saved


def _summary(df: pd.DataFrame) -> dict:
    rets = df["ret_strategy"]
    eq = df["equity"]
    n_years = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    return {
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sh, 3),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
    }


def main() -> int:
    print("Loading data...", flush=True)
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
    efa = efa.reindex(common)
    print(
        f"Range: {common.min().date()} → {common.max().date()} ({len(common)} bars)\n", flush=True
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    # Sweep grid: equity_weight values
    splits = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    engines = [
        ("Margin drift L=2", _engine_margin_drift, 2.0),
        ("3x leveraged ETF", _engine_3x_leveraged, 3.0),
    ]

    rows = []
    for engine_name, engine, L_max in engines:
        print(f"Engine: {engine_name}", flush=True)
        for eq_w in splits:
            bd_w = 1.0 - eq_w
            df = run_with_allocation(
                spy,
                efa,
                ief,
                hyg,
                tlt,
                samir_score,
                equity_weight=eq_w,
                bond_weight=bd_w,
                L_max=L_max,
                equity_engine=engine,
            )
            s = _summary(df)
            s["engine"] = engine_name
            s["equity_pct"] = int(eq_w * 100)
            s["bond_pct"] = int(bd_w * 100)
            rows.append(s)

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print("Equity/bond allocation sweep — improved Samir-Stack (I1+I2+I3)")
    print("=" * 110)
    for engine_name, _, _ in engines:
        sub = df_out[df_out["engine"] == engine_name].copy()
        sub = sub[["equity_pct", "bond_pct", "cagr", "vol", "sharpe", "max_dd", "calmar"]]
        print(f"\n{engine_name}:")
        print(sub.to_string(index=False))
        # Highlight the best on each metric
        print(
            f"  Best CAGR:   {sub.loc[sub['cagr'].idxmax(), 'equity_pct']}/{100 - sub.loc[sub['cagr'].idxmax(), 'equity_pct']}"
        )
        print(
            f"  Best Sharpe: {sub.loc[sub['sharpe'].idxmax(), 'equity_pct']}/{100 - sub.loc[sub['sharpe'].idxmax(), 'equity_pct']}"
        )
        print(
            f"  Best Calmar: {sub.loc[sub['calmar'].idxmax(), 'equity_pct']}/{100 - sub.loc[sub['calmar'].idxmax(), 'equity_pct']}"
        )
        print(
            f"  Lowest DD:   {sub.loc[sub['max_dd'].idxmax(), 'equity_pct']}/{100 - sub.loc[sub['max_dd'].idxmax(), 'equity_pct']}"
        )

    out_csv = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack" / "allocation_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
