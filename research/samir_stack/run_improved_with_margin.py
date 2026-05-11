"""Improved Samir-Stack (I1+I2+I3) × margin-on-standard-ETF engines.

Combines the two threads from this session:

  - Improvements I1+I2+I3 from `run_samir_improvements.py`: rate-shock
    score added to regime panel, IEF/HYG/cash bond rotation, opt-in
    EFA tactical overlay.
  - Margin engines from `margin_model.py`: hold a 1× standard ETF
    (CSPX/SPY) financed via IBKR margin instead of buying a daily-
    rebalanced 3× leveraged ETF (3USL).

Hypothesis: the improvements' edge transfers cleanly to the margin
engine because they're all gating / asset-selection logic — they don't
depend on the equity-engine internals. Practical implication: if the
margin variant carries the improvements with it, the live deployment
gets the lower per-instrument cost (CSPX 0.07% TER vs 3USL 0.75%) AND
the improved risk-adjusted profile.

Usage:
    uv run python research/samir_stack/run_improved_with_margin.py
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
from research.samir_stack.margin_model import (  # noqa: E402
    constant_leverage_margin_returns,
    drift_margin_returns,
)
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_samir_improvements import (  # noqa: E402
    bond_rotation_returns,
    compose_with_rate_shock,
)
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)
from research.samir_stack.synthetic_3x import synthetic_leveraged_returns  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Equity engines (re-exported with consistent signature) ─────────────────


def _engine_3x_leveraged(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        synthetic_leveraged_returns(
            spy, leverage=leverage, ter_annual=0.0075 if leverage > 1.0 else 0.0
        )
        .reindex(spy.index)
        .fillna(0.0)
    )


def _engine_margin_const(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        constant_leverage_margin_returns(spy, leverage=leverage, broker="ibkr_pro")
        .reindex(spy.index)
        .fillna(0.0)
    )


def _engine_margin_drift(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        drift_margin_returns(spy, initial_leverage=leverage, broker="ibkr_pro")["equity_ret"]
        .reindex(spy.index)
        .fillna(0.0)
    )


# ── Stacked-strategy runner with engine + improvements ─────────────────────


def run_improved_stack(
    spy: pd.Series,
    efa: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    tlt: pd.Series,
    samir_score: pd.Series,
    *,
    L_max: float,
    equity_engine,
    use_rate_shock: bool = True,
    use_bond_rotation: bool = True,
    use_optin_efa: bool = True,
) -> pd.DataFrame:
    """Run Samir-Stack with any combination of the three improvements
    AND any equity engine (leveraged ETF or margin variants).

    Implementation: monkey-patches ``stacked_strategy.synthetic_leveraged_returns``
    with the chosen engine, builds synthetic equity / bond series for
    the chosen overlays, then calls ``run_stacked_strategy``.
    """
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
    if use_optin_efa:
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
    else:
        equity_underlying = spy_a

    # I2: bond rotation
    if use_bond_rotation:
        rot_rets = bond_rotation_returns(ief_a, hyg_a)
        bond_underlying = (1.0 + rot_rets.reindex(common).fillna(0.0)).cumprod() * float(
            ief_a.iloc[0]
        )
    else:
        bond_underlying = ief_a

    # I1: rate-shock score blend
    score = compose_with_rate_shock(samir_a, tlt_a) if use_rate_shock else samir_a

    # Monkey-patch the equity engine into stacked_strategy's namespace
    import research.samir_stack.stacked_strategy as ss_mod

    saved = ss_mod.synthetic_leveraged_returns

    def patched(spy_series, leverage, **_kwargs):
        return equity_engine(spy_series, leverage)

    ss_mod.synthetic_leveraged_returns = patched
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=L_max,
        tier_thresholds=tuple([0.30, 0.50, 0.75][: int(L_max)]),
    )
    try:
        return run_stacked_strategy(equity_underlying, bond_underlying, score, cfg)
    finally:
        ss_mod.synthetic_leveraged_returns = saved


# ── Reporting ──────────────────────────────────────────────────────────────


def _summary(df: pd.DataFrame, name: str) -> dict:
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
        "variant": name,
        "n_years": round(n_years, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sh, 3),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
    }


CRISIS_WINDOWS = [
    ("GFC 2008", "2008-01-01", "2009-06-30"),
    ("Q4 2018", "2018-09-01", "2018-12-31"),
    ("COVID 2020", "2020-02-01", "2020-04-30"),
    ("Rate shock 2022", "2022-01-01", "2022-12-31"),
]


def _crisis_row(name: str, df: pd.DataFrame) -> dict:
    rec = {"variant": name}
    for crisis_name, start, end in CRISIS_WINDOWS:
        window = df.loc[start:end, "ret_strategy"]
        rec[crisis_name] = (
            round(float((1.0 + window).cumprod().iloc[-1] - 1.0) * 100, 2)
            if len(window) > 0
            else float("nan")
        )
    return rec


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

    # 8 variants: {3x ETF, margin const-L=2, margin const-L=3, margin drift L=2}
    # × {baseline (no improvements), I1+I2+I3 (all improvements)}
    cases = [
        ("3x leveraged ETF", _engine_3x_leveraged, 3.0),
        ("Margin const-L=2 CSPX", _engine_margin_const, 2.0),
        ("Margin const-L=3 CSPX", _engine_margin_const, 3.0),
        ("Margin drift L=2 CSPX", _engine_margin_drift, 2.0),
    ]

    results = []
    crisis_rows = []
    for label, engine, L_max in cases:
        # Baseline (no improvements)
        df_base = run_improved_stack(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            L_max=L_max,
            equity_engine=engine,
            use_rate_shock=False,
            use_bond_rotation=False,
            use_optin_efa=False,
        )
        results.append(_summary(df_base, f"{label}: baseline"))
        crisis_rows.append(_crisis_row(f"{label}: baseline", df_base))

        # All three improvements
        df_imp = run_improved_stack(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            L_max=L_max,
            equity_engine=engine,
            use_rate_shock=True,
            use_bond_rotation=True,
            use_optin_efa=True,
        )
        results.append(_summary(df_imp, f"{label}: + I1+I2+I3"))
        crisis_rows.append(_crisis_row(f"{label}: + I1+I2+I3", df_imp))

    summary = pd.DataFrame(results).set_index("variant")
    print("=" * 120)
    print("Improved Samir-Stack across equity engines: 3x leveraged ETF vs IBKR margin variants")
    print("=" * 120)
    print(summary.to_string())

    print("\n" + "=" * 120)
    print("Crisis-window cumulative return (%):")
    print("=" * 120)
    print(pd.DataFrame(crisis_rows).set_index("variant").to_string())

    out_csv = REPORTS_DIR / "improved_with_margin.csv"
    summary.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
