"""Samir-Stack overlay sweep: capitulation overlay + portfolio vol targeting.

Tests two independently-applied uplifts on top of the L=3 / 10/90 / MES
futures champion (from research/samir_stack/run_futures_sweep.py Part D):

  V1. **Capitulation overlay** — already implemented in
      research/samir_stack/capitulation.py and parameter-tuned in May
      2026. Currently disabled by default. Detects extreme VIX +
      drawdown + credit-panic events, waits for stabilisation
      (5d-bounce / VIX recovery / vol mean-revert), then enters at
      tier=2 in cash regimes that the slow regime gate would otherwise
      keep flat. Failed-bounce stop (-5% / 10 bars) limits damage from
      false dawns.

  V2. **Portfolio vol targeting** — multiplicative scaler applied to
      strategy daily returns post-hoc. Computes 30-day realised vol of
      the strategy itself (not the underlying), scales by
      ``target_vol / realised_vol`` (capped at ``max_scale``), shifted
      by 1 day to avoid look-ahead. Mean-reverting scaler that holds
      total portfolio vol roughly constant across regimes.

Sweep grid: {baseline, +V1, +V2, +V1+V2} × WFO over 16 OOS years.

The honest hypothesis is that V1 buys ~+0.05 Sharpe with material DD
improvement (especially in COVID-style V-bottoms) and V2 buys ~+0.10
Sharpe by smoothing realised vol. They should compose roughly
additively because they act on different parts of the return path
(V1 = entry timing, V2 = sizing).

Usage::

    uv run python research/samir_stack/run_overlay_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.capitulation import CapitulationConfig  # noqa: E402
from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.margin_model import futures_returns_tr  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_samir_improvements import (  # noqa: E402
    bond_rotation_returns,
    compose_with_rate_shock,
)
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Champion engine: MES futures L=3 ─────────────────────────────────


def _engine_futures(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return futures_returns_tr(spy, leverage=leverage).reindex(spy.index).fillna(0.0)


# ── V2: vol-targeting wrapper ────────────────────────────────────────


def vol_target(
    rets: pd.Series,
    *,
    target_vol_annual: float = 0.06,
    vol_window: int = 30,
    max_scale: float = 2.0,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Multiplicative vol-targeting scaler applied to a return series.

    Computes rolling realised vol from the *strategy's own* returns, not
    the underlying. Scales by ``target_vol / realised_vol``, capped at
    ``max_scale``, and **shifted by 1 day** so today's scale is computed
    from yesterday's window (no look-ahead).

    Returns the scaled return series with the same index. The first
    ``vol_window`` bars carry NaN scales which are filled with 1.0
    (un-scaled) so the equity curve doesn't break.
    """
    realised_vol = rets.rolling(vol_window).std() * np.sqrt(trading_days_per_year)
    scale = (target_vol_annual / realised_vol).clip(upper=max_scale)
    # Lag by 1 to avoid look-ahead. First bar after the window opens at scale=1.
    scale = scale.shift(1).fillna(1.0)
    return rets * scale


# ── Strategy runner with both overlays optional ──────────────────────


def run_with_overlays(
    spy: pd.Series,
    _efa: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    tlt: pd.Series,
    samir_score: pd.Series,
    indicator_panel: pd.DataFrame,
    *,
    L_max: float = 3.0,
    equity_weight: float = 0.10,
    bond_weight: float = 0.90,
    use_capitulation: bool = False,
    use_vol_target: bool = False,
    vol_target_annual: float = 0.06,
    vol_window: int = 30,
) -> pd.DataFrame:
    """Run improved Samir-Stack with optional V1 (capitulation) and V2 (vol
    target). Uses I1 (rate-shock) + I2 (bond rotation, properly lagged) +
    MES futures (corrected). I3 (opt-in EFA) DROPPED per remediation plan
    §0(1) — the ``_efa`` arg is retained positionally for back-compat but
    unused."""
    common = (
        spy.index.intersection(ief.index)
        .intersection(hyg.index)
        .intersection(tlt.index)
        .intersection(samir_score.index)
    )
    spy_a = spy.reindex(common)
    ief_a = ief.reindex(common)
    hyg_a = hyg.reindex(common)
    tlt_a = tlt.reindex(common)
    samir_a = samir_score.reindex(common)
    panel_a = indicator_panel.reindex(common)

    # I3 (opt-in EFA) DROPPED — equity underlying is plain SPY.
    equity_underlying = spy_a

    # I2: bond rotation (now properly lagged inside bond_rotation_returns)
    rot_rets = bond_rotation_returns(ief_a, hyg_a)
    bond_underlying = (1.0 + rot_rets.reindex(common).fillna(0.0)).cumprod() * float(ief_a.iloc[0])

    # I1: rate-shock score blend
    score = compose_with_rate_shock(samir_a, tlt_a)

    # Monkey-patch MES futures engine
    import research.samir_stack.stacked_strategy as ss_mod

    saved = ss_mod.synthetic_leveraged_returns

    def patched(spy_series, leverage, **_kwargs):
        return _engine_futures(spy_series, leverage)

    ss_mod.synthetic_leveraged_returns = patched

    cap_cfg = CapitulationConfig(enabled=True) if use_capitulation else None

    cfg = StackedConfig(
        equity_weight=equity_weight,
        bond_weight=bond_weight,
        L_max=L_max,
        tier_thresholds=(0.30, 0.50, 0.75),  # L_max=3 default
        capitulation=cap_cfg,
    )
    try:
        df = run_stacked_strategy(
            equity_underlying,
            bond_underlying,
            score,
            cfg,
            tlt_close=tlt_a,
            indicator_panel=panel_a if use_capitulation else None,
        )
    finally:
        ss_mod.synthetic_leveraged_returns = saved

    if use_vol_target:
        scaled = vol_target(
            df["ret_strategy"],
            target_vol_annual=vol_target_annual,
            vol_window=vol_window,
        )
        df = df.copy()
        df["ret_strategy"] = scaled
        df["equity"] = (1.0 + scaled.fillna(0.0)).cumprod()
    return df


# ── Stats / WFO helpers ──────────────────────────────────────────────


def _wfo_stitch(
    df: pd.DataFrame, *, is_days: int = 504, oos_days: int = 252, step: int = 252
) -> tuple[np.ndarray, list[dict]]:
    n = len(df)
    if n < is_days + oos_days:
        return np.array([]), []
    rets = df["ret_strategy"].to_numpy()
    fold_rows: list[dict] = []
    stitched: list[np.ndarray] = []
    fold_idx = 0
    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_end = oos_start + oos_days
        slice_rets = rets[oos_start:oos_end]
        sh = sharpe(slice_rets, periods_per_year=BARS_PER_YEAR["D"])
        eq = np.cumprod(1.0 + slice_rets)
        peak = np.maximum.accumulate(eq)
        maxdd = float(((eq - peak) / peak).min())
        fold_rows.append(
            {
                "fold": fold_idx,
                "sharpe": round(sh, 3),
                "max_dd": round(maxdd, 4),
            }
        )
        stitched.append(slice_rets)
        fold_idx += 1
        oos_start += step
    return np.concatenate(stitched) if stitched else np.array([]), fold_rows


def _summarise(label: str, df: pd.DataFrame) -> dict:
    stitched, fold_rows = _wfo_stitch(df)
    if len(stitched) == 0:
        return {"variant": label, "n_oos_years": 0.0}
    sh = sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"])
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    n_years = len(stitched) / 252.0
    eq = np.cumprod(1.0 + stitched)
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    vol = float(np.std(stitched) * np.sqrt(BARS_PER_YEAR["D"]))
    pos_pct = float(np.mean([f["sharpe"] > 0 for f in fold_rows])) if fold_rows else 0.0
    calmar = cagr / abs(dd) if dd < -1e-9 else 0.0
    return {
        "variant": label,
        "n_oos_years": round(n_years, 2),
        "stitched_sharpe": round(sh, 3),
        "ci95_lo": round(ci_lo, 3),
        "ci95_hi": round(ci_hi, 3),
        "stitched_cagr": round(cagr, 4),
        "stitched_vol": round(vol, 4),
        "stitched_max_dd": round(dd, 4),
        "calmar": round(calmar, 3),
        "pct_pos_folds": round(pos_pct, 3),
        "passes_gate": ci_lo > 0,
    }


def _crisis_returns(df: pd.DataFrame) -> dict:
    """Cumulative return during three stress windows."""
    crises = [
        ("GFC_2008", "2008-01-01", "2009-06-30"),
        ("COVID_2020", "2020-02-01", "2020-04-30"),
        ("Rate_Shock_2022", "2022-01-01", "2022-12-31"),
    ]
    rec = {}
    for name, start, end in crises:
        window = df.loc[start:end, "ret_strategy"]
        if len(window) == 0:
            rec[name] = float("nan")
        else:
            rec[name] = round(float((1.0 + window).cumprod().iloc[-1] - 1.0) * 100, 2)
    return rec


# ── Main ─────────────────────────────────────────────────────────────


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
    efa_a = efa.reindex(common)
    print(
        f"Range: {common.min().date()} -> {common.max().date()} ({len(common)} bars)\n",
        flush=True,
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    # Sweep grid
    cases = [
        ("baseline (L=3 10/90 + I1+I2+I3)", False, False),
        ("+ V1 capitulation overlay", True, False),
        ("+ V2 vol target 6%", False, True),
        ("+ V1+V2 (both)", True, True),
    ]

    summaries: list[dict] = []
    crisis_rows: list[dict] = []
    for label, use_cap, use_vt in cases:
        print(f"  {label}...", flush=True)
        df = run_with_overlays(
            spy,
            efa_a,
            ief,
            hyg,
            tlt,
            samir_score,
            panel,
            use_capitulation=use_cap,
            use_vol_target=use_vt,
        )
        summaries.append(_summarise(label, df))
        crisis_rows.append({"variant": label, **_crisis_returns(df)})

    summary_df = pd.DataFrame(summaries).set_index("variant")
    print()
    print("=" * 110)
    print("OVERLAY SWEEP — 16-fold WFO on MES L=3 10/90 champion")
    print("=" * 110)
    print(summary_df.to_string())

    # Compute deltas vs baseline
    base = summary_df.loc["baseline (L=3 10/90 + I1+I2+I3)"]
    print()
    print("=" * 110)
    print("DELTAS vs baseline:")
    print("=" * 110)
    for label, _, _ in cases[1:]:
        row = summary_df.loc[label]
        print(
            f"  {label:<35}  d_sharpe={row['stitched_sharpe'] - base['stitched_sharpe']:+.3f}  "
            f"d_cagr={(row['stitched_cagr'] - base['stitched_cagr']) * 100:+.2f}pp  "
            f"d_maxdd={(row['stitched_max_dd'] - base['stitched_max_dd']) * 100:+.2f}pp  "
            f"d_vol={(row['stitched_vol'] - base['stitched_vol']) * 100:+.2f}pp  "
            f"d_ci_lo={row['ci95_lo'] - base['ci95_lo']:+.3f}"
        )

    # ── Vol-target parameter sensitivity sweep ───────────────────────
    print()
    print("=" * 110)
    print("VOL-TARGET SENSITIVITY: sweep target_vol_annual at baseline (no capitulation)")
    print("=" * 110)
    vt_grid = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    vt_rows: list[dict] = []
    for tv in vt_grid:
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
        rec = _summarise(f"vol_target={tv:.2f}", df)
        rec["target_vol"] = tv
        vt_rows.append(rec)
    vt_df = pd.DataFrame(vt_rows).set_index("variant")
    print(vt_df.to_string())
    vt_df.to_csv(REPORTS_DIR / "vol_target_sensitivity.csv")

    # Crisis stress-window comparison
    crisis_df = pd.DataFrame(crisis_rows).set_index("variant")
    print()
    print("=" * 110)
    print("Crisis-window cumulative return (%):")
    print("=" * 110)
    print(crisis_df.to_string())

    summary_df.to_csv(REPORTS_DIR / "overlay_sweep_summary.csv")
    crisis_df.to_csv(REPORTS_DIR / "overlay_sweep_crises.csv")
    print(f"\nSaved: {REPORTS_DIR / 'overlay_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
