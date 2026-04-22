"""v4 Portfolio optimizer — uses the refined parameters from the
cross-asset parameter sweep (directives/Param Sweep 2026-04-22.md).

Changes vs v3:
  * HYG->IWB:  threshold 0.50 -> 0.25           (+22% Sharpe)
  * TIP->HYG:  hold 20 -> 40, threshold 0.50 -> 0.25   (+15% Sharpe)
  * LQD->HYG:  dropped (0.67 corr with TIP->HYG)
  * TLT->HYG:  dropped (same credit cluster, inferior to TIP)
  * TLT->QQQ:  unchanged (th=0.50 is its sweep-optimal)
  * MR AUD/JPY anchor=24: unchanged
  * IEF->GLD lb=60: kept as 5% diversifier slot (uncorrelated)
  * ML IWB:    overlay, reported standalone, not in the correlation mix

Output:
  .tmp/reports/param_sweep_2026_04_22/v4_portfolio.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "param_sweep_2026_04_22" / "v4_portfolio.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)


# ── Candidate loaders ──────────────────────────────────────────────────


def get_mr_audjpy_returns(vwap_anchor: int = 24) -> pd.Series:
    from research.mean_reversion.run_confluence_regime_test import (
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import run_mr_wfo

    df = load_h1("AUD_JPY")
    close = df["close"]
    deviation = compute_vwap_deviation(close, anchor_period=vwap_anchor)
    mask = build_confluence_disagreement_mask(df, "donchian_pos_20")
    r = run_mr_wfo(
        close,
        deviation,
        mask,
        [0.95, 0.98, 0.99, 0.999],
        is_bars=32000,
        oos_bars=8000,
    )
    s = r["stitched_returns"]
    s.name = f"mr_audjpy_anchor{vwap_anchor}"
    return s


def get_bond_equity_returns(
    bond: str, target: str, lookback: int, hold_days: int = 20, threshold: float = 0.50
) -> pd.Series:
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    r = run_bond_wfo(
        load_daily(bond),
        load_daily(target),
        lookback=lookback,
        hold_days=hold_days,
        threshold=threshold,
        is_days=504,
        oos_days=126,
        spread_bps=5.0,
    )
    s = r["stitched_returns"]
    s.name = f"{bond.lower()}_{target.lower()}_lb{lookback}_h{hold_days}_t{int(threshold * 100):03d}"
    return s


def get_ml_returns(instrument: str, oos_months: int = 2) -> pd.Series:
    from research.auto.evaluate import run_ml_wfo

    cfg = dict(
        strategy="stacking",
        timeframe="D",
        xgb_params=dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            verbosity=0,
        ),
        lstm_hidden=32,
        lookback=20,
        lstm_epochs=30,
        n_nested_folds=3,
        label_params=[
            dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
            dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
            dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
        ],
        signal_threshold=0.6,
        cost_bps=2.0,
        is_years=2,
        oos_months=oos_months,
    )
    r = run_ml_wfo(instrument, cfg, return_raw=True)
    s = r.get("stitched_returns", pd.Series(dtype=float))
    s.name = f"ml_{instrument.lower()}"
    return s


# v4 candidate set -- 5 strategies in the correlation mix + ML overlay.
CANDIDATES = [
    # Channel 1: credit -> equity (new threshold)
    ("hyg_iwb_th025", get_bond_equity_returns, ("HYG", "IWB", 10, 20, 0.25)),
    # Channel 2: inflation -> credit (new hold + threshold)
    ("tip_hyg_h40_th025", get_bond_equity_returns, ("TIP", "HYG", 60, 40, 0.25)),
    # Channel 3: duration -> growth (sweep-optimal is th=0.50)
    ("tlt_qqq_lb10", get_bond_equity_returns, ("TLT", "QQQ", 10, 20, 0.50)),
    # MR FX diversifier
    ("mr_audjpy_a24", get_mr_audjpy_returns, (24,)),
    # Commodity diversifier (5% slot)
    ("ief_gld_lb60", get_bond_equity_returns, ("IEF", "GLD", 60, 20, 0.50)),
]

# Prior v3-param variants — run them side-by-side for a clean "sweep paid off" comparison.
V3_PARAM_VARIANTS = [
    ("hyg_iwb_th050 (v3)", get_bond_equity_returns, ("HYG", "IWB", 10, 20, 0.50)),
    ("tip_hyg_h20_th050 (v3)", get_bond_equity_returns, ("TIP", "HYG", 60, 20, 0.50)),
]

ML_OVERLAY = [
    ("ml_iwb", get_ml_returns, ("IWB",)),
]


# ── Helpers ────────────────────────────────────────────────────────────


def to_bday(s: pd.Series) -> pd.Series:
    s = s.copy()
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.index = s.index.normalize()
    s = s.groupby(level=0).sum()
    idx = pd.bdate_range(s.index.min(), s.index.max())
    return s.reindex(idx, fill_value=0.0)


def sharpe(rets: pd.Series) -> float:
    from titan.research.metrics import BARS_PER_YEAR
    from titan.research.metrics import sharpe as _sh

    return float(_sh(rets, periods_per_year=BARS_PER_YEAR["D"]))


def max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def boot_ci(rets: pd.Series) -> tuple[float, float]:
    from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci

    return bootstrap_sharpe_ci(rets, periods_per_year=BARS_PER_YEAR["D"], n_resamples=1000, seed=42)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  v4 portfolio — new sweep parameters")
    print("=" * 70)

    series: dict[str, pd.Series] = {}
    for name, fn, args in CANDIDATES + V3_PARAM_VARIANTS + ML_OVERLAY:
        print(f"\n  Loading {name} ...", end=" ", flush=True)
        try:
            s = fn(*args)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if s is None or len(s) < 20:
            print("skipped (insufficient data)")
            continue
        s = to_bday(s)
        series[name] = s
        print(
            f"OK  bars={len(s)}  sharpe={sharpe(s):+.3f}  "
            f"range={s.index[0].date()}->{s.index[-1].date()}"
        )

    # Print v3-vs-v4 per-strategy delta first.
    print("\n" + "=" * 70)
    print("  v3 parameters vs v4 parameters (per-strategy)")
    print("=" * 70)
    comparisons = [
        ("HYG->IWB", "hyg_iwb_th050 (v3)", "hyg_iwb_th025"),
        ("TIP->HYG", "tip_hyg_h20_th050 (v3)", "tip_hyg_h40_th025"),
    ]
    for label, v3_key, v4_key in comparisons:
        if v3_key not in series or v4_key not in series:
            continue
        v3s = series[v3_key]
        v4s = series[v4_key]
        print(f"\n  {label}")
        print(
            f"    v3  Sharpe={sharpe(v3s):+.3f}  CI={boot_ci(v3s)}  DD={max_dd(v3s) * 100:+.1f}%"
        )
        print(
            f"    v4  Sharpe={sharpe(v4s):+.3f}  CI={boot_ci(v4s)}  DD={max_dd(v4s) * 100:+.1f}%"
        )

    # Portfolio analysis on the v4 correlation mix (5 strategies).
    v4_keys = [n for n, _, _ in CANDIDATES if n in series]
    if len(v4_keys) < 2:
        print("\nERROR: <2 v4 components loaded")
        sys.exit(1)

    common_start = max(series[k].index[0] for k in v4_keys)
    common_end = min(series[k].index[-1] for k in v4_keys)
    aligned = {k: series[k][common_start:common_end] for k in v4_keys}
    n_days = len(next(iter(aligned.values())))
    print(f"\n  v4 common window: {common_start.date()} -> {common_end.date()} ({n_days} bdays)")

    df = pd.DataFrame(aligned)
    corr = df.corr()
    max_pw = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()

    print("\n  v4 correlation matrix:\n")
    print("  " + corr.round(3).to_string().replace("\n", "\n  "))
    print(f"\n  Max |pairwise correlation|: {max_pw:.3f}")

    # IS/OOS split.
    split = len(df) // 2
    is_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    is_vols = is_df.std()
    inv_vols = 1.0 / is_vols.clip(lower=1e-9)
    inv_weights = (inv_vols / inv_vols.sum()).to_dict()
    capped = {k: min(v, 0.40) for k, v in inv_weights.items()}
    total = sum(capped.values())
    capped = {k: v / total for k, v in capped.items()}

    equal_weights = {k: 1.0 / len(df.columns) for k in df.columns}

    # Hand-proposed v4 weights.
    v4_proposed_weights_pct = {
        "hyg_iwb_th025": 25,
        "tip_hyg_h40_th025": 20,
        "tlt_qqq_lb10": 20,
        "mr_audjpy_a24": 20,
        "ief_gld_lb60": 15,
    }
    tot = sum(v4_proposed_weights_pct.values())
    v4_weights = {k: v / tot for k, v in v4_proposed_weights_pct.items() if k in df.columns}

    def port_rets(w: dict[str, float]) -> pd.Series:
        w_arr = np.array([w.get(c, 0.0) for c in oos_df.columns])
        return pd.Series(oos_df.values @ w_arr, index=oos_df.index)

    print("\n" + "=" * 70)
    print("  OOS portfolio performance by allocation scheme")
    print("=" * 70)

    results = []
    for scheme, w in [
        ("Equal-weight", equal_weights),
        ("IS inv-vol (cap 40%)", capped),
        ("v4 proposed (hand)", v4_weights),
    ]:
        port = port_rets(w)
        sh = sharpe(port)
        dd = max_dd(port)
        ci_lo, ci_hi = boot_ci(port)
        ann_ret = float(port.mean() * 252)
        wstr = ", ".join(f"{k}={w.get(k, 0):.0%}" for k in df.columns if w.get(k, 0) > 0)
        print(f"\n  {scheme}:")
        print(f"    weights: {wstr}")
        print(f"    Sharpe  : {sh:+.3f}   CI=({ci_lo:+.3f}, {ci_hi:+.3f})")
        print(f"    Max DD  : {dd * 100:+.1f}%")
        print(f"    Ann ret : {ann_ret * 100:+.1f}%")
        results.append((scheme, w, sh, ci_lo, ci_hi, dd, ann_ret))

    # Standalone OOS.
    print("\n  Per-strategy standalone OOS Sharpe:\n")
    standalone: list[tuple[str, float, float, float, float]] = []
    for c in df.columns:
        s = oos_df[c]
        sh = sharpe(s)
        ci_lo, ci_hi = boot_ci(s)
        dd = max_dd(s) * 100
        standalone.append((c, sh, ci_lo, ci_hi, dd))
        print(f"    {c:<22}  Sharpe={sh:+.3f}  CI=({ci_lo:+.3f}, {ci_hi:+.3f})  DD={dd:+.1f}%")

    # ── Markdown ──────────────────────────────────────────────────────
    md: list[str] = ["# v4 Portfolio Backtest (post-Param-Sweep)", ""]
    md.append(
        f"**Common window**: {common_start.date()} -> {common_end.date()} ({n_days} business days)"
    )
    md.append(f"**Max |pairwise correlation|**: {max_pw:.3f}")
    md.append("")
    md.append("## v3 vs v4 parameter comparison")
    md.append("")
    md.append("| Strategy | Config | Sharpe | CI_lo | CI_hi | Max DD |")
    md.append("|---|---|---:|---:|---:|---:|")
    for label, v3_key, v4_key in comparisons:
        if v3_key not in series or v4_key not in series:
            continue
        for tag, key in (("v3", v3_key), ("v4", v4_key)):
            s = series[key]
            sh = sharpe(s)
            lo, hi = boot_ci(s)
            md.append(
                f"| {label} ({tag}) | {key} | {sh:+.3f} | {lo:+.3f} | "
                f"{hi:+.3f} | {max_dd(s) * 100:+.1f}% |"
            )
    md.append("")
    md.append("## Full v4 correlation matrix")
    md.append("")
    md.append("| | " + " | ".join(corr.columns) + " |")
    md.append("|---|" + "---:|" * len(corr.columns))
    for lbl, row in corr.round(3).iterrows():
        md.append(f"| {lbl} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    md.append("")
    md.append("## Per-strategy standalone OOS")
    md.append("")
    md.append("| Strategy | Sharpe | CI_lo | CI_hi | Max DD |")
    md.append("|---|---:|---:|---:|---:|")
    for c, sh, lo, hi, dd in standalone:
        md.append(f"| {c} | {sh:+.3f} | {lo:+.3f} | {hi:+.3f} | {dd:+.1f}% |")
    md.append("")
    md.append("## Portfolio allocations — realized OOS")
    md.append("")
    md.append("| Scheme | Sharpe | CI_lo | CI_hi | Max DD | Ann ret |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for scheme, w, sh, lo, hi, dd, ann in results:
        md.append(
            f"| {scheme} | {sh:+.3f} | {lo:+.3f} | {hi:+.3f} | "
            f"{dd * 100:+.1f}% | {ann * 100:+.1f}% |"
        )
    md.append("")
    md.append("## ML IWB overlay (standalone only — sparse trades)")
    md.append("")
    if "ml_iwb" in series:
        s = series["ml_iwb"]
        md.append(
            f"Sharpe={sharpe(s):+.3f}  CI={boot_ci(s)}  "
            f"DD={max_dd(s) * 100:+.1f}%  bars={len(s)}"
        )
    else:
        md.append("ML IWB not loaded.")
    md.append("")
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Report: {REPORT}")


if __name__ == "__main__":
    main()
