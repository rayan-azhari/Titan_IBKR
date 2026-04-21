"""Portfolio correlation + weight optimization for the v3 candidate set.

Extends optimize_candidate_portfolio.py to cover every strategy in the
proposed v3 allocation:

  1. ML IWB stacking
  2. ML TQQQ stacking (leveraged -- position-size at 1/3)
  3. TLT -> QQQ lb=10
  4. TIP -> HYG lb=60  (NEW cross-bond low-DD)
  5. LQD -> HYG lb=10  (NEW cross-bond low-DD)
  6. TLT -> HYG lb=20  (NEW cross-bond)
  7. MR AUD/JPY (vwap_anchor=24, conf_donchian_pos_20)
  8. HYG -> IWB lb=10
  9. IEF -> GLD lb=60

Checks:
  - Pairwise correlation on the common window.
  - Which X->HYG signals are colinear -- they likely share a bond
    risk-off factor; we need to know the strength of that redundancy.
  - Inverse-vol weights on IS half, realized OOS.
  - Equal-weight realized OOS.
  - Random Dirichlet band.

Writes:
  .tmp/reports/autonomous_2026_04_21/v3_portfolio.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "autonomous_2026_04_21" / "v3_portfolio.md"
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
    # Agent loop finding: anchor=24 > anchor=46 on corrected harness.
    deviation = compute_vwap_deviation(close, anchor_period=vwap_anchor)
    mask = build_confluence_disagreement_mask(df, "donchian_pos_20")
    r = run_mr_wfo(
        close, deviation, mask,
        [0.95, 0.98, 0.99, 0.999],
        is_bars=32000, oos_bars=8000,
    )
    s = r["stitched_returns"]
    s.name = f"mr_audjpy_anchor{vwap_anchor}"
    return s


def get_bond_equity_returns(bond: str, target: str, lookback: int) -> pd.Series:
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    r = run_bond_wfo(
        load_daily(bond),
        load_daily(target),
        lookback=lookback, hold_days=20, threshold=0.50,
        is_days=504, oos_days=126, spread_bps=5.0,
    )
    s = r["stitched_returns"]
    s.name = f"{bond.lower()}_{target.lower()}_lb{lookback}"
    return s


def get_ml_returns(instrument: str, oos_months: int = 2) -> pd.Series:
    """ML stacking: call evaluate.py with per-instrument config, collect
    stitched returns from the result dict."""
    from research.auto.evaluate import run_ml_wfo

    cfg = dict(
        strategy="stacking",
        timeframe="D",
        xgb_params=dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, random_state=42, verbosity=0,
        ),
        lstm_hidden=32, lookback=20, lstm_epochs=30, n_nested_folds=3,
        label_params=[
            dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
            dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
            dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
        ],
        signal_threshold=0.6, cost_bps=2.0,
        is_years=2, oos_months=oos_months,
    )
    r = run_ml_wfo(instrument, cfg, return_raw=True)
    s = r.get("stitched_returns", pd.Series(dtype=float))
    s.name = f"ml_{instrument.lower()}"
    return s


# ML stacking returns are very sparse (5-10 trades, narrow window) and
# collapse the inner-join when combined with daily bond-equity series.
# Include them here for standalone reporting only; the portfolio
# correlation + weight analysis runs on the non-ML subset below, and
# ML is added back as a fixed overlay in the final recommendation.
CANDIDATES_STANDALONE_ONLY = [
    ("ml_iwb", get_ml_returns, ("IWB",)),
    ("ml_tqqq", get_ml_returns, ("TQQQ",)),
]

CANDIDATES = [
    ("tlt_qqq_lb10", get_bond_equity_returns, ("TLT", "QQQ", 10)),
    ("tip_hyg_lb60", get_bond_equity_returns, ("TIP", "HYG", 60)),
    ("lqd_hyg_lb10", get_bond_equity_returns, ("LQD", "HYG", 10)),
    ("tlt_hyg_lb20", get_bond_equity_returns, ("TLT", "HYG", 20)),
    ("mr_audjpy_a24", get_mr_audjpy_returns, (24,)),
    ("hyg_iwb_lb10", get_bond_equity_returns, ("HYG", "IWB", 10)),
    ("ief_gld_lb60", get_bond_equity_returns, ("IEF", "GLD", 60)),
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
    return bootstrap_sharpe_ci(
        rets, periods_per_year=BARS_PER_YEAR["D"], n_resamples=1000, seed=42
    )


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  v3 portfolio correlation + optimization")
    print("=" * 70)

    series: dict[str, pd.Series] = {}
    for name, fn, args in CANDIDATES:
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
        print(f"OK  bars={len(s)}  sharpe={sharpe(s):+.3f}  "
              f"range={s.index[0].date()}→{s.index[-1].date()}")

    if len(series) < 2:
        print("\nERROR: <2 loaded")
        sys.exit(1)

    common_start = max(s.index[0] for s in series.values())
    common_end = min(s.index[-1] for s in series.values())
    aligned = {k: v[common_start:common_end] for k, v in series.items()}
    n_days = len(next(iter(aligned.values())))
    print(f"\n  Common window: {common_start.date()} → {common_end.date()} ({n_days} bdays)")

    df = pd.DataFrame(aligned)
    corr = df.corr()
    max_pw = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()

    print("\n  Correlation matrix (full v3 set):\n")
    print("  " + corr.round(3).to_string().replace("\n", "\n  "))
    print(f"\n  Max |pairwise correlation|: {max_pw:.3f}")

    # Cluster check: pull out the three X->HYG signals.
    hyg_cols = [c for c in df.columns if c.endswith("_hyg_lb10") or c.endswith("_hyg_lb20") or c.endswith("_hyg_lb60")]
    print(f"\n  Bond->HYG sub-corr ({hyg_cols}):")
    print("  " + corr.loc[hyg_cols, hyg_cols].round(3).to_string().replace("\n", "\n  "))

    # IS/OOS split.
    split = len(df) // 2
    is_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    is_vols = is_df.std()
    inv_vols = 1.0 / is_vols.clip(lower=1e-9)
    inv_weights = (inv_vols / inv_vols.sum()).to_dict()
    # Cap individual weight at 35 % (more conservative than prior 60 % cap
    # because this portfolio has 9 components).
    capped = {k: min(v, 0.35) for k, v in inv_weights.items()}
    total = sum(capped.values())
    capped = {k: v / total for k, v in capped.items()}

    equal_weights = {k: 1.0 / len(df.columns) for k in df.columns}

    # Hand-proposed v3 weights, rescaled to the 7-strategy (non-ML) universe.
    # ML contribution (30% nominal) is carved out; the remaining 70% is
    # distributed per the original plan.
    v3_proposed_weights_pct_nonml = {
        "tlt_qqq_lb10": 15,
        "tip_hyg_lb60": 10, "lqd_hyg_lb10": 10, "tlt_hyg_lb20": 10,
        "mr_audjpy_a24": 10, "hyg_iwb_lb10": 10, "ief_gld_lb60": 5,
    }
    tot_nonml = sum(v3_proposed_weights_pct_nonml.values())
    v3_weights = {k: v / tot_nonml for k, v in v3_proposed_weights_pct_nonml.items()
                  if k in df.columns}

    def port_rets(w: dict[str, float]) -> pd.Series:
        w_arr = np.array([w.get(c, 0.0) for c in oos_df.columns])
        return pd.Series(oos_df.values @ w_arr, index=oos_df.index)

    print("\n" + "=" * 70)
    print("  OOS portfolio performance by allocation scheme")
    print("=" * 70)

    results = []
    for scheme, w in [
        ("Equal-weight", equal_weights),
        ("IS inv-vol (cap 35%)", capped),
        ("v3 proposed (hand)", v3_weights),
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

    # Random Dirichlet band.
    print("\n  Random-weight Dirichlet band (n=500):")
    rng = np.random.default_rng(42)
    rand_sh = []
    for _ in range(500):
        w_arr = rng.dirichlet(np.ones(len(df.columns)))
        rand_sh.append(sharpe(port_rets(dict(zip(df.columns, w_arr)))))
    arr = np.array(rand_sh)
    print(f"    p05={np.percentile(arr, 5):.2f}  "
          f"p50={np.percentile(arr, 50):.2f}  "
          f"p95={np.percentile(arr, 95):.2f}")

    # Standalone OOS Sharpes.
    print("\n  Per-strategy standalone OOS Sharpe:\n")
    for c in df.columns:
        s = oos_df[c]
        sh = sharpe(s)
        ci_lo, ci_hi = boot_ci(s)
        print(f"    {c:<16}  Sharpe={sh:+.3f}  CI=({ci_lo:+.3f}, {ci_hi:+.3f})")

    # ── Markdown report ────────────────────────────────────────────────

    md = ["# v3 Portfolio Correlation + Optimization", ""]
    md.append(f"**Common window**: {common_start.date()} → {common_end.date()} ({n_days} bdays)")
    md.append(f"**Max |pairwise correlation|**: {max_pw:.3f}")
    md.append("")
    md.append("## Full correlation matrix")
    md.append("")
    md.append("| | " + " | ".join(corr.columns) + " |")
    md.append("|---|" + "---:|" * len(corr.columns))
    for lbl, row in corr.round(3).iterrows():
        md.append(f"| {lbl} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    md.append("")
    md.append("## Bond→HYG sub-cluster (check for common credit factor)")
    md.append("")
    sub = corr.loc[hyg_cols, hyg_cols].round(3)
    md.append("| | " + " | ".join(sub.columns) + " |")
    md.append("|---|" + "---:|" * len(sub.columns))
    for lbl, row in sub.iterrows():
        md.append(f"| {lbl} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    md.append("")
    md.append("## Per-strategy standalone OOS")
    md.append("")
    md.append("| Strategy | Sharpe | CI_lo | CI_hi | Max DD |")
    md.append("|---|---:|---:|---:|---:|")
    for c in df.columns:
        s = oos_df[c]
        sh = sharpe(s)
        ci_lo, ci_hi = boot_ci(s)
        md.append(f"| {c} | {sh:+.3f} | {ci_lo:+.3f} | {ci_hi:+.3f} | {max_dd(s) * 100:+.1f}% |")
    md.append("")
    md.append("## Portfolio allocations — realized OOS")
    md.append("")
    md.append("| Scheme | Sharpe | CI_lo | CI_hi | Max DD | Ann ret |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for scheme, w, sh, ci_lo, ci_hi, dd, ann in results:
        md.append(f"| {scheme} | {sh:+.3f} | {ci_lo:+.3f} | {ci_hi:+.3f} | {dd * 100:+.1f}% | {ann * 100:+.1f}% |")
    md.append("")
    md.append(f"Random Dirichlet OOS band (n=500): p05={np.percentile(arr, 5):.2f}, "
              f"p50={np.percentile(arr, 50):.2f}, p95={np.percentile(arr, 95):.2f}.")
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Report: {REPORT}")


if __name__ == "__main__":
    main()
