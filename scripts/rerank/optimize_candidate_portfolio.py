"""Candidate portfolio optimization on the corrected harness.

Builds a portfolio from the top gate-passers discovered in the
post-remediation re-rank (2026-04-21):

  MR AUD/JPY conf_donchian_pos_20/conservative   — existing champion
  TLT->QQQ lookback=10                           — NEW (best CI_lo)
  HYG->QQQ lookback=10                           — NEW
  TIP->QQQ lookback=60                           — NEW (independent family)
  HYG->IWB lookback=10                           — existing champion
  IEF->GLD lookback=60                           — existing champion

For each candidate, pulls ``stitched_returns`` from the corrected WFO
runner, aligns on the common date range, computes:

  - pairwise correlation (check for bond-signal clustering)
  - inverse-vol weights on IS half, realized OOS
  - equal-weight realized OOS
  - random-weight Dirichlet band (sensitivity)
  - portfolio Sharpe / DD / Calmar with CI

Output: .tmp/reports/rerank_2026_04_21/candidate_portfolio.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "rerank_2026_04_21" / "candidate_portfolio.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)


# ── Candidate definitions ───────────────────────────────────────────────


def get_mr_audjpy_returns() -> pd.Series:
    from research.mean_reversion.run_confluence_regime_test import (
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import run_mr_wfo

    df = load_h1("AUD_JPY")
    close = df["close"]
    deviation = compute_vwap_deviation(close, anchor_period=24)
    mask = build_confluence_disagreement_mask(df, "donchian_pos_20")
    r = run_mr_wfo(
        close,
        deviation,
        mask,
        [0.95, 0.98, 0.99, 0.999],  # conservative
        is_bars=32000,
        oos_bars=8000,
    )
    s = r["stitched_returns"]
    s.name = "mr_audjpy"
    return s


def get_bond_equity_returns(bond: str, target: str, lookback: int) -> pd.Series:
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    r = run_bond_wfo(
        load_daily(bond),
        load_daily(target),
        lookback=lookback,
        hold_days=20,
        threshold=0.50,
        is_days=504,
        oos_days=126,
        spread_bps=5.0,
    )
    s = r["stitched_returns"]
    s.name = f"{bond}_{target}_lb{lookback}"
    return s


CANDIDATES = [
    ("mr_audjpy", get_mr_audjpy_returns, ()),
    ("tlt_qqq_lb10", get_bond_equity_returns, ("TLT", "QQQ", 10)),
    ("hyg_qqq_lb10", get_bond_equity_returns, ("HYG", "QQQ", 10)),
    ("tip_qqq_lb60", get_bond_equity_returns, ("TIP", "QQQ", 60)),
    ("hyg_iwb_lb10", get_bond_equity_returns, ("HYG", "IWB", 10)),
    ("ief_gld_lb60", get_bond_equity_returns, ("IEF", "GLD", 60)),
]

# Deduplicated set: drop HYG->QQQ because it's 0.93 correlated with HYG->IWB
# (same bond signal, near-identical equity target). Keeps one per distinct
# factor: FX MR (audjpy), duration (TLT->QQQ), inflation (TIP->QQQ), credit
# (HYG->IWB), gold (IEF->GLD).
DEDUPED_NAMES = {"mr_audjpy", "tlt_qqq_lb10", "tip_qqq_lb60", "hyg_iwb_lb10", "ief_gld_lb60"}


# ── Portfolio helpers ───────────────────────────────────────────────────


def to_business_day_series(s: pd.Series) -> pd.Series:
    """Normalize any return series to business-day calendar with zeros for
    non-trade days. The MR AUD/JPY series has H1 timestamps; the bond ones
    have daily. We want a common business-day axis for all."""
    s = s.copy()
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.index = s.index.normalize()
    # Sum within each day (multiple H1 bars collapse to one row per day)
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
    print("  Candidate Portfolio Optimization — Post-Remediation")
    print("=" * 70)

    # Step 1: load every candidate's OOS stitched returns.
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
        s = to_business_day_series(s)
        series[name] = s
        print(
            f"OK  bars={len(s)}  sharpe={sharpe(s):+.3f}  "
            f"range={s.index[0].date()}→{s.index[-1].date()}"
        )

    if len(series) < 2:
        print("\nERROR: <2 candidates loaded, cannot build portfolio.")
        sys.exit(1)

    # Step 2: align on common date range.
    common_start = max(s.index[0] for s in series.values())
    common_end = min(s.index[-1] for s in series.values())
    aligned = {k: v[common_start:common_end] for k, v in series.items()}
    n_days = len(next(iter(aligned.values())))
    print(f"\n  Common date range: {common_start.date()} → {common_end.date()}  ({n_days} bdays)")

    # Write markdown incrementally so partial failures don't lose results.
    md: list[str] = ["# Candidate Portfolio Optimization — 2026-04-21", ""]
    md.append("Post-remediation portfolio construction using the top gate-passers "
              "from the re-rank sweep. Evaluates two candidate sets: the full "
              "6-strategy set and a de-duplicated 5-strategy set that drops "
              "HYG->QQQ (0.93-correlated with HYG->IWB).")
    md.append("")
    md.append(f"**Common window**: {common_start.date()} → {common_end.date()} ({n_days} bdays)")
    md.append("")

    # Step 3: evaluate both the full set and the deduplicated set.
    for set_name, selected in [
        ("full (6 strategies)", list(aligned.keys())),
        ("deduplicated (5 strategies, drop HYG->QQQ)",
         [k for k in aligned.keys() if k in DEDUPED_NAMES]),
    ]:
        print("\n" + "=" * 70)
        print(f"  SET: {set_name}")
        print("=" * 70)

        df = pd.DataFrame({k: aligned[k] for k in selected})
        corr = df.corr()
        print("\n  Correlation matrix:\n")
        print("  " + corr.round(3).to_string().replace("\n", "\n  "))
        max_pw = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
        print(f"\n  Max |pairwise corr|: {max_pw:.3f}")

        # IS / OOS split for weight derivation + realized OOS.
        split = len(df) // 2
        is_df = df.iloc[:split]
        oos_df = df.iloc[split:]

        is_vols = is_df.std()
        inv_vols = 1.0 / is_vols.clip(lower=1e-9)
        inv_weights = (inv_vols / inv_vols.sum()).to_dict()
        capped = {k: min(v, 0.60) for k, v in inv_weights.items()}
        total = sum(capped.values())
        capped = {k: v / total for k, v in capped.items()}
        equal_weights = {k: 1.0 / len(df.columns) for k in df.columns}

        def port_rets(w: dict[str, float], _oos=oos_df) -> pd.Series:
            w_arr = np.array([w[c] for c in _oos.columns])
            return pd.Series(_oos.values @ w_arr, index=_oos.index)

        md.append(f"## {set_name}")
        md.append("")
        md.append(f"Max |pairwise correlation|: {max_pw:.3f}")
        md.append("")
        md.append("### Correlation matrix")
        md.append("")
        md.append("| | " + " | ".join(corr.columns) + " |")
        md.append("|---|" + "---:|" * len(corr.columns))
        for idx_label, row in corr.round(3).iterrows():
            md.append(f"| {idx_label} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
        md.append("")
        md.append("### Per-strategy standalone OOS")
        md.append("")
        md.append("| Strategy | Sharpe | CI_lo | CI_hi | Max DD |")
        md.append("|---|---:|---:|---:|---:|")
        print("\n  Per-strategy standalone OOS Sharpe:\n")
        for c in df.columns:
            s = oos_df[c]
            sh = sharpe(s)
            ci_lo, ci_hi = boot_ci(s)
            print(
                f"    {c:<18}  Sharpe={sh:+.3f}  CI=({ci_lo:+.3f}, {ci_hi:+.3f})  "
                f"maxDD={max_dd(s) * 100:+.1f}%"
            )
            md.append(
                f"| {c} | {sh:+.3f} | {ci_lo:+.3f} | {ci_hi:+.3f} | "
                f"{max_dd(s) * 100:+.1f}% |"
            )
        md.append("")
        md.append("### Portfolio allocations — realized OOS")
        md.append("")
        md.append("| Scheme | Weights | Sharpe | CI_lo | CI_hi | Max DD | Ann ret |")
        md.append("|---|---|---:|---:|---:|---:|---:|")

        print("\n  Portfolio allocations — realized OOS:")
        for scheme, w in [
            ("Equal-weight", equal_weights),
            ("IS inv-vol (cap 60%)", capped),
        ]:
            port = port_rets(w)
            sh = sharpe(port)
            dd = max_dd(port)
            ci_lo, ci_hi = boot_ci(port)
            ann_ret = float(port.mean() * 252)
            wstr = ", ".join(f"{k}={w[k]:.0%}" for k in df.columns)
            print(f"\n  {scheme}:")
            print(f"    weights: {wstr}")
            print(f"    Sharpe  : {sh:+.3f}   CI=({ci_lo:+.3f}, {ci_hi:+.3f})")
            print(f"    Max DD  : {dd * 100:+.1f}%")
            print(f"    Ann ret : {ann_ret * 100:+.1f}%")
            md.append(
                f"| {scheme} | {wstr} | {sh:+.3f} | {ci_lo:+.3f} | {ci_hi:+.3f} | "
                f"{dd * 100:+.1f}% | {ann_ret * 100:+.1f}% |"
            )

        # Random-weight sensitivity.
        rng = np.random.default_rng(42)
        rand_sharpes = []
        for _ in range(500):
            w_arr = rng.dirichlet(np.ones(len(df.columns)))
            rand_sharpes.append(sharpe(port_rets(dict(zip(df.columns, w_arr)))))
        arr = np.array(rand_sharpes)
        print(
            f"\n  Random Dirichlet band: "
            f"p05={np.percentile(arr, 5):.2f}  "
            f"p50={np.percentile(arr, 50):.2f}  "
            f"p95={np.percentile(arr, 95):.2f}"
        )
        md.append("")
        md.append(
            f"Random-weight Dirichlet band (n=500): "
            f"p05={np.percentile(arr, 5):.2f}, "
            f"p50={np.percentile(arr, 50):.2f}, "
            f"p95={np.percentile(arr, 95):.2f}."
        )
        md.append("")

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Report: {REPORT}")


if __name__ == "__main__":
    main()
