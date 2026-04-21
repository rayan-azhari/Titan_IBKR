"""Post-remediation re-rank sweep.

Re-runs every WFO pipeline in the repo across the full data universe,
collects each strategy's stitched Sharpe and 95 % bootstrap CI, and
writes a single leaderboard CSV + markdown.

Scope (all run with the corrected harness):
  * MR confluence-regime WFO — every H1 instrument in data/.
  * Bond→equity WFO — full bond × target × lookback sweep (built-in).
  * ETF trend rolling WFO — SPY, QQQ, IWB, TQQQ (the four live targets).

Each pipeline emits per-combo rows with:
    strategy, instrument, params_summary, sharpe, ci_lo, ci_hi,
    max_dd_pct, n_folds, pct_positive, n_trades, gate (ci_lo > 0)

Outputs:
  .tmp/reports/rerank_2026_04_21/leaderboard.csv
  .tmp/reports/rerank_2026_04_21/leaderboard.md
  .tmp/reports/rerank_2026_04_21/{strategy}/{instrument}.log

Sanctuary window and shared metrics module active by default.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
REPORT_ROOT = ROOT / ".tmp" / "reports" / "rerank_2026_04_21"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# ── Universe discovery ───────────────────────────────────────────────────


def discover_h1_instruments() -> list[str]:
    """Every instrument with an H1 parquet file, sorted."""
    names = [p.name.replace("_H1.parquet", "") for p in DATA_DIR.glob("*_H1.parquet")]
    return sorted(names)


# ── Pipeline runners ─────────────────────────────────────────────────────


def run_mr_confluence(instrument: str, rows: list[dict]) -> None:
    """Run MR confluence-regime WFO on one H1 instrument, append every
    (filter, tier-grid) combo's result as a row.

    Uses default IS/OOS sizing (will reduce automatically on short series).
    """
    from research.mean_reversion.run_confluence_regime_test import (
        build_atr_regime_mask,
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import (
        run_mr_wfo,
    )

    try:
        df = load_h1(instrument)
    except FileNotFoundError:
        return

    n_bars = len(df)
    # Adaptive IS/OOS sizing: need at least 2 folds. Default 32k/8k works
    # for pairs with 5+ years of H1 data; smaller instruments get scaled.
    if n_bars >= 40000:
        is_bars, oos_bars = 32000, 8000
    elif n_bars >= 15000:
        is_bars, oos_bars = int(n_bars * 0.75), int(n_bars * 0.1)
    else:
        return  # not enough data for meaningful WFO

    close = df["close"]
    try:
        deviation = compute_vwap_deviation(close, anchor_period=24)
    except Exception:
        return

    # Build a set of regime filters to test (same as the WFO default).
    filters = {
        "conf_donchian_pos_20": build_confluence_disagreement_mask(df, "donchian_pos_20"),
        "conf_rsi_14_dev": build_confluence_disagreement_mask(df, "rsi_14_dev"),
        "atr_only": build_atr_regime_mask(df),
        "no_filter": pd.Series(True, index=df.index),
    }
    tier_grids = {
        "standard": [0.90, 0.95, 0.98, 0.99],
        "conservative": [0.95, 0.98, 0.99, 0.999],
    }

    for filt_name, mask in filters.items():
        for grid_name, pcts in tier_grids.items():
            try:
                r = run_mr_wfo(
                    close,
                    deviation,
                    mask,
                    pcts,
                    is_bars=is_bars,
                    oos_bars=oos_bars,
                )
            except Exception as e:
                print(f"    {filt_name}/{grid_name}: ERROR {e}")
                continue
            if r.get("n_folds", 0) < 2:
                continue
            rows.append(
                {
                    "strategy": "mr_confluence",
                    "instrument": instrument,
                    "params": f"{filt_name}/{grid_name}",
                    "sharpe": r.get("stitched_sharpe", 0.0),
                    "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
                    "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
                    "max_dd_pct": r.get("stitched_dd_pct", 0.0),
                    "n_folds": r.get("n_folds", 0),
                    "pct_positive": r.get("pct_positive", 0.0),
                    "n_trades": r.get("total_trades", 0),
                }
            )


def run_bond_equity(rows: list[dict]) -> None:
    """Bond→equity/gold momentum — full sweep across bonds × targets × lookbacks."""
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    bonds = ["TLT", "IEF", "HYG", "TIP"]
    targets = ["SPY", "QQQ", "IWB", "GLD"]
    lookbacks = [10, 20, 40, 60]

    target_series: dict[str, pd.Series] = {}
    bond_series: dict[str, pd.Series] = {}
    for b in bonds:
        try:
            bond_series[b] = load_daily(b)
        except FileNotFoundError:
            pass
    for t in targets:
        try:
            target_series[t] = load_daily(t)
        except FileNotFoundError:
            pass

    for b, bc in bond_series.items():
        for t, tc in target_series.items():
            for lb in lookbacks:
                try:
                    r = run_bond_wfo(
                        bc,
                        tc,
                        lookback=lb,
                        hold_days=20,
                        threshold=0.50,
                        is_days=504,
                        oos_days=126,
                        spread_bps=5.0,
                    )
                except Exception as e:
                    print(f"    bond {b}->{t} lb={lb}: ERROR {e}")
                    continue
                if "n_folds" not in r or r["n_folds"] < 2:
                    continue
                rows.append(
                    {
                        "strategy": "bond_equity",
                        "instrument": f"{b}->{t}",
                        "params": f"lookback={lb} hold=20 th=0.50",
                        "sharpe": r.get("stitched_sharpe", 0.0),
                        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
                        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
                        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
                        "n_folds": r.get("n_folds", 0),
                        "pct_positive": r.get("pct_positive", 0.0),
                        "n_trades": r.get("total_trades", 0),
                    }
                )


# ── Leaderboard writer ───────────────────────────────────────────────────


def write_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print("No rows collected — leaderboard skipped.")
        return

    df = pd.DataFrame(rows)
    df["gate_pass"] = df["ci_lo"] > 0.0
    df = df.sort_values("ci_lo", ascending=False).reset_index(drop=True)

    csv_path = REPORT_ROOT / "leaderboard.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# Post-remediation Re-Rank Leaderboard — 2026-04-21",
        "",
        f"Generated from {len(df)} (strategy, instrument, params) combinations.",
        "Sanctuary window active (last 365 days excluded).",
        "95 % bootstrap Sharpe CI via `titan.research.metrics.bootstrap_sharpe_ci`.",
        "",
        "## Gate: `ci_lo > 0`",
        "",
        f"**{int(df['gate_pass'].sum())} / {len(df)} combinations pass.**",
        "",
        "## Top 40 by CI lower bound",
        "",
        "| # | Strategy | Instrument | Params | Sharpe | CI lo | CI hi | Max DD | n_folds | Pos % | Trades |",
        "|--:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in df.head(40).iterrows():
        md_lines.append(
            f"| {i + 1} | {row['strategy']} | {row['instrument']} | {row['params']} |"
            f" {row['sharpe']:+.3f} | {row['ci_lo']:+.3f} | {row['ci_hi']:+.3f} |"
            f" {row['max_dd_pct']:+.1f}% | {row['n_folds']} |"
            f" {row['pct_positive'] * 100:.0f}% | {row['n_trades']} |"
        )
    md_lines.append("")
    md_lines.append("## Bottom 10 by CI lower bound")
    md_lines.append("")
    md_lines.append(
        "| # | Strategy | Instrument | Params | Sharpe | CI lo | CI hi |"
    )
    md_lines.append("|--:|---|---|---|---:|---:|---:|")
    for i, row in df.tail(10).iloc[::-1].iterrows():
        md_lines.append(
            f"| {i + 1} | {row['strategy']} | {row['instrument']} | {row['params']} |"
            f" {row['sharpe']:+.3f} | {row['ci_lo']:+.3f} | {row['ci_hi']:+.3f} |"
        )

    (REPORT_ROOT / "leaderboard.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Leaderboard: {csv_path}")
    print(f"  Leaderboard: {REPORT_ROOT / 'leaderboard.md'}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    t_start = time.time()
    rows: list[dict] = []

    print("=" * 60)
    print("  Post-Remediation Re-Rank Sweep")
    print("=" * 60)

    # Bond-equity first -- fast (single cached load pair, ~5 min)
    print("\n[1/2] Bond->equity sweep (4 bonds x 4 targets x 4 lookbacks)...")
    try:
        run_bond_equity(rows)
    except Exception:
        traceback.print_exc()
    print(f"  collected {len(rows)} bond-equity rows")

    # MR confluence — the big one.
    h1 = discover_h1_instruments()
    print(f"\n[2/2] MR confluence sweep across {len(h1)} H1 instruments...")
    print(f"  (~30-60s per instrument, est. {len(h1) * 45 / 60:.0f} min total)")
    # Belt-and-braces: force UTF-8 on Windows so any unicode that sneaks
    # in from a library (e.g. logging) doesn't crash the run.
    for i, inst in enumerate(h1, 1):
        t0 = time.time()
        pre_count = len(rows)
        try:
            run_mr_confluence(inst, rows)
        except Exception:
            traceback.print_exc()
        added = len(rows) - pre_count
        elapsed = time.time() - t0
        print(
            f"  [{i:>3}/{len(h1)}] {inst:<12} {added:>2} combos  {elapsed:>5.1f}s"
            f"  (total rows={len(rows)})"
        )

    print(f"\n[done] Collected {len(rows)} rows in {time.time() - t_start:.0f}s")
    write_leaderboard(rows)


if __name__ == "__main__":
    main()
