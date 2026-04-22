"""Phase 1 of the EU strategy plan — "free" cross-asset sweeps.

5 channels, all using already-downloaded data, via the existing WFO
harness (research.cross_asset.run_bond_equity_wfo.run_bond_wfo). Each
channel is a full parameter grid (6 lookbacks x 4 holds x 4 thresholds
= 96 combos).

Channels:
  1. UUP   -> EUR/USD     Dollar strength leads EUR/USD reversion
  2. CL=F  -> ^GDAXI      Oil shocks transmit to Europe via energy costs
  3. GLD   -> EUR/USD     Safe-haven rotation; gold rallies often
                          coincide with USD weakness
  4. IGOV-IEF synthetic spread -> EUR/USD
                          International rates premium vs US rates
                          predicts FX
  5. DAX/SPY ratio -> ^GDAXI
                      Relative-momentum: when DAX outperforms SPY,
                      does DAX continue to outperform locally?

Writes:
  .tmp/reports/phase1_eu_native_2026_04_22/results.csv
  .tmp/reports/phase1_eu_native_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "phase1_eu_native_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

# 5 channels x 96 combos = 480 total. Bonferroni at this N requires
# a slightly less-strict CI_lo (the gate is ~2x CI-widening factor).
BONF_CI_LO = 0.45
BONF_MIN_FOLDS = 25
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0


# ── Signal builders ────────────────────────────────────────────────────


def load_series(sym: str) -> pd.Series:
    from research.cross_asset.run_bond_equity_wfo import load_daily

    s = load_daily(sym)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def build_spread(sym_a: str, sym_b: str) -> pd.Series:
    """IGOV - IEF: build a log-spread that behaves like a price series
    for the WFO harness. Not a true tradable but a clean z-scorable
    signal of international-vs-US rate momentum."""
    a = load_series(sym_a)
    b = load_series(sym_b)
    common = a.index.intersection(b.index)
    a = a.reindex(common).ffill()
    b = b.reindex(common).ffill()
    # Price-like spread: ratio, log. Stays positive and tradable-by-ratio.
    s = (a / b).astype(float)
    s.name = f"{sym_a}_over_{sym_b}"
    return s


def build_ratio(sym_num: str, sym_den: str) -> pd.Series:
    a = load_series(sym_num)
    b = load_series(sym_den)
    common = a.index.intersection(b.index)
    a = a.reindex(common).ffill()
    b = b.reindex(common).ffill()
    s = (a / b).astype(float)
    s.name = f"{sym_num}_over_{sym_den}"
    return s


# ── Runner ─────────────────────────────────────────────────────────────


def run_channel(
    name: str, signal: pd.Series, target: pd.Series, story: str
) -> list[dict]:
    from research.cross_asset.run_bond_equity_wfo import run_bond_wfo

    rows: list[dict] = []
    for lb in LOOKBACKS:
        for hold in HOLDS:
            for th in THRESHOLDS:
                try:
                    r = run_bond_wfo(
                        signal,
                        target,
                        lookback=lb,
                        hold_days=hold,
                        threshold=th,
                        is_days=504,
                        oos_days=126,
                        spread_bps=5.0,
                    )
                except Exception as e:
                    print(f"    ERROR: {name} lb={lb} hold={hold} th={th}: {e}")
                    continue
                if r.get("n_folds", 0) < 5:
                    continue
                rows.append(
                    {
                        "channel": name,
                        "lookback": lb,
                        "hold": hold,
                        "threshold": th,
                        "sharpe": r.get("stitched_sharpe", 0.0),
                        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
                        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
                        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
                        "n_folds": r.get("n_folds", 0),
                        "pct_positive": r.get("pct_positive", 0.0),
                        "n_trades": r.get("total_trades", 0),
                        "story": story,
                    }
                )
    print(f"  [{name}] {len(rows)} rows (story: {story})")
    return rows


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= BONF_MIN_FOLDS
        and r["pct_positive"] >= BONF_MIN_POS
        and r["max_dd_pct"] >= BONF_MAX_DD
    )


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  Phase 1: EU-native free cross-asset sweeps")
    print("=" * 70)
    t0 = time.time()

    # Load base series
    print("\n  Loading series ...")
    uup = load_series("UUP")
    eurusd = load_series("EUR_USD")
    cl = load_series("CL=F")
    gdaxi = load_series("GDAXI")
    gld = load_series("GLD")
    spy = load_series("SPY")
    spread_igov_ief = build_spread("IGOV", "IEF")
    ratio_gdaxi_spy = build_ratio("GDAXI", "SPY")

    rows: list[dict] = []

    # Channel 1: UUP -> EUR/USD (dollar strength -> EUR weakness)
    rows.extend(run_channel(
        "UUP->EURUSD", uup, eurusd,
        "Dollar strength -> EUR/USD momentum (USD up -> EUR down)",
    ))

    # Channel 2: CL=F -> ^GDAXI (oil -> German equity via energy imports)
    rows.extend(run_channel(
        "CL->GDAXI", cl, gdaxi,
        "Oil shock -> DAX (Germany is a net energy importer)",
    ))

    # Channel 3: GLD -> EUR/USD (safe-haven rotation; gold-up = USD-down)
    rows.extend(run_channel(
        "GLD->EURUSD", gld, eurusd,
        "Gold rally -> USD weakness -> EUR/USD up (safe-haven)",
    ))

    # Channel 4: IGOV-IEF spread -> EUR/USD
    rows.extend(run_channel(
        "IGOV_IEF->EURUSD", spread_igov_ief, eurusd,
        "International-vs-US rate premium -> EUR/USD (rate differential)",
    ))

    # Channel 5: DAX/SPY relative -> ^GDAXI
    rows.extend(run_channel(
        "DAX_SPY_ratio->GDAXI", ratio_gdaxi_spy, gdaxi,
        "DAX/SPY momentum -> DAX (relative-strength continuation)",
    ))

    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Total combos: {len(df)}, wall-clock: {time.time() - t0:.0f}s")

    perm = df[df["ci_lo"] > 0]
    bonf = df[df.apply(passes_bonferroni, axis=1)] if not df.empty else df
    print(f"  Permissive (ci_lo > 0): {len(perm)}")
    print(f"  Bonferroni (ci_lo >= {BONF_CI_LO}): {len(bonf)}")

    # Leaderboard
    lines: list[str] = [
        f"# Phase 1 — EU-Native Free Cross-Asset Sweeps ({len(df)} combos)",
        "",
        "5 channels using already-downloaded US ETF + EU index data, via "
        "the standard `run_bond_wfo` harness.",
        "",
        f"**Bonferroni gate**: CI_lo >= {BONF_CI_LO}, folds >= "
        f"{BONF_MIN_FOLDS}, pos >= {int(BONF_MIN_POS * 100)}%, "
        f"DD >= {BONF_MAX_DD}%.",
        "",
    ]

    # Per-channel summary
    lines.append("## Per-channel summary\n")
    lines.append("| Channel | Story | N | Max Sharpe | Max CI_lo | CI_lo > 0 | Bonf |")
    lines.append("|---|---|--:|---:|---:|--:|--:|")
    for ch in df["channel"].unique():
        sub = df[df["channel"] == ch]
        story = sub["story"].iloc[0]
        nbonf = sub.apply(passes_bonferroni, axis=1).sum() if not sub.empty else 0
        lines.append(
            f"| {ch} | {story[:50]}... | {len(sub)} | "
            f"{sub['sharpe'].max():+.3f} | {sub['ci_lo'].max():+.3f} | "
            f"{(sub['ci_lo'] > 0).sum()} | {nbonf} |"
        )
    lines.append("")

    if not bonf.empty:
        lines.append("## Bonferroni survivors\n")
        lines.append(
            "| # | Channel | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |"
        )
        lines.append("|--:|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
        for i, r in bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | {r['channel']} | {int(r['lookback'])} | "
                f"{int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )
    else:
        lines.append("## No Bonferroni survivors\n")
        lines.append(
            "None of the 5 channels produced a combo that clears the gate. "
            "Top-10 by CI_lo shown below for diagnostic value.\n"
        )
        top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
        lines.append("### Top 10 by CI_lo\n")
        lines.append(
            "| # | Channel | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |"
        )
        lines.append("|--:|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
        for i, r in top.iterrows():
            lines.append(
                f"| {i + 1} | {r['channel']} | {int(r['lookback'])} | "
                f"{int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
