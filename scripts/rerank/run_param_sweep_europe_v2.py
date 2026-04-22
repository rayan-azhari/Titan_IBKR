"""Three follow-up experiments to the (negative) Europe cross-asset sweep.

Tests whether the cross-asset edge fails on DAX/FTSE because of
(a) currency-translation noise, (b) wrong signal family, or (c)
the FX itself being the problem.

  Exp A -- USD-denominated EU ETFs as targets
           Tests whether EWG/EWU/IEV/VGK (USD-tradable) show the edge
           that ^GDAXI/^FTSE (local-currency indices) did not. If yes,
           the FX translation was diluting the signal.

  Exp B -- EU-native bond-ETF signals -> EU equity
           Tests whether IGOV/BWX/BNDX (international bond ETFs that
           include Bunds/OATs/BTPs) predict DAX/FTSE/EWG/EWU. If yes,
           the signal needs to be EU-rate-denominated to predict EU
           equity.

  Exp C -- Currency-hedged DAX as target
           Build DAX_hedged = ^GDAXI * (1 / EURUSD), test whether the
           US signals predict this synthetic hedged series. If yes,
           the equity component carries an edge but FX overwhelms it.

Writes:
  .tmp/reports/param_sweep_europe_v2_2026_04_22/exp_a_results.csv
  .tmp/reports/param_sweep_europe_v2_2026_04_22/exp_b_results.csv
  .tmp/reports/param_sweep_europe_v2_2026_04_22/exp_c_results.csv
  .tmp/reports/param_sweep_europe_v2_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "param_sweep_europe_v2_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

BONF_CI_LO = 0.45
BONF_MIN_FOLDS = 25
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0


def run_combo(
    signal_close: pd.Series,
    target_close: pd.Series,
    lookback: int,
    hold: int,
    threshold: float,
) -> dict | None:
    from research.cross_asset.run_bond_equity_wfo import run_bond_wfo

    try:
        r = run_bond_wfo(
            signal_close,
            target_close,
            lookback=lookback,
            hold_days=hold,
            threshold=threshold,
            is_days=504,
            oos_days=126,
            spread_bps=5.0,
        )
    except Exception:
        return None
    if r.get("n_folds", 0) < 5:
        return None
    return {
        "sharpe": r.get("stitched_sharpe", 0.0),
        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
        "n_folds": r.get("n_folds", 0),
        "pct_positive": r.get("pct_positive", 0.0),
        "n_trades": r.get("total_trades", 0),
    }


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= BONF_MIN_FOLDS
        and r["pct_positive"] >= BONF_MIN_POS
        and r["max_dd_pct"] >= BONF_MAX_DD
    )


def load_hedged_dax() -> pd.Series:
    """^GDAXI (local-currency DAX) * (1 / EURUSD). The result is a USD-
    denominated hedged return stream — pure equity exposure with the
    FX layer stripped out."""
    from research.cross_asset.run_bond_equity_wfo import load_daily

    dax = load_daily("GDAXI")
    eurusd = load_daily("EUR_USD")
    # Both series may carry duplicate timestamps on exchange holidays;
    # collapse first, then align on common index.
    dax = dax[~dax.index.duplicated(keep="last")].sort_index()
    eurusd = eurusd[~eurusd.index.duplicated(keep="last")].sort_index()
    common = dax.index.intersection(eurusd.index)
    dax_c = dax.reindex(common).ffill()
    eurusd_c = eurusd.reindex(common).ffill()
    # DAX is in EUR; EURUSD is USD per EUR → DAX_USD = DAX * EURUSD
    # (A German investor hedged to USD is the inverse: they synthesise
    # USD returns from EUR exposure. We want pure equity Sharpe so
    # we build the local-currency-denominated hedged series by
    # dividing out the FX move: DAX_hedged = DAX * (mean EURUSD) /
    # EURUSD. This keeps the series scale comparable to the unhedged.)
    ref = eurusd_c.mean()
    hedged = dax_c * (ref / eurusd_c)
    hedged.name = "GDAXI_hedged"
    return hedged


def run_experiment(
    name: str,
    signals: list[str],
    targets: list[str],
    target_loader=None,
) -> pd.DataFrame:
    """Run a signal x target x param grid. target_loader allows
    a synthetic target like the hedged DAX."""
    from research.cross_asset.run_bond_equity_wfo import load_daily

    rows: list[dict] = []
    t0 = time.time()
    combos: list[tuple[str, str, int, int, float]] = []
    for sig in signals:
        for tgt in targets:
            if sig == tgt:
                continue
            for lb in LOOKBACKS:
                for hold in HOLDS:
                    for th in THRESHOLDS:
                        combos.append((sig, tgt, lb, hold, th))
    print(f"\n  === {name} ===")
    print(f"  Combos: {len(combos)}")

    sig_cache: dict[str, pd.Series] = {}
    tgt_cache: dict[str, pd.Series] = {}

    tested = 0
    for sig, tgt, lb, hold, th in combos:
        tested += 1
        if sig not in sig_cache:
            try:
                sig_cache[sig] = load_daily(sig)
            except FileNotFoundError:
                continue
        if tgt not in tgt_cache:
            try:
                if target_loader is not None and tgt == "GDAXI_hedged":
                    tgt_cache[tgt] = target_loader()
                else:
                    tgt_cache[tgt] = load_daily(tgt)
            except FileNotFoundError:
                continue
        sc = sig_cache[sig]
        tc = tgt_cache[tgt]
        r = run_combo(sc, tc, lb, hold, th)
        if r is None:
            continue
        r.update(
            {"experiment": name, "signal": sig, "target": tgt, "lookback": lb, "hold": hold, "threshold": th}
        )
        rows.append(r)
        if tested % 200 == 0:
            print(
                f"  [{tested}/{len(combos)}] {sig}->{tgt} lb={lb} hold={hold} th={th}  "
                f"collected {len(rows)}"
            )

    df = pd.DataFrame(rows)
    print(f"  Done: tested {tested}, collected {len(df)}, {time.time() - t0:.0f}s")
    return df


def summarise(df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"## {title}\n"]
    if df.empty:
        lines.append("_No results._\n")
        return lines
    n_total = len(df)
    bonf = df[df.apply(passes_bonferroni, axis=1)]
    lines.append(f"- N combos: {n_total}")
    lines.append(f"- Max Sharpe: {df['sharpe'].max():+.3f}")
    lines.append(f"- Max CI_lo: {df['ci_lo'].max():+.3f}")
    lines.append(f"- CI_lo > 0 combos: {(df['ci_lo'] > 0).sum()}")
    lines.append(f"- **Bonferroni survivors (CI_lo >= {BONF_CI_LO}): {len(bonf)}**\n")

    top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
    lines.append("Top 10 by CI_lo:\n")
    lines.append(
        "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | CI_hi | Max DD | Folds | Pos% |"
    )
    lines.append("|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
    for i, r in top.iterrows():
        lines.append(
            f"| {i + 1} | {r['signal']} | {r['target']} | "
            f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
            f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | {r['ci_hi']:+.3f} | "
            f"{r['max_dd_pct']:.1f}% | {int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
        )
    lines.append("")
    return lines


def main() -> None:
    print("=" * 70)
    print("  European cross-asset follow-ups v2")
    print("=" * 70)

    # Exp A — USD-denominated EU ETFs
    us_signals = ["TLT", "IEF", "HYG", "TIP", "LQD", "UUP"]
    us_eu_etfs = ["EWG", "EWU", "IEV", "VGK"]
    df_a = run_experiment("exp_a_usd_eu_etfs", us_signals, us_eu_etfs)
    df_a.to_csv(REPORT / "exp_a_results.csv", index=False)

    # Exp B — EU-denominated bond signals
    eu_signals = ["IGOV", "BWX", "BNDX"]
    eu_targets = ["GDAXI", "FTSE", "EWG", "EWU"]
    df_b = run_experiment("exp_b_eu_bond_signals", eu_signals, eu_targets)
    df_b.to_csv(REPORT / "exp_b_results.csv", index=False)

    # Exp C — currency-hedged DAX
    df_c = run_experiment(
        "exp_c_hedged_dax",
        us_signals,
        ["GDAXI_hedged"],
        target_loader=load_hedged_dax,
    )
    df_c.to_csv(REPORT / "exp_c_results.csv", index=False)

    # ── Leaderboard ────────────────────────────────────────────────────
    lines: list[str] = [
        "# European Follow-Ups v2 — 2026-04-22",
        "",
        "Three follow-up experiments to the (negative) European "
        "cross-asset sweep. Tests whether the edge fails because of:",
        "  - (A) currency-translation noise (use USD-denominated ETFs)",
        "  - (B) wrong signal family (use EU-exposed bond signals)",
        "  - (C) the FX itself (build a synthetic currency-hedged DAX)",
        "",
        f"**Bonferroni gate**: CI_lo >= {BONF_CI_LO}, folds >= "
        f"{BONF_MIN_FOLDS}, pos >= {int(BONF_MIN_POS * 100)}%, "
        f"DD >= {BONF_MAX_DD}%.",
        "",
    ]
    lines.extend(summarise(df_a, "Exp A — USD-denominated EU ETFs (EWG, EWU, IEV, VGK)"))
    lines.extend(summarise(df_b, "Exp B — EU-exposed bond signals (IGOV, BWX, BNDX) -> EU equity"))
    lines.extend(summarise(df_c, "Exp C — Currency-hedged DAX (DAX * 1/EURUSD)"))

    # Final verdict
    survivors_a = df_a[df_a.apply(passes_bonferroni, axis=1)] if not df_a.empty else df_a
    survivors_b = df_b[df_b.apply(passes_bonferroni, axis=1)] if not df_b.empty else df_b
    survivors_c = df_c[df_c.apply(passes_bonferroni, axis=1)] if not df_c.empty else df_c
    total = len(survivors_a) + len(survivors_b) + len(survivors_c)

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Total Bonferroni survivors across all 3 experiments: **{total}**")
    if total == 0:
        lines.append("")
        lines.append(
            "No surviving (signal, target, params) combo in any of the 3 "
            "follow-ups. The US cross-asset edge does not transmit to "
            "European equity targets under any of the tested routes "
            "(USD-tradable proxies, EU-native bond signals, or FX-hedged "
            "synthetic). This **confirms** the original negative result — "
            "the European gap is structural, not a transmission artefact."
        )
    else:
        lines.append("")
        lines.append("Route(s) that produced survivors:")
        if not survivors_a.empty:
            lines.append(f"- **A (USD ETFs)**: {len(survivors_a)} survivors")
        if not survivors_b.empty:
            lines.append(f"- **B (EU bond signals)**: {len(survivors_b)} survivors")
        if not survivors_c.empty:
            lines.append(f"- **C (hedged DAX)**: {len(survivors_c)} survivors")

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
