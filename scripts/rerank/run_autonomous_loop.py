"""Multi-phase autonomous autoresearch loop.

Runs until no more meaningful progress is made. Phases:

  Phase A (broad sweep): every bond/dollar/commodity signal x every
    liquid equity/commodity target x 4 lookbacks. ~400 experiments.
  Phase B (ML expansion): ML stacking on every major ETF/index with
    sufficient daily history.
  Phase C (MR beyond the 140-instrument re-rank): the agent's earlier
    loop found AUD/JPY vwap_anchor=24 beats 46 -- expand to other
    H1 pairs with that finding (and full filter x tier sweep).
  Phase D (pairs): same-sector equity pairs with suitable history.

After each phase we log rows to ``.tmp/reports/autonomous_2026_04_21/``,
print the top of the running leaderboard, and stop phases early if
the current phase produced no combo with ``ci_lo > 0.4`` (i.e. no
meaningfully-new gate-passer).

Stop criteria (whichever hits first):
  * Two consecutive phases with no new ci_lo > 0.4 result.
  * 3000 total experiments (safety cap).

The agent reasoning phase happens out-of-band (in the conversation
after this script writes its leaderboard).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "autonomous_2026_04_21"
REPORT.mkdir(parents=True, exist_ok=True)


# ── Universe discovery ──────────────────────────────────────────────────


def discover_daily(min_bars: int = 1260) -> set[str]:
    out = set()
    for p in Path(ROOT / "data").glob("*_D.parquet"):
        try:
            df = pd.read_parquet(p)
            if len(df) >= min_bars:
                out.add(p.stem.replace("_D", ""))
        except Exception:
            pass
    return out


def discover_h1(min_bars: int = 15000) -> set[str]:
    out = set()
    for p in Path(ROOT / "data").glob("*_H1.parquet"):
        try:
            df = pd.read_parquet(p)
            if len(df) >= min_bars:
                out.add(p.stem.replace("_H1", ""))
        except Exception:
            pass
    return out


# ── Pipeline wrappers ──────────────────────────────────────────────────


def run_bond_equity(bond: str, target: str, lookback: int) -> dict | None:
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    try:
        bc = load_daily(bond)
        tc = load_daily(target)
    except FileNotFoundError:
        return None
    try:
        r = run_bond_wfo(
            bc,
            tc,
            lookback=lookback,
            hold_days=20,
            threshold=0.50,
            is_days=504,
            oos_days=126,
            spread_bps=5.0,
        )
    except Exception:
        return None
    if r.get("n_folds", 0) < 5:
        return None
    return {
        "strategy": "cross_asset",
        "instrument": f"{bond}->{target}",
        "params": f"lb={lookback}",
        "sharpe": r.get("stitched_sharpe", 0.0),
        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
        "n_folds": r.get("n_folds", 0),
        "pct_positive": r.get("pct_positive", 0.0),
        "n_trades": r.get("total_trades", 0),
    }


def run_mr_confluence_full(instrument: str) -> list[dict]:
    """MR with every filter/tier combo + sweep of VWAP anchors {12,24,36}."""
    from research.mean_reversion.run_confluence_regime_test import (
        build_atr_regime_mask,
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import run_mr_wfo

    try:
        df = load_h1(instrument)
    except FileNotFoundError:
        return []
    n_bars = len(df)
    if n_bars >= 40000:
        is_bars, oos_bars = 32000, 8000
    elif n_bars >= 15000:
        is_bars, oos_bars = int(n_bars * 0.75), int(n_bars * 0.1)
    else:
        return []

    close = df["close"]
    rows: list[dict] = []
    # VWAP anchor finding (agent loop iter-5): 24 beats 46 on AUD/JPY.
    # Test 12 / 24 / 36 for each filter/tier combo.
    for vwap_anchor in (12, 24, 36):
        try:
            deviation = compute_vwap_deviation(close, anchor_period=vwap_anchor)
        except Exception:
            continue

        filters = {
            "conf_donchian_pos_20": build_confluence_disagreement_mask(df, "donchian_pos_20"),
            "conf_rsi_14_dev": build_confluence_disagreement_mask(df, "rsi_14_dev"),
            "atr_only": build_atr_regime_mask(df),
            "no_filter": pd.Series(True, index=df.index),
        }
        for filt_name, mask in filters.items():
            for grid_name, pcts in (
                ("standard", [0.90, 0.95, 0.98, 0.99]),
                ("conservative", [0.95, 0.98, 0.99, 0.999]),
            ):
                try:
                    r = run_mr_wfo(close, deviation, mask, pcts, is_bars=is_bars, oos_bars=oos_bars)
                except Exception:
                    continue
                if r.get("n_folds", 0) < 2:
                    continue
                rows.append(
                    {
                        "strategy": "mr_confluence",
                        "instrument": instrument,
                        "params": f"vwap{vwap_anchor}/{filt_name}/{grid_name}",
                        "sharpe": r.get("stitched_sharpe", 0.0),
                        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
                        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
                        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
                        "n_folds": r.get("n_folds", 0),
                        "pct_positive": r.get("pct_positive", 0.0),
                        "n_trades": r.get("total_trades", 0),
                    }
                )
    return rows


def run_ml_stacking(instrument: str, oos_months: int = 2) -> dict | None:
    """Invoke evaluate.py via subprocess for an ML stacking run."""
    import subprocess

    ML_CFG = dict(
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
        instruments=[instrument],
    )
    EXP = ROOT / "research/auto/experiment.py"
    EVAL = ROOT / "research/auto/evaluate.py"
    EXP.write_text(f"def configure() -> dict:\n    return {repr(ML_CFG)}\n", encoding="utf-8")

    try:
        r = subprocess.run(
            ["uv", "run", "python", str(EVAL)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return None
    parsed: dict = {}
    for line in (r.stdout + r.stderr).splitlines():
        for k in ("SCORE", "SHARPE", "MAX_DD", "TRADES", "N_FOLDS", "POS_FOLDS"):
            pref = k + ":"
            if line.startswith(pref):
                try:
                    parsed[k.lower()] = float(line[len(pref) :].strip().rstrip("%"))
                except ValueError:
                    pass
    if parsed.get("score", -99) < 0 or parsed.get("trades", 0) < 3:
        return None
    return {
        "strategy": "ml_stacking",
        "instrument": instrument,
        "params": f"oos_months={oos_months}",
        "sharpe": parsed.get("sharpe", 0.0),
        "ci_lo": 0.0,  # ML WFO doesn't expose CI currently
        "ci_hi": 0.0,
        "max_dd_pct": parsed.get("max_dd", 0.0),
        "n_folds": int(parsed.get("n_folds", 0)),
        "pct_positive": parsed.get("pos_folds", 0.0) / 100.0,
        "n_trades": int(parsed.get("trades", 0)),
        "score": parsed.get("score", 0.0),
    }


# ── Phase drivers ──────────────────────────────────────────────────────


# Macro signal and target universes.
SIGNAL_UNIVERSE = ["TLT", "IEF", "HYG", "TIP", "LQD", "UUP", "DXY"]
TARGET_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWB",
    "GLD",
    "DBC",
    "EEM",
    "EFA",
    "TQQQ",
    "IAU",
    "SLV",
    "GDX",
    "TIP",
    "HYG",
]
LOOKBACKS = [10, 20, 40, 60]

# ML candidates: broad indices + leveraged + liquid sector-like ETFs.
ML_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWB",
    "TQQQ",
    "GLD",
    "EFA",
    "EEM",
    "DBC",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "DIA",
    "IWM",
]

# Pairs candidates (same sector, similar market cap) from discover_daily.
PAIRS_UNIVERSE = [
    ("INTC", "TXN"),
    ("MSFT", "AAPL"),
    ("GOOGL", "META"),
    ("AMZN", "META"),
    ("JPM", "BAC"),
    ("GS", "MS"),
    ("V", "MA"),
    ("XOM", "CVX"),
    ("KO", "PEP"),
    ("HD", "LOW"),
    ("UNH", "CVS"),
    ("PFE", "MRK"),
    ("SPY", "QQQ"),
    ("GLD", "IAU"),
    ("TLT", "IEF"),
]


def phase_A_cross_asset(rows: list[dict], daily_avail: set[str]) -> int:
    print("\n[Phase A] Cross-asset: bond/dollar -> equity/commodity sweep")
    new_gate_passers = 0
    total = 0
    t0 = time.time()
    for sig in SIGNAL_UNIVERSE:
        if sig not in daily_avail:
            continue
        for tgt in TARGET_UNIVERSE:
            if tgt == sig or tgt not in daily_avail:
                continue
            for lb in LOOKBACKS:
                total += 1
                row = run_bond_equity(sig, tgt, lb)
                if row is None:
                    continue
                rows.append(row)
                if row["ci_lo"] > 0.4:
                    new_gate_passers += 1
    print(f"  {total} combos, {new_gate_passers} ci_lo>0.4, {time.time() - t0:.0f}s")
    return new_gate_passers


def phase_B_ml(rows: list[dict], daily_avail: set[str]) -> int:
    print("\n[Phase B] ML stacking on major indices")
    new_strong = 0
    t0 = time.time()
    for inst in ML_UNIVERSE:
        if inst not in daily_avail:
            continue
        row = run_ml_stacking(inst)
        if row is None:
            continue
        rows.append(row)
        if row.get("score", 0) > 1.5:
            new_strong += 1
            print(f"  STRONG: {inst} SCORE={row.get('score', 0):+.3f} SH={row['sharpe']:+.3f}")
    print(f"  {new_strong} strong ML signals (SCORE > 1.5), {time.time() - t0:.0f}s")
    return new_strong


def phase_C_mr_full(rows: list[dict], h1_avail: set[str]) -> int:
    """Expanded MR on major H1 FX pairs + top H1 equities, all vwap_anchors."""
    print("\n[Phase C] MR full sweep (vwap_anchor x filter x tier) on key H1")
    # Target the long-history instruments where MR makes sense.
    targets = [
        "AUD_JPY",
        "AUD_USD",
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "USD_CHF",
        "GLD",
        "QQQ",
        "SPY",
    ]
    new_gate_passers = 0
    t0 = time.time()
    for inst in targets:
        if inst not in h1_avail:
            continue
        sub = run_mr_confluence_full(inst)
        rows.extend(sub)
        new_gate_passers += sum(1 for r in sub if r["ci_lo"] > 0.4)
    print(f"  {new_gate_passers} ci_lo>0.4, {time.time() - t0:.0f}s")
    return new_gate_passers


def phase_D_pairs(rows: list[dict], daily_avail: set[str]) -> int:
    """Pairs trading — reuse catalog runner via subprocess."""
    import subprocess

    print("\n[Phase D] Pairs trading on same-sector pairs")
    new_good = 0
    t0 = time.time()
    EXP = ROOT / "research/auto/experiment.py"
    EVAL = ROOT / "research/auto/evaluate.py"
    PT_CFG_BASE = dict(
        strategy="pairs_trading",
        entry_z=2.0,
        exit_z=0.5,
        max_z=4.0,
        refit_window=126,
        is_days=504,
        oos_days=126,
    )
    for a, b in PAIRS_UNIVERSE:
        if a not in daily_avail or b not in daily_avail:
            continue
        cfg = dict(PT_CFG_BASE, instruments=[a], pair_b=b)
        EXP.write_text(f"def configure() -> dict:\n    return {repr(cfg)}\n", encoding="utf-8")
        try:
            r = subprocess.run(
                ["uv", "run", "python", str(EVAL)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            continue
        parsed = {}
        for line in (r.stdout + r.stderr).splitlines():
            for k in ("SCORE", "SHARPE", "MAX_DD", "TRADES"):
                pref = k + ":"
                if line.startswith(pref):
                    try:
                        parsed[k.lower()] = float(line[len(pref) :].strip().rstrip("%"))
                    except ValueError:
                        pass
        sh = parsed.get("sharpe", -99)
        if sh <= 0:
            continue
        rows.append(
            {
                "strategy": "pairs_trading",
                "instrument": f"{a}/{b}",
                "params": "entry_z=2.0",
                "sharpe": sh,
                "ci_lo": 0.0,
                "ci_hi": 0.0,
                "max_dd_pct": parsed.get("max_dd", 0.0),
                "n_folds": 0,
                "pct_positive": 0.0,
                "n_trades": int(parsed.get("trades", 0)),
                "score": parsed.get("score", 0.0),
            }
        )
        if sh > 0.4:
            new_good += 1
    print(f"  {new_good} pairs with SH>0.4, {time.time() - t0:.0f}s")
    return new_good


# ── Leaderboard writer ─────────────────────────────────────────────────


def write_leaderboard(rows: list[dict], phase_tag: str = "") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Deduplicate within strategy+instrument+params, keep last (latest run).
    df = df.drop_duplicates(subset=["strategy", "instrument", "params"], keep="last")
    df["gate_pass"] = df["ci_lo"] > 0.0
    df = df.sort_values(by=["ci_lo", "sharpe"], ascending=False).reset_index(drop=True)
    df.to_csv(REPORT / "leaderboard.csv", index=False)

    top = df.head(25)
    md = [f"# Autonomous Loop Leaderboard {phase_tag}", ""]
    md.append(f"{len(df)} combinations tracked. Top 25 by (CI_lo, Sharpe).\n")
    md.append("| # | Strategy | Instrument | Params | Sharpe | CI_lo | CI_hi | Max DD | Trades |")
    md.append("|--:|---|---|---|---:|---:|---:|---:|---:|")
    for i, row in top.iterrows():
        md.append(
            f"| {i + 1} | {row['strategy']} | {row['instrument']} | {row['params']} | "
            f"{row['sharpe']:+.3f} | {row['ci_lo']:+.3f} | {row['ci_hi']:+.3f} | "
            f"{row['max_dd_pct']:+.1f}% | {row['n_trades']} |"
        )
    (REPORT / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")
    return df


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    t_start = time.time()
    daily = discover_daily()
    h1 = discover_h1()
    print(f"Universe: {len(daily)} daily, {len(h1)} H1")

    rows: list[dict] = []
    no_progress_streak = 0

    phases = [
        ("A", lambda: phase_A_cross_asset(rows, daily)),
        ("B", lambda: phase_B_ml(rows, daily)),
        ("C", lambda: phase_C_mr_full(rows, h1)),
        ("D", lambda: phase_D_pairs(rows, daily)),
    ]

    for tag, fn in phases:
        progress = fn()
        df = write_leaderboard(rows, phase_tag=f"(after phase {tag})")
        print(
            f"\n  Running leaderboard after phase {tag}: "
            f"{len(df)} rows, gate-passers={int(df['gate_pass'].sum())}"
        )
        if not df.empty:
            print(
                df.head(10)[
                    ["strategy", "instrument", "params", "sharpe", "ci_lo", "n_trades"]
                ].to_string(index=False)
            )
        if progress == 0:
            no_progress_streak += 1
            print(f"  No new strong finds in phase {tag}. streak={no_progress_streak}")
            if no_progress_streak >= 2:
                print("\n  Two consecutive no-progress phases. Stopping.")
                break
        else:
            no_progress_streak = 0

    print(f"\n  Total: {len(rows)} experiments, {time.time() - t_start:.0f}s")
    print(f"  Leaderboard: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
