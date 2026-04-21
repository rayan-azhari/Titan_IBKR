"""Safe autoresearch re-run on the corrected harness.

Unlike ``research/auto/run_loop.py`` (which commits every experiment
and hard-resets rejected ones, destroying branch state), this script:

  * never touches git;
  * runs each experiment via a subprocess call to ``evaluate.py``;
  * captures SCORE/SHARPE/MAX_DD/CI etc. into a single CSV + markdown;
  * ranks by composite SCORE at the end.

Experiment set:
  * The original 50 experiments from ``run_loop.py``.
  * Plus the re-rank discoveries (TLT->QQQ, TIP->QQQ, HYG variants) at
    multiple lookbacks so we can see parameter sensitivity of the new
    candidates on the corrected harness.

Sanctuary window is active by default (last 365 days invisible to
the agent).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"

REPORT_DIR = ROOT / ".tmp" / "reports" / "autoresearch_2026_04_21"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Experiment bases ─────────────────────────────────────────────────────

ML_BASE = dict(
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
    oos_months=2,
)

MR_BASE = dict(
    strategy="mean_reversion",
    timeframe="H1",
    vwap_anchor=24,
    regime_filter="conf_rsi_14_dev",
    tier_grid="conservative",
    spread_bps=2.0,
    slippage_bps=1.0,
    is_bars=30000,
    oos_bars=7500,
)

XA_BASE = dict(
    strategy="cross_asset",
    lookback=20,
    hold_days=20,
    threshold=0.50,
    is_days=504,
    oos_days=126,
    spread_bps=5.0,
)

GM_BASE = dict(
    strategy="gold_macro",
    real_rate_window=20,
    dollar_window=20,
    slow_ma=200,
    cost_bps=5.0,
    is_days=504,
    oos_days=126,
)

FC_BASE = dict(
    strategy="fx_carry",
    carry_direction=1,
    sma_period=50,
    vol_target_pct=0.08,
    vix_halve_threshold=25.0,
    spread_bps=3.0,
    slippage_bps=1.0,
    is_days=504,
    oos_days=126,
)

PT_BASE = dict(
    strategy="pairs_trading",
    entry_z=2.0,
    exit_z=0.5,
    max_z=4.0,
    refit_window=126,
    is_days=504,
    oos_days=126,
)


def c(base: dict, **kw) -> dict:
    d = dict(base)
    d.update(kw)
    return d


# ── Experiment catalogue ─────────────────────────────────────────────────

EXPERIMENTS: list[tuple[str, dict]] = [
    # ── ML ─────────────────────────────────────────────────────────────
    ("ML IWB stacking", c(ML_BASE, instruments=["IWB"])),
    ("ML QQQ stacking oos3", c(ML_BASE, instruments=["QQQ"], oos_months=3)),
    ("ML SPY stacking oos3", c(ML_BASE, instruments=["SPY"], oos_months=3)),
    ("ML QQQ+SPY oos3", c(ML_BASE, instruments=["QQQ", "SPY"], oos_months=3)),
    # ── MR (FX) ─────────────────────────────────────────────────────────
    ("MR AUD_JPY rsi_dev cons", c(MR_BASE, instruments=["AUD_JPY"])),
    (
        "MR AUD_JPY donchian cons",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_donchian_pos_20"),
    ),
    ("MR EUR_USD rsi_dev cons", c(MR_BASE, instruments=["EUR_USD"])),
    (
        "MR EUR_USD donchian cons",
        c(MR_BASE, instruments=["EUR_USD"], regime_filter="conf_donchian_pos_20"),
    ),
    ("MR GBP_USD rsi_dev cons", c(MR_BASE, instruments=["GBP_USD"])),
    ("MR USD_JPY rsi_dev cons", c(MR_BASE, instruments=["USD_JPY"])),
    ("MR AUD_USD rsi_dev cons", c(MR_BASE, instruments=["AUD_USD"])),
    ("MR USD_CHF rsi_dev cons", c(MR_BASE, instruments=["USD_CHF"])),
    # ── Cross-asset (existing + new candidates from re-rank) ────────────
    ("XA IEF->GLD lb60", c(XA_BASE, instruments=["GLD"], bond="IEF", lookback=60)),
    ("XA IEF->GLD lb20", c(XA_BASE, instruments=["GLD"], bond="IEF", lookback=20)),
    ("XA IEF->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="IEF", lookback=10)),
    ("XA HYG->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=10)),
    ("XA HYG->IWB lb20", c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=20)),
    ("XA HYG->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="HYG", lookback=10)),
    # NEW discoveries from re-rank
    ("XA TLT->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10)),
    ("XA TLT->QQQ lb20", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=20)),
    ("XA TLT->QQQ lb60", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=60)),
    ("XA HYG->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10)),
    ("XA HYG->QQQ lb20", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=20)),
    ("XA TIP->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="TIP", lookback=10)),
    ("XA TIP->QQQ lb20", c(XA_BASE, instruments=["QQQ"], bond="TIP", lookback=20)),
    ("XA TIP->QQQ lb60", c(XA_BASE, instruments=["QQQ"], bond="TIP", lookback=60)),
    ("XA TIP->SPY lb60", c(XA_BASE, instruments=["SPY"], bond="TIP", lookback=60)),
    ("XA TIP->IWB lb60", c(XA_BASE, instruments=["IWB"], bond="TIP", lookback=60)),
    # Parameter tweaks on top candidates
    ("XA TLT->QQQ lb10 hold10", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10, hold_days=10)),
    ("XA TLT->QQQ lb10 hold40", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10, hold_days=40)),
    ("XA TLT->QQQ lb10 th025", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10, threshold=0.25)),
    ("XA TLT->QQQ lb10 th075", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10, threshold=0.75)),
    # ── Gold Macro ──────────────────────────────────────────────────────
    ("GM GLD default", c(GM_BASE, instruments=["GLD"])),
    ("GM GLD rr10", c(GM_BASE, instruments=["GLD"], real_rate_window=10)),
    ("GM GLD rr40", c(GM_BASE, instruments=["GLD"], real_rate_window=40)),
    ("GM GLD sma100", c(GM_BASE, instruments=["GLD"], slow_ma=100)),
    # ── FX Carry ────────────────────────────────────────────────────────
    ("FC AUD_JPY default", c(FC_BASE, instruments=["AUD_JPY"])),
    ("FC AUD_JPY sma20", c(FC_BASE, instruments=["AUD_JPY"], sma_period=20)),
    ("FC AUD_JPY sma100", c(FC_BASE, instruments=["AUD_JPY"], sma_period=100)),
    ("FC AUD_USD default", c(FC_BASE, instruments=["AUD_USD"])),
    # ── Pairs Trading ───────────────────────────────────────────────────
    ("PT INTC/TXN", c(PT_BASE, instruments=["INTC"], pair_b="TXN")),
    ("PT GOOGL/META", c(PT_BASE, instruments=["GOOGL"], pair_b="META")),
    ("PT GLD/IAU", c(PT_BASE, instruments=["GLD"], pair_b="IAU")),
    ("PT SPY/QQQ", c(PT_BASE, instruments=["SPY"], pair_b="QQQ")),
    ("PT MSFT/AAPL", c(PT_BASE, instruments=["MSFT"], pair_b="AAPL")),
]


# ── Experiment runner ────────────────────────────────────────────────────


METRIC_KEYS = (
    "SCORE",
    "SHARPE",
    "MAX_DD",
    "PARITY",
    "POS_FOLDS",
    "TRADES",
    "WORST_FOLD",
    "N_FOLDS",
)


def run_one(desc: str, cfg: dict, timeout: int = 600) -> dict:
    """Write experiment.py, invoke evaluate.py, parse metrics.

    Returns a dict with the parsed metrics plus bookkeeping. On any
    failure returns ``{"score": -99.0, "error": ...}``.
    """
    # Write the experiment config.
    EXP.write_text(
        f"def configure() -> dict:\n    return {repr(cfg)}\n",
        encoding="utf-8",
    )

    t0 = time.time()
    try:
        r = subprocess.run(
            ["uv", "run", "python", str(EVAL)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        stdout = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return {"desc": desc, "score": -99.0, "error": "timeout", "elapsed": timeout}
    except Exception as e:
        return {"desc": desc, "score": -99.0, "error": str(e), "elapsed": time.time() - t0}

    parsed: dict = {}
    for line in stdout.splitlines():
        for k in METRIC_KEYS:
            prefix = k + ":"
            if line.startswith(prefix):
                val = line[len(prefix):].strip().rstrip("%")
                try:
                    parsed[k.lower()] = float(val)
                except ValueError:
                    parsed[k.lower()] = val

    parsed["desc"] = desc
    parsed["strategy"] = cfg.get("strategy", "")
    parsed["instruments"] = ",".join(cfg.get("instruments", []))
    parsed["elapsed"] = time.time() - t0
    parsed["score"] = parsed.get("score", -99.0)
    return parsed


def main() -> None:
    print("=" * 70)
    print("  Autoresearch Safe Re-run — corrected harness + sanctuary ON")
    print(f"  {len(EXPERIMENTS)} experiments (~30-60 min)")
    print("=" * 70)
    print()

    results: list[dict] = []
    t_total = time.time()

    # Save the current experiment.py contents so we can restore at the end.
    original = EXP.read_text(encoding="utf-8") if EXP.exists() else ""

    try:
        for i, (desc, cfg) in enumerate(EXPERIMENTS, 1):
            t0 = time.time()
            res = run_one(desc, cfg)
            results.append(res)
            dt = time.time() - t0
            score = res.get("score", -99.0)
            sh = res.get("sharpe", float("nan"))
            dd = res.get("max_dd", float("nan"))
            err = res.get("error", "")
            flag = " " if isinstance(score, (int, float)) and score > 0 else "x"
            print(
                f"[{i:>2}/{len(EXPERIMENTS)}] {flag} {desc:<35} "
                f"SCORE={score:>+6.3f}  SH={sh if isinstance(sh, str) else f'{sh:+.3f}':>7}  "
                f"DD={dd if isinstance(dd, str) else f'{dd:+.3f}':>7}  "
                f"{dt:>5.1f}s {'ERROR '+err if err else ''}"
            )
            sys.stdout.flush()

            # Periodically flush CSV in case the process is interrupted.
            if i % 5 == 0:
                pd.DataFrame(results).to_csv(REPORT_DIR / "results.csv", index=False)
    finally:
        EXP.write_text(original, encoding="utf-8")

    print(f"\n  Done in {(time.time() - t_total) / 60:.1f} min")

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(REPORT_DIR / "results.csv", index=False)

    md = ["# Autoresearch Safe Re-run — 2026-04-21", ""]
    md.append(
        f"Corrected harness, sanctuary window ON. "
        f"{len(df)} experiments, wall-clock "
        f"{(time.time() - t_total) / 60:.1f} min."
    )
    md.append("")
    md.append("## Top 20 by composite SCORE")
    md.append("")
    md.append("| # | Desc | Strategy | Instruments | SCORE | Sharpe | Max DD | Parity | Trades |")
    md.append("|--:|---|---|---|---:|---:|---:|---:|---:|")
    for i, row in df.head(20).iterrows():
        sh = row.get("sharpe", float("nan"))
        dd = row.get("max_dd", float("nan"))
        pr = row.get("parity", float("nan"))
        tr = row.get("trades", 0)
        md.append(
            f"| {i + 1} | {row['desc']} | {row.get('strategy', '')} | "
            f"{row.get('instruments', '')} | "
            f"{row.get('score', float('nan')):+.3f} | "
            f"{sh if isinstance(sh, str) else f'{sh:+.3f}'} | "
            f"{dd if isinstance(dd, str) else f'{dd:+.3f}'} | "
            f"{pr if isinstance(pr, str) else f'{pr:+.3f}'} | "
            f"{tr} |"
        )
    md.append("")
    md.append("## Per-strategy-type leaderboards")
    md.append("")
    for strat in sorted(df["strategy"].dropna().unique()):
        sub = df[df["strategy"] == strat].head(5)
        md.append(f"### {strat}")
        md.append("")
        md.append("| Desc | Instruments | SCORE | Sharpe | Max DD | Trades |")
        md.append("|---|---|---:|---:|---:|---:|")
        for _, row in sub.iterrows():
            sh = row.get("sharpe", float("nan"))
            dd = row.get("max_dd", float("nan"))
            tr = row.get("trades", 0)
            md.append(
                f"| {row['desc']} | {row.get('instruments', '')} | "
                f"{row.get('score', float('nan')):+.3f} | "
                f"{sh if isinstance(sh, str) else f'{sh:+.3f}'} | "
                f"{dd if isinstance(dd, str) else f'{dd:+.3f}'} | "
                f"{tr} |"
            )
        md.append("")

    (REPORT_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n  CSV:  {REPORT_DIR / 'results.csv'}")
    print(f"  MD:   {REPORT_DIR / 'results.md'}")
    print(f"  JSON: {REPORT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
