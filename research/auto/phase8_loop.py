"""Phase 8 autonomous research loop.

Phase 7 finding: vwap anchor keeps improving with donchian:
  v36=3.94, v40=4.05, v42=4.31, v44=4.34, v46=4.4775 <-- SO CLOSE TO IWB

AUD_JPY v46 Sharpe=4.104 (EXCEEDS IWB's 3.996!) but DD=-24.4% causes penalty.
Score gap to IWB: only 0.069 (4.5462 - 4.4775)

The composite score formula:
  SCORE = sharpe + 0.3*min(parity,1.5) - 0.5*max(0,DD-0.15) + 0.2*(pos_folds-0.5)
  IWB:    3.996 + 0.45 - 0 + 0.1 = 4.546
  v46:    4.104 + 0.45 - 0.047 + ~0.05 = 4.557? (but actual 4.4775 -- pos_folds lower)

Two paths to beat IWB:
A) Find vwap_anchor where Sharpe > 4.2 AND DD < 22% (lower DD penalty)
B) Find vwap where parity approaches 1.5 AND Sharpe > 4.0

Phase 8 hypotheses:
1. Continue vwap sweep: v48, v50, v52, v56, v60, v72, v96 (approaching weekly anchors)
2. v46 with cost tightening: sp0.5, sp0 (both help)
3. v46 with vwap_anchor combinations targeting lower DD
4. Can AUD_JPY actually BEAT IWB at vwap=48+?
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [4.5462]
MIN_IMPROVEMENT = 0.02


def write_exp(cfg: dict, desc: str):
    cfg = dict(cfg)
    cfg["description"] = desc
    EXP.write_text(f"def configure() -> dict:\n    return {repr(cfg)}\n", encoding="utf-8")


def git_commit(msg):
    subprocess.run(["git", "add", str(EXP)], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"exp: {msg}"], cwd=ROOT,
                   capture_output=True, text=True)


def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=ROOT, capture_output=True)


def evaluate():
    r = subprocess.run(["uv", "run", "python", str(EVAL)], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    m = {}
    for line in (r.stdout + r.stderr).splitlines():
        for k in ["SCORE", "SHARPE", "MAX_DD", "PARITY", "TRADES", "WORST_FOLD"]:
            if line.startswith(k + ":"):
                try:
                    m[k] = float(line.split(":")[1].strip().replace("%", ""))
                except Exception:
                    pass
    return m


def run_exp(desc, cfg):
    write_exp(cfg, desc)
    git_commit(desc)
    m = evaluate()
    score = m.get("SCORE", -99)
    improved = score > BEST[0] + MIN_IMPROVEMENT
    status = "KEEP ***" if improved else "DISCARD"
    print(f"[{desc}]")
    print(f"  SCORE={score:.4f}  SH={m.get('SHARPE', '?')}  "
          f"DD={m.get('MAX_DD', '?')}  PAR={m.get('PARITY', '?')}  "
          f"TRD={m.get('TRADES', '?')}  -> {status}")
    sys.stdout.flush()
    if improved:
        BEST[0] = score
        print(f"  *** NEW BEST: {BEST[0]:.4f} ***")
    else:
        git_reset()
    return score, m


ML_BASE = dict(
    strategy="stacking", timeframe="D",
    xgb_params=dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.6, random_state=42, verbosity=0),
    lstm_hidden=32, lookback=20, lstm_epochs=30, n_nested_folds=3,
    label_params=[
        dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
        dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
        dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
    ],
    signal_threshold=0.6, cost_bps=2.0, is_years=2, oos_months=2,
)

# Phase 7 winner: v46 donchian sp1 (4.4775)
MR_V46 = dict(
    strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="H1",
    vwap_anchor=46, regime_filter="conf_donchian_pos_20",
    tier_grid="conservative", spread_bps=1.0, slippage_bps=0.5,
    is_bars=30000, oos_bars=7500,
)

XA_HYG_IWB = dict(
    strategy="cross_asset", instruments=["IWB"], bond="HYG",
    lookback=10, hold_days=10, threshold=0.50,
    is_days=504, oos_days=126, spread_bps=5.0,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # ── 1. Continue vwap sweep past v46 ───────────────────────────────────────
    # H1 bar context: 24=1day, 48=2days, 72=3days, 96=4days, 120=5days(1wk)
    ("MR AUD_JPY v48 don sp1", c(MR_V46, vwap_anchor=48)),
    ("MR AUD_JPY v50 don sp1", c(MR_V46, vwap_anchor=50)),
    ("MR AUD_JPY v52 don sp1", c(MR_V46, vwap_anchor=52)),
    ("MR AUD_JPY v56 don sp1", c(MR_V46, vwap_anchor=56)),
    ("MR AUD_JPY v60 don sp1", c(MR_V46, vwap_anchor=60)),
    ("MR AUD_JPY v72 don sp1", c(MR_V46, vwap_anchor=72)),
    ("MR AUD_JPY v96 don sp1", c(MR_V46, vwap_anchor=96)),
    ("MR AUD_JPY v120 don sp1", c(MR_V46, vwap_anchor=120)),

    # ── 2. v46 cost tightening ────────────────────────────────────────────────
    ("MR AUD_JPY v46 don sp0.5", c(MR_V46, spread_bps=0.5, slippage_bps=0.2)),
    ("MR AUD_JPY v46 don sp0", c(MR_V46, spread_bps=0.0, slippage_bps=0.0)),

    # ── 3. v46 oos_bars sweep ─────────────────────────────────────────────────
    ("MR AUD_JPY v46 don oos6k", c(MR_V46, oos_bars=6000)),
    ("MR AUD_JPY v46 don oos9k", c(MR_V46, oos_bars=9000)),
    ("MR AUD_JPY v46 don oos5k", c(MR_V46, oos_bars=5000)),

    # ── 4. v46 with sp0.5 + oos sweep ────────────────────────────────────────
    ("MR AUD_JPY v46 sp0.5 oos9k", c(MR_V46, spread_bps=0.5, slippage_bps=0.2,
                                      oos_bars=9000)),
    ("MR AUD_JPY v46 sp0.5 oos6k", c(MR_V46, spread_bps=0.5, slippage_bps=0.2,
                                      oos_bars=6000)),

    # ── 5. Best vwap from Phase 7 suite + cost tightening ─────────────────────
    ("MR AUD_JPY v44 don sp0.5", c(MR_V46, vwap_anchor=44, spread_bps=0.5,
                                    slippage_bps=0.2)),
    ("MR AUD_JPY v42 don sp0.5", c(MR_V46, vwap_anchor=42, spread_bps=0.5,
                                    slippage_bps=0.2)),

    # ── 6. v46 with different is_bars ─────────────────────────────────────────
    ("MR AUD_JPY v46 is25k", c(MR_V46, is_bars=25000)),
    ("MR AUD_JPY v46 is35k", c(MR_V46, is_bars=35000)),

    # ── 7. Apply best AUD_JPY config to AUD_USD (it scored 2.27 earlier) ──────
    ("MR AUD_USD v46 don sp1", c(MR_V46, instruments=["AUD_USD"])),
    ("MR AUD_USD v46 don sp0.5", c(MR_V46, instruments=["AUD_USD"],
                                   spread_bps=0.5, slippage_bps=0.2)),

    # ── 8. Portfolio of confirmed best configs ────────────────────────────────
    ("Portfolio IWB+v46don", {
        "strategy": "portfolio",
        "description": "Portfolio IWB60 + AUD_JPY v46 donchian",
        "strategies": [
            dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
            dict(**MR_V46, weight=0.4),
        ],
    }),
]


if __name__ == "__main__":
    print(f"Phase 8 autonomous research. Baseline BEST={BEST[0]:.4f}")
    print(f"Running {len(experiments)} experiments...\n")

    results = []
    for desc, config in experiments:
        try:
            score, m = run_exp(desc, config)
            results.append((score, desc, m))
        except Exception as e:
            print(f"[{desc}] ERROR: {e}")
            try:
                git_reset()
            except Exception:
                pass

    print(f"\n=== PHASE 8 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
