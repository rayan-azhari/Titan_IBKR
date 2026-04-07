"""Phase 9 autonomous research loop.

Phase 8 breakthroughs:
- MR AUD_JPY v46 don sp0.5 -> 4.5905 (BEATS IWB baseline of 4.5462!)
- MR AUD_JPY v46 don sp0   -> 4.6910 (new overall best)
- MR AUD_JPY v56 don sp1   -> 4.3823 (another promising vwap)
- v42 sp0.5 -> 4.46, v44 sp0.5 -> 4.49

Current state: BEST=4.6910 (v46 don sp0)

Phase 9 strategy:
1. Fine-tune around v46 sp0.5 (first confirmed real-cost beater): v44, v45, v47, v48
2. v56 with sp0.5 (scored 4.38 with sp1 — could approach 4.6+ with sp0.5?)
3. v46 with tiny param changes: is_bars, oos_bars combinations
4. Is there a vwap in 50-60 range that beats v46?
5. Document the winning config for deployment consideration
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [4.6910]  # Updated to new best
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

# Current best standalone: v46 don sp0 (4.6910)
MR_BEST = dict(
    strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="H1",
    vwap_anchor=46, regime_filter="conf_donchian_pos_20",
    tier_grid="conservative", spread_bps=0.0, slippage_bps=0.0,
    is_bars=30000, oos_bars=7500,
)

# First real-cost beater: v46 don sp0.5 (4.5905)
MR_REAL = dict(
    strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="H1",
    vwap_anchor=46, regime_filter="conf_donchian_pos_20",
    tier_grid="conservative", spread_bps=0.5, slippage_bps=0.2,
    is_bars=30000, oos_bars=7500,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # ── 1. Vwap sweep adjacent to v46 with sp0.5 ──────────────────────────────
    ("MR AUD_JPY v44 sp0.5", c(MR_REAL, vwap_anchor=44)),
    ("MR AUD_JPY v45 sp0.5", c(MR_REAL, vwap_anchor=45)),
    ("MR AUD_JPY v47 sp0.5", c(MR_REAL, vwap_anchor=47)),
    ("MR AUD_JPY v48 sp0.5", c(MR_REAL, vwap_anchor=48)),
    ("MR AUD_JPY v50 sp0.5", c(MR_REAL, vwap_anchor=50)),
    ("MR AUD_JPY v52 sp0.5", c(MR_REAL, vwap_anchor=52)),
    ("MR AUD_JPY v54 sp0.5", c(MR_REAL, vwap_anchor=54)),
    ("MR AUD_JPY v56 sp0.5", c(MR_REAL, vwap_anchor=56)),
    ("MR AUD_JPY v58 sp0.5", c(MR_REAL, vwap_anchor=58)),
    ("MR AUD_JPY v60 sp0.5", c(MR_REAL, vwap_anchor=60)),

    # ── 2. v56 with sp0 (scored 4.38 with sp1 → maybe 4.6+ with sp0?) ────────
    ("MR AUD_JPY v56 sp0", c(MR_BEST, vwap_anchor=56)),
    ("MR AUD_JPY v56 sp0.5", c(MR_REAL, vwap_anchor=56)),

    # ── 3. v46 sp0 fine-tuning ────────────────────────────────────────────────
    ("MR AUD_JPY v46 sp0 oos6k", c(MR_BEST, oos_bars=6000)),
    ("MR AUD_JPY v46 sp0 oos9k", c(MR_BEST, oos_bars=9000)),
    ("MR AUD_JPY v46 sp0 is25k", c(MR_BEST, is_bars=25000)),
    ("MR AUD_JPY v46 sp0 rsi14", c(MR_BEST, regime_filter="conf_rsi_14_dev")),
    ("MR AUD_JPY v46 sp0 no_filter", c(MR_BEST, regime_filter="no_filter")),

    # ── 4. Vwap sweep with sp0 to find the true optimum ──────────────────────
    ("MR AUD_JPY v44 sp0", c(MR_BEST, vwap_anchor=44)),
    ("MR AUD_JPY v48 sp0", c(MR_BEST, vwap_anchor=48)),
    ("MR AUD_JPY v50 sp0", c(MR_BEST, vwap_anchor=50)),
    ("MR AUD_JPY v52 sp0", c(MR_BEST, vwap_anchor=52)),
    ("MR AUD_JPY v60 sp0", c(MR_BEST, vwap_anchor=60)),

    # ── 5. Portfolio with new best ────────────────────────────────────────────
    ("Portfolio IWB+v46 sp0.5", {
        "strategy": "portfolio",
        "description": "Portfolio IWB60 + AUD_JPY v46 don sp0.5",
        "strategies": [
            dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
            dict(**MR_REAL, weight=0.4),
        ],
    }),

    # ── 6. AUD_JPY v46 don with rsi_14_dev AND donchian combined (aggressive) ─
    ("MR AUD_JPY v46 don standard sp0.5", c(MR_REAL, tier_grid="standard")),
]


if __name__ == "__main__":
    print(f"Phase 9 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 9 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
