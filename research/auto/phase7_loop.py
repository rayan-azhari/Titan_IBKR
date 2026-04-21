"""Phase 7 autonomous research loop.

Phase 6 findings:
- MR AUD_JPY vwap40 don sp1 -> 4.0460 (new peak with realistic costs)
- MR AUD_JPY vwap36 don sp0 -> 4.1988 (zero cost)
- MR AUD_JPY vwap36 don sp0.5 -> 4.0665
- Portfolio IWB + AUD_JPY_star -> 4.3044 (best portfolio ever)
- AUD_USD vwap36 don sp1 -> 2.27 (only FX pair that works besides AUD_JPY)
- Longer IS (35k, 40k) consistently hurts — 30k is optimal
- oos=9k helps (3.69 vs 3.95 at 7.5k for vwap36 don)

Phase 7 strategy:
The fundamental gap to IWB (4.5462) is structural:
  - IWB: parity=6.57 (OOS better than IS!) with 5 trades
  - AUD_JPY: parity~1.2-1.5, 180 trades

To compete, we need to find a regime where AUD_JPY has IWB-like parity.
Hypothesis: shorter IS (fewer is_bars) might produce better OOS/IS ratio.

Phase 7 hypotheses:
1. vwap40 donchian refinement: oos9k, sp0.5, vwap sweep 38/42/44/46
2. Shorter IS windows (10k, 15k, 20k) for higher parity
3. AUD_JPY stacking (ML classifier on FX H1) — novel
4. AUD_USD with vwap40 donchian
5. Try the MOST extreme: is_bars=10000, oos_bars=2500 with donchian
6. Portfolio with best configs now known
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
    subprocess.run(["git", "commit", "-m", f"exp: {msg}"], cwd=ROOT, capture_output=True, text=True)


def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=ROOT, capture_output=True)


def evaluate():
    r = subprocess.run(
        ["uv", "run", "python", str(EVAL)], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
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
    print(
        f"  SCORE={score:.4f}  SH={m.get('SHARPE', '?')}  "
        f"DD={m.get('MAX_DD', '?')}  PAR={m.get('PARITY', '?')}  "
        f"TRD={m.get('TRADES', '?')}  -> {status}"
    )
    sys.stdout.flush()
    if improved:
        BEST[0] = score
        print(f"  *** NEW BEST: {BEST[0]:.4f} ***")
    else:
        git_reset()
    return score, m


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

# Phase 6 best: vwap40 donchian sp1
MR_V40 = dict(
    strategy="mean_reversion",
    instruments=["AUD_JPY"],
    timeframe="H1",
    vwap_anchor=40,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=1.0,
    slippage_bps=0.5,
    is_bars=30000,
    oos_bars=7500,
)

# Phase 5 star config for comparison
MR_STAR = dict(
    strategy="mean_reversion",
    instruments=["AUD_JPY"],
    timeframe="H1",
    vwap_anchor=36,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=1.0,
    slippage_bps=0.5,
    is_bars=30000,
    oos_bars=7500,
)

XA_HYG_IWB = dict(
    strategy="cross_asset",
    instruments=["IWB"],
    bond="HYG",
    lookback=10,
    hold_days=10,
    threshold=0.50,
    is_days=504,
    oos_days=126,
    spread_bps=5.0,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # ── 1. vwap40 donchian refinement ─────────────────────────────────────────
    ("MR AUD_JPY v40 don sp0.5", c(MR_V40, spread_bps=0.5, slippage_bps=0.2)),
    ("MR AUD_JPY v40 don sp0", c(MR_V40, spread_bps=0.0, slippage_bps=0.0)),
    ("MR AUD_JPY v40 don oos9k", c(MR_V40, oos_bars=9000)),
    ("MR AUD_JPY v40 don oos9k sp0.5", c(MR_V40, oos_bars=9000, spread_bps=0.5, slippage_bps=0.2)),
    ("MR AUD_JPY v40 don oos6k", c(MR_V40, oos_bars=6000)),
    ("MR AUD_JPY v40 don is25k", c(MR_V40, is_bars=25000)),
    ("MR AUD_JPY v42 don sp1", c(MR_V40, vwap_anchor=42)),
    ("MR AUD_JPY v44 don sp1", c(MR_V40, vwap_anchor=44)),
    ("MR AUD_JPY v46 don sp1", c(MR_V40, vwap_anchor=46)),
    ("MR AUD_JPY v38 don sp1", c(MR_V40, vwap_anchor=38)),
    ("MR AUD_JPY v39 don sp1", c(MR_V40, vwap_anchor=39)),
    ("MR AUD_JPY v41 don sp1", c(MR_V40, vwap_anchor=41)),
    # ── 2. Short IS: can we get parity >> 1 like IWB? ─────────────────────────
    ("MR AUD_JPY v36 don is15k", c(MR_STAR, is_bars=15000)),
    ("MR AUD_JPY v36 don is10k", c(MR_STAR, is_bars=10000)),
    ("MR AUD_JPY v40 don is15k", c(MR_V40, is_bars=15000)),
    ("MR AUD_JPY v40 don is10k", c(MR_V40, is_bars=10000)),
    # ── 3. AUD_USD with donchian (scored 2.27 with vwap36) ────────────────────
    ("MR AUD_USD v36 don sp1", c(MR_STAR, instruments=["AUD_USD"])),
    ("MR AUD_USD v40 don sp1", c(MR_V40, instruments=["AUD_USD"])),
    (
        "MR AUD_USD v36 don sp0.5",
        c(MR_STAR, instruments=["AUD_USD"], spread_bps=0.5, slippage_bps=0.2),
    ),
    # ── 4. ML stacking on AUD_JPY H1 (novel: ML applied to FX) ───────────────
    # Use D bars instead since H1 label machinery differs
    ("AUD_JPY stacking D cbars5", c(ML_BASE, instruments=["AUD_JPY"])),
    (
        "AUD_JPY stacking D cbars3",
        c(
            ML_BASE,
            instruments=["AUD_JPY"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=3, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=3, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=3, confirm_pct=0.005),
            ],
        ),
    ),
    # ── 5. AUD_JPY vwap40 don + rsi_14_dev combo (both filters) ──────────────
    # Not directly supported but try if there's an "atr_only" + donchian variant
    ("MR AUD_JPY v40 rsi14dev", c(MR_V40, regime_filter="conf_rsi_14_dev")),
    # ── 6. Best known portfolio combinations ──────────────────────────────────
    (
        "Portfolio IWB+v40don",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB + AUD_JPY vwap40 donchian sp1",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
                dict(**c(MR_V40), weight=0.4),
            ],
        },
    ),
    (
        "Portfolio IWB60 v40don30 HYG10",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB60 + v40don30 + HYG_IWB10",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
                dict(**c(MR_V40), weight=0.3),
                dict(**XA_HYG_IWB, weight=0.1),
            ],
        },
    ),
    # ── 7. Confirm vwap36 don sp0.5 at oos9k (best combined search) ──────────
    ("MR AUD_JPY v36 don sp0.5 oos9k", c(MR_STAR, spread_bps=0.5, slippage_bps=0.2, oos_bars=9000)),
    # ── 8. Try vwap around the rsi_14_dev optimum with donchian ──────────────
    (
        "MR AUD_JPY v36 don rsi14 sp0.5",
        dict(
            strategy="mean_reversion",
            instruments=["AUD_JPY"],
            timeframe="H1",
            vwap_anchor=36,
            regime_filter="conf_rsi_14_dev",
            tier_grid="conservative",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=30000,
            oos_bars=9000,
        ),
    ),
    # ── 9. Completely new: USD_JPY with vwap40 donchian ───────────────────────
    ("MR USD_JPY v40 don sp1", c(MR_V40, instruments=["USD_JPY"])),
    (
        "MR EUR_USD v40 don sp0.5",
        c(MR_V40, instruments=["EUR_USD"], spread_bps=0.5, slippage_bps=0.2),
    ),
]


if __name__ == "__main__":
    print(f"Phase 7 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 7 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
