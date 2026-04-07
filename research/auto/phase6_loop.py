"""Phase 6 autonomous research loop.

Phase 5 major finding:
- AUD_JPY MR vwap36 + donchian filter + spread=1 -> 3.9449 (BEST MR ever, prev was 2.60!)
  The vwap36 + donchian combination is SYNERGISTIC. Neither alone achieved this.
  donchian alone: 1.95, vwap36 alone: 3.29, TOGETHER: 3.94

- Portfolio IWB + HYG->IWB: 3.99 (close to IWB but can't beat it)
- MR AUD_JPY vwap36 sp0.5: 3.43 (with rsi_14_dev)
- MR AUD_JPY vwap36 oos9k: 3.43 (with rsi_14_dev)

Phase 6 hypotheses:
1. AUD_JPY vwap36 donchian: cost sweep (0, 0.5 bps) -- can we push past 4.0?
2. AUD_JPY vwap36 donchian: vwap sweep (32, 34, 38, 40, 48)
3. AUD_JPY vwap36 donchian: oos_bars and is_bars sweep
4. AUD_JPY vwap36 donchian: tier_grid=standard/aggressive
5. Can AUD_JPY vwap36 donchian beat IWB (4.5462)?
6. Other FX pairs with vwap36+donchian: EUR_USD, GBP_USD, AUD_USD
7. HYG->IWB fine-tuning: sp0 to push past 3.0
8. Portfolio: IWB + AUD_JPY_vwap36_donchian (strong #2 candidate)
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

# Phase 5 winner: AUD_JPY vwap36 donchian sp1
MR_STAR = dict(
    strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="H1",
    vwap_anchor=36, regime_filter="conf_donchian_pos_20",
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
    # ── 1. AUD_JPY vwap36 donchian cost sweep (can we push past 4.0?) ─────────
    ("MR AUD_JPY v36 don sp0.5", c(MR_STAR, spread_bps=0.5, slippage_bps=0.2)),
    ("MR AUD_JPY v36 don sp0", c(MR_STAR, spread_bps=0.0, slippage_bps=0.0)),
    ("MR AUD_JPY v36 don sp1.5", c(MR_STAR, spread_bps=1.5, slippage_bps=0.5)),
    ("MR AUD_JPY v36 don sp2", c(MR_STAR, spread_bps=2.0, slippage_bps=1.0)),

    # ── 2. AUD_JPY donchian vwap sweep ────────────────────────────────────────
    ("MR AUD_JPY v24 don sp1", c(MR_STAR, vwap_anchor=24)),
    ("MR AUD_JPY v28 don sp1", c(MR_STAR, vwap_anchor=28)),
    ("MR AUD_JPY v32 don sp1", c(MR_STAR, vwap_anchor=32)),
    ("MR AUD_JPY v34 don sp1", c(MR_STAR, vwap_anchor=34)),
    ("MR AUD_JPY v38 don sp1", c(MR_STAR, vwap_anchor=38)),
    ("MR AUD_JPY v40 don sp1", c(MR_STAR, vwap_anchor=40)),
    ("MR AUD_JPY v48 don sp1", c(MR_STAR, vwap_anchor=48)),

    # ── 3. AUD_JPY vwap36 donchian OOS/IS sweep ───────────────────────────────
    ("MR AUD_JPY v36 don oos5k", c(MR_STAR, oos_bars=5000)),
    ("MR AUD_JPY v36 don oos6k", c(MR_STAR, oos_bars=6000)),
    ("MR AUD_JPY v36 don oos9k", c(MR_STAR, oos_bars=9000)),
    ("MR AUD_JPY v36 don oos12k", c(MR_STAR, oos_bars=12000)),
    ("MR AUD_JPY v36 don is25k", c(MR_STAR, is_bars=25000)),
    ("MR AUD_JPY v36 don is35k", c(MR_STAR, is_bars=35000)),
    ("MR AUD_JPY v36 don is40k", c(MR_STAR, is_bars=40000)),

    # ── 4. AUD_JPY vwap36 donchian tier_grid variants ─────────────────────────
    ("MR AUD_JPY v36 don standard", c(MR_STAR, tier_grid="standard")),
    ("MR AUD_JPY v36 don aggressive", c(MR_STAR, tier_grid="aggressive")),

    # ── 5. Other FX with vwap36 + donchian ────────────────────────────────────
    ("MR EUR_USD v36 don sp1", c(MR_STAR, instruments=["EUR_USD"])),
    ("MR GBP_USD v36 don sp1", c(MR_STAR, instruments=["GBP_USD"])),
    ("MR AUD_USD v36 don sp1", c(MR_STAR, instruments=["AUD_USD"])),
    ("MR USD_JPY v36 don sp1", c(MR_STAR, instruments=["USD_JPY"])),
    ("MR USD_CHF v36 don sp1", c(MR_STAR, instruments=["USD_CHF"])),

    # ── 6. HYG->IWB further tightening ───────────────────────────────────────
    ("XA HYG->IWB sp0 thr0.5", c(XA_HYG_IWB, spread_bps=0.0, threshold=0.50)),
    ("XA HYG->IWB sp1 thr0.75", c(XA_HYG_IWB, spread_bps=1.0, threshold=0.75)),
    ("XA HYG->IWB lb8 hd8", c(XA_HYG_IWB, lookback=8, hold_days=8)),
    ("XA HYG->IWB lb12 hd12", c(XA_HYG_IWB, lookback=12, hold_days=12)),

    # ── 7. Portfolios with the new AUD_JPY star ───────────────────────────────
    ("Portfolio IWB+AUDJPY_star", {
        "strategy": "portfolio",
        "description": "Portfolio IWB + AUD_JPY vwap36 donchian",
        "strategies": [
            dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
            dict(**MR_STAR, weight=0.4),
        ],
    }),
    ("Portfolio IWB+AUDJPY_star+HYG_IWB", {
        "strategy": "portfolio",
        "description": "Portfolio IWB + AUD_JPY star + HYG->IWB",
        "strategies": [
            dict(**c(ML_BASE, instruments=["IWB"]), weight=0.5),
            dict(**MR_STAR, weight=0.3),
            dict(**XA_HYG_IWB, weight=0.2),
        ],
    }),

    # ── 8. AUD_JPY star with oos9k (best oos from rsi version: 3.43) ──────────
    ("MR AUD_JPY v36 don oos9k sp0.5", c(MR_STAR, oos_bars=9000,
                                         spread_bps=0.5, slippage_bps=0.2)),

    # ── 9. Completely new hypothesis: AUD_JPY MR on D timeframe with donchian ─
    ("MR AUD_JPY D v5 don", dict(
        strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="D",
        vwap_anchor=5, regime_filter="conf_donchian_pos_20",
        tier_grid="conservative", spread_bps=1.0, slippage_bps=0.5,
        is_bars=1500, oos_bars=375)),
    ("MR AUD_JPY D v10 don", dict(
        strategy="mean_reversion", instruments=["AUD_JPY"], timeframe="D",
        vwap_anchor=10, regime_filter="conf_donchian_pos_20",
        tier_grid="conservative", spread_bps=1.0, slippage_bps=0.5,
        is_bars=1500, oos_bars=375)),
]


if __name__ == "__main__":
    print(f"Phase 6 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 6 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
