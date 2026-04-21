"""Phase 5 autonomous research loop.

Phase 4 findings:
- AUD_JPY MR vwap36 spread=1 -> 3.2902 (best MR ever, was 2.60 baseline)
- XA HYG->IWB lb10 -> 3.0375 (new: high-yield spread predicts broad US equity)
- XA HYG->QQQ lb10 -> 2.8566 (same signal, QQQ)
- TQQQ cbars=6 -> 2.9479 (beats cbars=5 at 2.85)

Phase 5 hypotheses:
1. Fine-tune AUD_JPY vwap36 sp1: sweep vwap around 36, oos_bars, regime filter
2. Fine-tune HYG->IWB: lb, hold_days, threshold, spread
3. HYG cross-asset on other targets: EFA, EEM, DBC, SPY
4. TQQQ cbars=6 fine-tuning: oos, is_years, signal_threshold
5. Portfolio of best 3: IWB + AUD_JPY_vwap36_sp1 + HYG->IWB
6. Can HYG->IWB be a new strategy #2 (score > 3)?
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

# AUD_JPY best lead: vwap36, spread=1
MR_AJ_LEAD = dict(
    strategy="mean_reversion",
    instruments=["AUD_JPY"],
    timeframe="H1",
    vwap_anchor=36,
    regime_filter="conf_rsi_14_dev",
    tier_grid="conservative",
    spread_bps=1.0,
    slippage_bps=0.5,
    is_bars=30000,
    oos_bars=7500,
)

# HYG->IWB best lead
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

XA_HYG_QQQ = dict(
    strategy="cross_asset",
    instruments=["QQQ"],
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
    # ── 1. AUD_JPY vwap36 sp1 fine-tuning ─────────────────────────────────────
    ("MR AUD_JPY vwap30 sp1", c(MR_AJ_LEAD, vwap_anchor=30)),
    ("MR AUD_JPY vwap32 sp1", c(MR_AJ_LEAD, vwap_anchor=32)),
    ("MR AUD_JPY vwap34 sp1", c(MR_AJ_LEAD, vwap_anchor=34)),
    ("MR AUD_JPY vwap38 sp1", c(MR_AJ_LEAD, vwap_anchor=38)),
    ("MR AUD_JPY vwap40 sp1", c(MR_AJ_LEAD, vwap_anchor=40)),
    ("MR AUD_JPY vwap36 oos6k", c(MR_AJ_LEAD, oos_bars=6000)),
    ("MR AUD_JPY vwap36 oos9k", c(MR_AJ_LEAD, oos_bars=9000)),
    ("MR AUD_JPY vwap36 is25k", c(MR_AJ_LEAD, is_bars=25000)),
    ("MR AUD_JPY vwap36 is35k", c(MR_AJ_LEAD, is_bars=35000)),
    ("MR AUD_JPY vwap36 donchian", c(MR_AJ_LEAD, regime_filter="conf_donchian_pos_20")),
    ("MR AUD_JPY vwap36 no_filter", c(MR_AJ_LEAD, regime_filter="no_filter")),
    ("MR AUD_JPY vwap36 standard", c(MR_AJ_LEAD, tier_grid="standard")),
    ("MR AUD_JPY vwap36 sp0.5", c(MR_AJ_LEAD, spread_bps=0.5, slippage_bps=0.2)),
    # ── 2. HYG->IWB fine-tuning (best: 3.04) ─────────────────────────────────
    ("XA HYG->IWB lb5 hd5", c(XA_HYG_IWB, lookback=5, hold_days=5)),
    ("XA HYG->IWB lb15 hd15", c(XA_HYG_IWB, lookback=15, hold_days=15)),
    ("XA HYG->IWB lb20 hd20", c(XA_HYG_IWB, lookback=20, hold_days=20)),
    ("XA HYG->IWB lb10 thr0", c(XA_HYG_IWB, threshold=0.0)),
    ("XA HYG->IWB lb10 thr0.25", c(XA_HYG_IWB, threshold=0.25)),
    ("XA HYG->IWB lb10 thr0.75", c(XA_HYG_IWB, threshold=0.75)),
    ("XA HYG->IWB sp2", c(XA_HYG_IWB, spread_bps=2.0)),
    ("XA HYG->IWB sp1", c(XA_HYG_IWB, spread_bps=1.0)),
    ("XA HYG->IWB is756", c(XA_HYG_IWB, is_days=756, oos_days=189)),
    ("XA HYG->IWB is1008", c(XA_HYG_IWB, is_days=1008, oos_days=252)),
    # ── 3. HYG cross-asset on other targets ───────────────────────────────────
    ("XA HYG->SPY lb10", c(XA_HYG_QQQ, instruments=["SPY"])),
    ("XA HYG->GLD lb10", c(XA_HYG_QQQ, instruments=["GLD"])),
    ("XA HYG->EFA lb10", c(XA_HYG_QQQ, instruments=["EFA"])),
    ("XA HYG->EEM lb10", c(XA_HYG_QQQ, instruments=["EEM"])),
    ("XA HYG->DBC lb10", c(XA_HYG_QQQ, instruments=["DBC"])),
    (
        "XA LQD->IWB sp1",
        dict(
            strategy="cross_asset",
            instruments=["IWB"],
            bond="LQD",
            lookback=10,
            hold_days=10,
            threshold=0.50,
            is_days=504,
            oos_days=126,
            spread_bps=1.0,
        ),
    ),
    # ── 4. TQQQ cbars=6 fine-tuning ───────────────────────────────────────────
    (
        "TQQQ cbars6 oos1",
        c(
            ML_BASE,
            instruments=["TQQQ"],
            oos_months=1,
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=6, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=6, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=6, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "TQQQ cbars6 is3y",
        c(
            ML_BASE,
            instruments=["TQQQ"],
            is_years=3,
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=6, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=6, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=6, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "TQQQ cbars6 thr0.55",
        c(
            ML_BASE,
            instruments=["TQQQ"],
            signal_threshold=0.55,
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=6, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=6, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=6, confirm_pct=0.005),
            ],
        ),
    ),
    # ── 5. Portfolio: IWB + AUD_JPY_vwap36_sp1 + HYG->IWB ───────────────────
    (
        "Portfolio IWB+AJ_vwap36+HYG_IWB",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB + AUD_JPY vwap36 sp1 + HYG->IWB",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.5),
                dict(**c(MR_AJ_LEAD), weight=0.3),
                dict(**c(XA_HYG_IWB), weight=0.2),
            ],
        },
    ),
    (
        "Portfolio IWB+HYG_IWB",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB + HYG->IWB",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
                dict(**c(XA_HYG_IWB), weight=0.4),
            ],
        },
    ),
    # ── 6. Radical: IWB stacking with HYG as second instrument ───────────────
    ("IWB+HYG stacking", c(ML_BASE, instruments=["IWB", "HYG"])),
]


if __name__ == "__main__":
    print(f"Phase 5 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 5 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
