"""Phase 3 autonomous research loop.

Phase 2 finding: IWB stacking is fully converged at 4.5462 (same 5 trades regardless
of model architecture, lookback, threshold, or XGB params). XGBoost dominates — LSTM
and TCN are irrelevant. The 5 trades are the same across all variants.

Phase 3 hypotheses:
- Different random seeds in XGBoost (might select different 5 trades)
- lstm_e2e strategy type on IWB (different learning paradigm)
- Unexplored equity instruments: TQQQ, EFA, EEM, DBC, HYG, LQD standalone
- IWB stacking with completely different label geometry (cbars != 5)
- MR: NZD_USD and CAD-based pairs
- MR with much larger oos_bars (longer OOS)
- Cross-asset with HYG->GLD, LQD->QQQ
- Gold macro tuned params
- Combine XA IEF->GLD lb40 (1.46) with threshold sweep
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

MR_BASE = dict(
    strategy="mean_reversion", timeframe="H1",
    vwap_anchor=24, regime_filter="conf_rsi_14_dev",
    tier_grid="conservative", spread_bps=2.0, slippage_bps=1.0,
    is_bars=30000, oos_bars=7500,
)

XA_BASE = dict(
    strategy="cross_asset",
    lookback=10, hold_days=10, threshold=0.50,
    is_days=504, oos_days=126, spread_bps=5.0,
)

TF_BASE = dict(
    strategy="trend_following", timeframe="D",
    slow_ma=200, ma_type="SMA", cost_bps=2.0,
    is_days=504, oos_days=126,
)

GM_BASE = dict(
    strategy="gold_macro", instruments=["GLD"],
    real_rate_window=20, dollar_window=20, slow_ma=200,
    cost_bps=5.0, is_days=504, oos_days=126,
)

FC_BASE = dict(
    strategy="fx_carry",
    carry_direction=1, sma_period=50,
    vol_target_pct=0.08, vix_halve_threshold=25.0,
    spread_bps=3.0, slippage_bps=1.0,
    is_days=504, oos_days=126,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # ── 1. Different XGBoost seeds ─────────────────────────────────────────────
    # Phase 2 showed the same 5 trades at seed=42. Different seeds may find new trades.
    ("IWB stacking seed=7", c(ML_BASE, instruments=["IWB"],
                              xgb_params=dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                                              subsample=0.8, colsample_bytree=0.6,
                                              random_state=7, verbosity=0))),
    ("IWB stacking seed=123", c(ML_BASE, instruments=["IWB"],
                                xgb_params=dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                                                subsample=0.8, colsample_bytree=0.6,
                                                random_state=123, verbosity=0))),
    ("IWB stacking seed=999", c(ML_BASE, instruments=["IWB"],
                                xgb_params=dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                                                subsample=0.8, colsample_bytree=0.6,
                                                random_state=999, verbosity=0))),

    # ── 2. lstm_e2e on IWB (end-to-end LSTM without XGBoost) ──────────────────
    ("IWB lstm_e2e", c(ML_BASE, strategy="lstm_e2e", instruments=["IWB"])),
    ("IWB lstm_e2e hidden=64", c(ML_BASE, strategy="lstm_e2e", instruments=["IWB"],
                                 lstm_hidden=64)),

    # ── 3. Different label geometry on IWB ────────────────────────────────────
    # cbars=5 is the winner. Try nearby values with cbars as only variant.
    ("IWB cbars=4", c(ML_BASE, instruments=["IWB"],
                      label_params=[
                          dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=4, confirm_pct=0.005),
                          dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=4, confirm_pct=0.003),
                          dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=4, confirm_pct=0.005),
                      ])),
    ("IWB cbars=6", c(ML_BASE, instruments=["IWB"],
                      label_params=[
                          dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=6, confirm_pct=0.005),
                          dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=6, confirm_pct=0.003),
                          dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=6, confirm_pct=0.005),
                      ])),
    ("IWB cbars=7", c(ML_BASE, instruments=["IWB"],
                      label_params=[
                          dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=7, confirm_pct=0.005),
                          dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=7, confirm_pct=0.003),
                          dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=7, confirm_pct=0.005),
                      ])),
    ("IWB confirm_pct=0.003", c(ML_BASE, instruments=["IWB"],
                                label_params=[
                                    dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.003),
                                    dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
                                    dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.003),
                                ])),
    ("IWB confirm_pct=0.008", c(ML_BASE, instruments=["IWB"],
                                label_params=[
                                    dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.008),
                                    dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.008),
                                    dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.008),
                                ])),

    # ── 4. Other equity instruments with stacking ─────────────────────────────
    ("TQQQ stacking cbars5", c(ML_BASE, instruments=["TQQQ"])),
    ("EFA stacking cbars5", c(ML_BASE, instruments=["EFA"])),
    ("EEM stacking cbars5", c(ML_BASE, instruments=["EEM"])),
    ("HYG stacking cbars5", c(ML_BASE, instruments=["HYG"])),
    ("LQD stacking cbars5", c(ML_BASE, instruments=["LQD"])),
    ("DBC stacking cbars5", c(ML_BASE, instruments=["DBC"])),
    ("GDX stacking cbars5", c(ML_BASE, instruments=["GDX"])),
    ("TIP stacking cbars5", c(ML_BASE, instruments=["TIP"])),
    ("GLD stacking cbars5", c(ML_BASE, instruments=["GLD"])),

    # ── 5. Multi-instrument with TQQQ (previously best non-IWB at 2.85) ───────
    ("IWB+TQQQ stacking", c(ML_BASE, instruments=["IWB", "TQQQ"])),
    ("TQQQ cbars5 oos3", c(ML_BASE, instruments=["TQQQ"], oos_months=3)),

    # ── 6. MR new instruments ─────────────────────────────────────────────────
    ("MR GBP_JPY rsi_dev", c(MR_BASE, instruments=["GBP_JPY"])),
    ("MR NZD_USD rsi_dev", c(MR_BASE, instruments=["NZD_USD"])),
    ("MR EUR_GBP rsi_dev", c(MR_BASE, instruments=["EUR_GBP"])),
    ("MR AUD_JPY oos10k", c(MR_BASE, instruments=["AUD_JPY"], oos_bars=10000)),
    ("MR AUD_JPY spread1", c(MR_BASE, instruments=["AUD_JPY"],
                             spread_bps=1.0, slippage_bps=0.5)),
    ("MR AUD_JPY D", c(MR_BASE, instruments=["AUD_JPY"], timeframe="D",
                       is_bars=1500, oos_bars=375)),

    # ── 7. Cross-asset deeper ─────────────────────────────────────────────────
    # IEF->GLD lb40 scored 1.46 (best XA so far). Try fine-tuning.
    ("XA IEF->GLD lb40 hd20", c(XA_BASE, instruments=["GLD"], bond="IEF",
                                lookback=40, hold_days=20)),
    ("XA IEF->GLD lb30 hd15", c(XA_BASE, instruments=["GLD"], bond="IEF",
                                lookback=30, hold_days=15)),
    ("XA IEF->GLD lb20 thr0", c(XA_BASE, instruments=["GLD"], bond="IEF",
                                lookback=20, hold_days=20, threshold=0.0)),
    ("XA HYG->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="HYG")),
    ("XA LQD->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="LQD")),
    ("XA IEF->DBC lb20", c(XA_BASE, instruments=["DBC"], bond="IEF",
                           lookback=20, hold_days=20)),

    # ── 8. Gold macro fine-tuning ─────────────────────────────────────────────
    ("GM GLD rr10 dw10", c(GM_BASE, real_rate_window=10, dollar_window=10)),
    ("GM GLD rr10 sma100", c(GM_BASE, real_rate_window=10, slow_ma=100)),
    ("GM GLD rr20 dw10", c(GM_BASE, dollar_window=10)),
    ("GM GLD cost2", c(GM_BASE, cost_bps=2.0)),
    ("GM GLD is1008", c(GM_BASE, is_days=1008, oos_days=252)),

    # ── 9. Trend following fine-tuning ────────────────────────────────────────
    ("TF SPY SMA150 cost1", c(TF_BASE, instruments=["SPY"], slow_ma=150, cost_bps=1.0)),
    ("TF GLD EMA200", c(TF_BASE, instruments=["GLD"], ma_type="EMA")),
    ("TF GLD SMA150", c(TF_BASE, instruments=["GLD"], slow_ma=150)),
    ("TF IWB SMA150", c(TF_BASE, instruments=["IWB"], slow_ma=150)),

    # ── 10. FX carry on new pairs ─────────────────────────────────────────────
    ("FC NZD_JPY carry", c(FC_BASE, instruments=["NZD_JPY"])),
    ("FC GBP_JPY carry", c(FC_BASE, instruments=["GBP_JPY"])),
    ("FC AUD_JPY sma_200", c(FC_BASE, instruments=["AUD_JPY"], sma_period=200)),
]


if __name__ == "__main__":
    print(f"Phase 3 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 3 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase (by score):")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
