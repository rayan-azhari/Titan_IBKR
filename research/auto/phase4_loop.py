"""Phase 4 autonomous research loop.

Phase 3 findings:
- IWB stacking is fully converged at 4.5462 (same 5 trades regardless of seed/arch/params)
- New lead: AUD_JPY MR with spread=1bps scored 2.91 (best non-IWB)
- XA LQD->QQQ lb10 scored 1.55 (new cross-asset winner)
- TQQQ stacking scored 2.85 (promising leveraged variant)

Phase 4 hypotheses:
1. AUD_JPY MR cost sweep: tight spreads may push toward 3.0+
2. AUD_JPY MR with vwap=36, different oos_bars
3. TQQQ stacking: longer IS, different thresholds
4. XA LQD->QQQ fine-tuning: different lookback/hold/threshold
5. New cross-asset: LQD->IWB, HYG->QQQ
6. IWB stacking: try xgboost-only (no LSTM meta) via strategy='xgboost'
7. AUD_JPY MR + LQD->QQQ portfolio (both new leads)
8. XA IEF->GLD with spread reduction (currently 5bps - expensive)
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

# Best MR lead from Phase 3 (spread=1)
MR_LEAD = dict(
    strategy="mean_reversion",
    timeframe="H1",
    vwap_anchor=24,
    regime_filter="conf_rsi_14_dev",
    tier_grid="conservative",
    spread_bps=1.0,
    slippage_bps=0.5,
    is_bars=30000,
    oos_bars=7500,
)

XA_BASE = dict(
    strategy="cross_asset",
    lookback=10,
    hold_days=10,
    threshold=0.50,
    is_days=504,
    oos_days=126,
    spread_bps=5.0,
)

# Best XA lead from Phase 3 (LQD->QQQ)
XA_LEAD = dict(
    strategy="cross_asset",
    instruments=["QQQ"],
    bond="LQD",
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
    # ── 1. AUD_JPY MR cost sweep (Phase 3 lead: spread=1 scored 2.91) ─────────
    ("MR AUD_JPY spread0.5", c(MR_BASE, instruments=["AUD_JPY"], spread_bps=0.5, slippage_bps=0.2)),
    ("MR AUD_JPY spread0", c(MR_BASE, instruments=["AUD_JPY"], spread_bps=0.0, slippage_bps=0.0)),
    ("MR AUD_JPY spread1.5", c(MR_BASE, instruments=["AUD_JPY"], spread_bps=1.5, slippage_bps=0.5)),
    # ── 2. AUD_JPY MR param sweep around the spread=1 lead ───────────────────
    ("MR AUD_JPY sp1 vwap12", c(MR_LEAD, instruments=["AUD_JPY"], vwap_anchor=12)),
    ("MR AUD_JPY sp1 vwap36", c(MR_LEAD, instruments=["AUD_JPY"], vwap_anchor=36)),
    ("MR AUD_JPY sp1 vwap48", c(MR_LEAD, instruments=["AUD_JPY"], vwap_anchor=48)),
    ("MR AUD_JPY sp1 oos5k", c(MR_LEAD, instruments=["AUD_JPY"], oos_bars=5000)),
    ("MR AUD_JPY sp1 oos12k", c(MR_LEAD, instruments=["AUD_JPY"], oos_bars=12000)),
    ("MR AUD_JPY sp1 is20k", c(MR_LEAD, instruments=["AUD_JPY"], is_bars=20000)),
    ("MR AUD_JPY sp1 is40k", c(MR_LEAD, instruments=["AUD_JPY"], is_bars=40000)),
    ("MR AUD_JPY sp1 standard", c(MR_LEAD, instruments=["AUD_JPY"], tier_grid="standard")),
    (
        "MR AUD_JPY sp1 donchian",
        c(MR_LEAD, instruments=["AUD_JPY"], regime_filter="conf_donchian_pos_20"),
    ),
    ("MR AUD_JPY sp1 no_filter", c(MR_LEAD, instruments=["AUD_JPY"], regime_filter="no_filter")),
    # ── 3. Other FX pairs with spread=1 ──────────────────────────────────────
    ("MR EUR_USD sp1", c(MR_LEAD, instruments=["EUR_USD"])),
    ("MR GBP_USD sp1", c(MR_LEAD, instruments=["GBP_USD"])),
    ("MR AUD_USD sp1", c(MR_LEAD, instruments=["AUD_USD"])),
    # ── 4. XA LQD->QQQ fine-tuning (Phase 3 lead: 1.55) ─────────────────────
    ("XA LQD->QQQ lb20 hd20", c(XA_LEAD, lookback=20, hold_days=20)),
    ("XA LQD->QQQ lb5 hd5", c(XA_LEAD, lookback=5, hold_days=5)),
    ("XA LQD->QQQ lb10 thr0", c(XA_LEAD, threshold=0.0)),
    ("XA LQD->QQQ lb10 thr0.25", c(XA_LEAD, threshold=0.25)),
    ("XA LQD->QQQ sp2", c(XA_LEAD, spread_bps=2.0)),
    ("XA LQD->QQQ sp1", c(XA_LEAD, spread_bps=1.0)),
    ("XA LQD->QQQ is756", c(XA_LEAD, is_days=756, oos_days=189)),
    ("XA LQD->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="LQD")),
    ("XA HYG->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="HYG")),
    ("XA HYG->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="HYG")),
    # ── 5. IWB xgboost-only (no LSTM meta-learner) ───────────────────────────
    ("IWB xgboost only", c(ML_BASE, strategy="xgboost", instruments=["IWB"])),
    # ── 6. TQQQ deeper sweep (Phase 3: 2.85) ─────────────────────────────────
    ("TQQQ stacking is3y", c(ML_BASE, instruments=["TQQQ"], is_years=3)),
    ("TQQQ stacking oos1", c(ML_BASE, instruments=["TQQQ"], oos_months=1)),
    ("TQQQ stacking thr0.55", c(ML_BASE, instruments=["TQQQ"], signal_threshold=0.55)),
    (
        "TQQQ cbars=6",
        c(
            ML_BASE,
            instruments=["TQQQ"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=6, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=6, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=6, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "TQQQ cbars=4",
        c(
            ML_BASE,
            instruments=["TQQQ"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=4, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=4, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=4, confirm_pct=0.005),
            ],
        ),
    ),
    # ── 7. Portfolio: new leads combined ─────────────────────────────────────
    (
        "Portfolio IWB+AUDJPY sp1+LQDQQQ",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB+AUDJPY_sp1+LQD->QQQ",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.5),
                dict(**c(MR_LEAD, instruments=["AUD_JPY"]), weight=0.3),
                dict(**c(XA_LEAD), weight=0.2),
            ],
        },
    ),
    # ── 8. IEF->GLD with lower spread ────────────────────────────────────────
    ("XA IEF->GLD sp2", c(XA_BASE, instruments=["GLD"], bond="IEF", spread_bps=2.0)),
    ("XA IEF->GLD sp1", c(XA_BASE, instruments=["GLD"], bond="IEF", spread_bps=1.0)),
    ("XA IEF->GLD sp0", c(XA_BASE, instruments=["GLD"], bond="IEF", spread_bps=0.0)),
    # ── 9. Trend following with lower costs ───────────────────────────────────
    (
        "TF SPY SMA200 cost1",
        c(
            dict(
                strategy="trend_following",
                timeframe="D",
                slow_ma=200,
                ma_type="SMA",
                cost_bps=1.0,
                is_days=504,
                oos_days=126,
            ),
            instruments=["SPY"],
        ),
    ),
    (
        "TF GLD SMA200 cost2",
        c(
            dict(
                strategy="trend_following",
                timeframe="D",
                slow_ma=200,
                ma_type="SMA",
                cost_bps=2.0,
                is_days=504,
                oos_days=126,
            ),
            instruments=["GLD"],
        ),
    ),
    # ── 10. IWB with MUCH wider RSI bands (radical change) ───────────────────
    (
        "IWB rsi40/60",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=40, rsi_overbought=60, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.003),
                dict(rsi_oversold=42, rsi_overbought=58, confirm_bars=5, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "IWB rsi35/65",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=35, rsi_overbought=65, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=40, rsi_overbought=60, confirm_bars=5, confirm_pct=0.003),
                dict(rsi_oversold=37, rsi_overbought=63, confirm_bars=5, confirm_pct=0.005),
            ],
        ),
    ),
]


if __name__ == "__main__":
    print(f"Phase 4 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 4 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nTop 10 this phase:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
