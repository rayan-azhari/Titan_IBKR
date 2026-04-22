"""iter4: final plateau-confirmation probes.

After iter1-3 (181 experiments, BEST=4.5462 ML IWB), exhaust remaining
untested hypothesis families:
- Longer IS windows (is_years=3,4)
- Much wider RSI label bands
- M5 FX mean reversion (untested timeframe)
- Weekly FX mean reversion
- GLD/SLV/GDX pairs-trading and cross-asset permutations
- VIX-based cross-asset signals
- HYG as signal → LQD/TIP targets (credit-to-credit)
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
        ["uv", "run", "python", str(EVAL)], cwd=ROOT, capture_output=True, text=True, timeout=600
    )
    m = {}
    for line in (r.stdout + r.stderr).splitlines():
        for k in ["SCORE", "SHARPE", "MAX_DD", "PARITY", "TRADES", "WORST_FOLD", "N_FOLDS"]:
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
    print(
        f"[{desc}] SCORE={score:.4f} SH={m.get('SHARPE', '?')} "
        f"DD={m.get('MAX_DD', '?')} PAR={m.get('PARITY', '?')} "
        f"TRD={m.get('TRADES', '?')} WF={m.get('WORST_FOLD', '?')} -> {status}"
    )
    sys.stdout.flush()
    if improved:
        BEST[0] = score
        print(f"  *** NEW BEST: {BEST[0]:.4f} ***")
    else:
        git_reset()
    return score


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

XA_BASE = dict(
    strategy="cross_asset",
    lookback=20,
    hold_days=20,
    threshold=0.50,
    is_days=504,
    oos_days=126,
    spread_bps=5.0,
)

MR_BASE = dict(
    strategy="mean_reversion",
    timeframe="H1",
    vwap_anchor=24,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=0.5,
    slippage_bps=0.2,
    is_bars=32000,
    oos_bars=8000,
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


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # -- Longer IS windows on IWB ML --
    ("ML IWB stacking is_years=3", c(ML_BASE, instruments=["IWB"], is_years=3)),
    ("ML IWB stacking is_years=4", c(ML_BASE, instruments=["IWB"], is_years=4)),
    # -- Wider label bands (more candidate entries) --
    (
        "ML IWB stacking rsi35/65",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=35, rsi_overbought=65, confirm_bars=5, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "ML IWB stacking rsi30/70",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=30, rsi_overbought=70, confirm_bars=5, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "ML IWB stacking rsi42/58",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=42, rsi_overbought=58, confirm_bars=5, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "ML IWB stacking 4-label ensemble",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=40, rsi_overbought=60, confirm_bars=5, confirm_pct=0.01),
            ],
        ),
    ),
    # confirm_pct sweep
    (
        "ML IWB stacking confirm_pct=0.002",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.002),
            ],
        ),
    ),
    (
        "ML IWB stacking confirm_pct=0.01",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.01),
            ],
        ),
    ),
    # -- M5 FX mean reversion (completely new timeframe) --
    (
        "MR EUR_USD M5 v24",
        c(MR_BASE, instruments=["EUR_USD"], timeframe="M5", is_bars=50000, oos_bars=12000),
    ),
    (
        "MR EUR_USD M5 v48",
        c(
            MR_BASE,
            instruments=["EUR_USD"],
            timeframe="M5",
            vwap_anchor=48,
            is_bars=50000,
            oos_bars=12000,
        ),
    ),
    (
        "MR USD_CHF M5 v24",
        c(MR_BASE, instruments=["USD_CHF"], timeframe="M5", is_bars=50000, oos_bars=12000),
    ),
    # -- New pairs combinations (ETF-only) --
    ("PT SLV/SIL", c(PT_BASE, instruments=["SLV"], pair_b="SIL")),
    ("PT SLV/SIVR", c(PT_BASE, instruments=["SLV"], pair_b="SIVR")),
    ("PT GLD/GDX", c(PT_BASE, instruments=["GLD"], pair_b="GDX")),
    ("PT SPY/IWB", c(PT_BASE, instruments=["SPY"], pair_b="IWB")),
    ("PT QQQ/IWB", c(PT_BASE, instruments=["QQQ"], pair_b="IWB")),
    ("PT TIP/IEF", c(PT_BASE, instruments=["TIP"], pair_b="IEF")),
    ("PT EFA/IWB", c(PT_BASE, instruments=["EFA"], pair_b="IWB")),
    # -- Cross-asset: HYG → credit-adjacent targets, TIP → HYG --
    ("XA HYG->LQD lb10", c(XA_BASE, instruments=["LQD"], bond="HYG", lookback=10, hold_days=10)),
    ("XA HYG->TIP lb10", c(XA_BASE, instruments=["TIP"], bond="HYG", lookback=10, hold_days=10)),
    ("XA TIP->HYG lb10", c(XA_BASE, instruments=["HYG"], bond="TIP", lookback=10, hold_days=10)),
    ("XA TIP->HYG lb60", c(XA_BASE, instruments=["HYG"], bond="TIP", lookback=60, hold_days=40)),
    ("XA TLT->EFA lb10", c(XA_BASE, instruments=["EFA"], bond="TLT", lookback=10, hold_days=10)),
    ("XA TLT->EEM lb10", c(XA_BASE, instruments=["EEM"], bond="TLT", lookback=10, hold_days=10)),
    # -- GLD→ gold miners --
    ("XA GLD->GDX lb10", c(XA_BASE, instruments=["GDX"], bond="GLD", lookback=10, hold_days=10)),
    ("XA GLD->GDXJ lb10", c(XA_BASE, instruments=["GDXJ"], bond="GLD", lookback=10, hold_days=10)),
    ("XA SLV->SIL lb10", c(XA_BASE, instruments=["SIL"], bond="SLV", lookback=10, hold_days=10)),
    # -- Bond-on-bond cross asset (TLT signals IEF, etc) --
    ("XA TLT->IEF lb10", c(XA_BASE, instruments=["IEF"], bond="TLT", lookback=10, hold_days=10)),
    ("XA IEF->TLT lb10", c(XA_BASE, instruments=["TLT"], bond="IEF", lookback=10, hold_days=10)),
    ("XA IEF->LQD lb10", c(XA_BASE, instruments=["LQD"], bond="IEF", lookback=10, hold_days=10)),
    # -- UUP/DXY as cross-asset signal (dollar strength → risk) --
    ("XA UUP->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="UUP", lookback=10, hold_days=10)),
    ("XA UUP->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="UUP", lookback=10, hold_days=10)),
    # -- Cost sensitivity far from baseline on champion --
    ("ML IWB cost_bps=5.0", c(ML_BASE, instruments=["IWB"], cost_bps=5.0)),
    ("ML IWB cost_bps=0.5", c(ML_BASE, instruments=["IWB"], cost_bps=0.5)),
    # -- Larger XGBoost --
    (
        "ML IWB stacking xgb_n500",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    (
        "ML IWB stacking xgb_depth=5",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    (
        "ML IWB stacking xgb_lr=0.05",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
]


if __name__ == "__main__":
    print(f"iter4 autoresearch loop. BEST={BEST[0]:.4f}")
    print(f"Running {len(experiments)} experiments...\n")
    for desc, config in experiments:
        try:
            run_exp(desc, config)
        except Exception as e:
            print(f"[{desc}] ERROR: {e}")
            try:
                git_reset()
            except Exception:
                pass
    print(f"\n=== ITER4 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
