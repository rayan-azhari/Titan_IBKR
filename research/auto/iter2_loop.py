"""iter2: targeted variations around iter1 winners.

Runs after iter1 establishes BEST. Populated on the assumption that
the IWB stacking family will be the champion (SCORE ~4.55) -- we sweep
variants around it plus fresh plays in neighbouring families.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [4.5462]  # placeholder; re-read from iter1 before launch
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

MR_BASE = dict(
    strategy="mean_reversion",
    timeframe="H1",
    vwap_anchor=24,  # iter1: v24 beat v46 under sanctuary (1.66 vs 1.26)
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=0.5,
    slippage_bps=0.2,
    is_bars=32000,
    oos_bars=8000,
)

TF_BASE = dict(
    strategy="trend_following",
    timeframe="D",
    slow_ma=200,
    ma_type="SMA",
    cost_bps=2.0,
    is_days=504,
    oos_days=126,
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


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


# Ideas targeting the IWB ML family + orthogonal single-instrument probes.
experiments = [
    # IWB stacking variants
    ("ML IWB tcn_stacking", c(ML_BASE, instruments=["IWB"], strategy="tcn_stacking")),
    ("ML IWB ae_stacking", c(ML_BASE, instruments=["IWB"], strategy="ae_stacking")),
    ("ML IWB stacking threshold=0.55", c(ML_BASE, instruments=["IWB"], signal_threshold=0.55)),
    ("ML IWB stacking threshold=0.65", c(ML_BASE, instruments=["IWB"], signal_threshold=0.65)),
    ("ML IWB stacking threshold=0.70", c(ML_BASE, instruments=["IWB"], signal_threshold=0.70)),
    ("ML IWB stacking is_years=1.5", c(ML_BASE, instruments=["IWB"], is_years=1.5)),
    ("ML IWB stacking is_years=2.5", c(ML_BASE, instruments=["IWB"], is_years=2.5)),
    ("ML IWB stacking cost_bps=3.0", c(ML_BASE, instruments=["IWB"], cost_bps=3.0)),
    ("ML IWB stacking cost_bps=1.0", c(ML_BASE, instruments=["IWB"], cost_bps=1.0)),
    ("ML IWB stacking lookback=15", c(ML_BASE, instruments=["IWB"], lookback=15)),
    ("ML IWB stacking lookback=30", c(ML_BASE, instruments=["IWB"], lookback=30)),
    ("ML IWB stacking lstm_hidden=64", c(ML_BASE, instruments=["IWB"], lstm_hidden=64)),
    ("ML IWB stacking lstm_epochs=50", c(ML_BASE, instruments=["IWB"], lstm_epochs=50)),
    ("ML IWB stacking n_nested=5", c(ML_BASE, instruments=["IWB"], n_nested_folds=5)),
    ("ML IWB stacking oos_months=3", c(ML_BASE, instruments=["IWB"], oos_months=3)),
    ("ML IWB stacking oos_months=4", c(ML_BASE, instruments=["IWB"], oos_months=4)),
    # Label param tweaks (wider RSI bands, shorter confirm)
    (
        "ML IWB stacking rsi40/60 cbars=5",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=40, rsi_overbought=60, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
            ],
        ),
    ),
    (
        "ML IWB stacking cbars=10",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=10, confirm_pct=0.005),
            ],
        ),
    ),
    (
        "ML IWB stacking cbars=3",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=3, confirm_pct=0.005),
            ],
        ),
    ),
    # Try IWB + IEF (US broad equity + bond) dual-instrument
    ("ML IWB+IEF stacking", c(ML_BASE, instruments=["IWB", "IEF"])),
    ("ML IWB+TLT stacking", c(ML_BASE, instruments=["IWB", "TLT"])),
    # -- AUD_JPY MR deep sweep around v24 (iter1 winner under sanctuary) --
    ("MR AUD_JPY v20 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=20)),
    ("MR AUD_JPY v22 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=22)),
    ("MR AUD_JPY v26 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=26)),
    ("MR AUD_JPY v28 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=28)),
    ("MR AUD_JPY v30 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=30)),
    ("MR AUD_JPY v16 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=16)),
    ("MR AUD_JPY v12 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=12)),
    # Non-donchian regime filter on AUD_JPY v24
    (
        "MR AUD_JPY v24 rsi_dev sp0.5",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_rsi_14_dev"),
    ),
    ("MR AUD_JPY v24 atr sp0.5", c(MR_BASE, instruments=["AUD_JPY"], regime_filter="atr_only")),
    # Tier grid sweep on v24
    ("MR AUD_JPY v24 standard", c(MR_BASE, instruments=["AUD_JPY"], tier_grid="standard")),
    ("MR AUD_JPY v24 aggressive", c(MR_BASE, instruments=["AUD_JPY"], tier_grid="aggressive")),
    # Window sweep on v24
    (
        "MR AUD_JPY v24 is28k oos7k",
        c(MR_BASE, instruments=["AUD_JPY"], is_bars=28000, oos_bars=7000),
    ),
    (
        "MR AUD_JPY v24 is40k oos10k",
        c(MR_BASE, instruments=["AUD_JPY"], is_bars=40000, oos_bars=10000),
    ),
    # AUD_USD with v24 recipe
    ("MR AUD_USD v24 don sp0.5", c(MR_BASE, instruments=["AUD_USD"])),
    # -- Cross-asset HYG->equity deep sweep (iter1 winners ~1.35-1.40) --
    ("XA HYG->QQQ lb5", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=5, hold_days=5)),
    ("XA HYG->QQQ lb15", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=15, hold_days=15)),
    ("XA HYG->QQQ lb20", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=20, hold_days=20)),
    (
        "XA HYG->QQQ lb10 hold20",
        c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10, hold_days=20),
    ),
    (
        "XA HYG->QQQ lb20 hold10",
        c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=20, hold_days=10),
    ),
    (
        "XA HYG->QQQ lb10 th0.25",
        c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10, hold_days=10, threshold=0.25),
    ),
    (
        "XA HYG->QQQ lb10 th0.00",
        c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10, hold_days=10, threshold=0.00),
    ),
    (
        "XA HYG->QQQ lb10 th0.75",
        c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10, hold_days=10, threshold=0.75),
    ),
    (
        "XA HYG->QQQ lb10 sp2.0",
        c(
            XA_BASE,
            instruments=["QQQ"],
            bond="HYG",
            lookback=10,
            hold_days=10,
            spread_bps=2.0,
        ),
    ),
    # HYG -> IWB variations
    ("XA HYG->IWB lb5", c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=5, hold_days=5)),
    ("XA HYG->IWB lb15", c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=15, hold_days=15)),
    (
        "XA HYG->IWB lb10 th0.25",
        c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=10, hold_days=10, threshold=0.25),
    ),
    (
        "XA HYG->IWB lb10 th0.00",
        c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=10, hold_days=10, threshold=0.00),
    ),
    # HYG -> SPY
    ("XA HYG->SPY lb5", c(XA_BASE, instruments=["SPY"], bond="HYG", lookback=5, hold_days=5)),
    (
        "XA HYG->SPY lb10 th0.25",
        c(XA_BASE, instruments=["SPY"], bond="HYG", lookback=10, hold_days=10, threshold=0.25),
    ),
    # HYG -> TQQQ (leveraged version -- may amplify signal)
    ("XA HYG->TQQQ lb10", c(XA_BASE, instruments=["TQQQ"], bond="HYG", lookback=10, hold_days=10)),
    # Portfolio: IWB ML + HYG->QQQ + AUD_JPY MR
    (
        "Portfolio IWB-ML 60 + HYG-QQQ 20 + AJ-MR 20",
        dict(
            strategy="portfolio",
            instruments=["IWB"],
            timeframe="D",
            strategies=[
                c(ML_BASE, instruments=["IWB"], weight=0.6),
                c(
                    XA_BASE,
                    instruments=["QQQ"],
                    bond="HYG",
                    lookback=10,
                    hold_days=10,
                    weight=0.2,
                ),
                dict(
                    strategy="mean_reversion",
                    instruments=["AUD_JPY"],
                    timeframe="H1",
                    vwap_anchor=24,
                    regime_filter="conf_donchian_pos_20",
                    tier_grid="conservative",
                    spread_bps=0.5,
                    slippage_bps=0.2,
                    is_bars=32000,
                    oos_bars=8000,
                    weight=0.2,
                ),
            ],
        ),
    ),
    # Trend multi-asset -- try EMA, different slow MAs
    ("TF GLD EMA100", c(TF_BASE, instruments=["GLD"], slow_ma=100, ma_type="EMA")),
    ("TF GLD EMA150", c(TF_BASE, instruments=["GLD"], slow_ma=150, ma_type="EMA")),
    ("TF TQQQ SMA100", c(TF_BASE, instruments=["TQQQ"], slow_ma=100)),
    ("TF TQQQ EMA100", c(TF_BASE, instruments=["TQQQ"], slow_ma=100, ma_type="EMA")),
    ("TF TQQQ SMA50", c(TF_BASE, instruments=["TQQQ"], slow_ma=50)),
    ("TF TQQQ EMA50", c(TF_BASE, instruments=["TQQQ"], slow_ma=50, ma_type="EMA")),
]


if __name__ == "__main__":
    print(f"iter2 autoresearch loop. BEST={BEST[0]:.4f}")
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
    print(f"\n=== ITER2 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
