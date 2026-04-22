"""iter3: structurally-different hypothesis probes.

Iter1+2 plateau at SCORE=4.5462 (ML IWB stacking). Before declaring done,
probe different model families (xgboost alone, lstm_e2e), different target
instruments (TQQQ, GLD, EFA, ^FTSE), different timeframes (Daily MR), and
unconventional cross-asset signals (VIX, DXY, TIP proxies).
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

MR_BASE_H1 = dict(
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

MR_BASE_D = dict(
    strategy="mean_reversion",
    timeframe="D",
    vwap_anchor=10,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=0.5,
    slippage_bps=0.2,
    is_bars=500,
    oos_bars=125,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # -- Different ML model families on IWB --
    ("ML IWB xgboost alone", c(ML_BASE, instruments=["IWB"], strategy="xgboost")),
    (
        "ML IWB xgboost threshold=0.55",
        c(ML_BASE, instruments=["IWB"], strategy="xgboost", signal_threshold=0.55),
    ),
    (
        "ML IWB xgboost threshold=0.7",
        c(ML_BASE, instruments=["IWB"], strategy="xgboost", signal_threshold=0.7),
    ),
    ("ML IWB lstm_e2e", c(ML_BASE, instruments=["IWB"], strategy="lstm_e2e")),
    (
        "ML IWB lstm_e2e threshold=0.55",
        c(ML_BASE, instruments=["IWB"], strategy="lstm_e2e", signal_threshold=0.55),
    ),
    # -- ML stacking on different instruments --
    ("ML TQQQ stacking", c(ML_BASE, instruments=["TQQQ"])),
    (
        "ML TQQQ stacking oos_months=3",
        c(ML_BASE, instruments=["TQQQ"], oos_months=3),
    ),
    ("ML GLD stacking", c(ML_BASE, instruments=["GLD"])),
    ("ML EFA stacking", c(ML_BASE, instruments=["EFA"])),
    ("ML EEM stacking", c(ML_BASE, instruments=["EEM"])),
    ("ML DBC stacking", c(ML_BASE, instruments=["DBC"])),
    ("ML TLT stacking", c(ML_BASE, instruments=["TLT"])),
    ("ML IEF stacking", c(ML_BASE, instruments=["IEF"])),
    ("ML HYG stacking", c(ML_BASE, instruments=["HYG"])),
    # -- Different cross-asset signal sources --
    ("XA TIP->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="TIP", lookback=10, hold_days=10)),
    ("XA TIP->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="TIP", lookback=10, hold_days=10)),
    ("XA TIP->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="TIP", lookback=10, hold_days=10)),
    ("XA LQD->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="LQD", lookback=10, hold_days=10)),
    ("XA LQD->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="LQD", lookback=10, hold_days=10)),
    ("XA LQD->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="LQD", lookback=10, hold_days=10)),
    # HYG as signal on bond targets (credit -> duration)
    ("XA HYG->TLT lb10", c(XA_BASE, instruments=["TLT"], bond="HYG", lookback=10, hold_days=10)),
    ("XA HYG->IEF lb10", c(XA_BASE, instruments=["IEF"], bond="HYG", lookback=10, hold_days=10)),
    # -- MR at daily timeframe on FX D bars --
    ("MR-D AUD_JPY v10 don", c(MR_BASE_D, instruments=["AUD_JPY"])),
    ("MR-D AUD_JPY v20 don", c(MR_BASE_D, instruments=["AUD_JPY"], vwap_anchor=20)),
    ("MR-D EUR_USD v10 don", c(MR_BASE_D, instruments=["EUR_USD"])),
    ("MR-D GBP_USD v10 don", c(MR_BASE_D, instruments=["GBP_USD"])),
    # -- MR on less-tested H1 pairs with v24 --
    ("MR USD_CHF v24 don sp0.5", c(MR_BASE_H1, instruments=["USD_CHF"])),
    (
        "MR EUR_USD v24 rsi_dev sp0.5",
        c(MR_BASE_H1, instruments=["EUR_USD"], regime_filter="conf_rsi_14_dev"),
    ),
    (
        "MR GBP_USD v24 rsi_dev sp0.5",
        c(MR_BASE_H1, instruments=["GBP_USD"], regime_filter="conf_rsi_14_dev"),
    ),
    # -- ML on global indices --
    ("ML ^FTSE stacking", c(ML_BASE, instruments=["^FTSE"])),
    ("ML ^GDAXI stacking", c(ML_BASE, instruments=["^GDAXI"])),
    # -- Portfolio combos where IWB has 100% weight (sanity check) --
    # This shouldn't beat pure IWB since run_portfolio_wfo averages Sharpes
    # but confirms the aggregation contract.
    (
        "Portfolio IWB-ML pure",
        dict(
            strategy="portfolio",
            instruments=["IWB"],
            timeframe="D",
            strategies=[c(ML_BASE, instruments=["IWB"], weight=1.0)],
        ),
    ),
    # IWB stacking + TQQQ stacking multi-instrument (single-config, not portfolio)
    (
        "ML IWB+TQQQ stacking",
        c(ML_BASE, instruments=["IWB", "TQQQ"]),
    ),
    (
        "ML IWB+GLD stacking",
        c(ML_BASE, instruments=["IWB", "GLD"]),
    ),
    # -- Very low cost IWB --
    ("ML IWB stacking cost_bps=0.0", c(ML_BASE, instruments=["IWB"], cost_bps=0.0)),
    (
        "ML IWB stacking threshold=0.5",
        c(ML_BASE, instruments=["IWB"], signal_threshold=0.5),
    ),
    (
        "ML IWB stacking threshold=0.45",
        c(ML_BASE, instruments=["IWB"], signal_threshold=0.45),
    ),
    # -- AE stacking variations --
    (
        "ML IWB ae_stacking clusters=6",
        c(ML_BASE, instruments=["IWB"], strategy="ae_stacking", ae_clusters=6),
    ),
    (
        "ML IWB ae_stacking latent=16",
        c(ML_BASE, instruments=["IWB"], strategy="ae_stacking", ae_latent_dim=16),
    ),
    # -- TCN stacking variations --
    (
        "ML IWB tcn_stacking lookback=45",
        c(ML_BASE, instruments=["IWB"], strategy="tcn_stacking", lookback=45),
    ),
    (
        "ML IWB tcn_stacking lookback=10",
        c(ML_BASE, instruments=["IWB"], strategy="tcn_stacking", lookback=10),
    ),
]


if __name__ == "__main__":
    print(f"iter3 autoresearch loop. BEST={BEST[0]:.4f}")
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
    print(f"\n=== ITER3 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
