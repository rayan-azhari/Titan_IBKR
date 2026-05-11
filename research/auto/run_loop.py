"""Multi-strategy autonomous experiment runner.

Sweeps across all strategy types, instruments, and parameter variations.
Tracks per-strategy-type bests for portfolio construction.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [1.2630]  # Current best under rolling sanctuary (AUD_JPY MR v46 sp0.5 is32k oos8k)
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


# ==============================================================================
#  BASE CONFIGS PER STRATEGY TYPE
# ==============================================================================

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

GM_BASE = dict(
    strategy="gold_macro",
    real_rate_window=20,
    dollar_window=20,
    slow_ma=200,
    cost_bps=5.0,
    is_days=504,
    oos_days=126,
)

FC_BASE = dict(
    strategy="fx_carry",
    carry_direction=1,
    sma_period=50,
    vol_target_pct=0.08,
    vix_halve_threshold=25.0,
    spread_bps=3.0,
    slippage_bps=1.0,
    is_days=504,
    oos_days=126,
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


# ==============================================================================
#  EXPERIMENT DEFINITIONS
# ==============================================================================

experiments = [
    # -- Phase 1: Validate Known Winners --
    ("ML IWB stacking baseline", c(ML_BASE, instruments=["IWB"])),
    ("ML QQQ stacking oos3", c(ML_BASE, instruments=["QQQ"], oos_months=3)),
    ("ML SPY stacking oos3", c(ML_BASE, instruments=["SPY"], oos_months=3)),
    ("ML QQQ+SPY oos3", c(ML_BASE, instruments=["QQQ", "SPY"], oos_months=3)),
    # Mean Reversion (base now uses rsi_14_dev + conservative)
    ("MR AUD_JPY rsi_dev cons", c(MR_BASE, instruments=["AUD_JPY"])),
    ("MR EUR_USD rsi_dev cons", c(MR_BASE, instruments=["EUR_USD"])),
    (
        "MR EUR_USD donchian cons",
        c(MR_BASE, instruments=["EUR_USD"], regime_filter="conf_donchian_pos_20"),
    ),
    (
        "MR EUR_USD donchian std",
        c(
            MR_BASE,
            instruments=["EUR_USD"],
            regime_filter="conf_donchian_pos_20",
            tier_grid="standard",
        ),
    ),
    (
        "MR AUD_JPY donchian cons",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_donchian_pos_20"),
    ),
    ("MR GBP_USD rsi_dev cons", c(MR_BASE, instruments=["GBP_USD"])),
    ("MR USD_JPY rsi_dev cons", c(MR_BASE, instruments=["USD_JPY"])),
    ("MR AUD_USD rsi_dev cons", c(MR_BASE, instruments=["AUD_USD"])),
    ("MR USD_CHF rsi_dev cons", c(MR_BASE, instruments=["USD_CHF"])),
    # MR tuning
    ("MR EUR_USD aggressive", c(MR_BASE, instruments=["EUR_USD"], tier_grid="aggressive")),
    ("MR EUR_USD conservative", c(MR_BASE, instruments=["EUR_USD"], tier_grid="conservative")),
    ("MR EUR_USD vwap12", c(MR_BASE, instruments=["EUR_USD"], vwap_anchor=12)),
    ("MR EUR_USD vwap48", c(MR_BASE, instruments=["EUR_USD"], vwap_anchor=48)),
    # Trend Following
    ("TF SPY SMA200", c(TF_BASE, instruments=["SPY"])),
    ("TF SPY SMA150", c(TF_BASE, instruments=["SPY"], slow_ma=150)),
    ("TF SPY SMA100", c(TF_BASE, instruments=["SPY"], slow_ma=100)),
    ("TF SPY EMA200", c(TF_BASE, instruments=["SPY"], ma_type="EMA")),
    ("TF QQQ SMA200", c(TF_BASE, instruments=["QQQ"])),
    ("TF GLD SMA200", c(TF_BASE, instruments=["GLD"])),
    ("TF TLT SMA200", c(TF_BASE, instruments=["TLT"])),
    ("TF IWB SMA200", c(TF_BASE, instruments=["IWB"])),
    ("TF EFA SMA200", c(TF_BASE, instruments=["EFA"])),
    ("TF DBC SMA200", c(TF_BASE, instruments=["DBC"])),
    ("TF EEM SMA200", c(TF_BASE, instruments=["EEM"])),
    ("TF HYG SMA200", c(TF_BASE, instruments=["HYG"])),
    # Cross-Asset
    ("XA IEF->GLD lb20", c(XA_BASE, instruments=["GLD"], bond="IEF")),
    ("XA IEF->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="IEF", lookback=10, hold_days=10)),
    ("XA TLT->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="TLT", lookback=10, hold_days=10)),
    ("XA TLT->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="TLT", lookback=10, hold_days=10)),
    ("XA IEF->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="IEF", lookback=10, hold_days=10)),
    ("XA TLT->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="TLT", lookback=10, hold_days=10)),
    ("XA IEF->EFA lb20", c(XA_BASE, instruments=["EFA"], bond="IEF")),
    ("XA TLT->GLD lb60", c(XA_BASE, instruments=["GLD"], bond="TLT", lookback=60)),
    # Gold Macro
    ("GM GLD default", c(GM_BASE, instruments=["GLD"])),
    ("GM GLD rr10", c(GM_BASE, instruments=["GLD"], real_rate_window=10)),
    ("GM GLD rr40", c(GM_BASE, instruments=["GLD"], real_rate_window=40)),
    ("GM GLD sma100", c(GM_BASE, instruments=["GLD"], slow_ma=100)),
    # FX Carry
    ("FC AUD_JPY default", c(FC_BASE, instruments=["AUD_JPY"])),
    ("FC AUD_JPY sma20", c(FC_BASE, instruments=["AUD_JPY"], sma_period=20)),
    ("FC AUD_JPY sma100", c(FC_BASE, instruments=["AUD_JPY"], sma_period=100)),
    ("FC AUD_JPY vol10", c(FC_BASE, instruments=["AUD_JPY"], vol_target_pct=0.10)),
    ("FC AUD_USD default", c(FC_BASE, instruments=["AUD_USD"])),
    ("FC EUR_USD carry-1", c(FC_BASE, instruments=["EUR_USD"], carry_direction=-1)),
    # Pairs Trading (ETF-only per no-single-stock constraint)
    ("PT GLD/IAU", c(PT_BASE, instruments=["GLD"], pair_b="IAU")),
    ("PT SPY/QQQ", c(PT_BASE, instruments=["SPY"], pair_b="QQQ")),
    ("PT GLD/SLV", c(PT_BASE, instruments=["GLD"], pair_b="SLV")),
    ("PT IEF/TLT", c(PT_BASE, instruments=["IEF"], pair_b="TLT")),
    ("PT LQD/HYG", c(PT_BASE, instruments=["LQD"], pair_b="HYG")),
    ("PT GDX/GDXJ", c(PT_BASE, instruments=["GDX"], pair_b="GDXJ")),
    # -- Phase 2: rediscover champion direction under new sanctuary --
    # MR with donchian + spread_bps=0.5 sweep (current champion family)
    (
        "MR AUD_JPY v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR AUD_JPY v24 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=24,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR AUD_JPY v36 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=36,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR AUD_JPY v60 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=60,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR AUD_JPY v46 don sp0.5 is24k oos6k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=24000,
            oos_bars=6000,
        ),
    ),
    (
        "MR AUD_JPY v46 don sp0.5 is20k oos5k",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=20000,
            oos_bars=5000,
        ),
    ),
    # Try all FX pairs with the donchian+sp0.5 champion recipe
    (
        "MR EUR_USD v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["EUR_USD"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR GBP_USD v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["GBP_USD"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR AUD_USD v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["AUD_USD"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR USD_JPY v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["USD_JPY"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    (
        "MR USD_CHF v46 don sp0.5 is32k",
        c(
            MR_BASE,
            instruments=["USD_CHF"],
            vwap_anchor=46,
            regime_filter="conf_donchian_pos_20",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=32000,
            oos_bars=8000,
        ),
    ),
    # Cross-asset new pairs
    ("XA IEF->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="IEF", lookback=10, hold_days=10)),
    ("XA TLT->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="TLT", lookback=10, hold_days=10)),
    ("XA HYG->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="HYG", lookback=10, hold_days=10)),
    ("XA HYG->QQQ lb10", c(XA_BASE, instruments=["QQQ"], bond="HYG", lookback=10, hold_days=10)),
    ("XA HYG->SPY lb10", c(XA_BASE, instruments=["SPY"], bond="HYG", lookback=10, hold_days=10)),
    ("XA LQD->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="LQD", lookback=10, hold_days=10)),
    ("XA TIP->GLD lb20", c(XA_BASE, instruments=["GLD"], bond="TIP", lookback=20, hold_days=20)),
    ("XA TIP->GLD lb60", c(XA_BASE, instruments=["GLD"], bond="TIP", lookback=60, hold_days=60)),
    # Trend on more ETFs w/ multiple MAs
    ("TF GLD SMA100", c(TF_BASE, instruments=["GLD"], slow_ma=100)),
    ("TF GLD SMA150", c(TF_BASE, instruments=["GLD"], slow_ma=150)),
    ("TF DBC SMA150", c(TF_BASE, instruments=["DBC"], slow_ma=150)),
    ("TF SLV SMA200", c(TF_BASE, instruments=["SLV"])),
    ("TF TQQQ SMA200", c(TF_BASE, instruments=["TQQQ"])),
    ("TF IWB SMA100", c(TF_BASE, instruments=["IWB"], slow_ma=100)),
    ("TF GDX SMA200", c(TF_BASE, instruments=["GDX"])),
]

if __name__ == "__main__":
    print(f"Multi-strategy autoresearch loop. Baseline BEST={BEST[0]:.4f}")
    print(f"Running {len(experiments)} experiments across 7 strategy types...\n")

    for desc, config in experiments:
        try:
            run_exp(desc, config)
        except Exception as e:
            print(f"[{desc}] ERROR: {e}")
            try:
                git_reset()
            except Exception:
                pass

    print(f"\n=== COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
