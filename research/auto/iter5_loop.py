"""iter5: novel regime filter signals + extreme parameters.

After iter1-4 plateau (218 exps, 139 consecutive no-improvements), probe:
- MR with novel regime filter signals (stoch_k_dev, williams_r_dev, accel_rsi14,
  mom_accel_combo, donchian_rsi) -- only rsi_14_dev and donchian_pos_20 tried prior
- MR with extremely short vwap_anchor (4/6/8) -- pure-noise MR test
- Triple-instrument ML stacking (IWB+QQQ+SPY, IWB+TLT+GLD)
- Very short is_bars for recent-regime edges on MR (10000/3000, 6000/2000)
- Ultra-long XGBoost (n=1000, depth=2, lr=0.01)
- 4-5 strategy portfolio combos
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


experiments = [
    # -- MR AUD_JPY with novel regime filter signals --
    (
        "MR AUD_JPY v24 conf_stoch_k",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_stoch_k_dev"),
    ),
    (
        "MR AUD_JPY v24 conf_williams",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_williams_r_dev"),
    ),
    (
        "MR AUD_JPY v24 conf_accel",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_accel_rsi14"),
    ),
    (
        "MR AUD_JPY v24 conf_mom_accel",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_mom_accel_combo"),
    ),
    (
        "MR AUD_JPY v24 conf_donchian_rsi",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_donchian_rsi"),
    ),
    (
        "MR AUD_JPY v24 conf_rsi_7",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_rsi_7_dev"),
    ),
    (
        "MR AUD_JPY v24 conf_rsi_21",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_rsi_21_dev"),
    ),
    (
        "MR AUD_JPY v24 conf_ma_spread",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_ma_spread_5_20"),
    ),
    # -- MR with extremely short vwap anchor (near-instantaneous VWAP) --
    ("MR AUD_JPY v4 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=4)),
    ("MR AUD_JPY v6 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=6)),
    ("MR AUD_JPY v8 don sp0.5", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=8)),
    # -- MR with very short IS for recent-regime edges --
    (
        "MR AUD_JPY v24 is10k oos3k",
        c(MR_BASE, instruments=["AUD_JPY"], is_bars=10000, oos_bars=3000),
    ),
    (
        "MR AUD_JPY v24 is6k oos2k",
        c(MR_BASE, instruments=["AUD_JPY"], is_bars=6000, oos_bars=2000),
    ),
    (
        "MR AUD_JPY v24 is15k oos4k",
        c(MR_BASE, instruments=["AUD_JPY"], is_bars=15000, oos_bars=4000),
    ),
    # -- EUR_USD + donchian + v24 with novel regime filters --
    (
        "MR EUR_USD v24 conf_stoch_k",
        c(MR_BASE, instruments=["EUR_USD"], regime_filter="conf_stoch_k_dev"),
    ),
    (
        "MR EUR_USD v24 conf_williams",
        c(MR_BASE, instruments=["EUR_USD"], regime_filter="conf_williams_r_dev"),
    ),
    # -- Triple-instrument ML stacking --
    ("ML IWB+QQQ+SPY stacking", c(ML_BASE, instruments=["IWB", "QQQ", "SPY"])),
    ("ML IWB+TLT+GLD stacking", c(ML_BASE, instruments=["IWB", "TLT", "GLD"])),
    ("ML IWB+EFA+EEM stacking", c(ML_BASE, instruments=["IWB", "EFA", "EEM"])),
    # -- Extreme XGBoost hyperparameters --
    (
        "ML IWB xgb_n1000_depth2_lr01",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=1000,
                max_depth=2,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    (
        "ML IWB xgb_n100_depth6_lr1",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    (
        "ML IWB xgb_subsample=0.5",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.5,
                colsample_bytree=0.4,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    # -- Portfolio: 4 strategies, inverse-vol-ish allocation --
    (
        "Portfolio 4x (IWB ML 40 + AJ MR 20 + HYG-QQQ 20 + TIP-HYG 20)",
        dict(
            strategy="portfolio",
            instruments=["IWB"],
            timeframe="D",
            strategies=[
                c(ML_BASE, instruments=["IWB"], weight=0.4),
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
                c(
                    XA_BASE,
                    instruments=["QQQ"],
                    bond="HYG",
                    lookback=10,
                    hold_days=10,
                    weight=0.2,
                ),
                c(
                    XA_BASE,
                    instruments=["HYG"],
                    bond="TIP",
                    lookback=60,
                    hold_days=40,
                    weight=0.2,
                ),
            ],
        ),
    ),
    # -- More PT variants with ETF-only constraint --
    # Pairs_trading requires manual exploration across pair_b
    (
        "PT LQD/IEF",
        dict(
            strategy="pairs_trading",
            instruments=["LQD"],
            pair_b="IEF",
            entry_z=2.0,
            exit_z=0.5,
            max_z=4.0,
            refit_window=126,
            is_days=504,
            oos_days=126,
        ),
    ),
    (
        "PT DBC/GLD",
        dict(
            strategy="pairs_trading",
            instruments=["DBC"],
            pair_b="GLD",
            entry_z=2.0,
            exit_z=0.5,
            max_z=4.0,
            refit_window=126,
            is_days=504,
            oos_days=126,
        ),
    ),
    (
        "PT EEM/EFA",
        dict(
            strategy="pairs_trading",
            instruments=["EEM"],
            pair_b="EFA",
            entry_z=2.0,
            exit_z=0.5,
            max_z=4.0,
            refit_window=126,
            is_days=504,
            oos_days=126,
        ),
    ),
    # -- ML on ETF that hasn't been direct target --
    ("ML IWB+HYG stacking", c(ML_BASE, instruments=["IWB", "HYG"])),
    # -- MR on AUD_JPY with standard tier_grid + v24 + new filters --
    (
        "MR AUD_JPY v24 std conf_stoch",
        c(
            MR_BASE,
            instruments=["AUD_JPY"],
            regime_filter="conf_stoch_k_dev",
            tier_grid="standard",
        ),
    ),
]


if __name__ == "__main__":
    print(f"iter5 autoresearch loop. BEST={BEST[0]:.4f}")
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
    print(f"\n=== ITER5 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
