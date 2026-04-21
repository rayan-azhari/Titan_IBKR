"""Phase 2 autonomous research loop.

Explores beyond the fixed 52-experiment batch:
- Portfolio combinations of top 3 winners
- IWB stacking deeper param sweeps
- TCN stacking comparison
- AE stacking comparison
- More MR instruments and anchors
- Unexplored cross-asset combos
- IWB with longer training windows
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [4.5462]  # Confirmed IWB stacking cbars=5 baseline
MIN_IMPROVEMENT = 0.02


def write_exp(cfg: dict, desc: str):
    cfg = dict(cfg)
    cfg["description"] = desc
    EXP.write_text(f"def configure() -> dict:\n    return {repr(cfg)}\n", encoding="utf-8")


def git_commit(msg):
    subprocess.run(["git", "add", str(EXP)], cwd=ROOT, check=True)
    r = subprocess.run(
        ["git", "commit", "-m", f"exp: {msg}"], cwd=ROOT, capture_output=True, text=True
    )
    if r.returncode != 0 and "nothing to commit" not in r.stdout:
        # Not an error if nothing changed
        pass


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


# ── Base configs ──────────────────────────────────────────────────────────────

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

XA_BASE = dict(
    strategy="cross_asset",
    lookback=10,
    hold_days=10,
    threshold=0.50,
    is_days=504,
    oos_days=126,
    spread_bps=5.0,
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

PT_BASE = dict(
    strategy="pairs_trading",
    entry_z=2.0,
    exit_z=0.5,
    max_z=4.0,
    refit_window=126,
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


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


# ── Phase 2 experiments ───────────────────────────────────────────────────────

experiments = [
    # ── 1. Portfolio combinations ──────────────────────────────────────────────
    (
        "Portfolio IWB+AUDJPY+GLD",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB+AUDJPY+GLD",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.5),
                dict(**c(MR_BASE, instruments=["AUD_JPY"]), weight=0.3),
                dict(**c(XA_BASE, instruments=["GLD"], bond="IEF"), weight=0.2),
            ],
        },
    ),
    (
        "Portfolio IWB+AUDJPY",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB+AUDJPY",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.6),
                dict(**c(MR_BASE, instruments=["AUD_JPY"]), weight=0.4),
            ],
        },
    ),
    # ── 2. IWB stacking deeper sweep ──────────────────────────────────────────
    ("IWB stacking is_years=3", c(ML_BASE, instruments=["IWB"], is_years=3)),
    ("IWB stacking n_folds=5", c(ML_BASE, instruments=["IWB"], n_nested_folds=5)),
    ("IWB stacking threshold=0.55", c(ML_BASE, instruments=["IWB"], signal_threshold=0.55)),
    ("IWB stacking threshold=0.65", c(ML_BASE, instruments=["IWB"], signal_threshold=0.65)),
    ("IWB stacking threshold=0.70", c(ML_BASE, instruments=["IWB"], signal_threshold=0.70)),
    (
        "IWB stacking xgb depth=5",
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
        "IWB stacking xgb depth=3",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    (
        "IWB stacking xgb n500",
        c(
            ML_BASE,
            instruments=["IWB"],
            xgb_params=dict(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0,
            ),
        ),
    ),
    ("IWB stacking lookback=30", c(ML_BASE, instruments=["IWB"], lookback=30)),
    ("IWB stacking lookback=10", c(ML_BASE, instruments=["IWB"], lookback=10)),
    ("IWB stacking oos1", c(ML_BASE, instruments=["IWB"], oos_months=1)),
    (
        "IWB stacking 4labels",
        c(
            ML_BASE,
            instruments=["IWB"],
            label_params=[
                dict(rsi_oversold=45, rsi_overbought=55, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=50, rsi_overbought=50, confirm_bars=5, confirm_pct=0.003),
                dict(rsi_oversold=48, rsi_overbought=52, confirm_bars=5, confirm_pct=0.005),
                dict(rsi_oversold=43, rsi_overbought=57, confirm_bars=5, confirm_pct=0.008),
            ],
        ),
    ),
    # ── 3. TCN stacking on IWB ────────────────────────────────────────────────
    ("TCN stacking IWB", c(ML_BASE, strategy="tcn_stacking", instruments=["IWB"])),
    (
        "TCN stacking IWB lookback=30",
        c(ML_BASE, strategy="tcn_stacking", instruments=["IWB"], lookback=30),
    ),
    (
        "TCN stacking IWB lookback=10",
        c(ML_BASE, strategy="tcn_stacking", instruments=["IWB"], lookback=10),
    ),
    ("TCN stacking QQQ", c(ML_BASE, strategy="tcn_stacking", instruments=["QQQ"])),
    # ── 4. AE stacking on IWB ─────────────────────────────────────────────────
    (
        "AE stacking IWB k4",
        c(
            ML_BASE,
            strategy="ae_stacking",
            instruments=["IWB"],
            ae_latent_dim=8,
            ae_clusters=4,
            ae_epochs=100,
        ),
    ),
    (
        "AE stacking IWB k3",
        c(
            ML_BASE,
            strategy="ae_stacking",
            instruments=["IWB"],
            ae_latent_dim=8,
            ae_clusters=3,
            ae_epochs=100,
        ),
    ),
    (
        "AE stacking IWB k6",
        c(
            ML_BASE,
            strategy="ae_stacking",
            instruments=["IWB"],
            ae_latent_dim=8,
            ae_clusters=6,
            ae_epochs=100,
        ),
    ),
    (
        "AE stacking IWB latent16",
        c(
            ML_BASE,
            strategy="ae_stacking",
            instruments=["IWB"],
            ae_latent_dim=16,
            ae_clusters=4,
            ae_epochs=100,
        ),
    ),
    # ── 5. MR deeper sweep on AUD_JPY ─────────────────────────────────────────
    ("MR AUD_JPY vwap12", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=12)),
    ("MR AUD_JPY vwap48", c(MR_BASE, instruments=["AUD_JPY"], vwap_anchor=48)),
    (
        "MR AUD_JPY donchian",
        c(MR_BASE, instruments=["AUD_JPY"], regime_filter="conf_donchian_pos_20"),
    ),
    ("MR AUD_JPY no_filter", c(MR_BASE, instruments=["AUD_JPY"], regime_filter="no_filter")),
    ("MR AUD_JPY atr_only", c(MR_BASE, instruments=["AUD_JPY"], regime_filter="atr_only")),
    ("MR AUD_JPY is40k", c(MR_BASE, instruments=["AUD_JPY"], is_bars=40000, oos_bars=10000)),
    # ── 6. Cross-asset unexplored ─────────────────────────────────────────────
    ("XA LQD->GLD lb10", c(XA_BASE, instruments=["GLD"], bond="LQD")),
    ("XA IEF->GLD lb40", c(XA_BASE, instruments=["GLD"], bond="IEF", lookback=40, hold_days=20)),
    ("XA IEF->GLD thr0", c(XA_BASE, instruments=["GLD"], bond="IEF", threshold=0.0)),
    ("XA IEF->GLD thr0.25", c(XA_BASE, instruments=["GLD"], bond="IEF", threshold=0.25)),
    ("XA TLT->EEM lb10", c(XA_BASE, instruments=["EEM"], bond="TLT")),
    ("XA IEF->HYG lb20", c(XA_BASE, instruments=["HYG"], bond="IEF", lookback=20, hold_days=20)),
    ("XA TLT->IWB lb10", c(XA_BASE, instruments=["IWB"], bond="TLT")),
    # ── 7. Trend following unexplored ─────────────────────────────────────────
    ("TF IEF SMA150", c(TF_BASE, instruments=["IEF"], slow_ma=150)),
    ("TF TLT SMA150", c(TF_BASE, instruments=["TLT"], slow_ma=150)),
    ("TF QQQ EMA200", c(TF_BASE, instruments=["QQQ"], ma_type="EMA")),
    ("TF TQQQ SMA200", c(TF_BASE, instruments=["TQQQ"])),
    # ── 8. Pairs trading unexplored ───────────────────────────────────────────
    ("PT QQQ/SPY ez1.5", c(PT_BASE, instruments=["QQQ"], pair_b="SPY", entry_z=1.5)),
    ("PT QQQ/SPY ez2.5", c(PT_BASE, instruments=["QQQ"], pair_b="SPY", entry_z=2.5)),
    ("PT EFA/EEM", c(PT_BASE, instruments=["EFA"], pair_b="EEM")),
    ("PT TLT/IEF", c(PT_BASE, instruments=["TLT"], pair_b="IEF")),
    # ── 9. FX carry unexplored ────────────────────────────────────────────────
    ("FC AUD_JPY vol15", c(FC_BASE, instruments=["AUD_JPY"], vol_target_pct=0.15)),
    ("FC AUD_JPY vix20", c(FC_BASE, instruments=["AUD_JPY"], vix_halve_threshold=20.0)),
    ("FC AUD_JPY vix30", c(FC_BASE, instruments=["AUD_JPY"], vix_halve_threshold=30.0)),
    # ── 10. IWB + other multi-instrument ML ───────────────────────────────────
    ("ML IWB+EFA stacking", c(ML_BASE, instruments=["IWB", "EFA"])),
    ("ML IWB+GLD stacking", c(ML_BASE, instruments=["IWB", "GLD"])),
    ("ML IWB+TLT stacking", c(ML_BASE, instruments=["IWB", "TLT"])),
]


if __name__ == "__main__":
    print(f"Phase 2 autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 2 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    top = sorted(results, key=lambda x: -x[0])[:10]
    print("\nTop 10 this phase:")
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
