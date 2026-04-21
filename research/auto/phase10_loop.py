"""Phase 10 (final) autonomous research loop.

Phase 9 confirmed: AUD_JPY MR landscape is fully mapped.
Two peaks with donchian filter:
  Peak 1: vwap=46 → sp0.5=4.59, sp0=4.69 (BEST)
  Peak 2: vwap=56 → sp0.5=4.50, sp0=4.61

BEST = 4.6910 (v46 don sp0, zero cost)
Best realistic-cost = 4.5905 (v46 don sp0.5)

Phase 10 goals:
1. Try sp=0.25 at both peaks (v46, v56) — bridging the gap
2. Verify v46 don sp0.5 is robust by checking it at different periods
   - Use is_bars=28000 (slight variation), oos_bars=8000
3. Final portfolio optimization: find optimal IWB/AUD_JPY weight split
4. Try one completely fresh angle: MR on AUD_JPY with H4 timeframe
5. Wrap up — this is the final phase
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP = ROOT / "research/auto/experiment.py"
EVAL = ROOT / "research/auto/evaluate.py"
BEST = [4.6910]
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

MR_V46_SP0 = dict(
    strategy="mean_reversion",
    instruments=["AUD_JPY"],
    timeframe="H1",
    vwap_anchor=46,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=0.0,
    slippage_bps=0.0,
    is_bars=30000,
    oos_bars=7500,
)

MR_V46_REAL = dict(
    strategy="mean_reversion",
    instruments=["AUD_JPY"],
    timeframe="H1",
    vwap_anchor=46,
    regime_filter="conf_donchian_pos_20",
    tier_grid="conservative",
    spread_bps=0.5,
    slippage_bps=0.2,
    is_bars=30000,
    oos_bars=7500,
)


def c(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


experiments = [
    # ── 1. sp=0.25 at both peaks ──────────────────────────────────────────────
    ("MR AUD_JPY v46 sp0.25", c(MR_V46_SP0, spread_bps=0.25, slippage_bps=0.1)),
    ("MR AUD_JPY v56 sp0.25", c(MR_V46_SP0, vwap_anchor=56, spread_bps=0.25, slippage_bps=0.1)),
    ("MR AUD_JPY v56 sp0", c(MR_V46_SP0, vwap_anchor=56)),
    # ── 2. Verify v46 sp0.5 with slight is/oos variation ─────────────────────
    ("MR AUD_JPY v46 sp0.5 is28k oos8k", c(MR_V46_REAL, is_bars=28000, oos_bars=8000)),
    ("MR AUD_JPY v46 sp0.5 is32k oos8k", c(MR_V46_REAL, is_bars=32000, oos_bars=8000)),
    # ── 3. Portfolio weight sweep: IWB + AUD/JPY (v46 sp0.5) ─────────────────
    (
        "Portfolio IWB50+AJ50",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB50 + AUD_JPY v46 don sp0.5",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.5),
                dict(**MR_V46_REAL, weight=0.5),
            ],
        },
    ),
    (
        "Portfolio IWB70+AJ30",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB70 + AUD_JPY v46 don sp0.5",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.7),
                dict(**MR_V46_REAL, weight=0.3),
            ],
        },
    ),
    (
        "Portfolio IWB40+AJ40+HYG20",
        {
            "strategy": "portfolio",
            "description": "Portfolio IWB40 + AJ40 + HYG_IWB20",
            "strategies": [
                dict(**c(ML_BASE, instruments=["IWB"]), weight=0.4),
                dict(**MR_V46_REAL, weight=0.4),
                dict(
                    strategy="cross_asset",
                    instruments=["IWB"],
                    bond="HYG",
                    lookback=10,
                    hold_days=10,
                    threshold=0.50,
                    is_days=504,
                    oos_days=126,
                    spread_bps=5.0,
                    weight=0.2,
                ),
            ],
        },
    ),
    # ── 4. H4 timeframe for AUD/JPY (novel: 4-hour bars) ─────────────────────
    (
        "MR AUD_JPY H4 v12 don",
        dict(
            strategy="mean_reversion",
            instruments=["AUD_JPY"],
            timeframe="H4",
            vwap_anchor=12,
            regime_filter="conf_donchian_pos_20",
            tier_grid="conservative",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=7500,
            oos_bars=1875,
        ),
    ),
    (
        "MR AUD_JPY H4 v6 don",
        dict(
            strategy="mean_reversion",
            instruments=["AUD_JPY"],
            timeframe="H4",
            vwap_anchor=6,
            regime_filter="conf_donchian_pos_20",
            tier_grid="conservative",
            spread_bps=0.5,
            slippage_bps=0.2,
            is_bars=7500,
            oos_bars=1875,
        ),
    ),
    # ── 5. Extreme vwap values on v46 sp0 (can we push past 4.69?) ───────────
    ("MR AUD_JPY v43 sp0", c(MR_V46_SP0, vwap_anchor=43)),
    ("MR AUD_JPY v45 sp0", c(MR_V46_SP0, vwap_anchor=45)),
    ("MR AUD_JPY v47 sp0", c(MR_V46_SP0, vwap_anchor=47)),
    # ── 6. IWB stacking with is_years=2.5 (half-year more training) ──────────
    # Note: is_years is typically int but let's see if float works
    ("IWB stacking is2.5y", c(ML_BASE, instruments=["IWB"], is_years=2.5)),
]


if __name__ == "__main__":
    print(f"Phase 10 (FINAL) autonomous research. Baseline BEST={BEST[0]:.4f}")
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

    print(f"\n=== PHASE 10 COMPLETE. FINAL BEST={BEST[0]:.4f} ===")
    print("\nFull session summary:")
    top = sorted(results, key=lambda x: -x[0])[:10]
    for score, desc, m in top:
        print(f"  {score:.4f}  {desc}")
    print("\nAutoresearch session complete after 10 phases (~400+ experiments).")
