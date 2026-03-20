"""run_equity_longonly_pipeline.py
----------------------------------
Full Phase 3->5 pipeline for all daily equity parquets, long-only, with regime
filter sweep (no-gate, ADX<25, HMM low-vol state).

Phase 3  : VectorBT long-only backtest, threshold sweep [0.25..2.00],
           regime gate sweep [None, ADX<25, HMM], signal=rsi_21_dev.
Phase 4  : Rolling WFO (IS=504, OOS=126, 5 folds) on Phase 3 passers.
           HMM is re-fit on each fold's IS window independently (no lookahead).
Phase 5  : Monte Carlo gate on Phase 4 passers.

Regime gates:
  None    -- no filter (raw signal)
  ADX<25  -- only trade when ADX <= 25 (ranging/low-trend regime)
  HMM     -- 2-state Gaussian HMM; only trade in State 0 (low volatility,
             mean-reverting regime). Fitted on IS bars, predicted on full
             sequence using IS-fitted model parameters.

Outputs (all in .tmp/reports/):
  equity_longonly_phase3.csv   -- per-symbol best config + IS/OOS stats
  equity_longonly_phase4.csv   -- WFO fold stats for Phase 4 passers
  equity_longonly_phase5.csv   -- MC results for Phase 5 passers
  equity_longonly_leaderboard.csv -- final ranked leaderboard

Usage:
    uv run python research/ic_analysis/run_equity_longonly_pipeline.py
    uv run python research/ic_analysis/run_equity_longonly_pipeline.py --top 50
    uv run python research/ic_analysis/run_equity_longonly_pipeline.py --symbol AAPL
    uv run python research/ic_analysis/run_equity_longonly_pipeline.py --no-hmm
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_ic_backtest import (
    DEFAULT_RISK_PCT,
    DEFAULT_STOP_ATR,
    INIT_CASH,
    IS_RATIO,
    _build_and_align,
    build_composite,
    build_size_array,
    zscore_normalise,
)

import importlib.util as _ilu
HMM_AVAILABLE: bool = _ilu.find_spec("hmmlearn") is not None

REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
ADX_GATES: list = [None, 25, "hmm"]   # None=no filter, int=ADX<n, "hmm"=HMM state 0
MIN_OOS_TRADES = 5              # below this -> too few for WFO
IS_BARS_WFO = 504               # ~2yr daily
OOS_BARS_WFO = 126              # ~6mo daily

# Phase 3 pass gates
P3_MIN_OOS_SHARPE = 0.0
P3_MIN_PARITY = 0.50

# Phase 4 WFO gates
P4_MIN_FOLDS_POSITIVE = 0.60    # ≥60% folds OOS>0
P4_MIN_FOLDS_GT1 = 0.50         # ≥50% folds OOS>1
P4_WORST_FOLD_FLOOR = -2.0
P4_MIN_STITCHED = 1.5
P4_MIN_PARITY = 0.5

# Phase 5 Monte Carlo gates
P5_MC_N = 500
P5_MIN_5PCT_SHARPE = 0.5
P5_MIN_PROFITABLE = 0.80

# Excluded tickers (FX, ETFs, data-limited, non-equity)
EXCLUDE = {
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "AUD_JPY", "USD_CHF",
    "^VIX", "IWB", "QQQ", "SPY", "TQQQ",
    "GLD", "IAU", "SLV", "SIVR", "GDX", "GDXJ", "SIL", "PSLV",
    "SOLV", "KVUE", "VLTO", "EXE", "SNDK", "APP", "PLTR", "COIN",
    "DASH", "CRWD",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hmm_block_mask(df: pd.DataFrame, is_mask: pd.Series) -> pd.Series:
    """Fit 2-state HMM on IS bars, return boolean Series where True = block entry.

    State ordering: ascending by ewm_vol mean -> State 0 = low vol (allow),
    State 1 = high vol / trending (block). HMM parameters are IS-only; sequence
    prediction uses IS-fitted model applied to full dataset (standard pragmatic
    approach — no future model parameters leak into OOS).

    Returns a Series of all-False if HMM unavailable or insufficient data.
    """
    if not HMM_AVAILABLE:
        return pd.Series(False, index=df.index)

    # Load via importlib so the static analyser (no search roots) does not flag it
    _spec = _ilu.find_spec("research.ml_regime.regime_detection")
    if _spec is None:
        return pd.Series(False, index=df.index)
    _mod = _ilu.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    RegimeDetector = _mod.RegimeDetector

    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    ewm_vol = log_ret.ewm(span=20, adjust=False).std().fillna(0.0)
    features = np.column_stack([log_ret.values, ewm_vol.values])

    is_features = features[is_mask.values]
    if len(is_features) < 60:
        return pd.Series(False, index=df.index)

    try:
        det = RegimeDetector(n_states=2, n_iter=200, min_seq_len=60, vol_feature_idx=1)
        det.fit(is_features)
        states = det.predict_sequence(features)   # full sequence, IS-fitted model
    except Exception:
        return pd.Series(False, index=df.index)

    # State 0 = low vol = allow; State 1+ = high vol = block
    return pd.Series(states != 0, index=df.index)


def _adx14(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    dm_plus = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=close.index
    ).ewm(span=14, adjust=False).mean()
    dm_minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=close.index
    ).ewm(span=14, adjust=False).mean()
    safe_atr = atr.replace(0, np.nan)
    di_plus = dm_plus / safe_atr * 100
    di_minus = dm_minus / safe_atr * 100
    denom = (di_plus + di_minus).replace(0, np.nan)
    dx = (di_plus - di_minus).abs() / denom * 100
    return dx.ewm(span=14, adjust=False).mean()


def _sharpe_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return float("nan")
    std = pnl.std()
    if std == 0:
        return float("inf") if pnl.mean() > 0 else float("-inf")
    return float(pnl.mean() / std * np.sqrt(len(pnl)))


def _oos_trade_pnl(pf: vbt.Portfolio, is_n: int, base_index: pd.Index) -> np.ndarray:
    trades = pf.trades.records_readable
    if len(trades) == 0:
        return np.array([])
    pnls = []
    for _, t in trades.iterrows():
        try:
            pos = base_index.get_loc(t["Entry Timestamp"])
        except KeyError:
            continue
        if pos >= is_n:
            pnls.append(float(t["PnL"]))
    return np.array(pnls)


# ---------------------------------------------------------------------------
# Phase 3 -- single symbol
# ---------------------------------------------------------------------------


def phase3(symbol: str, spread: float = 0.05) -> dict | None:
    """Long-only threshold × ADX sweep. Returns best config or None if no pass."""
    try:
        tf_signals, base_index, base_df = _build_and_align(symbol, ["D"], base_tf="D")
    except Exception:
        return None

    base_close = base_df["close"]
    n = len(base_close)
    if n < 300:
        return None

    is_n = int(n * IS_RATIO)  # noqa: F841 (used implicitly via is_mask)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True

    composite = build_composite(tf_signals, base_close, ["D"], ["rsi_21_dev"], is_mask)
    sig_z = zscore_normalise(composite, is_mask)
    sizes = build_size_array(base_df, base_close, DEFAULT_RISK_PCT, DEFAULT_STOP_ATR)
    fees = spread / float(base_close.median())
    adx = _adx14(base_df)
    # Convert to numpy immediately so downstream ops are type-resolvable
    hmm_block: np.ndarray = np.asarray(_hmm_block_mask(base_df, is_mask), dtype=bool)

    best: dict | None = None

    for adx_gate in ADX_GATES:
        if adx_gate == "hmm" and not HMM_AVAILABLE:
            continue
        if adx_gate is None:
            gated = sig_z.copy()
            gated_pct = 0.0
        elif adx_gate == "hmm":
            gated = sig_z.mask(hmm_block.astype(bool), 0.0)
            gated_pct = float(hmm_block.astype(float).mean() * 100)
        else:
            mask_trend = adx > adx_gate
            gated = sig_z.where(~mask_trend, 0.0)
            gated_pct = float(mask_trend.mean() * 100)

        for th in THRESHOLDS:
            pf = vbt.Portfolio.from_signals(
                close=base_close,
                entries=gated > th,
                exits=gated < 0,
                size=sizes,
                size_type="percent",
                fees=fees,
                init_cash=INIT_CASH,
            )

            # IS Sharpe (daily returns)
            is_rets = pf.returns()[is_mask]
            is_sh = (
                float(is_rets.mean() / is_rets.std() * np.sqrt(252))
                if is_rets.std() > 0
                else 0.0
            )

            # OOS trade stats
            oos_pnl = _oos_trade_pnl(pf, is_n, base_index)
            if len(oos_pnl) < MIN_OOS_TRADES:
                continue

            oos_sh = _sharpe_from_pnl(oos_pnl)
            if not np.isfinite(oos_sh):
                continue

            parity = oos_sh / is_sh if is_sh > 0 else float("nan")
            if not (oos_sh > P3_MIN_OOS_SHARPE and (np.isnan(parity) or parity >= P3_MIN_PARITY)):
                continue

            wr = float((oos_pnl > 0).mean() * 100)
            oos_ret = float(oos_pnl.sum() / INIT_CASH * 100)

            gate_label = adx_gate if adx_gate is not None else 0
            candidate = {
                "symbol": symbol,
                "threshold": th,
                "adx_gate": gate_label,
                "adx_gated_pct": round(gated_pct, 1),
                "is_sharpe": round(is_sh, 3),
                "oos_sharpe": round(oos_sh, 3),
                "parity": round(parity, 3) if np.isfinite(parity) else float("nan"),
                "oos_wr_pct": round(wr, 1),
                "oos_trades": len(oos_pnl),
                "oos_ret_pct": round(oos_ret, 2),
                "n_bars": n,
            }

            # Prefer: more trades AND higher OOS Sharpe
            if best is None or (
                oos_sh > best["oos_sharpe"]  # type: ignore[operator]
                and len(oos_pnl) >= best["oos_trades"] // 2  # type: ignore[operator]
            ):
                best = candidate

    return best


# ---------------------------------------------------------------------------
# Phase 4 -- WFO (rolling, long-only, with optional ADX gate)
# ---------------------------------------------------------------------------


def phase4(symbol: str, threshold: float, adx_gate: "int | str", spread: float = 0.05) -> dict:
    """5-fold rolling WFO. Returns aggregate stats dict.

    adx_gate: 0 = no filter, int > 0 = ADX<n, "hmm" = HMM low-vol state only.
    For HMM gate the model is re-fit on each fold's IS window independently.
    """
    try:
        tf_signals, base_index, base_df = _build_and_align(symbol, ["D"], base_tf="D")
    except Exception:
        return {"symbol": symbol, "verdict": "ERROR"}

    base_close = base_df["close"]
    n = len(base_close)
    is_n_global = int(n * IS_RATIO)  # noqa: F841

    sizes = build_size_array(base_df, base_close, DEFAULT_RISK_PCT, DEFAULT_STOP_ATR)
    fees = spread / float(base_close.median())
    adx = _adx14(base_df)

    # Walk-forward folds
    total = n
    n_folds = max(1, (total - IS_BARS_WFO) // OOS_BARS_WFO)
    folds = []

    for fold_i in range(n_folds):
        is_start = fold_i * OOS_BARS_WFO
        is_end = is_start + IS_BARS_WFO
        oos_end = min(is_end + OOS_BARS_WFO, total)
        if is_end >= total:
            break

        is_mask = pd.Series(False, index=base_index)
        is_mask.iloc[is_start:is_end] = True

        composite = build_composite(
            tf_signals, base_close, ["D"], ["rsi_21_dev"], is_mask
        )
        sig_z = zscore_normalise(composite, is_mask)

        if adx_gate == "hmm":
            # Re-fit HMM on this fold's IS window only (anti-lookahead)
            hmm_block = np.asarray(_hmm_block_mask(base_df, is_mask), dtype=bool)
            sig_z = sig_z.where(~pd.Series(hmm_block, index=base_index), 0.0)
        elif isinstance(adx_gate, int) and adx_gate > 0:
            mask_trend = adx > adx_gate
            sig_z = sig_z.where(~mask_trend, 0.0)

        pf = vbt.Portfolio.from_signals(
            close=base_close,
            entries=sig_z > threshold,
            exits=sig_z < 0,
            size=sizes,
            size_type="percent",
            fees=fees,
            init_cash=INIT_CASH,
        )

        # IS Sharpe
        is_rets = pf.returns()[is_mask]
        is_sh = (
            float(is_rets.mean() / is_rets.std() * np.sqrt(252))
            if is_rets.std() > 0
            else 0.0
        )

        # OOS trades in this fold
        oos_pnl = []
        trades = pf.trades.records_readable
        for _, t in trades.iterrows():
            try:
                pos = base_index.get_loc(t["Entry Timestamp"])
            except KeyError:
                continue
            if is_end <= pos < oos_end:
                oos_pnl.append(float(t["PnL"]))

        oos_pnl_arr = np.array(oos_pnl)
        oos_sh = _sharpe_from_pnl(oos_pnl_arr) if len(oos_pnl) >= 1 else float("nan")
        wr = float((oos_pnl_arr > 0).mean() * 100) if len(oos_pnl) > 0 else float("nan")
        max_dd = 0.0

        folds.append({
            "fold": fold_i + 1,
            "is_sh": is_sh,
            "oos_sh": oos_sh,
            "oos_trades": len(oos_pnl),
            "oos_wr": wr,
        })

    if not folds:
        return {"symbol": symbol, "verdict": "NO_FOLDS"}

    oos_sharpes = [f["oos_sh"] for f in folds if np.isfinite(f["oos_sh"])]
    is_sharpes = [f["is_sh"] for f in folds if f["is_sh"] > 0]

    if not oos_sharpes:
        return {"symbol": symbol, "verdict": "NO_OOS_DATA"}

    pct_pos = sum(1 for s in oos_sharpes if s > 0) / len(folds)
    pct_gt1 = sum(1 for s in oos_sharpes if s > 1) / len(folds)
    worst = min(oos_sharpes)
    stitched = float(np.mean(oos_sharpes))
    mean_is = float(np.mean(is_sharpes)) if is_sharpes else float("nan")
    parity = stitched / mean_is if mean_is > 0 else float("nan")

    gates = {
        "folds_positive": pct_pos >= P4_MIN_FOLDS_POSITIVE,
        "folds_gt1": pct_gt1 >= P4_MIN_FOLDS_GT1,
        "worst_fold": worst >= P4_WORST_FOLD_FLOOR,
        "stitched": stitched >= P4_MIN_STITCHED,
        "parity": np.isnan(parity) or parity >= P4_MIN_PARITY,
    }
    passed = all(gates.values())

    return {
        "symbol": symbol,
        "threshold": threshold,
        "adx_gate": adx_gate,
        "n_folds": len(folds),
        "pct_folds_positive": round(pct_pos * 100, 1),
        "pct_folds_gt1": round(pct_gt1 * 100, 1),
        "worst_oos_sharpe": round(worst, 3),
        "stitched_oos_sharpe": round(stitched, 3),
        "parity": round(parity, 3) if np.isfinite(parity) else float("nan"),
        "mean_is_sharpe": round(mean_is, 3) if np.isfinite(mean_is) else float("nan"),
        "fold_details": folds,
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
    }


# ---------------------------------------------------------------------------
# Phase 5 -- Monte Carlo
# ---------------------------------------------------------------------------


def phase5_mc(symbol: str, threshold: float, adx_gate: "int | str", spread: float = 0.05) -> dict:
    """Monte Carlo gate on OOS trade P&L."""
    try:
        tf_signals, base_index, base_df = _build_and_align(symbol, ["D"], base_tf="D")
    except Exception:
        return {"symbol": symbol, "mc_verdict": "ERROR"}

    base_close = base_df["close"]
    n = len(base_close)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True

    composite = build_composite(tf_signals, base_close, ["D"], ["rsi_21_dev"], is_mask)
    sig_z = zscore_normalise(composite, is_mask)

    adx = _adx14(base_df)
    if adx_gate == "hmm":
        hmm_block = np.asarray(_hmm_block_mask(base_df, is_mask), dtype=bool)
        sig_z = sig_z.where(~pd.Series(hmm_block, index=base_index), 0.0)
    elif isinstance(adx_gate, int) and adx_gate > 0:
        sig_z = sig_z.where(adx <= adx_gate, 0.0)

    sizes = build_size_array(base_df, base_close, DEFAULT_RISK_PCT, DEFAULT_STOP_ATR)
    fees = spread / float(base_close.median())

    pf = vbt.Portfolio.from_signals(
        close=base_close,
        entries=sig_z > threshold,
        exits=sig_z < 0,
        size=sizes,
        size_type="percent",
        fees=fees,
        init_cash=INIT_CASH,
    )

    oos_pnl = _oos_trade_pnl(pf, is_n, base_index)
    if len(oos_pnl) < 5:
        return {"symbol": symbol, "mc_verdict": "SKIP_TOO_FEW_TRADES",
                "oos_trades": len(oos_pnl)}

    rng = np.random.default_rng(42)
    sim_sharpes = []
    for _ in range(P5_MC_N):
        s = _sharpe_from_pnl(rng.permutation(oos_pnl))
        if np.isfinite(s):
            sim_sharpes.append(s)
    sim_arr = np.array(sim_sharpes)

    pct5 = float(np.percentile(sim_arr, 5))
    pct_prof = float((sim_arr > 0).mean())
    gate = pct5 > P5_MIN_5PCT_SHARPE and pct_prof >= P5_MIN_PROFITABLE

    return {
        "symbol": symbol,
        "threshold": threshold,
        "adx_gate": adx_gate,
        "oos_trades": len(oos_pnl),
        "mc_5th_pct_sharpe": round(pct5, 3),
        "mc_pct_profitable": round(pct_prof * 100, 1),
        "mc_verdict": "PASS" if gate else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_candidates(top: int = 0, single: str = "") -> list[str]:
    """Return list of equity symbols to process."""
    # FX / ETF / data-limited exclusions
    fx = {"EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "AUD_JPY", "USD_CHF"}
    etf = {"IWB", "QQQ", "SPY", "TQQQ", "GLD", "IAU", "SLV", "SIVR",
           "GDX", "GDXJ", "SIL", "PSLV"}
    limited = {"SOLV", "KVUE", "VLTO", "EXE", "SNDK", "APP", "PLTR", "COIN",
               "DASH", "CRWD"}
    misc = {"^VIX"}
    excluded = fx | etf | limited | misc

    if single:
        return [single]

    master_path = REPORTS / "ic_sweep_daily_master.csv"
    if not master_path.exists():
        print("ERROR: Phase 1 master CSV not found. Run Phase 1 first.")
        sys.exit(1)

    master = pd.read_csv(master_path)
    sym_rank = (
        master[master["verdict"] == "STRONG"]
        .groupby("symbol")
        .agg(n_strong=("signal", "count"), best_ic=("abs_ic", "max"))
        .sort_values("n_strong", ascending=False)
        .reset_index()
    )

    syms = []
    for sym in sym_rank["symbol"]:
        if sym in excluded:
            continue
        p = DATA_DIR / f"{sym}_D.parquet"
        if not p.exists():
            continue
        try:
            n = len(pd.read_parquet(p))
        except Exception:
            continue
        if n < 1000:
            continue
        syms.append(sym)
        if top and len(syms) >= top:
            break
    return syms


def _gate_label(gate: "int | str") -> str:
    """Human-readable label for a regime gate value."""
    if gate == 0 or gate is None:
        return "no-gate"
    if gate == "hmm":
        return "HMM"
    return f"ADX<{gate}"


def main() -> None:  # noqa: PLR0912
    global ADX_GATES
    parser = argparse.ArgumentParser(
        description="Long-only IC equity pipeline: Phase 3->5 with regime gate sweep"
    )
    parser.add_argument("--top", type=int, default=0,
                        help="Limit to top N candidates by Phase 1 STRONG count (0=all)")
    parser.add_argument("--symbol", default="",
                        help="Run a single symbol only")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Skip HMM gate (faster; only runs no-gate and ADX<25)")
    args = parser.parse_args()

    active_gates = [g for g in ADX_GATES if not (g == "hmm" and args.no_hmm)]
    if not HMM_AVAILABLE and "hmm" in active_gates:
        print("  WARNING: hmmlearn not installed -- HMM gate will be skipped.")

    symbols = get_candidates(top=args.top, single=args.symbol)
    print(f"\n{'='*70}")
    print("  IC EQUITY LONG-ONLY PIPELINE -- Phase 3->5")
    print(f"  Symbols   : {len(symbols)}")
    print(f"  Signal    : rsi_21_dev | Long-only | Gates: {active_gates}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  HMM avail : {HMM_AVAILABLE}")
    print(f"{'='*70}\n")

    # Patch module-level gate list for this run
    ADX_GATES = active_gates  # type: ignore[assignment]

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print("PHASE 3 -- Threshold x regime gate sweep (long-only)...")
    p3_results = []
    p3_passers = []

    for i, sym in enumerate(symbols):
        t0 = time.time()
        result = phase3(sym)
        elapsed = time.time() - t0

        if result:
            p3_results.append(result)
            p3_passers.append(sym)
            gate_str = _gate_label(result["adx_gate"])
            print(
                f"  [{i+1:>4}/{len(symbols)}] {sym:<8} PASS | "
                f"th={result['threshold']:.2f} {gate_str:<8} | "
                f"OOS Sh={result['oos_sharpe']:+.3f} trades={result['oos_trades']} | "
                f"{elapsed:.1f}s"
            )
        else:
            if (i + 1) % 50 == 0:
                print(f"  [{i+1:>4}/{len(symbols)}] ... (continuing)")

    p3_df = pd.DataFrame(p3_results).sort_values("oos_sharpe", ascending=False)
    p3_df.to_csv(REPORTS / "equity_longonly_phase3.csv", index=False)
    print(f"\n  Phase 3: {len(p3_passers)}/{len(symbols)} passed")
    print("  Saved: equity_longonly_phase3.csv\n")

    if not p3_passers:
        print("No Phase 3 passers -- stopping.")
        return

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    print("PHASE 4 -- WFO validation on Phase 3 passers...")
    p4_results = []
    p4_passers = []

    p3_configs = {r["symbol"]: r for r in p3_results}

    for i, sym in enumerate(p3_passers):
        cfg = p3_configs[sym]
        result = phase4(sym, cfg["threshold"], cfg["adx_gate"])

        p4_results.append(result)
        verdict = result.get("verdict", "?")
        if verdict == "PASS":
            p4_passers.append(sym)
            print(
                f"  [{i+1:>4}/{len(p3_passers)}] {sym:<8} PASS | "
                f"stitched={result.get('stitched_oos_sharpe', '?'):+.3f} "
                f"parity={result.get('parity', '?'):.3f} "
                f"folds+={result.get('pct_folds_positive', '?'):.0f}%"
            )
        else:
            if (i + 1) % 20 == 0 or verdict != "FAIL":
                print(f"  [{i+1:>4}/{len(p3_passers)}] {sym:<8} {verdict}")

    # Flatten fold details for CSV
    p4_rows = []
    for r in p4_results:
        base = {k: v for k, v in r.items() if k not in ("fold_details", "gates")}
        base["gates_passed"] = sum(1 for v in r.get("gates", {}).values() if v)
        p4_rows.append(base)

    p4_df = pd.DataFrame(p4_rows).sort_values("stitched_oos_sharpe", ascending=False)
    p4_df.to_csv(REPORTS / "equity_longonly_phase4.csv", index=False)
    print(f"\n  Phase 4: {len(p4_passers)}/{len(p3_passers)} passed all WFO gates")
    print("  Saved: equity_longonly_phase4.csv\n")

    if not p4_passers:
        print("No Phase 4 passers -- stopping.")
        return

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    print("PHASE 5 -- Monte Carlo robustness on Phase 4 passers...")
    p5_results = []
    p4_configs = {r["symbol"]: r for r in p4_results if r.get("verdict") == "PASS"}

    for sym in p4_passers:
        cfg = p4_configs[sym]
        result = phase5_mc(sym, cfg["threshold"], cfg["adx_gate"])
        p5_results.append(result)
        print(
            f"  {sym:<8} MC 5th-pct={result.get('mc_5th_pct_sharpe', '?'):+.3f} "
            f"profitable={result.get('mc_pct_profitable', '?'):.1f}% "
            f"-> {result['mc_verdict']}"
        )

    p5_df = pd.DataFrame(p5_results).sort_values("mc_5th_pct_sharpe", ascending=False)
    p5_df.to_csv(REPORTS / "equity_longonly_phase5.csv", index=False)
    print("\n  Saved: equity_longonly_phase5.csv\n")

    # ── Final leaderboard ────────────────────────────────────────────────────
    lb = (
        p3_df.merge(
            p4_df[["symbol", "stitched_oos_sharpe", "parity", "pct_folds_positive",
                   "worst_oos_sharpe", "verdict"]].rename(
                columns={"verdict": "p4_verdict"}
            ),
            on="symbol", how="left",
        )
        .merge(
            p5_df[["symbol", "mc_5th_pct_sharpe", "mc_pct_profitable",
                   "mc_verdict"]].rename(columns={}),
            on="symbol", how="left",
        )
        .sort_values("stitched_oos_sharpe", ascending=False, na_position="last")
    )
    lb.to_csv(REPORTS / "equity_longonly_leaderboard.csv", index=False)

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  FINAL LEADERBOARD -- Long-Only rsi_21_dev Daily")
    print(f"{'='*90}")
    print(
        f"{'Symbol':<8} {'Thresh':>7} {'ADX':>6} {'P3 OOS':>8} {'P4 Stitch':>10} "
        f"{'Parity':>8} {'MC 5th':>8} {'P5':>6}"
    )
    print("-" * 90)

    p4_pass_set = set(p4_passers)
    p5_map = {r["symbol"]: r for r in p5_results}

    for _, row in lb.iterrows():
        sym = row["symbol"]
        if sym not in p4_pass_set:
            continue
        mc = p5_map.get(sym, {})
        gate_val = row.get("adx_gate", 0)
        if not gate_val or gate_val == 0:
            adx_str = "none"
        elif gate_val == "hmm":
            adx_str = "HMM"
        else:
            adx_str = f"<{int(gate_val)}"
        print(
            f"{sym:<8} {row['threshold']:>7.2f} {adx_str:>6} "
            f"{row['oos_sharpe']:>+8.3f} {row.get('stitched_oos_sharpe', float('nan')):>+10.3f} "
            f"{row.get('parity', float('nan')):>8.3f} "
            f"{mc.get('mc_5th_pct_sharpe', float('nan')):>+8.3f} "
            f"{mc.get('mc_verdict', '?'):>6}"
        )

    print(f"\n  Phase 3 passers : {len(p3_passers)}/{len(symbols)}")
    print(f"  Phase 4 passers : {len(p4_passers)}/{len(p3_passers)}")
    mc_pass = sum(1 for r in p5_results if r.get("mc_verdict") == "PASS")
    print(f"  Phase 5 MC PASS : {mc_pass}/{len(p4_passers)}")
    print("\n  Leaderboard: .tmp/reports/equity_longonly_leaderboard.csv")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
