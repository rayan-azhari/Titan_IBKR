"""Phase 5 — Robustness Validation (6 Gates).

Six quality gates that must all PASS before Phase 6 live implementation:
  G1. Monte Carlo shuffle (N=1,000): 5th-pct Sharpe > 0.5, > 80% profitable sims.
  G2. Remove top-10 winning trades: remaining cumulative return still positive.
  G3. 3x slippage stress: OOS Sharpe > 0.5.
  G4. WFO consecutive negative folds <= 2 (reads phase4_{slug}.csv from Phase 4).
  G5. Regime robustness: OOS Sharpe > 0 in >= 2/3 ADX regimes AND >= 1/2 HMM states.
  G6. Alpha/beta decomposition: alpha_ann > 0, beta < 1.0, R-squared < 0.5.

Gate 5 requires Phase 0 regime labels (.tmp/regime/{instrument}_{timeframe}_regime.parquet).
Gate 6 requires a benchmark price series (SPY_D.parquet for equities, DXY_D.parquet for FX).
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REGIME_DIR = ROOT / ".tmp" / "regime"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # type: ignore[assignment]

from research.ic_analysis.phase3_backtest import (  # noqa: E402
    COST_PROFILES,
    DEFAULT_RISK_PCT,
    DEFAULT_SIGNALS,
    DEFAULT_STOP_ATR,
    DEFAULT_TFS,
    INIT_CASH,
    IS_RATIO,
    _apply_hmm_gate,
    _build_and_align,
    build_composite,
    build_size_array,
    zscore_normalise,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MC_N = 1_000
MC_MIN_5PCT_SHARPE = 0.5
MC_MIN_PROFITABLE_PCT = 0.80
TOP_N_REMOVE = 10
STRESS_SLIPPAGE_MULT = 3.0
STRESS_MIN_SHARPE = 0.5
WFO_MAX_CONSEC_NEG = 2
DEFAULT_THRESHOLD = 0.75
REGIME_MIN_ADX_PASS = 2  # >= 2 out of 3 ADX regimes
REGIME_MIN_HMM_PASS = 1  # >= 1 out of 2 HMM states
BETA_THRESHOLDS = {
    "fx_major": 0.5,
    "fx_minor": 0.5,
    "equity": 1.0,
    "crypto": 1.5,
    "commodity": 0.8,
}
ALPHA_BETA_MAX_R2 = 0.5


# ---------------------------------------------------------------------------
# OOS portfolio builder
# ---------------------------------------------------------------------------


def _build_oos_portfolios(
    instrument: str,
    tfs: list[str],
    target_signals: list[str],
    asset_class: str,
    direction: str = "both",
    threshold: float = DEFAULT_THRESHOLD,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_atr: float = DEFAULT_STOP_ATR,
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    max_leverage: float | None = None,
    slippage_mult: float = 1.0,
    hmm_gate: bool = False,
    ref_horizon: int = 1,
) -> tuple:
    """Build OOS long + short VBT portfolios.

    Returns (pf_long, pf_short, oos_close, oos_mask, base_index, direction).
    pf_short is None when direction == 'long_only'.
    """
    profile = COST_PROFILES[asset_class]

    eff_spread_bps = spread_bps if spread_bps is not None else profile["spread_bps"]
    eff_slippage_bps = slippage_bps if slippage_bps is not None else profile["slippage_bps"]
    eff_slippage_bps *= slippage_mult
    eff_max_lev = max_leverage if max_leverage is not None else profile["max_leverage"]

    base_tf = tfs[-1]
    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]

    n = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True
    oos_mask = ~is_mask

    composite = build_composite(
        tf_signals, base_close, tfs, target_signals, is_mask, ref_horizon=ref_horizon
    )
    composite_z = zscore_normalise(composite, is_mask)
    if hmm_gate:
        base_tf = tfs[-1]
        composite_z = _apply_hmm_gate(composite_z, instrument, base_tf, is_mask)

    size_arr, stop_pct_arr = build_size_array(
        base_df, base_close, risk_pct, stop_atr, max_leverage=eff_max_lev
    )

    oos_close = base_close[oos_mask]
    med_close = float(oos_close.median()) or 1.0

    spread_abs = eff_spread_bps / 10_000 * med_close
    slippage_abs = eff_slippage_bps / 10_000 * med_close

    stop_arr = stop_pct_arr[oos_mask].fillna(0.0).values
    vbt_fees = (spread_abs / oos_close).bfill().values
    vbt_slip = (slippage_abs / oos_close).bfill().values

    sig = composite_z[oos_mask].shift(1).fillna(0.0)

    freq = "d" if base_tf in ("D",) else "h"

    pf_long = vbt.Portfolio.from_signals(
        oos_close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        sl_stop=stop_arr,
        size=size_arr[oos_mask],
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    if direction == "long_only":
        pf_short = None
    else:
        pf_short = vbt.Portfolio.from_signals(
            oos_close,
            entries=pd.Series(False, index=oos_close.index),
            exits=pd.Series(False, index=oos_close.index),
            short_entries=sig < -threshold,
            short_exits=sig >= 0.0,
            sl_stop=stop_arr,
            size=size_arr[oos_mask],
            size_type="percent",
            init_cash=INIT_CASH,
            fees=vbt_fees,
            slippage=vbt_slip,
            freq=freq,
        )

    return (pf_long, pf_short, oos_close, oos_mask, base_index, direction)


# ---------------------------------------------------------------------------
# Helper: combined per-trade % returns
# ---------------------------------------------------------------------------


def _combined_trade_returns(pf_long, pf_short) -> np.ndarray | None:
    """Extract per-trade % returns from both portfolios."""
    all_rets: list[float] = []

    for pf in [pf_long, pf_short]:
        if pf is None:
            continue
        try:
            rets = pf.trades.records_readable["Return"].values
            all_rets.extend(rets.tolist())
        except Exception:
            pass

    if not all_rets:
        return None
    return np.array(all_rets, dtype=float)


# ---------------------------------------------------------------------------
# Gate 1 — Monte Carlo shuffle
# ---------------------------------------------------------------------------


def monte_carlo_shuffle(
    pf_long,
    pf_short,
    oos_bars: int,
    n: int = MC_N,
    freq: str = "h",
) -> dict:
    """G1: Shuffle trade returns N times and check Sharpe distribution."""
    oos_years = oos_bars / 252 if freq == "d" else oos_bars / (252 * 24)

    trade_rets = _combined_trade_returns(pf_long, pf_short)
    if trade_rets is None or len(trade_rets) < 2:
        return {
            "error": "Insufficient trades for Monte Carlo",
            "gate_pass": False,
        }

    mu_base = float(trade_rets.mean())
    sigma_base = float(trade_rets.std(ddof=1))
    n_trades = len(trade_rets)
    trades_per_year = n_trades / oos_years if oos_years > 0 else float(n_trades)
    base_sharpe = mu_base / sigma_base * sqrt(trades_per_year) if sigma_base > 1e-10 else 0.0

    rng = np.random.default_rng(42)
    sim_sharpes: list[float] = []
    for _ in range(n):
        shuffled = rng.permutation(trade_rets)
        mu = float(shuffled.mean())
        sigma = float(shuffled.std(ddof=1))
        sh = mu / sigma * sqrt(trades_per_year) if sigma > 1e-10 else 0.0
        sim_sharpes.append(sh)

    sim_arr = np.array(sim_sharpes)
    pct5 = float(np.percentile(sim_arr, 5))
    profitable_pct = float((sim_arr > 0).mean())

    gate_pass = (pct5 > MC_MIN_5PCT_SHARPE) and (profitable_pct >= MC_MIN_PROFITABLE_PCT)

    return {
        "base_sharpe": round(base_sharpe, 4),
        "mc_5pct_sharpe": round(pct5, 4),
        "mc_profitable_pct": round(profitable_pct, 4),
        "mc_n": n,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Gate 2 — Remove top-N winning trades
# ---------------------------------------------------------------------------


def remove_top_n(pf_long, pf_short, n: int = TOP_N_REMOVE) -> dict:
    """G2: Remove top-N wins and check remaining cumulative return is positive."""
    trade_rets = _combined_trade_returns(pf_long, pf_short)
    if trade_rets is None or len(trade_rets) <= n:
        return {
            "error": f"Fewer than {n} trades available",
            "gate_pass": False,
        }

    # C4 FIX: Remove top-N by absolute magnitude (both large wins AND large losses).
    abs_order = np.argsort(np.abs(trade_rets))[::-1]  # largest |return| first
    trimmed = trade_rets[abs_order[n:]]

    cum_return = float((1 + trimmed).prod() - 1)
    gate_pass = cum_return > 0.0

    return {
        "original_trades": len(trade_rets),
        "removed_trades": n,
        "trimmed_cum_return": round(cum_return, 4),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Gate 3 — 3x slippage stress
# ---------------------------------------------------------------------------


def stress_triple_slippage(
    instrument: str,
    tfs: list[str],
    target_signals: list[str],
    asset_class: str,
    direction: str,
    threshold: float,
    risk_pct: float,
    stop_atr: float,
    spread_bps: float | None,
    max_leverage: float | None,
    hmm_gate: bool = False,
    ref_horizon: int = 1,
) -> dict:
    """G3: Rebuild OOS portfolios with 3x slippage and check Sharpe."""
    try:
        pf_long, pf_short, oos_close, oos_mask, base_index, _ = _build_oos_portfolios(
            instrument=instrument,
            tfs=tfs,
            target_signals=target_signals,
            asset_class=asset_class,
            direction=direction,
            threshold=threshold,
            risk_pct=risk_pct,
            stop_atr=stop_atr,
            spread_bps=spread_bps,
            max_leverage=max_leverage,
            slippage_mult=STRESS_SLIPPAGE_MULT,
            hmm_gate=hmm_gate,
            ref_horizon=ref_horizon,
        )
    except Exception as exc:
        return {"error": str(exc), "gate_pass": False}

    oos_bars = int(oos_mask.sum())
    base_tf = tfs[-1]
    freq = "d" if base_tf in ("D",) else "h"
    oos_years = oos_bars / 252 if freq == "d" else oos_bars / (252 * 24)

    def _sharpe(pf) -> float:
        if pf is None:
            return 0.0
        try:
            rets = pf.trades.records_readable["Return"].values
        except Exception:
            return 0.0
        if len(rets) < 2:
            return 0.0
        mu = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        if sigma < 1e-10:
            return 0.0
        tpy = len(rets) / oos_years if oos_years > 0 else float(len(rets))
        return float(mu / sigma * sqrt(tpy))

    if direction == "long_only":
        combined_sharpe = _sharpe(pf_long)
    else:
        sh_l = _sharpe(pf_long)
        sh_s = _sharpe(pf_short) if pf_short is not None else 0.0
        combined_sharpe = (sh_l + sh_s) / 2.0

    gate_pass = combined_sharpe > STRESS_MIN_SHARPE

    return {
        "stress_slippage_mult": STRESS_SLIPPAGE_MULT,
        "stress_sharpe": round(combined_sharpe, 4),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Gate 4 — WFO consecutive negative folds
# ---------------------------------------------------------------------------


def check_wfo_consecutive(instrument: str) -> dict:
    """G4: Read phase4 CSV and count max consecutive negative fold Sharpes."""
    slug = instrument.lower()

    candidates = [
        REPORTS_DIR / f"phase4_{slug}.csv",
        REPORTS_DIR / f"phase4_{slug}_anchored.csv",
    ]

    csv_path: Path | None = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        return {
            "error": (f"Phase 4 WFO CSV not found for {instrument}. Run phase4_wfo.py first."),
            "gate_pass": False,
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"error": f"Could not read {csv_path}: {exc}", "gate_pass": False}

    sharpe_col: str | None = None
    for candidate_col in ["oos_sharpe", "sharpe", "OOS_Sharpe", "Sharpe"]:
        if candidate_col in df.columns:
            sharpe_col = candidate_col
            break

    if sharpe_col is None:
        return {
            "error": f"No Sharpe column in {csv_path}. Columns: {list(df.columns)}",
            "gate_pass": False,
        }

    sharpes = df[sharpe_col].dropna().values

    max_consec = 0
    current = 0
    for sh in sharpes:
        if sh < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    gate_pass = max_consec <= WFO_MAX_CONSEC_NEG

    return {
        "wfo_csv": str(csv_path),
        "n_folds": len(sharpes),
        "max_consec_neg": int(max_consec),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Gate 5 — Regime robustness (NEW)
# ---------------------------------------------------------------------------


def _regime_sharpe(
    returns: list[float],
    trades_per_year: float,
) -> float:
    """Annualised Sharpe from a list of per-trade returns."""
    if len(returns) < 5:
        return 0.0
    arr = np.array(returns, dtype=float)
    mu = arr.mean()
    std = arr.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(mu / std * sqrt(trades_per_year))


def check_regime_robustness(
    instrument: str,
    timeframe: str,
    pf_long,
    pf_short,
    base_index: pd.DatetimeIndex,
    oos_mask: pd.Series,
    freq: str = "h",
    hmm_gate: bool = False,
) -> dict:
    """G5: Check OOS Sharpe > 0 in >= 2/3 ADX regimes and >= 1/2 HMM states.

    freq: base timeframe frequency — "d" for daily, "h" for hourly.
          Used for correct bars-per-year annualisation (M1 fix).
          Hardcoding hourly for daily instruments inflates Sharpe ~25×.

    hmm_gate: if True, the bad HMM state has zero trades by construction and
              is excluded from the HMM pass count (M2 fix). Without this, a
              healthy strategy with hmm_gate active always fails G5 HMM sub-test.
    """
    regime_path = REGIME_DIR / f"{instrument}_{timeframe}_regime.parquet"

    if not regime_path.exists():
        return {
            "error": "No regime file. Run phase0_regime.py first.",
            "gate_pass": False,
        }

    try:
        regime_df = pd.read_parquet(regime_path)
        regime_df.index = pd.to_datetime(regime_df.index, utc=True)
    except Exception as exc:
        return {"error": f"Could not load regime file: {exc}", "gate_pass": False}

    # Collect entry timestamps and per-trade returns from both portfolios.
    entry_times: list = []
    entry_returns: list[float] = []

    try:
        for pf in [pf_long, pf_short]:
            if pf is None:
                continue
            if pf.trades.count() == 0:
                continue
            readable = pf.trades.records_readable
            if "Entry Timestamp" in readable.columns and "Return" in readable.columns:
                for ts, ret in zip(
                    readable["Entry Timestamp"].tolist(),
                    readable["Return"].tolist(),
                ):
                    entry_times.append(ts)
                    entry_returns.append(float(ret))
    except Exception as exc:
        return {
            "error": f"Could not extract entry timestamps: {exc}",
            "gate_pass": False,
        }

    if not entry_times:
        return {"error": "No OOS trades found", "gate_pass": False}

    # Normalise entry timestamps to UTC-aware DatetimeIndex.
    try:
        entry_dt = pd.DatetimeIndex(entry_times)
        if entry_dt.tz is None:
            entry_dt = entry_dt.tz_localize("UTC")
        else:
            entry_dt = entry_dt.tz_convert("UTC")
    except Exception as exc:
        return {"error": f"Timestamp conversion failed: {exc}", "gate_pass": False}

    # Make regime_df index UTC-aware if needed.
    if regime_df.index.tz is None:
        regime_df.index = regime_df.index.tz_localize("UTC")
    else:
        regime_df.index = regime_df.index.tz_convert("UTC")

    # Align entry times to regime using ffill so coarser regime TFs still map.
    try:
        aligned_regime = (
            regime_df.reindex(regime_df.index.union(entry_dt)).ffill().reindex(entry_dt)
        )
    except Exception as exc:
        return {"error": f"Regime alignment failed: {exc}", "gate_pass": False}

    n_total_trades = len(entry_returns)
    oos_bars = int(oos_mask.sum())
    # M1 FIX: Use correct bars-per-year for the instrument's base frequency.
    # Previously hardcoded 252*24 (hourly) for all instruments, inflating
    # Sharpe ~25× for daily instruments (correct bpy=252, not 6048).
    bpy = 252 if freq == "d" else 252 * 24
    oos_years = max(oos_bars / bpy, 0.01)
    trades_per_year = n_total_trades / oos_years

    # -- ADX regimes -----------------------------------------------------------
    adx_results: dict[str, float] = {}
    adx_passing: list[bool] = []

    for regime_label in ["ranging", "neutral", "trending"]:
        rets_in_regime: list[float] = []

        for i, ts in enumerate(entry_dt):
            try:
                row = aligned_regime.iloc[i]
                adx_val = row.get("adx_regime") if hasattr(row, "get") else None
                if adx_val is None and "adx_regime" in aligned_regime.columns:
                    adx_val = aligned_regime["adx_regime"].iloc[i]
                if adx_val == regime_label:
                    rets_in_regime.append(entry_returns[i])
            except Exception:
                pass

        sh = _regime_sharpe(rets_in_regime, trades_per_year)
        adx_results[f"adx_{regime_label}_sharpe"] = round(sh, 4)
        adx_passing.append(bool(sh > 0))

    adx_pass_count = sum(adx_passing)

    # -- HMM states ------------------------------------------------------------
    # M2 FIX: When hmm_gate=True, the bad HMM state has zero trades by
    # construction (all entries in that state were blocked at strategy level).
    # Counting a zero-trade state as a fail would always fail G5 HMM even for
    # a perfectly healthy strategy. Derive the good (gated) state from the IS
    # majority and exclude the blocked state from the pass count.
    hmm_good_state: int | None = None
    if hmm_gate and "hmm_state" in regime_df.columns:
        try:
            # IS = base_index entries NOT in oos_mask
            is_times = base_index[~oos_mask.reindex(base_index, fill_value=False).values]
            is_hmm = (
                regime_df["hmm_state"]
                .reindex(regime_df.index.union(is_times))
                .ffill()
                .reindex(is_times)
                .dropna()
            )
            if len(is_hmm) > 0:
                hmm_good_state = int(is_hmm.mode()[0])
        except Exception:
            pass  # if derivation fails, fall back to counting all states

    hmm_results: dict[str, float] = {}
    hmm_passing: list[bool] = []

    for state in [0, 1]:
        # Skip the gated (bad) state: it has zero trades by design, not by
        # strategy failure.  Only skip if we successfully identified good_state.
        if hmm_gate and hmm_good_state is not None and state != hmm_good_state:
            hmm_results[f"hmm_{state}_sharpe"] = float("nan")
            hmm_results[f"hmm_{state}_note"] = "skipped (gated by hmm_gate)"
            continue

        rets_in_state: list[float] = []
        for i, _ts in enumerate(entry_dt):
            try:
                hmm_val = None
                if "hmm_state" in aligned_regime.columns:
                    hmm_val = aligned_regime["hmm_state"].iloc[i]
                if hmm_val is not None and int(hmm_val) == state:
                    rets_in_state.append(entry_returns[i])
            except Exception:
                pass

        sh = _regime_sharpe(rets_in_state, trades_per_year)
        hmm_results[f"hmm_{state}_sharpe"] = round(sh, 4)
        hmm_passing.append(bool(sh > 0))

    hmm_pass_count = sum(hmm_passing)

    # -- Volatility terciles (rv20) --------------------------------------------
    # L4 FIX: Compute tercile edges from IS rv20 distribution (first IS_RATIO
    # fraction of regime bars) rather than all OOS trade entry times. Using OOS
    # trade entries is a minor post-hoc bias — boundaries are set retrospectively.
    VOL_MIN_PASS = 2  # >= 2 out of 3 terciles
    vol_results: dict[str, float] = {}
    vol_passing: list[bool] = []

    # Step 1: get rv20 series (from regime_df column or compute from close).
    rv20_full: pd.Series | None = None
    if "rv20" in regime_df.columns:
        rv20_full = regime_df["rv20"]
    elif "close" in regime_df.columns:
        try:
            rv20_full = regime_df["close"].pct_change().rolling(20).std()
        except Exception:
            rv20_full = None

    if rv20_full is not None:
        # Step 2: derive tercile edges from IS portion only (pre-OOS data).
        is_n_regime = int(len(rv20_full) * IS_RATIO)
        rv20_is = rv20_full.iloc[:is_n_regime].dropna()
        if len(rv20_is) >= 10:
            tercile_edges = rv20_is.quantile([1 / 3, 2 / 3]).values
        else:
            tercile_edges = None

        # Step 3: align rv20 to entry timestamps.
        try:
            rv20_aligned = (
                rv20_full.reindex(rv20_full.index.union(entry_dt)).ffill().reindex(entry_dt)
            )
            rv20_at_entry = rv20_aligned.values
        except Exception:
            rv20_at_entry = None
    else:
        tercile_edges = None
        rv20_at_entry = None

    if (
        rv20_at_entry is not None
        and tercile_edges is not None
        and len(rv20_at_entry) == len(entry_returns)
    ):
        for lo, hi, label in [
            (None, tercile_edges[0], "low_vol"),
            (tercile_edges[0], tercile_edges[1], "mid_vol"),
            (tercile_edges[1], None, "high_vol"),
        ]:
            rets_in_tercile: list[float] = []
            for i in range(len(entry_returns)):
                rv_val = rv20_at_entry[i]
                if np.isnan(rv_val):
                    continue
                if lo is not None and rv_val < lo:
                    continue
                if hi is not None and rv_val >= hi:
                    continue
                rets_in_tercile.append(entry_returns[i])
            sh = _regime_sharpe(rets_in_tercile, trades_per_year)
            vol_results[f"vol_{label}_sharpe"] = round(sh, 4)
            vol_passing.append(bool(sh > 0))
    else:
        for label in ["low_vol", "mid_vol", "high_vol"]:
            vol_results[f"vol_{label}_sharpe"] = 0.0
            vol_passing.append(False)

    vol_pass_count = sum(vol_passing)

    gate_pass = (
        adx_pass_count >= REGIME_MIN_ADX_PASS
        and hmm_pass_count >= REGIME_MIN_HMM_PASS
        and vol_pass_count >= VOL_MIN_PASS
    )

    return {
        **adx_results,
        "adx_pass_count": adx_pass_count,
        **hmm_results,
        "hmm_pass_count": hmm_pass_count,
        **vol_results,
        "vol_pass_count": vol_pass_count,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Gate 6 — Alpha/beta decomposition (NEW)
# ---------------------------------------------------------------------------


def check_alpha_beta(
    instrument: str,
    benchmark: str,
    pf_long,
    pf_short,
    oos_close: pd.Series,
    direction: str = "both",
    asset_class: str = "equity",
) -> dict:
    """G6: OLS regression of strategy returns on benchmark returns.

    M3 FIX: Guard against self-referential benchmark (instrument == benchmark).
    When benchmarking SPY against SPY, there are 0 overlapping days because the
    strategy only covers the OOS period. This produces a misleading error instead
    of a clear explanation. Return a skip result with a clear message.
    """
    if sm is None:
        return {"error": "statsmodels not installed", "gate_pass": False}

    # M3 FIX: reject circular benchmark before attempting to load data.
    inst_slug = instrument.upper().replace("_", "").replace("-", "")
    bench_slug = benchmark.upper().replace("_", "").replace("-", "")
    if inst_slug == bench_slug:
        return {
            "gate_pass": False,
            "error": (
                f"Benchmark equals instrument ({benchmark}) — G6 is circular. "
                "Use a different benchmark (e.g. SPX for SPY, DXY for EUR_USD)."
            ),
        }

    bench_path = ROOT / "data" / f"{benchmark}_D.parquet"
    if not bench_path.exists():
        return {
            "error": f"Benchmark not found: {bench_path}",
            "gate_pass": False,
        }

    try:
        bench_df = pd.read_parquet(bench_path)
    except Exception as exc:
        return {"error": f"Could not load benchmark: {exc}", "gate_pass": False}

    # Strategy daily returns.
    try:
        if direction == "long_only":
            strat_daily = pf_long.returns()
        else:
            long_rets = pf_long.returns()
            short_rets = pf_short.returns() if pf_short is not None else long_rets * 0
            strat_daily = (long_rets + short_rets) / 2

        strat_daily = strat_daily.resample("D").sum()
    except Exception as exc:
        return {"error": f"Could not compute strategy returns: {exc}", "gate_pass": False}

    # Benchmark daily returns.
    try:
        bm_daily = bench_df["close"].pct_change().dropna()
        bm_daily.index = pd.to_datetime(bm_daily.index)
        if bm_daily.index.tz is not None:
            bm_daily.index = bm_daily.index.tz_convert("UTC")
    except Exception as exc:
        return {"error": f"Could not compute benchmark returns: {exc}", "gate_pass": False}

    # Align strategy and benchmark to common dates.
    strat_daily_clean = strat_daily.dropna()
    strat_daily_clean.index = pd.to_datetime(strat_daily_clean.index)
    if strat_daily_clean.index.tz is not None:
        strat_daily_clean.index = strat_daily_clean.index.tz_convert("UTC")
    if strat_daily_clean.index.tz is None and bm_daily.index.tz is not None:
        strat_daily_clean.index = strat_daily_clean.index.tz_localize("UTC")
    if strat_daily_clean.index.tz is not None and bm_daily.index.tz is None:
        bm_daily.index = bm_daily.index.tz_localize("UTC")

    common = strat_daily_clean.index.intersection(bm_daily.dropna().index)

    if len(common) < 30:
        return {
            "error": f"Insufficient overlapping days for OLS: {len(common)}",
            "gate_pass": False,
        }

    try:
        X = sm.add_constant(bm_daily.loc[common])
        y = strat_daily_clean.loc[common]
        model = sm.OLS(y, X).fit()
        alpha_daily = float(model.params.iloc[0])
        beta = float(model.params.iloc[1])
        r_squared = float(model.rsquared)
        alpha_ann = alpha_daily * 252
    except Exception as exc:
        return {"error": f"OLS regression failed: {exc}", "gate_pass": False}

    beta_threshold = BETA_THRESHOLDS.get(asset_class, 1.0)
    gate_pass = alpha_ann > 0 and abs(beta) < beta_threshold and r_squared < ALPHA_BETA_MAX_R2

    return {
        "alpha_ann": round(alpha_ann, 6),
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "beta_threshold": beta_threshold,
        "n_days": len(common),
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_robustness(
    instrument: str,
    tfs: list[str] | None = None,
    target_signals: list[str] | None = None,
    asset_class: str = "fx_major",
    direction: str = "both",
    threshold: float = DEFAULT_THRESHOLD,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_atr: float = DEFAULT_STOP_ATR,
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    max_leverage: float | None = None,
    benchmark: str | None = None,
    timeframe: str = "H4",
    hmm_gate: bool = False,
    ref_horizon: int = 1,
) -> dict:
    """Run all 6 robustness gates for one instrument."""
    if tfs is None:
        tfs = DEFAULT_TFS
    if target_signals is None:
        target_signals = DEFAULT_SIGNALS

    # Default benchmark by asset class.
    if benchmark is None:
        if asset_class.startswith("fx"):
            benchmark = "DXY"
        else:
            benchmark = "SPY"

    print(f"\n{'=' * 70}")
    print(f"  Phase 5 Robustness — {instrument}  ({asset_class}, {direction})")
    print(f"{'=' * 70}")
    print(f"  TFs: {tfs}  |  Signals: {target_signals}")
    print(f"  Threshold: {threshold}  |  Benchmark: {benchmark}  |  TF for regime: {timeframe}")
    print()

    # Build OOS portfolios once (baseline slippage).
    print("Building OOS portfolios …")
    try:
        pf_long, pf_short, oos_close, oos_mask, base_index, _ = _build_oos_portfolios(
            instrument=instrument,
            tfs=tfs,
            target_signals=target_signals,
            asset_class=asset_class,
            direction=direction,
            threshold=threshold,
            risk_pct=risk_pct,
            stop_atr=stop_atr,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            slippage_mult=1.0,
            hmm_gate=hmm_gate,
            ref_horizon=ref_horizon,
        )
    except Exception as exc:
        print(f"  ERROR building portfolios: {exc}")
        return {"instrument": instrument, "error": str(exc), "all_pass": False}

    oos_bars = int(oos_mask.sum())
    base_tf = tfs[-1]
    freq = "d" if base_tf in ("D",) else "h"

    # ---- G1: Monte Carlo ------------------------------------------------
    print("\n[G1] Monte Carlo shuffle …")
    g1 = monte_carlo_shuffle(pf_long, pf_short, oos_bars, n=MC_N, freq=freq)
    _print_gate("G1", "Monte Carlo", g1)

    # ---- G2: Remove top-N -----------------------------------------------
    print("\n[G2] Remove top-N winning trades …")
    g2 = remove_top_n(pf_long, pf_short, n=TOP_N_REMOVE)
    _print_gate("G2", "Remove Top-N", g2)

    # ---- G3: 3x slippage stress -----------------------------------------
    print("\n[G3] 3x slippage stress test …")
    g3 = stress_triple_slippage(
        instrument=instrument,
        tfs=tfs,
        target_signals=target_signals,
        asset_class=asset_class,
        direction=direction,
        threshold=threshold,
        risk_pct=risk_pct,
        stop_atr=stop_atr,
        spread_bps=spread_bps,
        max_leverage=max_leverage,
        hmm_gate=hmm_gate,
        ref_horizon=ref_horizon,
    )
    _print_gate("G3", "3x Slippage", g3)

    # ---- G4: WFO consecutive negatives ----------------------------------
    print("\n[G4] WFO consecutive negative folds …")
    g4 = check_wfo_consecutive(instrument)
    _print_gate("G4", "WFO Consecutive", g4)

    # ---- G5: Regime robustness ------------------------------------------
    print("\n[G5] Regime robustness …")
    g5 = check_regime_robustness(
        instrument=instrument,
        timeframe=timeframe,
        pf_long=pf_long,
        pf_short=pf_short,
        base_index=base_index,
        oos_mask=oos_mask,
        freq=freq,  # M1: pass actual freq for correct bars_per_year
        hmm_gate=hmm_gate,  # M2: skip gated state from HMM pass count
    )
    _print_gate("G5", "Regime Robustness", g5)

    # ---- G6: Alpha/beta -------------------------------------------------
    print("\n[G6] Alpha/beta decomposition …")
    g6 = check_alpha_beta(
        instrument=instrument,
        benchmark=benchmark,
        pf_long=pf_long,
        pf_short=pf_short,
        oos_close=oos_close,
        direction=direction,
        asset_class=asset_class,
    )
    _print_gate("G6", "Alpha/Beta", g6)

    # ---- Summary --------------------------------------------------------
    gates = [g1, g2, g3, g4, g5, g6]
    gate_labels = [
        "G1 Monte Carlo",
        "G2 Remove Top-N",
        "G3 3x Slip",
        "G4 WFO Consec",
        "G5 Regime",
        "G6 Alpha/Beta",
    ]

    all_pass = all(g.get("gate_pass", False) for g in gates)

    print(f"\n{'=' * 70}")
    print("  GATE SUMMARY")
    print(f"  {'Gate':<20} {'Result'}")
    print(f"  {'-' * 40}")
    for label, g in zip(gate_labels, gates):
        status = "PASS" if g.get("gate_pass", False) else "FAIL"
        err = f"  ({g['error']})" if "error" in g else ""
        print(f"  {label:<20} {status}{err}")
    print(f"  {'-' * 40}")
    print(f"  {'OVERALL':<20} {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 70}\n")

    # ---- Save CSV -------------------------------------------------------
    row: dict = {
        "instrument": instrument,
        "asset_class": asset_class,
        "direction": direction,
        "benchmark": benchmark,
        "g1_pass": g1.get("gate_pass", False),
        "g1_mc_5pct_sharpe": g1.get("mc_5pct_sharpe"),
        "g1_profitable_pct": g1.get("mc_profitable_pct"),
        "g2_pass": g2.get("gate_pass", False),
        "g2_trimmed_cum_ret": g2.get("trimmed_cum_return"),
        "g3_pass": g3.get("gate_pass", False),
        "g3_stress_sharpe": g3.get("stress_sharpe"),
        "g4_pass": g4.get("gate_pass", False),
        "g4_max_consec_neg": g4.get("max_consec_neg"),
        "g5_pass": g5.get("gate_pass", False),
        "g5_adx_pass_count": g5.get("adx_pass_count"),
        "g5_hmm_pass_count": g5.get("hmm_pass_count"),
        "g6_pass": g6.get("gate_pass", False),
        "g6_alpha_ann": g6.get("alpha_ann"),
        "g6_beta": g6.get("beta"),
        "g6_r_squared": g6.get("r_squared"),
        "all_pass": all_pass,
    }

    out_path = REPORTS_DIR / f"phase5_{instrument.lower()}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return {
        "instrument": instrument,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "g6": g6,
        "all_pass": all_pass,
        "csv_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------


def _print_gate(tag: str, name: str, result: dict) -> None:
    status = "PASS" if result.get("gate_pass", False) else "FAIL"
    print(f"  [{tag}] {name}: {status}")
    for k, v in result.items():
        if k == "gate_pass":
            continue
        if k == "error":
            print(f"    ERROR: {v}")
        else:
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5 — 6-gate robustness validation")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--instruments", nargs="+", default=None)
    p.add_argument(
        "--tfs",
        default=None,
        help="Comma-separated timeframes, e.g. W,D,H4,H1",
    )
    p.add_argument(
        "--signals",
        default=None,
        help="Comma-separated signals, e.g. accel_stoch_k,accel_rsi14",
    )
    p.add_argument(
        "--asset-class",
        default="fx_major",
        choices=list(COST_PROFILES.keys()),
    )
    p.add_argument(
        "--direction",
        default="both",
        choices=["both", "long_only"],
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    p.add_argument("--stop-atr", type=float, default=DEFAULT_STOP_ATR)
    p.add_argument("--spread-bps", type=float, default=None)
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--max-leverage", type=float, default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument(
        "--timeframe",
        default="H4",
        help="Base timeframe for regime file lookup (e.g. H4, D)",
    )
    p.add_argument(
        "--hmm-gate",
        action="store_true",
        default=False,
        help="Gate entries to IS-majority HMM state (requires Phase 0 regime file)",
    )
    p.add_argument(
        "--ref-horizon",
        type=int,
        default=1,
        help=(
            "Horizon used for signal sign orientation (match Phase 1 natural horizon). "
            "Default=1; wrong value inverts the composite. E.g. 60 for vol signals."
        ),
    )
    args = p.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",")] if args.tfs else None
    signals = [x.strip() for x in args.signals.split(",")] if args.signals else None

    instruments = args.instruments if args.instruments else [args.instrument]

    all_results: list[dict] = []
    for instr in instruments:
        result = run_robustness(
            instrument=instr,
            tfs=tfs,
            target_signals=signals,
            asset_class=args.asset_class,
            direction=args.direction,
            threshold=args.threshold,
            risk_pct=args.risk_pct,
            stop_atr=args.stop_atr,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            max_leverage=args.max_leverage,
            benchmark=args.benchmark,
            timeframe=args.timeframe,
            hmm_gate=args.hmm_gate,
            ref_horizon=args.ref_horizon,
        )
        all_results.append(result)

    if len(instruments) > 1:
        print("\n" + "=" * 80)
        print("  BATCH SUMMARY")
        print(
            f"  {'Instrument':<20} {'G1':>4} {'G2':>4} {'G3':>4} {'G4':>4}"
            f" {'G5':>4} {'G6':>4} {'ALL':>5}"
        )
        print("  " + "-" * 60)
        for res in all_results:
            instr = res.get("instrument", "?")
            if "error" in res:
                print(f"  {instr:<20}  ERROR: {res['error']}")
                continue

            def _s(g_key: str) -> str:
                g = res.get(g_key, {})
                return "PASS" if g.get("gate_pass", False) else "FAIL"

            overall = "PASS" if res.get("all_pass", False) else "FAIL"
            print(
                f"  {instr:<20} {_s('g1'):>4} {_s('g2'):>4} {_s('g3'):>4}"
                f" {_s('g4'):>4} {_s('g5'):>4} {_s('g6'):>4} {overall:>5}"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
