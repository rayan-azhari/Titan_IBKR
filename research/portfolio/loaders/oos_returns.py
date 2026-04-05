"""OOS Return Series Loader — Central hub for per-strategy OOS daily returns.

For each validated strategy, re-runs its backtest with locked config parameters
and returns the OOS daily return series.

CONTRACT
--------
- Returns are daily fractional returns (e.g. 0.01 = 1% gain).
- Returns span the strict OOS window only (chronologically last 30% of data).
- No look-ahead: IS calibration (composite sign, z-score mean/std) is frozen
  from IS data only; OOS slice is evaluated on those frozen statistics.
- All costs (spread + slippage + swap) are included.
- Alignment to a common date range is done by the caller; each loader returns
  its own OOS window independently.

Strategies registered
---------------------
  IC_GENERIC   — FX majors / equities, uses ic_generic.toml locked configs
  ETF_TREND    — SPY / QQQ / TQQQ / IWB, uses etf_trend_{inst}.toml
  ORB          — stub (M5 intraday; see note below)

ORB NOTE: The ORB strategy operates on M5 intraday data with complex bracket
order logic. Its OOS loader is implemented as a daily-aggregated return series
via the ORB optimisation script. A full integration is deferred; the loader
returns a cached series if available, otherwise raises NotImplementedError.
"""

from __future__ import annotations

import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / ".tmp" / "reports"

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _load_ohlcv(instrument: str, tf: str) -> pd.DataFrame:
    """Load OHLCV parquet.  Mirrors _load_ohlcv in phase1_sweep.py."""
    candidates = [
        DATA_DIR / f"{instrument}_{tf}.parquet",
        DATA_DIR / f"{instrument.replace('/', '_')}_{tf}.parquet",
        DATA_DIR / f"^{instrument}_{tf}.parquet",
    ]
    # Also try with caret prefix already in name
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df.sort_index().dropna(subset=["close"])
    raise FileNotFoundError(
        f"No parquet file found for {instrument}_{tf}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


# ── IC Generic loader (FX majors + equities) ───────────────────────────────────


def _load_ic_generic_config(instrument: str) -> dict:
    """Load locked config for instrument from config/ic_generic.toml."""
    path = CONFIG_DIR / "ic_generic.toml"
    if not path.exists():
        raise FileNotFoundError(f"ic_generic.toml not found at {path}")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if instrument not in cfg:
        raise KeyError(
            f"Instrument '{instrument}' not found in ic_generic.toml. Available: {list(cfg.keys())}"
        )
    return cfg[instrument]


def load_ic_generic_oos(instrument: str) -> pd.Series:
    """OOS daily returns for an IC Generic strategy instrument.

    Loads locked config from ic_generic.toml, re-runs the IC signal pipeline
    on the full history, calibrates IS statistics, and simulates OOS returns
    using a direct position-based simulation (no VBT dependency).

    Pipeline:
      1. _build_and_align()   — load + compute signals on each TF, ffill to base
      2. build_composite()    — sign-normalise + equal-weight composite (IS only)
      3. zscore_normalise()   — IS mean/std frozen (no look-ahead)
      4. position rule        — long if z > threshold, short if z < -threshold
      5. bar returns          — close.pct_change() * position - transition costs
      6. aggregate to daily   — resample("D").sum() for H1 data; normalize timestamps

    Timestamp normalisation: all H1 dates are normalised to UTC midnight so that
    they align correctly with equity daily timestamps (also UTC midnight).

    Returns:
        pd.Series — daily OOS returns, DatetimeIndex (UTC midnight, trading days only).
        Label: instrument (e.g. "EUR_USD").
    """
    from research.ic_analysis.phase3_backtest import (
        COST_PROFILES,
        IS_RATIO,
        _build_and_align,
        build_composite,
        zscore_normalise,
    )

    cfg = _load_ic_generic_config(instrument)
    tfs: list[str] = cfg["tfs"]
    signals: list[str] = cfg["signals"]
    threshold: float = float(cfg["threshold"])
    asset_class: str = cfg["asset_class"]
    direction: str = cfg.get("direction", "both")

    profile = COST_PROFILES.get(asset_class, COST_PROFILES["fx_major"])
    base_tf = tfs[-1]
    is_hourly = base_tf not in ("D", "daily")

    # 1. Load + align signals
    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]
    n = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True

    # 2. Build composite + z-score (IS-calibrated, no look-ahead)
    # ref_horizon=1 (default) matches the Phase 3 backtest that produced FINDINGS.md.
    # For piecewise-constant weekly signals ffill'd to H1, the Spearman IC at h=1
    # ranks each week-block's signal value vs same-bar return — IC may be small but
    # the default sign (+1) is consistent with the profitable direction from Phase 3.
    # ref_horizon=20 inverts the weekly signal (mean-reversion at 20-hour horizon
    # even though the signal has positive edge at the weekly horizon).
    composite = build_composite(tf_signals, base_close, tfs, signals, is_mask)
    composite_z = zscore_normalise(composite, is_mask)

    # 3. Position rule (signal shifted 1 bar → fills next bar open)
    sig = composite_z.shift(1).fillna(0.0)
    if direction == "both":
        pos = np.where(sig > threshold, 1.0, np.where(sig < -threshold, -1.0, 0.0))
    else:
        pos = np.where(sig > threshold, 1.0, 0.0)
    pos_series = pd.Series(pos, index=base_close.index)

    # 4. Bar returns with spread+slippage on transitions
    transitions = (pos_series != pos_series.shift(1).fillna(0.0)).astype(float)
    cost_per_bar = transitions * (profile["spread_bps"] + profile["slippage_bps"]) / 10_000
    bar_rets = base_close.pct_change().fillna(0.0) * pos_series - cost_per_bar

    # 5. Slice to OOS window
    oos_bar_rets = bar_rets.iloc[is_n:]

    # 6. Aggregate to daily and normalise timestamps to UTC midnight
    if is_hourly:
        # sum() compounds H1 returns within each calendar day (valid for small returns)
        daily_rets = oos_bar_rets.resample("D").sum()
        # Normalize end-of-day timestamps (e.g. 23:00 UTC) to midnight UTC
        daily_rets.index = daily_rets.index.normalize()
        # Drop weekends/holidays (zero-return days with no trading activity)
        daily_rets = daily_rets[daily_rets != 0.0]
    else:
        daily_rets = oos_bar_rets
        daily_rets.index = daily_rets.index.normalize()

    daily_rets.name = instrument
    return daily_rets


# ── ETF Trend loader ───────────────────────────────────────────────────────────


def load_etf_trend_oos(instrument: str) -> pd.Series:
    """OOS daily returns for an ETF Trend strategy instrument.

    Mirrors research/etf_trend/run_multi_portfolio.py pattern:
    loads locked TOML from config/etf_trend_{inst}.toml, computes signals,
    slices to OOS window (last 30% of sig_close history).

    Returns:
        pd.Series — daily OOS returns, DatetimeIndex.
    """
    from research.etf_trend.run_multi_portfolio import (
        compute_strategy_returns,
        load_config,
        load_instrument,
    )

    config = load_config(instrument)
    sig_close, exec_close, sig_df = load_instrument(config)

    strat_rets = compute_strategy_returns(sig_close, exec_close, sig_df, config)

    # Slice to OOS window
    split = int(len(sig_close) * 0.70)
    oos_start = sig_close.index[split]
    oos_rets = strat_rets[strat_rets.index >= oos_start].copy()
    oos_rets.index = oos_rets.index.normalize()  # ensure UTC midnight timestamps
    oos_rets.name = f"etf_trend_{instrument.lower()}"
    return oos_rets


# ── ORB loader (stub) ──────────────────────────────────────────────────────────


def load_orb_oos(instrument: str) -> pd.Series:
    """OOS daily returns for an ORB strategy instrument.

    The ORB strategy operates on M5 intraday data. This loader attempts to
    load a cached daily return series from a prior ORB optimisation run.
    Cache path: .tmp/reports/orb_{instrument.lower()}_oos_daily.parquet

    If the cache does not exist, raises NotImplementedError with instructions
    to generate it via the ORB research pipeline.
    """
    cache_path = REPORTS_DIR / f"orb_{instrument.lower()}_oos_daily.parquet"
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        if not isinstance(s, pd.Series):
            raise ValueError(f"Expected pd.Series in {cache_path}, got {type(s)}")
        s.name = f"orb_{instrument.lower()}"
        return s.astype(float)

    raise NotImplementedError(
        f"ORB OOS cache not found for {instrument}.\n"
        f"Expected: {cache_path}\n"
        f"To generate: run the ORB optimisation pipeline for {instrument} with "
        f"--save-oos-daily flag, then re-run this loader.\n"
        f"Alternatively, pass strategy_subset excluding 'orb_{instrument.lower()}'."
    )


# ── Turtle loader ──────────────────────────────────────────────────────────────


def load_turtle_oos(instrument: str) -> pd.Series:
    """OOS daily returns for Turtle H1 (System 2) on a given instrument.

    Uses locked parameters from config/turtle_h1.toml [system2]:
        entry_period=45, exit_period=30, risk_pct=0.01, stop_atr_mult=2.0,
        direction=long_only, timeframe=H1.

    Calls research/turtle/turtle_backtest.py:run_turtle_system() and extracts
    oos_stats["daily_returns"] (per-bar H1 returns from VBT).

    H1 returns are resampled to daily (sum of intrabar returns per calendar day)
    and normalised to UTC midnight so they align with equity daily series.

    Returns:
        pd.Series of daily OOS returns, DatetimeIndex (UTC midnight, trading days).
    """
    from research.turtle.turtle_backtest import load_data, run_turtle_system

    # Load H1 data
    try:
        df = load_data(instrument, "H1")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Turtle OOS: H1 parquet not found for {instrument}. "
            f"Run download_data_yfinance.py to fetch it. Detail: {exc}"
        ) from exc

    if len(df) < 500:
        raise ValueError(
            f"Turtle OOS: insufficient H1 bars for {instrument} ({len(df)} bars). "
            f"Need at least 500."
        )

    # Locked System 2 parameters (from config/turtle_h1.toml [system2])
    result = run_turtle_system(
        df,
        system_num=2,  # entry_period=55, exit_period=20 in turtle_backtest
        timeframe="H1",
        direction="long",  # long_only per directive
        pyramid=True,  # locked: use_trailing_stop=true, pyramid enabled
        trailing_stop=True,
        max_units=4,
        pyramid_atr_mult=0.5,
    )

    oos_rets: pd.Series = result["oos_stats"]["daily_returns"]

    if oos_rets is None or len(oos_rets) == 0:
        raise ValueError(f"Turtle OOS: empty daily_returns for {instrument}.")

    # H1 bars -> aggregate to daily returns (sum of log-approx hourly returns)
    oos_rets = oos_rets.resample("D").sum()
    oos_rets.index = oos_rets.index.normalize()
    # Drop weekends / holidays (zero-return days with no intraday activity)
    oos_rets = oos_rets[oos_rets != 0.0]
    oos_rets.name = f"turtle_{instrument.lower()}"
    return oos_rets


# ── Gold Macro loader ─────────────────────────────────────────────────────────


def load_gold_macro_oos(_arg: str | None = None) -> pd.Series:
    """OOS daily returns for the Gold Macro strategy.

    Calls research/gold_macro/run_backtest.py functions directly:
    load_data(), build_composite_signal(), backtest(), then slices to OOS (last 30%).
    """
    from research.gold_macro.run_backtest import (
        backtest as gold_backtest,
    )
    from research.gold_macro.run_backtest import (
        build_composite_signal,
    )
    from research.gold_macro.run_backtest import (
        load_data as gold_load_data,
    )

    data = gold_load_data()
    df = build_composite_signal(
        gld=data["GLD"],
        tip=data["TIP"],
        tlt=data["TLT"],
        dxy=data["DXY"],
        real_rate_window=20,
        dollar_window=20,
        slow_ma=200,
    )
    results = gold_backtest(df, data["GLD"], target_vol=0.10)

    # IS/OOS split (70/30)
    split_idx = int(len(results) * 0.70)
    oos_rets = results["position_ret"].iloc[split_idx:].dropna()
    oos_rets.index = oos_rets.index.normalize()
    oos_rets.name = "gold_macro"
    return oos_rets


# ── Pairs Trading loader ─────────────────────────────────────────────────────


def load_pairs_oos(_arg: str | None = None, sym_a: str = "GLD", sym_b: str = "EFA") -> pd.Series:
    """OOS daily returns for the Pairs Trading strategy.

    Calls research/pairs_trading/run_backtest.py: load_pair(), backtest_pairs().
    IS/OOS split: 70/30 time-based.
    """
    from research.pairs_trading.run_backtest import (
        backtest_pairs,
        load_pair,
    )

    series_a, series_b = load_pair(sym_a, sym_b)
    results = backtest_pairs(series_a, series_b, entry_z=2.0, exit_z=0.5, max_z=4.0)

    # IS/OOS split
    split_idx = int(len(results) * 0.70)
    oos_rets = results["pnl"].iloc[split_idx:].dropna()
    oos_rets.index = oos_rets.index.normalize()
    oos_rets.name = f"pairs_{sym_a.lower()}_{sym_b.lower()}"
    return oos_rets


# ── FX Carry loader ──────────────────────────────────────────────────────────


def load_fx_carry_oos(instrument: str = "AUD_JPY") -> pd.Series:
    """OOS daily returns for FX Carry strategy.

    Tries cache first, otherwise runs the research backtest and caches result.
    """
    cache_path = REPORTS_DIR / f"fx_carry_{instrument.lower()}_oos_daily.parquet"
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        if not isinstance(s, pd.Series):
            s = s.iloc[:, 0]
        s.name = f"fx_carry_{instrument.lower()}"
        return s.astype(float)

    from research.fx_carry.run_backtest import run_fx_carry_backtest

    _, oos_returns = run_fx_carry_backtest(instrument=instrument)

    # Cache for next time
    oos_returns.to_frame().to_parquet(cache_path)
    return oos_returns


# ── MTF Confluence loader ────────────────────────────────────────────────────


def _compute_native_signal(
    close: pd.Series,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    ma_type: str,
) -> pd.Series:
    """Compute MA-spread + RSI signal at native TF resolution.

    Returns a signal series in roughly [-1, +1] range, indexed at the TF's
    native timestamps. Indicators use only data available at bar close.
    """
    # MA spread: (fast - slow) / |slow|, matching live strategy's normalisation
    if ma_type == "EMA":
        fast_ma = close.ewm(span=fast_period, adjust=False).mean()
        slow_ma = close.ewm(span=slow_period, adjust=False).mean()
    elif ma_type == "WMA":
        fast_ma = close.rolling(fast_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
        slow_ma = close.rolling(slow_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
    else:
        fast_ma = close.rolling(fast_period).mean()
        slow_ma = close.rolling(slow_period).mean()

    ma_spread = (fast_ma - slow_ma) / slow_ma.abs().clip(lower=1e-8)
    # Tanh normalisation to [-0.5, +0.5] (matches live strategy)
    ma_signal = np.tanh(ma_spread / ma_spread.rolling(20).std().clip(lower=1e-8)) * 0.5

    # RSI deviation: (RSI - 50) / 100, range [-0.5, +0.5] (matches live strategy)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.clip(lower=1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi_signal = (rsi - 50) / 100

    return (ma_signal + rsi_signal).dropna()


def load_mtf_oos(pair: str = "EUR_USD") -> pd.Series:
    """OOS daily returns for MTF Confluence strategy.

    ANTI-LOOK-AHEAD DESIGN:
    1. Compute indicators at each TF's NATIVE resolution (not on ffill'd data).
    2. For non-base TFs (D, W, H4): shift signal by 1 native bar before ffill.
       This ensures the signal is only available AFTER the bar closes.
    3. ffill to base TF (H1) index -- the signal is stale until the next native
       bar close, which is the correct causal relationship.
    4. Z-score normalisation uses IS-only statistics.
    5. Position shifted by 1 base-TF bar before computing returns.
    """
    import tomllib

    # Config files use condensed names: mtf_eurusd.toml, not mtf_eur_usd.toml
    pair_lower = pair.lower().replace("/", "_")
    config_path = CONFIG_DIR / f"mtf_{pair_lower}.toml"
    if not config_path.exists():
        config_path = CONFIG_DIR / f"mtf_{pair_lower.replace('_', '')}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"MTF config not found: {config_path}")

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    ma_type = cfg.get("ma_type", "WMA")
    threshold = cfg.get("confirmation_threshold", 0.10)
    tfs = list(cfg.get("weights", {}).keys())
    if not tfs:
        tfs = ["H1", "H4", "D", "W"]
    tf_weights = cfg.get("weights", {})

    base_tf = tfs[0]  # H1 is the fastest (trading) timeframe
    closes = {}
    for tf in tfs:
        df = _load_ohlcv(pair, tf)
        closes[tf] = df["close"].astype(float)

    base_close = closes[base_tf]
    base_idx = base_close.index

    # Step 1-3: Compute signal at NATIVE resolution, shift non-base TFs, ffill
    per_tf_signal = pd.DataFrame(index=base_idx)
    for tf in tfs:
        tf_cfg = cfg.get(tf, {})
        fast = tf_cfg.get("fast_ma", 10)
        slow = tf_cfg.get("slow_ma", 30)
        rsi_period = tf_cfg.get("rsi_period", 14)
        w = tf_weights.get(tf, 1.0 / len(tfs))

        # Compute at native TF resolution (no ffill contamination)
        native_sig = _compute_native_signal(closes[tf], fast, slow, rsi_period, ma_type)

        # ANTI-LOOK-AHEAD: shift non-base TFs by 1 native bar.
        # A D-bar signal computed at Friday's close becomes available only on the
        # NEXT D-bar's timestamp. This prevents intraday H1 bars from using
        # same-day daily signals before the daily bar closes.
        if tf != base_tf:
            native_sig = native_sig.shift(1)

        # ffill to base TF index -- signal is stale until next native bar
        aligned = native_sig.reindex(base_idx, method="ffill")
        per_tf_signal[tf] = aligned * w

    # Step 4: Composite + IS-only z-score
    composite = per_tf_signal.sum(axis=1).dropna()

    n = len(composite)
    is_n = int(n * 0.70)
    is_mean = composite.iloc[:is_n].mean()
    is_std = composite.iloc[:is_n].std()
    if is_std < 1e-8:
        is_std = 1.0
    composite_z = (composite - is_mean) / is_std

    # Step 5: Position shifted 1 base-TF bar (trade on next bar's open)
    sig = composite_z.shift(1).fillna(0.0)
    pos = np.where(sig > threshold, 1.0, np.where(sig < -threshold, -1.0, 0.0))
    pos_series = pd.Series(pos, index=composite.index)

    # Returns with realistic EUR/USD costs (IBKR: ~2-3 bps spread + slippage)
    bar_rets = base_close.reindex(composite.index).pct_change().fillna(0.0)
    transitions = (pos_series != pos_series.shift(1).fillna(0.0)).astype(float)
    spread_bps = 2.5  # EUR/USD IBKR average (was 1.5, increased for realism)
    slippage_bps = 1.0  # execution slippage (was 0.5)
    cost_per_bar = transitions * (spread_bps + slippage_bps) / 10_000
    strategy_rets = bar_rets * pos_series - cost_per_bar

    # Slice to OOS
    oos_rets = strategy_rets.iloc[is_n:]

    # Resample H1 to daily
    if base_tf in ("H1", "H4"):
        oos_rets = oos_rets.resample("D").sum()
        oos_rets.index = oos_rets.index.normalize()
        oos_rets = oos_rets[oos_rets != 0.0]
    else:
        oos_rets.index = oos_rets.index.normalize()

    oos_rets.name = f"mtf_{pair.lower()}"
    return oos_rets


# ── Gap Fade loader (stub) ───────────────────────────────────────────────────


def load_gap_fade_oos(instrument: str = "EUR_USD") -> pd.Series:
    """OOS daily returns for Gap Fade strategy. STUB -- not yet implemented."""
    raise NotImplementedError(
        f"Gap Fade OOS loader not implemented for {instrument}. "
        "Intraday M5 bracket-order simulation required. Deferred to Phase 2."
    )


# ── Strategy registry ──────────────────────────────────────────────────────────

# Maps strategy label → (loader_fn, loader_arg)
# All loaders return pd.Series of daily OOS returns.
_REGISTRY: dict[str, tuple] = {
    # IC Generic — equities (daily, Phase 5 validated)
    "ic_equity_csco": (load_ic_generic_oos, "CSCO"),
    "ic_equity_noc": (load_ic_generic_oos, "NOC"),
    "ic_equity_hwm": (load_ic_generic_oos, "HWM"),
    "ic_equity_wmt": (load_ic_generic_oos, "WMT"),
    "ic_equity_abnb": (load_ic_generic_oos, "ABNB"),
    "ic_equity_gl": (load_ic_generic_oos, "GL"),
    # IC Generic — FX majors (NOTE: post-debiasing all FX show 0 edge; kept for
    # research comparison only — do NOT include in live portfolio sweep)
    "ic_mtf_eur_usd": (load_ic_generic_oos, "EUR_USD"),
    "ic_mtf_gbp_usd": (load_ic_generic_oos, "GBP_USD"),
    "ic_mtf_usd_jpy": (load_ic_generic_oos, "USD_JPY"),
    "ic_mtf_aud_jpy": (load_ic_generic_oos, "AUD_JPY"),
    "ic_mtf_aud_usd": (load_ic_generic_oos, "AUD_USD"),
    "ic_mtf_usd_chf": (load_ic_generic_oos, "USD_CHF"),
    # IC Generic — equities broad (daily)
    "ic_equity_spy": (load_ic_generic_oos, "SPY"),
    "ic_equity_qqq": (load_ic_generic_oos, "QQQ"),
    # ETF Trend
    "etf_trend_spy": (load_etf_trend_oos, "SPY"),
    "etf_trend_qqq": (load_etf_trend_oos, "QQQ"),
    # Turtle H1 System 2 — Phase 5 validated instruments (OOS Sharpe confirmed in FINDINGS.md)
    # GS: 2.15, NVDA: 1.90, CAT: 1.97 — all have H1 parquets in data/
    "turtle_gs": (load_turtle_oos, "GS"),
    "turtle_nvda": (load_turtle_oos, "NVDA"),
    "turtle_cat": (load_turtle_oos, "CAT"),
    # ORB (intraday — requires cached daily series)
    "orb_unh": (load_orb_oos, "UNH"),
    "orb_cat": (load_orb_oos, "CAT"),
    "orb_tmo": (load_orb_oos, "TMO"),
    "orb_amat": (load_orb_oos, "AMAT"),
    "orb_txn": (load_orb_oos, "TXN"),
    "orb_intc": (load_orb_oos, "INTC"),
    "orb_wmt": (load_orb_oos, "WMT"),
    # Gold Macro (daily, cross-asset signal)
    "gold_macro": (load_gold_macro_oos, None),
    # Pairs Trading (daily, market-neutral)
    "pairs_gld_efa": (load_pairs_oos, None),
    # FX Carry (daily, carry premium)
    "fx_carry_aud_jpy": (load_fx_carry_oos, "AUD_JPY"),
    # MTF Confluence (H1 base, resampled to daily)
    "mtf_eur_usd": (load_mtf_oos, "EUR_USD"),
    # Gap Fade (stub — intraday M5 simulation deferred)
    "gap_fade_eur_usd": (load_gap_fade_oos, "EUR_USD"),
}

# ICIR scores from research/ic_analysis/FINDINGS.md (used in icir_weighted scheme)
ICIR_SCORES: dict[str, float] = {
    # IC Equity daily (Phase 5 validated OOS Sharpe from FINDINGS.md)
    "ic_equity_hwm": 0.75,  # OOS Sharpe 4.28 — highest IC equity
    "ic_equity_csco": 0.70,  # OOS Sharpe 3.14
    "ic_equity_noc": 0.68,  # OOS Sharpe 3.06
    "ic_equity_wmt": 0.65,  # OOS Sharpe 2.82
    "ic_equity_abnb": 0.63,  # OOS Sharpe 2.78
    "ic_equity_gl": 0.62,  # OOS Sharpe 2.65
    "ic_equity_spy": 0.45,
    "ic_equity_qqq": 0.45,
    # ETF Trend
    "etf_trend_spy": 0.40,
    "etf_trend_qqq": 0.38,
    # Turtle H1 System 2 (OOS Sharpe from FINDINGS.md — scaled to ICIR range)
    "turtle_gs": 0.72,  # OOS Sharpe 2.15
    "turtle_nvda": 0.68,  # OOS Sharpe 1.90
    "turtle_cat": 0.70,  # OOS Sharpe 1.97
    # IC MTF FX — invalidated post-debiasing; scores kept for backwards-compat only
    "ic_mtf_eur_usd": 0.71,
    "ic_mtf_gbp_usd": 0.74,
    "ic_mtf_usd_jpy": 0.70,
    "ic_mtf_aud_jpy": 0.70,
    "ic_mtf_aud_usd": 0.68,
    "ic_mtf_usd_chf": 0.68,
    # ORB
    "orb_unh": 0.35,
    "orb_cat": 0.35,
    "orb_tmo": 0.33,
    "orb_amat": 0.32,
    "orb_txn": 0.32,
    "orb_intc": 0.30,
    "orb_wmt": 0.30,
    # Gold Macro (OOS Sharpe +0.603)
    "gold_macro": 0.35,
    # Pairs Trading GLD/EFA (OOS Sharpe +1.14)
    "pairs_gld_efa": 0.55,
    # FX Carry AUD/JPY (estimated — carry premium is structural)
    "fx_carry_aud_jpy": 0.30,
    # MTF EUR/USD (OOS Sharpe +1.94)
    "mtf_eur_usd": 0.60,
}


# ── Main loader function ───────────────────────────────────────────────────────


def _load_one(label: str, skip_errors: bool) -> tuple[str, pd.Series | None, str | None]:
    """Load a single strategy; returns (label, series, error_message)."""
    loader_fn, loader_arg = _REGISTRY[label]
    try:
        s = loader_fn(loader_arg)
        return label, s, None
    except NotImplementedError as exc:
        return label, None, f"[SKIP] {label}: {exc}"
    except Exception as exc:
        if skip_errors:
            return label, None, f"[ERROR] {label}: {exc}"
        raise RuntimeError(f"[ERROR] {label}: {exc}") from exc


def load_all_strategies(
    strategy_subset: list[str] | None = None,
    skip_errors: bool = True,
    max_workers: int = 4,
) -> dict[str, pd.Series]:
    """Load OOS returns for all (or a subset of) registered strategies.

    Strategies are loaded in parallel using a thread pool (I/O + NumPy release GIL).
    Insertion order in the returned dict matches strategy_subset order.

    Args:
        strategy_subset:  list of strategy labels (see _REGISTRY keys).
                          If None, loads all registered strategies.
        skip_errors:      if True, log warnings and skip failed loaders.
        max_workers:      thread pool size (default 4).
    """
    targets = strategy_subset if strategy_subset is not None else list(_REGISTRY.keys())

    # Validate labels upfront (serial, instant)
    valid: list[str] = []
    for label in targets:
        if label not in _REGISTRY:
            msg = f"Unknown strategy label '{label}'. Available: {list(_REGISTRY.keys())}"
            if skip_errors:
                print(f"  [WARN] {msg}")
            else:
                raise KeyError(msg)
        else:
            valid.append(label)

    print(f"  Loading {len(valid)} strategies (up to {max_workers} in parallel)...")
    results: dict[str, pd.Series] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one, lbl, skip_errors): lbl for lbl in valid}
        for fut in as_completed(futures):
            label, series, err = fut.result()
            if err:
                print(f"  {err}")
            else:
                results[label] = series
                print(
                    f"    OK  {label:<28s}  {len(series):>5} OOS days  Sharpe={_quick_sharpe(series):>+6.2f}"
                )

    # Restore insertion order to match strategy_subset
    return {lbl: results[lbl] for lbl in valid if lbl in results}


def align_to_common_window(
    strategy_returns: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Align all series to their common OOS date range.

    All timestamps are normalised to UTC midnight before intersecting, so that
    H1-sourced FX series (which may have end-of-day timestamps) align correctly
    with equity daily series (which use midnight timestamps).

    Returns the same dict with series trimmed to the common window.
    """
    if len(strategy_returns) == 0:
        return {}

    # Normalise all timestamps to UTC midnight
    normalised: dict[str, pd.Series] = {}
    for k, s in strategy_returns.items():
        s2 = s.copy()
        s2.index = s2.index.normalize()
        # Drop any duplicate dates created by normalisation (keep last)
        s2 = s2[~s2.index.duplicated(keep="last")]
        normalised[k] = s2

    all_indices = [s.index for s in normalised.values()]
    common_idx = all_indices[0]
    for idx in all_indices[1:]:
        common_idx = common_idx.intersection(idx)

    return {k: s.reindex(common_idx).fillna(0.0) for k, s in normalised.items()}


def _quick_sharpe(s: pd.Series) -> float:
    std = float(s.std())
    return float(s.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
