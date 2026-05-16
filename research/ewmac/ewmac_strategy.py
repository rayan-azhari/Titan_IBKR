"""B2 -- Carver EWMAC trend ensemble.

Pre-registered in ``directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md``.

Mechanism (per asset i, per bar t):

1. For each speed (fast_hl, slow_hl):
       ewmac_s(t) = EWMA(close, halflife=fast_hl) - EWMA(close, halflife=slow_hl)
2. Vol-normalise to make scale-invariant:
       norm_s(t) = ewmac_s(t) / stdev(diff(close), window=vol_lookback)
3. Multiply by Carver forecast scalar so |forecast| ~ 10 on average:
       scaled_s(t) = norm_s(t) * forecast_scalar_s
4. Clip to forecast_cap (default ±20 — Carver's "double cap"):
       capped_s(t) = clip(scaled_s(t), -cap, +cap)
5. Combine across speeds with diversification multiplier:
       combined(t) = clip(FDM * mean(capped_s(t)), -cap, +cap)
6. Position-sizing: position = combined(t) * target_vol /
       (instrument_vol * 10). Forecast = 10 -> full target-vol allocation.

Causality (L04 / A1): EWMAs use only past data through t-1; the position
effective at t earns return from t -> t+1. Implemented as
``position.shift(1) * log_return``.

Pre-reg note: Carver's published forecast scalars (Systematic Trading, p.286)
are used for the standard speeds (16/64, 32/128, 64/256). For non-standard
speeds the scalar defaults to ``1.0``; the audit harness can compute
IS-frozen scalars per pre-reg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR

# Carver Systematic Trading (2015), Table 31, p.286 — forecast scalars
# normalising the vol-divided EWMAC so the long-run mean abs forecast
# equals ~10. Keys are (fast_halflife, slow_halflife) day pairs.
CARVER_FORECAST_SCALARS: dict[tuple[int, int], float] = {
    (2, 8): 12.1,
    (4, 16): 8.53,
    (8, 32): 5.95,
    (16, 64): 4.10,
    (32, 128): 2.79,
    (64, 256): 1.91,
    (128, 512): 1.32,
}

# Forecast diversification multiplier — Carver Table 21, p.187. Compensates
# for the volatility reduction when averaging N partially-correlated
# forecasts. Keys = N speeds.
CARVER_FDM: dict[int, float] = {
    1: 1.0,
    2: 1.20,
    3: 1.35,
    4: 1.43,
    5: 1.47,
    6: 1.51,
    7: 1.53,
    8: 1.54,
}


@dataclass(frozen=True)
class BroadTrendFilterConfig:
    """B2c trend-of-trend regime filter.

    When set on an EwmacConfig, every per-asset combined forecast is
    multiplied by a gate derived from an equal-weight cum-log-return
    universe index. The gate is +1 / 0 / -1 depending on mode:

        - ``absolute_trend``: gate = +1 when ``|broad_ewmac| > deadband``,
          else 0. Trades enabled in either trending direction; flat in
          quiescent regimes. Per-asset signs unchanged.
        - ``directional``: gate = sign(broad_ewmac) when ``|broad_ewmac| >
          deadband``, else 0. Negative broad-trend FLIPS per-asset signal
          signs (e.g., long signals become short in a bear regime).

    ``deadband`` is in the same units as the vol-normalised forecast
    (i.e., the Carver scale where ~10 = average absolute forecast).
    """

    fast_hl: int = 64
    slow_hl: int = 256
    mode: Literal["absolute_trend", "directional"] = "absolute_trend"
    deadband: float = 0.0


@dataclass(frozen=True)
class VolRegimeFilterConfig:
    """B2d realised-vol regime filter.

    Activates trades only when the broad-universe realised volatility is
    within a "moderate" percentile band. Cleanly separates trend-friendly
    regimes (moderate vol) from quiescent regimes (low vol → no signal)
    AND crisis regimes (extreme vol → everything-correlates).

    Mechanism:
        1. Compute equal-weight cum-log-return broad-universe series.
        2. Compute its rolling realised-vol over ``vol_lookback_days``.
        3. Compute rolling percentile of the vol over
           ``percentile_window_days`` (longer-window historical context).
        4. Gate = 1 when ``pct_lo <= current_pct <= pct_hi``, else 0.

    Causality: vol uses past returns; percentile is rolling backward.
    """

    vol_lookback_days: int = 60
    percentile_window_days: int = 252
    pct_lo: float = 20.0  # inclusive lower bound (0-100)
    pct_hi: float = 80.0  # inclusive upper bound


@dataclass(frozen=True)
class EwmacConfig:
    """One row of the B2 pre-reg grid."""

    # Speeds as tuple of (fast_hl, slow_hl) day pairs.
    speeds: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))

    # Forecast cap (Carver's double cap = 20.0).
    forecast_cap: float = 20.0

    # Forecast diversification multiplier (FDM). Default None -> look up
    # from CARVER_FDM by N. Setting explicitly allows pre-reg override.
    fdm: float | None = None

    # Vol-normalisation lookback for the daily-change stdev.
    vol_lookback_days: int = 20

    # Per-asset realised-vol lookback for position sizing.
    instrument_vol_lookback_days: int = 60

    # Position-sizing target.
    target_vol_annual: float = 0.10  # 10% portfolio vol
    target_forecast: float = 10.0  # canonical Carver normalisation

    # Rebalance frequency. Carver default = daily-mark, monthly-target.
    # We follow the TSMOM monthly rebalance for cost parity.
    rebalance: Literal["daily", "weekly", "monthly"] = "monthly"

    # Costs (CME futures; matches B4 model).
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0
    cost_fixed_usd_per_fill: float = 1.0
    notional_usd_per_leg: float = 30_000.0

    # Forecast scalar source. "carver" looks up CARVER_FORECAST_SCALARS;
    # "is_frozen" computes from the visible/IS window and freezes them;
    # "unit" uses 1.0 (raw vol-normalised EWMAC, no rescaling).
    forecast_scalar_mode: Literal["carver", "is_frozen", "unit"] = "carver"

    # Optional pre-frozen overrides: dict mapping speed pair to scalar.
    forecast_scalars: dict[tuple[int, int], float] | None = field(default=None)

    # B2c trend-of-trend regime filter (None = disabled = B2 baseline).
    broad_trend_filter: BroadTrendFilterConfig | None = None

    # B2d realised-vol regime filter (None = disabled). Mutually exclusive
    # with broad_trend_filter; if both set, vol_regime_filter is applied
    # AFTER broad_trend_filter (i.e., both gates AND together).
    vol_regime_filter: VolRegimeFilterConfig | None = None

    # I1 per-asset HMM regime gate (None = disabled). When set, the audit
    # harness must supply ``is_end_idx`` to ``compute_ewmac_forecast`` via
    # the ``is_end_idx`` kwarg so the HMM training stays IS-only. Cannot
    # be co-applied with broad gates in this implementation (would require
    # a per-asset broad-vs-narrow combination decision out of scope).
    per_asset_regime_gate: "object | None" = None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _ewma(series: pd.Series, halflife: int) -> pd.Series:
    return series.ewm(halflife=halflife, adjust=False).mean()


def _forecast_scalar(
    speed: tuple[int, int],
    cfg: EwmacConfig,
    is_close: pd.Series | None = None,
) -> float:
    """Return the forecast scalar for one speed, honouring cfg.forecast_scalar_mode.

    For ``"is_frozen"``: requires ``is_close`` (the in-sample series). The
    raw vol-normalised EWMAC over the IS window is computed, its abs mean
    is taken, and the scalar = target_forecast / abs_mean. This is V3.1
    honest — frozen on IS, applied to all of visible.
    """
    if cfg.forecast_scalars and speed in cfg.forecast_scalars:
        return float(cfg.forecast_scalars[speed])
    if cfg.forecast_scalar_mode == "carver":
        return float(CARVER_FORECAST_SCALARS.get(speed, 1.0))
    if cfg.forecast_scalar_mode == "unit":
        return 1.0
    if cfg.forecast_scalar_mode == "is_frozen":
        if is_close is None or is_close.empty:
            return float(CARVER_FORECAST_SCALARS.get(speed, 1.0))
        fast_hl, slow_hl = speed
        norm = _vol_normalised_ewmac(is_close, fast_hl, slow_hl, vol_lookback=cfg.vol_lookback_days)
        norm_clean = norm.replace([np.inf, -np.inf], np.nan).dropna()
        if len(norm_clean) == 0:
            return 1.0
        abs_mean = float(norm_clean.abs().mean())
        if abs_mean <= 0:
            return 1.0
        return float(cfg.target_forecast / abs_mean)
    raise ValueError(f"Unknown forecast_scalar_mode: {cfg.forecast_scalar_mode!r}")


def _fdm_for(cfg: EwmacConfig) -> float:
    if cfg.fdm is not None:
        return float(cfg.fdm)
    return float(CARVER_FDM.get(len(cfg.speeds), CARVER_FDM[max(CARVER_FDM)]))


def _vol_normalised_ewmac(
    close: pd.Series, fast_hl: int, slow_hl: int, *, vol_lookback: int
) -> pd.Series:
    """Carver vol-normalised EWMAC for a single asset and single speed."""
    fast = _ewma(close, halflife=fast_hl)
    slow = _ewma(close, halflife=slow_hl)
    ewmac = fast - slow
    # Normalise by stdev of daily price CHANGE (not return) — Carver's
    # convention. Floored at small epsilon to avoid division blow-ups
    # on very-stable instruments.
    price_diff = close.diff()
    vol = price_diff.rolling(vol_lookback, min_periods=vol_lookback).std(ddof=1)
    return ewmac / vol.replace(0, np.nan)


def _compute_vol_regime_gate(
    closes_df: pd.DataFrame,
    *,
    filter_cfg: VolRegimeFilterConfig,
) -> pd.Series:
    """B2d realised-vol regime gate.

    Equal-weight broad universe cum-log-return → rolling realised vol →
    rolling percentile-of-vol over a longer window → gate = 1 when within
    [pct_lo, pct_hi] band, else 0. Causal (uses only data through t).
    """
    if closes_df.empty:
        return pd.Series(dtype=float, index=closes_df.index, name="vol_regime_gate")
    log_ret = np.log(closes_df / closes_df.shift(1))
    mean_log_ret = log_ret.mean(axis=1, skipna=True).fillna(0.0)
    # Annualised realised vol of the broad-index daily returns.
    broad_vol = mean_log_ret.rolling(
        filter_cfg.vol_lookback_days, min_periods=filter_cfg.vol_lookback_days
    ).std(ddof=1) * np.sqrt(BARS_PER_YEAR["D"])
    # Rolling percentile of broad_vol over the percentile window.
    # rank(pct=True) returns 0..1; multiply by 100 to align with pct_lo/hi.
    pct = (
        broad_vol.rolling(
            filter_cfg.percentile_window_days, min_periods=filter_cfg.percentile_window_days
        ).rank(pct=True)
        * 100.0
    )
    in_band = (pct >= filter_cfg.pct_lo) & (pct <= filter_cfg.pct_hi)
    gate = in_band.astype(float)
    gate.name = "vol_regime_gate"
    return gate


def _compute_broad_trend_gate(
    closes_df: pd.DataFrame,
    *,
    filter_cfg: BroadTrendFilterConfig,
) -> pd.Series:
    """B2c trend-of-trend gate.

    Builds an equal-weight cum-log-return universe index that is
    scale-invariant (FX rates of ~1 mix safely with equity ETFs of ~100s),
    runs the configured EWMAC on the LOG of that index, and returns a
    per-bar gate signal in {-1, 0, +1} per the configured mode.

    Causality: only daily log-returns are used (each return is computed
    from close[t]/close[t-1]); the EWMA itself is causal by definition.
    The gate Series returned has the same index as ``closes_df``.
    """
    if closes_df.empty:
        return pd.Series(dtype=float, index=closes_df.index, name="broad_trend_gate")
    log_ret = np.log(closes_df / closes_df.shift(1))
    # Equal-weight cum-log-return mean across columns at each date. NaN-safe
    # (instruments with fewer leading bars don't contribute until they have data).
    mean_log_ret = log_ret.mean(axis=1, skipna=True)
    cum_log = mean_log_ret.fillna(0.0).cumsum()
    # Run Carver vol-normalised EWMAC on this constructed broad-index log.
    norm = _vol_normalised_ewmac(
        cum_log,
        filter_cfg.fast_hl,
        filter_cfg.slow_hl,
        vol_lookback=20,  # standard 20-day lookback for the broad index
    )
    norm = norm.fillna(0.0)
    if filter_cfg.mode == "absolute_trend":
        gate = (norm.abs() > filter_cfg.deadband).astype(float)
    elif filter_cfg.mode == "directional":
        sign = np.sign(norm)
        gate = sign.where(norm.abs() > filter_cfg.deadband, 0.0)
    else:
        raise ValueError(
            f"BroadTrendFilterConfig.mode must be 'absolute_trend' or 'directional', got {filter_cfg.mode!r}"
        )
    gate.name = "broad_trend_gate"
    return gate


def compute_ewmac_forecast(
    closes_df: pd.DataFrame,
    *,
    cfg: EwmacConfig,
    is_closes_df: pd.DataFrame | None = None,
    is_end_idx: int | None = None,
) -> pd.DataFrame:
    """Per-asset combined EWMAC forecast (causal, scaled, capped, FDM-blended,
    optionally broad-trend-gated).

    Returns a DataFrame indexed like closes_df, columns = assets.
    Cell [t, i] = combined forecast in [-cap, +cap]; canonical full-position
    long = +10, max = +20. When ``cfg.broad_trend_filter`` is set, each
    per-asset forecast is multiplied by the broad-trend gate at the same
    date (B2c regime-filter mechanism).
    """
    closes = _normalize_index(closes_df)
    is_closes = _normalize_index(is_closes_df) if is_closes_df is not None else None

    fdm = _fdm_for(cfg)
    combined = pd.DataFrame(0.0, index=closes.index, columns=closes.columns, dtype=float)

    for col in closes.columns:
        series = closes[col]
        is_series = is_closes[col] if is_closes is not None and col in is_closes.columns else None
        # Per-speed forecast for this asset.
        scaled_stack: list[pd.Series] = []
        for speed in cfg.speeds:
            scalar = _forecast_scalar(speed, cfg, is_close=is_series)
            fast_hl, slow_hl = speed
            norm = _vol_normalised_ewmac(
                series, fast_hl, slow_hl, vol_lookback=cfg.vol_lookback_days
            )
            scaled = (norm * scalar).clip(lower=-cfg.forecast_cap, upper=cfg.forecast_cap)
            scaled_stack.append(scaled)
        if not scaled_stack:
            continue
        per_speed = pd.concat(scaled_stack, axis=1)
        per_speed.columns = list(range(len(scaled_stack)))
        mean_scaled = per_speed.mean(axis=1)
        capped = (fdm * mean_scaled).clip(lower=-cfg.forecast_cap, upper=cfg.forecast_cap)
        combined[col] = capped

    # B2c broad-trend gate (multiplicative on every column).
    if cfg.broad_trend_filter is not None:
        gate = _compute_broad_trend_gate(closes, filter_cfg=cfg.broad_trend_filter)
        # Multiply each column by the gate; broadcast aligns on index.
        combined = combined.mul(gate, axis=0).fillna(0.0)

    # B2d vol-regime gate (AND-combined with broad-trend gate if both set).
    if cfg.vol_regime_filter is not None:
        vol_gate = _compute_vol_regime_gate(closes, filter_cfg=cfg.vol_regime_filter)
        combined = combined.mul(vol_gate, axis=0).fillna(0.0)

    # I1 per-asset HMM regime gate. Imported lazily so EwmacConfig can
    # type-hint without forcing hmmlearn at import-time.
    if cfg.per_asset_regime_gate is not None:
        if is_end_idx is None:
            raise ValueError(
                "EwmacConfig.per_asset_regime_gate is set but is_end_idx not provided "
                "to compute_ewmac_forecast(). HMM training requires explicit IS boundary."
            )
        from research.regime.hmm_gate import compute_per_asset_regime_gate

        per_asset_gate = compute_per_asset_regime_gate(
            closes,
            cfg=cfg.per_asset_regime_gate,
            is_end_idx=is_end_idx,
        )
        combined = (combined * per_asset_gate).fillna(0.0)

    return combined


def _instrument_vol(closes_df: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
    """Per-asset rolling annualised vol of daily log returns."""
    log_ret = np.log(closes_df / closes_df.shift(1)).fillna(0.0)
    return log_ret.rolling(lookback_days, min_periods=lookback_days).std(ddof=1) * np.sqrt(
        BARS_PER_YEAR["D"]
    )


def _rebalance_mask(idx: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    if rebalance == "daily":
        return np.ones(len(idx), dtype=bool)
    if rebalance in ("monthly", "weekly"):
        period_code = "M" if rebalance == "monthly" else "W"
        period = idx.to_period(period_code)
        last_seen: dict = {}
        for i, p in enumerate(period):
            last_seen[p] = i
        mask = np.zeros(len(idx), dtype=bool)
        for i in last_seen.values():
            mask[i] = True
        return mask
    raise ValueError(f"rebalance must be 'daily', 'weekly', or 'monthly', got {rebalance!r}")


def build_positions(
    forecast_df: pd.DataFrame,
    instrument_vol_df: pd.DataFrame,
    *,
    cfg: EwmacConfig,
) -> pd.DataFrame:
    """Convert combined forecast into per-asset positions (target-vol scaled).

    Rebalances on cfg.rebalance dates only — between rebalances, position
    is held constant (no daily-mark intra-month adjustment, matching B4
    cost parity).
    """
    idx = forecast_df.index
    cols = list(forecast_df.columns)
    positions = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)
    mask = _rebalance_mask(idx, cfg.rebalance)
    current = pd.Series(0.0, index=cols, dtype=float)

    n_assets = len(cols)
    for i in range(len(idx)):
        if mask[i]:
            forecast_row = forecast_df.iloc[i]
            vol_row = instrument_vol_df.iloc[i] if i < len(instrument_vol_df) else None
            new_pos = pd.Series(0.0, index=cols, dtype=float)
            for asset in cols:
                fcst = forecast_row[asset]
                if not np.isfinite(fcst):
                    continue
                if vol_row is None:
                    continue
                vol = vol_row.get(asset, np.nan)
                if not np.isfinite(vol) or vol <= 0:
                    continue
                # Carver's position formula:
                # position = forecast / target_forecast * target_vol_per_asset / instrument_vol
                # The /n_assets split portfolio target equally across assets.
                target_vol_per_asset = cfg.target_vol_annual / max(n_assets, 1)
                new_pos[asset] = (fcst / cfg.target_forecast) * (target_vol_per_asset / vol)
            current = new_pos
        positions.iloc[i] = current.values

    return positions


def ewmac_returns(
    closes_df: pd.DataFrame,
    *,
    cfg: EwmacConfig | None = None,
    is_closes_df: pd.DataFrame | None = None,
    is_end_idx: int | None = None,
) -> pd.Series:
    """Per-bar cost-adjusted EWMAC portfolio returns.

    ``is_end_idx`` (number of leading rows that constitute the IS window)
    is required when ``cfg.per_asset_regime_gate`` is set so the HMM fit
    stays IS-only. For non-HMM cells the kwarg is ignored.
    """
    if cfg is None:
        cfg = EwmacConfig()
    closes = _normalize_index(closes_df)
    forecast = compute_ewmac_forecast(
        closes, cfg=cfg, is_closes_df=is_closes_df, is_end_idx=is_end_idx
    )
    instrument_vol = _instrument_vol(closes, lookback_days=cfg.instrument_vol_lookback_days)
    positions = build_positions(forecast, instrument_vol, cfg=cfg)
    log_ret = np.log(closes / closes.shift(1)).fillna(0.0)

    # Causal: positions at close[t] reflect data <=t-1; earn return at t+1.
    held_lagged = positions.shift(1).fillna(0.0)
    gross = (held_lagged * log_ret).sum(axis=1)

    if cfg.apply_costs:
        dpos = positions.diff().abs().fillna(0.0)
        n_fills_per_bar = (dpos > 1e-9).sum(axis=1).astype(float)
        bps_drag = (dpos.sum(axis=1) * cfg.cost_bps_per_turnover) / 10_000.0
        fixed_drag = (
            n_fills_per_bar * cfg.cost_fixed_usd_per_fill / max(cfg.notional_usd_per_leg, 1.0)
        )
        net = gross - bps_drag - fixed_drag
    else:
        net = gross

    return net.rename("ewmac_returns")


def ewmac_assert_causal(
    closes_df: pd.DataFrame,
    *,
    cfg: EwmacConfig | None = None,
    n_trials: int = 3,
    seed: int = 42,
) -> None:
    """A10 causality smoke test (date-aligned per L44-test pattern)."""
    if cfg is None:
        cfg = EwmacConfig()
    base = ewmac_returns(closes_df, cfg=cfg)
    if len(base) < 200:
        return
    return_dates = base.index
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        ret_pos = int(rng.integers(50, len(return_dates) - 5))
        t_corrupt_date = return_dates[ret_pos]
        corrupt = closes_df.copy()
        mask = corrupt.index >= t_corrupt_date
        corrupt.loc[mask] = corrupt.loc[mask] * 1.5
        altered = ewmac_returns(corrupt, cfg=cfg)
        past_base = base.loc[base.index < t_corrupt_date]
        past_corr = altered.loc[altered.index < t_corrupt_date]
        if not past_base.equals(past_corr):
            diff = past_base != past_corr
            raise AssertionError(
                f"ewmac_assert_causal: future corruption at "
                f"date={t_corrupt_date.date()} changed {int(diff.sum())} past returns"
            )
