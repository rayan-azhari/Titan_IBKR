"""B4 -- Time-series momentum strategy (MOP 2012).

Pre-registered in ``directives/Pre-Reg B4 TSMOM 2026-05-15.md``.

Inputs:
    closes_df: DataFrame indexed by daily-bar timestamp (date-only),
        columns = commodity roots (CL, GC, ZC, ...). Each cell = close
        price (yfinance M1 continuous).

Output:
    per-bar portfolio return (pd.Series, cost-adjusted).

Algorithm (per month-end rebalance t):
    1. Per asset i: trailing log-return sum over months
       [t - momentum_window, t - skip]. Canonical: window=12, skip=1.
    2. Signal: sign(cum_return). +1 (long), -1 (short), or 0.
    3. Per-asset annualised vol over `vol_lookback_days` (60 default).
    4. Inv-vol weight: w_i = sign_i × target_vol / (vol_i × N_active).
    5. Hold between monthly rebalances.

Causality (V3.6 A1 / L04):
    Signal uses returns through close[t-1]; position effective from t+1.
    Implemented as weights.shift(1) * returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR


@dataclass(frozen=True)
class TsmomConfig:
    """One row of the B4 (single-window) or B4c (window-ensemble) pre-reg grid.

    B4 form: ``momentum_window_months: int`` (e.g. ``12``) — single-window TSMOM.
    B4c form: ``momentum_window_months: tuple[int, ...]`` (e.g. ``(9, 12, 15)``) +
    ``ensemble_aggregation`` — combines multiple lookbacks into one robust
    signal per L43 mitigation.
    """

    signal_mode: Literal["sign", "raw"] = "sign"  # binary sign vs continuous magnitude
    momentum_window_months: int | tuple[int, ...] = 12
    skip_months: int = 1
    # B4c ensemble settings — only consulted when momentum_window_months
    # is a tuple of length > 1. Singleton tuple == single int (B4 parity).
    ensemble_aggregation: Literal["vote", "weighted_sum"] = "vote"

    # Position sizing.
    weighting: Literal["inv_vol", "equal"] = "inv_vol"
    target_vol_annual: float = 0.10  # 10% portfolio vol target
    vol_lookback_days: int = 60

    # Rebalance frequency.
    rebalance: Literal["monthly", "weekly"] = "monthly"

    # Costs (CME futures).
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0
    cost_fixed_usd_per_fill: float = 1.0
    notional_usd_per_leg: float = 30_000.0


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Date-only, tz-naive index."""
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _log_returns(closes_df: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; NaN-safe (returns 0 for non-trading days)."""
    return np.log(closes_df / closes_df.shift(1)).fillna(0.0)


def _month_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask of the LAST trading day of each calendar month."""
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    period = idx.to_period("M")
    last_seen: dict = {}
    for i, p in enumerate(period):
        last_seen[p] = i
    mask = np.zeros(len(idx), dtype=bool)
    for i in last_seen.values():
        mask[i] = True
    return mask


def _week_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask of the last trading day of each calendar week."""
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    period = idx.to_period("W")
    last_seen: dict = {}
    for i, p in enumerate(period):
        last_seen[p] = i
    mask = np.zeros(len(idx), dtype=bool)
    for i in last_seen.values():
        mask[i] = True
    return mask


def _normalize_windows(window_arg: int | tuple[int, ...]) -> tuple[int, ...]:
    """Coerce config.momentum_window_months to a tuple."""
    if isinstance(window_arg, int):
        return (window_arg,)
    return tuple(int(w) for w in window_arg)


def compute_tsmom_signal(
    closes_df: pd.DataFrame,
    *,
    cfg: TsmomConfig,
) -> pd.DataFrame:
    """Per-asset TSMOM signal, populated on rebalance rows.

    Returns a DataFrame indexed like closes_df.index, columns = assets.
    On non-rebalance rows: NaN. On rebalance rows:

    - Single window (B4): sign or raw cum_return.
    - Multi-window (B4c): aggregated signal per ``cfg.ensemble_aggregation``.
      * ``vote`` (default): mean of sign() over windows, in [-1, +1].
      * ``weighted_sum``: mean of sign(W) * |cum_return_W|. Windows with
        a strong signal get heavier weight; the result is on the same
        scale as a raw cum_return summed across windows.

    For ``cfg.signal_mode='raw'`` with a multi-window ensemble: each
    window's RAW cum_return is averaged (vote) or weighted by its own
    magnitude (weighted_sum then equals the mean of |cum_ret| times sign).
    """
    closes = _normalize_index(closes_df)
    idx = closes.index
    log_ret = _log_returns(closes).values  # (T, N)
    windows = _normalize_windows(cfg.momentum_window_months)
    skip_bars = cfg.skip_months * 21

    if cfg.rebalance == "monthly":
        me_mask = _month_end_mask(idx)
    elif cfg.rebalance == "weekly":
        me_mask = _week_end_mask(idx)
    else:
        raise ValueError(f"rebalance must be 'monthly' or 'weekly', got {cfg.rebalance!r}")

    if cfg.signal_mode not in ("sign", "raw"):
        raise ValueError(f"signal_mode must be 'sign' or 'raw', got {cfg.signal_mode!r}")
    if cfg.ensemble_aggregation not in ("vote", "weighted_sum"):
        raise ValueError(
            f"ensemble_aggregation must be 'vote' or 'weighted_sum', "
            f"got {cfg.ensemble_aggregation!r}"
        )

    out = pd.DataFrame(np.nan, index=idx, columns=closes.columns, dtype=float)

    for me_pos in np.where(me_mask)[0]:
        # For each window, compute cum_return over [start_W, end). The
        # window with the LARGEST W determines minimum data requirement.
        end = me_pos - skip_bars + 1  # exclusive end (skip the most recent skip_months)
        per_window_cum: list[np.ndarray] = []
        skipped = False
        for w_months in windows:
            momentum_bars = w_months * 21
            start = end - momentum_bars + skip_bars
            if start < 0:
                skipped = True
                break
            per_window_cum.append(log_ret[start:end].sum(axis=0))
        if skipped:
            continue

        stack = np.stack(per_window_cum, axis=0)  # (W, N)

        if cfg.ensemble_aggregation == "vote":
            if cfg.signal_mode == "sign":
                # Mean of signs over windows.
                aggregated = np.mean(np.sign(stack), axis=0)
            else:  # raw
                aggregated = np.mean(stack, axis=0)
        else:  # weighted_sum
            if cfg.signal_mode == "sign":
                # sign(W) weighted by |cum_ret_W|, then averaged across W.
                weighted = np.sign(stack) * np.abs(stack)
                aggregated = np.mean(weighted, axis=0)
            else:  # raw
                weighted = stack * np.abs(stack)
                aggregated = np.mean(weighted, axis=0)
        out.iloc[me_pos] = aggregated

    return out


def _realised_vol(closes_df: pd.DataFrame, *, lookback_days: int = 60) -> pd.DataFrame:
    """Per-asset rolling annualised vol of daily log returns."""
    log_ret = _log_returns(closes_df)
    return log_ret.rolling(lookback_days, min_periods=lookback_days).std(ddof=1) * np.sqrt(
        BARS_PER_YEAR["D"]
    )


def build_portfolio_weights(
    signal_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    *,
    cfg: TsmomConfig,
) -> pd.DataFrame:
    """Convert per-asset signals + vol estimates into portfolio weights.

    Returns a DataFrame indexed like signal_df, columns same. Weights
    are held constant between rebalance rows (rows where signal is
    non-all-NaN).
    """
    idx = signal_df.index
    cols = list(signal_df.columns)
    weights = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)
    current = pd.Series(0.0, index=cols, dtype=float)

    for i in range(len(idx)):
        row = signal_df.iloc[i]
        if row.notna().any():
            # Rebalance day.
            valid = row.dropna()
            if cfg.signal_mode == "sign":
                signs = np.sign(valid.values)
            else:
                # raw mode: use sign of the cumulative return as the
                # discrete direction, scaled by relative magnitude.
                signs = np.sign(valid.values)
            n_active = int(np.sum(signs != 0))
            if n_active == 0:
                weights.iloc[i] = current.values
                continue
            vol_row = vol_df.iloc[i] if i < len(vol_df) else None
            new_w = pd.Series(0.0, index=cols, dtype=float)
            if cfg.weighting == "inv_vol":
                if vol_row is None:
                    weights.iloc[i] = current.values
                    continue
                for ticker, sign in zip(valid.index, signs, strict=False):
                    if sign == 0:
                        continue
                    v = vol_row.get(ticker, np.nan)
                    if not np.isfinite(v) or v <= 0:
                        continue
                    new_w[ticker] = float(sign) * cfg.target_vol_annual / (v * n_active)
            else:  # equal weight
                per_leg = 1.0 / n_active
                for ticker, sign in zip(valid.index, signs, strict=False):
                    if sign == 0:
                        continue
                    new_w[ticker] = float(sign) * per_leg
            current = new_w
        weights.iloc[i] = current.values

    return weights


def tsmom_returns(
    closes_df: pd.DataFrame,
    *,
    cfg: TsmomConfig | None = None,
) -> pd.Series:
    """Per-bar cost-adjusted TSMOM portfolio returns."""
    if cfg is None:
        cfg = TsmomConfig()
    closes = _normalize_index(closes_df)
    signal = compute_tsmom_signal(closes, cfg=cfg)
    vol = _realised_vol(closes, lookback_days=cfg.vol_lookback_days)
    weights = build_portfolio_weights(signal, vol, cfg=cfg)
    log_ret = _log_returns(closes)

    # Causal: weights at close[t] reflect signal from data <=t-1; earn r[t+1].
    held_lagged = weights.shift(1).fillna(0.0)
    gross = (held_lagged * log_ret).sum(axis=1)

    if cfg.apply_costs:
        dw = weights.diff().abs().fillna(0.0)
        n_fills_per_bar = (dw > 1e-9).sum(axis=1).astype(float)
        bps_drag = (dw.sum(axis=1) * cfg.cost_bps_per_turnover) / 10_000.0
        fixed_drag = (
            n_fills_per_bar * cfg.cost_fixed_usd_per_fill / max(cfg.notional_usd_per_leg, 1.0)
        )
        net = gross - bps_drag - fixed_drag
    else:
        net = gross

    return net.rename("tsmom_returns")


def tsmom_assert_causal(
    closes_df: pd.DataFrame,
    *,
    cfg: TsmomConfig | None = None,
    n_trials: int = 3,
    seed: int = 42,
) -> None:
    """A10 causality smoke test.

    Corrupt future closes at random t; assert per-bar strategy returns
    at t' < t are bit-exact unchanged.
    """
    if cfg is None:
        cfg = TsmomConfig()
    n = len(closes_df)
    max_window = max(_normalize_windows(cfg.momentum_window_months))
    min_bars = max_window * 21 + 252
    if n < min_bars:
        return
    base = tsmom_returns(closes_df, cfg=cfg)
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        t_corrupt = int(rng.integers(min_bars, n - 5))
        corrupt = closes_df.copy()
        corrupt.iloc[t_corrupt:] = corrupt.iloc[t_corrupt:] * 1.5
        altered = tsmom_returns(corrupt, cfg=cfg)
        past_base = base.iloc[:t_corrupt]
        past_corr = altered.iloc[:t_corrupt]
        if not past_base.equals(past_corr):
            diff = past_base != past_corr
            raise AssertionError(
                f"tsmom_assert_causal: future corruption at t={t_corrupt} "
                f"changed {int(diff.sum())} past returns"
            )
