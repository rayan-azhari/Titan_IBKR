"""A1 -- Residual momentum strategy.

Pre-registered in ``directives/Pre-Reg A1 Residual Momentum 2026-05-15.md``.

Inputs:
    stocks_close: DataFrame indexed by daily-bar timestamp (date-only,
        tz-naive), columns = ticker symbols. Each cell = adjusted close
        for that stock on that day. Yfinance adj-close (total return).
    ff3_df: DataFrame indexed by same daily-bar timestamps with columns
        ``["mkt_rf", "smb", "hml", "rf"]`` (decimal daily returns).

Output:
    per-bar long-short portfolio returns (pd.Series, cost-adjusted,
    indexed like stocks_close.index after warmup).

Algorithm (per rebalance month-end t):
    1. For each stock i with valid history:
       a. Compute daily excess returns r_excess_i,τ = r_i,τ - r_rf,τ over
          the trailing `regression_months` months (default 36).
       b. Regress r_excess_i,τ on [1, mkt_rf, smb, hml] over the same window.
       c. Take residuals ε_i,τ for τ in the window.
       d. Sum residuals over months [t-window, t-skip-1] (default
          [t-12, t-2]) -> skip-1 cumulative residual return.
       e. Compute residual std over the FULL regression window (monthly).
       f. Signal_i = cumulative_residual / residual_std (t-stat style).
    2. Cross-sectional rank: long top quintile, short bottom quintile,
       equal-weighted, dollar-neutral.
    3. Hold the position until the next month-end rebalance.

Causality (V3.6 A1 / L04):
    Position at bar t reflects the signal computed at month-end t' < t
    using data through close[t'-1]. Implemented as: build weights on
    month-end rows from rolling regressions ending at THAT date; hold
    constant until next rebalance; per-bar return = w[t-1] * r[t].

Cost model (L23):
    US large-cap equity: 0.5 bps/turnover (spread+slip) + $0.5/fill
    (IBKR Pro). Notional per leg = $30k. Monthly rebalance with ~200
    names produces ~100-150 fills/month → ~200 bps/yr drag baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualConfig:
    """One row of the A1 pre-reg grid."""

    # Signal mode.
    signal_mode: Literal["residual", "raw"] = "residual"
    # Window (months) used for the cumulative momentum sum.
    momentum_window_months: int = 12
    # Skip the most recent N months (BHM canonical = 1, avoids short-term
    # reversal in week-1).
    skip_months: int = 1
    # Regression-window length (months) for residual estimation.
    regression_months: int = 36

    # Portfolio construction.
    breadth_pct: float = 0.20  # quintile = 0.20; tercile = 0.333
    long_only: bool = False
    weighting: Literal["equal", "inverse_vol"] = "equal"
    inverse_vol_lookback_days: int = 60  # only used when weighting="inverse_vol"

    # Costs.
    apply_costs: bool = True
    cost_bps_per_turnover: float = 0.5
    cost_fixed_usd_per_fill: float = 0.5
    notional_usd_per_leg: float = 30_000.0


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


def _stocks_log_returns(stocks_close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, NaN-safe (returns 0.0 where prior price missing)."""
    return np.log(stocks_close / stocks_close.shift(1)).fillna(0.0)


def _ff3_align(ff3_df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Align FF3 factors to the stock index. Missing days forward-filled
    up to 3 bars (handles FF release lag); longer gaps remain NaN.
    """
    ff = ff3_df.copy()
    ff.index = pd.to_datetime(ff.index).tz_localize(None).normalize()
    target = pd.to_datetime(idx).tz_localize(None).normalize()
    return ff.reindex(target).ffill(limit=3)


def _compute_residuals_for_window(
    stock_excess: np.ndarray,
    factors: np.ndarray,
) -> tuple[np.ndarray, float]:
    """OLS-regress a single stock's excess returns on factors over the window.

    Args:
        stock_excess: 1-D array length T (daily excess returns).
        factors: 2-D array shape (T, K) — first column is 1s (intercept).

    Returns:
        (residuals, residual_std) — residuals shape (T,), std is float
        on monthly-aggregated residuals.
    """
    # Drop NaN rows.
    valid = np.isfinite(stock_excess) & np.all(np.isfinite(factors), axis=1)
    if valid.sum() < factors.shape[1] + 2:
        # Need at least K+2 obs to fit + estimate residual variance.
        return np.full_like(stock_excess, np.nan), np.nan
    se = stock_excess[valid]
    f = factors[valid]
    # OLS: beta = (X'X)^-1 X'y
    try:
        beta, *_ = np.linalg.lstsq(f, se, rcond=None)
    except np.linalg.LinAlgError:
        return np.full_like(stock_excess, np.nan), np.nan
    pred = f @ beta
    resid_valid = se - pred
    # Put residuals back in the full-length array.
    resid = np.full_like(stock_excess, np.nan)
    resid[valid] = resid_valid
    # Residual std on the in-window residuals.
    resid_std = float(np.std(resid_valid, ddof=1)) if resid_valid.size > 1 else np.nan
    return resid, resid_std


def compute_residual_signal(
    stocks_close: pd.DataFrame,
    ff3_df: pd.DataFrame,
    *,
    cfg: ResidualConfig,
) -> pd.DataFrame:
    """Compute the cross-sectional residual-momentum signal per stock.

    Returns a DataFrame indexed like stocks_close.index, columns = tickers.
    Cell [t, i] holds the standardised cumulative residual return through
    t-1, ONLY POPULATED ON MONTH-END ROWS. Non-rebalance rows are NaN.

    For cfg.signal_mode='raw', skips the regression step and returns
    cumulative LOG-RETURN momentum (BHM C2 control cell).

    Implementation: vectorised OLS. For each rebalance window, we solve
    one (T, K) -> (T, N) least-squares problem (N = number of stocks)
    rather than N separate lstsq calls. The (X'X)^-1 X' projection is
    identical across stocks; we factor it once per window.
    """
    idx = stocks_close.index
    if cfg.signal_mode not in ("residual", "raw"):
        raise ValueError(f"signal_mode must be 'residual' or 'raw', got {cfg.signal_mode!r}")

    log_returns = _stocks_log_returns(stocks_close)
    me_mask = _month_end_mask(idx)

    if cfg.signal_mode == "raw":
        return _compute_raw_signal(log_returns, idx, cfg)

    # ── Residual signal (vectorised) ──────────────────────────────────────
    ff = _ff3_align(ff3_df, idx)
    factors_all = np.column_stack(
        [
            np.ones(len(idx)),
            ff["mkt_rf"].values,
            ff["smb"].values,
            ff["hml"].values,
        ]
    )
    rf_all = ff["rf"].values  # shape (T,)

    regression_bars = cfg.regression_months * 21
    momentum_bars = cfg.momentum_window_months * 21
    skip_bars = cfg.skip_months * 21

    log_ret_arr = log_returns.values  # shape (T, N)
    tickers = list(stocks_close.columns)
    out = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
    cum_start_rel = regression_bars - momentum_bars
    cum_end_rel = regression_bars - skip_bars

    for me_pos in np.where(me_mask)[0]:
        start = me_pos - regression_bars + 1
        if start < 0:
            continue
        end = me_pos + 1
        if cum_start_rel < 0 or cum_end_rel <= cum_start_rel:
            continue
        # Window slices.
        win_factors = factors_all[start:end]  # (W, K)
        win_rf = rf_all[start:end]  # (W,)
        win_log_ret = log_ret_arr[start:end]  # (W, N)
        # Excess returns (W, N) = log returns minus risk-free (broadcasted).
        win_excess = win_log_ret - win_rf[:, None]
        # Mask rows where ANY factor is NaN; mask stocks where ANY bar is NaN.
        factor_valid_mask = np.all(np.isfinite(win_factors), axis=1)  # (W,)
        if factor_valid_mask.sum() < win_factors.shape[1] + 2:
            continue
        f_valid = win_factors[factor_valid_mask]
        # We'll loop over per-stock NaN masks because they can differ; but
        # we batch the projection so each stock costs an O(K*W) matvec.
        # Precompute the projection matrix once: (X'X)^-1 X' is shape (K, W).
        try:
            xtx_inv = np.linalg.inv(f_valid.T @ f_valid)
        except np.linalg.LinAlgError:
            continue
        proj = xtx_inv @ f_valid.T  # (K, W')
        for j in range(win_excess.shape[1]):
            stock_excess = win_excess[:, j]
            row_valid = factor_valid_mask & np.isfinite(stock_excess)
            if row_valid.sum() < f_valid.shape[1] + 2:
                continue
            # If the per-stock valid mask differs from factor_valid_mask,
            # fall back to a per-stock lstsq for that case (rare).
            if not np.array_equal(row_valid, factor_valid_mask):
                stock_valid = stock_excess[row_valid]
                f_stock = win_factors[row_valid]
                try:
                    beta = np.linalg.lstsq(f_stock, stock_valid, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                resid_valid = stock_valid - f_stock @ beta
            else:
                stock_valid = stock_excess[factor_valid_mask]
                beta = proj @ stock_valid  # (K,)
                resid_valid = stock_valid - f_valid @ beta
            if resid_valid.size < 2:
                continue
            resid_std = float(resid_valid.std(ddof=1))
            if not np.isfinite(resid_std) or resid_std == 0:
                continue
            # Cumulative residual over the momentum window. We need to
            # locate the [cum_start_rel, cum_end_rel) slice in
            # window-position coordinates. Because residuals were
            # computed only on valid rows, recompute index mapping.
            valid_positions = np.where(
                row_valid if not np.array_equal(row_valid, factor_valid_mask) else factor_valid_mask
            )[0]
            cum_mask = (valid_positions >= cum_start_rel) & (valid_positions < cum_end_rel)
            if cum_mask.sum() == 0:
                continue
            cum_resid = float(resid_valid[cum_mask].sum())
            n_window_bars = cum_end_rel - cum_start_rel
            out.iloc[me_pos, j] = cum_resid / (resid_std * np.sqrt(n_window_bars))

    return out


def _compute_raw_signal(
    log_returns: pd.DataFrame,
    idx: pd.DatetimeIndex,
    cfg: ResidualConfig,
) -> pd.DataFrame:
    """Raw cumulative skip-1 12-month log return (BHM C2 control)."""
    momentum_bars = cfg.momentum_window_months * 21
    skip_bars = cfg.skip_months * 21
    me_mask = _month_end_mask(idx)
    out = pd.DataFrame(np.nan, index=idx, columns=log_returns.columns, dtype=float)
    for me_pos in np.where(me_mask)[0]:
        end = me_pos - skip_bars
        start = end - momentum_bars + skip_bars  # = me_pos - momentum_bars
        if start < 0:
            continue
        cum = log_returns.iloc[start:end].sum(axis=0)
        out.iloc[me_pos] = cum.values
    return out


def build_portfolio_weights(
    signal_df: pd.DataFrame,
    *,
    cfg: ResidualConfig,
    realised_vol_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional L/S weights from the signal.

    On each month-end (rows where signal is non-all-NaN), rank stocks by
    signal: long top breadth_pct, short bottom breadth_pct. Equal-weight
    within each leg; long-leg sum = +0.5, short-leg = -0.5 (or +1.0 if
    long_only). Between month-ends weights are HELD CONSTANT.
    """
    if cfg.weighting == "inverse_vol" and realised_vol_df is None:
        raise ValueError("CarryConfig.weighting='inverse_vol' requires realised_vol_df")

    idx = signal_df.index
    cols = list(signal_df.columns)
    weights = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)
    current = pd.Series(0.0, index=cols, dtype=float)

    for i in range(len(idx)):
        row = signal_df.iloc[i]
        if row.notna().any():
            # Rebalance day.
            valid = row.dropna()
            if len(valid) >= 5:
                n = len(valid)
                top_n = max(1, int(round(n * cfg.breadth_pct)))
                sorted_assets = valid.sort_values(ascending=False)
                long_names = sorted_assets.iloc[:top_n].index.tolist()
                short_names = sorted_assets.iloc[-top_n:].index.tolist()

                new_w = pd.Series(0.0, index=cols, dtype=float)
                if cfg.weighting == "equal":
                    if cfg.long_only:
                        for n_ in long_names:
                            new_w[n_] = 1.0 / top_n
                    else:
                        for n_ in long_names:
                            new_w[n_] = 0.5 / top_n
                        for n_ in short_names:
                            new_w[n_] = -0.5 / top_n
                else:  # inverse_vol
                    rv = realised_vol_df.iloc[i]
                    if cfg.long_only:
                        iv = pd.Series(
                            [1.0 / max(rv[n_], 1e-6) for n_ in long_names], index=long_names
                        )
                        iv = iv / iv.sum()
                        for n_ in long_names:
                            new_w[n_] = iv[n_]
                    else:
                        ivl = pd.Series(
                            [1.0 / max(rv[n_], 1e-6) for n_ in long_names], index=long_names
                        )
                        ivl = ivl / ivl.sum() * 0.5
                        ivs = pd.Series(
                            [1.0 / max(rv[n_], 1e-6) for n_ in short_names], index=short_names
                        )
                        ivs = ivs / ivs.sum() * 0.5
                        for n_ in long_names:
                            new_w[n_] = ivl[n_]
                        for n_ in short_names:
                            new_w[n_] = -ivs[n_]
                current = new_w
        weights.iloc[i] = current.values

    return weights


def residual_returns(
    stocks_close: pd.DataFrame,
    ff3_df: pd.DataFrame,
    *,
    cfg: ResidualConfig | None = None,
) -> pd.Series:
    """Per-bar cost-adjusted strategy returns for one cell."""
    if cfg is None:
        cfg = ResidualConfig()

    signal = compute_residual_signal(stocks_close, ff3_df, cfg=cfg)

    realised_vol = None
    if cfg.weighting == "inverse_vol":
        log_returns = _stocks_log_returns(stocks_close)
        realised_vol = log_returns.rolling(
            cfg.inverse_vol_lookback_days, min_periods=cfg.inverse_vol_lookback_days
        ).std(ddof=1)

    weights = build_portfolio_weights(signal, cfg=cfg, realised_vol_df=realised_vol)
    log_returns = _stocks_log_returns(stocks_close)

    # Causal hold: weights as of close[t] earn return at close[t] using
    # the position decided at close[t-1]. weights are already "post-rebalance"
    # at month-end rows and held constant; multiply by next-day return.
    held_lagged = weights.shift(1).fillna(0.0)
    gross = (held_lagged * log_returns).sum(axis=1)

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

    return net.rename("residual_momentum_returns")


def residual_assert_causal(
    stocks_close: pd.DataFrame,
    ff3_df: pd.DataFrame,
    *,
    cfg: ResidualConfig | None = None,
    n_trials: int = 3,
    seed: int = 42,
) -> None:
    """A10 causality smoke test.

    Corrupt future stock prices + future FF3 at random t; assert per-bar
    strategy returns at t' < t are bit-exact unchanged.
    """
    if cfg is None:
        cfg = ResidualConfig()
    n = len(stocks_close)
    if n < 252 * 5:  # need at least 5 years for the 36-month regression to fit
        return
    base = residual_returns(stocks_close, ff3_df, cfg=cfg)
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        t_corrupt = int(rng.integers(252 * 4, n - 5))
        corrupt_stocks = stocks_close.copy()
        corrupt_stocks.iloc[t_corrupt:] = corrupt_stocks.iloc[t_corrupt:] * 1.5
        corrupt_ff3 = ff3_df.copy()
        if isinstance(corrupt_ff3.index, pd.DatetimeIndex):
            t_ts = stocks_close.index[t_corrupt]
            corrupt_ff3.loc[corrupt_ff3.index >= t_ts] *= 1.5
        altered = residual_returns(corrupt_stocks, corrupt_ff3, cfg=cfg)
        past_base = base.iloc[:t_corrupt]
        past_corr = altered.iloc[:t_corrupt]
        if not past_base.equals(past_corr):
            diff = past_base != past_corr
            raise AssertionError(
                f"residual_assert_causal: future corruption at t={t_corrupt} "
                f"changed {int(diff.sum())} past returns"
            )
