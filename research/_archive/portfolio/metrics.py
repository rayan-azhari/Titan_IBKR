"""Portfolio-level statistics and risk-of-ruin calculations.

All functions accept pd.Series of daily returns (fractional, e.g. 0.01 = 1%).
No look-ahead bias: statistics are computed on the provided series only.

Key outputs:
    compute_portfolio_stats()    — full stats dict for a combined return series
    compute_ror_montecarlo()     — block-bootstrap P(ruin at 25% DD)
    compute_ror_balsara()        — analytical Balsara adapted for daily returns
    apply_correlation_constraint() — reduce weights for correlated strategy pairs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

RUIN_THRESHOLD_PCT: float = 0.25  # 25% DD = ruin
ANNUAL_BARS: int = 252  # trading days per year
BLOCK_SIZE: int = 20  # block-bootstrap block size (≈ 1 trading month)
N_SIM_DEFAULT: int = 10_000


# ── Risk of Ruin — Monte Carlo block bootstrap ─────────────────────────────────


def compute_ror_montecarlo(
    daily_returns: pd.Series,
    n_simulations: int = N_SIM_DEFAULT,
    ruin_threshold_pct: float = RUIN_THRESHOLD_PCT,
    horizon_days: int = 504,  # 2 years
    block_size: int = BLOCK_SIZE,
    random_seed: int = 42,
) -> float:
    """P(equity falls below 1 - ruin_threshold_pct at any point in horizon_days).

    Uses block bootstrap to preserve serial correlation and volatility clustering.
    Fully vectorised over all simulation paths — no Python inner loop.

    Memory: n_simulations × horizon_days × 8 bytes ≈ 20 MB for default params.
    """
    if n_simulations <= 0:
        return 0.0  # caller opted out of MC (fast mode); Balsara used instead

    rng = np.random.default_rng(random_seed)
    rets = daily_returns.dropna().values
    n_obs = len(rets)

    if n_obs < block_size * 2:
        return 1.0

    ruin_level = 1.0 - ruin_threshold_pct

    # How many blocks needed to cover horizon_days (with a small buffer)
    n_blocks = int(np.ceil(horizon_days / block_size)) + 1

    # Sample all block start indices at once: shape (n_simulations, n_blocks)
    starts = rng.integers(0, n_obs - block_size + 1, size=(n_simulations, n_blocks))

    # Build index array into rets: shape (n_simulations, n_blocks, block_size)
    offsets = np.arange(block_size)  # (block_size,)
    indices = starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]

    # Gather returns, flatten to (n_simulations, n_blocks*block_size), truncate
    path_rets = rets[indices].reshape(n_simulations, -1)[:, :horizon_days]

    # Cumulative equity via log-sum for numerical stability
    log1p = np.log1p(path_rets)           # (n_simulations, horizon_days)
    cum_equity = np.exp(np.cumsum(log1p, axis=1))  # (n_simulations, horizon_days)

    # Ruin = any point where equity <= ruin_level
    ruined = (cum_equity <= ruin_level).any(axis=1)  # (n_simulations,)
    return float(ruined.mean())


# ── Risk of Ruin — Balsara adapted for daily returns ──────────────────────────


def compute_ror_balsara(
    daily_returns: pd.Series,
    ruin_threshold_pct: float = RUIN_THRESHOLD_PCT,
) -> float:
    """Balsara formula adapted to treat each daily return as a 'trade'.

    Normalises to R-multiples using average |daily_loss| as the risk unit.
    Matches the C3 fix in research/ic_analysis/phase3_backtest.py:_risk_of_ruin().

    This is an analytical approximation — less accurate than Monte Carlo for
    multi-strategy portfolios but fast and provides a lower bound.

    Returns:
        float in [0, 1] — P(ruin at ruin_threshold_pct DD).
    """
    rets = daily_returns.dropna()
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    if len(losses) == 0:
        return 0.0  # no losing days

    win_rate = len(wins) / len(rets)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss_abs = float(losses.abs().mean())  # positive value

    if avg_loss_abs < 1e-12:
        return 0.0

    # Normalise to R-multiples using avg |daily_loss| as the risk unit
    avg_win_R = avg_win / avg_loss_abs
    avg_loss_R = 1.0  # by definition

    edge = win_rate * avg_win_R - (1.0 - win_rate) * avg_loss_R
    if edge <= 0:
        return 1.0  # negative edge → certain ruin

    # cap_units = ruin_threshold / risk_unit_as_fraction_of_equity
    # risk_unit = avg_loss_abs; expressed as fraction of equity already
    cap_units = ruin_threshold_pct / avg_loss_abs
    ratio = (1.0 - edge) / (1.0 + edge)
    if ratio <= 0:
        return 0.0  # edge so large ruin is impossible
    return float(np.clip(ratio**cap_units, 0.0, 1.0))


# ── Correlation constraint ─────────────────────────────────────────────────────


def apply_correlation_constraint(
    weights: dict[str, float],
    component_returns: dict[str, pd.Series],
    corr_threshold: float = 0.70,
) -> dict[str, float]:
    """Reduce weights for highly correlated strategy pairs.

    For each pair (A, B) where |r| > corr_threshold: the combined weight of the
    pair is capped to the weight of a single uncorrelated strategy (average weight).
    Implementation: w_A = w_B = 0.5 * (w_A + w_B), then re-normalise.

    Per Ensemble Strategy Framework.md: threshold = 0.70.
    """
    names = list(weights.keys())
    w = dict(weights)  # copy

    # Build correlation matrix from aligned returns
    rets_df = pd.DataFrame({k: component_returns[k] for k in names}).dropna()
    if len(rets_df) < 10:
        return w

    corr = rets_df.corr()

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) > corr_threshold:
                combined = w[a] + w[b]
                w[a] = combined * 0.5
                w[b] = combined * 0.5

    # Re-normalise
    total = sum(w.values())
    if total > 1e-9:
        w = {k: v / total for k, v in w.items()}
    return w


# ── Full portfolio statistics ──────────────────────────────────────────────────


def compute_portfolio_stats(
    port_returns: pd.Series,
    weights: dict[str, float],
    component_returns: dict[str, pd.Series],
    n_sim_ror: int = N_SIM_DEFAULT,
) -> dict:
    """Compute full portfolio-level statistics.

    Args:
        port_returns:       Combined portfolio daily returns (pre-computed).
        weights:            Strategy weights dict {label: weight}.
        component_returns:  Individual strategy return series (aligned to same index).
        n_sim_ror:          Monte Carlo simulations for RoR (set lower for fast runs).

    Returns dict with keys:
        sharpe, calmar, max_dd, cagr, win_rate_daily, avg_win_daily, avg_loss_daily,
        diversification_ratio, avg_pairwise_corr, ror_montecarlo, ror_balsara,
        ror_composite, individual_sharpes, n_days, n_years.
    """
    rets = port_returns.dropna()
    if len(rets) < 20:
        return {
            k: float("nan")
            for k in [
                "sharpe",
                "calmar",
                "max_dd",
                "cagr",
                "ror_montecarlo",
                "ror_balsara",
                "ror_composite",
                "diversification_ratio",
                "avg_pairwise_corr",
            ]
        }

    # ── Core metrics ───────────────────────────────────────────────────────
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(ANNUAL_BARS)) if std > 1e-9 else 0.0

    equity = (1.0 + rets).cumprod()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = float(dd.min())

    n_days = len(rets)
    n_years = n_days / ANNUAL_BARS
    total_ret = float(equity.iloc[-1] - 1.0)
    cagr = float((1.0 + total_ret) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate_daily = len(wins) / n_days if n_days > 0 else 0.0
    avg_win_daily = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss_daily = float(losses.mean()) if len(losses) > 0 else 0.0

    # ── Diversification ratio (Choueifaty-Coignard) ────────────────────────
    w_arr = np.array([weights.get(k, 0.0) for k in component_returns])
    comp_vols = np.array(
        [component_returns[k].std() * np.sqrt(ANNUAL_BARS) for k in component_returns]
    )
    port_vol = std * np.sqrt(ANNUAL_BARS)
    dr = float(np.dot(w_arr, comp_vols) / port_vol) if port_vol > 1e-9 else 1.0

    # ── Correlation matrix ─────────────────────────────────────────────────
    ret_df = pd.DataFrame(component_returns).dropna()
    corr = ret_df.corr()
    n_strat = len(corr)
    if n_strat > 1:
        upper_idx = np.triu_indices(n_strat, k=1)
        off_diag = corr.values[upper_idx]
        avg_corr = float(off_diag.mean())
    else:
        avg_corr = 0.0

    # ── Risk of Ruin ───────────────────────────────────────────────────────
    ror_mc = compute_ror_montecarlo(rets, n_simulations=n_sim_ror)
    ror_bs = compute_ror_balsara(rets)

    # ── Per-component Sharpe ───────────────────────────────────────────────
    individual_sharpes: dict[str, float] = {}
    for k, s in component_returns.items():
        s_clean = s.dropna()
        sd = float(s_clean.std())
        individual_sharpes[k] = (
            float(s_clean.mean() / sd * np.sqrt(ANNUAL_BARS)) if sd > 1e-9 else 0.0
        )

    return {
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "win_rate_daily": round(win_rate_daily, 4),
        "avg_win_daily": round(avg_win_daily, 6),
        "avg_loss_daily": round(avg_loss_daily, 6),
        "diversification_ratio": round(dr, 3),
        "avg_pairwise_corr": round(avg_corr, 3),
        "ror_montecarlo": round(ror_mc, 5),
        "ror_balsara": round(ror_bs, 5),
        "ror_composite": round(max(ror_mc, ror_bs), 5),
        "individual_sharpes": individual_sharpes,
        "correlation_matrix": corr,
        "n_days": n_days,
        "n_years": round(n_years, 2),
    }
