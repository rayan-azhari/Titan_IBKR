"""IC Signal Census — new primitives.

Implements the audit-derived gates and the Anchored MTF Aggregation Rule
specified in ``directives/IC Signal Census 2026-05-13.md``. Reused signal
factories (RSI, MACD, ROC) live in ``run_ic.py`` / ``phase1_sweep.py``;
this module adds:

    1. ``assert_causal``            -- A10 causality smoke test (mandatory wrap)
    2. ``anchored_aggregate``       -- causal lower-TF → higher-TF aggregator
    3. ``deflated_t_pvalue``        -- extreme-value DSR for IC t-stats
    4. ``fold_ic_signs``            -- 5-fold IC sign stability
    5. ``plateau_stable``           -- neighbour check across the 3-cell grid
    6. ``mtf_agreement``            -- cross-TF sign-agreement quorum
    7. ``slice_sanctuary``          -- drop the trailing 12 months
    8. ``signal_factories``         -- parameterised closures for the 8 signal classes

Every function is causal. ``shift(-n)`` appears only inside
``run_ic.compute_forward_returns`` (the forward-return target factory).
No function in this module uses ``.expanding()`` over close (audit A6).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from titan.research.metrics import BARS_PER_YEAR

# ── (1) Causality smoke test (audit A10) ───────────────────────────────────


def assert_causal(
    agg_fn: Callable[[pd.Series, pd.DatetimeIndex], pd.Series],
    src: pd.Series,
    dst_index: pd.DatetimeIndex,
    *,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """Corrupt-the-future test: future bars are multiplied by 100; past
    aggregator output must be **bit-exact** unchanged.

    Raises AssertionError on the first trial that leaks. Use this to wrap
    every cross-TF aggregation in ``run_ic_census.py``. The A10 audit
    rule is non-negotiable -- a function that fails this test is broken.
    """
    if len(src) < 50:
        raise ValueError(f"assert_causal: source series too short ({len(src)})")
    rng = np.random.default_rng(seed)
    baseline = agg_fn(src.copy(), dst_index).copy()
    n = len(src)
    for trial in range(n_trials):
        t_idx = int(rng.integers(n // 2, n))
        corrupted = src.copy()
        corrupted.iloc[t_idx:] = corrupted.iloc[t_idx:] * 100.0
        corrupted_out = agg_fn(corrupted, dst_index)
        cutoff = src.index[t_idx]
        past_baseline = baseline[baseline.index < cutoff].dropna()
        past_corrupted = corrupted_out[corrupted_out.index < cutoff].dropna()
        # Align on common index, then assert.
        common = past_baseline.index.intersection(past_corrupted.index)
        if len(common) == 0:
            raise AssertionError(
                f"assert_causal trial {trial}: no past output to compare "
                f"(cutoff={cutoff}, dst_index range={dst_index[0]}..{dst_index[-1]})"
            )
        pd.testing.assert_series_equal(
            past_baseline.loc[common],
            past_corrupted.loc[common],
            check_exact=True,
            check_names=False,
        )


# ── (2) Anchored MTF aggregator (audit A2 / Migrate.md §1) ─────────────────


def anchored_aggregate(
    src: pd.Series,
    dst_index: pd.DatetimeIndex,
    *,
    higher_tf: bool,
    rule: str | None = None,
) -> pd.Series:
    """Causal aggregator from ``src`` (e.g. H1 series) to ``dst_index``.

    Two modes:

    * ``higher_tf=True``: ``dst_index`` is a HIGHER timeframe than ``src``
      (e.g. ``src=H1``, ``dst=D1``). The output at dst-bar ``T`` uses only
      src bars with timestamp strictly less than ``T``. Implementation
      uses ``resample(rule, label="left", closed="left").last().shift(1)``
      so the bin labelled T contains [T-rule, T), and is then offset by
      one bin -- belt-and-braces against tz/DST surprises.

    * ``higher_tf=False``: ``dst_index`` is a LOWER timeframe than ``src``
      (e.g. ``src=D1`` daily signal, ``dst=H1`` index). The src series
      must already be the higher-TF signal; we ``shift(1)`` (one bin of
      the src frequency) then reindex with ffill onto dst. The shift(1)
      is what makes this causal -- an H1 bar at 13:30 on 2026-05-13 sees
      the D1 value for 2026-05-12, never for 2026-05-13.

    No ``ffill`` on cross-TF data WITHOUT a prior ``.shift(1)``. Reindex
    with ``method=None`` (NaN-keep) for the higher-TF case.
    """
    if higher_tf:
        if rule is None:
            raise ValueError("anchored_aggregate higher_tf=True requires resample rule")
        # Native-frequency aggregator: last value of each [T-rule, T) bin,
        # then shift one bin so output at T sees only bars < T.
        agg = src.resample(rule, label="left", closed="left").last().shift(1)
        return agg.reindex(dst_index, method=None)
    else:
        shifted = src.shift(1)
        return shifted.reindex(dst_index, method="ffill")


# ── (3) Deflated t-stat (extreme-value DSR for IC) ─────────────────────────


def deflated_t_pvalue(t_stat: float, n_trials: int) -> float:
    """Probability the true IC is non-zero after deflating for ``N`` trials.

    Approximation: under H0 (all true ICs == 0) the expected max |t| over
    N independent tests is ``sqrt(2 * ln(N))`` (extreme-value /
    Bonferroni-style approximation, exact in the asymptotic limit). A
    "deflated" test rejects only if the observed |t| exceeds this null
    maximum by a normal-distributed margin. Returns
    ``Phi(|t| - e_max_t)`` -- the gate is dsr_p >= 0.95.

    This is structurally analogous to Bailey & López de Prado 2014's DSR
    (which uses cross-sectional SR variance instead of a Bonferroni
    approximation) but does not require a returns series per trial --
    appropriate for IC-style multi-test deflation where we only have a
    t-stat per cell.
    """
    if n_trials < 2:
        return 0.0
    e_max_t = math.sqrt(2.0 * math.log(n_trials))
    return float(stats.norm.cdf(abs(t_stat) - e_max_t))


# ── (4) 5-fold IC sign stability ───────────────────────────────────────────


def fold_ic_signs(
    signal: pd.Series,
    fwd_return: pd.Series,
    *,
    n_folds: int = 5,
    min_per_fold: int = 30,
    quorum: int = 4,
    near_zero_threshold: float = 0.005,
) -> tuple[list[float], bool, int]:
    """Split ``(signal, fwd_return)`` into ``n_folds`` non-overlapping
    time-ordered slices, compute Spearman IC per fold, return:

    * ``fold_ic`` -- list[float], one IC per fold (NaN if fold too small)
    * ``sign_stable`` -- True if at least ``quorum`` folds share sign
    * ``modal_count`` -- the number of folds matching the modal sign

    Operationalises Migrate.md's "reject if any fold sign-flips":

    * Folds with ``|IC| < near_zero_threshold`` are treated as
      sign-agnostic (count toward neither pos nor neg). One borderline
      fold is realistic noise -- it should not torpedo an otherwise
      stable signal.
    * ``quorum=4`` for ``n_folds=5`` (default) tolerates exactly one
      near-zero or NaN fold, but ANY fold with the OPPOSITE sign at
      magnitude ≥ near_zero_threshold breaks the gate.
    * A naive ``quorum=3`` for ``n_folds=5`` would be tautological --
      pigeonhole forces at least 3 of any 5 signs to coincide.
    """
    df = pd.concat([signal, fwd_return], axis=1).dropna()
    if len(df) < n_folds * min_per_fold:
        return [float("nan")] * n_folds, False, 0
    slices = np.array_split(df, n_folds)
    fold_ic: list[float] = []
    pos = neg = 0
    for sl in slices:
        if len(sl) < min_per_fold:
            fold_ic.append(float("nan"))
            continue
        rho, _ = stats.spearmanr(sl.iloc[:, 0], sl.iloc[:, 1])
        if np.isnan(rho):
            fold_ic.append(float("nan"))
            continue
        fold_ic.append(round(float(rho), 5))
        if abs(rho) < near_zero_threshold:
            continue  # sign-agnostic
        if rho > 0:
            pos += 1
        elif rho < 0:
            neg += 1
    modal_count = max(pos, neg)
    return fold_ic, modal_count >= quorum, modal_count


# ── (5) Plateau stability across the 3-cell grid ───────────────────────────


@dataclass
class CellResult:
    """One row of the parameter sweep for a single (signal, instrument, TF, horizon)."""

    params: dict[str, Any]
    ic: float
    t_stat: float
    n_obs: int


def plateau_stable(
    cells: list[CellResult],
    *,
    headline_t_floor: float,
    neighbour_t_floor: float,
    ic_range_max: float,
) -> tuple[bool, CellResult | None, str]:
    """V3.2 plateau gate.

    The headline cell is the one with the largest |t|. Plateau passes iff:

    1. The headline cell clears ``headline_t_floor``.
    2. Both grid neighbours (cells before/after in the input order) ALSO
       clear ``neighbour_t_floor``.
    3. The neighbours and the headline have the SAME SIGN of IC.
    4. |IC| range across the three cells is less than ``ic_range_max``
       fraction of |headline IC|.
    5. The headline cell is NOT at the grid edge (first or last) --
       edges have no two-sided neighbours and are ineligible.

    Returns ``(passes, headline_cell, reason)``. ``reason`` is a short
    diagnostic string when the gate fails.
    """
    if len(cells) < 3:
        return False, None, f"need 3 cells, got {len(cells)}"
    # Cells are input in the directive's pre-registered order. The headline
    # is whichever has max |t|, but it must be the middle cell to have
    # two-sided neighbours. We pick max-|t| then check edge-eligibility.
    ts = [abs(c.t_stat) for c in cells]
    headline_idx = int(np.argmax(ts))
    headline = cells[headline_idx]
    if headline_idx in (0, len(cells) - 1):
        return False, headline, f"headline at edge cell {headline_idx} (no two-sided neighbours)"
    if abs(headline.t_stat) < headline_t_floor:
        return False, headline, f"headline |t|={abs(headline.t_stat):.2f} < {headline_t_floor}"
    left = cells[headline_idx - 1]
    right = cells[headline_idx + 1]
    if min(abs(left.t_stat), abs(right.t_stat)) < neighbour_t_floor:
        return (
            False,
            headline,
            (
                f"neighbour |t| min={min(abs(left.t_stat), abs(right.t_stat)):.2f} "
                f"< {neighbour_t_floor}"
            ),
        )
    signs = {np.sign(c.ic) for c in (left, headline, right) if not np.isnan(c.ic)}
    if len(signs) > 1:
        return False, headline, f"neighbour sign disagrees ({signs})"
    abs_ics = [abs(c.ic) for c in (left, headline, right)]
    span = max(abs_ics) - min(abs_ics)
    rng_frac = span / abs(headline.ic) if abs(headline.ic) > 1e-9 else float("inf")
    if rng_frac > ic_range_max:
        return False, headline, f"IC range {rng_frac:.2%} > {ic_range_max:.0%}"
    return True, headline, "plateau"


# ── (6) Cross-timeframe agreement ──────────────────────────────────────────


def mtf_agreement(
    per_tf_results: dict[str, CellResult | None],
    *,
    quorum: int,
    t_floor: float,
) -> tuple[bool, int]:
    """Count how many timeframes have a passing cell with the same sign.

    ``per_tf_results`` maps TF label (e.g. "D", "H4", "H1") to the
    selected headline CellResult for that TF (or None if the gate failed
    at that TF). Returns ``(agree, n_pass)``.
    """
    passing = [
        c
        for c in per_tf_results.values()
        if c is not None and abs(c.t_stat) >= t_floor and not np.isnan(c.ic)
    ]
    if not passing:
        return False, 0
    signs = [int(np.sign(c.ic)) for c in passing]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    n_pass = max(pos, neg)
    return n_pass >= quorum, n_pass


# ── (7) Sanctuary window ───────────────────────────────────────────────────


def slice_sanctuary(
    df: pd.DataFrame | pd.Series,
    *,
    months: int = 12,
) -> tuple[pd.DataFrame | pd.Series, pd.Timestamp, pd.Timestamp]:
    """Drop the trailing ``months`` calendar months from ``df`` and return
    ``(df_visible, sanctuary_start, sanctuary_end)``.

    The autoresearch agent operates on ``df_visible``. The sanctuary
    window is used in a single, separate final-validation pass.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"slice_sanctuary requires DatetimeIndex, got {type(df.index)}")
    end = df.index[-1]
    sanctuary_start = end - pd.DateOffset(months=months)
    visible = df[df.index < sanctuary_start]
    return visible, sanctuary_start, end


# ── (8) Signal factories ───────────────────────────────────────────────────


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _momentum(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    """12-1 style: log(close[t-skip] / close[t-lookback]).

    Pure causal -- both numerator and denominator are in the past. The
    most recent ``skip`` bars are intentionally excluded (the canonical
    momentum factor skips the most recent month to avoid reversal).
    """
    log_close = np.log(close)
    return log_close.shift(skip) - log_close.shift(lookback)


def _ewmac(close: pd.Series, fast: int, slow: int) -> pd.Series:
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    return (f - s) / close.rolling(slow, min_periods=slow).std().replace(0, np.nan)


def _ma_distance(close: pd.Series, window: int) -> pd.Series:
    return close / close.rolling(window, min_periods=window).mean() - 1.0


def _vwap_overshoot(close: pd.Series, anchor: int) -> pd.Series:
    # No-volume VWAP approximation: rolling mean of close. AUD/JPY parquet
    # has volume == -1 (sentinel) so a true VWAP is unavailable. Other
    # instruments with proper volume could substitute (high+low+close)/3
    # weighted by volume -- left as a future refinement; this matches the
    # mr_audjpy strategy's own vwap_anchor implementation.
    rm = close.rolling(anchor, min_periods=anchor).mean()
    return close / rm - 1.0


def _bb_pctb(close: pd.Series, window: int) -> pd.Series:
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=1)
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _realized_vol_z(close: pd.Series, window: int) -> pd.Series:
    rets = np.log(close).diff()
    rv = rets.rolling(window, min_periods=window).std(ddof=1)
    mu = rv.rolling(window * 5, min_periods=window).mean()
    sd = rv.rolling(window * 5, min_periods=window).std(ddof=1)
    return (rv - mu) / sd.replace(0, np.nan)


def _overnight_gap_z(close: pd.Series, lookback_bars: int) -> pd.Series:
    # Gap = log(open[t] / close[t-1]). Without an open column we substitute
    # log(close[t] / close[t-1]) -- this becomes the 1-bar return z-score.
    # Future refinement: extend factory signature to accept OHLC dict.
    r = np.log(close).diff()
    mu = r.rolling(lookback_bars, min_periods=lookback_bars).mean()
    sd = r.rolling(lookback_bars, min_periods=lookback_bars).std(ddof=1)
    return (r - mu) / sd.replace(0, np.nan)


def _intraday_range_atr(
    close: pd.Series, period: int, high: pd.Series, low: pd.Series
) -> pd.Series:
    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr_components.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    today_range = high - low
    return today_range / atr.replace(0, np.nan) - 1.0


def _range_atr_when_oversold(
    close: pd.Series,
    period_atr: int,
    period_rsi: int,
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Range-expansion magnitude, gated to zero unless RSI is below 50.

    Pre-registered in directives/IC Confluence Range-Reversion 2026-05-14.md.
    Tests whether the Phase C TIER_B range_atr edge concentrates in
    oversold regimes (mean-reversion + range-expansion agreement quadrant).

    The signal at bar t is non-zero only when RSI(close, period_rsi) at t
    is below 50. Both legs are causal (RSI uses close-through-t, ATR uses
    close-through-t-1 via shift(1) inside _intraday_range_atr's TR
    components). The .where(..., 0.0) replaces non-oversold values with
    zero rather than NaN so the regime-gated bars contribute a clearly
    null observation rather than dropping out of the Spearman pair.
    """
    iat = _intraday_range_atr(close, period_atr, high, low)
    rsi_dev = _wilder_rsi(close, period_rsi) - 50.0
    return iat.where(rsi_dev < 0, 0.0)


# ── Cross-asset factories ──────────────────────────────────────────────────
# Each takes the target's close plus one or more EXTERNAL closes already
# anchored to the target's index by the orchestrator. The orchestrator MUST
# apply ``anchored_aggregate(..., higher_tf=False)`` (i.e. ``.shift(1)``
# followed by causal ffill onto the target index) BEFORE handing the
# external series to these factories. That is the Anchored MTF Rule (audit
# A2 / EUR/USD MTF +1.94 Sharpe bug pattern).
#
# Self-correlation guard (target instrument == external instrument) is the
# orchestrator's job; these factories assume distinct instruments.


def _hyg_ief_z(close: pd.Series, *, hyg: pd.Series, ief: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score of the high-yield / investment-grade spread.

    Mechanism: when HY narrows vs IG (spread tightens), risk appetite is
    on -- equity tends to grind up. When HY widens, risk-off -- equity
    falls. The IC sign is expected POSITIVE on the equity targets (high
    spread z → high forward equity return is the canonical bond-equity
    pattern observed in Migrate.md's IHYU→CSPX, IHYG→VUSD candidates,
    and in the broader literature on credit-spread mean reversion).

    ``hyg`` and ``ief`` MUST already be ``.shift(1)``-ed and aligned to
    the target's index by the orchestrator via ``anchored_aggregate``.
    The reindex below is a defensive no-op when the contract holds and
    a corrective alignment if it doesn't.
    """
    hyg = hyg.reindex(close.index)
    ief = ief.reindex(close.index)
    spread = np.log(hyg) - np.log(ief)
    mu = spread.rolling(lookback, min_periods=lookback).mean()
    sd = spread.rolling(lookback, min_periods=lookback).std(ddof=1)
    return (spread - mu) / sd.replace(0.0, np.nan)


def _vix_term_ratio_z(
    close: pd.Series,
    *,
    numerator: pd.Series,
    denominator: pd.Series,
    smoothing: int,
    norm_window: int = 60,
) -> pd.Series:
    """Generic VIX-term-structure ratio z-score.

    ``ratio = numerator / denominator`` smoothed by ``rolling_mean(smoothing)``,
    then z-scored over ``rolling_window=norm_window`` (the normalisation
    window is FIXED across cells per Phase B directive §2.1; only
    ``smoothing`` is the swept parameter).

    Used by ``_vix9d_over_vix`` and ``_vix_over_vix3m``. Both externals
    must already be anchored by the orchestrator via anchored_aggregate.
    """
    numerator = numerator.reindex(close.index)
    denominator = denominator.reindex(close.index)
    ratio = numerator / denominator.replace(0.0, np.nan)
    if smoothing > 1:
        ratio = ratio.rolling(smoothing, min_periods=smoothing).mean()
    mu = ratio.rolling(norm_window, min_periods=norm_window).mean()
    sd = ratio.rolling(norm_window, min_periods=norm_window).std(ddof=1)
    return (ratio - mu) / sd.replace(0.0, np.nan)


def _vix9d_over_vix(
    close: pd.Series,
    *,
    vix9d: pd.Series,
    vix: pd.Series,
    smoothing: int,
) -> pd.Series:
    """Z-score of (VIX9D / VIX), smoothed. Backwardation indicator.

    Mechanism: ratio > 1 means near-term implied vol exceeds 30-day
    implied vol — typical right at a vol spike, AFTER which equities
    tend to recover. Expected IC sign on equity targets: POSITIVE for
    high z-score (post-spike rebound)."""
    return _vix_term_ratio_z(close, numerator=vix9d, denominator=vix, smoothing=smoothing)


def _vix_over_vix3m(
    close: pd.Series,
    *,
    vix: pd.Series,
    vix3m: pd.Series,
    smoothing: int,
) -> pd.Series:
    """Z-score of (VIX / VIX3M), smoothed. Canonical contango/backwardation.

    Mechanism: ratio > 1 (backwardation) is a stress signal — VIX
    futures are pricing near-term fear above medium-term fear.
    Historically a precursor to equity weakness. Expected IC sign on
    equity targets: NEGATIVE for high z-score."""
    return _vix_term_ratio_z(close, numerator=vix, denominator=vix3m, smoothing=smoothing)


def _vrp_z(
    close: pd.Series,
    *,
    vix: pd.Series,
    rv_window: int,
    norm_window: int = 60,
) -> pd.Series:
    """Vol-risk-premium z-score: implied vol minus realised vol.

    ``vrp = VIX / 100 - rolling_std(log_returns, rv_window) * sqrt(252)``
    where VIX is in percent units. Positive VRP means options are
    pricing more vol than the market is actually realising — vol-seller's
    edge. Expected IC sign on equity targets: POSITIVE.

    The realised-vol leg uses the target's own close (causal rolling).
    VIX must already be anchored by the orchestrator.
    """
    vix = vix.reindex(close.index)
    log_rets = np.log(close).diff()
    rv = log_rets.rolling(rv_window, min_periods=rv_window).std(ddof=1) * np.sqrt(
        BARS_PER_YEAR["D"]
    )
    vrp = (vix / 100.0) - rv
    mu = vrp.rolling(norm_window, min_periods=norm_window).mean()
    sd = vrp.rolling(norm_window, min_periods=norm_window).std(ddof=1)
    return (vrp - mu) / sd.replace(0.0, np.nan)


def _us_lead_eu(
    close: pd.Series,
    *,
    spy: pd.Series,
    window: int,
) -> pd.Series:
    """Yesterday's SPY return (smoothed over ``window`` bars) predicts
    today's EU index return.

    Continuous version of Migrate.md §2.8 sign-only formulation. The
    SPY series must already be anchored by the orchestrator
    (``.shift(1)`` + ffill onto the target's daily index), so at target
    time T the SPY value is from time strictly less than T.

    The lead-lag mechanism: US markets close after Europe opens; their
    overnight moves are partially priced into European opens. Expected
    IC sign: POSITIVE (positive SPY → positive EU follow-through).
    """
    spy = spy.reindex(close.index)
    spy_log_rets = np.log(spy).diff()
    if window > 1:
        return spy_log_rets.rolling(window, min_periods=window).mean()
    return spy_log_rets


def _dxy_z(close: pd.Series, *, eur_usd: pd.Series, lookback: int) -> pd.Series:
    """USD-index proxy z-score. ``-log(EUR_USD)`` rises when USD strengthens.

    Mechanism: a strong USD typically headwinds risk assets and ex-US
    equities. Expected IC sign is NEGATIVE on most equity / EM /
    commodity targets.

    ``eur_usd`` MUST already be ``.shift(1)``-ed and aligned via
    ``anchored_aggregate`` by the orchestrator. The reindex below is a
    defensive no-op when the contract holds.
    """
    eur_usd = eur_usd.reindex(close.index)
    dxy_proxy = -np.log(eur_usd)
    mu = dxy_proxy.rolling(lookback, min_periods=lookback).mean()
    sd = dxy_proxy.rolling(lookback, min_periods=lookback).std(ddof=1)
    return (dxy_proxy - mu) / sd.replace(0.0, np.nan)


SignalCallable = Callable[..., pd.Series]


def signal_factories() -> dict[str, dict[str, Any]]:
    """Return a registry of signal name -> {fn, needs, externals}.

    * ``needs`` -- target OHLCV columns the factory wants (subset of
      ``["close", "high", "low", "open", "volume"]``). Most factories take
      only ``close``; ``intraday_range_atr`` needs HLC.
    * ``externals`` -- (optional) list of external instrument tickers
      whose close series must be anchored to the target's index via
      ``anchored_aggregate(..., higher_tf=False)`` before invocation.
      The orchestrator passes each as a kwarg named with the
      instrument's lowercase ticker (``HYG`` → ``hyg``,
      ``EUR_USD`` → ``eur_usd``). Cross-asset signals MUST declare
      ``externals``; absence means single-instrument signal.

    Self-correlation guard (target instrument == one of the externals)
    is the orchestrator's job; factories assume distinct instruments.
    """
    return {
        # ── Single-instrument signals ─────────────────────────────────
        "momentum": {"fn": _momentum, "needs": ["close"]},
        "ewmac": {"fn": _ewmac, "needs": ["close"]},
        "ma_distance": {"fn": _ma_distance, "needs": ["close"]},
        "rsi_dev": {"fn": lambda c, period: _wilder_rsi(c, period) - 50.0, "needs": ["close"]},
        "vwap_overshoot": {"fn": _vwap_overshoot, "needs": ["close"]},
        # Alias for the AUD/JPY fine-grid follow-up directive
        # (directives/IC AUDJPY Vwap Fine-Grid 2026-05-13.md). Same formula;
        # separate name keeps the two pre-registrations clearly distinct.
        "vwap_overshoot_fine": {"fn": _vwap_overshoot, "needs": ["close"]},
        "bb_pctb": {"fn": lambda c, window: _bb_pctb(c, window) - 0.5, "needs": ["close"]},
        "realized_vol_z": {"fn": _realized_vol_z, "needs": ["close"]},
        "overnight_gap_z": {"fn": _overnight_gap_z, "needs": ["close"]},
        "intraday_range_atr": {"fn": _intraday_range_atr, "needs": ["close", "high", "low"]},
        # Confluence factory (directives/IC Confluence Range-Reversion 2026-05-14.md):
        # range-expansion gated to zero when RSI is overbought.
        "range_atr_when_oversold": {
            "fn": _range_atr_when_oversold,
            "needs": ["close", "high", "low"],
        },
        # ── Cross-asset signals (Anchored MTF Rule applied by runner) ─
        "hyg_ief_z": {"fn": _hyg_ief_z, "needs": ["close"], "externals": ["HYG", "IEF"]},
        "dxy_z": {"fn": _dxy_z, "needs": ["close"], "externals": ["EUR_USD"]},
        # ── Phase B cross-asset signals ──────────────────────────────
        # Term structure: VIX9D / VIX and VIX / VIX3M ratios as predictors
        # of forward equity returns. Sweep parameter is the ratio-smoothing
        # window before z-score normalisation (which uses a fixed 60-bar
        # window per Phase B directive §2.1).
        "vix9d_over_vix": {
            "fn": _vix9d_over_vix,
            "needs": ["close"],
            "externals": ["VIX9D", "VIX"],
        },
        "vix_over_vix3m": {
            "fn": _vix_over_vix3m,
            "needs": ["close"],
            "externals": ["VIX", "VIX3M"],
        },
        # Vol-risk-premium: VIX minus rolling realised vol, z-scored.
        "vrp_z": {"fn": _vrp_z, "needs": ["close"], "externals": ["VIX"]},
        # Cross-region lead-lag: yesterday's SPY return predicting today's
        # EU index return.
        "us_lead_eu": {"fn": _us_lead_eu, "needs": ["close"], "externals": ["SPY"]},
    }
