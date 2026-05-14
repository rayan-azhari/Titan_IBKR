"""GEM live-logic state machine.

A pure-python class that wraps the research-side ``gem_returns`` /
``gem_target_weights`` so that:

  1. The live trading layer (NautilusTrader) has a single object to push
     bars into and pull target weights from.
  2. Parity tests (V3.6 A10) can validate bit-exact equivalence between
     the live class outputs and the research-side batch computation,
     bar by bar, WITHOUT booting NautilusTrader.

Decoupled from NautilusTrader so it's unit-testable in isolation.

Usage::

    from research.gem.gem_strategy import GemConfig
    from titan.strategies.gem.live_logic import GemLiveLogic

    cfg = GemConfig(
        lookback_blend=(3, 6, 12), buffer_pct=0.005,
        defensive_switch=True, ann_vol_target=0.10,
        vol_lookback_days=20, max_leverage=2.0,
    )
    logic = GemLiveLogic(cfg)

    # Push bars (in chronological order):
    for ts, spy, efa, ief, vix, hyg in stream:
        logic.add_bar(ts, spy, efa, ief, vix=vix, hyg=hyg)

    # Read the current target weights at any time:
    w = logic.current_weights()  # -> {"SPY": 1.0, "EFA": 0.0, "IEF": 0.0}
"""

from __future__ import annotations

import pandas as pd

from research.gem.gem_strategy import (
    GEM_UNIVERSE,
    GemConfig,
    gem_target_weights,
)


class GemLiveLogic:
    """State machine wrapping the research-side weight computation.

    Maintains a rolling DataFrame of closes (and optional VIX/HYG history),
    re-computes the full weight series on every new bar, and exposes the
    LATEST weight via :py:meth:`current_weights`.

    This implementation is intentionally simple: it calls the research
    function on the full history each bar. For a 22-year backtest that's
    ~5600 bars and a few seconds per bar; for LIVE use it's <0.1s per
    decision (only the latest weight is needed; we could cache state to
    optimise but it's not the bottleneck).

    Parity guarantee: at any bar t, ``logic.current_weights()`` after
    ``add_bar`` on bars 0..t returns the exact value that
    ``gem_target_weights(closes[0..t]).iloc[t]`` would return. This is
    true BY CONSTRUCTION because the live class delegates to the research
    function -- V3.6 A10 satisfied without engineering effort.
    """

    def __init__(self, cfg: GemConfig | None = None) -> None:
        self.cfg = cfg or GemConfig()
        self._closes_history = pd.DataFrame(columns=list(GEM_UNIVERSE))
        self._vix_history: pd.Series = pd.Series(dtype=float, name="VIX")
        self._hyg_history: pd.Series = pd.Series(dtype=float, name="HYG")

    @property
    def n_bars(self) -> int:
        """Number of bars currently in the history."""
        return len(self._closes_history)

    def add_bar(
        self,
        timestamp: pd.Timestamp,
        spy: float,
        efa: float,
        ief: float,
        *,
        vix: float | None = None,
        hyg: float | None = None,
    ) -> None:
        """Append a new daily bar to the history.

        Timestamps should be naive (no tz) and date-normalised (00:00
        time-of-day) to match the project convention -- L20 of V3.6.
        Out-of-order or duplicate timestamps raise.
        """
        timestamp = pd.Timestamp(timestamp).tz_localize(None).normalize()
        if timestamp in self._closes_history.index:
            raise ValueError(f"GemLiveLogic.add_bar: duplicate timestamp {timestamp}")
        if len(self._closes_history) > 0 and timestamp <= self._closes_history.index[-1]:
            raise ValueError(
                f"GemLiveLogic.add_bar: out-of-order timestamp "
                f"{timestamp} <= last bar {self._closes_history.index[-1]}"
            )
        self._closes_history.loc[timestamp] = [float(spy), float(efa), float(ief)]
        if vix is not None:
            self._vix_history.loc[timestamp] = float(vix)
        if hyg is not None:
            self._hyg_history.loc[timestamp] = float(hyg)

    def add_bars_dataframe(self, closes_df: pd.DataFrame) -> None:
        """Convenience: bulk-load history from a DataFrame (e.g., during warmup).

        ``closes_df`` must have columns SPY/EFA/IEF (optional VIX/HYG).
        The index must be a sorted, deduplicated, naive DatetimeIndex
        with date-only time-of-day.
        """
        if not isinstance(closes_df.index, pd.DatetimeIndex):
            raise TypeError("closes_df must have a DatetimeIndex")
        missing = set(GEM_UNIVERSE) - set(closes_df.columns)
        if missing:
            raise ValueError(f"closes_df missing required columns: {sorted(missing)}")
        idx = closes_df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
        if not idx.is_monotonic_increasing:
            raise ValueError("closes_df index must be sorted ascending")
        if idx.duplicated().any():
            raise ValueError("closes_df index has duplicates")
        if len(self._closes_history) > 0 and idx[0] <= self._closes_history.index[-1]:
            raise ValueError("closes_df overlaps existing history")

        chunk = closes_df[list(GEM_UNIVERSE)].astype(float).copy()
        chunk.index = idx
        self._closes_history = pd.concat([self._closes_history, chunk])

        if "VIX" in closes_df.columns:
            vix_chunk = closes_df["VIX"].astype(float).copy()
            vix_chunk.index = idx
            self._vix_history = pd.concat([self._vix_history, vix_chunk])
        if "HYG" in closes_df.columns:
            hyg_chunk = closes_df["HYG"].astype(float).copy()
            hyg_chunk.index = idx
            self._hyg_history = pd.concat([self._hyg_history, hyg_chunk])

    def current_weights(self) -> dict[str, float]:
        """Return the target weights for the CURRENT bar.

        Returns a dict with keys SPY / EFA / IEF and float weights summing
        to a value in [0, max_leverage]. During warmup (insufficient history
        for the longest lookback) returns all zeros.

        Note on causality: the research function shifts weights by one bar
        (decision at close[t-1] earns return at close[t]). So the "current"
        weight returned here is the weight to HOLD INTO the NEXT bar -- which
        for live trading is exactly what the operator wants.
        """
        if len(self._closes_history) < 2:
            return {sym: 0.0 for sym in GEM_UNIVERSE}
        vix = self._vix_history if len(self._vix_history) > 0 else None
        hyg = self._hyg_history if len(self._hyg_history) > 0 else None
        weights = gem_target_weights(
            self._closes_history,
            cfg=self.cfg,
            vix=vix,
            hyg=hyg,
            ief_for_credit=self._closes_history["IEF"],
        )
        return {sym: float(weights.iloc[-1][sym]) for sym in GEM_UNIVERSE}

    def weight_history(self) -> pd.DataFrame:
        """Return the full computed weight history. Used by parity tests."""
        if len(self._closes_history) < 2:
            return pd.DataFrame(columns=list(GEM_UNIVERSE))
        vix = self._vix_history if len(self._vix_history) > 0 else None
        hyg = self._hyg_history if len(self._hyg_history) > 0 else None
        return gem_target_weights(
            self._closes_history,
            cfg=self.cfg,
            vix=vix,
            hyg=hyg,
            ief_for_credit=self._closes_history["IEF"],
        )
