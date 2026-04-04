"""
feature_engine.py

The central fix to the research/production parity problem.
A single class whose step() method is called tick-by-tick in Nautilus
and whose batch() method wraps step() sequentially in research.

Invariant: feature_engine.batch(prices) == [feature_engine.step(p) for p in prices]

This is not a convenience wrapper -- it is the architectural guarantee that
live features are byte-for-byte identical to training features.
"""

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngine:
    """
    Computes three stationary features incrementally:
      - log_return:  log(p_t / p_{t-1})
      - ewm_vol:     exponentially weighted std of log returns over `vol_span` periods
      - frac_diff:   fractionally differenced price, window truncated at `frac_tau`

    The frac_diff window is computed once at init via the tau-threshold method
    (de Prado 2018, Ch. 5). The same truncated window is used in both research
    and production, resolving the core memory-mismatch bug.
    """

    def __init__(
        self,
        vol_span: int = 20,
        frac_d: float = 0.4,
        frac_tau: float = 1e-4,
    ):
        self.vol_span = vol_span
        self.frac_d = frac_d
        self.frac_tau = frac_tau

        # Compute and store weights once
        self._weights = self._compute_frac_weights(frac_d, frac_tau)
        self._frac_window = len(self._weights)

        # Internal state -- reset between batch() calls
        self._price_buf: deque = deque(maxlen=self._frac_window)
        self._ret_buf: deque = deque(maxlen=vol_span * 4)
        self._prev_price: Optional[float] = None

    # -- Public API ----------------------------------------------------------

    def step(self, price: float) -> Optional[np.ndarray]:
        """
        Process one new price. Returns [log_return, ewm_vol, frac_diff]
        or None during warm-up.
        Called by Nautilus on_bar(), and internally by batch().
        """
        self._price_buf.append(price)

        log_ret = None
        if self._prev_price is not None:
            log_ret = np.log(price / self._prev_price)
            self._ret_buf.append(log_ret)
        self._prev_price = price

        if len(self._price_buf) < self._frac_window or len(self._ret_buf) < self.vol_span:
            return None

        vol = self._ewm_vol(np.array(self._ret_buf), self.vol_span)
        fd = float(np.dot(self._weights, np.array(self._price_buf)[-self._frac_window :]))

        return np.array([log_ret, vol, fd], dtype=np.float64)

    def batch(self, prices: pd.Series) -> pd.DataFrame:
        """
        Research entrypoint. Resets state and calls step() sequentially.
        Output is guaranteed identical to on_bar() sequence for the same prices.
        """
        self._reset()
        records = []
        for ts, p in prices.items():
            feat = self.step(float(p))
            if feat is not None:
                records.append((ts, feat))

        if not records:
            return pd.DataFrame(columns=["log_return", "ewm_vol", "frac_diff"])

        idx, feats = zip(*records)
        return pd.DataFrame(
            np.vstack(feats),
            index=pd.DatetimeIndex(idx),
            columns=["log_return", "ewm_vol", "frac_diff"],
        )

    @property
    def min_history(self) -> int:
        """Bars required before the first feature is emitted."""
        return self._frac_window + self.vol_span

    @property
    def frac_window(self) -> int:
        """The tau-truncated memory window for frac diff."""
        return self._frac_window

    # -- Private helpers -----------------------------------------------------

    def _reset(self):
        self._price_buf.clear()
        self._ret_buf.clear()
        self._prev_price = None

    @staticmethod
    def _compute_frac_weights(d: float, tau: float) -> np.ndarray:
        """
        Compute fractional differencing weights w_k = prod_{i=0}^{k-1} (d-i)/(i+1).
        Truncate when |w_k| < tau. Returns array oldest-first for dot product.
        """
        weights = [1.0]
        k = 1
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < tau:
                break
            weights.append(w)
            k += 1
        return np.array(weights[::-1])  # oldest first

    @staticmethod
    def _ewm_vol(returns: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1)
        n = len(returns)
        w = (1 - alpha) ** np.arange(n)[::-1]
        w /= w.sum()
        mu = np.dot(w, returns)
        var = np.dot(w, (returns - mu) ** 2)
        return float(np.sqrt(var))
