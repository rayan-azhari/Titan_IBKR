"""Phase 0 — Regime Labelling. Produces per-bar ADX + HMM regime labels for every bar
in an OHLCV parquet file. Output is consumed by Phase 1 (conditional IC splits) and
Phase 5 (regime robustness gate).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_ohlcv(instrument: str, timeframe: str) -> pd.DataFrame:
    path = ROOT / "data" / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError(f"Cannot resolve timestamp index: {path}")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df[["open", "high", "low", "close"]].dropna()


# ---------------------------------------------------------------------------
# Fractional differencing
# ---------------------------------------------------------------------------


def _frac_diff(series: pd.Series, d: float = 0.35, threshold: float = 1e-5) -> pd.Series:
    """Fractionally difference a series with order d using fixed-width window."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    w_arr = np.array(weights[::-1])  # oldest first
    width = len(w_arr)
    result = series.copy() * np.nan
    for i in range(width - 1, len(series)):
        result.iloc[i] = float(w_arr @ series.iloc[i - width + 1 : i + 1].values)
    return result


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def compute_regime_labels(
    df: pd.DataFrame,
    hmm_n_states: int = 2,
    frac_diff: bool = False,
) -> pd.DataFrame:
    """Compute regime labels for every bar in an OHLCV DataFrame.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns open, high, low, close.
    hmm_n_states:
        Number of HMM states (default 2).
    frac_diff:
        When True, append fractionally differenced close (d=0.35).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with columns appended:
        - adx_regime: "ranging" / "neutral" / "trending"
        - hmm_state: int (0-indexed HMM state)
        - frac_diff_close: (only when frac_diff=True)
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print(
            "hmmlearn is not installed. Install it with:\n"
            "    pip install hmmlearn\n"
            "or:\n"
            "    uv add hmmlearn"
        )
        sys.exit(1)

    try:
        from titan.strategies.ml.features import adx
    except ImportError as exc:
        log.error("Could not import adx from titan.strategies.ml.features: %s", exc)
        raise

    out = df.copy()

    # --- ADX regime ---
    log.info("Computing ADX(14)...")
    adx_series = adx(df, 14)
    out["adx_regime"] = pd.cut(
        adx_series,
        bins=[-np.inf, 20, 25, np.inf],
        labels=["ranging", "neutral", "trending"],
    )

    # --- HMM regime ---
    log.info("Fitting %d-state Gaussian HMM (IS split: 70%%)...", hmm_n_states)
    log_ret = np.log(df["close"]).diff().fillna(0.0)
    rvol20 = log_ret.rolling(20).std().bfill()

    # Prevent look-ahead bias: Fit HMM and calculate normalisation statistics ONLY on IS data.
    is_n = max(100, int(len(df) * 0.70))
    log_ret_is = log_ret.iloc[:is_n]
    rvol20_is = rvol20.iloc[:is_n]

    # Normalise full series using IS mean/std
    ret_z = (log_ret - log_ret_is.mean()) / log_ret_is.std()
    vol_z = (rvol20 - rvol20_is.mean()) / rvol20_is.std()

    X_full = np.column_stack([ret_z, vol_z])
    X_is = X_full[:is_n]

    model = GaussianHMM(
        n_components=hmm_n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(X_is)
    hmm_state = model.predict(X_full)
    out["hmm_state"] = hmm_state.astype(int)

    # M5 FIX: Warn when OOS features fall outside the IS normalisation range.
    # If OOS volatility far exceeds IS volatility (e.g. COVID crash vs. calm IS
    # period), OOS features are extreme outliers and the HMM may classify all OOS
    # bars into a single state, making regime labels unreliable.
    oos_X = X_full[is_n:]
    if len(oos_X) > 0:
        outlier_mask = np.abs(oos_X).max(axis=1) > 3.0
        outlier_frac = float(outlier_mask.mean())
        if outlier_frac > 0.20:
            log.warning(
                "HMM REGIME RELIABILITY WARNING: %.1f%% of OOS bars have "
                "|feature_z| > 3 (IS normalisation range exceeded). "
                "HMM regime labels may be unreliable for the OOS period. "
                "Consider extending IS window or retraining on a wider period.",
                outlier_frac * 100,
            )

    # --- Fractional differencing (optional) ---
    if frac_diff:
        log.info("Computing fractionally differenced close (d=0.35)...")
        out["frac_diff_close"] = _frac_diff(df["close"], d=0.35)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 — Regime Labelling (ADX + HMM)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Multiple instruments (overrides --instrument)",
    )
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument(
        "--frac-diff",
        action="store_true",
        help="Add fractionally differenced close (d=0.35)",
    )
    parser.add_argument("--hmm-states", type=int, default=2)
    args = parser.parse_args()

    instruments = args.instruments or [args.instrument]
    for inst in instruments:
        log.info("Processing %s %s...", inst, args.timeframe)
        df = _load_ohlcv(inst, args.timeframe)
        labelled = compute_regime_labels(
            df,
            hmm_n_states=args.hmm_states,
            frac_diff=args.frac_diff,
        )
        out_dir = ROOT / ".tmp" / "regime"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inst}_{args.timeframe}_regime.parquet"
        labelled.to_parquet(out_path)
        print(f"  Saved: {out_path.relative_to(ROOT)}")
        print(f"  ADX regime distribution:\n{labelled['adx_regime'].value_counts().to_string()}")
        print(f"  HMM state distribution:\n{labelled['hmm_state'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
