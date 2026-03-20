"""Swap Curve Cost Model

Provides historical time-varying swap values for accurate backtesting.
Falls back to static values if historical data is unavailable.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SWAP_DATA_DIR = ROOT / "data" / "rates"

def load_swap_curve(
    instrument: str,
    date_index: pd.DatetimeIndex,
    fallback_long_pips: float,
    fallback_short_pips: float,
) -> pd.DataFrame:
    """Return DataFrame with [swap_long_pips, swap_short_pips] aligned to date_index.
    
    If data/rates/{instrument}_swap.parquet exists, it uses the historical time series.
    Otherwise, returns a DataFrame filled with the static fallback values to ensure
    backward compatibility during the backtest.
    """
    swap_df = pd.DataFrame(index=date_index)
    
    data_path = SWAP_DATA_DIR / f"{instrument}_swap.parquet"
    if data_path.exists():
        try:
            curve = pd.read_parquet(data_path)
            # Forward fill missing dates from broker
            curve = curve.reindex(date_index).ffill()
            
            # Fill remaining NaNs at the beginning with fallbacks
            if "swap_long_pips" in curve.columns:
                swap_df["swap_long_pips"] = curve["swap_long_pips"].fillna(fallback_long_pips)
            else:
                swap_df["swap_long_pips"] = fallback_long_pips
                
            if "swap_short_pips" in curve.columns:
                swap_df["swap_short_pips"] = curve["swap_short_pips"].fillna(fallback_short_pips)
            else:
                swap_df["swap_short_pips"] = fallback_short_pips
                
            logger.info("Loaded historical swap curve for %s", instrument)
            return swap_df
        except Exception as exc:
            logger.warning("Error loading swap curve %s, using static fallbacks: %s", data_path, exc)

    # Fallback to static values
    swap_df["swap_long_pips"] = fallback_long_pips
    swap_df["swap_short_pips"] = fallback_short_pips
    
    return swap_df
