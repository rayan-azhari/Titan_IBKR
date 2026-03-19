"""State manager for ETF Trend Optimization Workflow.

Handles persistence of optimization results between stages using per-instrument JSON files.
Each instrument gets its own state file: .tmp/etf_trend_state_{instrument_lower}.json

Modelled on research/mtf/state_manager.py.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = PROJECT_ROOT / ".tmp"


def _state_file(instrument: str = "spy") -> Path:
    inst_lower = instrument.lower().replace(".", "_").replace("-", "_")
    return STATE_DIR / f"etf_trend_state_{inst_lower}.json"


def load_state(instrument: str = "spy") -> Dict[str, Any]:
    """Load the current optimization state for an instrument."""
    path = _state_file(instrument)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_state(updates: Dict[str, Any], instrument: str = "spy") -> None:
    """Update and save the optimization state for an instrument."""
    current = load_state(instrument)
    current.update(updates)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_file(instrument)
    with open(path, "w") as f:
        json.dump(current, f, indent=4)
    print(f"[State] Updated {path.name}: {list(updates.keys())}")


# ── Stage 1: MA type + periods ─────────────────────────────────────────────


def save_stage1(
    ma_type: str,
    slow_ma: int,
    instrument: str = "spy",
    fast_ma: Optional[int] = None,
) -> None:
    """Save Stage 1 results (MA type, slow period, optional fast period)."""
    save_state(
        {"stage1": {"ma_type": ma_type, "slow_ma": slow_ma, "fast_ma": fast_ma}},
        instrument=instrument,
    )


def get_stage1(instrument: str = "spy") -> Optional[tuple[str, int, Optional[int]]]:
    """Retrieve Stage 1 results.

    Returns (ma_type, slow_ma, fast_ma) or None.
    fast_ma is None when entry_mode is slow_only.
    """
    state = load_state(instrument)
    s1 = state.get("stage1")
    if s1:
        return s1.get("ma_type"), s1.get("slow_ma"), s1.get("fast_ma")
    return None


# ── Stage 2: Deceleration signals ──────────────────────────────────────────


def save_stage2(
    signals: List[str],
    weights: Dict[str, float],
    instrument: str = "spy",
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Save Stage 2 results (selected decel signals, composite weights, signal params)."""
    save_state(
        {
            "stage2": {
                "decel_signals": signals,
                "decel_weights": weights,
                "decel_params": params or {},
            }
        },
        instrument=instrument,
    )


def get_stage2(
    instrument: str = "spy",
) -> Optional[tuple[List[str], Dict[str, float], Dict[str, Any]]]:
    """Retrieve Stage 2 results.

    Returns (signals, weights, params) or None.
    params holds winning signal hyperparameters (d_pct_smooth, rv_window, macd_fast).
    """
    state = load_state(instrument)
    s2 = state.get("stage2")
    if s2:
        return s2.get("decel_signals"), s2.get("decel_weights"), s2.get("decel_params", {})
    return None


# ── Stage 3: Exit mode ─────────────────────────────────────────────────────


def save_stage3(
    exit_mode: str,
    exit_decel_thresh: float,
    entry_mode: str = "decel_positive",
    fast_reentry_ma: Optional[int] = None,
    exit_confirm_days: int = 1,
    decel_confirm_days: int = 1,
    instrument: str = "spy",
) -> None:
    """Save Stage 3 results (exit mode, threshold, entry mode, optional fast re-entry MA,
    SMA confirmation days, and decel confirmation days)."""
    save_state(
        {
            "stage3": {
                "exit_mode": exit_mode,
                "exit_decel_thresh": exit_decel_thresh,
                "entry_mode": entry_mode,
                "fast_reentry_ma": fast_reentry_ma,
                "exit_confirm_days": exit_confirm_days,
                "decel_confirm_days": decel_confirm_days,
            }
        },
        instrument=instrument,
    )


def get_stage3(
    instrument: str = "spy",
) -> Optional[tuple[str, float, str, Optional[int], int, int]]:
    """Retrieve Stage 3 results.

    Returns (exit_mode, exit_decel_thresh, entry_mode, fast_reentry_ma,
    exit_confirm_days, decel_confirm_days) or None.
    fast_reentry_ma is None unless entry_mode == 'asymmetric'.
    """
    state = load_state(instrument)
    s3 = state.get("stage3")
    if s3:
        return (
            s3.get("exit_mode"),
            s3.get("exit_decel_thresh"),
            s3.get("entry_mode", "decel_positive"),
            s3.get("fast_reentry_ma"),
            int(s3.get("exit_confirm_days", 1)),
            int(s3.get("decel_confirm_days", 1)),
        )
    return None


# ── Stage 4: Sizing ────────────────────────────────────────────────────────


def save_stage4(
    sizing_mode: str,
    atr_stop_mult: float,
    instrument: str = "spy",
    vol_target: float = 0.15,
    max_leverage: float = 1.0,
) -> None:
    """Save Stage 4 results (sizing mode, ATR stop multiplier, vol-target params)."""
    save_state(
        {
            "stage4": {
                "sizing_mode": sizing_mode,
                "atr_stop_mult": atr_stop_mult,
                "vol_target": vol_target,
                "max_leverage": max_leverage,
            }
        },
        instrument=instrument,
    )


def get_stage4(instrument: str = "spy") -> Optional[Dict[str, Any]]:
    """Retrieve Stage 4 results."""
    state = load_state(instrument)
    return state.get("stage4")
