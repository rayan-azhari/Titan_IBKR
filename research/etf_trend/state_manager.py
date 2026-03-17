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


def save_stage1(ma_type: str, fast_ma: int, slow_ma: int, instrument: str = "spy") -> None:
    """Save Stage 1 results (MA type and entry periods)."""
    save_state(
        {"stage1": {"ma_type": ma_type, "fast_ma": fast_ma, "slow_ma": slow_ma}},
        instrument=instrument,
    )


def get_stage1(instrument: str = "spy") -> Optional[tuple[str, int, int]]:
    """Retrieve Stage 1 results. Returns (ma_type, fast_ma, slow_ma) or None."""
    state = load_state(instrument)
    s1 = state.get("stage1")
    if s1:
        return s1.get("ma_type"), s1.get("fast_ma"), s1.get("slow_ma")
    return None


# ── Stage 2: Deceleration signals ──────────────────────────────────────────


def save_stage2(signals: List[str], weights: Dict[str, float], instrument: str = "spy") -> None:
    """Save Stage 2 results (selected decel signals + composite weights)."""
    save_state(
        {"stage2": {"decel_signals": signals, "decel_weights": weights}},
        instrument=instrument,
    )


def get_stage2(instrument: str = "spy") -> Optional[tuple[List[str], Dict[str, float]]]:
    """Retrieve Stage 2 results. Returns (signals, weights) or None."""
    state = load_state(instrument)
    s2 = state.get("stage2")
    if s2:
        return s2.get("decel_signals"), s2.get("decel_weights")
    return None


# ── Stage 3: Exit mode ─────────────────────────────────────────────────────


def save_stage3(exit_mode: str, exit_decel_thresh: float, instrument: str = "spy") -> None:
    """Save Stage 3 results (exit mode + threshold)."""
    save_state(
        {"stage3": {"exit_mode": exit_mode, "exit_decel_thresh": exit_decel_thresh}},
        instrument=instrument,
    )


def get_stage3(instrument: str = "spy") -> Optional[tuple[str, float]]:
    """Retrieve Stage 3 results. Returns (exit_mode, exit_decel_thresh) or None."""
    state = load_state(instrument)
    s3 = state.get("stage3")
    if s3:
        return s3.get("exit_mode"), s3.get("exit_decel_thresh")
    return None


# ── Stage 4: Sizing ────────────────────────────────────────────────────────


def save_stage4(
    sizing_mode: str,
    vol_target: float,
    vol_window: int,
    atr_stop_mult: float,
    instrument: str = "spy",
) -> None:
    """Save Stage 4 results (sizing mode, vol target, ATR stop)."""
    save_state(
        {
            "stage4": {
                "sizing_mode": sizing_mode,
                "vol_target": vol_target,
                "vol_window": vol_window,
                "atr_stop_mult": atr_stop_mult,
            }
        },
        instrument=instrument,
    )


def get_stage4(instrument: str = "spy") -> Optional[Dict[str, Any]]:
    """Retrieve Stage 4 results."""
    state = load_state(instrument)
    return state.get("stage4")
