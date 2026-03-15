"""State manager for MTF Optimization Workflow.

Handles persistence of optimization results between stages using per-pair JSON files.

Each pair gets its own state file: .tmp/mtf_state_{pair_lower}.json
Backward compat: default pair="eurusd" preserves original .tmp/mtf_state_eurusd.json
(callers that used the old .tmp/mtf_state.json will still work via the eurusd default).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = PROJECT_ROOT / ".tmp"


def _state_file(pair: str = "eurusd") -> Path:
    pair_lower = pair.lower().replace("_", "")
    return STATE_DIR / f"mtf_state_{pair_lower}.json"


def load_state(pair: str = "eurusd") -> Dict[str, Any]:
    """Load the current optimization state for a pair."""
    path = _state_file(pair)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_state(updates: Dict[str, Any], pair: str = "eurusd") -> None:
    """Update and save the optimization state for a pair."""
    current = load_state(pair)
    current.update(updates)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_file(pair)
    with open(path, "w") as f:
        json.dump(current, f, indent=4)
    print(f"[State] Updated {path.name}: {list(updates.keys())}")


def save_stage1(ma_type: str, threshold: float, pair: str = "eurusd") -> None:
    """Save Stage 1 results (Global Params)."""
    save_state({"stage1": {"ma_type": ma_type, "threshold": threshold}}, pair=pair)


def get_stage1(pair: str = "eurusd") -> Optional[Tuple[str, float]]:
    """Retrieve Stage 1 results. Returns (ma_type, threshold) or None."""
    state = load_state(pair)
    s1 = state.get("stage1")
    if s1:
        return s1.get("ma_type"), s1.get("threshold")
    return None


def save_stage2(weights: Dict[str, float], pair: str = "eurusd") -> None:
    """Save Stage 2 results (Weights)."""
    save_state({"stage2": {"weights": weights}}, pair=pair)


def get_stage2(pair: str = "eurusd") -> Optional[Dict[str, float]]:
    """Retrieve Stage 2 results (Weights)."""
    state = load_state(pair)
    s2 = state.get("stage2")
    return s2.get("weights") if s2 else None


def save_stage3(params: Dict[str, Any], pair: str = "eurusd") -> None:
    """Save Stage 3 results (Indicator Params)."""
    save_state({"stage3": {"params": params}}, pair=pair)


def save_stage4(atr_mult: float, pair: str = "eurusd") -> None:
    """Save Stage 4 results (ATR multiplier)."""
    save_state({"stage4": {"atr_stop_mult": atr_mult}}, pair=pair)


def get_stage4(pair: str = "eurusd") -> Optional[float]:
    """Retrieve Stage 4 ATR multiplier."""
    state = load_state(pair)
    s4 = state.get("stage4")
    return s4.get("atr_stop_mult") if s4 else None
