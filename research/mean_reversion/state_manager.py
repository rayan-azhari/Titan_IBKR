"""state_manager.py — Persist research results between pipeline stages.

State is stored in .tmp/mr_state_eurusd.json.  Each stage saves its best
parameters so subsequent stages can load them without re-running the sweep.

Pattern mirrors research/mtf/state_manager.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = PROJECT_ROOT / ".tmp"
STATE_FILE = STATE_DIR / "mr_state_eurusd.json"


def load_state() -> dict[str, Any]:
    """Load the full state dict from disk."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_state(updates: dict[str, Any]) -> None:
    """Merge updates into the existing state and persist to disk."""
    current = load_state()
    current.update(updates)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(current, f, indent=4)
    print(f"[State] Updated {STATE_FILE.name}: {list(updates.keys())}")


# ---------------------------------------------------------------------------
# Stage-specific helpers
# ---------------------------------------------------------------------------


def save_stage1(
    method: str,
    vwap_window: int,
    best_pct_window: int,
    best_pct: float,
    sharpe: float,
) -> None:
    """Save Stage 1 best signal params."""
    save_state({
        "stage1": {
            "method": method,
            "vwap_window": vwap_window,
            "best_pct_window": best_pct_window,
            "best_tier1_pct": best_pct,
            "sharpe": sharpe,
        }
    })


def get_stage1() -> dict[str, Any] | None:
    """Retrieve Stage 1 results."""
    return load_state().get("stage1")


def save_stage2(
    ranging_state_idx: int,
    p_thresh: float,
    hurst_thresh: float,
    sharpe_lift: float,
    hmm_model_path: str,
) -> None:
    """Save Stage 2 regime filter params."""
    save_state({
        "stage2": {
            "ranging_state_idx": ranging_state_idx,
            "p_thresh": p_thresh,
            "hurst_thresh": hurst_thresh,
            "sharpe_lift": sharpe_lift,
            "hmm_model_path": hmm_model_path,
        }
    })


def get_stage2() -> dict[str, Any] | None:
    """Retrieve Stage 2 results."""
    return load_state().get("stage2")


def save_stage3(
    is_sharpe: float,
    oos_sharpe: float,
    oos_is_ratio: float,
    win_rate: float,
    max_dd: float,
    n_oos_trades: int,
    gates_passed: bool,
    gate_results: dict[str, bool],
) -> None:
    """Save Stage 3 validation results."""
    save_state({
        "stage3": {
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_is_ratio": oos_is_ratio,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "n_oos_trades": n_oos_trades,
            "all_gates_passed": gates_passed,
            "gate_results": gate_results,
        }
    })


def get_stage3() -> dict[str, Any] | None:
    """Retrieve Stage 3 results."""
    return load_state().get("stage3")
