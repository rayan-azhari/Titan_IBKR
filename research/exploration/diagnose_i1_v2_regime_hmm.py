"""I1 v2 -- pre-pre-reg diagnostic: does a multi-feature HMM on the regime panel
yield non-degenerate regime states (i.e., escape L51)?

L51 v1 failure: HMM on raw per-asset daily returns produced a near-uniform state
distribution (one state ~90%+), so the gate degenerated to a no-op.

v2 hypothesis: a multi-feature HMM on the 7-feature regime panel
(`data/i1_regime_panel.parquet`) will identify meaningful regime states because
the feature panel captures the actual structural variables (vol, credit, dollar,
drawdown velocity) that drive market regime, not just price.

Diagnostic checks (all IS-only, last 12 months held out):
    1. Fit 2-state, 3-state, 4-state Gaussian HMMs.
    2. Per model, report state-occupation distribution (should NOT be ~uniform
       with one state >85%).
    3. Per model, report mean feature vector per state (should be
       economically interpretable, e.g., one state = high-VIX + high-credit-
       spread + drawdown = "crisis").
    4. Report state-transition flickering (mean run-length per state; <5d
       = noisy gate, >50d = stable regime).
    5. Seed-stability check (2 different seeds, Hamming distance between
       state paths < 20%).

If 2-state or 3-state passes the non-degeneracy + stability gates, v2 is worth
a pre-reg. If all degenerate, the regime panel itself is insufficient and we
need a different approach.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/diagnose_i1_v2_regime_hmm.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PANEL_FP = PROJECT_ROOT / "data" / "i1_regime_panel.parquet"
SANCTUARY_MONTHS = 12
SEEDS = [42, 11]
N_STATES_GRID = [2, 3, 4]


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_FP)
    df = df.dropna(how="any")
    return df.sort_index()


def fit_hmm_panel(features: np.ndarray, n_states: int, seed: int) -> hmm.GaussianHMM | None:
    if features.shape[0] < n_states * 60:
        return None
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(model.transmat_)):
        return None
    return model


def state_run_lengths(states: np.ndarray) -> dict[int, float]:
    """Mean run-length per state (consecutive-day occupation)."""
    if len(states) == 0:
        return {}
    out: dict[int, list[int]] = {}
    cur = int(states[0])
    run = 1
    for s in states[1:]:
        if s == cur:
            run += 1
        else:
            out.setdefault(cur, []).append(run)
            cur = int(s)
            run = 1
    out.setdefault(cur, []).append(run)
    return {st: float(np.mean(rl)) for st, rl in out.items()}


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of bars where labels differ (after optimal relabel)."""
    n_states = max(int(a.max()), int(b.max())) + 1
    # Try all label permutations of b to match a; report best hamming.
    from itertools import permutations
    best = 1.0
    for perm in permutations(range(n_states)):
        b_relabeled = np.array([perm[s] for s in b])
        diff = float((a != b_relabeled).mean())
        if diff < best:
            best = diff
    return best


def main() -> None:
    print("=" * 84)
    print("I1 v2 -- pre-pre-reg regime HMM diagnostic")
    print("=" * 84)
    panel = load_panel()
    print(f"\nPanel: {panel.shape[0]} days x {panel.shape[1]} features")
    print(f"  range: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"  features: {list(panel.columns)}")

    cutoff = panel.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_panel = panel.loc[:cutoff]
    sanc_panel = panel.loc[cutoff:]
    print(f"\nIS slice:      {len(is_panel)} bars ({is_panel.index[0].date()} -> {is_panel.index[-1].date()})")
    print(f"Sanctuary:     {len(sanc_panel)} bars ({sanc_panel.index[0].date()} -> {sanc_panel.index[-1].date()}) [HELD OUT]")

    X_is = is_panel.to_numpy()
    feature_names = list(panel.columns)

    for n_states in N_STATES_GRID:
        print("\n" + "-" * 84)
        print(f"  N_STATES = {n_states}")
        print("-" * 84)

        models = {}
        state_paths = {}
        for seed in SEEDS:
            mdl = fit_hmm_panel(X_is, n_states=n_states, seed=seed)
            if mdl is None:
                print(f"  [seed={seed}] FIT FAILED")
                continue
            models[seed] = mdl
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # predict() is Viterbi; for diagnostic on IS only this is fine
                # (no causality concern because we're not building a live gate).
                state_paths[seed] = mdl.predict(X_is)

        if SEEDS[0] not in models:
            print("  [skip] primary seed failed")
            continue

        mdl = models[SEEDS[0]]
        states = state_paths[SEEDS[0]]

        # 1. State occupation distribution
        occ = pd.Series(states).value_counts(normalize=True).sort_index()
        print(f"  State occupation: {occ.round(3).to_dict()}")
        max_occ = float(occ.max())
        if max_occ > 0.85:
            print(f"  [WARN] state {occ.idxmax()} dominates {max_occ:.1%} -- borderline L51 no-op")
        elif max_occ > 0.65:
            print(f"  [info] state {occ.idxmax()} occupies {max_occ:.1%} -- skewed but OK")
        else:
            print("  [OK] all states have <65% occupation -- balanced regimes")

        # 2. Per-state mean feature vector (economic interpretation)
        print("  State mean feature values (z-scores, except spy_above_sma200 binary):")
        means_df = pd.DataFrame(mdl.means_, columns=feature_names).round(3)
        print(means_df.to_string())

        # 3. Run-length per state (regime stability)
        rl = state_run_lengths(states)
        print(f"  Mean run-length per state (days): {rl}")
        too_flickery = [s for s, r in rl.items() if r < 5]
        if too_flickery:
            print(f"  [WARN] states {too_flickery} flicker (<5d mean run-length)")
        else:
            print("  [OK] all states have stable runs (>=5d)")

        # 4. Seed stability
        if SEEDS[1] in models:
            stab = 1.0 - hamming(state_paths[SEEDS[0]], state_paths[SEEDS[1]])
            print(f"  Seed stability (1 - hamming after relabel): {stab:.3f}  "
                  f"({'OK' if stab > 0.80 else 'WARN'})")
        else:
            print("  Seed stability: alt seed failed")

        # 5. Transition matrix snapshot (diagonal = stay-prob)
        diag = np.diag(mdl.transmat_)
        print(f"  Self-transition probs (diag): {[round(float(d), 3) for d in diag]}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    print("""
If at least one N_STATES has:
  - max occupation <= 0.65
  - all run-lengths >= 5 days
  - seed stability >= 0.80
then the regime panel admits meaningful HMM regimes -> proceed with I1 v2 pre-reg.
If all configurations degenerate (max occupation > 0.85), the panel still
isn't enough; need to revisit feature engineering.
""")


if __name__ == "__main__":
    main()
