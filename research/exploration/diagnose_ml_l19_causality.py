"""ml Wave C -- L19 same-bar look-ahead causality diagnostic.

Verifies whether the live ml strategy (titan/strategies/ml/{features,strategy}.py)
exhibits the L19 same-bar look-ahead bug flagged in `directives/V1-era Re-audit
Sweep Roster 2026-05-16.md`.

L19 same-bar look-ahead pattern:
    features[t] use info from close[t]
    labels[t]   pair with predicted_return[t]   (e.g., close[t+1] - close[t])
    strategy executes at close[t] earning return[t]   (close[t] - close[t-1])
                                                       ^^^^^ leak: features[t] knew this

If feature[t] -> predict label[t] -> act at close[t] earning return[t-1->t],
that's L19. Correct: act at close[t] earning return[t->t+1].

Test:
    1. Take EUR/USD H4 data.
    2. Build features at every row.
    3. Corrupt close[T] (replace with 100x). Re-build features.
    4. Check: do features at row T+1 differ from baseline? They MUST (they use close[T]).
       Do features at row T-1 differ? They MUST NOT (no look-ahead).
    5. Repeat: build_tbm_labels then verify label[t] depends only on close[t+1..t+24],
       not on close[t-1].

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/diagnose_ml_l19_causality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.strategies.ml.features import build_features, load_feature_config  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"


def _load_h4_eurusd() -> pd.DataFrame:
    fp = DATA_DIR / "EUR_USD_H4.parquet"
    if not fp.exists():
        # Fallback to H1.
        fp = DATA_DIR / "EUR_USD_H1.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"]).tail(2000)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def test_features_causal(df: pd.DataFrame) -> dict:
    """Verify build_features() is causal: corrupting close[T] should NOT
    change features at row < T.
    """
    feat_cfg = load_feature_config()
    baseline = build_features(df, context_data={}, cfg=feat_cfg)

    # Corrupt the last 100 bars: multiply close by 100.
    n_corrupt = 100
    df_c = df.copy()
    df_c.iloc[-n_corrupt:, df_c.columns.get_loc("close")] *= 100.0
    df_c.iloc[-n_corrupt:, df_c.columns.get_loc("high")] *= 100.0
    df_c.iloc[-n_corrupt:, df_c.columns.get_loc("low")] *= 100.0
    perturbed = build_features(df_c, context_data={}, cfg=feat_cfg)

    cutoff = len(df) - n_corrupt - 5
    # Compare features at rows [0..cutoff). They must be bit-exact.
    base_past = baseline.iloc[:cutoff]
    pert_past = perturbed.iloc[:cutoff]
    common_cols = sorted(set(base_past.columns).intersection(pert_past.columns))

    diffs = {}
    for col in common_cols:
        a = base_past[col].dropna()
        b = pert_past[col].dropna()
        common_idx = a.index.intersection(b.index)
        if len(common_idx) == 0:
            continue
        max_abs_diff = float((a.reindex(common_idx) - b.reindex(common_idx)).abs().max())
        diffs[col] = max_abs_diff
    return diffs


def test_labels_causal(df: pd.DataFrame) -> dict:
    """Verify TBM labels depend only on future bars.

    Corrupt close[T-5]. Check label[T-5] is unchanged (it depends on
    close[T-4..T-5+24]; close[T-5] itself only sets the barrier offsets,
    not the bars examined for breaches).

    Actually for L19 specifically: label[t] = f(close[t], close[t+1..t+24]).
    The barrier OFFSETS depend on close[t] + ATR[t-13..t]; the BREACH
    detection depends on high[t+1..t+24] / low[t+1..t+24]. So label[t]
    legitimately uses close[t] but NOT close[t-1] for breach checks.

    Test: corrupt close[T]. Label[T] is allowed to change (barriers shift).
    But label[t < T] must NOT change because it doesn't use close[T].
    """
    try:
        from research.ml.build_tbm_labels import _tbm_kernel
    except ImportError:
        return {"error": "_tbm_kernel not importable"}

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    # Simple ATR proxy: 14-bar rolling true range.
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean().to_numpy()

    labels_base = _tbm_kernel(close, high, low, atr, 2.0, 1.0, 24)

    # Corrupt close[T] = 100x for the last 50 bars.
    n_corrupt = 50
    close_c = close.copy()
    high_c = high.copy()
    low_c = low.copy()
    close_c[-n_corrupt:] *= 100.0
    high_c[-n_corrupt:] *= 100.0
    low_c[-n_corrupt:] *= 100.0
    # ATR is computed from close.shift(1) + high + low; corrupting last 50 affects
    # ATR's last (50 + 13) bars. Use the perturbed ATR.
    tr_c = pd.concat(
        [
            (pd.Series(high_c) - pd.Series(low_c)),
            (pd.Series(high_c) - pd.Series(close_c).shift(1)).abs(),
            (pd.Series(low_c) - pd.Series(close_c).shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_c = tr_c.rolling(14, min_periods=14).mean().to_numpy()
    labels_pert = _tbm_kernel(close_c, high_c, low_c, atr_c, 2.0, 1.0, 24)

    # Labels at index < (n - n_corrupt - 24 - 14) should be unchanged (24 is
    # forward-looking window, 14 is ATR backward window).
    cutoff = len(close) - n_corrupt - 24 - 14
    base_past = labels_base[:cutoff]
    pert_past = labels_pert[:cutoff]
    n_diff = int((base_past != pert_past).sum())
    return {
        "n_compared": int(len(base_past)),
        "n_different": n_diff,
        "fraction_different": float(n_diff / max(len(base_past), 1)),
        "verdict": "PASS" if n_diff == 0 else "FAIL",
    }


def test_strategy_inference_alignment() -> str:
    """Inspect titan/strategies/ml/strategy.py for the inference-execution
    alignment. Returns a textual verdict on the L19 risk.

    The bug pattern to flag:
        - features computed using close[t]
        - prediction at close[t] is for "next bar return"
        - strategy enters order AT close[t]; order earns return[t]->[t+1]
        - This is correct IF training labels are also forward-looking.

        - features computed using close[t]
        - prediction at close[t] is for "current bar return"
        - strategy enters AT close[t]; earns return[t-1]->[t] (already past!)
        - This is the L19 bug -- features knew the return that they're "predicting".
    """
    strat_fp = PROJECT_ROOT / "titan" / "strategies" / "ml" / "strategy.py"
    if not strat_fp.exists():
        return "strategy.py NOT FOUND"
    src = strat_fp.read_text(encoding="utf-8")
    if "predict_proba" in src and "self.model.predict(latest_features)" in src:
        # Look for the comment about training-time label alignment.
        flag = "training labels must be forward-looking"
        has_doc = (
            "L19" in src
            or "look-ahead" in src.lower()
            or "lookahead" in src.lower()
            or "same-bar" in src.lower()
        )
        if has_doc:
            return (
                "Strategy inference uses latest_features = X.iloc[[-1]] (close[t]) "
                "and predicts to act at close[t]. Documentation references "
                "look-ahead discipline."
            )
        return (
            "Strategy inference uses latest_features = X.iloc[[-1]] (close[t]) "
            "and predicts to act at close[t]. NO explicit L19 / look-ahead "
            "comment found. Requires checking training-time label alignment "
            f"to confirm forward-looking (i.e., {flag})."
        )
    return "strategy.py predict() pattern not recognised"


def main() -> None:
    print("=" * 84)
    print("ml Wave C -- L19 same-bar look-ahead causality diagnostic")
    print("=" * 84)

    df = _load_h4_eurusd()
    print(f"\nData: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print("\n[1] build_features() causality:")
    diffs = test_features_causal(df)
    finite_diffs = {k: v for k, v in diffs.items() if np.isfinite(v)}
    if finite_diffs:
        max_diff_col = max(finite_diffs, key=finite_diffs.get)
        print(f"    {len(finite_diffs)} feature columns checked")
        print(f"    max abs-diff column: {max_diff_col} = {finite_diffs[max_diff_col]:.6e}")
        any_leaks = any(v > 1e-9 for v in finite_diffs.values())
        if any_leaks:
            print("    [FAIL] Some feature columns leak future-bar info into past rows.")
            leaks = sorted(
                [(k, v) for k, v in finite_diffs.items() if v > 1e-9], key=lambda kv: -kv[1]
            )[:10]
            for col, d in leaks:
                print(f"      - {col}: {d:.6e}")
        else:
            print("    [PASS] all feature columns are causal.")
    else:
        print("    no comparable features (empty diff dict)")

    print("\n[2] TBM label causality (build_tbm_labels._tbm_kernel):")
    tbm = test_labels_causal(df)
    if "error" in tbm:
        print(f"    [skip] {tbm['error']}")
    else:
        print(
            f"    {tbm['n_different']} / {tbm['n_compared']} past labels changed "
            f"after corrupting last 50 close bars  -- {tbm['verdict']}"
        )

    print("\n[3] Strategy inference alignment:")
    print(f"    {test_strategy_inference_alignment()}")

    print("\n" + "=" * 84)
    print("INTERPRETATION")
    print("=" * 84)
    print("""
- If [1] PASS and [2] PASS: feature & label computations are causal in isolation.
  The L19 risk (if any) lies in the training-time PAIRING: features[t] paired
  with labels[t] is correct ONLY if labels[t] are constructed from FUTURE bars
  (which build_tbm_labels does via the forward 24-bar window). Then live
  inference at close[t] predicting the next-bar trade is consistent.

- If [1] FAIL: there's an explicit feature look-ahead. Fix the corrupted columns
  before re-training.

- If [2] FAIL: there's a label look-back (label[t] depends on close[t-1] or
  earlier), which means the model would be predicting an already-resolved
  return. This is the strict L19 pattern.

- The L19 fix in either case is to .shift(1) the features so feat[t] uses
  close[t-1] and predict the move from close[t]->close[t+1]. The current
  build_features uses no .shift(1), so the live alignment relies on
  *forward-looking labels* in training. Verify the training notebook / script
  builds labels from close[t+1..t+24] (TBM does so by construction).
""")


if __name__ == "__main__":
    main()
