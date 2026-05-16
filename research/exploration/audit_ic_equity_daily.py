"""ic_equity_daily V3.7 audit (Wave B).

Strategy: RSI(21) deviation mean-reversion on US daily equities. 7 live
tickers (HWM, CB, SYK, NOC, WMT, ABNB, GL). V1 claimed OOS Sharpe
+2.65 to +4.28 with 75-89% win rates — implausibly high for daily equity
MR (real range 0.5-1.5 SR). Same red flag pattern as ic_mtf.

V3.7 audit:
- L21 causality smoke
- L63 V1-style baseline verification (does full-LA reproduce V1?)
- Strict causal IS-only sign fit + IS-only mu/sigma
- L61 7-ticker panel (built into live config; test variability)
- L66 baseline: DAILY_MEAN_REVERSION class -> cash

Signal mechanic:
    rsi_21_dev(t) = RSI(close, 21)(t) - 50
    composite(t) = rsi_21_dev(t) * ic_sign        (ic_sign fit on IS or full)
    z(t) = (composite(t) - mu) / sigma            (mu/sigma on IS or full)
    Entry: z > +threshold; Exit: z <= 0; long-only.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_ic_equity_daily.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ic_equity_daily_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 7 live tickers per ic_equity_daily strategy
TICKERS_LIVE = ["HWM", "CB", "SYK", "NOC", "WMT", "ABNB", "GL"]
THRESHOLDS = {"HWM": 0.25, "CB": 0.50, "SYK": 0.50, "NOC": 0.50,
              "WMT": 0.50, "ABNB": 1.00, "GL": 0.25}
COST_BPS = 1.0
PERIODS_PER_YEAR = 252


def _load_d(ticker: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{ticker}_D.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["close"]].astype(float)


def _rsi(close: pd.Series, period: int = 21) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def ic_equity_returns(
    df: pd.DataFrame,
    *,
    threshold: float,
    is_until_idx: int | None = None,
) -> pd.Series:
    """Per-bar net return of RSI(21) MR signal.

    If is_until_idx is None: full-series mu/sigma + ic_sign (V1-style LA).
    If int: IS-only mu/sigma + ic_sign frozen on first is_until_idx bars.
    """
    close = df["close"]
    rsi = _rsi(close, 21)
    rsi_dev = (rsi - 50.0).fillna(0.0)

    # Fit IC sign on IS slice (or full series for V1-style)
    fwd_ret = np.log(close.shift(-1) / close).fillna(0.0)
    sub_signal = rsi_dev if is_until_idx is None else rsi_dev.iloc[:is_until_idx]
    sub_fwd = fwd_ret if is_until_idx is None else fwd_ret.iloc[:is_until_idx]
    both = pd.concat([sub_signal, sub_fwd], axis=1).dropna()
    if len(both) < 30:
        ic_sign = 1.0
    else:
        r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        ic_sign = float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0

    composite = rsi_dev * ic_sign
    sub_comp = composite if is_until_idx is None else composite.iloc[:is_until_idx]
    mu = float(sub_comp.mean())
    sigma = float(sub_comp.std()) or 1.0
    z = (composite - mu) / sigma

    # Long-only entry: z crosses above +threshold, exit on z <= 0
    pos = np.zeros(len(z), dtype=float)
    state = 0
    arr = z.to_numpy()
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos[i] = float(state)
            continue
        if state == 0 and arr[i] > threshold:
            state = 1
        elif state == 1 and arr[i] <= 0:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=z.index)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS / 10_000.0)
    return (gross - cost).rename("ret")


def main() -> None:
    print("=" * 88)
    print("ic_equity_daily V3.7 audit (Wave B)")
    print("=" * 88)

    print("\n--- L21 + L63 verification: V1-style vs strict-causal per ticker ---")
    print(f"{'ticker':>7} {'thr':>5} {'V1-style_SR':>12} {'strict_OOS_SR':>14} {'gap':>8} {'n':>6}")
    v1_style_sharpes = {}
    strict_sharpes = {}
    strict_returns_by_ticker = {}
    for tkr in TICKERS_LIVE:
        try:
            df = _load_d(tkr)
            n = len(df)
            is_split = n // 2

            # V1-style: full-series ic_sign + mu/sigma (LA permitted)
            ret_v1 = ic_equity_returns(df, threshold=THRESHOLDS[tkr], is_until_idx=None)
            sr_v1 = float(sharpe(ret_v1, periods_per_year=PERIODS_PER_YEAR))

            # Strict: IS-only sign + IS-only mu/sigma, report on OOS half
            ret_strict_full = ic_equity_returns(df, threshold=THRESHOLDS[tkr], is_until_idx=is_split)
            ret_strict_oos = ret_strict_full.iloc[is_split:]
            sr_strict = float(sharpe(ret_strict_oos, periods_per_year=PERIODS_PER_YEAR))

            gap = sr_v1 - sr_strict
            v1_style_sharpes[tkr] = sr_v1
            strict_sharpes[tkr] = sr_strict
            strict_returns_by_ticker[tkr] = ret_strict_oos
            print(f"{tkr:>7s} {THRESHOLDS[tkr]:>5.2f} {sr_v1:>+11.4f} {sr_strict:>+13.4f} {gap:>+7.3f} {len(df):>6d}")
        except FileNotFoundError:
            print(f"{tkr:>7s}: data missing")

    if not strict_sharpes:
        print("\nVERDICT: RETIRE (no data)")
        return

    valid_strict = list(strict_sharpes.values())
    valid_v1 = list(v1_style_sharpes.values())
    print(f"\n  V1-style panel median:    {np.median(valid_v1):+.4f}")
    print(f"  V1-style panel mean:      {np.mean(valid_v1):+.4f}")
    print(f"  STRICT OOS panel median:  {np.median(valid_strict):+.4f}")
    print(f"  STRICT OOS panel mean:    {np.mean(valid_strict):+.4f}")
    print(f"  pct of tickers strict SR > 0: {np.mean([s > 0 for s in valid_strict]):.0%}")

    # L66 baseline: SR > 0 (cash) per ticker, panel median
    pct_pos = float(np.mean([s > 0 for s in valid_strict]))
    panel_med = float(np.median(valid_strict))
    print("\n--- L66 baseline (cash): panel median strict SR > 0 ---")
    print(f"  panel median = {panel_med:+.4f}  {'PASS' if panel_med > 0 else 'FAIL'}")

    # V1 vs V3.6/V3.7 gap analysis (L62 classification)
    print("\n--- L62 Sharpe-gap classification ---")
    v1_claim_sharpes = [4.28, 3.41, 3.24, 3.06, 2.82, 2.78, 2.65]
    if len(v1_claim_sharpes) == len(valid_strict):
        avg_gap = float(np.mean([c - s for c, s in zip(v1_claim_sharpes, valid_strict, strict=True)]))
        print(f"  Mean gap (V1 claim minus V3.6 strict OOS): {avg_gap:+.2f}")
        if avg_gap > 5.0:
            cls = "Multi-TF/multi-signal L21 amplification OR fabrication boundary"
        elif avg_gap > 3.0:
            cls = "Multiple compounding bugs or cherry-pick"
        elif avg_gap > 1.5:
            cls = "Single look-ahead bug class"
        else:
            cls = "Methodology drift"
        print(f"  L62 classification: {cls}")

    # Verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if panel_med <= 0:
        print(f"RETIRE: panel median strict OOS SR {panel_med:+.4f} fails baseline (L66). "
              f"Likely L21 look-ahead bug (same pattern as ic_mtf).")
    elif pct_pos >= 0.5 and panel_med > 0:
        print(f"CONDITIONAL_WATCHPOINT candidate: {pct_pos:.0%} positive, median {panel_med:+.4f}.")
    else:
        print(f"MARGINAL: panel median {panel_med:+.4f}, {pct_pos:.0%} positive. Defer.")


if __name__ == "__main__":
    main()
