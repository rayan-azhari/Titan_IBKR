"""IC validation for regime indicators.

For each candidate indicator, compute its Spearman rank correlation with
forward 21-day SPY returns (standard IC) plus rolling 252-bar ICIR. Apply
Benjamini-Hochberg FDR correction for multiple testing.

This is the project's standard pattern (see directives/IC Signal Analysis.md).
A regime indicator that doesn't predict forward returns at the 21-day
horizon shouldn't enter the regime-score ensemble.

Note: regime indicators are designed for *risk classification*, not pure
return prediction. So we expect smaller |IC| than for traditional alpha
signals (target |IC| > 0.03 rather than > 0.05). What matters more is
ICIR consistency and survival of BH FDR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forward_log_return(
    close: pd.Series, *, horizon: int, vol_window: int = 20, vol_adjust: bool = True
) -> pd.Series:
    """Volatility-adjusted forward log return at horizon h.

    Same convention as the IC pipeline at directives/IC Signal Analysis.md:
    raw forward log return divided by rolling realised vol times sqrt(h).
    """
    log_ret_1d = np.log(close / close.shift(1))
    raw_fwd = np.log(close.shift(-horizon) / close)
    if not vol_adjust:
        return raw_fwd
    vol = log_ret_1d.rolling(vol_window).std() * np.sqrt(horizon)
    return raw_fwd / vol.replace(0.0, np.nan)


def spearman_ic(signal: pd.Series, fwd_return: pd.Series) -> tuple[float, float, int]:
    """Spearman rank correlation + p-value + sample size on aligned non-NaN bars."""
    df = pd.concat([signal, fwd_return], axis=1, join="inner").dropna()
    if len(df) < 30:
        return (float("nan"), float("nan"), len(df))
    rho, pval = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return (float(rho), float(pval), len(df))


def rolling_spearman_ic(
    signal: pd.Series, fwd_return: pd.Series, *, window: int = 252
) -> pd.Series:
    """Rolling Spearman IC over `window` bars. Returns Series indexed at the right edge.

    Skips windows where either input is constant (correlation undefined).
    """
    df = pd.concat([signal, fwd_return], axis=1, join="inner").dropna()
    if len(df) < window + 10:
        return pd.Series(dtype=float)
    arr_s = df.iloc[:, 0].values
    arr_f = df.iloc[:, 1].values
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(window, n):
        s_win = arr_s[i - window : i]
        f_win = arr_f[i - window : i]
        if s_win.std() < 1e-12 or f_win.std() < 1e-12:
            continue
        rho, _ = stats.spearmanr(s_win, f_win)
        out[i] = rho
    return pd.Series(out, index=df.index)


def icir(rolling_ic: pd.Series) -> float:
    """Information ratio of the rolling-IC series. Returns 0.0 if too few bars."""
    s = rolling_ic.dropna()
    if len(s) < 30:
        return 0.0
    sd = float(s.std())
    if sd < 1e-12:
        return 0.0
    return float(s.mean() / sd)


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of booleans (True = significant)."""
    pvals_arr = np.array(pvals, dtype=float)
    n = len(pvals_arr)
    order = np.argsort(pvals_arr)
    ranked = pvals_arr[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passes = ranked <= thresholds
    if not passes.any():
        return [False] * n
    cutoff_rank = np.max(np.where(passes)[0])
    cutoff_p = ranked[cutoff_rank]
    return [bool(p <= cutoff_p) for p in pvals_arr]


def regime_conditional_metrics(
    signal: pd.Series,
    spy_close: pd.Series,
    *,
    horizon: int = 21,
    benign_threshold: float = 0.7,
    hostile_threshold: float = 0.3,
) -> dict:
    """Conditional metrics — the right validator for regime indicators.

    Slices forward 21d returns by indicator state (benign vs hostile) and
    measures whether the regime classification produces meaningfully
    different forward distributions:

    * conditional_mean — does benign predict positive mean return?
    * conditional_vol — does hostile predict higher realised vol?
    * conditional_sharpe — risk-adjusted return per regime
    * crash_hit_rate — when hostile fires, fraction followed by a 5%+ DD
      within ``horizon`` days
    * mean_diff_t — t-statistic of difference in means (benign vs hostile)
    """
    fwd_raw = (spy_close.shift(-horizon) / spy_close - 1.0).rename("fwd_ret")
    df = pd.concat([signal.rename("sig"), fwd_raw], axis=1).dropna()
    if len(df) < 100:
        return {"error": "insufficient data"}

    benign_mask = df["sig"] >= benign_threshold
    hostile_mask = df["sig"] <= hostile_threshold

    out: dict = {
        "n_benign": int(benign_mask.sum()),
        "n_hostile": int(hostile_mask.sum()),
        "n_total": len(df),
        "frac_benign": round(float(benign_mask.mean()), 3),
        "frac_hostile": round(float(hostile_mask.mean()), 3),
    }

    if benign_mask.sum() < 30 or hostile_mask.sum() < 30:
        out["error"] = "insufficient regime samples"
        return out

    benign_ret = df.loc[benign_mask, "fwd_ret"]
    hostile_ret = df.loc[hostile_mask, "fwd_ret"]

    out["benign_mean"] = round(float(benign_ret.mean()), 5)
    out["hostile_mean"] = round(float(hostile_ret.mean()), 5)
    out["benign_vol"] = round(float(benign_ret.std()), 5)
    out["hostile_vol"] = round(float(hostile_ret.std()), 5)
    out["benign_sharpe_h"] = (
        round(float(benign_ret.mean() / benign_ret.std()), 3) if benign_ret.std() > 1e-12 else 0.0
    )
    out["hostile_sharpe_h"] = (
        round(float(hostile_ret.mean() / hostile_ret.std()), 3)
        if hostile_ret.std() > 1e-12
        else 0.0
    )

    # Crash hit rate: when hostile, fraction of times forward return < -5%
    out["crash_hit_rate"] = round(float((hostile_ret < -0.05).mean()), 3)
    out["benign_crash_rate"] = round(float((benign_ret < -0.05).mean()), 3)

    # Welch's t-test on the mean difference
    t_stat, t_p = stats.ttest_ind(benign_ret, hostile_ret, equal_var=False)
    out["mean_diff_t"] = round(float(t_stat), 3)
    out["mean_diff_p"] = round(float(t_p), 6)

    # Levene's test on variance difference (does hostile predict higher vol?)
    lev_stat, lev_p = stats.levene(benign_ret, hostile_ret)
    out["vol_diff_p"] = round(float(lev_p), 6)
    out["vol_ratio_h_b"] = round(float(hostile_ret.std() / benign_ret.std()), 2)

    return out


def validate_indicators(
    panel: pd.DataFrame, spy_close: pd.Series, *, horizon: int = 21, ic_window: int = 252
) -> pd.DataFrame:
    """Run IC + regime-conditional validation on every indicator.

    For each indicator returns:
        ic, p_val, icir — standard linear-prediction IC (regime indicators
            often score low here because risk classification ≠ direction)
        n_benign, n_hostile, frac_hostile — regime sample sizes
        benign_mean, hostile_mean — conditional forward returns
        vol_ratio_h_b — hostile vol / benign vol (>1 means indicator
            successfully predicts higher dispersion in hostile state)
        crash_hit_rate — fraction of hostile firings followed by 5%+ DD
        mean_diff_p, vol_diff_p — significance of the conditional differences
        verdict — REGIME_STRONG / REGIME_USABLE / REGIME_WEAK / NOISE
    """
    fwd = forward_log_return(spy_close, horizon=horizon)
    aligned_panel = panel.reindex(spy_close.index)

    rows = []
    for name in aligned_panel.columns:
        sig = aligned_panel[name]
        ic, pval, n = spearman_ic(sig, fwd)
        roll = rolling_spearman_ic(sig, fwd, window=ic_window)
        ic_ratio = icir(roll)

        common = pd.concat([sig, fwd], axis=1, join="inner").dropna()
        recent = common.iloc[-min(1000, len(common)) :]
        if len(recent) >= 50 and recent.iloc[:, 0].std() > 1e-12:
            r_recent, _ = stats.spearmanr(recent.iloc[:, 0], recent.iloc[:, 1])
        else:
            r_recent = float("nan")

        cond = regime_conditional_metrics(sig, spy_close, horizon=horizon)

        rows.append(
            {
                "indicator": name,
                "ic": round(ic, 4),
                "p_val": round(pval, 6) if pd.notna(pval) else float("nan"),
                "icir": round(ic_ratio, 3),
                "ic_recent": round(r_recent, 4) if pd.notna(r_recent) else float("nan"),
                "frac_hostile": cond.get("frac_hostile", float("nan")),
                "benign_mean": cond.get("benign_mean", float("nan")),
                "hostile_mean": cond.get("hostile_mean", float("nan")),
                "vol_ratio_h_b": cond.get("vol_ratio_h_b", float("nan")),
                "crash_hit_rate": cond.get("crash_hit_rate", float("nan")),
                "benign_crash_rate": cond.get("benign_crash_rate", float("nan")),
                "mean_diff_p": cond.get("mean_diff_p", float("nan")),
                "vol_diff_p": cond.get("vol_diff_p", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    bh_results = benjamini_hochberg([r if pd.notna(r) else 1.0 for r in df["p_val"]], alpha=0.05)
    df["bh_significant"] = bh_results

    # Regime-aware verdict
    def _verdict(row) -> str:
        # Primary: vol ratio (hostile should be more volatile) AND crash hit rate
        # AND benign should beat hostile in mean OR vol
        vol_ok = pd.notna(row["vol_ratio_h_b"]) and row["vol_ratio_h_b"] > 1.2
        mean_ok = (
            pd.notna(row["benign_mean"])
            and pd.notna(row["hostile_mean"])
            and row["benign_mean"] > row["hostile_mean"]
            and pd.notna(row["mean_diff_p"])
            and row["mean_diff_p"] < 0.05
        )
        crash_lift = (
            pd.notna(row["crash_hit_rate"])
            and pd.notna(row["benign_crash_rate"])
            and row["crash_hit_rate"] > 2.0 * max(row["benign_crash_rate"], 0.01)
        )
        ic_ok = pd.notna(row["ic"]) and abs(row["ic"]) > 0.03 and row["bh_significant"]

        passes = sum([vol_ok, mean_ok, crash_lift, ic_ok])
        if passes >= 3:
            return "REGIME_STRONG"
        if passes >= 2:
            return "REGIME_USABLE"
        if passes >= 1:
            return "REGIME_WEAK"
        return "NOISE"

    df["verdict"] = df.apply(_verdict, axis=1)
    return df.sort_values(by="vol_ratio_h_b", ascending=False).reset_index(drop=True)
