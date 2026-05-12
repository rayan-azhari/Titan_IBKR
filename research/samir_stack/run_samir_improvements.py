"""Samir-Stack improvement candidates — head-to-head test of three ideas.

After establishing that GEM doesn't help (see
``directives/Samir-Stack + GEM Hybrid 2026-05-11.md``), this script tests
the three improvements that the GEM analysis flagged as more promising:

  I1. **Rate-shock indicator** — extend Samir's regime panel with a slow
      TLT-momentum component that should have flagged 2022's rate shock
      (when VIX/credit indicators stayed quiet but TLT crashed).

  I2. **DMFI-style bond rotation** — replace the static 60% IEF bond
      sleeve with a dynamic rotation between IEF / HYG / cash based on
      60-day momentum. Should help when both IEF and HYG drop together
      (2022) by going to cash.

  I3. **Opt-in EFA tactical overlay** — only rotate SPY → EFA when the
      relative-momentum gap is large (>5pp 12m). Captures EFA's
      occasional outperformance without taking on its full tail risk.

All three are run side-by-side with the Samir baseline and SPY buy-hold,
plus combinations to see if they're additive or substitutes.

Usage:
    uv run python research/samir_stack/run_samir_improvements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.dual_momentum_gem import gem_signal  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── I1: Rate-shock score ───────────────────────────────────────────────────


def rate_shock_score(
    tlt_close: pd.Series,
    *,
    lookback_days: int = 60,
    shock_threshold: float = -0.10,
) -> pd.Series:
    """Slow indicator: how 'rate-shocked' is the long-bond market?

    Score in [0, 1]: 1.0 when TLT 60d return is positive (rates stable),
    0.0 when TLT has dropped >= ``shock_threshold`` (e.g. -10%) in the
    trailing 60 days. Linear interpolation between.

    The 2022 rate shock saw TLT down -30% over 60 days at one point —
    well below the -10% threshold, so this score would have been 0
    throughout most of that year, kicking the equity sleeve to cash.
    """
    ret_60d = tlt_close.pct_change(lookback_days)
    # Linear: ret >= 0 → 1.0; ret <= shock_threshold → 0.0
    score = (ret_60d - shock_threshold) / (0.0 - shock_threshold)
    return score.clip(0.0, 1.0).fillna(1.0)


def compose_with_rate_shock(
    samir_score: pd.Series,
    tlt_close: pd.Series,
    *,
    lookback_days: int = 60,
    shock_threshold: float = -0.10,
) -> pd.Series:
    """Min-blend Samir score with the rate-shock score.

    The min-blend is intentionally pessimistic: any indicator going
    negative knocks the equity sleeve out, even if the others are still
    benign. This is what we want for tail-risk control.
    """
    rs = rate_shock_score(tlt_close, lookback_days=lookback_days, shock_threshold=shock_threshold)
    common = samir_score.index.intersection(rs.index)
    return pd.concat([samir_score.reindex(common), rs.reindex(common)], axis=1).min(axis=1)


# ── I2: DMFI bond rotation ─────────────────────────────────────────────────


def bond_rotation_returns(
    ief_close: pd.Series,
    hyg_close: pd.Series,
    *,
    lookback_days: int = 60,
) -> pd.Series:
    """DMFI-inspired bond-sleeve rotation: pick highest 60d-momentum
    among {IEF, HYG, cash (0%)}. Returns the chosen daily return per bar.

    Contract: returned value at bar t = decision_made_at_(t-1) * asset_ret_t.
    The winner at t-1 (computed from closes through t-1) is what determines
    which asset's return is collected at t. This is the only legitimate
    semantics for a backtest — a real implementation can only act on
    yesterday's close at today's open.

    A prior version of this function used winner_at_t (computed from close
    at t) to assign asset_ret_t — that is a same-bar look-ahead and was
    found by the 2026-05-12 audit to inflate Sharpe by +1.36 and CAGR by
    +10.06pp on the bond sleeve. Fixed by lagging the winner mask one bar.
    """
    common = ief_close.index.intersection(hyg_close.index)
    ief = ief_close.reindex(common)
    hyg = hyg_close.reindex(common)
    ief_mom = ief.pct_change(lookback_days)
    hyg_mom = hyg.pct_change(lookback_days)
    ief_ret = ief.pct_change().fillna(0.0)
    hyg_ret = hyg.pct_change().fillna(0.0)

    # Pick winner per bar; ties broken IEF > HYG > cash
    winner = pd.Series("CASH", index=common)
    winner[(ief_mom > 0) & (ief_mom >= hyg_mom)] = "IEF"
    winner[(hyg_mom > 0) & (hyg_mom > ief_mom)] = "HYG"
    # Lag by one bar — today's return is earned under yesterday's decision.
    winner = winner.shift(1).fillna("CASH")

    out = pd.Series(0.0, index=common)
    out[winner == "IEF"] = ief_ret[winner == "IEF"]
    out[winner == "HYG"] = hyg_ret[winner == "HYG"]
    return out


def run_stacked_with_bond_rotation(
    spy: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    samir_score: pd.Series,
    *,
    L_max: float = 3.0,
) -> pd.DataFrame:
    """Run Samir-Stack with the bond sleeve rotated.

    Implementation trick: feed ``run_stacked_strategy`` a synthetic
    'IEF' close whose daily returns equal the rotated-bond returns.
    Downstream `pct_change()` recovers the rotated returns.
    """
    rot_rets = bond_rotation_returns(ief, hyg)
    common = spy.index.intersection(rot_rets.index).intersection(samir_score.index)
    rot_rets = rot_rets.reindex(common)
    synthetic_bond = (1.0 + rot_rets).cumprod() * float(ief.reindex(common).iloc[0])
    cfg = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=L_max)
    return run_stacked_strategy(
        spy.reindex(common), synthetic_bond, samir_score.reindex(common), cfg
    )


# ── I3: Opt-in EFA when relmom gap is large ───────────────────────────────


def run_stacked_with_optin_efa(
    spy: pd.Series,
    efa: pd.Series,
    ief: pd.Series,
    samir_score: pd.Series,
    *,
    L_max: float = 3.0,
    relmom_gap_pct: float = 0.05,
) -> pd.DataFrame:
    """DEPRECATED — I3 overlay dropped per 2026-05-12 remediation plan §0(1).

    Contains a same-bar look-ahead in the ``use_efa`` mask (uses today's
    close for both signal and return). Audit 2026-05-12 measured +2.33pp
    CAGR / +0.11 Sharpe inflation on the equity sleeve. Kept for
    historical reproductions of pre-2026-05-12 backtests only.

    DO NOT call from new code. Use plain SPY in the equity sleeve.
    """
    sig = gem_signal(spy, efa)
    common = sig.index.intersection(samir_score.index).intersection(ief.index)
    spy_a = spy.reindex(common)
    efa_a = efa.reindex(common)
    spy_ret = spy_a.pct_change().fillna(0.0)
    efa_ret = efa_a.pct_change().fillna(0.0)

    # Use EFA only when EFA beats SPY by gap_pct on the trailing 12m.
    spy_12m = sig["spy_excess"].reindex(common) + (
        sig["efa_excess"].reindex(common) - sig["efa_excess"].reindex(common)
    )  # spy_12m = spy_excess + rf_12m, but easier to recompute
    spy_12m = spy_a.pct_change(252)
    efa_12m = efa_a.pct_change(252)
    use_efa = (efa_12m - spy_12m) > relmom_gap_pct

    chosen_ret = pd.Series(
        np.where(use_efa.fillna(False), efa_ret.values, spy_ret.values),
        index=common,
    )
    synthetic_close = (1.0 + chosen_ret).cumprod() * float(spy_a.iloc[0])
    cfg = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=L_max)
    return run_stacked_strategy(
        synthetic_close, ief.reindex(common), samir_score.reindex(common), cfg
    )


# ── Buy-hold reference ─────────────────────────────────────────────────────


def run_buy_hold(spy: pd.Series) -> pd.DataFrame:
    rets = spy.pct_change().fillna(0.0)
    eq = (1.0 + rets).cumprod()
    return pd.DataFrame(
        {"ret_strategy": rets, "equity": eq, "equity_pos": 1.0, "bond_pos": 0.0, "traded": False},
        index=spy.index,
    )


# ── Stats / reporting ──────────────────────────────────────────────────────


def _summary(df: pd.DataFrame, name: str) -> dict:
    rets = df["ret_strategy"]
    eq = df["equity"]
    n_years = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    trades_per_year = (
        float(df["traded"].sum() / n_years) if "traded" in df.columns else float("nan")
    )
    return {
        "variant": name,
        "n_years": round(n_years, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sh, 3),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
        "trades_per_year": round(trades_per_year, 1),
    }


CRISIS_WINDOWS = [
    ("GFC 2008", "2008-01-01", "2009-06-30"),
    ("Vol Aug 2015", "2015-07-01", "2015-10-31"),
    ("Q4 2018", "2018-09-01", "2018-12-31"),
    ("COVID 2020", "2020-02-01", "2020-04-30"),
    ("Rate shock 2022", "2022-01-01", "2022-12-31"),
]


def _crisis_row(name: str, df: pd.DataFrame) -> dict:
    rec = {"variant": name}
    for crisis_name, start, end in CRISIS_WINDOWS:
        window = df.loc[start:end, "ret_strategy"]
        if len(window) == 0:
            rec[crisis_name] = float("nan")
        else:
            rec[crisis_name] = round(float((1.0 + window).cumprod().iloc[-1] - 1.0) * 100, 2)
    return rec


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    print("Loading data...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")

    common = data["spy"].index.intersection(efa.index).intersection(data["tlt"].index)
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa = efa.reindex(common)
    print(
        f"Range: {common.min().date()} → {common.max().date()} ({len(common)} bars)\n", flush=True
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    # Variants
    results = []
    crisis_rows = []

    # Baseline + control
    bh = run_buy_hold(spy)
    results.append(_summary(bh, "A: SPY buy-hold"))
    crisis_rows.append(_crisis_row("A: SPY buy-hold", bh))

    cfg_base = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=3.0)
    df_base = run_stacked_strategy(spy, ief, samir_score, cfg_base)
    results.append(_summary(df_base, "B: Samir baseline"))
    crisis_rows.append(_crisis_row("B: Samir baseline", df_base))

    # I1: rate-shock
    score_i1 = compose_with_rate_shock(samir_score, tlt)
    df_i1 = run_stacked_strategy(spy, ief, score_i1, cfg_base)
    results.append(_summary(df_i1, "I1: + rate-shock score"))
    crisis_rows.append(_crisis_row("I1: + rate-shock score", df_i1))

    # I2: bond rotation
    df_i2 = run_stacked_with_bond_rotation(spy, ief, hyg, samir_score, L_max=3.0)
    results.append(_summary(df_i2, "I2: + bond rotation"))
    crisis_rows.append(_crisis_row("I2: + bond rotation", df_i2))

    # I3: opt-in EFA
    df_i3 = run_stacked_with_optin_efa(spy, efa, ief, samir_score, L_max=3.0)
    results.append(_summary(df_i3, "I3: + opt-in EFA (5pp gap)"))
    crisis_rows.append(_crisis_row("I3: + opt-in EFA (5pp gap)", df_i3))

    # Combinations
    score_i12 = compose_with_rate_shock(samir_score, tlt)
    df_i12 = run_stacked_with_bond_rotation(spy, ief, hyg, score_i12, L_max=3.0)
    results.append(_summary(df_i12, "I1+I2: rate-shock + bond rot"))
    crisis_rows.append(_crisis_row("I1+I2: rate-shock + bond rot", df_i12))

    df_i123 = run_stacked_with_optin_efa(spy, efa, ief, score_i12, L_max=3.0)
    # I3 doesn't include bond-rot. Add bond-rot-aware version manually:
    # rebuild by running optin-efa equity then merge with rotated bond
    # via the synthetic-bond trick. Easier: just do a sleeve with both
    # synthetic equity AND synthetic bond.
    sig = gem_signal(spy, efa)
    common_all = (
        sig.index.intersection(samir_score.index)
        .intersection(ief.index)
        .intersection(hyg.index)
        .intersection(tlt.index)
    )
    spy_a = spy.reindex(common_all)
    efa_a = efa.reindex(common_all)
    spy_ret = spy_a.pct_change().fillna(0.0)
    efa_ret = efa_a.pct_change().fillna(0.0)
    spy_12m = spy_a.pct_change(252)
    efa_12m = efa_a.pct_change(252)
    use_efa = (efa_12m - spy_12m) > 0.05
    chosen_ret = pd.Series(
        np.where(use_efa.fillna(False), efa_ret.values, spy_ret.values),
        index=common_all,
    )
    synth_eq = (1.0 + chosen_ret).cumprod() * float(spy_a.iloc[0])
    rot_rets = bond_rotation_returns(ief.reindex(common_all), hyg.reindex(common_all))
    synth_bond = (1.0 + rot_rets.reindex(common_all).fillna(0.0)).cumprod() * float(
        ief.reindex(common_all).iloc[0]
    )
    score_i123 = compose_with_rate_shock(samir_score.reindex(common_all), tlt.reindex(common_all))
    df_i123 = run_stacked_strategy(synth_eq, synth_bond, score_i123, cfg_base)
    results.append(_summary(df_i123, "I1+I2+I3: all three"))
    crisis_rows.append(_crisis_row("I1+I2+I3: all three", df_i123))

    summary = pd.DataFrame(results).set_index("variant")
    print("=" * 110)
    print("Samir-Stack improvement candidates — head-to-head")
    print("=" * 110)
    print(summary.to_string())

    print("\n" + "=" * 110)
    print("Crisis-window cumulative return (%):")
    print("=" * 110)
    crisis_df = pd.DataFrame(crisis_rows).set_index("variant")
    print(crisis_df.to_string())

    out_csv = REPORTS_DIR / "samir_improvements.csv"
    summary.to_csv(out_csv)
    crisis_df.to_csv(REPORTS_DIR / "samir_improvements_crisis.csv")
    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
