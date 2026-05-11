"""Samir-Stack + GEM dual-momentum hybrid — head-to-head comparison.

Runs five variants over the same 2003-2026 history with identical bond
sleeve (60% IEF) and identical leverage tiering, varying only the
regime / asset-selection layer:

  A. SPY buy-hold (raw market control)
  B. GEM standalone (Antonacci's original — abs-mom on SPY, rel-mom
     between SPY and EFA, default to bonds)
  C. Samir-Stack baseline (regime-gated 40/60 stack at L_max=3 on SPY)
  D. Samir + GEM ABS-MOM filter only (Samir's regime score is forced
     to 0 whenever SPY's 12m excess return is negative)
  E. Samir + GEM ABS+REL (D, plus the equity sleeve trades whichever
     of SPY/EFA had higher 12m return)

Hypothesis: variants D and E should reduce max drawdown by adding a
slow regime gate that catches multi-month bear markets the faster
21-day Samir score may not flag (the 2008 grind-down, 2022 rate shock).

Usage:
    uv run python research/samir_stack/run_samir_gem_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.dual_momentum_gem import (  # noqa: E402
    compose_with_samir_regime,
    gem_signal,
)
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Standalone GEM equity curve ────────────────────────────────────────────


def run_gem_standalone(
    spy: pd.Series,
    efa: pd.Series,
    bonds: pd.Series,
    *,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Antonacci's original GEM, monthly rebalance, daily-bar simulation.

    Holds whichever asset GEM points to — switches at month-ends only
    so the path tracks the published implementation cadence.
    """
    sig = gem_signal(spy, efa)
    common = sig.index.intersection(efa.index).intersection(bonds.index)
    spy = spy.reindex(common)
    efa = efa.reindex(common)
    bonds = bonds.reindex(common)
    sig = sig.reindex(common)

    # Monthly rebalance: only flip on the first trading day of each month.
    month_starts = pd.Series(common.month, index=common).diff().fillna(0) != 0
    month_starts.iloc[0] = True

    spy_ret = spy.pct_change().fillna(0.0)
    efa_ret = efa.pct_change().fillna(0.0)
    bond_ret = bonds.pct_change().fillna(0.0)

    held = "BONDS"
    rows = []
    for i, ts in enumerate(common):
        if bool(month_starts.iat[i]):
            new_held = sig["asset_choice"].iat[i]
        else:
            new_held = held
        traded = new_held != held
        tc = (transaction_cost_bps / 10_000.0) if traded else 0.0
        if held == "SPY":
            r = spy_ret.iat[i]
        elif held == "EFA":
            r = efa_ret.iat[i]
        else:
            r = bond_ret.iat[i]
        r -= tc
        rows.append({"date": ts, "ret_strategy": r, "asset": held, "traded": traded})
        held = new_held

    df = pd.DataFrame(rows).set_index("date")
    df["equity"] = (1.0 + df["ret_strategy"]).cumprod()
    df["equity_pos"] = (df["asset"] != "BONDS").astype(float) * 0.40
    df["bond_pos"] = (df["asset"] == "BONDS").astype(float)
    return df


# ── Samir+GEM hybrid ───────────────────────────────────────────────────────


def run_samir_with_gem(
    spy: pd.Series,
    efa: pd.Series,
    ief: pd.Series,
    samir_score: pd.Series,
    *,
    L_max: float = 3.0,
    use_relative_momentum: bool = False,
) -> pd.DataFrame:
    """Samir-Stack with GEM as a hard outer gate.

    If ``use_relative_momentum`` is True, the equity sleeve trades EFA
    on bars where rel-mom favors EFA. Otherwise the sleeve always uses
    SPY (only the abs-mom gate is applied).
    """
    sig = gem_signal(spy, efa)
    common = sig.index.intersection(samir_score.index).intersection(ief.index)
    samir_aligned = samir_score.reindex(common)
    composed = compose_with_samir_regime(sig, samir_aligned)

    spy_aligned = spy.reindex(common)
    efa_aligned = efa.reindex(common)

    if not use_relative_momentum:
        # Equity sleeve = SPY tiered by Samir, gated by GEM
        cfg = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=L_max)
        return run_stacked_strategy(spy_aligned, ief.reindex(common), composed, cfg)

    # use_relative_momentum=True: substitute EFA for SPY on EFA-winner bars.
    # Easiest implementation: build a synthetic "winner" close series
    # whose daily return matches whichever asset was chosen on each bar
    # (with a transition cost on switch days that's already absorbed by
    # Samir's transaction_cost_bps when sleeve weight changes).
    spy_ret = spy_aligned.pct_change().fillna(0.0)
    efa_ret = efa_aligned.pct_change().fillna(0.0)
    winner = sig["rel_winner"].reindex(common).fillna("SPY")
    chosen_ret = pd.Series(
        np.where(winner == "EFA", efa_ret.values, spy_ret.values),
        index=common,
    )
    # Build a synthetic close from the chosen-return series so the
    # downstream `synthetic_leveraged_returns(L)` math still applies.
    synthetic_close = (1.0 + chosen_ret).cumprod() * float(spy_aligned.iloc[0])
    cfg = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=L_max)
    return run_stacked_strategy(synthetic_close, ief.reindex(common), composed, cfg)


# ── Buy-hold control ───────────────────────────────────────────────────────


def run_buy_hold(spy: pd.Series) -> pd.DataFrame:
    rets = spy.pct_change().fillna(0.0)
    eq = (1.0 + rets).cumprod()
    return pd.DataFrame(
        {
            "ret_strategy": rets,
            "equity": eq,
            "equity_pos": 1.0,
            "bond_pos": 0.0,
            "traded": False,
        },
        index=spy.index,
    )


# ── Main ───────────────────────────────────────────────────────────────────


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
    if "traded" in df.columns:
        trades_per_year = float(df["traded"].sum() / n_years)
    else:
        trades_per_year = float("nan")
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


def main() -> int:
    print("Loading data...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    # EFA is loaded separately (not part of the standard panel).
    efa = _load_close("EFA_D.parquet")
    common = data["spy"].index.intersection(efa.index)
    print(
        f"Common SPY ∩ EFA range: {common.min().date()} → {common.max().date()} "
        f"({len(common)} bars)",
        flush=True,
    )

    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    efa = efa.reindex(common)

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=data["hyg"].reindex(common),
        ief_close=ief,
        tlt_close=data["tlt"].reindex(common),
    )
    samir_score = regime_score_equal(panel)

    # Quick sanity print of GEM's composition
    sig = gem_signal(spy, efa)
    asset_share = sig["asset_choice"].value_counts(normalize=True).round(3)
    print(f"\nGEM time-share: {asset_share.to_dict()}\n")

    print("Running 5 variants...", flush=True)
    rows = []

    # A: SPY buy-hold
    rows.append(_summary(run_buy_hold(spy), "A: SPY buy-hold"))

    # B: GEM standalone
    rows.append(_summary(run_gem_standalone(spy, efa, ief), "B: GEM standalone"))

    # C: Samir baseline (3x leveraged ETF, regime gate, no GEM)
    cfg_baseline = StackedConfig(equity_weight=0.40, bond_weight=0.60, L_max=3.0)
    df_c = run_stacked_strategy(spy, ief, samir_score, cfg_baseline)
    rows.append(_summary(df_c, "C: Samir baseline (L=3, SPY only)"))

    # D: Samir + GEM abs-mom filter (still SPY as equity)
    df_d = run_samir_with_gem(spy, efa, ief, samir_score, L_max=3.0, use_relative_momentum=False)
    rows.append(_summary(df_d, "D: Samir + GEM abs-mom filter"))

    # E: Samir + GEM abs+rel (equity = SPY or EFA per rel-mom)
    df_e = run_samir_with_gem(spy, efa, ief, samir_score, L_max=3.0, use_relative_momentum=True)
    rows.append(_summary(df_e, "E: Samir + GEM abs+rel-mom"))

    # F: Samir at L_max=2 + GEM abs-mom (lower-leverage variant for risk-adj)
    df_f = run_samir_with_gem(spy, efa, ief, samir_score, L_max=2.0, use_relative_momentum=False)
    rows.append(_summary(df_f, "F: Samir L=2 + GEM abs-mom"))

    summary = pd.DataFrame(rows).set_index("variant")
    print("\n" + "=" * 110)
    print("Samir-Stack + Dual-Momentum (GEM) hybrid — head-to-head")
    print("Period: 2003-04 → 2026-04 (limited by EFA inception). Bond sleeve: 60% IEF.")
    print("=" * 110)
    print(summary.to_string())

    # Stress: how each variant did during 2008 + 2022 (the regimes
    # where GEM should help most).
    print("\n" + "=" * 110)
    print("Crisis windows (cumulative return; lower = worse):")
    print("=" * 110)
    crisis_windows = [
        ("GFC 2008", "2008-01-01", "2009-06-30"),
        ("COVID 2020", "2020-02-01", "2020-04-30"),
        ("Rate shock 2022", "2022-01-01", "2022-12-31"),
    ]
    crisis_rows = []
    for name, df, _stub in [
        ("A: SPY buy-hold", run_buy_hold(spy), None),
        ("B: GEM standalone", run_gem_standalone(spy, efa, ief), None),
        ("C: Samir baseline", df_c, None),
        ("D: Samir + GEM abs", df_d, None),
        ("E: Samir + GEM abs+rel", df_e, None),
        ("F: Samir L=2 + GEM abs", df_f, None),
    ]:
        rec = {"variant": name}
        for crisis_name, start, end in crisis_windows:
            window = df.loc[start:end, "ret_strategy"]
            if len(window) == 0:
                rec[crisis_name] = float("nan")
            else:
                rec[crisis_name] = round(float((1.0 + window).cumprod().iloc[-1] - 1.0) * 100, 2)
        crisis_rows.append(rec)
    print(pd.DataFrame(crisis_rows).set_index("variant").to_string())

    # Save
    out_csv = REPORTS_DIR / "samir_gem_hybrid.csv"
    summary.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
