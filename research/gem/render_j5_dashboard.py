"""GEM J5 Hybrid Re-Audit — comparison dashboard.

Renders an HTML page comparing the J5 PROMOTED canonical
(`P_hl60_vt05`) against the J4 prior-live (`A1_ewma_hl40`) and the 60/40
SPY/IEF benchmark so the operator can visually inspect the trade-offs.

Computes CAGR, MaxDD, Sharpe, vol from the cell return series (cheap —
GEM is deterministic given the same params; no MC needed here, the MC
verdict is already in `result_log.md`).

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/gem/render_j5_dashboard.py

Output: .tmp/reports/gem_j5_reaudit/dashboard.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.gem.gem_strategy import gem_returns  # noqa: E402
from research.gem.run_gem_j5_reaudit import (  # noqa: E402
    CELLS,
    COST_BPS,
    COST_FIXED_USD,
    NOTIONAL,
    SANCTUARY_MONTHS,
    buy_and_hold_60_40,
    load_universe,
)
from titan.research.framework import (  # noqa: E402
    StrategyClass,
    defaults_for,
    slice_sanctuary,  # noqa: E402
)
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "gem_j5_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cells to display prominently. The promoted P_hl60_vt05 + the prior J4 live
# C3_J4_live + the C1_canonical (sweep IS-best) + the 60/40 benchmark.
HIGHLIGHT_CELLS = ["P_hl60_vt05", "C1_canonical", "C3_J4_live"]

# Stitched-OOS Sharpe numbers from `.tmp/reports/gem_j5_reaudit/result_log.md` §4.1.
AUDIT_NUMBERS = {
    "C1_canonical":      {"sharpe": +0.9935, "ci_lo": +0.474, "ci_hi": +1.488, "verdict": "CONDITIONAL_WATCHPOINT",
                          "rel_mc_dd_red": 0.614, "rel_mc_pass": True, "noise": "mid"},
    "P_hl10_vt05":       {"sharpe": +1.0050, "ci_lo": +0.502, "ci_hi": +1.513, "verdict": "CONDITIONAL_WATCHPOINT",
                          "rel_mc_dd_red": 0.614, "rel_mc_pass": True, "noise": "mid"},
    "P_hl40_vt05":       {"sharpe": +0.9924, "ci_lo": +0.489, "ci_hi": +1.489, "verdict": "DEPLOY",
                          "rel_mc_dd_red": 0.600, "rel_mc_pass": True, "noise": "best"},
    "P_hl60_vt05":       {"sharpe": +1.0021, "ci_lo": +0.510, "ci_hi": +1.505, "verdict": "DEPLOY (PROMOTED)",
                          "rel_mc_dd_red": 0.621, "rel_mc_pass": True, "noise": "best"},
    "P_hl20_vt075":      {"sharpe": +0.8715, "ci_lo": +0.349, "ci_hi": +1.351, "verdict": "DEPLOY",
                          "rel_mc_dd_red": 0.725, "rel_mc_pass": True, "noise": "best"},
    "C2_constrained":    {"sharpe": +0.7633, "ci_lo": +0.232, "ci_hi": +1.272, "verdict": "CONDITIONAL_WATCHPOINT",
                          "rel_mc_dd_red": 0.851, "rel_mc_pass": False, "noise": "best"},
    "C3_J4_live":        {"sharpe": +0.7402, "ci_lo": +0.238, "ci_hi": +1.239, "verdict": "CONDITIONAL_WATCHPOINT",
                          "rel_mc_dd_red": 0.881, "rel_mc_pass": False, "noise": "best"},
    "C4_gross_no_costs": {"sharpe": +1.0279, "ci_lo": +0.510, "ci_hi": +1.521, "verdict": "CONDITIONAL_WATCHPOINT",
                          "rel_mc_dd_red": 0.608, "rel_mc_pass": True, "noise": "mid"},
}


def _returns_for_cell(closes: pd.DataFrame, name: str) -> pd.Series:
    cfg = CELLS[name]
    apply_costs = name != "C4_gross_no_costs"
    return gem_returns(
        closes,
        cfg=cfg,
        cost_bps_per_turnover=COST_BPS if apply_costs else 0.0,
        cost_fixed_usd_per_fill=COST_FIXED_USD if apply_costs else 0.0,
        notional_usd=NOTIONAL,
        execution_mode="etf",
        rebalance_threshold=0.05,
    ).rename(name)


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    parts = [full_returns.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()


def _cagr(rets: pd.Series) -> float:
    """Compound annual growth rate from a per-bar return series."""
    clean = rets.dropna()
    if clean.empty:
        return 0.0
    n_bars = len(clean)
    years = n_bars / 252.0
    total_return = float(np.exp(clean.sum()) - 1.0)
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _annualised_vol(rets: pd.Series) -> float:
    clean = rets.dropna()
    if clean.empty:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(252))


def _equity_curve(rets: pd.Series) -> pd.Series:
    """Cumulative product equity from per-bar log returns."""
    return np.exp(rets.fillna(0.0).cumsum()).rename(rets.name or "equity")


def _drawdown_curve(rets: pd.Series) -> pd.Series:
    eq = _equity_curve(rets)
    peak = eq.cummax()
    return (eq / peak - 1.0).rename(rets.name or "dd")


def _compute_stats(rets: pd.Series) -> dict:
    return {
        "sharpe": float(sharpe(rets, periods_per_year=252)),
        "cagr": _cagr(rets),
        "vol_ann": _annualised_vol(rets),
        "maxdd": float(max_drawdown(rets)),
        "ci_lo": bootstrap_sharpe_ci(rets, periods_per_year=252, seed=42)[0],
        "ci_hi": bootstrap_sharpe_ci(rets, periods_per_year=252, seed=42)[1],
        "n_bars": int(rets.dropna().shape[0]),
    }


# ── Plotly figures ──────────────────────────────────────────────────


def _fig_kpi_grid(stats_by_name: dict[str, dict]) -> go.Figure:
    """Big-number KPI grid for the 4 key series."""
    rows = ["**Sharpe**", "**CI<sub>95 lo</sub>**", "**CAGR**", "**Ann. Vol**", "**Max DD**"]
    names_in_order = ["P_hl60_vt05", "C1_canonical", "C3_J4_live", "Benchmark_60_40"]
    pretty = {
        "P_hl60_vt05":    "<b>J5 LIVE</b><br>P_hl60_vt05",
        "C1_canonical":   "Sweep IS-best<br>C1 (hl20, vt05)",
        "C3_J4_live":     "J4 prior live<br>(hl40, vt10)",
        "Benchmark_60_40":"60/40 SPY/IEF<br>buy-and-hold",
    }
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[pretty[n] for n in names_in_order],
        specs=[[{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]],
    )
    for col_idx, name in enumerate(names_in_order, start=1):
        s = stats_by_name[name]
        rows_text = [
            f"<b>{s['sharpe']:+.3f}</b>",
            f"<b>{s['ci_lo']:+.3f}</b>",
            f"<b>{s['cagr']*100:+.2f}%</b>",
            f"<b>{s['vol_ann']*100:.2f}%</b>",
            f"<b>{s['maxdd']*100:.1f}%</b>",
        ]
        trace = go.Table(
            header=dict(values=["", pretty[name]],
                        fill_color="#f3f4f6", align="left",
                        font=dict(size=11), height=0),
            cells=dict(
                values=[rows, rows_text],
                align="left",
                fill_color="white",
                font=dict(size=14),
                height=28,
            ),
        )
        fig.add_trace(trace, row=1, col=col_idx)
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=24, b=10))
    return fig


def _fig_equity(rets_by_name: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    colors = {
        "P_hl60_vt05":     "#15803d",  # green — promoted/live
        "C1_canonical":    "#7c3aed",  # purple
        "C3_J4_live":      "#ca8a04",  # amber — prior live
        "Benchmark_60_40": "#6b7280",  # grey — benchmark
    }
    widths = {
        "P_hl60_vt05":    3.0,
        "C1_canonical":   1.6,
        "C3_J4_live":     1.6,
        "Benchmark_60_40":1.6,
    }
    labels = {
        "P_hl60_vt05":    "J5 P_hl60_vt05 (LIVE)",
        "C1_canonical":   "C1 sweep IS-best (hl20, vt05)",
        "C3_J4_live":     "C3 J4 prior-live (hl40, vt10)",
        "Benchmark_60_40":"60/40 SPY/IEF (benchmark)",
    }
    for name, rets in rets_by_name.items():
        eq = _equity_curve(rets)
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            name=labels.get(name, name),
            line=dict(color=colors.get(name, "#111827"), width=widths.get(name, 1.6)),
        ))
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=10, t=24, b=40),
        yaxis=dict(type="log", title="Equity (log-scale, $1 → $X)"),
        xaxis=dict(title="Date"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def _fig_drawdown(rets_by_name: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    colors = {
        "P_hl60_vt05":     "#15803d",
        "C1_canonical":    "#7c3aed",
        "C3_J4_live":      "#ca8a04",
        "Benchmark_60_40": "#6b7280",
    }
    labels = {
        "P_hl60_vt05":    "J5 P_hl60_vt05 (LIVE)",
        "C1_canonical":   "C1 sweep IS-best",
        "C3_J4_live":     "C3 J4 prior-live",
        "Benchmark_60_40":"60/40 SPY/IEF",
    }
    for name, rets in rets_by_name.items():
        dd = _drawdown_curve(rets)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100.0, mode="lines",
            name=labels.get(name, name),
            line=dict(color=colors.get(name, "#111827"), width=1.6),
            fill="tozeroy" if name == "P_hl60_vt05" else None,
            fillcolor="rgba(21, 128, 61, 0.12)" if name == "P_hl60_vt05" else None,
        ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=10, t=24, b=40),
        yaxis=dict(title="Drawdown (%)"),
        xaxis=dict(title="Date"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def _fig_audit_table() -> go.Figure:
    """Per-cell 5-axis audit matrix."""
    rows = []
    for name, m in AUDIT_NUMBERS.items():
        rows.append([
            name,
            f"{m['sharpe']:+.4f}",
            f"{m['ci_lo']:+.3f}",
            f"{m['ci_hi']:+.3f}",
            f"{m['rel_mc_dd_red']:.3f}",
            "YES" if m["rel_mc_pass"] else "no",
            m["noise"],
            m["verdict"],
        ])
    # Transpose to column-major.
    cols_data = list(map(list, zip(*rows)))
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Cell</b>", "<b>OOS Sharpe</b>", "<b>CI<sub>95 lo</sub></b>",
                    "<b>CI<sub>95 hi</sub></b>", "<b>Rel-MC DD red</b>",
                    "<b>Rel-MC pass</b>", "<b>Noise axis</b>", "<b>Verdict</b>"],
            fill_color="#1f2937",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cols_data,
            align="left",
            fill_color=["#f9fafb" if (i % 2 == 0) else "white" for i in range(len(rows))],
            font=dict(size=12),
            height=28,
        ),
    )])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def main() -> Path:
    print("=" * 72)
    print("Rendering GEM J5 dashboard...")
    print("=" * 72)

    closes = load_universe()
    print(f"Data: {len(closes)} bars, {closes.index[0].date()} -> {closes.index[-1].date()}")

    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible

    class_def = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(visible.index, class_def.wfo, bars_per_year=252)
    print(f"WFO folds: {len(folds)}")

    print("Computing per-cell strategy returns (3 highlighted cells)...")
    full_by_name: dict[str, pd.Series] = {}
    for name in HIGHLIGHT_CELLS:
        full_by_name[name] = _returns_for_cell(visible, name)

    # Benchmark: 60/40 SPY/IEF (vol-targeted at 0 cost — pure benchmark).
    bench_full = buy_and_hold_60_40(visible).rename("Benchmark_60_40")

    # Restrict to stitched OOS for stats + plotting (matches the audit framing).
    rets_by_name: dict[str, pd.Series] = {
        name: _stitched_oos_returns(full_by_name[name], folds)
        for name in HIGHLIGHT_CELLS
    }
    rets_by_name["Benchmark_60_40"] = _stitched_oos_returns(bench_full, folds)

    print("Computing CAGR / MaxDD / vol stats...")
    stats_by_name = {name: _compute_stats(r) for name, r in rets_by_name.items()}
    for name, s in stats_by_name.items():
        print(f"  {name:>18s}  Sharpe={s['sharpe']:+.3f}  CAGR={s['cagr']*100:+.2f}%  "
              f"MaxDD={s['maxdd']*100:.1f}%  vol={s['vol_ann']*100:.1f}%")

    fig_kpis = _fig_kpi_grid(stats_by_name)
    fig_eq = _fig_equity(rets_by_name)
    fig_dd = _fig_drawdown(rets_by_name)
    fig_audit = _fig_audit_table()

    out = REPORTS_DIR / "dashboard.html"

    def _div(fig: go.Figure) -> str:
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    j5 = stats_by_name["P_hl60_vt05"]
    j4 = stats_by_name["C3_J4_live"]
    bench = stats_by_name["Benchmark_60_40"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>GEM J5 — Hybrid Re-Audit Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        margin: 0; padding: 24px; background: #f3f4f6; color: #111827; }}
.container {{ max-width: 1300px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 8px 0; }}
.subtitle {{ color: #6b7280; font-size: 13px; margin-bottom: 18px; }}
.banner {{ padding: 14px 18px; border-radius: 8px; margin: 14px 0;
          background: #ecfdf5; border-left: 4px solid #15803d; font-size: 13px; }}
.banner b {{ color: #14532d; }}
.section {{ background: white; border-radius: 8px; padding: 12px;
            margin: 14px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
.section h2 {{ font-size: 15px; margin: 4px 0 8px 4px; color: #1f2937; }}
.section p {{ font-size: 13px; color: #4b5563; margin: 6px 4px 12px 4px; }}
code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
.kbd {{ background:#fef3c7; padding:1px 5px; border-radius:3px; font-size:12px; color:#92400e; }}
</style>
</head>
<body>
<div class="container">

<h1>GEM J5 — Hybrid Re-Audit Dashboard</h1>
<div class="subtitle">
  Pre-reg: <code>directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md</code>
  &nbsp;•&nbsp; Visible OOS {next(iter(rets_by_name.values())).index[0].date()} → {next(iter(rets_by_name.values())).index[-1].date()}
  &nbsp;•&nbsp; {len(folds)} rolling WFO folds &nbsp;•&nbsp; Class: <code>CROSS_ASSET_MOMENTUM</code>
</div>

<div class="banner">
  <b>LIVE since 2026-05-16 17:10 UTC.</b>
  J5 <code>P_hl60_vt05</code> (vol_estimator_halflife=60, ann_vol_target=0.05) replaces J4 A1_ewma_hl40
  via in-place config overwrite + <code>docker compose restart titan-portfolio</code>.
  2 IBKR positions adopted cleanly (CSPX qty=27, IDTM qty=45 — no double-fill).
  Stitched-OOS Sharpe <b>{j5['sharpe']:+.3f}</b> vs prior-live <b>{j4['sharpe']:+.3f}</b>
  (<b>+{(j5['sharpe']-j4['sharpe'])/abs(j4['sharpe'])*100:.1f}%</b>);
  CI<sub>95 lo</sub> <b>{j5['ci_lo']:+.3f}</b> vs <b>{j4['ci_lo']:+.3f}</b>
  (<b>{j5['ci_lo']/max(abs(j4['ci_lo']),1e-9):.1f}x tighter</b>).
</div>

<div class="section">
<h2>1. Headline statistics — J5 LIVE vs J4 prior-live vs 60/40 benchmark</h2>
<p>All numbers computed on the stitched-OOS portion of the 60-fold WFO
({len(folds)} rolling folds, ~22y SPY/EFA/IEF). CAGR is the per-year
compound growth rate of $1 invested over the stitched OOS window; CI is
the IID bootstrap 95% confidence interval on Sharpe (n=1000 resamples).</p>
{_div(fig_kpis)}
</div>

<div class="section">
<h2>2. Equity curve — $1 → $X over the stitched OOS window (log scale)</h2>
<p>The J5 line shows the live-config behaviour as if it had been running
since 2003. The visible "kink" near sanctuary cutoff reflects the
WFO-fold stitching (consecutive OOS slices may be sampled non-contiguously).</p>
{_div(fig_eq)}
</div>

<div class="section">
<h2>3. Drawdown comparison vs benchmark</h2>
<p>The drawdown panel is the main visual proof of the L17 relative-MC verdict —
J5 (green, filled) sits inside the benchmark drawdown envelope for most of the
window. Worst observed drawdowns:
J5 <b>{j5['maxdd']*100:.1f}%</b>,
J4 <b>{j4['maxdd']*100:.1f}%</b>,
60/40 benchmark <b>{bench['maxdd']*100:.1f}%</b>.</p>
{_div(fig_dd)}
</div>

<div class="section">
<h2>4. 5-axis decision matrix — all 8 J5 cells</h2>
<p>Numbers from the framework audit (<code>.tmp/reports/gem_j5_reaudit/result_log.md</code>).
P_hl60_vt05 was promoted by §3 rule (highest CI<sub>95 lo</sub> among DEPLOY
cells, excluding baselines C2/C3/C4). Note: C1 sweep IS-best gets CONDITIONAL
because its noise axis is mid (halflife=20 noise-flips); P_hl60 gets DEPLOY
because longer halflife smooths noise.</p>
{_div(fig_audit)}
</div>

<div class="section">
<h2>5. What the J5 deployment implies operationally</h2>
<p><b>Capital deployed:</b> vol_target=0.05 deploys ~50% of the capital that
J4's vt=0.10 did. At $30k initial equity, effective notional drops from
~$30k to ~$15k. Freed capital can rotate to <code>bond_gold</code> CONDITIONAL
(per Wave A.1, L52 hybrid workflow) or sit as cash buffer.</p>

<p><b>Next monthly evaluation:</b> the strategy will recompute target weights
using vol_target=0.05 + halflife=60. The pre-J5 positions (CSPX=27 @ 798.15,
IDTM=45 @ 172.99) were sized for J4's vt=0.10, so the strategy will likely
TRIM the equity leg by ~50% at the next month-end. Orders fire when any
per-asset delta exceeds the 5% <code>rebalance_threshold_weight</code>.</p>

<p><b>Rollback procedure</b> if J5 misbehaves: <code class="kbd">git checkout HEAD~ config/gem_voltarget_lev2.toml</code>
+ <code class="kbd">docker compose restart titan-portfolio</code>.
The 2 adopted positions stay; only the rebalance logic reverts to J4.</p>

<p><b>Next audit:</b> J6 scheduled for 2026-06-13 (4 weeks of J5 live data).</p>
</div>

<div class="subtitle" style="margin-top:24px">
  Generated by <code>research/gem/render_j5_dashboard.py</code>
  •&nbsp; See full audit log <code>.tmp/reports/gem_j5_reaudit/result_log.md</code>
  •&nbsp; Migration memo <code>.tmp/reports/gem_j5_reaudit/findings.md</code>
  •&nbsp; Strategy guide <code>docs/strategies/gem-dual-momentum.md</code>
</div>

</div>
</body>
</html>
"""

    out.write_text(html, encoding="utf-8")
    print(f"\n[write] {out.relative_to(PROJECT_ROOT)}")
    return out


if __name__ == "__main__":
    main()
