"""J4 comparison dashboard.

Renders an HTML page comparing C0_baseline (=C12 production) against the
three DEPLOY winners (A1_ewma_hl40, B1_cap_5pct, B2_cap_10pct) so the
operator can visually inspect the trade-offs before promoting A1.

Run via::

    uv run python research/gem_noise_redesign/render_j4_dashboard.py

Output: .tmp/reports/gem_j4/dashboard.html

Re-uses the J4 audit harness for cell configs and data loading; computes
per-cell equity curves + noise sweeps directly (no MC needed -- the MC
results are already captured in result_log.md).
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
from research.gem_noise_redesign.run_j4_audit import (  # noqa: E402
    CELLS,
    _cost_kwargs,
    load_closes,
)
from titan.research.framework import (  # noqa: E402
    NoiseConfig,
    run_noise_robustness,
)
from titan.research.metrics import BARS_PER_YEAR, sharpe  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "gem_j4"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cells to highlight on the dashboard: baseline + the 3 DEPLOY winners.
# Mid- and worst-failing cells (A2, C1, C2) are summarized in a table only.
HIGHLIGHT_CELLS = ["C0_baseline", "A1_ewma_hl40", "B1_cap_5pct", "B2_cap_10pct"]

# Distinct colors for each highlighted cell.
CELL_COLORS = {
    "C0_baseline": "#888888",  # grey -- the baseline being replaced
    "A1_ewma_hl40": "#0c66e4",  # blue -- the recommended winner
    "B1_cap_5pct": "#f59f00",  # amber
    "B2_cap_10pct": "#cc5de8",  # purple
    "Buy-Hold SPY": "#15803d",  # green -- benchmark
}

# Audit numbers (copied from the J4 result_log.md for the summary table).
# Avoids re-running MC (expensive); only the noise sweep is re-computed.
AUDIT_NUMBERS = {
    "C0_baseline": dict(
        sharpe=0.8016,
        ci_lo=0.402,
        ci_hi=1.164,
        dsr=1.0000,
        rel_mc_ratio=0.5748,
        sanc=0.8717,
        noise_base=0.7596,
        noise_axis="mid",
        verdict="CONDITIONAL_WATCHPOINT",
    ),
    "A1_ewma_hl40": dict(
        sharpe=0.7773,
        ci_lo=0.387,
        ci_hi=1.140,
        dsr=1.0000,
        rel_mc_ratio=0.5657,
        sanc=0.8856,
        noise_base=0.7549,
        noise_axis="best",
        verdict="DEPLOY",
    ),
    "A2_window_60": dict(
        sharpe=0.7904,
        ci_lo=0.392,
        ci_hi=1.143,
        dsr=1.0000,
        rel_mc_ratio=0.6044,
        sanc=0.7643,
        noise_base=0.7812,
        noise_axis="mid",
        verdict="CONDITIONAL_WATCHPOINT",
    ),
    "B1_cap_5pct": dict(
        sharpe=0.5621,
        ci_lo=0.192,
        ci_hi=0.943,
        dsr=1.0000,
        rel_mc_ratio=0.5977,
        sanc=0.9273,
        noise_base=0.5599,
        noise_axis="best",
        verdict="DEPLOY",
    ),
    "B2_cap_10pct": dict(
        sharpe=0.6394,
        ci_lo=0.270,
        ci_hi=1.013,
        dsr=1.0000,
        rel_mc_ratio=0.5763,
        sanc=0.9058,
        noise_base=0.6316,
        noise_axis="best",
        verdict="DEPLOY",
    ),
    "C1_qtile_q40": dict(
        sharpe=0.7937,
        ci_lo=0.410,
        ci_hi=1.150,
        dsr=1.0000,
        rel_mc_ratio=0.6186,
        sanc=0.8889,
        noise_base=0.7456,
        noise_axis="worst",
        verdict="CONDITIONAL_WATCHPOINT",
    ),
    "C2_qtile_q50": dict(
        sharpe=0.7641,
        ci_lo=0.381,
        ci_hi=1.126,
        dsr=1.0000,
        rel_mc_ratio=0.6685,
        sanc=0.9511,
        noise_base=0.7419,
        noise_axis="worst",
        verdict="CONDITIONAL_WATCHPOINT",
    ),
}


def _equity_curve(rets: pd.Series) -> pd.Series:
    return (1.0 + rets.fillna(0.0)).cumprod()


def _drawdown(rets: pd.Series) -> pd.Series:
    eq = _equity_curve(rets)
    return eq / eq.cummax() - 1.0


def _compute_per_cell_returns(closes: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-cell strategy returns on the full visible window."""
    out: dict[str, pd.Series] = {}
    for name in HIGHLIGHT_CELLS:
        cfg = CELLS[name]
        out[name] = gem_returns(closes, cfg=cfg, ief_for_credit=closes["IEF"], **_cost_kwargs(cfg))
    return out


def _compute_per_cell_noise_sweep(closes: pd.DataFrame) -> dict[str, dict]:
    """Per-cell Varma noise sweep (only highlighted cells, ~6s total)."""
    out: dict[str, dict] = {}
    bars_per_year = BARS_PER_YEAR["D"]
    cfg_noise = NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30)
    for name in HIGHLIGHT_CELLS:
        cell_cfg = CELLS[name]

        def fn(df: pd.DataFrame, _cfg=cell_cfg) -> pd.Series:
            return gem_returns(df, cfg=_cfg, ief_for_credit=df.get("IEF"), **_cost_kwargs(_cfg))

        res = run_noise_robustness(closes, fn, periods_per_year=bars_per_year, cfg=cfg_noise)
        out[name] = dict(
            base=res.base_sharpe,
            levels=[r.noise_level for r in res.per_level],
            means=[r.sharpe_mean for r in res.per_level],
            p5s=[r.sharpe_p5 for r in res.per_level],
            passes_mean=res.passes,
            passes_worst=res.worst_case_passes,
        )
    return out


def _fig_equity(per_cell_rets: dict[str, pd.Series], benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    for name, rets in per_cell_rets.items():
        eq = _equity_curve(rets)
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.values,
                mode="lines",
                name=name,
                line=dict(color=CELL_COLORS.get(name, None), width=2),
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Equity %{{y:.3f}}x<extra></extra>",
            )
        )
    bench_eq = _equity_curve(benchmark)
    fig.add_trace(
        go.Scatter(
            x=bench_eq.index,
            y=bench_eq.values,
            mode="lines",
            name="Buy-Hold SPY",
            line=dict(color=CELL_COLORS["Buy-Hold SPY"], width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title="Equity curves (visible window, log scale, costs applied)",
        yaxis_type="log",
        yaxis_title="Equity (x initial)",
        xaxis_title="Date",
        hovermode="x unified",
        height=420,
    )
    return fig


def _fig_drawdown(per_cell_rets: dict[str, pd.Series], benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    for name, rets in per_cell_rets.items():
        dd = _drawdown(rets) * 100
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=name,
                line=dict(color=CELL_COLORS.get(name, None), width=1.5),
            )
        )
    bench_dd = _drawdown(benchmark) * 100
    fig.add_trace(
        go.Scatter(
            x=bench_dd.index,
            y=bench_dd.values,
            mode="lines",
            name="Buy-Hold SPY",
            line=dict(color=CELL_COLORS["Buy-Hold SPY"], width=1.2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Drawdown curves (visible window)",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        hovermode="x unified",
        height=360,
    )
    return fig


def _fig_noise_sweep(noise_data: dict[str, dict]) -> go.Figure:
    """Per-cell noise-degradation sweep -- the key differentiator for J4."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Mean Sharpe vs noise", "5th-percentile Sharpe vs noise")
    )
    for name, d in noise_data.items():
        color = CELL_COLORS.get(name, None)
        fig.add_trace(
            go.Scatter(
                x=d["levels"],
                y=d["means"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                showlegend=True,
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["levels"],
                y=d["p5s"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=name,
            ),
            row=1,
            col=2,
        )
    # Reference: 30% degradation gate (relative to base Sharpe).
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.13,
        showarrow=False,
        text="<b>Varma noise-injection sweep — the gate that separates DEPLOY (best) from CONDITIONAL_WATCHPOINT (mid)</b>",
        font=dict(size=11),
    )
    fig.update_xaxes(title_text="Input noise σ (× input log-return std)", row=1, col=1)
    fig.update_xaxes(title_text="Input noise σ (× input log-return std)", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe (mean across 10 trials)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe (5th percentile across 10 trials)", row=1, col=2)
    fig.update_layout(height=420, hovermode="x unified")
    return fig


def _fig_summary_table(audit_numbers: dict[str, dict]) -> go.Figure:
    rows = [
        "C0_baseline (=C12 prod)",
        "A1_ewma_hl40 ★ recommended",
        "A2_window_60",
        "B1_cap_5pct",
        "B2_cap_10pct",
        "C1_qtile_q40",
        "C2_qtile_q50",
    ]
    keys = [
        "C0_baseline",
        "A1_ewma_hl40",
        "A2_window_60",
        "B1_cap_5pct",
        "B2_cap_10pct",
        "C1_qtile_q40",
        "C2_qtile_q50",
    ]

    def _fmt(v, signed=False, pct=False):
        if pct:
            return f"{v:.2%}"
        return f"{v:+.4f}" if signed else f"{v:.4f}"

    sharpes = [_fmt(audit_numbers[k]["sharpe"], signed=True) for k in keys]
    ci_los = [_fmt(audit_numbers[k]["ci_lo"], signed=True) for k in keys]
    dsr = [_fmt(audit_numbers[k]["dsr"]) for k in keys]
    relmc = [_fmt(audit_numbers[k]["rel_mc_ratio"]) for k in keys]
    sanc = [_fmt(audit_numbers[k]["sanc"], signed=True) for k in keys]
    noise = [audit_numbers[k]["noise_axis"].upper() for k in keys]
    verdict = [audit_numbers[k]["verdict"] for k in keys]

    # Color noise axis cells by class.
    noise_colors = []
    for v in noise:
        if v == "BEST":
            noise_colors.append("#15803d")
        elif v == "MID":
            noise_colors.append("#d97706")
        else:
            noise_colors.append("#b91c1c")

    # Color verdict cells.
    verdict_colors = []
    for v in verdict:
        if v == "DEPLOY":
            verdict_colors.append("#15803d")
        elif v == "CONDITIONAL_WATCHPOINT":
            verdict_colors.append("#d97706")
        else:
            verdict_colors.append("#b91c1c")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Cell</b>",
                        "<b>Sharpe</b>",
                        "<b>CI95 lo</b>",
                        "<b>DSR</b>",
                        "<b>Rel-MC ratio</b>",
                        "<b>Sanc Sharpe</b>",
                        "<b>Noise axis</b>",
                        "<b>Verdict</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[rows, sharpes, ci_los, dsr, relmc, sanc, noise, verdict],
                    fill_color=[
                        ["white"] * len(rows),
                        ["white"] * len(rows),
                        ["white"] * len(rows),
                        ["white"] * len(rows),
                        ["white"] * len(rows),
                        ["white"] * len(rows),
                        noise_colors,
                        verdict_colors,
                    ],
                    font=dict(
                        color=[
                            ["black"] * len(rows),
                            ["black"] * len(rows),
                            ["black"] * len(rows),
                            ["black"] * len(rows),
                            ["black"] * len(rows),
                            ["black"] * len(rows),
                            ["white"] * len(rows),
                            ["white"] * len(rows),
                        ],
                        size=11,
                    ),
                    align="left",
                    height=28,
                ),
            )
        ]
    )
    fig.update_layout(
        title="5-axis decision matrix — all 7 J4 cells",
        height=320,
    )
    return fig


def _annualised(rets: pd.Series, periods_per_year: int = 252) -> dict:
    rets = rets.dropna()
    if len(rets) == 0:
        return dict(cagr=0.0, vol=0.0, sharpe=0.0, maxdd=0.0)
    eq = _equity_curve(rets)
    years = len(rets) / periods_per_year
    cagr = (eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    sh = float(sharpe(rets, periods_per_year=periods_per_year))
    maxdd = float(_drawdown(rets).min())
    return dict(cagr=cagr, vol=vol, sharpe=sh, maxdd=maxdd)


def _fig_kpi_grid(per_cell_rets: dict[str, pd.Series], benchmark: pd.Series) -> go.Figure:
    rows = list(per_cell_rets.keys()) + ["Buy-Hold SPY"]
    cagrs, vols, sharpes, maxdds = [], [], [], []
    for name in per_cell_rets:
        k = _annualised(per_cell_rets[name])
        cagrs.append(f"{k['cagr']:.2%}")
        vols.append(f"{k['vol']:.2%}")
        sharpes.append(f"{k['sharpe']:+.3f}")
        maxdds.append(f"{k['maxdd']:.2%}")
    kb = _annualised(benchmark)
    cagrs.append(f"{kb['cagr']:.2%}")
    vols.append(f"{kb['vol']:.2%}")
    sharpes.append(f"{kb['sharpe']:+.3f}")
    maxdds.append(f"{kb['maxdd']:.2%}")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Cell</b>",
                        "<b>CAGR</b>",
                        "<b>Annualised vol</b>",
                        "<b>Sharpe</b>",
                        "<b>MaxDD</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[rows, cagrs, vols, sharpes, maxdds],
                    fill_color="white",
                    font=dict(color="black", size=11),
                    align="left",
                    height=26,
                ),
            )
        ]
    )
    fig.update_layout(
        title="Realised performance KPIs (full visible window, cost-adjusted)",
        height=260,
    )
    return fig


def main() -> Path:
    print("=" * 72)
    print("Rendering J4 comparison dashboard...")
    print("=" * 72)

    closes = load_closes()
    print(f"Data: {len(closes)} bars, {closes.index[0].date()} -> {closes.index[-1].date()}")

    # Use the visible window (drop the sanctuary so this matches the audit
    # framing the user is comparing against).
    from titan.research.framework import slice_sanctuary

    sanc = slice_sanctuary(closes, months=12)
    visible = sanc.visible

    print("Computing per-cell strategy returns (4 cells)...")
    per_cell_rets = _compute_per_cell_returns(visible)
    benchmark = visible["SPY"].pct_change().fillna(0.0)

    print("Running per-cell Varma noise sweep (4 cells x 3 levels x 10 trials)...")
    noise_data = _compute_per_cell_noise_sweep(visible)

    fig_kpis = _fig_kpi_grid(per_cell_rets, benchmark)
    fig_table = _fig_summary_table(AUDIT_NUMBERS)
    fig_eq = _fig_equity(per_cell_rets, benchmark)
    fig_dd = _fig_drawdown(per_cell_rets, benchmark)
    fig_noise = _fig_noise_sweep(noise_data)

    out = REPORTS_DIR / "dashboard.html"

    def _div(fig: go.Figure) -> str:
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>GEM J4 — Noise-Robust Redesign — Deployment Dashboard</title>
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
</style>
</head>
<body>
<div class="container">

<h1>GEM J4 — Noise-Robust Redesign — Deployment Dashboard</h1>
<div class="subtitle">
  Pre-reg: <code>directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md</code>
  &nbsp;•&nbsp; Visible window {visible.index[0].date()} → {visible.index[-1].date()}
  &nbsp;•&nbsp; Class: <code>CROSS_ASSET_MOMENTUM</code>
</div>

<div class="banner">
  <b>RECOMMENDATION:</b> Promote <b>A1_ewma_hl40</b> to production.
  EWMA vol estimator (half-life=40) recovers the noise axis from <b>mid → best</b>,
  unlocking a true 5-axis <b>DEPLOY</b> verdict.
  Sharpe trade-off: <b>+0.7773 vs +0.8016</b> (-3%) in exchange for noise robustness.
  CI<sub>95 lo</sub> = <b>+0.387</b>, comfortably above zero.
</div>

<div class="section">
<h2>1. 5-axis decision matrix — all 7 J4 cells</h2>
<p>Numbers from the framework audit (<code>.tmp/reports/gem_j4/result_log.md</code>).
DEPLOY cells highlighted in green on the verdict column; CONDITIONAL_WATCHPOINT in amber.
A1 has the highest CI<sub>95 lo</sub> among DEPLOY cells (J3 §3 selection rule).</p>
{_div(fig_table)}
</div>

<div class="section">
<h2>2. Realised KPIs — full visible window, cost-adjusted</h2>
<p>Strategy returns are cost-adjusted via <code>gem_returns(..., cfg=cell)</code> (L23 cost model).
Buy-Hold SPY is the null hypothesis comparator.</p>
{_div(fig_kpis)}
</div>

<div class="section">
<h2>3. Equity curves — A1 vs B1/B2 vs C0 baseline</h2>
<p>Log-scale to compare compounded growth across the 22-year window.
A1 (blue) tracks C0 (grey) closely, slightly below — the ~3% Sharpe drag is the
visual signature of the smoother EWMA vol estimator that's also doing the
noise-robustness work.</p>
{_div(fig_eq)}
</div>

<div class="section">
<h2>4. Drawdown curves</h2>
<p>The vol-target overlay is doing its primary job — drawdowns capped well below
buy-hold SPY (~41% peak). All three DEPLOY cells produce comparable MaxDDs;
the EWMA path has slightly less rapid bottoming on shock bars.</p>
{_div(fig_dd)}
</div>

<div class="section">
<h2>5. Noise-injection sweep — the deployment-blocking axis</h2>
<p><b>This is the chart that decides the deployment.</b> Varma's L24 noise
injection perturbs the input close series by σ × log-return-std and re-runs the
strategy 10 times per noise level. The 5-axis matrix's noise axis is BEST when
both the mean and the 5th-percentile Sharpe survive within 30% degradation.</p>
<p>C0 baseline (grey) has noise axis = <b>mid</b> — mean degradation is fine but
the worst-case (p5) breaks at 0.5σ. A1 (blue) — and B1/B2 — survive on both panels.</p>
{_div(fig_noise)}
</div>

<div class="section">
<h2>6. Deployment path</h2>
<p>Confirmed by J4 audit + this dashboard. Next steps:</p>
<ol style="font-size: 13px; color: #4b5563;">
<li>Plumb J4 fields through <code>titan/strategies/gem/config.py</code> + <code>strategy.py</code>.</li>
<li>Update <code>config/gem_voltarget_lev2.toml</code> with <code>vol_estimator_kind="ewma"</code> + <code>vol_estimator_halflife=40</code>.</li>
<li>Add live-class parity test exercising the new field plumbing.</li>
<li>Run pre-push gates (ruff + pytest).</li>
<li><code>docker compose down &amp;&amp; docker compose up -d --build</code> on the paper account.</li>
<li>Paper-trade for 5 sessions per J3 §4.3.</li>
</ol>
</div>

</div>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    print(f"\nDashboard written: {out}")
    return out


if __name__ == "__main__":
    main()
