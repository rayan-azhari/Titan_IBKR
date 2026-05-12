"""Plot 20-year bootstrap equity curves for vol-target levels 6%, 8%, 12%.

Generates 5,000 bootstrap paths per target_vol (5-day block bootstrap on
the OOS daily return series), then visualises:

  - Background: 200 random sample paths (alpha-light grey)
  - Shaded envelopes: 5th-95th percentile band (1-in-20 to 19-in-20)
  - Bold median path
  - Red worst-of-5000 path (the realistic tail-risk scenario)

Three side-by-side subplots so you can see the dramatic widening of the
distribution as target_vol increases from 6% to 12%. Output is saved to
``.tmp/reports/samir_stack/bootstrap_equity_curves.png``.

Usage::

    uv run python research/samir_stack/plot_bootstrap_equity_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_overlay_sweep import run_with_overlays  # noqa: E402
from research.samir_stack.run_risk_of_ruin import _block_bootstrap_paths  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _wfo_stitched(df) -> np.ndarray:
    n = len(df)
    is_days, oos_days, step = 504, 252, 252
    if n < is_days + oos_days:
        return np.array([])
    rets = df["ret_strategy"].to_numpy()
    chunks = []
    oos_start = is_days
    while oos_start + oos_days <= n:
        chunks.append(rets[oos_start : oos_start + oos_days])
        oos_start += step
    return np.concatenate(chunks)


def main() -> int:
    print("Loading data and building strategies...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")
    common = (
        data["spy"]
        .index.intersection(efa.index)
        .intersection(data["tlt"].index)
        .intersection(data["hyg"].index)
        .intersection(data["ief"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa_a = efa.reindex(common)
    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    targets = [0.06, 0.08, 0.12]
    n_paths = 5000
    horizon_years = 20.0
    horizon_days = int(round(horizon_years * 252))
    block_size = 5
    seed = 42

    paths_by_target: dict[float, np.ndarray] = {}
    for tv in targets:
        df = run_with_overlays(
            spy,
            efa_a,
            ief,
            hyg,
            tlt,
            samir_score,
            panel,
            use_capitulation=False,
            use_vol_target=True,
            vol_target_annual=tv,
        )
        rets = _wfo_stitched(df)
        rng = np.random.default_rng(seed)
        print(
            f"  target_vol={tv:.0%}: bootstrapping {n_paths} paths over {horizon_years:.0f}y...",
            flush=True,
        )
        boot = _block_bootstrap_paths(
            rets,
            horizon_days=horizon_days,
            n_paths=n_paths,
            block_size=block_size,
            rng=rng,
        )
        # Equity = cumulative product of (1 + daily return), starting at 1.0
        eq = np.cumprod(1.0 + boot, axis=1)
        paths_by_target[tv] = eq

    print("\nPlotting...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
    years = np.arange(horizon_days) / 252.0

    for ax, tv in zip(axes, targets, strict=True):
        eq = paths_by_target[tv]

        # Background: 200 random paths (alpha-light)
        rng_bg = np.random.default_rng(seed + 1)
        sample_idx = rng_bg.choice(len(eq), size=200, replace=False)
        for i in sample_idx:
            ax.plot(years, eq[i], color="grey", alpha=0.04, linewidth=0.4)

        # Percentile envelopes
        p5 = np.percentile(eq, 5, axis=0)
        p25 = np.percentile(eq, 25, axis=0)
        p50 = np.percentile(eq, 50, axis=0)
        p75 = np.percentile(eq, 75, axis=0)
        p95 = np.percentile(eq, 95, axis=0)

        ax.fill_between(years, p5, p95, color="steelblue", alpha=0.15, label="5th-95th pct")
        ax.fill_between(years, p25, p75, color="steelblue", alpha=0.30, label="25th-75th pct")
        ax.plot(years, p50, color="navy", linewidth=2.0, label="median path")

        # Worst path: the one with the lowest ending wealth
        worst_idx = int(np.argmin(eq[:, -1]))
        ax.plot(years, eq[worst_idx], color="darkred", linewidth=1.6, label="worst-of-5000")

        # Best path for context
        best_idx = int(np.argmax(eq[:, -1]))
        ax.plot(
            years,
            eq[best_idx],
            color="darkgreen",
            linewidth=1.2,
            alpha=0.6,
            label="best-of-5000",
        )

        # Reference line at 1.0 (starting capital)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_yscale("log")
        ax.set_title(f"target_vol = {tv:.0%}", fontsize=14, fontweight="bold")
        ax.set_xlabel("years")
        ax.set_ylabel("equity multiple (log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)

        # Annotate end-of-horizon stats
        median_end = float(p50[-1])
        worst_end = float(eq[worst_idx, -1])
        worst_dd = _max_dd(eq[worst_idx])
        median_dd = _median_max_dd(eq)
        text = (
            f"Median end: {median_end:.1f}x\n"
            f"Worst end:  {worst_end:.2f}x\n"
            f"Median MaxDD: {median_dd * 100:.1f}%\n"
            f"Worst MaxDD:  {worst_dd * 100:.1f}%"
        )
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "grey"},
        )

    fig.suptitle(
        f"Samir-Stack (MES L=3, 10/90) — 20-year equity-curve distribution\n"
        f"({n_paths:,} bootstrap paths, {block_size}-day blocks)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = REPORTS_DIR / "bootstrap_equity_curves_20y.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # Also save the underlying summary stats
    summary_lines = [
        "Bootstrap 20-year equity-curve summary",
        "=" * 60,
        f"{'target':>8}  {'median_end':>11}  {'5pct_end':>11}  "
        f"{'1pct_end':>11}  {'worst_end':>11}  {'best_end':>11}",
    ]
    for tv in targets:
        eq = paths_by_target[tv]
        ends = eq[:, -1]
        summary_lines.append(
            f"{tv:>7.0%}  {np.median(ends):>11.2f}x  "
            f"{np.percentile(ends, 5):>11.2f}x  "
            f"{np.percentile(ends, 1):>11.2f}x  "
            f"{ends.min():>11.2f}x  "
            f"{ends.max():>11.2f}x"
        )
    print("\n" + "\n".join(summary_lines))

    out_txt = REPORTS_DIR / "bootstrap_equity_summary_20y.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved: {out_txt}")
    return 0


def _max_dd(eq_path: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq_path)
    return float(((eq_path - peak) / peak).min())


def _median_max_dd(eq_paths: np.ndarray) -> float:
    """Median of MaxDD across all paths (one MaxDD per path)."""
    dds = np.array([_max_dd(p) for p in eq_paths])
    return float(np.median(dds))


if __name__ == "__main__":
    sys.exit(main())
