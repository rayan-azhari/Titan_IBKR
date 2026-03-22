"""run_cat_amat_strategy.py -- CAT + AMAT Regime-Gated Long-Only Portfolio.

Two-stock equal-weight portfolio using the ADX regime-gated composite:

  ADX > 25 (Trending) -> Trend composite (momentum / moving-average signals)
  ADX < 20 (Ranging)  -> Oscillator composite (mean-reversion signals)
  ADX 20-25 (Neutral) -> Flat -- no new entries, hold existing

Long-only.  ATR-based position sizing (1% risk per trade, 1.5x ATR14 stop).
Realistic stock costs: 10 bps spread + 5 bps slippage per fill.

Best thresholds (from single-instrument long-only sweep):
  CAT  : 1.0z
  AMAT : 0.75z

Usage:
  uv run python research/ic_analysis/run_cat_amat_strategy.py
  uv run python research/ic_analysis/run_cat_amat_strategy.py --sweep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTRUMENTS = ["CAT", "AMAT"]
BEST_THRESHOLDS = {"CAT": 1.00, "AMAT": 0.75}   # from long-only sweep
THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50]

IS_RATIO  = 0.70
INIT_CASH = 10_000.0     # per instrument
RISK_PCT  = 0.01
STOP_ATR  = 1.5
MAX_LEV   = 10.0
FREQ      = "d"

SPREAD   = 0.0010   # 10 bps round-trip
SLIPPAGE = 0.0005   # 5 bps per fill

ADX_RANGING  = 20.0
ADX_TRENDING = 25.0

RANGING_SIGNALS = [
    "zscore_50", "bb_zscore_50", "cci_20", "stoch_k_dev",
    "donchian_pos_10", "zscore_20", "bb_zscore_20",
]
TRENDING_SIGNALS = [
    "ma_spread_50_200", "ma_spread_20_100", "zscore_expanding",
    "zscore_100", "roc_60", "price_pct_rank_60", "ema_slope_20",
]

W = 80
REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Signal / composite helpers (shared with run_regime_backtest.py)
# ---------------------------------------------------------------------------

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _build_composite(
    signals: pd.DataFrame,
    close: pd.Series,
    signal_names: list,
    is_mask: pd.Series,
) -> pd.Series:
    fwd_is = np.log(close.shift(-1) / close)[is_mask]
    parts = []
    for name in signal_names:
        if name not in signals.columns:
            continue
        sign = _ic_sign(signals[name][is_mask], fwd_is)
        parts.append(signals[name] * sign)
    if not parts:
        return pd.Series(0.0, index=signals.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _zscore(series: pd.Series, is_mask: pd.Series) -> pd.Series:
    is_vals = series[is_mask].dropna()
    mu, sigma = float(is_vals.mean()), float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _size_series(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    atr = _atr14(df)
    stop_pct = (STOP_ATR * atr) / close.where(close > 0)
    return (RISK_PCT / stop_pct.where(stop_pct > 0)).clip(upper=MAX_LEV).fillna(0.0)


# ---------------------------------------------------------------------------
# Prepare one instrument: returns aligned OOS arrays
# ---------------------------------------------------------------------------

def prepare_instrument(instrument: str) -> dict:
    """Load, compute signals, build gated_z and size. Returns dict of arrays."""
    df    = _load_ohlcv(instrument, "D")
    close = df["close"]
    signals = build_all_signals(df)

    valid    = signals.notna().any(axis=1) & close.notna()
    signals  = signals[valid]
    close    = close[valid]
    df       = df.loc[valid]
    n        = len(signals)
    is_n     = int(n * IS_RATIO)
    is_mask  = pd.Series(False, index=signals.index)
    is_mask.iloc[:is_n] = True

    adx    = signals["adx_14"]
    regime = pd.Series("neutral", index=signals.index)
    regime[adx < ADX_RANGING]  = "ranging"
    regime[adx > ADX_TRENDING] = "trending"

    # Build composites IS-calibrated
    ranging_comp  = _build_composite(signals, close, RANGING_SIGNALS,  is_mask)
    trending_comp = _build_composite(signals, close, TRENDING_SIGNALS, is_mask)

    is_ranging  = is_mask & (regime == "ranging")
    is_trending = is_mask & (regime == "trending")

    ranging_z  = _zscore(ranging_comp,  is_ranging)
    trending_z = _zscore(trending_comp, is_trending)

    gated_z = pd.Series(0.0, index=signals.index)
    gated_z[regime == "ranging"]  = ranging_z[regime == "ranging"]
    gated_z[regime == "trending"] = trending_z[regime == "trending"]

    size = _size_series(df, close)

    return {
        "instrument": instrument,
        "close":      close,
        "gated_z":    gated_z,
        "size":       size,
        "is_mask":    is_mask,
        "oos_mask":   ~is_mask,
        "regime":     regime,
        "is_n":       is_n,
        "n":          n,
        "index":      signals.index,
    }


# ---------------------------------------------------------------------------
# Single-instrument long-only backtest
# ---------------------------------------------------------------------------

def _run_long(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    size: pd.Series,
) -> dict:
    sig = signal_z.shift(1).fillna(0.0)
    med = float(close.median()) or 1.0
    fees = SPREAD / med
    slip = SLIPPAGE / med

    pf = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size.values,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=fees,
        slippage=slip,
        freq=FREQ,
    )
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "ret":    float(pf.total_return()),
        "dd":     float(pf.max_drawdown()),
        "trades": int(n),
        "wr":     float(pf.trades.win_rate()) if n > 0 else 0.0,
        "annual": float(pf.annualized_return()),
        "value":  pf.value(),
    }


# ---------------------------------------------------------------------------
# Threshold sweep (long-only per instrument)
# ---------------------------------------------------------------------------

def sweep_instrument(data: dict) -> pd.DataFrame:
    oos = data["oos_mask"]
    close = data["close"][oos]
    gz    = data["gated_z"][oos]
    size  = data["size"][oos]

    rows = []
    for thr in THRESHOLDS:
        r = _run_long(close, gz, thr, size)
        rows.append({
            "threshold": thr,
            "sharpe":    r["sharpe"],
            "ret":       r["ret"],
            "dd":        r["dd"],
            "trades":    r["trades"],
            "wr":        r["wr"],
            "annual":    r["annual"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio combination
# ---------------------------------------------------------------------------

def _portfolio_stats(value_series_list: list, index: pd.DatetimeIndex) -> dict:
    """Combine per-instrument value series into equal-weight portfolio."""
    # Align on common index
    aligned = pd.concat(
        [v.reindex(index) for v in value_series_list], axis=1
    ).ffill()
    # Portfolio value = sum of individual portfolios
    port_val  = aligned.sum(axis=1)
    port_ret  = port_val.pct_change().dropna()
    total_ret = float(port_val.iloc[-1] / port_val.iloc[0] - 1)
    ann_ret   = float((1 + total_ret) ** (252 / len(port_ret)) - 1)
    sharpe    = float(port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0.0
    peak      = port_val.cummax()
    dd        = float(((port_val - peak) / peak).min())
    return {
        "sharpe":  sharpe,
        "ret":     total_ret,
        "annual":  ann_ret,
        "dd":      dd,
        "values":  port_val,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(do_sweep: bool = False) -> None:
    print("\n" + "=" * W)
    print("  CAT + AMAT -- REGIME-GATED LONG-ONLY PORTFOLIO")
    print(f"  Spread: {SPREAD*100:.2f}%  Slippage: {SLIPPAGE*100:.2f}%  "
          f"Risk/trade: {RISK_PCT*100:.1f}%  Stop: {STOP_ATR}xATR14")
    print("=" * W)

    # 1. Prepare both instruments
    all_data: dict[str, dict] = {}
    for inst in INSTRUMENTS:
        print(f"\nPreparing {inst}...")
        d = prepare_instrument(inst)
        all_data[inst] = d
        oos = d["oos_mask"]
        print(f"  Bars: {d['n']:,}  IS: {d['is_n']:,}  OOS: {oos.sum():,}")
        print(f"  IS : {d['index'][0].date()} -> {d['index'][d['is_n']-1].date()}")
        print(f"  OOS: {d['index'][d['is_n']].date()} -> {d['index'][-1].date()}")
        regime_oos = d["regime"][oos]
        for lbl in ["ranging", "neutral", "trending"]:
            pct = (regime_oos == lbl).mean() * 100
            print(f"  {lbl.capitalize():<10}: {pct:.0f}% of OOS bars")

    # 2. Optional threshold sweep
    if do_sweep:
        for inst, d in all_data.items():
            sweep_df = sweep_instrument(d)
            print(f"\n{'='*W}")
            print(f"  THRESHOLD SWEEP (LONG-ONLY) -- {inst}")
            print(f"  {'Thresh':>7}  {'Sharpe':>8}  {'Annual':>8}  "
                  f"{'MaxDD':>8}  {'Trades':>7}  {'WR%':>6}")
            print("  " + "-" * 55)
            for _, row in sweep_df.iterrows():
                print(
                    f"  {row['threshold']:>7.2f}  {row['sharpe']:>+8.3f}  "
                    f"{row['annual']:>+8.1%}  {row['dd']:>+8.1%}  "
                    f"{int(row['trades']):>7}  {row['wr']*100:>6.1f}%"
                )

    # 3. Per-instrument OOS results at best threshold
    print(f"\n{'='*W}")
    print("  PER-INSTRUMENT OOS RESULTS (LONG-ONLY, BEST THRESHOLD)")
    print(f"{'='*W}")
    print(f"  {'Instrument':<12}  {'Thresh':>6}  {'Sharpe':>8}  {'Annual':>8}  "
          f"{'MaxDD':>8}  {'Ret':>8}  {'Trades':>7}  {'WR%':>6}")
    print("  " + "-" * (W - 2))

    value_series = []
    oos_index    = None
    per_inst     = {}

    for inst, d in all_data.items():
        thr   = BEST_THRESHOLDS[inst]
        oos   = d["oos_mask"]
        r     = _run_long(d["close"][oos], d["gated_z"][oos], thr, d["size"][oos])
        per_inst[inst] = r
        value_series.append(r["value"])
        if oos_index is None:
            oos_index = d["close"][oos].index
        print(
            f"  {inst:<12}  {thr:>6.2f}  {r['sharpe']:>+8.3f}  "
            f"{r['annual']:>+8.1%}  {r['dd']:>+8.1%}  "
            f"{r['ret']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
        )

    # 4. Combined portfolio
    # Align value series on common OOS index
    common_idx = value_series[0].index.intersection(value_series[1].index)
    port = _portfolio_stats(value_series, common_idx)

    print(f"\n{'='*W}")
    print("  COMBINED PORTFOLIO (CAT 50% + AMAT 50%, EQUAL-WEIGHT)")
    print(f"{'='*W}")
    print(f"  {'OOS Sharpe':<22}: {port['sharpe']:>+.3f}")
    print(f"  {'OOS Annual Return':<22}: {port['annual']:>+.1%}")
    print(f"  {'OOS Total Return':<22}: {port['ret']:>+.1%}")
    print(f"  {'OOS Max Drawdown':<22}: {port['dd']:>+.1%}")
    total_trades = sum(per_inst[i]["trades"] for i in INSTRUMENTS)
    avg_wr = np.mean([per_inst[i]["wr"] for i in INSTRUMENTS if per_inst[i]["trades"] > 0])
    print(f"  {'Total Trades':<22}: {total_trades}")
    print(f"  {'Avg Win Rate':<22}: {avg_wr*100:.1f}%")

    # 5. Annual breakdown
    port_rets = port["values"].pct_change().dropna()
    annual_df = (
        pd.DataFrame({"ret": port_rets})
        .assign(year=lambda x: x.index.year)
        .groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
    )
    print(f"\n  {'Year':<8}  {'Portfolio Return':>18}")
    print("  " + "-" * 28)
    for yr, val in annual_df.items():
        print(f"  {yr:<8}  {val:>+18.1%}")

    # 6. Save CSV
    rows = []
    for inst in INSTRUMENTS:
        r = per_inst[inst]
        rows.append({
            "instrument": inst,
            "threshold":  BEST_THRESHOLDS[inst],
            "sharpe":     r["sharpe"],
            "annual":     r["annual"],
            "ret":        r["ret"],
            "dd":         r["dd"],
            "trades":     r["trades"],
            "wr":         r["wr"],
        })
    rows.append({
        "instrument": "PORTFOLIO",
        "threshold":  "mixed",
        "sharpe":     port["sharpe"],
        "annual":     port["annual"],
        "ret":        port["ret"],
        "dd":         port["dd"],
        "trades":     total_trades,
        "wr":         avg_wr,
    })
    out = REPORTS / "cat_amat_strategy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Results saved: {out}")

    # 7. Optional: Plotly equity curve
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        cat_val  = per_inst["CAT"]["value"]
        amat_val = per_inst["AMAT"]["value"]
        port_val = port["values"]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=["Equity Curve (OOS)", "Drawdown"],
        )
        # Normalize to 100
        def norm(v):
            return v / v.iloc[0] * 100

        fig.add_trace(go.Scatter(
            x=cat_val.index, y=norm(cat_val),
            name="CAT", line=dict(color="#2196F3")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=amat_val.index, y=norm(amat_val),
            name="AMAT", line=dict(color="#FF9800")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=port_val.index, y=norm(port_val),
            name="Portfolio", line=dict(color="#4CAF50", width=2.5)
        ), row=1, col=1)

        # Drawdown
        def _dd_series(v):
            pk = v.cummax()
            return (v - pk) / pk * 100
        fig.add_trace(go.Scatter(
            x=port_val.index, y=_dd_series(port_val),
            name="Portfolio DD", fill="tozeroy",
            line=dict(color="#F44336"), fillcolor="rgba(244,67,54,0.2)"
        ), row=2, col=1)

        fig.update_layout(
            title="CAT + AMAT | Regime-Gated Long-Only | OOS",
            height=700,
            template="plotly_dark",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Value (rebased 100)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %",          row=2, col=1)

        html_out = REPORTS / "cat_amat_strategy.html"
        fig.write_html(str(html_out))
        print(f"  Chart saved:   {html_out}")
    except Exception as e:
        print(f"  [INFO] Plotly chart skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CAT + AMAT Regime-Gated Long-Only Portfolio")
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sweep for each instrument before final result")
    args = parser.parse_args()
    run(do_sweep=args.sweep)


if __name__ == "__main__":
    main()
