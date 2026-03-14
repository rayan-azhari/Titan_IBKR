"""analyse_backtest_meta.py — Deep-dive on OOS equity curve quality.

Checks:
  1. Year-by-year return breakdown (is the Sharpe spread across years or concentrated?)
  2. Rolling 12-month Sharpe (regime stability)
  3. Monthly return heatmap
  4. Drawdown period analysis
  5. Trade PnL distribution (fat tails? single outlier?)
  6. Position sizing audit (is VBT compounding distorting Sharpe?)
  7. Flat-sizing re-run (fixed $1 per unit, no compounding) — true signal quality
"""

import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed.")
    sys.exit(1)

PAIR           = "EUR_USD"
META_THRESHOLD = 0.60
IS_SPLIT       = 0.70
FEES           = 0.00015
INIT_CASH      = 100_000.0
FREQ           = "1h"


# ── Reuse helpers from run_backtest_meta.py ──────────────────────────────────

def _load_parquet(pair, gran):
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def _rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100.0 - (100.0 / (1.0 + gain / loss))


def _tf_signal(close, fast_ma, slow_ma, rsi_period):
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    r    = _rsi(close, rsi_period)
    return (
        pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
        + pd.Series(np.where(r > 50, 0.5, -0.5), index=close.index)
    )


def compute_mtf_signals(pair, mtf_cfg):
    weights   = mtf_cfg["weights"]
    threshold = mtf_cfg["confirmation_threshold"]
    h1 = _load_parquet(pair, "H1")
    tfs = {
        "H1": h1,
        "H4": _load_parquet(pair, "H4"),
        "D":  _load_parquet(pair, "D"),
        "W":  _load_parquet(pair, "W"),
    }
    confluence = pd.Series(0.0, index=h1.index)
    for tf_name, df_tf in tfs.items():
        cfg = mtf_cfg[tf_name]
        sig = _tf_signal(df_tf["close"], cfg["fast_ma"], cfg["slow_ma"], cfg["rsi_period"])
        confluence += sig.reindex(h1.index, method="ffill") * weights[tf_name]
    primary = pd.Series(0, index=h1.index, dtype=int)
    primary[confluence >  threshold] =  1
    primary[confluence < -threshold] = -1
    return primary, confluence, h1


def _bb_bw(close, p=20, s=2.0):
    mid = close.rolling(p).mean()
    return (2 * close.rolling(p).std() * s) / mid.replace(0, np.nan)


def build_features(h1, confluence, primary):
    close = h1["close"]
    feats = pd.DataFrame(index=h1.index)
    for lag in [1, 2, 3, 5, 10]:
        feats[f"ret_{lag}"] = close.pct_change(lag)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - h1["close"].shift()).abs(),
        (h1["low"]  - h1["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    feats["atr_14"]  = tr.rolling(14).mean()
    feats["atr_pct"] = feats["atr_14"] / close.replace(0, np.nan)
    feats["boll_bw"] = _bb_bw(close)
    feats["rsi_14"]  = _rsi(close, 14)
    feats["rsi_21"]  = _rsi(close, 21)
    feats["rsi_overextended"] = ((feats["rsi_14"] > 70) | (feats["rsi_14"] < 30)).astype(int)
    feats["adx_proxy"] = feats["atr_14"].rolling(14).mean() / feats["atr_14"].rolling(14).std()
    feats["confluence_score"] = confluence
    feats["confluence_abs"]   = confluence.abs()
    feats["primary_signal"]   = primary
    feats["hour_utc"]    = h1.index.hour
    feats["is_london"]   = ((feats["hour_utc"] >= 7)  & (feats["hour_utc"] < 16)).astype(int)
    feats["is_new_york"] = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 21)).astype(int)
    feats["is_overlap"]  = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 16)).astype(int)
    feats["day_of_week"] = h1.index.dayofweek
    return feats


def build_meta_labels(primary):
    tbm = pd.read_parquet(FEATURES_DIR / f"{PAIR}_H1_tbm_labels.parquet")["tbm_label"]
    tbm = tbm.reindex(primary.index).fillna(0).astype(int)
    meta = pd.Series(0, index=primary.index, dtype=int)
    meta[(primary ==  1) & (tbm ==  1)] = 1
    meta[(primary == -1) & (tbm == -1)] = 1
    return meta


def run_portfolio(close, long_e, short_e, long_x, short_x, size=np.inf, size_type="value"):
    """Run VBT portfolio. size=np.inf means use all cash (default compounding).
       size=1 with size_type='value' gives flat $1/unit for signal quality check."""
    kwargs = dict(
        close=close, entries=long_e, exits=long_x,
        short_entries=short_e, short_exits=short_x,
        init_cash=INIT_CASH, fees=FEES, freq=FREQ, accumulate=False,
    )
    if size != np.inf:
        kwargs["size"]      = size
        kwargs["size_type"] = size_type
    return vbt.Portfolio.from_signals(**kwargs)


# ── Analysis helpers ─────────────────────────────────────────────────────────

def annual_returns(pf) -> pd.Series:
    """Year-by-year returns from equity curve."""
    eq = pf.value()
    return eq.resample("YE").last().pct_change().dropna() * 100


def rolling_sharpe(pf, window_months: int = 12) -> pd.Series:
    """Rolling annualised Sharpe on hourly returns."""
    ret = pf.returns()
    window_bars = window_months * 21 * 24  # ~21 trading days/month, 24h bars
    ann_factor  = np.sqrt(252 * 24)
    roll_mean   = ret.rolling(window_bars).mean()
    roll_std    = ret.rolling(window_bars).std()
    return (roll_mean / roll_std * ann_factor).dropna()


def monthly_returns_table(pf) -> pd.DataFrame:
    """Monthly return grid (years x months)."""
    eq  = pf.value()
    mon = eq.resample("ME").last().pct_change() * 100
    mon.index = mon.index.to_period("M")
    tbl = mon.to_frame("ret")
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    return tbl.pivot(index="year", columns="month", values="ret")


def drawdown_periods(pf, top_n: int = 5) -> pd.DataFrame:
    """Top N drawdown events by depth."""
    dd = pf.drawdowns  # property not callable
    if dd.count() == 0:
        return pd.DataFrame()
    rec = dd.records_readable.copy()
    # Compute max drawdown % from peak/valley values
    rec["DD_pct"] = (rec["Valley Value"] - rec["Peak Value"]) / rec["Peak Value"] * 100
    rec["Duration"] = rec["End Timestamp"] - rec["Start Timestamp"]
    rec = rec.sort_values("DD_pct").head(top_n)
    return rec[["Start Timestamp", "Valley Timestamp", "End Timestamp", "DD_pct", "Duration"]].reset_index(drop=True)


def trade_pnl_stats(pf) -> dict:
    """PnL distribution stats to detect fat-tail / single-winner distortion."""
    if pf.trades.count() == 0:
        return {}
    pnl = pf.trades.records_readable["PnL"]
    total_pnl = pnl.sum()
    top1_pnl  = pnl.nlargest(1).sum()
    top3_pnl  = pnl.nlargest(3).sum()
    top5_pnl  = pnl.nlargest(5).sum()
    return {
        "total_pnl":      total_pnl,
        "top1_contrib_%": top1_pnl / total_pnl * 100 if total_pnl else 0,
        "top3_contrib_%": top3_pnl / total_pnl * 100 if total_pnl else 0,
        "top5_contrib_%": top5_pnl / total_pnl * 100 if total_pnl else 0,
        "skewness":       float(pnl.skew()),
        "kurtosis":       float(pnl.kurtosis()),
        "pct_positive":   (pnl > 0).mean() * 100,
    }


def flat_size_sharpe(close, long_e, short_e, long_x, short_x) -> float:
    """Sharpe on fixed-lot portfolio (no compounding) — true signal quality."""
    pf = run_portfolio(close, long_e, short_e, long_x, short_x,
                       size=10_000, size_type="value")
    try:
        return pf.sharpe_ratio()
    except Exception:
        return float("nan")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("OOS EQUITY CURVE INVESTIGATION")
    print("=" * 70)

    with open(PROJECT_ROOT / "config" / "mtf.toml", "rb") as f:
        mtf_cfg = tomllib.load(f)

    primary, confluence, h1_df = compute_mtf_signals(PAIR, mtf_cfg)
    close = h1_df["close"]

    print("\n[1] Building signals and training meta-model on IS...")
    feats       = build_features(h1_df, confluence, primary)
    active_mask = primary != 0
    feats_act   = feats[active_mask].dropna()
    y_active    = build_meta_labels(primary)[feats_act.index]

    split_idx    = int(len(feats_act) * IS_SPLIT)
    is_end_ts    = feats_act.index[split_idx - 1]
    oos_start_ts = feats_act.index[split_idx]

    X_is, y_is   = feats_act.iloc[:split_idx], y_active.iloc[:split_idx]
    X_oos        = feats_act.iloc[split_idx:]

    meta_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    meta_model.fit(X_is, y_is)

    meta_prob = pd.Series(np.nan, index=h1_df.index)
    meta_prob.loc[X_is.index]  = meta_model.predict_proba(X_is)[:, 1]
    meta_prob.loc[X_oos.index] = meta_model.predict_proba(X_oos)[:, 1]

    gate = meta_prob >= META_THRESHOLD
    long_e_raw  = (primary ==  1);  long_x_raw  = (primary !=  1)
    short_e_raw = (primary == -1);  short_x_raw = (primary != -1)
    long_e_meta  = (primary ==  1) & gate;  long_x_meta  = long_x_raw
    short_e_meta = (primary == -1) & gate;  short_x_meta = short_x_raw

    oos_mask = close.index >= oos_start_ts
    close_oos = close[oos_mask]

    pf_raw  = run_portfolio(
        close_oos,
        long_e_raw[oos_mask],  short_e_raw[oos_mask],
        long_x_raw[oos_mask],  short_x_raw[oos_mask],
    )
    pf_meta = run_portfolio(
        close_oos,
        long_e_meta[oos_mask], short_e_meta[oos_mask],
        long_x_meta[oos_mask], short_x_meta[oos_mask],
    )

    # ── 1. Year-by-year returns ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. YEAR-BY-YEAR RETURNS (OOS: 2019-2026)")
    print("=" * 70)
    ann_raw  = annual_returns(pf_raw)
    ann_meta = annual_returns(pf_meta)

    all_years = sorted(set(ann_raw.index.year.tolist() + ann_meta.index.year.tolist()))
    print(f"\n  {'Year':<6} {'Raw MTF':>10} {'Meta-Filt':>10}  {'Regime'}")
    print("  " + "-" * 42)
    for yr in all_years:
        r = ann_raw[ann_raw.index.year == yr]
        m = ann_meta[ann_meta.index.year == yr]
        rv = r.values[0] if len(r) else float("nan")
        mv = m.values[0] if len(m) else float("nan")
        regime = "BULL" if rv > 15 else ("BEAR" if rv < -5 else "FLAT")
        print(f"  {yr:<6} {rv:>+9.1f}%  {mv:>+9.1f}%   {regime}")

    pos_years_raw  = sum(1 for yr in all_years
                         for r in [ann_raw[ann_raw.index.year == yr]]
                         if len(r) and r.values[0] > 0)
    print(f"\n  Positive years (Raw):  {pos_years_raw}/{len(all_years)}")

    # ── 2. Rolling 12-month Sharpe ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. ROLLING 12-MONTH SHARPE — regime stability check")
    print("=" * 70)
    rs_raw = rolling_sharpe(pf_raw, 12)
    pct_above_1 = (rs_raw > 1.0).mean() * 100
    pct_negative = (rs_raw < 0).mean() * 100
    print(f"\n  Raw MTF rolling Sharpe (12-month window):")
    print(f"    Min:            {rs_raw.min():>8.3f}")
    print(f"    25th pct:       {rs_raw.quantile(0.25):>8.3f}")
    print(f"    Median:         {rs_raw.median():>8.3f}")
    print(f"    75th pct:       {rs_raw.quantile(0.75):>8.3f}")
    print(f"    Max:            {rs_raw.max():>8.3f}")
    print(f"    % of time > 1:  {pct_above_1:>7.1f}%")
    print(f"    % of time < 0:  {pct_negative:>7.1f}%")

    # ── 3. Monthly returns heatmap (text) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("3. MONTHLY RETURN GRID (Raw MTF OOS)")
    print("=" * 70)
    tbl = monthly_returns_table(pf_raw)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    hdr = f"  {'Year':>4}  " + "  ".join(f"{m:>5}" for m in months)
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for year, row in tbl.iterrows():
        vals = []
        for m in range(1, 13):
            v = row.get(m, float("nan"))
            if np.isnan(v):
                vals.append("     ")
            else:
                vals.append(f"{v:>+5.1f}")
        print(f"  {year:>4}  {'  '.join(vals)}")
    year_totals = tbl.sum(axis=1)
    print(f"\n  Annual totals (sum of months):")
    for yr, tot in year_totals.items():
        bar = "+" * int(abs(tot) / 5) if tot > 0 else "-" * int(abs(tot) / 5)
        print(f"    {yr}: {tot:>+7.1f}%  {bar}")

    # ── 4. Drawdown analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. DRAWDOWN PERIODS (Top 5 worst, Raw MTF OOS)")
    print("=" * 70)
    dd_df = drawdown_periods(pf_raw, top_n=5)
    if not dd_df.empty:
        print(f"\n  {dd_df.to_string(index=False)}")
    else:
        print("  No significant drawdowns found.")

    # ── 5. Trade PnL distribution ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. TRADE PnL DISTRIBUTION — single-winner distortion check")
    print("=" * 70)
    ts = trade_pnl_stats(pf_raw)
    if ts:
        print(f"\n  Raw MTF OOS ({pf_raw.trades.count()} trades):")
        print(f"    Total PnL:         ${ts['total_pnl']:>12,.0f}")
        print(f"    Top-1 trade:       {ts['top1_contrib_%']:>8.1f}% of total P&L")
        print(f"    Top-3 trades:      {ts['top3_contrib_%']:>8.1f}% of total P&L")
        print(f"    Top-5 trades:      {ts['top5_contrib_%']:>8.1f}% of total P&L")
        print(f"    PnL skewness:      {ts['skewness']:>8.3f}  (>0 = right tail, positive)")
        print(f"    PnL kurtosis:      {ts['kurtosis']:>8.3f}  (>3 = fat tails)")
        print(f"    % profitable:      {ts['pct_positive']:>8.1f}%")

        # Percentile breakdown
        pnl = pf_raw.trades.records_readable["PnL"]
        print(f"\n  PnL percentiles:")
        for p in [5, 25, 50, 75, 90, 95, 99]:
            print(f"    {p:>3}th pct:  ${pnl.quantile(p/100):>12,.2f}")

    # ── 6. Position sizing audit ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. POSITION SIZING AUDIT — compounding effect check")
    print("=" * 70)
    print("\n  Running flat-size ($10k fixed) portfolio for comparison...")

    sharpe_flat_raw = flat_size_sharpe(
        close_oos,
        long_e_raw[oos_mask], short_e_raw[oos_mask],
        long_x_raw[oos_mask], short_x_raw[oos_mask],
    )
    pf_flat = run_portfolio(
        close_oos,
        long_e_raw[oos_mask], short_e_raw[oos_mask],
        long_x_raw[oos_mask], short_x_raw[oos_mask],
        size=10_000, size_type="value",
    )
    total_ret_flat = pf_flat.total_return() * 100

    print(f"\n  Compounding (all-cash):    Sharpe {pf_raw.sharpe_ratio():.3f}  |  Return {pf_raw.total_return()*100:+.1f}%")
    print(f"  Flat-size ($10k/trade):    Sharpe {sharpe_flat_raw:.3f}  |  Return {total_ret_flat:+.1f}%")
    print(
        f"\n  Verdict: {'Compounding adds {:.1f}x to returns but Sharpe difference is {:.2f}.'.format(pf_raw.total_return()/pf_flat.total_return(), pf_raw.sharpe_ratio()-sharpe_flat_raw)}"
        if pf_flat.total_return() > 0 else "  (flat portfolio had zero/negative return)"
    )

    # ── 7. Sub-period breakdown ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. SUB-PERIOD BREAKDOWN — 2020-2022 vs 2022-2026")
    print("=" * 70)
    periods = [
        ("COVID/Recovery (2020-2021)", "2020-01-01", "2021-12-31"),
        ("Inflation/USD (2022-2023)", "2022-01-01", "2023-12-31"),
        ("Recent (2024-2026)",        "2024-01-01", "2026-12-31"),
    ]
    print(f"\n  {'Period':<30} {'Sharpe':>8} {'Return':>9} {'Trades':>7} {'WinRate':>8}")
    print("  " + "-" * 64)
    for label, start, end in periods:
        mask = (close_oos.index >= start) & (close_oos.index <= end)
        if mask.sum() < 100:
            continue
        c_sub = close_oos[mask]
        try:
            pf_sub = run_portfolio(
                c_sub,
                long_e_raw[close_oos.index][mask],
                short_e_raw[close_oos.index][mask],
                long_x_raw[close_oos.index][mask],
                short_x_raw[close_oos.index][mask],
            )
            sh  = pf_sub.sharpe_ratio()
            ret = pf_sub.total_return() * 100
            n   = pf_sub.trades.count()
            wr  = pf_sub.trades.win_rate() * 100 if n > 0 else float("nan")
            print(f"  {label:<30} {sh:>8.3f} {ret:>+8.1f}% {n:>7,d} {wr:>7.1f}%")
        except Exception as e:
            print(f"  {label:<30} ERROR: {e}")

    # ── 8. Equity curve HTML ──────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.06,
            subplot_titles=["Equity Curve", "Drawdown (%)", "Rolling 12m Sharpe"],
        )

        # Equity
        for pf, name, color in [
            (pf_raw,  "Raw MTF",       "#2196F3"),
            (pf_meta, "Meta-Filtered", "#4CAF50"),
        ]:
            eq = pf.value()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
                                     line=dict(color=color, width=1.5)), row=1, col=1)

        # Drawdown
        dd_pct = pf_raw.drawdown() * 100
        fig.add_trace(go.Scatter(
            x=dd_pct.index, y=dd_pct.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#F44336", width=1),
        ), row=2, col=1)

        # Rolling Sharpe
        rs = rolling_sharpe(pf_raw, 12)
        colors_rs = ["#4CAF50" if v >= 0 else "#F44336" for v in rs.values]
        fig.add_trace(go.Bar(x=rs.index, y=rs.values, name="Roll.Sharpe",
                             marker_color=colors_rs), row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                      annotation_text="Sharpe=1", row=3, col=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(
            title="EUR/USD H1 OOS Analysis — Raw MTF Confluence",
            template="plotly_dark", height=800, showlegend=True,
        )
        path = REPORTS_DIR / "meta_oos_analysis.html"
        fig.write_html(str(path))
        print(f"\n  Chart saved: {path.name}")
    except Exception as e:
        print(f"\n  (Chart skipped: {e})")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
