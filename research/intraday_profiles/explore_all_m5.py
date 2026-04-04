"""explore_all_m5.py — EDA across all M5 parquet files.

Runs the same pattern analysis (hourly, DOW, monthly, AM/PM, session
transition, volatility clustering) for every M5 instrument with sufficient
history (>= 60 trading days).

Instruments with < 60 trading days are skipped with a note.

Detects FX vs equity automatically:
  - FX:     bars span 00:00-22:00 UTC (24h near-continuous)
  - Equity: bars cluster in 14:30-21:00 UTC window (US session only)

Outputs per instrument: .tmp/reports/eda_m5/<SYMBOL>/01_hourly_profile.html etc.
Cross-instrument summary: .tmp/reports/eda_m5/00_summary.html

Usage:
    uv run python research/intraday_profiles/explore_all_m5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
BASE_OUT  = PROJECT_ROOT / ".tmp" / "reports" / "eda_m5"
BASE_OUT.mkdir(parents=True, exist_ok=True)

MIN_TRADING_DAYS = 60

DOW_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_m5(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_close     = df["close"].shift(1)
    df["bar_ret"]  = (df["close"] - df["open"]) / df["open"]
    df["abs_ret"]  = df["bar_ret"].abs()
    df["log_ret"]  = np.log(df["close"] / prev_close)
    df["tr"]       = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"]      = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    df["clv"]      = df["clv"].fillna(0.0)
    df["hour"]     = df.index.hour
    df["minute"]   = df.index.minute
    df["dow"]      = df.index.day_of_week
    df["month"]    = df.index.month
    df["hhmm"]     = df["hour"] * 60 + df["minute"]
    df["date"]     = df.index.date
    return df


def detect_type(df: pd.DataFrame) -> str:
    """Return 'fx' if bars exist outside 12:00-22:00 UTC, else 'equity'."""
    outside = ((df.index.hour < 12) | (df.index.hour >= 22)).mean()
    return "fx" if outside > 0.10 else "equity"


# ---------------------------------------------------------------------------
# Individual chart functions (instrument-aware)
# ---------------------------------------------------------------------------


def plot_hourly(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    grp = df.groupby("hour").agg(
        abs_ret=("abs_ret", "mean"),
        drift=("bar_ret", "mean"),
        volume=("volume", "mean"),
        n=("bar_ret", "count"),
    ).reset_index()

    pvals, t_stats = [], []
    for h in grp["hour"]:
        vals = df.loc[df["hour"] == h, "bar_ret"].dropna()
        if len(vals) < 30:
            pvals.append(1.0); t_stats.append(0.0)
        else:
            t, p = stats.ttest_1samp(vals, 0.0)
            pvals.append(p); t_stats.append(t)
    grp["p"] = pvals
    grp["sig"] = grp["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

    fig = make_subplots(rows=3, cols=1, subplot_titles=[
        "Volatility — mean |bar return| (pips)",
        "Directional Drift — mean bar return (pips)  * = p<0.05",
        "Mean Volume (normalised)",
    ])
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["abs_ret"] * 10000,
                         marker_color="steelblue", name="Vol"), row=1, col=1)

    colours = ["red" if d < 0 else "green" for d in grp["drift"]]
    labels  = [f"{d*10000:.2f}{s}" for d, s in zip(grp["drift"], grp["sig"])]
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["drift"] * 10000,
                         text=labels, textposition="outside",
                         marker_color=colours, name="Drift"), row=2, col=1)

    mv = grp["volume"].mean()
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["volume"] / max(mv, 1e-9),
                         marker_color="purple", name="RelVol"), row=3, col=1)

    fig.update_layout(title=f"{symbol} — Hourly Profile", height=850, showlegend=False)
    fig.write_html(str(out / "01_hourly_profile.html"))

    sig = grp[grp["p"] < 0.05]
    return {
        "sig_drift_hours": [(int(r.hour), round(r.drift * 10000, 2), r.sig)
                            for _, r in sig.iterrows()],
    }


def plot_dow(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[df["dow"] < 5]
    daily = df_wd.groupby(["dow", "date"]).agg(day_ret=("log_ret", "sum")).reset_index()
    dow_d = daily.groupby("dow").agg(
        mean=("day_ret", "mean"), std=("day_ret", "std"), n=("day_ret", "count"),
    ).reset_index()
    dow_d["name"] = dow_d["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})

    pvals = []
    for d in dow_d["dow"]:
        vals = daily.loc[daily["dow"] == d, "day_ret"]
        pvals.append(stats.ttest_1samp(vals, 0.0)[1] if len(vals) >= 10 else 1.0)
    dow_d["p"] = pvals
    dow_d["sig"] = dow_d["p"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")

    vol_grp = df_wd.groupby("dow")["abs_ret"].mean().reset_index()
    vol_grp["name"] = vol_grp["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Mean M5 Volatility by Day of Week",
        "Mean Daily Return by Day of Week — * = p<0.05",
    ])
    fig.add_trace(go.Bar(x=vol_grp["name"], y=vol_grp["abs_ret"] * 10000,
                         marker_color="steelblue", name="Vol"), row=1, col=1)

    colours = ["red" if d < 0 else "green" for d in dow_d["mean"]]
    labels  = [f"{d*10000:.1f}{s}" for d, s in zip(dow_d["mean"], dow_d["sig"])]
    fig.add_trace(go.Bar(x=dow_d["name"], y=dow_d["mean"] * 10000,
                         text=labels, textposition="outside",
                         error_y=dict(type="data",
                                      array=(dow_d["std"] / np.sqrt(dow_d["n"]) * 10000).tolist()),
                         marker_color=colours, name="DayRet"), row=2, col=1)

    fig.update_layout(title=f"{symbol} — Day-of-Week Profile", height=600, showlegend=False)
    fig.write_html(str(out / "02_dow_profile.html"))

    sig = dow_d[dow_d["p"] < 0.05]
    return {
        "dow_means_pips": dict(zip(dow_d["name"], dow_d["mean"].mul(10000).round(1))),
        "sig_dow": [(r["name"], round(r["mean"]*10000, 1), r["sig"]) for _, r in sig.iterrows()],
    }


def plot_intraday_path_by_dow(df: pd.DataFrame, symbol: str, out: Path) -> None:
    df_wd = df[df["dow"] < 5].copy()
    df_wd["cum_ret"] = df_wd.groupby("date")["log_ret"].cumsum()
    day_std = df_wd.groupby("date")["log_ret"].std().rename("day_std")
    df_wd   = df_wd.join(day_std, on="date")
    df_wd["cum_norm"] = df_wd["cum_ret"] / df_wd["day_std"].replace(0, np.nan)

    fig = go.Figure()
    for dow in range(5):
        sub = df_wd[df_wd["dow"] == dow]
        if len(sub) == 0:
            continue
        avg = sub.groupby("hhmm")["cum_norm"].mean()
        fig.add_trace(go.Scatter(x=avg.index, y=avg.values,
                                 name=DOW_NAMES[dow], mode="lines"))

    for h, lbl in [(0,"00:00"),(7,"London"),(12,"AM/PM"),(13,"NY open"),(17,"Lon close"),(22,"NY close")]:
        fig.add_vline(x=h*60, line_dash="dot", line_color="grey",
                      annotation_text=lbl, annotation_position="top")

    fig.update_layout(title=f"{symbol} — Mean Normalised Intraday Path by DOW",
                      xaxis_title="Minute of day (UTC)", height=480)
    fig.write_html(str(out / "03_intraday_path_dow.html"))


def plot_hour_dow_heatmap(df: pd.DataFrame, symbol: str, out: Path) -> None:
    df_wd = df[df["dow"] < 5]
    pivot = df_wd.groupby(["dow", "hour"])["bar_ret"].mean().unstack("hour") * 10000

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=[DOW_NAMES[d] for d in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=pivot.round(2).values, texttemplate="%{text}",
        colorbar_title="Drift (pips)",
    ))
    fig.update_layout(title=f"{symbol} — Drift (pips) by Hour × DOW", height=320)
    fig.write_html(str(out / "04_hour_dow_heatmap.html"))


def plot_monthly(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[df["dow"] < 5]
    daily = df_wd.groupby("date").agg(
        day_ret=("log_ret", "sum"), month=("month", "first"),
    ).reset_index()
    monthly = daily.groupby("month").agg(
        mean=("day_ret", "mean"), std=("day_ret", "std"), n=("day_ret", "count"),
    ).reset_index()

    pvals = []
    for m in monthly["month"]:
        vals = daily.loc[daily["month"] == m, "day_ret"]
        pvals.append(stats.ttest_1samp(vals, 0.0)[1] if len(vals) >= 10 else 1.0)
    monthly["p"] = pvals
    monthly["sig"] = monthly["p"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
    monthly["name"] = monthly["month"].apply(lambda m: MONTH_NAMES[m-1])

    colours = ["red" if d < 0 else "green" for d in monthly["mean"]]
    labels  = [f"{d*10000:.1f}{s}" for d, s in zip(monthly["mean"], monthly["sig"])]
    fig = go.Figure(go.Bar(
        x=monthly["name"], y=monthly["mean"] * 10000,
        text=labels, textposition="outside",
        error_y=dict(type="data",
                     array=(monthly["std"] / np.sqrt(monthly["n"].clip(lower=1)) * 10000).tolist()),
        marker_color=colours,
    ))
    for _, row in monthly.iterrows():
        fig.add_annotation(x=row["name"], y=monthly["mean"].min()*10000 - 2,
                           text=f"n={int(row['n'])}", showarrow=False, font=dict(size=9))
    fig.update_layout(title=f"{symbol} — Mean Daily Return by Month  * = p<0.05",
                      yaxis_title="Pips", height=430, showlegend=False)
    fig.write_html(str(out / "05_monthly_profile.html"))

    sig = monthly[monthly["p"] < 0.05]
    return {"sig_months": [(r["name"], round(r["mean"]*10000, 1), r["sig"]) for _, r in sig.iterrows()]}


def plot_am_pm(df: pd.DataFrame, symbol: str, out: Path, is_fx: bool) -> dict:
    df_wd = df[df["dow"] < 5].copy()
    if is_fx:
        am_end, pm_start, pm_end = 12, 12, 22
    else:
        am_end, pm_start, pm_end = 17, 17, 21   # equity: AM = first half of session

    am = df_wd[df_wd["hour"] < am_end].groupby("date")["log_ret"].sum().rename("am")
    pm = df_wd[(df_wd["hour"] >= pm_start) & (df_wd["hour"] < pm_end)].groupby("date")["log_ret"].sum().rename("pm")
    both = pd.concat([am, pm], axis=1).dropna()
    if len(both) < 20:
        return {"am_pm_r": float("nan"), "am_pm_p": float("nan")}

    r, p = stats.pearsonr(both["am"], both["pm"])
    am_up   = both[both["am"] > 0]["pm"]
    am_down = both[both["am"] < 0]["pm"]

    fig = px.scatter(both, x="am", y="pm",
                     labels={"am": f"AM return (00-{am_end}h UTC)", "pm": f"PM return ({pm_start}-{pm_end}h UTC)"},
                     title=f"{symbol} — AM vs PM Return  (r={r:.3f}, p={p:.3f})",
                     opacity=0.5)
    xr = np.linspace(both["am"].min(), both["am"].max(), 100)
    slope, intercept, *_ = stats.linregress(both["am"], both["pm"])
    fig.add_trace(go.Scatter(x=xr, y=slope*xr+intercept, mode="lines",
                             line=dict(color="black", dash="dash"), name="OLS"))
    fig.update_layout(height=450)
    fig.write_html(str(out / "06_am_pm_correlation.html"))

    return {
        "am_pm_r": round(r, 3),
        "am_pm_p": round(p, 3),
        "pm_when_am_up_pips":   round(float(am_up.mean()) * 10000, 1) if len(am_up) else float("nan"),
        "pm_when_am_down_pips": round(float(am_down.mean()) * 10000, 1) if len(am_down) else float("nan"),
    }


def plot_session_transition(df: pd.DataFrame, symbol: str, out: Path, is_fx: bool) -> dict:
    if not is_fx:
        return {}
    df_wd = df[df["dow"] < 5]
    london = df_wd[(df_wd["hour"] >= 7) & (df_wd["hour"] < 12)].groupby("date")["log_ret"].sum().rename("london")
    ny     = df_wd[(df_wd["hour"] >= 13) & (df_wd["hour"] < 17)].groupby("date")["log_ret"].sum().rename("ny")
    both   = pd.concat([london, ny], axis=1).dropna()
    if len(both) < 20:
        return {}

    r, p = stats.pearsonr(both["london"], both["ny"])
    lon_up   = both[both["london"] > 0]["ny"]
    lon_down = both[both["london"] < 0]["ny"]
    _, p_up   = stats.ttest_1samp(lon_up, 0) if len(lon_up) >= 10 else (0, 1.0)
    _, p_down = stats.ttest_1samp(lon_down, 0) if len(lon_down) >= 10 else (0, 1.0)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "NY open return by London direction",
        "London vs NY open scatter",
    ])
    fig.add_trace(go.Histogram(x=lon_up*10000, name="London Up",
                               opacity=0.7, marker_color="green", nbinsx=40), row=1, col=1)
    fig.add_trace(go.Histogram(x=lon_down*10000, name="London Down",
                               opacity=0.7, marker_color="red", nbinsx=40), row=1, col=1)
    fig.add_trace(go.Scatter(x=both["london"]*10000, y=both["ny"]*10000, mode="markers",
                             marker=dict(color="steelblue", size=5, opacity=0.6),
                             name="obs"), row=1, col=2)
    xr = np.linspace(both["london"].min(), both["london"].max(), 100) * 10000
    slope, intercept, *_ = stats.linregress(both["london"]*10000, both["ny"]*10000)
    fig.add_trace(go.Scatter(x=xr, y=slope*xr+intercept, mode="lines",
                             line=dict(color="black", dash="dash"), name="OLS"), row=1, col=2)
    fig.update_layout(title=f"{symbol} — London -> NY Follow-Through  (r={r:.3f}, p={p:.3f})",
                      height=430, barmode="overlay")
    fig.write_html(str(out / "08_session_transition.html"))

    return {
        "london_ny_r": round(r, 3), "london_ny_p": round(p, 3),
        "ny_when_lon_up_pips":   round(float(lon_up.mean())*10000, 1),
        "ny_when_lon_down_pips": round(float(lon_down.mean())*10000, 1),
        "p_ny_when_lon_up":   round(p_up, 3),
        "p_ny_when_lon_down": round(p_down, 3),
    }


def plot_vol_autocorr(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[df["dow"] < 5]
    hourly_tr = df_wd.groupby([df_wd.index.date, "hour"])["tr"].mean()
    hourly_tr = hourly_tr.reset_index(level=0, drop=True)

    lags = list(range(1, 13))
    h_acf = [hourly_tr.autocorr(lag=l) for l in lags]
    b_acf = [df_wd["tr"].autocorr(lag=l) for l in range(1, 7)]

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Hourly TR Autocorrelation",
        "M5 Bar TR Autocorrelation",
    ])
    ci_h = 1.96 / np.sqrt(max(len(hourly_tr), 1))
    ci_b = 1.96 / np.sqrt(max(len(df_wd), 1))

    fig.add_trace(go.Bar(x=lags, y=h_acf,
                         marker_color=["red" if a < 0 else "steelblue" for a in h_acf]), row=1, col=1)
    fig.add_hline(y=ci_h, line_dash="dot", line_color="orange", row=1, col=1)
    fig.add_hline(y=-ci_h, line_dash="dot", line_color="orange", row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(1,7)), y=b_acf,
                         marker_color=["red" if a < 0 else "steelblue" for a in b_acf]), row=2, col=1)
    fig.add_hline(y=ci_b, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=-ci_b, line_dash="dot", line_color="orange", row=2, col=1)

    fig.update_layout(title=f"{symbol} — Volatility Autocorrelation", height=580, showlegend=False)
    fig.write_html(str(out / "09_vol_autocorr.html"))

    return {"tr_lag1h_acf": round(h_acf[0], 3), "tr_lag1bar_acf": round(b_acf[0], 3)}


def plot_clv_heatmap(df: pd.DataFrame, symbol: str, out: Path) -> None:
    df_wd = df[df["dow"] < 5]
    pivot = df_wd.groupby(["dow", "hour"])["clv"].mean().unstack("hour")

    sig = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for dow in pivot.index:
        for hour in pivot.columns:
            vals = df_wd[(df_wd["dow"]==dow) & (df_wd["hour"]==hour)]["clv"]
            if len(vals) >= 30:
                _, p = stats.ttest_1samp(vals, 0.0)
                sig.loc[dow, hour] = "★" if p < 0.01 else ("·" if p < 0.05 else "")

    text = pivot.round(3).astype(str) + sig.values

    fig = go.Figure(go.Heatmap(
        z=pivot.values.astype(float),
        x=[f"{h:02d}h" for h in pivot.columns],
        y=[DOW_NAMES[d] for d in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=text.values, texttemplate="%{text}",
        colorbar_title="Mean CLV",
    ))
    fig.update_layout(title=f"{symbol} — Mean CLV by Hour × DOW  (★=p<0.01, ·=p<0.05)", height=320)
    fig.write_html(str(out / "10_clv_heatmap.html"))


# ---------------------------------------------------------------------------
# Per-instrument runner
# ---------------------------------------------------------------------------


def run_instrument(path: Path) -> dict | None:
    sym = path.stem
    print(f"\n{'='*55}")
    print(f"  {sym}")
    print(f"{'='*55}")

    df_raw = load_m5(path)
    n_days = len(np.unique(df_raw.index.date))
    print(f"  {len(df_raw):,} bars  [{df_raw.index[0].date()} -> {df_raw.index[-1].date()}]  "
          f"{n_days} trading days")

    if n_days < MIN_TRADING_DAYS:
        print(f"  SKIP: only {n_days} trading days (need >= {MIN_TRADING_DAYS})")
        return {"symbol": sym, "skipped": True, "n_days": n_days,
                "reason": f"only {n_days} trading days"}

    df      = enrich(df_raw)
    is_fx   = detect_type(df)
    type_lbl = "FX (24h)" if is_fx == "fx" else "Equity (US session)"
    print(f"  Type: {type_lbl}")

    out = BASE_OUT / sym
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {"symbol": sym, "skipped": False, "n_days": n_days,
                     "type": type_lbl,
                     "date_range": f"{df.index[0].date()} -> {df.index[-1].date()}"}

    print("  [01] Hourly profile...")
    results.update(plot_hourly(df, sym, out))

    if n_days >= 60:
        print("  [02] Day-of-week...")
        results.update(plot_dow(df, sym, out))

        print("  [03] Intraday path by DOW...")
        plot_intraday_path_by_dow(df, sym, out)

        print("  [04] Hour x DOW heatmap...")
        plot_hour_dow_heatmap(df, sym, out)

    if n_days >= 90:
        print("  [05] Monthly seasonality...")
        results.update(plot_monthly(df, sym, out))

    print("  [06] AM/PM correlation...")
    results.update(plot_am_pm(df, sym, out, is_fx == "fx"))

    if is_fx == "fx":
        print("  [08] Session transition...")
        results.update(plot_session_transition(df, sym, out, True))

    print("  [09] Volatility autocorrelation...")
    results.update(plot_vol_autocorr(df, sym, out))

    print("  [10] CLV heatmap...")
    plot_clv_heatmap(df, sym, out)

    # Print key findings
    if results.get("sig_drift_hours"):
        print(f"  ** Significant drift hours: {results['sig_drift_hours']}")
    if results.get("sig_dow"):
        print(f"  ** Significant DOW:         {results['sig_dow']}")
    if results.get("sig_months"):
        print(f"  ** Significant months:      {results['sig_months']}")
    if results.get("am_pm_r") is not None:
        print(f"  ** AM/PM correlation: r={results.get('am_pm_r')}  p={results.get('am_pm_p')}")
    if results.get("london_ny_r") is not None:
        print(f"  ** London->NY r={results.get('london_ny_r')}  p={results.get('london_ny_p')}")
    print(f"  ** Vol lag-1h ACF: {results.get('tr_lag1h_acf')}")

    return results


# ---------------------------------------------------------------------------
# Cross-instrument summary chart
# ---------------------------------------------------------------------------


def build_summary(all_results: list[dict]) -> None:
    valid = [r for r in all_results if not r.get("skipped")]
    if not valid:
        return

    syms = [r["symbol"] for r in valid]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Volatility Lag-1h Autocorrelation (higher = stronger vol clustering)",
            "AM/PM Correlation r (closer to 0 = AM gives no info about PM)",
            "London -> NY Follow-Through r (FX only)",
        ],
        vertical_spacing=0.12,
    )

    acf_vals = [r.get("tr_lag1h_acf", float("nan")) for r in valid]
    fig.add_trace(go.Bar(x=syms, y=acf_vals,
                         marker_color="steelblue", name="Vol ACF lag-1h"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", row=1, col=1)

    am_pm_r = [r.get("am_pm_r", float("nan")) for r in valid]
    colours  = ["green" if abs(v) > 0.1 else "grey" for v in am_pm_r]
    fig.add_trace(go.Bar(x=syms, y=am_pm_r,
                         marker_color=colours, name="AM/PM r"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", row=2, col=1)

    lon_r    = [r.get("london_ny_r", float("nan")) for r in valid]
    fig.add_trace(go.Bar(x=syms, y=lon_r,
                         marker_color="orange", name="London->NY r"), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", row=3, col=1)

    fig.update_layout(title="M5 Pattern Analysis — Cross-Instrument Summary",
                      height=750, showlegend=False)
    path = BASE_OUT / "00_summary.html"
    fig.write_html(str(path))
    print(f"\n  Summary chart -> {path}")

    # Print text table
    print("\n" + "=" * 75)
    print(f"  {'Symbol':<18} {'Days':>5} {'Vol ACF':>8} {'AM/PM r':>8} {'Lon->NY r':>10}  Sig patterns")
    print("  " + "-" * 73)
    for r in valid:
        sig_parts = []
        if r.get("sig_drift_hours"):
            sig_parts.append(f"hours={[h for h,_,_ in r['sig_drift_hours']]}")
        if r.get("sig_dow"):
            sig_parts.append(f"DOW={[d for d,_,_ in r['sig_dow']]}")
        if r.get("sig_months"):
            sig_parts.append(f"months={[m for m,_,_ in r['sig_months']]}")
        print(f"  {r['symbol']:<18} {r['n_days']:>5} "
              f"{r.get('tr_lag1h_acf', float('nan')):>8.3f} "
              f"{r.get('am_pm_r', float('nan')):>8.3f} "
              f"{r.get('london_ny_r', float('nan')):>10.3f}  "
              f"{', '.join(sig_parts) if sig_parts else 'none'}")

    skipped = [r for r in all_results if r.get("skipped")]
    if skipped:
        print(f"\n  Skipped (insufficient data): {[r['symbol'] for r in skipped]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 55)
    print("  M5 Pattern Analysis — All Instruments")
    print("=" * 55)

    m5_files = sorted(DATA_DIR.glob("*_M5.parquet"))
    print(f"  Found {len(m5_files)} M5 files: {[f.stem for f in m5_files]}")

    all_results = []
    for path in m5_files:
        result = run_instrument(path)
        if result:
            all_results.append(result)

    print("\n\nCross-instrument summary:")
    build_summary(all_results)
    print(f"\n  All outputs in: {BASE_OUT}")


if __name__ == "__main__":
    main()
