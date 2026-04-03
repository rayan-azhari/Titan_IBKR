"""explore_eurusd_2024.py — Pure EDA of EUR/USD M5, 2024-2026.

Goal: find consistent daily profiles, time-of-day, day-of-week, and
seasonal patterns. No trading model, no Sharpe ratios.

Outputs (all saved to .tmp/reports/eda_eurusd/):
  01_hourly_profile.html        — volatility + drift + volume by UTC hour
  02_dow_profile.html           — day-of-week directional and volatility profile
  03_intraday_avg_path.html     — mean cumulative return path by day-of-week
  04_hour_dow_heatmap.html      — 2D heatmap: hour × day-of-week drift
  05_monthly_profile.html       — month-of-year and week-of-year seasonality
  06_am_pm_correlation.html     — AM session return vs PM session return scatter
  07_open_gap_analysis.html     — Monday open gap distribution + close tendency
  08_session_transition.html    — London close → NY session directional follow-through
  09_volatility_clustering.html — does high-volatility hour predict next-hour vol?
  10_summary_table.html         — ranked "most consistent" patterns by effect size

Usage:
    uv run python research/intraday_profiles/explore_eurusd_2024.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data"
OUT_DIR     = PROJECT_ROOT / ".tmp" / "reports" / "eda_eurusd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load() -> pd.DataFrame:
    path = DATA_DIR / "EUR_USD_M5.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Filter to 2024-2026 only (continuous data)
    df = df["2024-01-01":]
    print(f"  Loaded EUR/USD M5 (2024-2026): {len(df):,} bars  "
          f"[{df.index[0].date()} -> {df.index[-1].date()}]")

    # Core derived columns
    df["bar_ret"]  = (df["close"] - df["open"]) / df["open"]     # M5 bar return
    df["abs_ret"]  = df["bar_ret"].abs()
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    df["tr"]       = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["clv"]      = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
                     (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"]      = df["clv"].fillna(0.0)

    df["hour"]     = df.index.hour
    df["minute"]   = df.index.minute
    df["dow"]      = df.index.day_of_week   # 0=Mon, 4=Fri
    df["month"]    = df.index.month
    df["week"]     = df.index.isocalendar().week.astype(int)
    df["date"]     = df.index.date
    df["hhmm"]     = df["hour"] * 60 + df["minute"]  # minute-of-day

    return df


# ---------------------------------------------------------------------------
# Helper: t-test annotation
# ---------------------------------------------------------------------------


def ttest_label(values: pd.Series) -> str:
    if len(values) < 10:
        return ""
    t, p = stats.ttest_1samp(values.dropna(), 0.0)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"p={p:.3f}{stars}"


# ---------------------------------------------------------------------------
# 01 — Hourly profile
# ---------------------------------------------------------------------------


def plot_hourly(df: pd.DataFrame) -> None:
    grp = df.groupby("hour").agg(
        abs_ret=("abs_ret", "mean"),
        drift=("bar_ret", "mean"),
        volume=("volume", "mean"),
        n=("bar_ret", "count"),
    ).reset_index()

    # t-test per hour
    pvals = []
    for h in grp["hour"]:
        vals = df.loc[df["hour"] == h, "bar_ret"].dropna()
        if len(vals) < 30:
            pvals.append(1.0)
        else:
            _, p = stats.ttest_1samp(vals, 0.0)
            pvals.append(p)
    grp["p_drift"] = pvals
    grp["sig"]     = grp["p_drift"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=[
                            "Volatility (mean |bar return|) by UTC hour",
                            "Directional Drift (mean bar return) by UTC hour — * = p<0.05",
                            "Mean Volume by UTC hour (normalised)",
                        ])
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["abs_ret"] * 10000,
                         name="Volatility (pips)", marker_color="steelblue"), row=1, col=1)

    colours = ["red" if d < 0 else "green" for d in grp["drift"]]
    labels  = [f"{d*10000:.2f}{s}" for d, s in zip(grp["drift"], grp["sig"])]
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["drift"] * 10000,
                         text=labels, textposition="outside",
                         name="Drift (pips)", marker_color=colours), row=2, col=1)

    mean_vol = grp["volume"].mean()
    fig.add_trace(go.Bar(x=grp["hour"], y=grp["volume"] / mean_vol,
                         name="Rel Volume", marker_color="purple"), row=3, col=1)

    fig.update_xaxes(tickvals=list(range(0, 24)), ticktext=[f"{h:02d}h" for h in range(24)])
    fig.update_yaxes(title_text="Pips", row=1, col=1)
    fig.update_yaxes(title_text="Pips", row=2, col=1)
    fig.update_yaxes(title_text="Rel Vol", row=3, col=1)
    fig.update_layout(title="EUR/USD M5 (2024-2026) — Hourly Profile", height=900, showlegend=False)

    path = OUT_DIR / "01_hourly_profile.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")

    # Print significant hours
    sig = grp[grp["p_drift"] < 0.05]
    if not sig.empty:
        print(f"     Significant drift hours (p<0.05): "
              f"{[(int(r.hour), round(r.drift*10000,2), r.sig) for _, r in sig.iterrows()]}")


# ---------------------------------------------------------------------------
# 02 — Day-of-week profile
# ---------------------------------------------------------------------------


def plot_dow(df: pd.DataFrame) -> None:
    df_wd = df[df["dow"] < 5]   # Mon-Fri only
    grp = df_wd.groupby("dow").agg(
        abs_ret=("abs_ret", "mean"),
        drift=("bar_ret", "mean"),
        volume=("volume", "mean"),
        n=("bar_ret", "count"),
    ).reset_index()
    grp["dow_name"] = grp["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

    # Also compute daily return per day-of-week
    daily = df_wd.groupby(["dow", "date"]).agg(
        day_ret=("log_ret", "sum"),
    ).reset_index()
    dow_daily = daily.groupby("dow").agg(
        mean_day_ret=("day_ret", "mean"),
        std_day_ret=("day_ret", "std"),
        n_days=("day_ret", "count"),
    ).reset_index()
    dow_daily["dow_name"] = dow_daily["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

    # t-test on daily returns
    pvals = []
    for d in dow_daily["dow"]:
        vals = daily.loc[daily["dow"] == d, "day_ret"]
        if len(vals) < 10:
            pvals.append(1.0)
        else:
            _, p = stats.ttest_1samp(vals, 0.0)
            pvals.append(p)
    dow_daily["p"] = pvals
    dow_daily["sig"] = dow_daily["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            "Mean M5 Bar Volatility by Day of Week",
                            "Mean Daily Return (%) by Day of Week — * = p<0.05",
                        ])
    fig.add_trace(go.Bar(x=grp["dow_name"], y=grp["abs_ret"] * 10000,
                         name="Volatility (pips)", marker_color="steelblue"), row=1, col=1)

    colours = ["red" if d < 0 else "green" for d in dow_daily["mean_day_ret"]]
    labels  = [f"{d*100:.3f}%{s}" for d, s in zip(dow_daily["mean_day_ret"], dow_daily["sig"])]
    fig.add_trace(go.Bar(x=dow_daily["dow_name"], y=dow_daily["mean_day_ret"] * 100,
                         text=labels, textposition="outside",
                         error_y=dict(type="data",
                                      array=(dow_daily["std_day_ret"] / np.sqrt(dow_daily["n_days"]) * 100).tolist()),
                         name="Mean Daily Ret", marker_color=colours), row=2, col=1)

    fig.update_layout(title="EUR/USD M5 (2024-2026) — Day-of-Week Profile", height=600, showlegend=False)
    path = OUT_DIR / "02_dow_profile.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")
    print(f"     DOW daily returns: {dict(zip(dow_daily['dow_name'], (dow_daily['mean_day_ret']*10000).round(1).tolist()))} (pips)")


# ---------------------------------------------------------------------------
# 03 — Average intraday cumulative path by day-of-week
# ---------------------------------------------------------------------------


def plot_intraday_path_by_dow(df: pd.DataFrame) -> None:
    """Mean normalised cumulative return path for each day of week."""
    df_wd = df[df["dow"] < 5].copy()
    df_wd["cum_ret"] = df_wd.groupby("date")["log_ret"].cumsum()

    # Normalise by day std so we compare shape, not magnitude
    day_std = df_wd.groupby("date")["log_ret"].std().rename("day_std")
    df_wd   = df_wd.join(day_std, on="date")
    df_wd["cum_ret_norm"] = df_wd["cum_ret"] / df_wd["day_std"].replace(0, np.nan)

    fig = go.Figure()
    for dow in range(5):
        sub = df_wd[df_wd["dow"] == dow]
        avg = sub.groupby("hhmm")["cum_ret_norm"].mean()
        fig.add_trace(go.Scatter(
            x=avg.index, y=avg.values,
            name=DOW_NAMES[dow],
            mode="lines",
        ))

    # Mark session boundaries
    for h, label in [(0, "00:00 open"), (7, "London"), (12, "AM/PM cut"),
                     (13, "NY open"), (17, "London close"), (22, "NY close")]:
        fig.add_vline(x=h * 60, line_dash="dot", line_color="grey",
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title="EUR/USD M5 (2024-2026) — Mean Normalised Cumulative Return by Day-of-Week",
        xaxis_title="Minute of day (UTC)",
        yaxis_title="Normalised cumulative return",
        height=500,
    )
    path = OUT_DIR / "03_intraday_avg_path.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# 04 — Hour × Day-of-week drift heatmap
# ---------------------------------------------------------------------------


def plot_hour_dow_heatmap(df: pd.DataFrame) -> None:
    df_wd = df[df["dow"] < 5]
    pivot = df_wd.groupby(["dow", "hour"])["bar_ret"].mean().unstack("hour") * 10000

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=[DOW_NAMES[d] for d in pivot.index],
        colorscale="RdYlGn",
        zmid=0,
        text=pivot.round(2).values,
        texttemplate="%{text}",
        colorbar_title="Drift (pips)",
    ))
    fig.update_layout(
        title="EUR/USD M5 (2024-2026) — Mean Bar Drift (pips) by UTC Hour × Day-of-Week",
        xaxis_title="UTC Hour",
        yaxis_title="Day of Week",
        height=350,
    )
    path = OUT_DIR / "04_hour_dow_heatmap.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# 05 — Monthly and weekly seasonality
# ---------------------------------------------------------------------------


def plot_seasonal(df: pd.DataFrame) -> None:
    # Group by calendar date first, then aggregate
    daily = df[df["dow"] < 5].groupby("date").agg(
        day_ret=("log_ret", "sum"),
        month=("month", "first"),
        week=("week", "first"),
    ).reset_index()

    # Monthly
    monthly = daily.groupby("month").agg(
        mean_ret=("day_ret", "mean"),
        std_ret=("day_ret", "std"),
        n=("day_ret", "count"),
    ).reset_index()

    pvals = []
    for m in monthly["month"]:
        vals = daily.loc[daily["month"] == m, "day_ret"]
        if len(vals) < 5:
            pvals.append(1.0)
        else:
            _, p = stats.ttest_1samp(vals, 0.0)
            pvals.append(p)
    monthly["p"]   = pvals
    monthly["sig"] = monthly["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    monthly["name"] = monthly["month"].apply(lambda m: MONTH_NAMES[m - 1])

    fig = make_subplots(rows=1, cols=1)
    colours = ["red" if d < 0 else "green" for d in monthly["mean_ret"]]
    labels  = [f"{d*100:.3f}%{s}" for d, s in zip(monthly["mean_ret"], monthly["sig"])]
    fig.add_trace(go.Bar(
        x=monthly["name"], y=monthly["mean_ret"] * 100,
        text=labels, textposition="outside",
        error_y=dict(type="data",
                     array=(monthly["std_ret"] / np.sqrt(monthly["n"]) * 100).tolist()),
        marker_color=colours,
        name="Mean Daily Ret",
    ))

    # Add sample size annotation
    for _, row in monthly.iterrows():
        fig.add_annotation(x=row["name"], y=-0.015, text=f"n={int(row['n'])}",
                           showarrow=False, font=dict(size=9))

    fig.update_layout(
        title="EUR/USD M5 (2024-2026) — Mean Daily Return by Calendar Month — * = p<0.05",
        yaxis_title="Mean Daily Return (%)",
        height=450,
        showlegend=False,
    )
    path = OUT_DIR / "05_monthly_profile.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")
    sig_months = monthly[monthly["p"] < 0.05]
    if not sig_months.empty:
        print(f"     Significant months: {list(zip(sig_months['name'], sig_months['mean_ret'].mul(10000).round(1)))}")


# ---------------------------------------------------------------------------
# 06 — AM return vs PM return (scatter + correlation)
# ---------------------------------------------------------------------------


def plot_am_pm_correlation(df: pd.DataFrame) -> None:
    """Does the AM session return predict the PM session return?"""
    df_wd = df[df["dow"] < 5].copy()

    # AM: 00:00–12:00 UTC cumulative log return
    # PM: 12:00–22:00 UTC cumulative log return
    am = df_wd[df_wd["hour"] < 12].groupby("date")["log_ret"].sum().rename("am_ret")
    pm = df_wd[(df_wd["hour"] >= 12) & (df_wd["hour"] < 22)].groupby("date")["log_ret"].sum().rename("pm_ret")

    both = pd.concat([am, pm], axis=1).dropna()
    both["dow"] = pd.to_datetime(both.index.astype(str)).day_of_week
    both["dow_name"] = both["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

    r, p = stats.pearsonr(both["am_ret"], both["pm_ret"])
    print(f"     AM vs PM correlation: r={r:.3f}  p={p:.3f}")

    # Also split by direction: AM up vs AM down -> what does PM do?
    am_up   = both[both["am_ret"] > 0]["pm_ret"]
    am_down = both[both["am_ret"] < 0]["pm_ret"]
    am_flat = both[both["am_ret"].abs() < both["am_ret"].std() * 0.5]["pm_ret"]

    print(f"     PM return when AM up   (n={len(am_up)}):   mean={am_up.mean()*10000:.1f} pips")
    print(f"     PM return when AM down (n={len(am_down)}): mean={am_down.mean()*10000:.1f} pips")
    print(f"     PM return when AM flat (n={len(am_flat)}): mean={am_flat.mean()*10000:.1f} pips")

    # Scatter coloured by day of week
    fig = px.scatter(
        both, x="am_ret", y="pm_ret", color="dow_name",
        labels={"am_ret": "AM Return (00:00-12:00 UTC)", "pm_ret": "PM Return (12:00-22:00 UTC)"},
        title=f"EUR/USD M5 (2024-2026) — AM vs PM Session Return  (r={r:.3f}, p={p:.3f})",
    )
    # Add regression line
    xr = np.linspace(both["am_ret"].min(), both["am_ret"].max(), 100)
    slope, intercept, *_ = stats.linregress(both["am_ret"], both["pm_ret"])
    fig.add_trace(go.Scatter(x=xr, y=slope * xr + intercept,
                             mode="lines", name="OLS fit", line=dict(color="black", dash="dash")))
    fig.update_layout(height=500)
    path = OUT_DIR / "06_am_pm_correlation.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# 07 — Monday open gap + close tendency
# ---------------------------------------------------------------------------


def plot_monday_gap(df: pd.DataFrame) -> None:
    """Monday opens with a gap from Friday close. Does the gap close?"""
    mondays = df[df["dow"] == 0].copy()
    # First bar of each Monday = gap relative to last Friday close
    first_bar = mondays.groupby("date").first().reset_index()
    first_bar["gap"] = (first_bar["open"] - first_bar["open"].shift(5)).dropna()  # rough

    # Better: compare Monday open to last Friday's last close
    friday_close = df[df["dow"] == 4].groupby("date")["close"].last()
    monday_open  = df[df["dow"] == 0].groupby("date")["open"].first()
    monday_close = df[df["dow"] == 0].groupby("date")["close"].last()

    gap_df = pd.DataFrame({
        "fri_close":   friday_close,
        "mon_open":    monday_open,
        "mon_close":   monday_close,
    }).dropna()
    gap_df["gap"]        = (gap_df["mon_open"] - gap_df["fri_close"]) * 10000   # pips
    gap_df["mon_return"] = (gap_df["mon_close"] - gap_df["mon_open"]) * 10000   # pips
    gap_df["gap_closed"] = (gap_df["gap"] * gap_df["mon_return"] < 0).astype(int)

    print(f"     Monday gap stats: mean={gap_df['gap'].mean():.1f} pips  "
          f"std={gap_df['gap'].std():.1f} pips  "
          f"gap-closed rate={gap_df['gap_closed'].mean():.1%}")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Monday Open Gap Distribution (pips)",
                                        "Gap vs Monday Return: does gap close?"])
    fig.add_trace(go.Histogram(x=gap_df["gap"], nbinsx=40, name="Gap (pips)",
                               marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=gap_df["gap"], y=gap_df["mon_return"],
                             mode="markers", name="Gap vs Monday Return",
                             marker=dict(color=gap_df["gap_closed"],
                                         colorscale=["red", "green"], size=6)),
                 row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", row=1, col=2)
    fig.add_vline(x=0, line_dash="dot", row=1, col=2)
    fig.update_layout(title="EUR/USD M5 (2024-2026) — Monday Gap Analysis", height=450, showlegend=False)
    path = OUT_DIR / "07_monday_gap.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# 08 — Session transition: London → NY follow-through
# ---------------------------------------------------------------------------


def plot_session_transition(df: pd.DataFrame) -> None:
    """Does the London session direction continue or reverse into NY?"""
    df_wd = df[df["dow"] < 5].copy()

    london = df_wd[(df_wd["hour"] >= 7) & (df_wd["hour"] < 12)].groupby("date")["log_ret"].sum().rename("london")
    ny     = df_wd[(df_wd["hour"] >= 13) & (df_wd["hour"] < 17)].groupby("date")["log_ret"].sum().rename("ny_open")

    both = pd.concat([london, ny], axis=1).dropna()

    # When London up / down, what does NY opening 2h do?
    lon_up   = both[both["london"] > 0]["ny_open"]
    lon_down = both[both["london"] < 0]["ny_open"]

    _, p_up   = stats.ttest_1samp(lon_up, 0)
    _, p_down = stats.ttest_1samp(lon_down, 0)

    print(f"     NY open (13-17h) when London up   (n={len(lon_up)}):   "
          f"mean={lon_up.mean()*10000:.1f} pips  p={p_up:.3f}")
    print(f"     NY open (13-17h) when London down (n={len(lon_down)}): "
          f"mean={lon_down.mean()*10000:.1f} pips  p={p_down:.3f}")

    r, p = stats.pearsonr(both["london"], both["ny_open"])
    print(f"     London vs NY correlation: r={r:.3f}  p={p:.3f}")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            "NY Open Return distribution by London direction",
                            "London Return vs NY Open Return scatter",
                        ])
    fig.add_trace(go.Histogram(x=lon_up   * 10000, name="London Up -> NY", opacity=0.7,
                               marker_color="green", nbinsx=40), row=1, col=1)
    fig.add_trace(go.Histogram(x=lon_down * 10000, name="London Down -> NY", opacity=0.7,
                               marker_color="red", nbinsx=40), row=1, col=1)
    fig.add_trace(go.Scatter(x=both["london"] * 10000, y=both["ny_open"] * 10000,
                             mode="markers", name="Daily obs",
                             marker=dict(color="steelblue", size=5, opacity=0.6)), row=1, col=2)
    xr = np.linspace(both["london"].min(), both["london"].max(), 100) * 10000
    slope, intercept, *_ = stats.linregress(both["london"] * 10000, both["ny_open"] * 10000)
    fig.add_trace(go.Scatter(x=xr, y=slope * xr + intercept, mode="lines",
                             name="OLS", line=dict(color="black", dash="dash")), row=1, col=2)
    fig.update_layout(
        title=f"EUR/USD M5 (2024-2026) — London -> NY Follow-Through  (r={r:.3f}, p={p:.3f})",
        height=450, barmode="overlay",
    )
    path = OUT_DIR / "08_session_transition.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# 09 — Volatility autocorrelation (does high vol hour predict next-hour vol?)
# ---------------------------------------------------------------------------


def plot_vol_clustering(df: pd.DataFrame) -> None:
    """True range autocorrelation by lag."""
    df_wd = df[df["dow"] < 5].copy()
    # Hourly aggregated TR
    hourly_tr = df_wd.groupby([df_wd.index.date, "hour"])["tr"].mean()
    hourly_tr = hourly_tr.reset_index(level=1)["tr"]

    lags    = list(range(1, 25))
    autocorrs = [hourly_tr.autocorr(lag=lag) for lag in lags]

    # Also compute 5-min autocorr
    bar_autocorrs = [df_wd["tr"].autocorr(lag=lag) for lag in range(1, 13)]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            "Hourly True Range Autocorrelation (lag in hours)",
                            "M5 Bar True Range Autocorrelation (lag in M5 bars)",
                        ])
    fig.add_trace(go.Bar(x=lags, y=autocorrs, name="Hourly TR ACF",
                         marker_color=["red" if a < 0 else "steelblue" for a in autocorrs]),
                 row=1, col=1)
    fig.add_hline(y=1.96 / np.sqrt(len(hourly_tr)), line_dash="dot", line_color="orange",
                  annotation_text="95% CI", row=1, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(len(hourly_tr)), line_dash="dot", line_color="orange",
                  row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(1, 13)), y=bar_autocorrs, name="M5 TR ACF",
                         marker_color=["red" if a < 0 else "steelblue" for a in bar_autocorrs]),
                 row=2, col=1)
    fig.add_hline(y=1.96 / np.sqrt(len(df_wd)), line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(len(df_wd)), line_dash="dot", line_color="orange", row=2, col=1)

    fig.update_layout(title="EUR/USD M5 (2024-2026) — Volatility Autocorrelation", height=600, showlegend=False)
    path = OUT_DIR / "09_volatility_clustering.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")

    print(f"     Lag-1 hourly TR ACF: {autocorrs[0]:.3f}  (strong = vol clusters across hours)")
    print(f"     Lag-1 M5 TR ACF:    {bar_autocorrs[0]:.3f}")


# ---------------------------------------------------------------------------
# 10 — CLV direction by hour × DOW (who wins each 5-min slot?)
# ---------------------------------------------------------------------------


def plot_clv_heatmap(df: pd.DataFrame) -> None:
    """Mean CLV by UTC hour and day-of-week — shows who dominates each slot."""
    df_wd = df[df["dow"] < 5]
    pivot = df_wd.groupby(["dow", "hour"])["clv"].mean().unstack("hour")

    # Statistical significance: t-test CLV vs 0 per cell
    sig_pivot = pd.DataFrame(index=pivot.index, columns=pivot.columns, data="")
    for dow in pivot.index:
        for hour in pivot.columns:
            vals = df_wd[(df_wd["dow"] == dow) & (df_wd["hour"] == hour)]["clv"]
            if len(vals) >= 30:
                _, p = stats.ttest_1samp(vals, 0.0)
                sig_pivot.loc[dow, hour] = "★" if p < 0.01 else ("·" if p < 0.05 else "")

    text = pivot.round(3).astype(str) + sig_pivot.values

    fig = go.Figure(go.Heatmap(
        z=pivot.values.astype(float),
        x=[f"{h:02d}h" for h in pivot.columns],
        y=[DOW_NAMES[d] for d in pivot.index],
        colorscale="RdYlGn",
        zmid=0,
        text=text.values,
        texttemplate="%{text}",
        colorbar_title="Mean CLV",
    ))
    fig.update_layout(
        title="EUR/USD M5 (2024-2026) — Mean CLV by Hour × DOW  (★=p<0.01, ·=p<0.05)",
        height=350,
    )
    path = OUT_DIR / "10_clv_heatmap.html"
    fig.write_html(str(path))
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("EUR/USD M5 2024-2026 — Exploratory Pattern Analysis")
    print("=" * 60)

    df = load()

    print("\n[01] Hourly volatility / drift / volume profile")
    plot_hourly(df)

    print("\n[02] Day-of-week profile")
    plot_dow(df)

    print("\n[03] Average intraday cumulative path by day-of-week")
    plot_intraday_path_by_dow(df)

    print("\n[04] Hour x Day-of-week drift heatmap")
    plot_hour_dow_heatmap(df)

    print("\n[05] Monthly seasonality")
    plot_seasonal(df)

    print("\n[06] AM vs PM session return correlation")
    plot_am_pm_correlation(df)

    print("\n[07] Monday open gap analysis")
    plot_monday_gap(df)

    print("\n[08] London -> NY session follow-through")
    plot_session_transition(df)

    print("\n[09] Volatility autocorrelation")
    plot_vol_clustering(df)

    print("\n[10] CLV by hour x DOW heatmap")
    plot_clv_heatmap(df)

    print(f"\n  All charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
