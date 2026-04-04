"""explore_h1.py — EDA across H1 instruments (equity + FX).

Same pattern analysis as explore_all_m5.py but for H1 bars:
  - Hourly drift / volatility / volume profile
  - Day-of-week directional profile
  - Mean normalised intraday cumulative path by DOW
  - Hour × DOW drift heatmap
  - Monthly / seasonal profile
  - AM/PM session correlation (AM = pre-13h UTC, PM = 13-21h UTC for equity)
  - Opening-hour drift (first hour of session — strongest edge in equities)
  - Closing-hour drift (last hour — institutional rebalancing)
  - Volatility autocorrelation (lag-1h, 2h, 3h)
  - CLV heatmap

Usage:
    # Single instrument:
    uv run python research/intraday_profiles/explore_h1.py --symbol SPY

    # All H1 instruments with >= MIN_DAYS trading days:
    uv run python research/intraday_profiles/explore_h1.py --all

    # Filter to specific list:
    uv run python research/intraday_profiles/explore_h1.py --symbols SPY QQQ GLD
"""

from __future__ import annotations

import argparse
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
BASE_OUT = PROJECT_ROOT / ".tmp" / "reports" / "eda_h1"
BASE_OUT.mkdir(parents=True, exist_ok=True)

MIN_TRADING_DAYS = 200  # ~1 year minimum for seasonal patterns

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# US equity session hours UTC
SESSION_OPEN_UTC = 14  # 09:30 ET ≈ 14:30 UTC → first full H1 bar at 14h
SESSION_CLOSE_UTC = 21  # 16:00 ET = 21:00 UTC


# ---------------------------------------------------------------------------
# Loading + enrichment
# ---------------------------------------------------------------------------


def load_h1(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol}_H1.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # Skip files where timestamps are clearly corrupt (epoch 1970)
    if df.index[-1].year < 2000:
        return None
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_c = df["close"].shift(1)
    df["bar_ret"] = (df["close"] - df["open"]) / df["open"]
    df["abs_ret"] = df["bar_ret"].abs()
    df["log_ret"] = np.log(df["close"] / prev_c)
    df["tr"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"] - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    df["clv"] = df["clv"].fillna(0.0)
    df["hour"] = df.index.hour
    df["dow"] = df.index.day_of_week
    df["month"] = df.index.month
    df["hhmm"] = df.index.hour * 60 + df.index.minute
    df["date"] = df.index.date
    # Session flags
    df["in_session"] = (df["hour"] >= SESSION_OPEN_UTC) & (df["hour"] < SESSION_CLOSE_UTC)
    return df


def detect_type(df: pd.DataFrame) -> str:
    outside = ((df["hour"] < 12) | (df["hour"] >= 22)).mean()
    return "fx" if outside > 0.10 else "equity"


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------


def plot_hourly(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    """Volatility, drift, volume by UTC hour."""
    grp = (
        df.groupby("hour")
        .agg(
            abs_ret=("abs_ret", "mean"),
            drift=("bar_ret", "mean"),
            volume=("volume", "mean"),
            n=("bar_ret", "count"),
        )
        .reset_index()
    )

    pvals = []
    for h in grp["hour"]:
        vals = df.loc[df["hour"] == h, "bar_ret"].dropna()
        pvals.append(stats.ttest_1samp(vals, 0.0)[1] if len(vals) >= 30 else 1.0)
    grp["p"] = pvals
    grp["sig"] = grp["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Volatility — mean |bar return| (%)",
            "Directional Drift — mean bar return (%)  * = p<0.05",
            "Mean Volume (normalised)",
        ],
    )
    fig.add_trace(
        go.Bar(x=grp["hour"], y=grp["abs_ret"] * 100, marker_color="steelblue"), row=1, col=1
    )

    colours = ["red" if d < 0 else "green" for d in grp["drift"]]
    labels = [f"{d * 100:.3f}%{s}" for d, s in zip(grp["drift"], grp["sig"])]
    fig.add_trace(
        go.Bar(
            x=grp["hour"],
            y=grp["drift"] * 100,
            text=labels,
            textposition="outside",
            marker_color=colours,
        ),
        row=2,
        col=1,
    )

    mv = grp["volume"].mean()
    fig.add_trace(
        go.Bar(x=grp["hour"], y=grp["volume"] / max(mv, 1e-9), marker_color="purple"), row=3, col=1
    )

    # Mark session boundaries
    for h, lbl in [(SESSION_OPEN_UTC, "open"), (SESSION_CLOSE_UTC, "close")]:
        for row in [1, 2, 3]:
            fig.add_vline(x=h, line_dash="dot", line_color="grey", row=row, col=1)

    fig.update_xaxes(tickvals=list(range(8, 23)), ticktext=[f"{h:02d}h" for h in range(8, 23)])
    fig.update_layout(title=f"{symbol} H1 — Hourly Profile (UTC)", height=850, showlegend=False)
    fig.write_html(str(out / "01_hourly_profile.html"))

    sig = grp[grp["p"] < 0.05]
    return {
        "sig_drift_hours": [
            (int(r.hour), round(r.drift * 100, 4), r.sig) for _, r in sig.iterrows()
        ],
    }


def plot_open_close_hours(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    """Focus on first and last hour of session — strongest equity patterns."""
    df_sess = df[df["dow"] < 5]
    is_fx = detect_type(df)

    if is_fx == "equity":
        first_hour = SESSION_OPEN_UTC  # 14h UTC
        last_hour = SESSION_CLOSE_UTC - 1  # 20h UTC
    else:
        first_hour = 7
        last_hour = 21

    results = {}
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"First session hour ({first_hour}h UTC) return distribution",
            f"Last session hour ({last_hour}h UTC) return distribution",
        ],
    )

    for col, hour, label in [(1, first_hour, "Open"), (2, last_hour, "Close")]:
        vals = df_sess[df_sess["hour"] == hour]["bar_ret"].dropna() * 100
        if len(vals) < 20:
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        colour = "green" if vals.mean() > 0 else "red"
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=50, name=label, marker_color=colour, opacity=0.7),
            row=1,
            col=col,
        )
        fig.add_vline(
            x=vals.mean(),
            line_dash="dash",
            line_color="black",
            row=1,
            col=col,
            annotation_text=f"mean={vals.mean():.3f}%{sig}",
        )
        results[f"{label.lower()}_hour_mean_pct"] = round(float(vals.mean()), 4)
        results[f"{label.lower()}_hour_p"] = round(p, 4)
        results[f"{label.lower()}_hour_sig"] = sig

    fig.update_layout(title=f"{symbol} H1 — Open vs Close Hour Distribution", height=400)
    fig.write_html(str(out / "02_open_close_hours.html"))
    return results


def plot_dow(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    daily = df_wd.groupby(["dow", "date"]).agg(day_ret=("log_ret", "sum")).reset_index()
    dow_d = (
        daily.groupby("dow")
        .agg(
            mean=("day_ret", "mean"),
            std=("day_ret", "std"),
            n=("day_ret", "count"),
        )
        .reset_index()
    )
    dow_d["name"] = dow_d["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

    pvals = []
    for d in dow_d["dow"]:
        vals = daily.loc[daily["dow"] == d, "day_ret"]
        pvals.append(stats.ttest_1samp(vals, 0.0)[1] if len(vals) >= 10 else 1.0)
    dow_d["p"] = pvals
    dow_d["sig"] = dow_d["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    vol_grp = df_wd.groupby("dow")["abs_ret"].mean().reset_index()
    vol_grp["name"] = vol_grp["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "Mean Hourly Volatility by Day of Week",
            "Mean Daily Return by Day of Week — * = p<0.05",
        ],
    )
    fig.add_trace(
        go.Bar(x=vol_grp["name"], y=vol_grp["abs_ret"] * 100, marker_color="steelblue"),
        row=1,
        col=1,
    )

    colours = ["red" if d < 0 else "green" for d in dow_d["mean"]]
    labels = [f"{d * 100:.3f}%{s}" for d, s in zip(dow_d["mean"], dow_d["sig"])]
    fig.add_trace(
        go.Bar(
            x=dow_d["name"],
            y=dow_d["mean"] * 100,
            text=labels,
            textposition="outside",
            error_y=dict(type="data", array=(dow_d["std"] / np.sqrt(dow_d["n"]) * 100).tolist()),
            marker_color=colours,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(title=f"{symbol} H1 — Day-of-Week Profile", height=600, showlegend=False)
    fig.write_html(str(out / "03_dow_profile.html"))

    sig = dow_d[dow_d["p"] < 0.05]
    return {
        "dow_means_pct": dict(zip(dow_d["name"], dow_d["mean"].mul(100).round(4))),
        "sig_dow": [(r["name"], round(r["mean"] * 100, 4), r["sig"]) for _, r in sig.iterrows()],
    }


def plot_intraday_path_by_dow(df: pd.DataFrame, symbol: str, out: Path) -> None:
    df_wd = df[(df["dow"] < 5) & df["in_session"]].copy()
    df_wd["cum_ret"] = df_wd.groupby("date")["log_ret"].cumsum()
    day_std = df_wd.groupby("date")["log_ret"].std().rename("day_std")
    df_wd = df_wd.join(day_std, on="date")
    df_wd["cum_norm"] = df_wd["cum_ret"] / df_wd["day_std"].replace(0, np.nan)

    fig = go.Figure()
    for dow in range(5):
        sub = df_wd[df_wd["dow"] == dow]
        if len(sub) == 0:
            continue
        avg = sub.groupby("hour")["cum_norm"].mean()
        fig.add_trace(
            go.Scatter(x=avg.index, y=avg.values, name=DOW_NAMES[dow], mode="lines+markers")
        )

    fig.add_vline(x=SESSION_OPEN_UTC, line_dash="dot", line_color="grey", annotation_text="open")
    fig.add_vline(x=SESSION_CLOSE_UTC, line_dash="dot", line_color="grey", annotation_text="close")
    fig.update_layout(
        title=f"{symbol} H1 — Mean Normalised Intraday Path by DOW",
        xaxis_title="UTC Hour",
        yaxis_title="Normalised cumulative return",
        height=450,
    )
    fig.write_html(str(out / "04_intraday_path_dow.html"))


def plot_hour_dow_heatmap(df: pd.DataFrame, symbol: str, out: Path) -> None:
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    pivot = df_wd.groupby(["dow", "hour"])["bar_ret"].mean().unstack("hour") * 100

    sig = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for dow in pivot.index:
        for hour in pivot.columns:
            vals = df_wd[(df_wd["dow"] == dow) & (df_wd["hour"] == hour)]["bar_ret"]
            if len(vals) >= 30:
                _, p = stats.ttest_1samp(vals, 0.0)
                sig.loc[dow, hour] = "★" if p < 0.01 else ("·" if p < 0.05 else "")

    text = pivot.round(3).astype(str) + sig.values

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values.astype(float),
            x=[f"{h:02d}h" for h in pivot.columns],
            y=[DOW_NAMES[d] for d in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=text.values,
            texttemplate="%{text}",
            colorbar_title="Drift (%)",
        )
    )
    fig.update_layout(
        title=f"{symbol} H1 — Drift (%) by Hour × DOW  (★=p<0.01, ·=p<0.05)",
        height=320,
    )
    fig.write_html(str(out / "05_hour_dow_heatmap.html"))


def plot_monthly(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    daily = (
        df_wd.groupby("date")
        .agg(
            day_ret=("log_ret", "sum"),
            month=("month", "first"),
        )
        .reset_index()
    )
    monthly = (
        daily.groupby("month")
        .agg(
            mean=("day_ret", "mean"),
            std=("day_ret", "std"),
            n=("day_ret", "count"),
        )
        .reset_index()
    )

    pvals = []
    for m in monthly["month"]:
        vals = daily.loc[daily["month"] == m, "day_ret"]
        pvals.append(stats.ttest_1samp(vals, 0.0)[1] if len(vals) >= 10 else 1.0)
    monthly["p"] = pvals
    monthly["sig"] = monthly["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    monthly["name"] = monthly["month"].apply(lambda m: MONTH_NAMES[m - 1])

    colours = ["red" if d < 0 else "green" for d in monthly["mean"]]
    labels = [f"{d * 100:.3f}%{s}" for d, s in zip(monthly["mean"], monthly["sig"])]
    fig = go.Figure(
        go.Bar(
            x=monthly["name"],
            y=monthly["mean"] * 100,
            text=labels,
            textposition="outside",
            error_y=dict(
                type="data",
                array=(monthly["std"] / np.sqrt(monthly["n"].clip(lower=1)) * 100).tolist(),
            ),
            marker_color=colours,
        )
    )
    for _, row in monthly.iterrows():
        fig.add_annotation(
            x=row["name"],
            y=monthly["mean"].min() * 100 - 0.05,
            text=f"n={int(row['n'])}",
            showarrow=False,
            font=dict(size=9),
        )
    fig.update_layout(
        title=f"{symbol} H1 — Mean Daily Return by Month  * = p<0.05",
        yaxis_title="%",
        height=430,
        showlegend=False,
    )
    fig.write_html(str(out / "06_monthly_profile.html"))

    sig = monthly[monthly["p"] < 0.05]
    return {
        "sig_months": [(r["name"], round(r["mean"] * 100, 4), r["sig"]) for _, r in sig.iterrows()]
    }


def plot_am_pm(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    """Early session vs late session correlation."""
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    is_fx = detect_type(df)
    mid_h = 17 if not is_fx == "fx" else 12  # equity midpoint: 17h UTC = noon ET

    early = df_wd[df_wd["hour"] < mid_h].groupby("date")["log_ret"].sum().rename("early")
    late = df_wd[df_wd["hour"] >= mid_h].groupby("date")["log_ret"].sum().rename("late")
    both = pd.concat([early, late], axis=1).dropna()
    if len(both) < 20:
        return {}

    r, p = stats.pearsonr(both["early"], both["late"])

    # Conditional means: when early session up/down, what does late session do?
    early_up = both[both["early"] > 0]["late"]
    early_down = both[both["early"] < 0]["late"]
    _, p_up = stats.ttest_1samp(early_up, 0.0) if len(early_up) >= 10 else (0, 1.0)
    _, p_down = stats.ttest_1samp(early_down, 0.0) if len(early_down) >= 10 else (0, 1.0)

    both["dow"] = pd.to_datetime(both.index.astype(str)).day_of_week
    both["dow_name"] = both["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})
    fig = px.scatter(
        both,
        x="early",
        y="late",
        color="dow_name",
        opacity=0.5,
        labels={
            "early": f"Early session (<{mid_h}h UTC)",
            "late": f"Late session (>={mid_h}h UTC)",
        },
        title=f"{symbol} H1 — Early vs Late Session  (r={r:.3f}, p={p:.3f})",
    )
    xr = np.linspace(both["early"].min(), both["early"].max(), 100)
    slope, intercept, *_ = stats.linregress(both["early"], both["late"])
    fig.add_trace(
        go.Scatter(
            x=xr,
            y=slope * xr + intercept,
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="OLS",
        )
    )
    fig.update_layout(height=450)
    fig.write_html(str(out / "07_early_late_correlation.html"))

    return {
        "early_late_r": round(r, 3),
        "early_late_p": round(p, 3),
        "late_when_early_up_pct": round(float(early_up.mean()) * 100, 4),
        "late_when_early_down_pct": round(float(early_down.mean()) * 100, 4),
        "p_late_when_early_up": round(p_up, 3),
        "p_late_when_early_down": round(p_down, 3),
    }


def plot_vol_autocorr(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    lags = list(range(1, 8))
    acf = [df_wd["tr"].autocorr(lag=l) for l in lags]
    ci = 1.96 / np.sqrt(max(len(df_wd), 1))

    fig = go.Figure(
        go.Bar(
            x=lags,
            y=acf,
            marker_color=["red" if a < 0 else "steelblue" for a in acf],
        )
    )
    fig.add_hline(y=ci, line_dash="dot", line_color="orange", annotation_text="95% CI")
    fig.add_hline(y=-ci, line_dash="dot", line_color="orange")
    fig.update_layout(
        title=f"{symbol} H1 — True Range Autocorrelation (session bars only)",
        xaxis_title="Lag (H1 bars)",
        yaxis_title="ACF",
        height=380,
        showlegend=False,
    )
    fig.write_html(str(out / "08_vol_autocorr.html"))
    return {"tr_lag1_acf": round(acf[0], 3), "tr_lag2_acf": round(acf[1], 3)}


def plot_clv_heatmap(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    df_wd = df[(df["dow"] < 5) & df["in_session"]]
    pivot = df_wd.groupby(["dow", "hour"])["clv"].mean().unstack("hour")

    sig = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    sig_cells: list[tuple[str, int, float, float, str]] = []  # (dow, hour, mean_clv, p, marker)
    for dow in pivot.index:
        for hour in pivot.columns:
            vals = df_wd[(df_wd["dow"] == dow) & (df_wd["hour"] == hour)]["clv"]
            if len(vals) >= 30:
                _, p = stats.ttest_1samp(vals, 0.0)
                if p < 0.01:
                    marker = "★"
                elif p < 0.05:
                    marker = "·"
                else:
                    marker = ""
                sig.loc[dow, hour] = marker
                if p < 0.05:
                    sig_cells.append(
                        (
                            DOW_NAMES[dow],
                            int(hour),
                            round(float(pivot.loc[dow, hour]), 4),
                            round(float(p), 4),
                            marker,
                        )
                    )

    text = pivot.round(3).astype(str) + sig.values
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values.astype(float),
            x=[f"{h:02d}h" for h in pivot.columns],
            y=[DOW_NAMES[d] for d in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=text.values,
            texttemplate="%{text}",
            colorbar_title="CLV",
        )
    )
    fig.update_layout(
        title=f"{symbol} H1 — Mean CLV by Hour × DOW  (★=p<0.01, ·=p<0.05)",
        height=320,
    )
    fig.write_html(str(out / "09_clv_heatmap.html"))
    # Sort: strongest signal first (lowest p)
    sig_cells.sort(key=lambda x: x[3])
    return {"sig_clv_cells": sig_cells}


def plot_yearly_drift(df: pd.DataFrame, symbol: str, out: Path) -> dict:
    """Year-over-year drift — is there a consistent positive bias?"""
    df_wd = df[(df["dow"] < 5) & df["in_session"]].copy()
    df_wd["year"] = df_wd.index.year
    yearly = df_wd.groupby(["year", "date"]).agg(day_ret=("log_ret", "sum")).reset_index()
    yr_grp = (
        yearly.groupby("year")
        .agg(
            mean=("day_ret", "mean"),
            std=("day_ret", "std"),
            n=("day_ret", "count"),
        )
        .reset_index()
    )

    colours = ["red" if d < 0 else "steelblue" for d in yr_grp["mean"]]
    fig = go.Figure(
        go.Bar(
            x=yr_grp["year"].astype(str),
            y=yr_grp["mean"] * 100,
            error_y=dict(type="data", array=(yr_grp["std"] / np.sqrt(yr_grp["n"]) * 100).tolist()),
            marker_color=colours,
            text=[f"{v:.3f}%" for v in yr_grp["mean"] * 100],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{symbol} H1 — Mean Daily Return by Year",
        yaxis_title="%",
        height=400,
        showlegend=False,
    )
    fig.write_html(str(out / "10_yearly_drift.html"))

    pos_years = int((yr_grp["mean"] > 0).sum())
    neg_years = int((yr_grp["mean"] <= 0).sum())
    return {
        "pos_years": pos_years,
        "neg_years": neg_years,
        "pct_positive_years": round(pos_years / max(len(yr_grp), 1), 2),
    }


# ---------------------------------------------------------------------------
# Per-instrument runner
# ---------------------------------------------------------------------------


def run_instrument(symbol: str) -> dict | None:
    df_raw = load_h1(symbol)
    if df_raw is None:
        print(f"  {symbol}: file not found or corrupt — skip")
        return None

    n_days = len(np.unique(df_raw.index.date))
    if n_days < MIN_TRADING_DAYS:
        print(f"  {symbol}: only {n_days} days — skip")
        return {"symbol": symbol, "skipped": True, "n_days": n_days}

    df = enrich(df_raw)
    is_type = detect_type(df)
    print(f"\n{'=' * 55}")
    print(f"  {symbol}  ({is_type}, {n_days} days, {df.index[0].date()} -> {df.index[-1].date()})")
    print(f"{'=' * 55}")

    out = BASE_OUT / symbol
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "symbol": symbol,
        "skipped": False,
        "n_days": n_days,
        "type": is_type,
        "date_range": f"{df.index[0].date()} -> {df.index[-1].date()}",
    }

    results.update(plot_hourly(df, symbol, out))
    results.update(plot_open_close_hours(df, symbol, out))
    results.update(plot_dow(df, symbol, out))
    plot_intraday_path_by_dow(df, symbol, out)
    plot_hour_dow_heatmap(df, symbol, out)
    results.update(plot_monthly(df, symbol, out))
    results.update(plot_am_pm(df, symbol, out))
    results.update(plot_vol_autocorr(df, symbol, out))
    results.update(plot_clv_heatmap(df, symbol, out))
    results.update(plot_yearly_drift(df, symbol, out))

    # Print key findings
    if results.get("sig_drift_hours"):
        print(f"  ** Sig drift hours:  {results['sig_drift_hours']}")
    if results.get("sig_dow"):
        print(f"  ** Sig DOW:          {results['sig_dow']}")
    if results.get("sig_months"):
        print(f"  ** Sig months:       {results['sig_months']}")
    if results.get("open_hour_sig"):
        print(
            f"  ** Open hour:        mean={results.get('open_hour_mean_pct')}%  "
            f"p={results.get('open_hour_p')}  {results.get('open_hour_sig')}"
        )
    if results.get("close_hour_sig"):
        print(
            f"  ** Close hour:       mean={results.get('close_hour_mean_pct')}%  "
            f"p={results.get('close_hour_p')}  {results.get('close_hour_sig')}"
        )
    print(f"  ** Early/late r:     {results.get('early_late_r')}  p={results.get('early_late_p')}")
    print(f"  ** Vol lag-1 ACF:    {results.get('tr_lag1_acf')}")
    if results.get("sig_clv_cells"):
        for dow_name, hour, mean_clv, p, marker in results["sig_clv_cells"][:6]:
            sig_str = "**" if marker == "\u2605" else " *"
            direction = "+" if mean_clv > 0 else "-"
            print(
                f"  ** CLV{sig_str} {dow_name} {hour:02d}h: {direction}{abs(mean_clv):.4f}  (p={p:.4f})"
            )
    print(f"  ** Positive years:   {results.get('pct_positive_years', 'n/a')}")

    return results


# ---------------------------------------------------------------------------
# Cross-instrument summary
# ---------------------------------------------------------------------------


def build_summary(all_results: list[dict]) -> None:
    valid = [r for r in all_results if not r.get("skipped")]
    if not valid:
        return

    syms = [r["symbol"] for r in valid]

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            "Vol lag-1 ACF (higher = stronger vol clustering)",
            "Early/Late session r (>0 = momentum, <0 = reversal)",
            "% positive years (>0.5 = persistent upward bias)",
        ],
    )
    acf_vals = [r.get("tr_lag1_acf", float("nan")) for r in valid]
    el_r_vals = [r.get("early_late_r", float("nan")) for r in valid]
    pos_yrs = [r.get("pct_positive_years", float("nan")) for r in valid]

    fig.add_trace(go.Bar(x=syms, y=acf_vals, marker_color="steelblue"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", row=1, col=1)

    colours_el = ["green" if v > 0.1 else "red" if v < -0.1 else "grey" for v in el_r_vals]
    fig.add_trace(go.Bar(x=syms, y=el_r_vals, marker_color=colours_el), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", row=2, col=1)

    fig.add_trace(go.Bar(x=syms, y=pos_yrs, marker_color="orange"), row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dot", row=3, col=1)

    fig.update_layout(
        title="H1 Pattern Analysis — Cross-Instrument Summary", height=750, showlegend=False
    )
    path = BASE_OUT / "00_summary.html"
    fig.write_html(str(path))
    print(f"\n  Summary chart -> {path}")

    # Text table
    print("\n" + "=" * 110)
    print(
        f"  {'Symbol':<14} {'Days':>5} {'VolACF':>7} {'E/L r':>7} {'E/L p':>7} "
        f"{'PosYrs':>7} {'CLV**':>5}  Sig patterns"
    )
    print("  " + "-" * 108)
    for r in sorted(valid, key=lambda x: x.get("tr_lag1_acf", 0), reverse=True):
        parts = []
        if r.get("sig_drift_hours"):
            parts.append(f"h={[h for h, _, _ in r['sig_drift_hours']]}")
        if r.get("sig_dow"):
            parts.append(f"DOW={[d for d, _, _ in r['sig_dow']]}")
        if r.get("sig_months"):
            parts.append(f"mo={[m for m, _, _ in r['sig_months']]}")
        if r.get("open_hour_sig"):
            parts.append(f"open{r['open_hour_sig']}")
        if r.get("close_hour_sig"):
            parts.append(f"close{r['close_hour_sig']}")
        n_clv_sig = len([c for c in r.get("sig_clv_cells", []) if c[4] == "\u2605"])
        print(
            f"  {r['symbol']:<14} {r['n_days']:>5} "
            f"{r.get('tr_lag1_acf', float('nan')):>7.3f} "
            f"{r.get('early_late_r', float('nan')):>7.3f} "
            f"{r.get('early_late_p', float('nan')):>7.3f} "
            f"{r.get('pct_positive_years', float('nan')):>7.2f} "
            f"{n_clv_sig:>5}  "
            f"{', '.join(parts) if parts else '-'}"
        )

    skipped = [r["symbol"] for r in all_results if r.get("skipped")]
    if skipped:
        print(f"\n  Skipped: {skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global MIN_TRADING_DAYS  # noqa: PLW0603
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Single symbol, e.g. SPY")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols")
    parser.add_argument("--all", action="store_true", help="Run all H1 files")
    parser.add_argument("--min-days", type=int, default=MIN_TRADING_DAYS)
    args = parser.parse_args()

    MIN_TRADING_DAYS = args.min_days

    if args.symbol:
        targets = [args.symbol]
    elif args.symbols:
        targets = args.symbols
    elif args.all:
        targets = [f.stem.replace("_H1", "") for f in sorted(DATA_DIR.glob("*_H1.parquet"))]
    else:
        parser.print_help()
        sys.exit(0)

    print(f"\nH1 EDA — {len(targets)} instrument(s)")
    all_results = []
    for sym in targets:
        result = run_instrument(sym)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        build_summary(all_results)

    print(f"\n  Outputs in: {BASE_OUT}")


if __name__ == "__main__":
    main()
