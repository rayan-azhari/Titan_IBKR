"""run_orb_dsr_audit.py -- DSR audit of the 9 ORB-deployed instruments.

Pre-registered in:
    directives/ORB DSR Audit N482 2026-05-14.md

Computes Sharpe + DSR-prob (at N=482, the screener pool size) for each
of the 9 instruments in config/orb_live.toml. Uses a simple ORB
simulator with per-ticker config (ATR multiplier, RR ratio, time-decay
cutoff). Filters use_sma / use_rsi flagged in the config are
approximated by simple SMA(50) trend-with-direction + RSI(14) momentum
filters; use_gauss is approximated by a wider SMA(20) filter (Gaussian
Channel is a smoothed midline; SMA is the most common proxy and
captures the same trend-context idea).

For DSR: uses the variance of Sharpes ACROSS the 9 deployed cells as a
LOWER BOUND on the true sr_var (the 9 are post-selection survivors;
pre-selection variance over all 482 candidates is larger). Lower-bound
sr_var produces OPTIMISTIC DSR-prob -- if even optimistic prob fails
the gate, true prob definitely fails.

Usage::

    python research/orb/run_orb_dsr_audit.py
"""

from __future__ import annotations

import math
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.run_phase5_joint_sweep import deflated_sharpe_prob  # noqa: E402
from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402

N_SCREENED = 482   # original screening pool size per Strategy Re-validation §1.5
BARS_PER_DAY_M5 = 78
BARS_PER_YEAR_M5 = BARS_PER_DAY_M5 * 252
SPREAD_BPS = 2.0   # half-spread, per fill
SLIPPAGE_BPS = 1.0


@dataclass(frozen=True)
class OrbConfig:
    instrument: str
    atr_multiplier: float
    rr_ratio: float
    use_sma: bool
    use_rsi: bool
    use_gauss: bool
    orb_window_end: str
    entry_cutoff: str = "11:00"


def load_configs(path: Path = PROJECT_ROOT / "config" / "orb_live.toml") -> list[OrbConfig]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return [
        OrbConfig(
            instrument=inst,
            atr_multiplier=float(cfg[inst]["atr_multiplier"]),
            rr_ratio=float(cfg[inst]["rr_ratio"]),
            use_sma=bool(cfg[inst].get("use_sma", False)),
            use_rsi=bool(cfg[inst].get("use_rsi", False)),
            use_gauss=bool(cfg[inst].get("use_gauss", False)),
            orb_window_end=cfg[inst].get("orb_window_end", "09:45"),
            entry_cutoff=cfg[inst].get("entry_cutoff", "11:00"),
        )
        for inst in cfg
    ]


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _parse_hhmm_to_minutes(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def _to_eastern(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert("US/Eastern")
    return idx.tz_convert("US/Eastern")


def simulate_orb(df: pd.DataFrame, cfg: OrbConfig) -> tuple[pd.Series, dict]:
    """Run a simple ORB simulator. Returns (per_bar_returns, stats)."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df.index = _to_eastern(df.index)
    df = df.between_time("09:30", "16:00").copy()
    df["atr"] = _atr(df, 14)
    df["rsi"] = _rsi(df["close"], 14) if cfg.use_rsi else 50.0
    df["sma50"] = df["close"].rolling(50).mean() if cfg.use_sma else df["close"]
    df["sma20"] = df["close"].rolling(20).mean() if cfg.use_gauss else df["close"]

    orb_end_min = _parse_hhmm_to_minutes(cfg.orb_window_end)
    entry_cutoff_min = _parse_hhmm_to_minutes(cfg.entry_cutoff)

    cost_per_side = (SPREAD_BPS + SLIPPAGE_BPS) / 10_000
    per_bar_ret = pd.Series(0.0, index=df.index)
    trades: list[dict] = []

    for _date, day in df.groupby(df.index.date):
        if len(day) < 10:
            continue
        orb_window = day[
            (day.index.hour * 60 + day.index.minute) <= orb_end_min
        ]
        if len(orb_window) < 1:
            continue
        or_high = float(orb_window["high"].max())
        or_low = float(orb_window["low"].min())

        post_orb = day[(day.index.hour * 60 + day.index.minute) > orb_end_min]
        position = 0
        entry_px = 0.0
        sl = 0.0
        tp = 0.0
        entry_ts = None
        for ts, row in post_orb.iterrows():
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["atr"]) if not pd.isna(row["atr"]) else (or_high - or_low)
            t_min = ts.hour * 60 + ts.minute

            if position == 0:
                if t_min >= entry_cutoff_min:
                    continue
                # Context filters per cfg
                ctx_long = True
                ctx_short = True
                if cfg.use_sma:
                    sma50 = row["sma50"]
                    if not pd.isna(sma50):
                        ctx_long = ctx_long and close > sma50
                        ctx_short = ctx_short and close < sma50
                if cfg.use_rsi:
                    rsi = row["rsi"]
                    if not pd.isna(rsi):
                        ctx_long = ctx_long and rsi > 50.0
                        ctx_short = ctx_short and rsi < 50.0
                if cfg.use_gauss:
                    sma20 = row["sma20"]
                    if not pd.isna(sma20):
                        ctx_long = ctx_long and close > sma20
                        ctx_short = ctx_short and close < sma20
                risk = atr * cfg.atr_multiplier
                if close > or_high and ctx_long:
                    position = 1
                    entry_px = close
                    sl = close - risk
                    tp = close + risk * cfg.rr_ratio
                    entry_ts = ts
                elif close < or_low and ctx_short:
                    position = -1
                    entry_px = close
                    sl = close + risk
                    tp = close - risk * cfg.rr_ratio
                    entry_ts = ts
            else:
                # check stops / targets / time exit (end-of-day)
                exit_px = None
                exit_reason = None
                if position == 1:
                    if low <= sl:
                        exit_px = sl
                        exit_reason = "stop"
                    elif high >= tp:
                        exit_px = tp
                        exit_reason = "target"
                else:
                    if high >= sl:
                        exit_px = sl
                        exit_reason = "stop"
                    elif low <= tp:
                        exit_px = tp
                        exit_reason = "target"
                if exit_px is None and (t_min >= 15 * 60 + 55 or ts == day.index[-1]):
                    exit_px = close
                    exit_reason = "eod"
                if exit_px is not None:
                    gross_ret = (exit_px - entry_px) / entry_px * position
                    net_ret = gross_ret - 2.0 * cost_per_side  # round-trip
                    per_bar_ret.loc[ts] = net_ret
                    trades.append({
                        "entry_ts": entry_ts, "exit_ts": ts,
                        "side": position, "entry_px": entry_px, "exit_px": exit_px,
                        "gross_ret": gross_ret, "net_ret": net_ret, "reason": exit_reason,
                    })
                    position = 0

    return per_bar_ret, {"n_trades": len(trades), "trades": trades}


def main() -> None:
    configs = load_configs()
    print("=" * 100)
    print(f"  ORB DSR AUDIT -- N_screened = {N_SCREENED}, M5 bars/yr = {BARS_PER_YEAR_M5}")
    print(f"  Lower-bound sr_var uses variance of the {len(configs)} deployed Sharpes")
    print(f"  null max SR ~ sqrt(2 ln N) * sr_std = {np.sqrt(2 * np.log(N_SCREENED)):.2f} * sr_std")
    print("=" * 100)

    rows: list[dict] = []
    for cfg in configs:
        path = PROJECT_ROOT / "data" / f"{cfg.instrument}_M5.parquet"
        if not path.exists():
            print(f"  SKIP {cfg.instrument}: parquet missing")
            continue
        df = pd.read_parquet(path)
        per_bar_ret, stats = simulate_orb(df, cfg)
        n_bars = len(per_bar_ret)
        n_trades = stats["n_trades"]
        if n_trades < 5:
            print(f"  {cfg.instrument}: only {n_trades} trades -- skip")
            continue
        non_zero = per_bar_ret[per_bar_ret != 0]
        sh = sharpe(per_bar_ret, periods_per_year=BARS_PER_YEAR_M5)
        ci_lo, ci_hi = bootstrap_sharpe_ci(per_bar_ret, periods_per_year=BARS_PER_YEAR_M5, n_resamples=500, seed=42)
        eq = (1.0 + per_bar_ret).cumprod()
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        avg_per_trade = float(non_zero.mean()) if len(non_zero) > 0 else 0.0
        rows.append({
            "instrument": cfg.instrument,
            "n_bars": int(n_bars),
            "n_trades": int(n_trades),
            "win_rate": round(float((non_zero > 0).mean()) if len(non_zero) else 0.0, 4),
            "avg_per_trade": round(avg_per_trade, 6),
            "sharpe": round(sh, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "max_drawdown": round(mdd, 4),
            "atr_mult": cfg.atr_multiplier,
            "rr_ratio": cfg.rr_ratio,
        })
        print(
            f"  {cfg.instrument:<5} sharpe={sh:+.3f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"trades={n_trades:>3}  win={float((non_zero > 0).mean()) if len(non_zero) else 0:.1%}  "
            f"DD={mdd*100:.1f}%"
        )

    if not rows:
        print("\n  No instruments produced results.")
        return

    # DSR with lower-bound sr_var
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    sr_var = float(np.var(sharpes, ddof=1))
    sr_std = math.sqrt(sr_var)
    null_max = sr_std * (
        (1 - 0.5772156649) * math.erf(0)  # placeholder; actual formula uses norm.ppf
        + 0
    )
    # Use the actual deflated_sharpe_prob formula
    from scipy.stats import norm
    EULER_GAMMA = 0.5772156649
    e_max_sr = sr_std * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N_SCREENED)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (N_SCREENED * math.e))
    )
    print(f"\n  Cross-cell sr_std (lower bound) = {sr_std:.4f}")
    print(f"  Expected null max SR at N={N_SCREENED} = {e_max_sr:.3f}")
    print()

    for r in rows:
        T = max(r["n_bars"], 30)
        dsr = deflated_sharpe_prob(r["sharpe"], sr_var, 0.0, 3.0, T, N_SCREENED)
        r["dsr_prob"] = round(float(dsr), 4)
        if dsr >= 0.95:
            verdict = "DEPLOYABLE"
        elif dsr >= 0.50:
            verdict = "CONDITIONAL"
        else:
            verdict = "RETIRE"
        r["verdict"] = verdict

    print("  Per-cell DSR + verdict:")
    print(f"  {'Inst':<5}  {'Sharpe':>7}  {'CI_lo':>7}  {'DSR_prob':>8}  {'e_max_SR':>9}  {'gap':>7}  Verdict")
    print("  " + "-" * 80)
    for r in rows:
        gap = r["sharpe"] - e_max_sr
        print(
            f"  {r['instrument']:<5}  {r['sharpe']:>+7.3f}  {r['ci_lo']:>+7.3f}  "
            f"{r['dsr_prob']:>8.4f}  {e_max_sr:>+9.3f}  {gap:>+7.3f}  {r['verdict']}"
        )

    # Class-level decision: more than half failing => flag the strategy
    n_retire = sum(1 for r in rows if r["verdict"] == "RETIRE")
    print()
    print("=" * 100)
    if n_retire > len(rows) / 2:
        print(f"  CLASS-LEVEL FLAG: {n_retire}/{len(rows)} cells failed DSR. "
              f"ORB strategy class flagged for re-evaluation.")
    else:
        print(f"  CLASS-LEVEL: {n_retire}/{len(rows)} cells in RETIRE bucket. "
              f"Strategy class survives at the majority threshold.")
    print("=" * 100)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / ".tmp" / "reports" / "orb_dsr_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{stamp}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
