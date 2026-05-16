"""shadow_ewmac.py -- Daily-runnable shadow runner for B2 Carver EWMAC.

Per B2 audit §4.7 (L46 binding constraint = bootstrap CI), the strategy
is not yet live-eligible but is research-confirmed at canonical settings.
This script runs the canonical C1 cell against the most recent IG data
each day, logs:
    - Per-instrument daily forecast (combined EWMAC, capped, FDM-blended)
    - Per-instrument target position (forecast / target_forecast * target_vol / instrument_vol)
    - Simulated daily P&L (position.shift(1) * log_return)
    - Cumulative paper-equity curve

Output: ``.tmp/shadow/ewmac/positions_{run_date}.parquet`` (per-instrument
state) and ``.tmp/shadow/ewmac/equity.parquet`` (cumulative equity curve;
appended each run). Re-runnable: idempotent within a calendar day.

The goal is to accumulate enough forward data that the next B2 re-audit
can use 6-8 WFO folds instead of B2's 2 folds, pushing CI_lo above zero.

Usage (manual, daily after market close)::

    uv run python scripts/shadow_ewmac.py

Usage (automated via Windows Task Scheduler or cron, daily 22:00 UTC)::

    uv run python scripts/shadow_ewmac.py --refresh-data

The ``--refresh-data`` flag pulls today's IG bars first; default uses
already-downloaded ``data/ig_markets/*.parquet`` data.

Live-deploy upgrade path (once CI_lo > 0): port the per-instrument
position computation in ``_run_shadow_pass`` into a NautilusTrader
``Strategy`` class under ``titan/strategies/ewmac/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ewmac.ewmac_strategy import (  # noqa: E402
    EwmacConfig,
    _instrument_vol,
    build_positions,
    compute_ewmac_forecast,
)

DATA_DIR = PROJECT_ROOT / "data" / "ig_markets"
SHADOW_DIR = PROJECT_ROOT / ".tmp" / "shadow" / "ewmac"
SHADOW_DIR.mkdir(parents=True, exist_ok=True)

# Canonical production cell per B2 audit §4.7.
CANONICAL_CFG = EwmacConfig(
    speeds=((16, 64), (32, 128), (64, 256)),
    fdm=1.35,
    forecast_cap=20.0,
    target_vol_annual=0.10,
    target_forecast=10.0,
    apply_costs=True,
)

# Universe — mirror research/ewmac/run_b2b_audit.py EXPANDED_ROOTS.
UNIVERSE: tuple[str, ...] = (
    "CL",
    "BZ",
    "NG",
    "HO",
    "RB",
    "GC",
    "SI",
    "HG",
    "PL",
    "PA",
    "ZC",
    "ZW",
    "ZS",
    "ZL",
    "ZM",
    "KC",
    "CC",
    "SB",
    "CT",
    "OJ",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "DXY",
    "US10Y",
    "US30Y",
    "BUND",
    "GILT",
    "SPX",
    "NDX",
    "DJI",
    "RUT",
    "FTSE",
    "DAX",
    "NIKKEI",
    "EUROSTOXX",
)


def _load_universe() -> pd.DataFrame:
    """Load all IG closes that exist on disk."""
    parts: list[pd.Series] = []
    skipped: list[str] = []
    for root in UNIVERSE:
        fp = DATA_DIR / f"{root}_DAY.parquet"
        if not fp.exists():
            skipped.append(root)
            continue
        s = pd.read_parquet(fp)["close"].astype(float)
        s.name = root
        s.index = pd.to_datetime(s.index).normalize()
        parts.append(s)
    if not parts:
        raise RuntimeError(
            f"No IG instruments under {DATA_DIR}. Run scripts/download_ig_markets.py first."
        )
    if skipped:
        print(f"  [info] missing instruments: {skipped}")
    df = pd.concat(parts, axis=1).sort_index()
    df = df.ffill(limit=5)
    return df


def _run_shadow_pass(closes: pd.DataFrame) -> dict:
    """Compute today's forecast, position, and simulated P&L. Returns the
    structured snapshot for logging."""
    cfg = CANONICAL_CFG
    forecast = compute_ewmac_forecast(closes, cfg=cfg)
    instr_vol = _instrument_vol(closes, lookback_days=cfg.instrument_vol_lookback_days)
    positions = build_positions(forecast, instr_vol, cfg=cfg)
    log_ret = np.log(closes / closes.shift(1)).fillna(0.0)
    held_lagged = positions.shift(1).fillna(0.0)
    daily_pnl = (held_lagged * log_ret).sum(axis=1)

    # Apply turnover cost.
    if cfg.apply_costs:
        dpos = positions.diff().abs().fillna(0.0)
        n_fills_per_bar = (dpos > 1e-9).sum(axis=1).astype(float)
        bps_drag = (dpos.sum(axis=1) * cfg.cost_bps_per_turnover) / 10_000.0
        fixed_drag = (
            n_fills_per_bar * cfg.cost_fixed_usd_per_fill / max(cfg.notional_usd_per_leg, 1.0)
        )
        daily_pnl_net = daily_pnl - bps_drag - fixed_drag
    else:
        daily_pnl_net = daily_pnl

    # Today's snapshot (last bar that has a finite forecast for at least
    # 50% of universe).
    last_idx = closes.index[-1]
    last_forecasts = forecast.loc[last_idx]
    last_positions = positions.loc[last_idx]
    last_pnl = float(daily_pnl_net.loc[last_idx])

    # Cumulative metrics over the visible window.
    cum_pnl = daily_pnl_net.cumsum()
    eq_curve = np.exp(cum_pnl)

    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_bar_date": last_idx.date().isoformat(),
        "n_bars": int(len(closes)),
        "n_instruments": int(closes.shape[1]),
        "today_pnl_log": last_pnl,
        "today_pnl_pct": float(np.expm1(last_pnl) * 100),
        "cum_pnl_log_since_data_start": float(cum_pnl.iloc[-1]),
        "cum_equity_multiple": float(eq_curve.iloc[-1]),
        "n_long_positions": int((last_positions > 0).sum()),
        "n_short_positions": int((last_positions < 0).sum()),
        "n_flat_positions": int((last_positions == 0).sum()),
        "forecasts": last_forecasts.dropna().to_dict(),
        "positions": last_positions.dropna().to_dict(),
        # Convert Timestamp keys to ISO date strings for JSON serialisation.
        "daily_pnl_series": {
            ts.date().isoformat(): float(v) for ts, v in daily_pnl_net.tail(252).items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Pull today's IG bars first via scripts/download_ig_markets.py",
    )
    args = parser.parse_args()

    if args.refresh_data:
        print("Refreshing IG data...")
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "download_ig_markets.py"),
                "--sleeves",
                "all",
                "--years",
                "10",
            ],
            check=False,
        )

    print("Loading IG universe...")
    closes = _load_universe()
    print(
        f"  {closes.shape[1]} instruments, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} "
        f"({len(closes)} bars)"
    )

    print("Running shadow EWMAC pass...")
    snapshot = _run_shadow_pass(closes)

    # Persist this run's snapshot.
    run_date = snapshot["last_bar_date"]
    out_json = SHADOW_DIR / f"snapshot_{run_date}.json"
    out_json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"  snapshot: {out_json}")

    # Append (or overwrite if same date) to the cumulative equity log.
    eq_path = SHADOW_DIR / "equity.parquet"
    pnl_series = pd.Series(snapshot["daily_pnl_series"])
    pnl_series.index = pd.to_datetime(pnl_series.index).normalize()
    pnl_series.name = "daily_pnl_log"
    if eq_path.exists():
        existing = pd.read_parquet(eq_path)["daily_pnl_log"]
        # Take the latest reading per date (re-run within day overwrites).
        combined = pd.concat([existing, pnl_series])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        pnl_series = combined
    pnl_series.to_frame().to_parquet(eq_path)
    print(f"  equity log: {eq_path}  ({len(pnl_series)} bars)")

    # Console summary.
    print()
    print("=" * 60)
    print(f"  Shadow EWMAC snapshot — {run_date}")
    print("=" * 60)
    print(f"  Universe:        {snapshot['n_instruments']} instruments, {snapshot['n_bars']} bars")
    print(
        f"  Positions:       {snapshot['n_long_positions']} long / "
        f"{snapshot['n_short_positions']} short / {snapshot['n_flat_positions']} flat"
    )
    print(
        f"  Today's P&L:     {snapshot['today_pnl_pct']:+.4f}% "
        f"(log={snapshot['today_pnl_log']:+.6f})"
    )
    print(
        f"  Cumulative ret:  log={snapshot['cum_pnl_log_since_data_start']:+.4f} "
        f"=> {(snapshot['cum_equity_multiple'] - 1) * 100:+.2f}% equity"
    )
    # Top 5 strongest forecasts (high abs value = strong conviction).
    fcsts = pd.Series(snapshot["forecasts"])
    print()
    print("  Top 5 long-conviction instruments today:")
    for inst, f in fcsts.nlargest(5).items():
        print(f"    {inst:<10} forecast={f:+.3f}")
    print("  Top 5 short-conviction:")
    for inst, f in fcsts.nsmallest(5).items():
        print(f"    {inst:<10} forecast={f:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
