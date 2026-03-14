"""run_backtest_meta.py — VectorBT backtest: MTF Confluence vs Meta-Filtered MTF.

Compares two strategies on EUR/USD H1 (2005-2026) with IS/OOS 70/30 split:
  1. Raw MTF Confluence  — take every signal the MTF strategy fires
  2. Meta-Filtered MTF   — only signals where meta_prob >= META_THRESHOLD

Meta-model is retrained on IS data only so OOS stats are fully honest
(no look-ahead from the saved production model).

Usage:
    .venv/Scripts/python.exe scripts/run_backtest_meta.py
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

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt is not installed. Run: uv add vectorbt")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost is not installed. Run: uv add xgboost")
    sys.exit(1)

# ── Constants ───────────────────────────────────────────────────────────────
PAIR = "EUR_USD"
META_THRESHOLD = 0.60
IS_SPLIT = 0.70  # First 70% = in-sample, last 30% = out-of-sample
FEES = 0.00015  # ~1.5 pips per side, round-trip ~3 pips (EUR/USD typical)
SLIPPAGE = 0.00010  # 1.0 pip per side (conservative baseline for H1 market-order fills)
# Asymmetric Tom-Next carry: IBKR markup means BOTH sides typically pay a net cost.
# Long pays USD−EUR differential + markup ≈ 2.0%/yr (2019-2026 average)
# Short pays EUR−USD differential + markup ≈ 0.8%/yr (markup exceeds near-parity rate diff)
CARRY_LONG_ANNUAL = 0.020
CARRY_SHORT_ANNUAL = 0.008
INIT_CASH = 100_000.0
FREQ = "1h"


# ── Data loading ────────────────────────────────────────────────────────────


def _load_parquet(pair: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


# ── MTF Signal ──────────────────────────────────────────────────────────────


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi_val = _rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi_val > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_mtf_signals(pair: str, mtf_cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Return (primary_signal {-1,0,+1}, confluence_score) on H1 index."""
    weights = mtf_cfg["weights"]
    threshold = mtf_cfg["confirmation_threshold"]

    h1_df = _load_parquet(pair, "H1")
    h4_df = _load_parquet(pair, "H4")
    d_df = _load_parquet(pair, "D")
    w_df = _load_parquet(pair, "W")

    tfs = {"H1": h1_df, "H4": h4_df, "D": d_df, "W": w_df}
    h1_index = h1_df.index
    confluence = pd.Series(0.0, index=h1_index)

    for tf_name, df_tf in tfs.items():
        cfg = mtf_cfg[tf_name]
        sig = _tf_signal(df_tf["close"], cfg["fast_ma"], cfg["slow_ma"], cfg["rsi_period"])
        sig_aligned = sig.reindex(h1_index, method="ffill")
        confluence += sig_aligned * weights[tf_name]

    primary = pd.Series(0, index=h1_index, dtype=int)
    primary[confluence > threshold] = 1
    primary[confluence < -threshold] = -1

    return primary, confluence


# ── Feature Engineering ─────────────────────────────────────────────────────


def _bollinger_bandwidth(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    band = close.rolling(period).std() * std
    return (2 * band) / mid.replace(0, np.nan)


def build_meta_features(
    h1_df: pd.DataFrame, confluence: pd.Series, primary: pd.Series
) -> pd.DataFrame:
    """Identical feature set to run_metalabeling.py — 20 features."""
    close = h1_df["close"]
    feats = pd.DataFrame(index=h1_df.index)

    for lag in [1, 2, 3, 5, 10]:
        feats[f"ret_{lag}"] = close.pct_change(lag)

    tr = pd.concat(
        [
            h1_df["high"] - h1_df["low"],
            (h1_df["high"] - h1_df["close"].shift()).abs(),
            (h1_df["low"] - h1_df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feats["atr_14"] = tr.rolling(14).mean()
    feats["atr_pct"] = feats["atr_14"] / close.replace(0, np.nan)
    feats["boll_bw"] = _bollinger_bandwidth(close, 20, 2.0)

    feats["rsi_14"] = _rsi(close, 14)
    feats["rsi_21"] = _rsi(close, 21)
    feats["rsi_overextended"] = ((feats["rsi_14"] > 70) | (feats["rsi_14"] < 30)).astype(int)

    feats["adx_proxy"] = feats["atr_14"].rolling(14).mean() / feats["atr_14"].rolling(14).std()

    feats["confluence_score"] = confluence
    feats["confluence_abs"] = confluence.abs()
    feats["primary_signal"] = primary

    feats["hour_utc"] = h1_df.index.hour
    feats["is_london"] = ((feats["hour_utc"] >= 7) & (feats["hour_utc"] < 16)).astype(int)
    feats["is_new_york"] = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 21)).astype(int)
    feats["is_overlap"] = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 16)).astype(int)
    feats["day_of_week"] = h1_df.index.dayofweek

    return feats


# ── Meta-label construction ──────────────────────────────────────────────────


def build_meta_labels(primary: pd.Series) -> pd.Series:
    """Load TBM labels and derive binary meta-labels.

    meta_label = 1  if TBM outcome matches primary direction (trade would have won)
    meta_label = 0  if SL hit, time-out, or direction mismatch
    """
    tbm_path = FEATURES_DIR / f"{PAIR}_H1_tbm_labels.parquet"
    if not tbm_path.exists():
        raise FileNotFoundError(
            f"TBM labels not found at {tbm_path}. Run: "
            ".venv/Scripts/python.exe research/ml/build_tbm_labels.py"
        )
    tbm = pd.read_parquet(tbm_path)["tbm_label"]
    tbm = tbm.reindex(primary.index).fillna(0).astype(int)

    meta_label = pd.Series(0, index=primary.index, dtype=int)
    # Long signal wins if TBM = +1 (upper barrier hit)
    meta_label[(primary == 1) & (tbm == 1)] = 1
    # Short signal wins if TBM = -1 (lower barrier hit)
    meta_label[(primary == -1) & (tbm == -1)] = 1

    return meta_label


# ── Meta-model training ──────────────────────────────────────────────────────


def train_meta_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── Portfolio statistics printer ─────────────────────────────────────────────


def print_stats(label: str, pf) -> dict:
    """Print comprehensive VectorBT portfolio statistics."""
    sep = "-" * 56

    # Core returns
    total_ret = pf.total_return() * 100
    try:
        cagr = pf.annualized_return() * 100
    except Exception:
        cagr = float("nan")

    # Risk-adjusted
    try:
        sharpe = pf.sharpe_ratio()
    except Exception:
        sharpe = float("nan")
    try:
        sortino = pf.sortino_ratio()
    except Exception:
        sortino = float("nan")
    try:
        calmar = pf.calmar_ratio()
    except Exception:
        calmar = float("nan")

    # Drawdown
    max_dd = pf.max_drawdown() * 100

    # Trade-level
    n_trades = pf.trades.count()
    if n_trades > 0:
        win_rate = pf.trades.win_rate() * 100
        try:
            profit_factor = pf.trades.profit_factor()
        except Exception:
            profit_factor = float("nan")
        try:
            expectancy = pf.trades.expectancy()
        except Exception:
            expectancy = float("nan")
        try:
            avg_pnl = pf.trades.records_readable["PnL"].mean()
        except Exception:
            avg_pnl = float("nan")
        try:
            best = pf.trades.records_readable["PnL"].max()
            worst = pf.trades.records_readable["PnL"].min()
        except Exception:
            best = worst = float("nan")
    else:
        win_rate = profit_factor = expectancy = avg_pnl = best = worst = float("nan")

    # Final equity
    final_equity = pf.final_value()

    print(f"\n  {label}")
    print(f"  {sep}")
    print(f"    Initial Capital:  ${INIT_CASH:>12,.0f}")
    print(f"    Final Equity:     ${final_equity:>12,.0f}")
    print(f"    Total Return:     {total_ret:>+10.2f}%")
    print(f"    CAGR:             {cagr:>+10.2f}%")
    print(f"  {sep}")
    print(f"    Sharpe Ratio:     {sharpe:>10.3f}")
    print(f"    Sortino Ratio:    {sortino:>10.3f}")
    print(f"    Calmar Ratio:     {calmar:>10.3f}")
    print(f"    Max Drawdown:     {max_dd:>10.2f}%")
    print(f"  {sep}")
    print(f"    Total Trades:     {n_trades:>10,d}")
    print(f"    Win Rate:         {win_rate:>10.1f}%")
    print(f"    Profit Factor:    {profit_factor:>10.3f}")
    print(f"    Expectancy:       {expectancy:>10.4f}")
    print(f"    Avg PnL/Trade:    {avg_pnl:>10.4f}")
    print(f"    Best Trade:       {best:>10.4f}")
    print(f"    Worst Trade:      {worst:>10.4f}")

    return {
        "label": label,
        "total_return_pct": total_ret,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd_pct": max_dd,
        "trades": n_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def run_vbt_portfolio(
    close: pd.Series,
    long_entries: pd.Series,
    short_entries: pd.Series,
    long_exits: pd.Series,
    short_exits: pd.Series,
    label: str,
    sl_stop: pd.Series | None = None,
    open_prices: pd.Series | None = None,
    slippage: float = SLIPPAGE,
    size_pct: float | None = None,
):
    """Run combined long+short VectorBT portfolio.

    sl_stop:     ATR as a fraction of price (e.g. 0.008 = exit if price moves
                 0.8% against position from entry). sl_trail=False (fixed stop).
    open_prices: If provided, fills execute at open[t+1] (next bar's open) rather
                 than close[t], eliminating look-ahead bias from same-bar fills.
    slippage:    Additional per-side slippage fraction (default 1.0 pip = 0.00010).
    size_pct:    If set, deploy this fraction of INIT_CASH as fixed notional per trade
                 (e.g. 0.10 → $10,000 fixed per trade). Uses size_type="value" because
                 VBT SizeType.Percent does not support signal-based reversals.
                 Default (None) uses VBT all-cash sizing (100% of equity per trade).
    """
    kwargs: dict = {
        "close": close,
        "entries": long_entries,
        "exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
        "init_cash": INIT_CASH,
        "fees": FEES,
        "freq": FREQ,
        "accumulate": False,
    }
    # Fix look-ahead bias: fill at next bar's open, not current close
    if open_prices is not None:
        kwargs["price"] = open_prices.shift(-1).reindex(close.index).ffill()
    # Add execution slippage
    if slippage > 0:
        kwargs["slippage"] = slippage
    if sl_stop is not None:
        kwargs["sl_stop"] = sl_stop.reindex(close.index).ffill().fillna(sl_stop.median())
        kwargs["sl_trail"] = False  # fixed stop from entry, not trailing
    if size_pct is not None:
        # SizeType.Percent doesn't support signal reversals in VBT; use fixed value instead
        kwargs["size"] = size_pct * INIT_CASH
        kwargs["size_type"] = "value"
    pf = vbt.Portfolio.from_signals(**kwargs)
    return pf


# ── Carry cost analysis ──────────────────────────────────────────────────────


def compute_carry_impact(pf) -> dict:
    """Estimate Tom-Next carry cost impact on portfolio returns.

    Uses an asymmetric broker model: IBKR's markup on the Tom-Next differential
    means both long and short positions typically pay a net carry cost. There is
    no free carry credit for shorts in a near-parity rate environment.
    """
    if pf.trades.count() == 0:
        return {}
    trades = pf.trades.records_readable.copy()
    if "Direction" in trades.columns:
        direction_col = "Direction"
    elif "Side" in trades.columns:
        direction_col = "Side"
    else:
        return {}
    is_long = trades[direction_col].astype(str).str.upper() == "LONG"
    trades["hours_held"] = (
        trades["Exit Timestamp"] - trades["Entry Timestamp"]
    ).dt.total_seconds() / 3600
    long_cost_hr = CARRY_LONG_ANNUAL / (252 * 24)
    short_cost_hr = CARRY_SHORT_ANNUAL / (252 * 24)
    # Both sides pay carry (asymmetric broker markup model — always a positive cost)
    trades["carry_cost"] = np.where(
        is_long,
        trades["hours_held"] * long_cost_hr * INIT_CASH,
        trades["hours_held"] * short_cost_hr * INIT_CASH,
    )
    total_carry = float(trades["carry_cost"].sum())  # always > 0
    raw_pnl = pf.total_profit()
    adj_pnl = raw_pnl - total_carry
    return {
        "carry_pnl_usd": total_carry,
        "carry_pct_equity": total_carry / INIT_CASH * 100,
        "raw_total_return_pct": pf.total_return() * 100,
        "adj_total_return_pct": adj_pnl / INIT_CASH * 100,
        "long_trades": int(trades[direction_col].astype(str).str.upper().eq("LONG").sum()),
        "short_trades": int(trades[direction_col].astype(str).str.upper().ne("LONG").sum()),
    }


# ── ATR stop sensitivity sweep ───────────────────────────────────────────────


def atr_sensitivity_sweep(
    close_oos: pd.Series,
    long_entries: pd.Series,
    short_entries: pd.Series,
    long_exits: pd.Series,
    short_exits: pd.Series,
    sl_oos_1x: pd.Series,
    open_oos: pd.Series,
) -> None:
    """Test OOS Sharpe across ATR stop multipliers 0.5x to 3.0x.

    Reveals whether the strategy is brittle (only works at 1.0x) or
    robust (profitable across a wide range of stop distances).
    """
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    results = []
    for mult in multipliers:
        sl = sl_oos_1x * mult
        pf = run_vbt_portfolio(
            close_oos,
            long_entries,
            short_entries,
            long_exits,
            short_exits,
            f"ATR {mult}x",
            sl_stop=sl,
            open_prices=open_oos,
        )
        n = pf.trades.count()
        sharpe = pf.sharpe_ratio() if n > 0 else float("nan")
        cagr = pf.annualized_return() * 100 if n > 0 else float("nan")
        max_dd = pf.max_drawdown() * 100
        wr = pf.trades.win_rate() * 100 if n > 0 else float("nan")
        results.append(
            {
                "mult": mult,
                "sharpe": sharpe,
                "cagr": cagr,
                "max_dd": max_dd,
                "win_rate": wr,
                "trades": n,
            }
        )

    print("\n" + "=" * 70)
    print("ATR STOP MULTIPLIER SENSITIVITY  (OOS, Raw MTF signals)")
    print("  Tests robustness: is 1.0x ATR special, or profitable across a range?")
    print("=" * 70)
    print(
        f"  {'ATR Mult':>8} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WinRate%':>10} {'Trades':>7}"
    )
    print("  " + "-" * 54)
    for r in results:
        flag = " <-- baseline" if r["mult"] == 1.0 else ""
        print(
            f"  {r['mult']:>8.2f} {r['sharpe']:>8.3f} {r['cagr']:>8.2f} "
            f"{r['max_dd']:>8.2f} {r['win_rate']:>10.1f} {r['trades']:>7,d}{flag}"
        )
    n_above = sum(1 for r in results if r["sharpe"] > 1.0)
    verdict = "ROBUST" if n_above >= 5 else "BRITTLE"
    print(f"\n  Sharpe > 1.0 in {n_above}/{len(multipliers)} scenarios  ->  {verdict}")


# ── Slippage stress test ─────────────────────────────────────────────────────


SLIPPAGE_LEVELS = [0.00005, 0.00010, 0.00015, 0.00020, 0.00030]


def slippage_stress_test(
    close_oos: pd.Series,
    long_entries: pd.Series,
    short_entries: pd.Series,
    long_exits: pd.Series,
    short_exits: pd.Series,
    sl_oos: pd.Series,
    open_oos: pd.Series,
) -> None:
    """Test OOS performance across 5 slippage levels (0.5 to 3.0 pips/side).

    Reveals whether profitability survives realistic high-volatility fill costs.
    Pass criteria: Sharpe > 1.0 at all 5 levels.
    """
    print("\n" + "=" * 70)
    print("SLIPPAGE STRESS TEST  (OOS, ATR 1.0x stop, Raw MTF signals)")
    print("  Tests survival at 0.5, 1.0, 1.5, 2.0, 3.0 pips per side.")
    print("=" * 70)
    print(f"  {'Slip (pip)':>10} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>7} {'Trades':>7}")
    print("  " + "-" * 52)

    above: list[bool] = []
    for slip in SLIPPAGE_LEVELS:
        pf = run_vbt_portfolio(
            close_oos,
            long_entries,
            short_entries,
            long_exits,
            short_exits,
            f"Slip {slip * 10000:.1f}pip",
            sl_stop=sl_oos,
            open_prices=open_oos,
            slippage=slip,
        )
        n = pf.trades.count()
        sharpe = float(pf.sharpe_ratio()) if n > 0 else float("nan")
        cagr = float(pf.annualized_return()) * 100 if n > 0 else float("nan")
        max_dd = float(pf.max_drawdown()) * 100
        wr = float(pf.trades.win_rate()) * 100 if n > 0 else float("nan")
        flag = " <-- baseline" if abs(slip - SLIPPAGE) < 1e-9 else ""
        pip_str = f"{slip * 10000:.1f}"
        print(
            f"  {pip_str:>10} {sharpe:>8.3f} {cagr:>+8.2f} {max_dd:>8.2f} {wr:>7.1f} {n:>7,d}{flag}"
        )
        above.append(sharpe > 1.0)

    n_above = sum(above)
    verdict = "ROBUST" if n_above == len(SLIPPAGE_LEVELS) else "FRAGILE"
    print(f"\n  Sharpe > 1.0 in {n_above}/{len(SLIPPAGE_LEVELS)} levels  ->  {verdict}")
    if n_above < len(SLIPPAGE_LEVELS):
        print("  WARNING: Strategy breaks at elevated slippage — fragile to execution friction.")


# ── Fixed-fraction sizing comparison ─────────────────────────────────────────


SIZING_LEVELS = [1.0, 0.10, 0.01]


def fixed_fraction_comparison(
    close_oos: pd.Series,
    long_entries: pd.Series,
    short_entries: pd.Series,
    long_exits: pd.Series,
    short_exits: pd.Series,
    sl_oos: pd.Series,
    open_oos: pd.Series,
) -> None:
    """Compare OOS performance at 100%, 10%, and 1% of INIT_CASH per trade (fixed notional).

    Uses a fixed dollar amount per trade (not % of current equity) because VBT
    SizeType.Percent does not support signal-based reversals. At 10% = $10k/trade
    vs 100% = $100k/trade, this isolates signal edge from compounding inflation.
    Pass criteria: Sharpe > 1.0 at 10% sizing confirms genuine per-trade edge.
    """
    print("\n" + "=" * 70)
    print("FIXED-FRACTION SIZING COMPARISON  (OOS, ATR 1.0x stop)")
    print("  Isolates signal edge from compounding effects.")
    print("=" * 70)
    print(f"  {'Size':>8} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'R/D':>8} {'Trades':>7}")
    print("  " + "-" * 52)

    for sz in SIZING_LEVELS:
        pf = run_vbt_portfolio(
            close_oos,
            long_entries,
            short_entries,
            long_exits,
            short_exits,
            f"Size {sz:.0%}",
            sl_stop=sl_oos,
            open_prices=open_oos,
            size_pct=sz,
        )
        n = pf.trades.count()
        sharpe = pf.sharpe_ratio() if n > 0 else float("nan")
        cagr = pf.annualized_return() * 100 if n > 0 else float("nan")
        max_dd = float(pf.max_drawdown()) * 100  # negative value
        rd = abs(cagr / max_dd) if max_dd != 0 else float("nan")
        flag = " <-- default (all-cash)" if sz == 1.0 else ""
        print(f"  {sz:>7.0%} {sharpe:>8.3f} {cagr:>+8.2f} {max_dd:>8.2f} {rd:>8.2f} {n:>7,d}{flag}")

    print("\n  Interpretation:")
    print("    If Sharpe collapses at 10% -> edge was compounding, not signal quality.")
    print("    If Sharpe stays > 1.0 at 10% -> genuine per-trade edge confirmed.")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("META-STRATEGY BACKTEST: MTF Confluence vs Meta-Filtered MTF")
    print(
        f"Instrument: EUR/USD H1  |  IS/OOS: {int(IS_SPLIT * 100)}/{int((1 - IS_SPLIT) * 100)}"
        f"  |  Fees: {FEES * 10000:.1f} pips/side  |  Slippage: {SLIPPAGE * 10000:.1f} pip/side"
    )
    print("  Fills: next-bar open (no look-ahead)  |  Stop: 1x ATR fixed")
    print("=" * 70)

    # 1. Load MTF config
    with open(PROJECT_ROOT / "config" / "mtf.toml", "rb") as f:
        mtf_cfg = tomllib.load(f)

    # 2. Load H1 price data
    h1_df = _load_parquet(PAIR, "H1")
    close = h1_df["close"]
    open_prices = h1_df["open"]
    print(
        f"\n  Data: {len(h1_df):,} H1 bars  |  {h1_df.index[0].date()} to {h1_df.index[-1].date()}"
    )

    # Compute ATR-based stop-loss fraction: sl_stop = ATR14 / close
    # VBT exits long if price drops sl_stop% below entry; short if price rises sl_stop% above entry
    tr = pd.concat(
        [
            h1_df["high"] - h1_df["low"],
            (h1_df["high"] - h1_df["close"].shift(1)).abs(),
            (h1_df["low"] - h1_df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    sl_stop = (atr14 / close).fillna(method="bfill")  # fraction of price, ~0.008 for EUR/USD H1
    print(f"  ATR stop: avg {sl_stop.mean() * 10000:.1f} pips  |  SL multiplier: 1x ATR")

    # 3. Compute MTF primary signals
    print("\n[1] Computing MTF primary signals...")
    primary, confluence = compute_mtf_signals(PAIR, mtf_cfg)
    print(
        f"  Long: {primary.eq(1).sum():,}  "
        f"Short: {primary.eq(-1).sum():,}  "
        f"Flat: {primary.eq(0).sum():,}"
    )

    # 4. Build feature matrix
    print("\n[2] Building meta-model feature matrix...")
    feats = build_meta_features(h1_df, confluence, primary)
    active_mask = primary != 0
    feats_active = feats[active_mask].dropna()
    print(f"  {len(feats_active):,} active-signal rows, {feats_active.shape[1]} features")

    # 5. Build meta-labels (TBM outcomes aligned to primary direction)
    print("\n[3] Building meta-labels from TBM outcomes...")
    meta_label = build_meta_labels(primary)
    y_active = meta_label[feats_active.index]
    print(f"  Wins: {y_active.sum():,}  ({y_active.mean() * 100:.1f}%)")

    # 6. IS/OOS split on active rows
    split_idx = int(len(feats_active) * IS_SPLIT)
    is_end_ts = feats_active.index[split_idx - 1]
    oos_start_ts = feats_active.index[split_idx]

    X_is = feats_active.iloc[:split_idx]
    y_is = y_active.iloc[:split_idx]
    X_oos = feats_active.iloc[split_idx:]
    y_oos = y_active.iloc[split_idx:]

    print(f"\n  IS:  {X_is.index[0].date()} to {X_is.index[-1].date()}  ({len(X_is):,} rows)")
    print(f"  OOS: {X_oos.index[0].date()} to {X_oos.index[-1].date()}  ({len(X_oos):,} rows)")

    # 7. Train meta-model on IS ONLY (OOS is fully honest)
    print("\n[4] Training meta-model on IS data only...")
    meta_model = train_meta_model(X_is, y_is)

    # IS in-sample predictions
    is_proba_is = meta_model.predict_proba(X_is)[:, 1]
    # OOS out-of-sample predictions
    is_proba_oos = meta_model.predict_proba(X_oos)[:, 1]

    print(f"  IS  class balance: {y_is.mean() * 100:.1f}% wins")
    print(f"  OOS class balance: {y_oos.mean() * 100:.1f}% wins")

    # 8. Build probability series on full H1 index (NaN where primary==0)
    meta_prob = pd.Series(np.nan, index=h1_df.index)
    meta_prob.loc[X_is.index] = is_proba_is
    meta_prob.loc[X_oos.index] = is_proba_oos

    # 9. Build entry/exit signal arrays (cast to Series so type-checkers are happy)
    # Primary (raw MTF signals)
    long_entries_raw: pd.Series = primary.eq(1)
    long_exits_raw: pd.Series = primary.ne(1)
    short_entries_raw: pd.Series = primary.eq(-1)
    short_exits_raw: pd.Series = primary.ne(-1)

    # Meta-filtered (primary AND meta_prob >= threshold)
    meta_gate: pd.Series = meta_prob >= META_THRESHOLD
    long_entries_meta: pd.Series = primary.eq(1) & meta_gate
    short_entries_meta: pd.Series = primary.eq(-1) & meta_gate
    # Exits: when primary changes OR meta_gate drops (prob falls below threshold)
    long_exits_meta: pd.Series = long_exits_raw | (primary.eq(1) & ~meta_gate)
    short_exits_meta: pd.Series = short_exits_raw | (primary.eq(-1) & ~meta_gate)

    # Meta-Entry-Only: entries gated by meta_prob, exits follow raw signal only.
    # Isolates the value of entry-filtering from the damage caused by probability-gate exits.
    # If this variant underperforms raw, the meta-model adds no entry value either.
    long_entries_meta_e: pd.Series = primary.eq(1) & meta_gate
    short_entries_meta_e: pd.Series = primary.eq(-1) & meta_gate
    long_exits_meta_e: pd.Series = long_exits_raw
    short_exits_meta_e: pd.Series = short_exits_raw

    # 10. IS and OOS price slices
    is_mask = close.index <= is_end_ts
    oos_mask = close.index >= oos_start_ts

    close_is = close[is_mask]
    close_oos = close[oos_mask]
    open_is = open_prices[is_mask]
    open_oos = open_prices[oos_mask]

    print("\n[5] Running VectorBT backtests (next-bar open fills + ATR stop + slippage)...")

    sl_is = sl_stop[is_mask]
    sl_oos = sl_stop[oos_mask]

    # ── IS Primary
    pf_is_raw = run_vbt_portfolio(
        close_is,
        long_entries_raw[is_mask],
        short_entries_raw[is_mask],
        long_exits_raw[is_mask],
        short_exits_raw[is_mask],
        "IS Raw Primary",
        sl_stop=sl_is,
        open_prices=open_is,
    )

    # ── IS Meta-filtered
    pf_is_meta = run_vbt_portfolio(
        close_is,
        long_entries_meta[is_mask],
        short_entries_meta[is_mask],
        long_exits_meta[is_mask],
        short_exits_meta[is_mask],
        "IS Meta-Filtered",
        sl_stop=sl_is,
        open_prices=open_is,
    )

    # ── OOS Primary
    pf_oos_raw = run_vbt_portfolio(
        close_oos,
        long_entries_raw[oos_mask],
        short_entries_raw[oos_mask],
        long_exits_raw[oos_mask],
        short_exits_raw[oos_mask],
        "OOS Raw Primary",
        sl_stop=sl_oos,
        open_prices=open_oos,
    )

    # ── OOS Meta-filtered
    pf_oos_meta = run_vbt_portfolio(
        close_oos,
        long_entries_meta[oos_mask],
        short_entries_meta[oos_mask],
        long_exits_meta[oos_mask],
        short_exits_meta[oos_mask],
        "OOS Meta-Filtered",
        sl_stop=sl_oos,
        open_prices=open_oos,
    )

    # ── IS Meta-Entry-Only
    pf_is_meta_e = run_vbt_portfolio(
        close_is,
        long_entries_meta_e[is_mask],
        short_entries_meta_e[is_mask],
        long_exits_meta_e[is_mask],
        short_exits_meta_e[is_mask],
        "IS Meta-Entry-Only",
        sl_stop=sl_is,
        open_prices=open_is,
    )

    # ── OOS Meta-Entry-Only
    pf_oos_meta_e = run_vbt_portfolio(
        close_oos,
        long_entries_meta_e[oos_mask],
        short_entries_meta_e[oos_mask],
        long_exits_meta_e[oos_mask],
        short_exits_meta_e[oos_mask],
        "OOS Meta-Entry-Only",
        sl_stop=sl_oos,
        open_prices=open_oos,
    )

    # 11. Print results
    print("\n" + "=" * 70)
    print("IN-SAMPLE RESULTS")
    print("=" * 70)
    stats_is_raw = print_stats("RAW MTF Confluence (IS)", pf_is_raw)
    stats_is_meta_e = print_stats("META-ENTRY-ONLY (IS)", pf_is_meta_e)
    stats_is_meta = print_stats(f"META-FULL (IS, thresh={META_THRESHOLD})", pf_is_meta)

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE RESULTS  <-- honest evaluation")
    print("=" * 70)
    stats_oos_raw = print_stats("RAW MTF Confluence (OOS)", pf_oos_raw)
    stats_oos_meta_e = print_stats("META-ENTRY-ONLY (OOS)", pf_oos_meta_e)
    stats_oos_meta = print_stats(f"META-FULL (OOS, thresh={META_THRESHOLD})", pf_oos_meta)

    # 12. Summary comparison table (5 columns: Raw / Entry-Only / Full-Meta, IS and OOS)
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    hdr = (
        f"  {'Metric':<22} {'IS Raw':>9} {'IS EntryOnly':>13}"
        f" {'IS Meta':>9} {'OOS Raw':>9} {'OOS EntryOnly':>13} {'OOS Meta':>9}"
    )
    print(hdr)
    print("  " + "-" * 90)

    rows = [
        ("Total Return %", "total_return_pct", "{:>+9.2f}"),
        ("CAGR %", "cagr_pct", "{:>+9.2f}"),
        ("Sharpe Ratio", "sharpe", "{:>9.3f}"),
        ("Sortino Ratio", "sortino", "{:>9.3f}"),
        ("Calmar Ratio", "calmar", "{:>9.3f}"),
        ("Max Drawdown %", "max_dd_pct", "{:>9.2f}"),
        ("# Trades", "trades", "{:>9,d}"),
        ("Win Rate %", "win_rate_pct", "{:>9.1f}"),
        ("Profit Factor", "profit_factor", "{:>9.3f}"),
        ("Expectancy", "expectancy", "{:>9.4f}"),
    ]

    all_stats = [
        stats_is_raw,
        stats_is_meta_e,
        stats_is_meta,
        stats_oos_raw,
        stats_oos_meta_e,
        stats_oos_meta,
    ]
    for name, key, fmt in rows:
        vals = []
        for s in all_stats:
            v = s.get(key, float("nan"))
            try:
                vals.append(fmt.format(v))
            except Exception:
                vals.append(f"{'N/A':>9}")
        print(f"  {name:<22} {''.join(vals)}")

    # 13. IS/OOS consistency check
    is_sharpe = stats_is_raw["sharpe"]
    oos_sharpe = stats_oos_raw["sharpe"]
    ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else float("nan")

    is_meta_e_sharpe = stats_is_meta_e["sharpe"]
    oos_meta_e_sharpe = stats_oos_meta_e["sharpe"]
    meta_e_ratio = oos_meta_e_sharpe / is_meta_e_sharpe if is_meta_e_sharpe > 0 else float("nan")

    is_meta_sharpe = stats_is_meta["sharpe"]
    oos_meta_sharpe = stats_oos_meta["sharpe"]
    meta_ratio = oos_meta_sharpe / is_meta_sharpe if is_meta_sharpe > 0 else float("nan")

    print("\n  IS/OOS Consistency (OOS/IS Sharpe ratio, reject if < 0.5):")
    raw_pass = "PASS" if ratio >= 0.5 else "FAIL"
    meta_e_pass = "PASS" if meta_e_ratio >= 0.5 else "FAIL"
    meta_pass = "PASS" if meta_ratio >= 0.5 else "FAIL"
    print(f"    Raw Primary:       OOS/IS = {ratio:.2f}   {raw_pass}")
    print(f"    Meta-Entry-Only:   OOS/IS = {meta_e_ratio:.2f}   {meta_e_pass}")
    print(f"    Meta-Full:         OOS/IS = {meta_ratio:.2f}   {meta_pass}")

    # 14. Meta-model diagnosis
    print("\n" + "=" * 70)
    print("META-MODEL DIAGNOSIS")
    print("=" * 70)
    raw_t = stats_oos_raw["trades"]
    me_t = stats_oos_meta_e["trades"]
    full_t = stats_oos_meta["trades"]
    raw_wr = stats_oos_raw["win_rate_pct"]
    me_wr = stats_oos_meta_e["win_rate_pct"]
    full_wr = stats_oos_meta["win_rate_pct"]
    raw_sh = stats_oos_raw["sharpe"]
    me_sh = stats_oos_meta_e["sharpe"]
    full_sh = stats_oos_meta["sharpe"]

    print(f"  {'Variant':<22} {'Trades':>7} {'WinRate%':>9} {'Sharpe':>8}")
    print("  " + "-" * 50)
    print(f"  {'Raw (control)':<22} {raw_t:>7,d} {raw_wr:>9.1f} {raw_sh:>8.3f}")
    print(f"  {'Meta-Entry-Only':<22} {me_t:>7,d} {me_wr:>9.1f} {me_sh:>8.3f}")
    print(f"  {'Meta-Full':<22} {full_t:>7,d} {full_wr:>9.1f} {full_sh:>8.3f}")

    # Entry-filter value: Entry-Only vs Raw
    entry_value = me_sh - raw_sh
    # Exit-churn damage: Full-Meta vs Entry-Only
    churn_damage = full_sh - me_sh

    print(f"\n  Entry-filter value (Entry-Only vs Raw):  {entry_value:>+.3f} Sharpe")
    print(f"  Exit-churn damage  (Full vs Entry-Only): {churn_damage:>+.3f} Sharpe")

    if me_sh >= raw_sh:
        verdict = "KEEP ENTRY GATE, DISCARD PROBABILITY-GATE EXITS"
    elif me_sh >= raw_sh * 0.9:
        verdict = "MARGINAL — DISCARD META, TRADE RAW SIGNAL"
    else:
        verdict = "DISCARD META ENTIRELY — TRADE RAW SIGNAL"
    print(f"\n  Verdict: {verdict}")

    # 15. Carry cost impact (Tom-Next swap — asymmetric broker model)
    print("\n" + "=" * 70)
    print("CARRY COST IMPACT  (Tom-Next, asymmetric: long ~2.0%/yr, short ~0.8%/yr)")
    print("  IBKR markup means both sides pay a net cost — no free carry credit.")
    print("=" * 70)
    carry = compute_carry_impact(pf_oos_raw)
    if carry:
        print(
            f"  OOS Raw MTF | Long trades: {carry['long_trades']}"
            f"  Short trades: {carry['short_trades']}"
        )
        cost_pct = carry["carry_pct_equity"]
        print(f"  Total carry cost:  ${carry['carry_pnl_usd']:>10,.0f}  ({cost_pct:>+.2f}% equity)")
        print(f"  Raw total return:  {carry['raw_total_return_pct']:>+.2f}%")
        print(f"  Carry-adj return:  {carry['adj_total_return_pct']:>+.2f}%  (estimate)")

    # 15. ATR stop multiplier sensitivity sweep
    le_oos: pd.Series = long_entries_raw[oos_mask]
    se_oos: pd.Series = short_entries_raw[oos_mask]
    lx_oos: pd.Series = long_exits_raw[oos_mask]
    sx_oos: pd.Series = short_exits_raw[oos_mask]
    atr_sensitivity_sweep(close_oos, le_oos, se_oos, lx_oos, sx_oos, sl_oos, open_oos)

    # 16. Slippage stress test
    slippage_stress_test(close_oos, le_oos, se_oos, lx_oos, sx_oos, sl_oos, open_oos)

    # 17. Fixed-fraction sizing comparison (isolates compounding from signal edge)
    fixed_fraction_comparison(close_oos, le_oos, se_oos, lx_oos, sx_oos, sl_oos, open_oos)

    # 18. Save equity curves
    eq_path = REPORTS_DIR / "meta_backtest_equity.html"
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for pf, name, color in [
            (pf_oos_raw, "OOS Raw MTF", "#9E9E9E"),
            (pf_oos_meta, "OOS Meta-Filtered", "#4CAF50"),
        ]:
            eq = pf.value()
            fig.add_trace(
                go.Scatter(
                    x=eq.index,
                    y=eq.values,
                    name=name,
                    line=dict(color=color, width=1.5),
                )
            )
        fig.update_layout(
            title="EUR/USD H1 — OOS Equity Curves: Raw MTF vs Meta-Filtered",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=500,
        )
        fig.write_html(str(eq_path))
        print(f"\n  Equity curve saved: {eq_path.name}")
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print("Backtest complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
