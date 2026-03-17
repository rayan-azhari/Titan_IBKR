Here's the complete system architecture first, then the full plan phase by phase.---

## Environment Setup

Before any code, get the stack installed cleanly. Create a dedicated virtual environment in VS Code — mixing Nautilus Trader with other packages can cause dependency conflicts, so isolation matters.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install vectorbt pandas numpy yfinance ib_insync
pip install nautilus_trader
pip install discord-webhook python-telegram-bot schedule
```

Nautilus Trader has a Rust compilation step — if you hit build errors on Windows, use their pre-built wheel from PyPI rather than building from source. Keep separate `requirements-research.txt` (vectorbt stack) and `requirements-live.txt` (nautilus + ib_insync) since vectorbt's numba dependency can clash with Nautilus's Rust extensions on some platforms.

Project structure:
```
strategy/
├── data/           # cached historical CSVs
├── signals/        # signal construction modules
├── research/       # vectorbt notebooks and sweep scripts
├── engine/         # nautilus strategy classes
├── execution/      # IBKR integration
├── monitoring/     # alerting webhooks
└── config.py       # parameters (MA windows, thresholds, universe)
```

---

## Phase 1 — Data Pipeline

**Universe definition.** Start with 8 assets chosen for low correlation:

| Asset | ETF | Role |
|---|---|---|
| US large cap | SPY | Core equity trend |
| US small cap | IWM | Risk-on/off signal |
| International | EFA | Diversification |
| Emerging | EEM | High-beta tail |
| Gold | GLD | Crisis hedge |
| Long bonds | TLT | Counter-equity trend |
| Commodities | DJP or GSG | Inflation regime |
| Short-term bonds | IEF | Cash proxy |

The reason to include bonds and gold specifically is that they trend in the *opposite* direction to equities in most bear markets — so the long/short signals on them will naturally hedge the equity longs without you needing to force market neutrality.

**Historical data pull.** Use `yfinance` for everything pre-2020 (free, reliable for daily OHLCV), then IBKR for recent data and live feeds. Write a unified `DataLoader` class that normalises both sources into the same pandas DataFrame format (DatetimeIndex, columns = asset tickers, values = adjusted close). Cache everything to `/data/` as Parquet — reloading from disk takes milliseconds vs seconds from the API.

```python
import yfinance as yf
import pandas as pd

UNIVERSE = ['SPY', 'IWM', 'EFA', 'EEM', 'GLD', 'TLT', 'DJP', 'IEF']

def fetch_history(tickers, start='1993-01-01'):
    raw = yf.download(tickers, start=start, auto_adjust=True)['Close']
    return raw.dropna(how='all')
```

Critically, you need to handle the fact that not all ETFs exist back to 1993 — TLT started in 2002, EEM in 2003. Your backtests need to handle variable universe sizes gracefully. A simple approach: only include an asset in the universe for dates where it has at least 252 days of history (one full year, enough to compute the 250-day SMA meaningfully).

---

## Phase 2 — Signal Design

Build each signal as a pure function that takes a price series and returns a float series (or boolean). Keep them modular — the sweep later will mix and match them.

**Trend filter (regime gate).** This is binary: either the market is in a trend-following regime or it isn't. Apply it independently to each asset.

```python
def trend_regime(prices, fast=50, slow=250):
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    # True = long regime active, False = flat/short regime
    return (prices > sma_fast) & (prices > sma_slow)
```

**Deceleration signals.** These are continuous — they output a float between 0 and 1 (or -1 and 1) representing signal strength. The position sizer will consume this.

*D% (your distance-from-MA idea):*
```python
def distance_pct(prices, window=250, smooth=10):
    sma = prices.rolling(window).mean()
    d = (prices - sma) / sma * 100
    return d.ewm(span=smooth).mean()  # smoothed to reduce noise
```

*SMA slope:*
```python
def sma_slope(prices, window=250, lookback=20):
    sma = prices.rolling(window).mean()
    return sma.diff(lookback) / sma.shift(lookback)  # normalised slope
```

*ADX:* Use `pandas-ta` or `ta-lib` rather than implementing from scratch — ADX has a 3-smoothing-step calculation that's error-prone to hand-code. `ta.adx(high, low, close, length=14)['ADX_14']`.

*MACD histogram:*
```python
def macd_hist(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line  # positive and shrinking = deceleration
```

*Realised volatility:*
```python
def realised_vol(prices, window=20):
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)
```

**Signal combination.** Rather than combining signals with fixed weights upfront, keep them separate going into the sweep — you want vectorbt to tell you which combinations work best. The combination logic will look like:

```python
def composite_signal(trend, d_pct, adx, macd_h, rv, weights):
    # Normalise each signal to [-1, 1]
    # Weight and sum
    # Return: long score, short score, position_fraction
```

**Position sizing.** Volatility-scaled sizing is the key mechanism:

```python
def position_size(signal_strength, realised_vol, target_vol=0.10, max_position=1.0):
    # Target 10% annualised vol contribution per position
    vol_scalar = target_vol / (realised_vol + 1e-6)
    raw_size = signal_strength * vol_scalar
    return np.clip(raw_size, -max_position, max_position)
```

---

## Phase 3 — VectorBT Parameter Sweep

This is the most computation-heavy phase. VectorBT's power is running thousands of parameter combinations simultaneously using numpy broadcasting — what would take hours in a loop takes minutes.

**Parameter grid.** These are the axes you want to sweep:

```python
param_grid = {
    'fast_ma':      [20, 30, 40, 50, 60],          # 5 values
    'slow_ma':      [150, 200, 250, 300],           # 4 values
    'adx_thresh':   [20, 25, 30],                   # 3 values
    'dpct_smooth':  [5, 10, 20],                    # 3 values
    'macd_fast':    [8, 12, 16],                    # 3 values
    'target_vol':   [0.08, 0.10, 0.15],             # 3 values
}
# Total combinations: 5 × 4 × 3 × 3 × 3 × 3 = 1,620 backtests per asset
```

Run this across 8 assets = ~13,000 backtests. VectorBT handles this without a loop via its portfolio simulation engine.

**Backtest structure:**

```python
import vectorbt as vbt

# Build signal arrays for all parameter combinations at once
entries = vbt.pd_acc.signals.generate_combs(...)  # shape: (dates, n_combinations)
exits   = ...

# Single call runs all combinations in parallel
pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    short_entries=short_entries,
    short_exits=short_exits,
    freq='1D',
    fees=0.001,   # 10bps round trip — conservative for liquid ETFs
    slippage=0.001
)
```

**Metrics to extract.** For each combination, compute:

- Sharpe ratio (primary rank)
- Calmar ratio (return / max drawdown — good for drawdown-sensitive strategies)
- Maximum drawdown
- Average drawdown duration
- Win rate in trending vs choppy regimes (split the sample by ADX > 25 to test regime sensitivity)
- Correlation of returns to SPY (you want low correlation — that's the point of multi-asset)

**Regime split analysis.** This is the most important diagnostic — a strategy that looks great in aggregate but only works in bull markets is not what you want. Split the backtest period into:
- Strong trend regime (SPY > SMA250, ADX > 25)
- Choppy regime (SPY near SMA250, ADX < 20)
- Bear trend (SPY < SMA250)

And check Sharpe in each regime independently. Your target: positive Sharpe in all three, not just overall.

**Sweep visualisation.** VectorBT's built-in heatmaps are excellent for 2D parameter slices. Plot fast_ma vs slow_ma with Sharpe as the colour — you want a broad plateau of good parameters rather than a narrow spike (which would indicate overfitting).

```python
pf.sharpe_ratio().vbt.heatmap(
    x_level='fast_ma',
    y_level='slow_ma'
).show()
```

**Selecting the final parameters.** Don't just pick the highest Sharpe combination. Use a "robust region" approach: identify the cluster of parameter sets that all perform well (e.g. top quartile), then pick the centroid of that cluster. This gives you a parameter set that is likely to continue working even if the true optimal shifts slightly in live trading.

---

## Phase 4 — Nautilus Trader Event-Driven Validation

The VectorBT backtest is vectorised — it sees all bars simultaneously and has no concept of execution latency, slippage on order time, or look-ahead bias at bar boundaries. Nautilus validates that the strategy works identically in a proper event-driven simulation where each bar arrives one at a time and orders execute on the *next* bar's open.

**Strategy class structure:**

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

class MultiAssetTrendStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.fast_ma = config.fast_ma
        self.slow_ma = config.slow_ma
        self.target_vol = config.target_vol
        self._price_history = {}  # per-instrument buffer

    def on_start(self):
        # Subscribe to daily bars for each asset in universe
        for instrument_id in self.config.universe:
            self.subscribe_bars(
                bar_type=BarType.from_str(f"{instrument_id}-1-DAY-LAST-EXTERNAL")
            )

    def on_bar(self, bar: Bar):
        instrument_id = bar.bar_type.instrument_id
        # Update rolling price buffer
        self._update_history(instrument_id, bar.close)

        if len(self._price_history[instrument_id]) < self.slow_ma:
            return  # warm-up period

        # Compute signals
        trend_signal = self._compute_trend(instrument_id)
        decel_signal = self._compute_deceleration(instrument_id)
        rv = self._compute_realised_vol(instrument_id)

        # Position sizing
        target_fraction = self._size_position(trend_signal, decel_signal, rv)

        # Generate orders if target differs materially from current
        self._rebalance(instrument_id, target_fraction)
```

**Time-of-day trigger.** Rather than rebalancing on every bar, set a timer that fires 15 minutes before the US close (20:45 UK time) to evaluate all signals and queue orders. This replicates the real execution behaviour and avoids acting on intraday noise.

```python
def on_start(self):
    # Set daily timer for 20:45 UK time
    self.clock.set_time_alert(
        name="end_of_day_eval",
        alert_time=self._next_eod_time()
    )

def on_time_event(self, event):
    if event.name == "end_of_day_eval":
        self._evaluate_all_positions()
        self._reset_timer()
```

**Cross-checking vectorbt.** After running the Nautilus backtest on the same historical data, compute the correlation of daily P&L between the two. They won't be identical (different execution assumptions), but the equity curves should be strongly correlated (>0.95) and the final Sharpe should be within ~10% of each other. Any larger divergence indicates a look-ahead bias or timing assumption difference in the vectorbt model that needs fixing before going live.

---

## Phase 5 — IBKR Production Deployment

**IB Gateway configuration.** Run IB Gateway (headless) rather than TWS on your cloud server — it uses fewer resources and can be controlled programmatically. Set it to auto-restart on disconnect.

```python
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersExecClientConfig
)

ibkr_config = InteractiveBrokersExecClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=4001,        # 4001 for Gateway live, 4002 for paper
    ibg_client_id=1,
    account_id="YOUR_ACCOUNT"
)
```

**Order types.** Use Market-on-Close (MOC) orders for all ETFs — these execute at the official closing auction price and have the tightest bid-ask spreads of the day on high-volume instruments like SPY, GLD, and TLT. MOC orders placed before 15:45 ET are guaranteed to fill at close.

```python
from nautilus_trader.model.orders import MarketOnCloseOrder

order = self.order_factory.market_on_close(
    instrument_id=instrument_id,
    order_side=OrderSide.BUY if target > current else OrderSide.SELL,
    quantity=Quantity.from_int(shares_delta)
)
self.submit_order(order)
```

**Scheduling.** On your cloud server (a small VPS — even a £4/month instance is sufficient for this), use a cron job to start everything at 20:30 UK time on trading days only:

```bash
# crontab -e
30 20 * * 1-5 /path/to/.venv/bin/python /path/to/run_strategy.py
```

The script should: check the market is open (use `pandas_market_calendars` to skip US holidays), start IB Gateway if not already running, connect Nautilus, run the daily evaluation, place orders, log results, then shut down. The whole process should complete in under 10 minutes — well before the close.

**Monitoring and alerting.** Wire up a Discord webhook so the system pushes a daily message showing: which assets were evaluated, the computed signal and target allocation for each, which orders were placed, and whether they filled successfully.

```python
import requests

def send_daily_report(allocations, orders, fills):
    webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK"
    message = {
        "content": f"**Daily Strategy Report — {date.today()}**\n"
                   f"Allocations: {allocations}\n"
                   f"Orders placed: {len(orders)}\n"
                   f"Fills confirmed: {len(fills)}"
    }
    requests.post(webhook_url, json=message)
```

---

## Testing & Integrity Checks

Before risking any capital, run through this validation checklist:

**Look-ahead bias test.** Take any signal computation function. Shift the input prices forward by one bar. If the backtest performance *improves*, you have look-ahead bias — you're accidentally using tomorrow's data to make today's decision. Every signal should be computed with `prices.shift(1)` applied before use (i.e. yesterday's close drives today's decision).

**Transaction cost sensitivity.** Rerun the vectorbt sweep with fees of 20bps, 30bps, and 50bps. Any strategy combination that becomes unprofitable at 30bps is too dependent on frictionless execution and should be discarded. Robust strategies should survive 3–5× the expected real-world fee.

**Walk-forward validation.** Split the history into in-sample (earliest 70%) and out-of-sample (most recent 30%). Run the parameter sweep on in-sample only, select the robust parameter set, then run it *unchanged* on out-of-sample. If performance collapses on out-of-sample, the parameter selection is overfit. You want out-of-sample Sharpe to be at least 60% of in-sample Sharpe.

**Paper trading period.** Regardless of how good the backtest looks, run the live system in paper trading mode (IBKR port 4002) for at least 60 trading days before switching to real capital. This validates the execution logic, timing, order fill rates, and monitoring pipeline with zero financial risk.

---

## Suggested Build Order

Given the dependencies between phases, work in this sequence:

1. Data pipeline and universe loading (2–3 days)
2. Individual signal functions with unit tests (2–3 days)
3. VectorBT single-asset backtest on SPY to validate signal logic (1–2 days)
4. Full multi-asset sweep and robust parameter selection (1–2 days)
5. Nautilus strategy class and event-driven backtest (3–4 days)
6. IBKR paper trading integration and monitoring (3–5 days)
7. 60-day paper trading observation period
8. Live capital deployment

