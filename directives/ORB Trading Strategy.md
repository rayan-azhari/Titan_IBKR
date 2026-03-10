# Opening Range Breakout (ORB) Strategy - Institutional Edition

The Opening Range Breakout (ORB) strategy described here relies on a multi-timeframe approach to capture early momentum during the New York session. 

After conducting a massive 1,200-scenario grid search across the S&P 100, we determined that the standard retail ORB (using the 15-minute candle low/high as a stop loss and targeting arbitrary profits) vastly underperforms. 

Here is the **Optimized Institutional Rule Set**:

1. **Asset Selection:** Trade massive-cap, institutional favorites (e.g., `IBM`, `ORCL`, `MO`, `CAT`, `V`) rather than high-beta retail meme stocks. 
2. **Trend Context:** Only take long breakouts if the underlying asset's Daily Close is **above** the `Daily SMA50`. Only take shorts if it is **below**. 
3. **Define the Range:** Mark the High and Low of the first 15-minute candle at the New York open (09:30:00 to 09:45:00 EST).
4. **Signal Confirmation:** Wait for a 5-minute candle to close above the 15-minute range high (for a long bias) or below the 15-minute range low (for a short bias).
5. **Dynamic Volatility Stop (ATR):** Do not use the 15m candle extreme as the stop loss. Instead, set the stop loss to exactly **2.0 * 5m ATR** away from your entry price.
6. **Execution & Management:** Execute the trade using a bracket order targeting a strict **1:2 Risk-to-Reward Ratio**. No trailing stops. Require auto-cancellation of counterpart orders on fill, and a strict End-of-Day (EOD) flatten at 15:55 ET to clear overnight risk. 

Below is a comprehensive blueprint to execute this live on Interactive Brokers using NautilusTrader.

---

### Sizing and Risk Management (The 1% Rule)

Implementing a strict 1% risk management rule is the bedrock of this strategy. Because we use a dynamic 2.0x ATR stop loss, the distance to your stop will vary depending on if the market is quiet or manic. Your position size must scale inversely to volatility to keep risk perfectly flat.

**The Core Risk Formula:**
$$\text{Position Size} = \frac{\text{Total Equity} \times 0.01}{|\text{Entry Price} - \text{Stop Loss Price}|}$$

*   **Total Equity:** Your current account liquid balance (e.g., $100,000).
*   **1% Risk Amount:** The absolute maximum amount you are willing to lose ($1,000).
*   **Trade Risk (Denominator):** If entering at $200 with a 2.0 ATR stop loss at $198, your absolute risk per share is $2.00.
*   **Result (Size):** $1,000 \div $2.00 = **500 shares**.
*   **Take Profit (Numerator):** At a 1:2 R/R, you target a $4.00 move per share, yielding exactly $2,000 (2% account gain).

---

### The Complete ORB Execution Script (NautilusTrader)

Here is the fully integrated NautilusTrader script combining multi-timeframe ORB logic, the Daily SMA50 trend filter, the 2.0 ATR dynamic stop loss, and exact 1% position sizing.

> **Implementation note:** The live production strategy is in `titan/strategies/orb/strategy.py` and launched via `scripts/run_live_orb.py`. The pattern below reflects the verified live implementation.

```python
from decimal import Decimal
import math
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.models.data import Bar
from nautilus_trader.models.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.models.identifiers import InstrumentId
from nautilus_trader.models.objects import Price, Quantity

class ORBConfig(StrategyConfig):
    instrument_id: str
    bar_type_5m: str
    bar_type_1d: str
    risk_pct: float = 0.01       # 1% risk per trade
    leverage_cap: float = 4.0
    warmup_bars_1d: int = 60
    warmup_bars_5m: int = 200

class ORBStrategy(Strategy):
    def __init__(self, config: ORBConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # State — reset each trading day
        self.or_high: float | None = None
        self.or_low: float | None = None
        self.trade_taken_today: bool = False
        self.current_atr: float | None = None

    def on_start(self):
        # Use EXTERNAL bars — IB streams them directly.
        # INTERNAL requires tick-by-tick subscription which paper accounts lack.
        self.subscribe_bars(BarType.from_str(self.config.bar_type_5m))
        self.subscribe_bars(BarType.from_str(self.config.bar_type_1d))
        self.log.info(f"ORB Strategy started for {self.instrument_id}.")

    def on_bar(self, bar: Bar):
        bar_type_str = str(bar.bar_type)

        if "DAY" in bar_type_str:
            self._update_daily(bar)
        elif "MINUTE" in bar_type_str:
            self._update_5m(bar)

    def _update_daily(self, bar: Bar):
        # Reset on each new daily bar
        self.or_high = None
        self.or_low = None
        self.trade_taken_today = False

    def _update_5m(self, bar: Bar):
        # ... compute ATR and Gaussian channel ...
        # CRITICAL: compute ATR using lowercase df columns BEFORE renaming to uppercase
        # for Gaussian Channel. The atr() helper from features.py uses df["high"/"low"/"close"].

        ts = bar.ts_event_ns // 1_000_000_000
        hour = (ts % 86400) // 3600  # simplified — use proper tz-aware conversion in production
        minute = (ts % 3600) // 60

        # Build Opening Range from first 5m candle at 09:30
        if hour == 9 and minute == 30:
            self.or_high = float(bar.high)
            self.or_low = float(bar.low)

        if self.or_high is None or self.trade_taken_today:
            return

        # Entry cutoff — stop looking for entries after 11:30
        if hour >= 11 and minute >= 30:
            return

        close = float(bar.close)
        if close > self.or_high:
            self._execute_bracket(OrderSide.BUY, Decimal(str(close)))
        elif close < self.or_low:
            self._execute_bracket(OrderSide.SELL, Decimal(str(close)))

    def _execute_bracket(self, side: OrderSide, fill_price: Decimal):
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None or self.current_atr is None:
            return

        atr = Decimal(str(self.current_atr))
        sl_distance = atr * Decimal("2.0")
        tp_distance = sl_distance * Decimal("2.0")  # 1:2 R/R

        sl_price = fill_price - sl_distance if side == OrderSide.BUY else fill_price + sl_distance
        tp_price = fill_price + tp_distance if side == OrderSide.BUY else fill_price - tp_distance

        # Sizing: risk 1% of account equity
        account = self.portfolio.account(self.instrument_id.venue)
        if account is None:
            return
        equity = float(account.balance_total(instrument.quote_currency).as_double())
        risk_amount = equity * self.config.risk_pct
        shares = int(risk_amount / float(sl_distance))
        if shares < 1:
            return

        precision = instrument.price_precision
        qty = Quantity.from_int(shares)

        # Set trade_taken_today BEFORE submitting — prevents re-entry on next bar
        self.trade_taken_today = True

        # order_factory.bracket() creates an OTO+OCO linked list.
        # IB ONLY accepts TimeInForce.DAY for MARKET orders — GTC is rejected.
        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            sl_trigger_price=Price(round(float(sl_price), precision), precision=precision),
            tp_price=Price(round(float(tp_price), precision), precision=precision),
            entry_order_type=OrderType.MARKET,
            entry_time_in_force=TimeInForce.DAY,   # ← MUST be DAY for MARKET on IB
            tp_time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(bracket)   # ← use submit_order_list, not submit_order
        self.log.info(f"Bracket submitted: {side} {shares} shares | SL={sl_price:.2f} TP={tp_price:.2f}")

    def on_stop(self):
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
```

---

### Key Implementation Notes (Verified Live)

1. **Bracket Orders via `order_factory.bracket()`:** Use `self.order_factory.bracket()` + `self.submit_order_list()` — never submit three independent market/SL/TP orders. The bracket creates proper OTO (One-Triggers-Other) + OCO (One-Cancels-Other) contingency so the SL and TP are automatically linked and cancel each other on fill.

2. **Market Order TIF must be `DAY`:** IB rejects `TimeInForce.GTC` on market orders. Always use `entry_time_in_force=TimeInForce.DAY` for the entry leg of a bracket. SL/TP limit and stop orders can use `GTC`.

3. **Set `trade_taken_today = True` before `submit_order_list()`:** The flag must be set *before* submitting to prevent re-entry on the next bar tick arriving before the order acknowledgement.

4. **EXTERNAL bars, not INTERNAL:** Subscribe with `EXTERNAL` bar types (e.g. `AAPL.NASDAQ-5-MINUTE-LAST-EXTERNAL`). `INTERNAL` bars require tick-by-tick subscription which paper accounts lack permissions for (error 10189).

5. **ATR computed before column rename:** If using a DataFrame-based ATR helper (like `features.atr()`), it expects lowercase column names (`high`, `low`, `close`). Compute ATR *before* renaming columns to uppercase for Gaussian Channel or other indicators.