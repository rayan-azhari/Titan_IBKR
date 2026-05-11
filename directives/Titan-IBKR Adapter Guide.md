# Titan-IBKR Adapter Guide

## 1. Overview

The **Titan-IBKR Adapter** is a custom `nautilus_trader` integration designed to trade effectively on the IBKR v20 API. It uses a **Hybrid Wrapper Architecture** to bridge the high-performance, event-driven world of NautilusTrader with the HTTP/REST and Streaming APIs of IBKR.

### Key Features
*   **Event-Driven:** Converts IBKR stream events (`ORDER_FILL`, `ORDER_CANCEL`) into internal Nautilus events.
*   **Resilient:** Handles network interruptions and automatically reconciles order/position state on reconnection.
*   **Production-Ready:** Support for Market, Limit, Stop, and Market-If-Touched orders.
*   **Stateless Mapping:** Maps orders 1:1 using `ClientOrderId`, avoiding complex local databases.

---

## 2. Titan Package Structure

The `titan` package is the core of this repository. It is structured to separate concerns between data, execution, and strategy logic.

```text
titan/
├── adapters/                  # [CORE] Nautilus-IBKR Integration
│   └── ibkr/
│       ├── execution.py       # Order submission & reconciliation
│       ├── data.py            # Live price streaming (ticks/bars)
│       └── instruments.py     # Instrument symbol & precision mapping
├── config/                    # Configuration loading (TOML -> Python)
├── data/                      # Data fetching (historic) & validation
├── indicators/                # Shared technical indicators (Numba/VectorBT)
├── models/                    # Quantitative models (Spread, Slippage, ML)
├── strategies/                # NautilusTrader Strategies
│   ├── mtf_confluence.py      # Multi-Timeframe Logic
│   └── ml_strategy.py         # Machine Learning Inference
└── utils/                     # Logging, Notification (Slack/Discord), Math
```

---

## 3. Adapter Capabilities

The adapter is currently verified for the following functionality:

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Market Orders** | ✅ Supported | Use `TimeInForce.GTC` for FX (IDEALPRO). `FOK` is **rejected** by IB for FX. `DAY` is valid for equities. |
| **Limit Orders** | ✅ Supported | GTC (Good-Til-Cancelled). |
| **Stop Orders** | ✅ Supported | Triggers market order when price is hit. |
| **Market-If-Touched** | ✅ Supported | Like a Limit, but triggers Market order (slippage possible). |
| **Stop Loss / Take Profit** | 🚧 Partial | Can be attached to orders, but standalone management via Nautilus is limited. |
| **Trailing Stop** | ❌ Unsupported | **Reason:** IBKR requires Trailing Stops to be linked to a specific `tradeID`. The current adapter is stateless and doesn't track individual trade IDs locally. |
| **GSLO** | ❌ Unsupported | Guaranteed Stop Loss Orders are not currently mapped. |

---

## 4. Configuration

The system uses `dotenv` and `TOML` for configuration.

### Environment Variables (`.env`)
Required for IBKR connection:
```ini
IBKR_HOST="127.0.0.1"
IBKR_PORT=4002 # Gateway Paper (7497 for TWS Paper)
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID="DUxxxxxxx"
```

### Instruments (`config/instruments.toml`)
Defines the universe of tradeable assets:
```toml
[instruments]
EUR_USD = { type = "CURRENCY_PAIR", precision = 5, lot_size = 1 }
GBP_USD = { type = "CURRENCY_PAIR", precision = 5, lot_size = 1 }
```

---

## 5. Usage Guide

### Running a Live Strategy

**A. ORB Strategy (7 large-cap equities)**
```bash
python scripts/run_live_orb.py
```

**B. Multi-Timeframe Confluence strategy**
```bash
uv run python scripts/run_live_mtf.py
```

Both scripts will:
1.  Connect to IBKR Gateway/TWS.
2.  Reconcile open positions.
3.  Warm up indicators with historical data.
4.  Begin trading.

> **Port mapping:** 7496 = TWS Live, 7497 = TWS Paper, 4001 = Gateway Live, 4002 = Gateway Paper. The runner auto-selects `REALTIME` vs `DELAYED_FROZEN` market data based on the port.

### Verifying the Setup
To verify adapter health and order handling:
```bash
uv run python scripts/verify_connection.py
```
This tests the `ibapi` connectivity to the local socket server.

---

## 6. Troubleshooting

### Common Errors

**1. Instrument not found / bar subscription fails (`Cannot start bar aggregation: no instrument found for X.EXCHANGE`)**
*   **Cause:** The `IBContract` was defined without `primaryExchange`, so the adapter generates `SYMBOL.SMART` as the instrument ID instead of `SYMBOL.NYSE` / `SYMBOL.NASDAQ`.
*   **Fix:** Always set `primaryExchange` on every `IBContract`. The generated instrument ID is `{localSymbol}.{primaryExchange}`, so `bar_type` and `instrument_id` in the strategy config must match exactly.
    ```python
    IBContract(secType="STK", symbol="UNH", exchange="SMART", primaryExchange="NYSE", currency="USD")
    # → instrument_id = "UNH.NYSE"
    ```
*   **Tip:** WMT routes to `NASDAQ` on IB (tradingClass=NMS), not NYSE, despite being NYSE-listed. Always verify with IB's contract API.

**2. `AttributeError: 'TradingNode' object has no attribute 'add_strategy'`**
*   **Cause:** Calling `node.add_strategy()` — this method does not exist on `TradingNode`.
*   **Fix:** Use `node.trader.add_strategy(strategy)`. The `Trader` object is accessed via `node.trader`.

**3. Strategies not receiving data / `KeyError` on indicators**
*   **Cause:** `node.build()` was called *after* `node.trader.add_strategy()`. Strategies must be added after the node is built.
*   **Fix:** Always call `node.build()` first, then add strategies:
    ```python
    node.build()
    node.trader.add_strategy(strategy)  # after build
    ```

**4. Error 10189 — "No market data permissions for NYSE/ISLAND STK"**
*   **Cause:** Using `INTERNAL` bar types causes NautilusTrader to subscribe to tick-by-tick data. Paper accounts lack the required market data permissions for tick streaming.
*   **Fix:** Use `EXTERNAL` bar types — IB streams the bars directly, no tick subscription needed:
    ```python
    bar_5m = "UNH.NYSE-5-MINUTE-LAST-EXTERNAL"   # ✅
    # NOT: "UNH.NYSE-5-MINUTE-LAST-INTERNAL"      # ❌ requires tick permissions
    ```

**5. IB rejects Market order — "Order type 'MKT' not valid with TIF 'GTC'"**
*   **Cause:** Market orders submitted with `TimeInForce.GTC`. IB only accepts `DAY` for market orders.
*   **Fix:** Use `TimeInForce.DAY` for the entry order. The entry TIF is the `time_in_force` parameter (not `entry_time_in_force` — see entry 17). GTC is valid for limit/stop SL and TP legs.
    ```python
    bracket = self.order_factory.bracket(
        entry_order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,    # ← must be DAY; parameter is 'time_in_force' not 'entry_time_in_force'
        tp_post_only=False,               # ← required; see entry 18
        tp_time_in_force=TimeInForce.GTC,
        sl_time_in_force=TimeInForce.GTC,
    )
    ```

**6. Data and Exec clients use different `client_id` — two TCP connections**
*   **Cause:** Assigning different `ibg_client_id` to `InteractiveBrokersDataClientConfig` and `InteractiveBrokersExecClientConfig`.
*   **Fix:** Use the same `ibg_client_id` for both. The adapter reuses a single `InteractiveBrokersClient` socket.

**7. Wrong client key — factories not matched**
*   **Cause:** Using a string key like `"IBKR"` or `"IB"` (string literal) in `TradingNodeConfig` instead of the canonical `IB` constant.
*   **Fix:** Import and use the canonical constant:
    ```python
    from nautilus_trader.adapters.interactive_brokers.common import IB
    node_config = TradingNodeConfig(
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    ```

**8. "Order Not Found" during Cancellation**
*   **Cause:** The IBKR stream event for `ORDER_CREATE` arrived *after* the strategy tried to cancel the order.
*   **Fix:** The adapter captures the Order ID immediately from the REST response. If you see this, check your network latency.

**9. `NotImplementedError: generate_order_status_report`**
*   **Cause:** The engine tried to query a single order, but the adapter only supported bulk reconciliation.
*   **Fix:** Update `titan` package to the latest version (Resolved in Feb 2026 update).

**10. "Trade ID Unspecified" (Trailing Stop)**
*   **Cause:** Trying to place a Trailing Stop without linking it to an open trade.
*   **Fix:** Use standard Stop Loss orders managed primarily by the strategy logic, rather than IBKR-side trailing stops, until stateful tracking is implemented.

**11. `no execution client configured for NASDAQ or client_id None`**
*   **Cause:** The exec client is registered under the `INTERACTIVE_BROKERS` venue. Orders for `INTC.NASDAQ` or `UNH.NYSE` instruments route by their venue, which doesn't match.
*   **Fix:** Set `routing=RoutingConfig(default=True)` on `InteractiveBrokersExecClientConfig`. Import from `nautilus_trader.live.config`:
    ```python
    from nautilus_trader.live.config import RoutingConfig
    exec_config = InteractiveBrokersExecClientConfig(
        ...,
        routing=RoutingConfig(default=True),
    )
    ```

**12. `TypeError: 'currency' argument was None` in `account.balance_total()`**
*   **Cause:** IB margin accounts have `base_currency=None`. Calling `balance_total()` with no argument raises this error.
*   **Fix:** Get currency dynamically from account balances:
    ```python
    accounts = self.cache.accounts()
    account = accounts[0]
    currencies = list(account.balances().keys())
    equity = float(account.balance_total(currencies[0]).as_double())
    ```

**13. `on_timer` never fires / close logic silently skipped**
*   **Cause:** `on_timer` does not exist in NautilusTrader ≥1.200. Overriding it is silently ignored — the base class `pass` runs instead.
*   **Fix:** Override `on_event` and check the event name:
    ```python
    def on_event(self, event):
        if hasattr(event, "name") and event.name == "close_position":
            # your close logic here
    ```

**14. `close_all_positions()` silently does nothing**
*   **Cause:** Nautilus position state becomes inconsistent when an order goes through `CANCELED → ACCEPTED → FILLED` (IB sends acknowledgement late). The portfolio tracker doesn't recognise the position as open.
*   **Fix:** Submit an explicit counter-side market order instead of relying on `close_all_positions()`:
    ```python
    close_side = OrderSide.BUY if entry_side == OrderSide.SELL else OrderSide.SELL
    close_order = self.order_factory.market(
        instrument_id=self.instrument_id,
        order_side=close_side,
        quantity=qty,
        time_in_force=TimeInForce.DAY,
    )
    self.submit_order(close_order)
    ```

**15. Error 321 — "API interface is currently in Read-Only mode"**
*   **Cause:** TWS/Gateway has the Read-Only API toggle enabled. All order submissions are rejected.
*   **Fix:** In TWS/Gateway: **Edit → Global Configuration → API → Settings → uncheck "Read-Only API"**.

**16. Error 326 — "Unable to connect as the client id is already in use"**
*   **Cause:** A previous strategy or test process is still running and holding the same `ibg_client_id` socket connection.
*   **Fix:** Kill old Python processes. Use distinct `ibg_client_id` values per node (e.g. ORB strategy = env var value, test scripts = hardcoded 22+). The `IBKR_CLIENT_ID` env var is shared — test scripts must hardcode a different ID.

**17. `TypeError: bracket() got an unexpected keyword argument 'entry_time_in_force'`**
*   **Cause:** The `bracket()` method does not have an `entry_time_in_force` parameter. The entry order's time-in-force is set via the shared `time_in_force` parameter.
*   **Fix:** Replace `entry_time_in_force` with `time_in_force`:
    ```python
    # WRONG — raises TypeError, bracket never submitted
    bracket = self.order_factory.bracket(
        entry_time_in_force=TimeInForce.DAY,  # ❌ parameter does not exist
        ...
    )

    # CORRECT
    bracket = self.order_factory.bracket(
        time_in_force=TimeInForce.DAY,        # ✅ controls entry order TIF
        ...
    )
    ```
*   **Impact:** Every bracket order submission silently fails. No trade is executed even when all entry conditions are met.

**18. IB rejects bracket order — TP limit leg rejected, cascades to entry rejection**
*   **Cause:** `order_factory.bracket()` defaults to `tp_post_only=True`. IB rejects limit orders with the post-only flag when they are part of a bracket (OTO/OCO) order group.
*   **Symptom:** Log shows `ORDER REJECTED: ... reason='The order has been rejected due to the rejection of the order with ClientOrderId(...) in the list'` immediately after bracket submission. The entry order is the one visibly rejected, but the root cause is the TP limit leg.
*   **Fix:** Explicitly set `tp_post_only=False`:
    ```python
    bracket = self.order_factory.bracket(
        ...
        tp_post_only=False,   # ← required; default True causes IB to reject TP in brackets
        ...
    )
    ```
*   **Impact:** Every bracket order rejected immediately. No position is ever opened.

**19. Paper account — GTC TP/SL orders do not simulate fills (DELAYED_FROZEN)**
*   **Cause:** IB paper accounts using `DELAYED_FROZEN` market data do not stream live price updates mid-session for new subscriptions. The paper trading engine cannot simulate TP/SL fills if the price feed is frozen.
*   **Symptom:** Entry fills correctly, but TP and SL orders sit open indefinitely without filling. Only the fallback cancel + explicit close path fires.
*   **Fix / Workaround:**
    - The bracket structure itself is correct — use `scripts/test_bracket.py` to verify submission, acceptance, and entry fill.
    - For full TP/SL fill simulation, start the strategy before market open (connection established before 09:30 ET gets a live delayed stream).
    - On a live account (`REALTIME` data), TP/SL fills occur normally.
    - The 5-minute fallback in `test_bracket.py` (cancel + explicit close) provides a confirmed working close path regardless.

**20. FX market orders reject with `FOK` time-in-force (IDEALPRO)**
*   **Cause:** `TimeInForce.FOK` (Fill-or-Kill) is not valid for FX spot orders on IDEALPRO. IB rejects with error 201: *"The time-in-force FOK is invalid for this combination of exchange and security type."*
*   **Symptom:** Market entry immediately rejected. Log shows `ORDER REJECTED` with error 201 on every bar signal.
*   **Fix:** Use `TimeInForce.GTC` for FX market orders on IDEALPRO:
    ```python
    order = self.order_factory.market(
        instrument_id=self.instrument_id,
        order_side=side,
        quantity=qty,
        time_in_force=TimeInForce.GTC,   # ✅ valid for FX
    )
    ```
*   **Note:** For equities on SMART, `TimeInForce.DAY` is the correct choice for market orders.

**21. Account equity is `None` on GBP-denominated paper accounts**
*   **Cause:** `account.balance_total(Currency.from_str("USD"))` returns `None` when the account holds GBP, not USD. IB paper accounts are created in the account's home currency (GBP for UK clients).
*   **Symptom:** `AttributeError: 'NoneType' object has no attribute 'as_double'` in position sizing code.
*   **Fix:** Detect currency dynamically, then FX-convert to USD for consistent sizing:
    ```python
    currencies = list(account.balances().keys())
    acct_ccy = currencies[0]
    equity_raw = float(account.balance_total(acct_ccy).as_double())

    equity = equity_raw
    if str(acct_ccy) != "USD":
        try:
            usd = Currency.from_str("USD")
            fx = self.portfolio.exchange_rate(acct_ccy, usd, PriceType.MID)
            if fx and fx > 0:
                equity = equity_raw * fx
            else:
                raise ValueError("zero rate")
        except Exception:
            self.log.warning("No FX rate cached; using raw balance for sizing.")
    ```
*   **Note:** If no GBPUSD rate is cached (common at startup), position sizing uses raw GBP balance — ~20% undersized vs. USD-equivalent. This clears once the first quote tick is received.

**22. Multiple bar timeframes firing duplicate entry signals before position cache updates**
*   **Cause:** `on_bar` fires for every subscribed timeframe (H1, H4, D, W). If all four bars arrive before the position is reflected in `self.cache.positions()`, `_execute_bias` sees `current_dir=0` on each bar and submits multiple market entries.
*   **Symptom:** 3–4 SELL MKT orders submitted simultaneously; multiple fills; oversized position.
*   **Fix:** Add an `_entry_pending` flag set on order submission and cleared on fill or rejection:
    ```python
    def _open_position(self, side, price):
        if self._entry_pending:
            return  # duplicate bar guard
        ...
        self._entry_pending = True
        self.submit_order(order)

    def on_order_filled(self, event):
        if event.client_order_id in self._entry_order_ids:
            self._entry_pending = False
            ...

    def on_order_rejected(self, event):
        if event.client_order_id in self._entry_order_ids:
            self._entry_pending = False
    ```

**23. ATR stop orders lost across strategy restart (state not reconciled)**
*   **Cause:** `_stop_order_ids` is an in-memory set reset to empty on every restart. After reconnect, NautilusTrader reconciliation re-accepts open stop orders but the strategy doesn't know they belong to it — `_cancel_stops()` silently skips them.
*   **Symptom:** On signal flip after restart, position is closed but ATR stop order remains open in TWS — orphaned GTC stop that can re-open a position unexpectedly.
*   **Fix:** Re-register stop orders during reconciliation via `on_order_accepted`:
    ```python
    def on_order_accepted(self, event):
        order = self.cache.order(event.client_order_id)
        if (
            order is not None
            and order.order_type == OrderType.STOP_MARKET
            and event.client_order_id not in self._stop_order_ids
            and event.client_order_id not in self._entry_order_ids
        ):
            self._stop_order_ids.add(event.client_order_id)
            self.log.info(f"Reconciled stop order: {event.client_order_id}")
    ```

---

## 6a. Multi-Currency Account Pitfalls (April 2026 audit)

The April 21 portfolio-risk audit surfaced a class of bugs that live adjacent to
the adapter: anywhere strategy code reads `account.balances()` or
`account.balance_total(ccy)`, non-determinism and silent currency mismatches
can corrupt sizing.

### 6a.1 Never use `list(balances.keys())[0]`

Python dict ordering reflects insertion order, and IBKR multi-currency
accounts insert in an order the broker chooses. On an account that has held
USD, JPY, and EUR at various times, `list(account.balances().keys())[0]` can
return **any** of them across process restarts. Every strategy that used this
pattern was migrated on April 21 -- reintroducing it is a regression.

**Correct pattern:**
```python
from titan.risk.strategy_equity import get_base_balance

equity = get_base_balance(account, "USD")  # explicit, deterministic
if equity is None or equity <= 0:
    self.log.warning("No USD balance on account; skipping bar.")
    return
```

`get_base_balance` returns `None` if USD is absent, which is **correct**
behaviour -- the strategy should log and skip rather than silently trade
against the wrong currency.

### 6a.2 FX unit conversion for non-USD-quoted instruments

An AUD/JPY spot price is quoted in JPY per 1 AUD. Dividing a USD notional by
that price gives a meaningless unit count:

```python
# ❌ WRONG for AUD/JPY (price is JPY per AUD)
units = int(notional_usd / price)  # ~108 "units" at price=95 -- garbage

# ✅ CORRECT
from titan.risk.strategy_equity import convert_notional_to_units
units = convert_notional_to_units(
    notional_base=notional_usd,
    price=price,                     # JPY per AUD
    quote_ccy="JPY",
    base_ccy="USD",
    fx_rate_quote_to_base=0.0067,    # JPY -> USD rate -- MUST be explicit
)
# ~15,710 AUD units for a 10,000 USD notional at price=95, rate=0.0067
```

The helper **raises `ValueError`** if `quote_ccy != base_ccy` and no FX rate
is supplied -- it refuses to silently assume 1.0. This is the intended
behaviour; configure the rate on your strategy config or subscribe to a
spot feed.

### 6a.3 Diagnostic procedure when sizing looks wrong

1. Log the raw equity your strategy passed to the PRM.
2. Check `get_base_balance(account, "USD")` returns the value you expect.
3. For FX strategies, verify `quote_ccy` and `fx_rate_quote_to_base` in the
   sizing call.
4. Run the 13-test regression suite: `uv run pytest tests/test_portfolio_risk_april2026_fixes.py -v`.

---

## 7. Future Roadmap

### **1. Trailing Stop Support (High Priority)**
To support IBKR's native Trailing Stops, the adapter must be upgraded to be **Stateful**.
*   **Requirement:** Maintain a local mapping of `ClientOrderId` -> `IBKRTradeId`.
*   **Implementation:** When an `ORDER_FILL` event is received, extract the `tradeOpened.id` or `tradeReduced.id` and store it in a local SQLite DB or in-memory cache.
*   **Usage:** When sending a Trailing Stop, look up the `tradeID` for the corresponding position and attach it to the API request.

### **2. Resilience Upgrades**
*   **Rate Limit Handling:** Implement specific `429 Too Many Requests` backoff logic in `execution.py`.
*   **Circuit Breaker:** Automatically pause trading if error rate > 5% in 1 minute.

### **3. Scalability**
*   **Docker:** Finalize the `Dockerfile` for easy deployment to AWS/GCP.
*   **Logging:** Ship logs to CloudWatch or Datadog for remote monitoring.
