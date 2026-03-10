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
| **Market Orders** | ✅ Supported | Immediate fill or FOK (Fill-Or-Kill). |
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
*   **Fix:** Use `TimeInForce.DAY` for market entry orders. GTC is valid for limit/stop SL and TP legs.
    ```python
    bracket = self.order_factory.bracket(
        entry_order_type=OrderType.MARKET,
        entry_time_in_force=TimeInForce.DAY,  # ← must be DAY
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
