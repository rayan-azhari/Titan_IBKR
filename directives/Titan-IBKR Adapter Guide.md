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
To launch the Multi-Timeframe Confluence strategy:
```bash
uv run python scripts/run_live_mtf.py
```
This script will:
1.  Connect to IBKR Gateway/TWS.
2.  Reconcile open positions.
3.  Warm up indicators with historical data.
4.  Begin trading.

### Verifying the Setup
To verify adapter health and order handling:
```bash
uv run python scripts/verify_connection.py
```
This tests the `ibapi` connectivity to the local socket server.

---

## 6. Troubleshooting

### Common Errors

**1. "Order Not Found" during Cancellation**
*   **Cause:** The IBKR stream event for `ORDER_CREATE` arrived *after* the strategy tried to cancel the order.
*   **Fix:** The adapter now captures the Order ID immediately from the REST response. if you see this, check your network latency.

**2. `NotImplementedError: generate_order_status_report`**
*   **Cause:** The engine tried to query a single order, but the adapter only supported bulk reconciliation.
*   **Fix:** Update `titan` package to the latest version (Resolved in Feb 2026 update).

**3. "Trade ID Unspecified" (Trailing Stop)**
*   **Cause:** Trying to place a Trailing Stop without linking it to an open trade.
*   **Fix:** Use standard Stop Loss orders managed primarily by the strategy logic, rather than IBKR-side trailing stops, until stateful tracking is implemented.

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
