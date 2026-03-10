# IBKR & NautilusTrader API Reference

Comprehensive technical reference for building, configuring, and executing strategies via the NautilusTrader Interactive Brokers integration and the raw TWS API.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Installation](#2-installation)
3. [Connection Setup](#3-connection-setup)
4. [Instrument / Contract Specification](#4-instrument--contract-specification)
5. [Instrument Provider Configuration](#5-instrument-provider-configuration)
6. [Data Client — Market Data Subscriptions](#6-data-client--market-data-subscriptions)
7. [Execution Client — Orders](#7-execution-client--orders)
8. [Order Types & Parameters](#8-order-types--parameters)
9. [IB-Specific Order Tags](#9-ib-specific-order-tags)
10. [Account & Portfolio Data](#10-account--portfolio-data)
11. [Historical Data](#11-historical-data)
12. [Full TradingNode Setup](#12-full-tradingnode-setup)
13. [Multi-Account Configuration](#13-multi-account-configuration)
14. [Raw TWS API (ibapi) — Direct Access](#14-raw-tws-api-ibapi--direct-access)
15. [Key Import Paths](#15-key-import-paths)
16. [Common Errors & Fixes](#16-common-errors--fixes)

---

## 1. Architecture Overview

### NautilusTrader IB Adapter Layer

```
Strategy (your code)
    │
    ▼
TradingNode
    ├── InteractiveBrokersDataClient     ← market data subscriptions
    └── InteractiveBrokersExecutionClient ← order management
            │
            ▼
    InteractiveBrokersClient             ← central connection manager
    (ConnectionMixin / ErrorMixin / AccountMixin /
     ContractMixin / MarketDataMixin / OrderMixin)
            │
            ▼
    TWS / IB Gateway (socket, port 4001/4002/7496/7497)
```

### Core API Classes

| Class | Purpose |
|---|---|
| `InteractiveBrokersClient` | Central component managing connections, errors, trades, data |
| `InteractiveBrokersDataClient` | Real-time market data subscriptions |
| `InteractiveBrokersExecutionClient` | Order management and account information |
| `InteractiveBrokersInstrumentProvider` | Instrument definitions and contract details |
| `HistoricInteractiveBrokersClient` | Historical data retrieval for backtesting |
| `InteractiveBrokersLiveDataClientFactory` | Factory to instantiate data clients |
| `InteractiveBrokersLiveExecClientFactory` | Factory to instantiate execution clients |

### Configuration Classes

| Class | Purpose |
|---|---|
| `InteractiveBrokersDataClientConfig` | Data client settings |
| `InteractiveBrokersExecClientConfig` | Execution client settings |
| `InteractiveBrokersInstrumentProviderConfig` | Instrument provider configuration |
| `DockerizedIBGatewayConfig` | Docker gateway settings |
| `IBContract` | Contract/instrument specification |
| `IBOrderTags` | IB-specific order parameters and tags |
| `IBMarketDataTypeEnum` | Enum for market data feed type |
| `SymbologyMethod` | Enum for symbol format (IB_SIMPLIFIED, IB_RAW) |

---

## 2. Installation

```bash
uv pip install "nautilus_trader[ib,docker]"

# Or install all extras
uv sync --all-extras
```

> NautilusTrader repackages `ibapi` as `nautilus-ibapi` on PyPI (official IB wheels are unavailable on PyPI).

---

## 3. Connection Setup

### Port Reference

| Mode | Application | Port |
|---|---|---|
| Live Trading | TWS | **7496** |
| Paper Trading | TWS | **7497** |
| Live Trading | IB Gateway | **4001** |
| Paper Trading | IB Gateway | **4002** |

### Option A — Existing TWS or IB Gateway (direct socket)

```python
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
)

data_config = InteractiveBrokersDataClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,          # TWS paper
    ibg_client_id=1,
)

exec_config = InteractiveBrokersExecClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,
    ibg_client_id=1,
    account_id="DU123456",  # or set env var TWS_ACCOUNT
)
```

### Option B — Dockerized IB Gateway (programmatic container)

```python
from nautilus_trader.adapters.interactive_brokers.config import DockerizedIBGatewayConfig
from nautilus_trader.adapters.interactive_brokers.gateway import DockerizedIBGateway

gateway_config = DockerizedIBGatewayConfig(
    username="your_ib_username",   # or env: TWS_USERNAME
    password="your_ib_password",   # or env: TWS_PASSWORD
    trading_mode="paper",          # "paper" | "live"
    read_only_api=False,
    timeout=300,
)

gateway = DockerizedIBGateway(config=gateway_config)
gateway.start()
```

### Connection Parameters Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ibg_host` | str | `"127.0.0.1"` | Hostname of TWS/IB Gateway |
| `ibg_port` | int | required | Port number |
| `ibg_client_id` | int | required | Unique identifier per connection (0–31) |
| `account_id` | str | None | IB account number; falls back to `TWS_ACCOUNT` env var |
| `connection_timeout` | int | 300 | Seconds before connection timeout |
| `request_timeout_secs` | int | 60 | Seconds to wait for a request response |

### Environment Variables

```bash
export TWS_USERNAME="your_ib_username"
export TWS_PASSWORD="your_ib_password"
export TWS_ACCOUNT="your_account_id"    # e.g. DU123456
export IB_MAX_CONNECTION_ATTEMPTS="5"
```

---

## 4. Instrument / Contract Specification

The `IBContract` class maps to an IB instrument. Use it for both the instrument provider and historical data client.

```python
from nautilus_trader.adapters.interactive_brokers.common import IBContract
```

### By Asset Class

```python
# Stocks / ETFs
IBContract(secType='STK', exchange='SMART', primaryExchange='ARCA',    symbol='SPY')
IBContract(secType='STK', exchange='SMART', primaryExchange='NASDAQ',  symbol='AAPL')
IBContract(secType='STK', exchange='SMART', primaryExchange='NASDAQ',  symbol='MSFT')

# Forex
IBContract(secType='CASH', exchange='IDEALPRO', symbol='EUR', currency='USD')
IBContract(secType='CASH', exchange='IDEALPRO', symbol='GBP', currency='JPY')

# Futures
IBContract(secType='FUT', exchange='CME',   symbol='ES', lastTradeDateOrContractMonth='20240315')
IBContract(secType='FUT', exchange='NYMEX', symbol='CL', lastTradeDateOrContractMonth='20240315')

# Continuous Futures (chain)
IBContract(secType='CONTFUT', exchange='CME',  symbol='ES', build_futures_chain=True)
IBContract(secType='CONTFUT', exchange='NYMEX', symbol='CL',
           build_futures_chain=True, min_expiry_days=30, max_expiry_days=180)

# Equity Options (single contract)
IBContract(secType='OPT', exchange='SMART', symbol='SPY',
           lastTradeDateOrContractMonth='20251219', strike=500, right='C')

# Equity Options Chain (all strikes/expirations)
IBContract(secType='STK', exchange='SMART', primaryExchange='ARCA', symbol='SPY',
           build_options_chain=True, min_expiry_days=10, max_expiry_days=60)

# Options on Futures
IBContract(secType='FOP', exchange='CME', symbol='ES',
           lastTradeDateOrContractMonth='20240315', strike=4200, right='C')

# Crypto
IBContract(secType='CRYPTO', symbol='BTC', exchange='PAXOS', currency='USD')
IBContract(secType='CRYPTO', symbol='ETH', exchange='PAXOS', currency='USD')

# Indices
IBContract(secType='IND', symbol='SPX', exchange='CBOE')
IBContract(secType='IND', symbol='NDX', exchange='NASDAQ')

# CFDs
IBContract(secType='CFD', symbol='IBUS30')
IBContract(secType='CFD', symbol='DE40EUR', exchange='SMART')

# Bonds (by ISIN or CUSIP)
IBContract(secType='BOND', secIdType='ISIN',  secId='US03076KAA60')
IBContract(secType='BOND', secIdType='CUSIP', secId='912828XE8')

# Commodities
IBContract(secType='CMDTY', symbol='XAUUSD', exchange='SMART')
```

### IBContract Field Reference

| Field | Description |
|---|---|
| `secType` | `STK`, `BOND`, `OPT`, `FUT`, `CONTFUT`, `FOP`, `CASH`, `CRYPTO`, `IND`, `CFD`, `CMDTY` |
| `symbol` | Ticker symbol |
| `exchange` | Routing exchange: `SMART`, `CME`, `NYMEX`, `IDEALPRO`, `PAXOS`, `CBOE`, etc. |
| `primaryExchange` | Listing exchange: `ARCA`, `NASDAQ`, `NYSE` |
| `currency` | `USD`, `EUR`, `GBP`, `JPY`, etc. |
| `lastTradeDateOrContractMonth` | Expiry: `YYYYMMDD` or `YYYYMM` |
| `strike` | Option strike price (float) |
| `right` | Option type: `'C'` (call) or `'P'` (put) |
| `secIdType` | `ISIN` or `CUSIP` (bonds only) |
| `secId` | The ISIN or CUSIP value |
| `build_options_chain` | Bool — load all strikes/expirations for equity options |
| `build_futures_chain` | Bool — load all contract months |
| `options_chain_exchange` | Override exchange for options chain loading |
| `min_expiry_days` | Minimum days to expiration filter |
| `max_expiry_days` | Maximum days to expiration filter |

---

## 5. Instrument Provider Configuration

```python
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersInstrumentProviderConfig,
    SymbologyMethod,
)

# Simple: load by symbol string
instrument_provider_config = InteractiveBrokersInstrumentProviderConfig(
    symbology_method=SymbologyMethod.IB_SIMPLIFIED,
    load_ids=frozenset([
        "EUR/USD.IDEALPRO",
        "GBP/USD.IDEALPRO",
        "SPY.ARCA",
        "QQQ.NASDAQ",
        "AAPL.NASDAQ",
        "ESM4.CME",
        "BTC/USD.PAXOS",
        "^SPX.CBOE",
    ]),
)

# Advanced: load by IBContract (options chains, futures chains, etc.)
advanced_config = InteractiveBrokersInstrumentProviderConfig(
    symbology_method=SymbologyMethod.IB_SIMPLIFIED,
    build_futures_chain=True,
    build_options_chain=True,
    min_expiry_days=7,
    max_expiry_days=90,
    load_contracts=frozenset([
        IBContract(secType='STK', symbol='SPY', exchange='SMART',
                   primaryExchange='ARCA', build_options_chain=True),
        IBContract(secType='CONTFUT', exchange='CME', symbol='ES',
                   build_futures_chain=True),
    ]),
)

# With MIC venue conversion
mic_config = InteractiveBrokersInstrumentProviderConfig(
    convert_exchange_to_mic_venue=True,
    symbology_method=SymbologyMethod.IB_SIMPLIFIED,
    symbol_to_mic_venue={"ES": "XCME", "SPY": "ARCX"},
)
```

### Symbology Format — IB_SIMPLIFIED (default)

| Asset Class | Format | Example |
|---|---|---|
| Forex | `{symbol}/{currency}.{exchange}` | `EUR/USD.IDEALPRO` |
| Stocks | `{localSymbol}.{primaryExchange}` | `SPY.ARCA` |
| Futures | `{localSymbol}.{exchange}` | `ESM4.CME` |
| Continuous Futures | `{symbol}.{exchange}` | `ES.CME` |
| Options | `{localSymbol}.{exchange}` | `AAPL230217P00155000.SMART` |
| Indices | `^{localSymbol}.{exchange}` | `^SPX.CBOE` |
| Crypto | `{symbol}/{currency}.{exchange}` | `BTC/USD.PAXOS` |

### MIC Venue Conversions

`CME` → `XCME` | `NASDAQ` → `XNAS` | `NYSE` → `XNYS` | `ARCA` → `ARCX` | `LSE` → `XLON`

---

## 6. Data Client — Market Data Subscriptions

### Data Client Config

```python
from nautilus_trader.adapters.interactive_brokers.config import (
    IBMarketDataTypeEnum,
    InteractiveBrokersDataClientConfig,
)

data_client_config = InteractiveBrokersDataClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,
    ibg_client_id=1,
    use_regular_trading_hours=True,        # RTH-only bars for equities
    market_data_type=IBMarketDataTypeEnum.DELAYED_FROZEN,  # paper/dev
    ignore_quote_tick_size_updates=False,  # filter size-only quote updates
    handle_revised_bars=True,              # process IB bar revisions
    instrument_provider=instrument_provider_config,
    connection_timeout=300,
    request_timeout_secs=60,
)

# Production — real-time
prod_data_config = InteractiveBrokersDataClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=4001,
    ibg_client_id=1,
    use_regular_trading_hours=False,
    market_data_type=IBMarketDataTypeEnum.REALTIME,
    ignore_quote_tick_size_updates=True,
    handle_revised_bars=True,
    instrument_provider=instrument_provider_config,
    dockerized_gateway=dockerized_gateway_config,
)
```

### Market Data Type Enum

| Value | Meaning |
|---|---|
| `IBMarketDataTypeEnum.REALTIME` | Live streaming (requires subscription) |
| `IBMarketDataTypeEnum.FROZEN` | Last available snapshot when market closes |
| `IBMarketDataTypeEnum.DELAYED` | 10–15 min delayed (free) |
| `IBMarketDataTypeEnum.DELAYED_FROZEN` | Delayed + frozen fallback |

### Subscribing Within a Strategy

```python
from nautilus_trader.trading.strategy import Strategy

class MyStrategy(Strategy):
    def on_start(self):
        instrument_id = self.instrument.id

        # Quote ticks (bid/ask with sizes)
        self.subscribe_quote_ticks(instrument_id)

        # Trade ticks (last price and volume)
        self.subscribe_trade_ticks(instrument_id)

        # Bars (OHLCV)
        from nautilus_trader.model.data import BarType
        bar_type = BarType.from_str("EUR/USD.IDEALPRO-1-MINUTE-MID-EXTERNAL")
        self.subscribe_bars(bar_type)

        # Level 2 order book
        self.subscribe_order_book_deltas(instrument_id)

    def on_quote_tick(self, tick):
        print(f"Bid: {tick.bid_price}  Ask: {tick.ask_price}")

    def on_trade_tick(self, tick):
        print(f"Last: {tick.price}  Size: {tick.size}")

    def on_bar(self, bar):
        print(f"O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}")
```

---

## 7. Execution Client — Orders

### Exec Client Config

```python
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersExecClientConfig
from nautilus_trader.config import RoutingConfig

exec_client_config = InteractiveBrokersExecClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,
    ibg_client_id=1,
    account_id="DU123456",
    instrument_provider=instrument_provider_config,
    connection_timeout=300,
    routing=RoutingConfig(default=True),
    fetch_all_open_orders=False,
    track_option_exercise_from_position_update=False,
)
```

### Placing Orders in a Strategy

```python
from nautilus_trader.model.enums import OrderSide, TimeInForce

class MyStrategy(Strategy):
    def buy_market(self):
        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(100),
        )
        self.submit_order(order)

    def buy_limit(self, price: float):
        order = self.order_factory.limit(
            instrument_id=self.instrument.id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(100),
            price=self.instrument.make_price(price),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def sell_stop(self, trigger_price: float):
        order = self.order_factory.stop_market(
            instrument_id=self.instrument.id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(100),
            trigger_price=self.instrument.make_price(trigger_price),
        )
        self.submit_order(order)

    def bracket_entry(self, entry: float, tp: float, sl: float):
        bracket = self.order_factory.bracket(
            instrument_id=self.instrument.id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(100),
            entry_price=self.instrument.make_price(entry),
            tp_price=self.instrument.make_price(tp),
            sl_trigger_price=self.instrument.make_price(sl),
        )
        self.submit_order_list(bracket)

    def cancel_specific(self, order):
        self.cancel_order(order)

    def cancel_all(self):
        self.cancel_all_orders(self.instrument.id)
```

---

## 8. Order Types & Parameters

### Supported Order Types (NautilusTrader → IB mapping)

| Nautilus Type | IB Equivalent | Notes |
|---|---|---|
| `MARKET` | `MKT` | Immediate execution at market |
| `LIMIT` | `LMT` | Execute at limit price or better |
| `STOP_MARKET` | `STP` | Triggers market order at stop price |
| `STOP_LIMIT` | `STP LMT` | Triggers limit order at stop price |
| `MARKET_IF_TOUCHED` | `MIT` | Like limit, but triggers market |
| `LIMIT_IF_TOUCHED` | `LIT` | Like stop, but triggers limit |
| `TRAILING_STOP_MARKET` | `TRAIL` | Trailing stop → market |
| `TRAILING_STOP_LIMIT` | `TRAIL LMT` | Trailing stop → limit |
| `MARKET` + `AT_THE_CLOSE` | `MOC` | Market On Close |
| `LIMIT` + `AT_THE_CLOSE` | `LOC` | Limit On Close |

### Time In Force Options

| TIF | Meaning |
|---|---|
| `DAY` | Day order (default) |
| `GTC` | Good Till Cancelled |
| `IOC` | Immediate or Cancel |
| `FOK` | Fill or Kill |
| `GTD` | Good Till Date |
| `AT_THE_OPEN` | Execute at the open |
| `AT_THE_CLOSE` | Execute at the close |

---

## 9. IB-Specific Order Tags

Use `IBOrderTags` to pass IB-native parameters that have no direct Nautilus equivalent.

```python
from nautilus_trader.adapters.interactive_brokers.common import IBOrderTags

# Basic IB-specific params
order_tags = IBOrderTags(
    allOrNone=True,                          # All-or-nothing
    ocaGroup="MY_OCA_GROUP",                 # OCA group name
    ocaType=1,                               # 1=Cancel All, 2=Reduce block, 3=Reduce no block
    activeStartTime="20240315 09:30:00 EST", # GTC window start
    activeStopTime="20240315 16:00:00 EST",  # GTC window end
    goodAfterTime="20240315 09:35:00 EST",   # Activation time
)

order = self.order_factory.limit(
    instrument_id=self.instrument.id,
    order_side=OrderSide.BUY,
    quantity=self.instrument.make_qty(100),
    price=self.instrument.make_price(100.0),
    tags=[order_tags.value],
)
```

### OCA (One-Cancels-All) Orders

```python
oca_tags = IBOrderTags(ocaGroup="MY_OCA_GROUP", ocaType=1)

bracket = self.order_factory.bracket(
    instrument_id=self.instrument.id,
    order_side=OrderSide.BUY,
    quantity=self.instrument.make_qty(100),
    entry_price=self.instrument.make_price(100.0),
    tp_price=self.instrument.make_price(110.0),
    sl_trigger_price=self.instrument.make_price(90.0),
    tp_tags=[oca_tags.value],
    sl_tags=[oca_tags.value],
)
self.submit_order_list(bracket)
```

### Conditional Orders

```python
# Price condition — trigger when AAPL (conId=265598) crosses above $250
price_condition = {
    "type": "price",
    "conId": 265598,
    "exchange": "SMART",
    "isMore": True,
    "price": 250.00,
    "triggerMethod": 0,   # 0=default, 1=double-bid/ask, 2=last, 4=mid-point, 8=bid/ask
    "conjunction": "and", # "and" | "or" for chaining multiple conditions
}

# Time condition
time_condition = {
    "type": "time",
    "time": "20250315-09:30:00",
    "isMore": True,
    "conjunction": "and",
}

# Volume condition
volume_condition = {
    "type": "volume",
    "conId": 265598,
    "exchange": "SMART",
    "isMore": True,
    "volume": 10_000_000,
    "conjunction": "and",
}

# Margin condition
margin_condition = {
    "type": "margin",
    "percent": 75,
    "isMore": True,
    "conjunction": "and",
}

# Percent-change condition
pct_condition = {
    "type": "percent_change",
    "conId": 495512563,
    "exchange": "CME",
    "changePercent": 5.0,
    "isMore": True,
    "conjunction": "and",
}

order_tags = IBOrderTags(
    conditions=[price_condition, time_condition],
    conditionsCancelOrder=False,  # False=submit order when conditions met; True=cancel
)

order = self.order_factory.limit(
    instrument_id=self.instrument.id,
    order_side=OrderSide.BUY,
    quantity=self.instrument.make_qty(100),
    price=self.instrument.make_price(251.00),
    tags=[order_tags.value],
)
```

### IBOrderTags Parameter Reference

| Parameter | Description |
|---|---|
| `allOrNone` | All-or-nothing fill requirement |
| `ocaGroup` | OCA group identifier string |
| `ocaType` | 1=Cancel All, 2=Reduce with block, 3=Reduce no block |
| `activeStartTime` | GTC time window start (`"YYYYMMDD HH:MM:SS TZ"`) |
| `activeStopTime` | GTC time window stop |
| `goodAfterTime` | Order activation time |
| `conditions` | List of condition dicts (price, time, volume, execution, margin, percent_change) |
| `conditionsCancelOrder` | `True`=cancel on conditions met; `False`=submit |

---

## 10. Account & Portfolio Data

NautilusTrader exposes account data via the `Portfolio` object within a strategy.

```python
class MyStrategy(Strategy):
    def check_account(self):
        account = self.portfolio.account(self.exec_client.account_id)

        # Cash / margin
        cash = account.balance_total()
        free = account.balance_free()
        locked = account.balance_locked()

        # Position
        position = self.portfolio.net_position(self.instrument.id)
        is_flat = self.portfolio.is_flat(self.instrument.id)
        is_net_long = self.portfolio.is_net_long(self.instrument.id)
        is_net_short = self.portfolio.is_net_short(self.instrument.id)

        # Unrealized PnL
        upnl = self.portfolio.unrealized_pnl(self.instrument.id)

        # Realized PnL
        rpnl = self.portfolio.realized_pnl(self.instrument.id)
```

---

## 11. Historical Data

### HistoricInteractiveBrokersClient

Used for downloading historical data for backtesting or catalog population.

```python
import asyncio
import datetime
from nautilus_trader.adapters.interactive_brokers.common import IBContract
from nautilus_trader.adapters.interactive_brokers.historical.client import (
    HistoricInteractiveBrokersClient,
)
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from ibapi.common import MarketDataTypeEnum

async def download_history():
    client = HistoricInteractiveBrokersClient(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        market_data_type=MarketDataTypeEnum.DELAYED_FROZEN,
        log_level="INFO",
    )

    await client.connect()
    await asyncio.sleep(2)  # Allow TWS handshake to complete

    contracts = [
        IBContract(secType="STK", symbol="AAPL", exchange="SMART", primaryExchange="NASDAQ"),
        IBContract(secType="CASH", symbol="EUR",  currency="USD",   exchange="IDEALPRO"),
    ]

    # Load instrument definitions
    instruments = await client.request_instruments(contracts=contracts)

    # Request OHLCV bars
    # Bar spec format: "{size}-{timeframe}-{price_type}"
    # Price types: LAST | MID | BID | ASK
    bars = await client.request_bars(
        bar_specifications=["1-MINUTE-LAST", "5-MINUTE-MID", "1-HOUR-LAST", "1-DAY-LAST"],
        start_date_time=datetime.datetime(2023, 11, 1, 9, 30),
        end_date_time=datetime.datetime(2023, 11, 6, 16, 30),
        tz_name="America/New_York",
        contracts=contracts,
        use_rth=True,
        timeout=120,
    )

    # Request tick data
    ticks = await client.request_ticks(
        tick_types=["TRADES", "BID_ASK"],
        start_date_time=datetime.datetime(2023, 11, 6, 9, 30),
        end_date_time=datetime.datetime(2023, 11, 6, 16, 30),
        tz_name="America/New_York",
        contracts=contracts,
        use_rth=True,
        timeout=120,
    )

    # Persist to Parquet catalog
    catalog = ParquetDataCatalog("./catalog")
    catalog.write_data(instruments)
    catalog.write_data(bars)
    catalog.write_data(ticks)

    await client.disconnect()

asyncio.run(download_history())
```

### Bar Specification Examples

| Spec String | Meaning |
|---|---|
| `"1-SECOND-LAST"` | 1-second bars, last price |
| `"1-MINUTE-LAST"` | 1-minute bars, last price |
| `"5-MINUTE-MID"` | 5-minute bars, midpoint |
| `"1-HOUR-LAST"` | 1-hour bars, last price |
| `"1-DAY-LAST"` | Daily bars, last price |
| `"1-MONTH-LAST"` | Monthly bars, last price |

---

## 12. Full TradingNode Setup

### Paper Trading (standard setup)

```python
import os
from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.config import (
    IBMarketDataTypeEnum,
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
    SymbologyMethod,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import (
    LiveDataEngineConfig,
    LoggingConfig,
    RoutingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode

# --- Instrument provider ---
instrument_provider_config = InteractiveBrokersInstrumentProviderConfig(
    symbology_method=SymbologyMethod.IB_SIMPLIFIED,
    load_ids=frozenset(["SPY.ARCA", "EUR/USD.IDEALPRO"]),
)

# --- Data client ---
data_client_config = InteractiveBrokersDataClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,
    ibg_client_id=1,
    use_regular_trading_hours=True,
    market_data_type=IBMarketDataTypeEnum.DELAYED_FROZEN,
    instrument_provider=instrument_provider_config,
)

# --- Exec client ---
exec_client_config = InteractiveBrokersExecClientConfig(
    ibg_host="127.0.0.1",
    ibg_port=7497,
    ibg_client_id=1,
    account_id=os.getenv("TWS_ACCOUNT"),
    instrument_provider=instrument_provider_config,
    routing=RoutingConfig(default=True),
)

# --- Strategy ---
from my_strategy import MyStrategyConfig, MyStrategy
strategy_config = MyStrategyConfig(instrument_id="SPY.ARCA")
strategy = MyStrategy(config=strategy_config)

# --- Node ---
config_node = TradingNodeConfig(
    trader_id="TITAN-PAPER-001",
    logging=LoggingConfig(log_level="INFO"),
    data_clients={IB: data_client_config},
    exec_clients={IB: exec_client_config},
    data_engine=LiveDataEngineConfig(
        time_bars_timestamp_on_close=False,
        validate_data_sequence=True,
    ),
    timeout_connection=90.0,
    timeout_reconciliation=5.0,
    timeout_portfolio=5.0,
    timeout_disconnection=5.0,
    timeout_post_stop=2.0,
)

node = TradingNode(config=config_node)
node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
node.build()
node.add_strategy(strategy)

try:
    node.run()
finally:
    node.dispose()
```

### Live Trading (Dockerized Gateway)

```python
from nautilus_trader.adapters.interactive_brokers.config import DockerizedIBGatewayConfig

dockerized_gateway_config = DockerizedIBGatewayConfig(
    username=os.getenv("TWS_USERNAME"),
    password=os.getenv("TWS_PASSWORD"),
    trading_mode="live",
    read_only_api=False,
    timeout=300,
)

data_client_config = InteractiveBrokersDataClientConfig(
    ibg_client_id=1,
    use_regular_trading_hours=False,
    market_data_type=IBMarketDataTypeEnum.REALTIME,
    instrument_provider=instrument_provider_config,
    dockerized_gateway=dockerized_gateway_config,
)

exec_client_config = InteractiveBrokersExecClientConfig(
    ibg_client_id=1,
    account_id=os.getenv("TWS_ACCOUNT"),
    instrument_provider=instrument_provider_config,
    dockerized_gateway=dockerized_gateway_config,
    routing=RoutingConfig(default=True),
)
```

### Strategy Skeleton

```python
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from pydantic import Field


class MyStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str = "SPY.ARCA-1-MINUTE-LAST-EXTERNAL"
    trade_size: float = 100.0


class MyStrategy(Strategy):
    def __init__(self, config: MyStrategyConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.trade_size = config.trade_size

    def on_start(self):
        self.instrument = self.cache.instrument(self.instrument_id)
        self.subscribe_bars(self.bar_type)
        self.subscribe_quote_ticks(self.instrument_id)

    def on_bar(self, bar: Bar):
        # Trading logic here
        if self.should_buy():
            self.enter_long()

    def on_quote_tick(self, tick: QuoteTick):
        pass  # real-time bid/ask updates

    def enter_long(self):
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)

    def on_stop(self):
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

    def should_buy(self) -> bool:
        return False  # implement your logic
```

---

## 13. Multi-Account Configuration

```python
config_node = TradingNodeConfig(
    trader_id="TITAN-MULTI-001",
    data_clients={"IB": data_client_config},
    exec_clients={
        "IB-PAPER": InteractiveBrokersExecClientConfig(
            ibg_host="127.0.0.1", ibg_port=7497, ibg_client_id=2,
            account_id="DU123456",
            instrument_provider=instrument_provider_config,
            routing=RoutingConfig(default=False),
        ),
        "IB-LIVE": InteractiveBrokersExecClientConfig(
            ibg_host="127.0.0.1", ibg_port=4001, ibg_client_id=3,
            account_id="U987654",
            instrument_provider=instrument_provider_config,
            routing=RoutingConfig(default=True),
        ),
    },
)
```

Account identifier format: `AccountId("IB-{key}-{account_id}")`
e.g., `AccountId("IB-PAPER-DU123456")`, `AccountId("IB-LIVE-U987654")`

```python
from nautilus_trader.model.identifiers import AccountId, ClientId

class MultiAccountStrategy(Strategy):
    def on_start(self):
        self.paper_account = AccountId("IB-PAPER-DU123456")
        self.live_account  = AccountId("IB-LIVE-U987654")

    def submit_to_paper(self, order):
        self.submit_order(order, client_id=ClientId("IB-PAPER"))

    def submit_to_live(self, order):
        self.submit_order(order, client_id=ClientId("IB-LIVE"))
```

---

## 14. Raw TWS API (ibapi) — Direct Access

Use this when you need capabilities not yet exposed by NautilusTrader's IB adapter (e.g., scanner subscriptions, fundamental data, WSH calendar events).

### Connection Pattern

```python
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from decimal import Decimal

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None

    def error(self, reqId, errorTime, errorCode, errorMsg, advancedOrderRejectJson=""):
        print(f"reqId:{reqId} errorCode:{errorCode} msg:{errorMsg}")

    def nextValidId(self, orderId: int):
        self.nextValidOrderId = orderId

app = IBapi()
app.connect("127.0.0.1", 7497, clientId=1)

# Start the message loop in a background thread
thread = threading.Thread(target=app.run, daemon=True)
thread.start()

import time; time.sleep(1)  # Wait for connection
```

### Contract Definition

```python
# Stock
c = Contract()
c.symbol = "AAPL"
c.secType = "STK"
c.currency = "USD"
c.exchange = "SMART"
c.primaryExch = "NASDAQ"

# Futures
c = Contract()
c.symbol = "ES"
c.secType = "FUT"
c.exchange = "CME"
c.currency = "USD"
c.lastTradeDateOrContractMonth = "202512"

# Forex
c = Contract()
c.symbol = "EUR"
c.secType = "CASH"
c.currency = "USD"
c.exchange = "IDEALPRO"

# Option
c = Contract()
c.symbol = "AAPL"
c.secType = "OPT"
c.exchange = "SMART"
c.currency = "USD"
c.lastTradeDateOrContractMonth = "20251219"
c.strike = 200.0
c.right = "C"
c.multiplier = "100"
```

### Market Data

```python
# Streaming data
app.reqMktData(tickerId=1001, contract=c, genericTickList="233,236",
               snapshot=False, regulatorySnapshot=False, mktDataOptions=[])
app.cancelMktData(1001)

# Market data type
app.reqMarketDataType(1)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen

# Tick-by-tick
app.reqTickByTickData(1002, c, tickType="BidAsk", numberOfTicks=0, ignoreSize=False)
# tickType: "Last" | "AllLast" | "BidAsk" | "MidPoint"

# Level 2 depth
app.reqMarketDepth(1003, c, numRows=10, isSmartDepth=True, mktDepthOptions=[])
app.cancelMktDepth(1003, isSmartDepth=True)

# Real-time 5-second bars
app.reqRealTimeBars(1004, c, barSize=5, whatToShow="TRADES", useRTH=False,
                   realTimeBarsOptions=[])
app.cancelRealTimeBars(1004)
```

### Historical Data

```python
app.reqHistoricalData(
    reqId=2001,
    contract=c,
    endDateTime="",               # "" = now; or "YYYYMMDD HH:MM:SS TZ"
    durationStr="5 D",            # "60 S" | "5 D" | "2 W" | "3 M" | "1 Y"
    barSizeSetting="1 hour",      # see bar size table below
    whatToShow="TRADES",          # TRADES | MIDPOINT | BID | ASK | BID_ASK | ADJUSTED_LAST
    useRTH=1,                     # 1=RTH only, 0=all hours
    formatDate=1,                 # 1=string, 2=UNIX timestamp
    keepUpToDate=False,           # True=subscribe to live updates
    chartOptions=[]
)
app.cancelHistoricalData(2001)
```

**Valid barSizeSetting values:**
```
"1 secs"  "5 secs"  "10 secs"  "15 secs"  "30 secs"
"1 min"   "2 mins"  "3 mins"   "5 mins"   "10 mins"  "15 mins"  "20 mins"  "30 mins"
"1 hour"  "2 hours" "3 hours"  "4 hours"  "8 hours"
"1 day"   "1 week"  "1 month"
```

### Order Placement

```python
# Market order
order = Order()
order.action = "BUY"           # "BUY" | "SELL"
order.orderType = "MKT"
order.totalQuantity = Decimal("100")
order.tif = "DAY"
app.placeOrder(app.nextValidOrderId, c, order)
app.nextValidOrderId += 1

# Limit order
order = Order()
order.action = "BUY"
order.orderType = "LMT"
order.totalQuantity = Decimal("100")
order.lmtPrice = 150.00
order.tif = "GTC"
app.placeOrder(app.nextValidOrderId, c, order)
app.nextValidOrderId += 1

# Stop order
order = Order()
order.action = "SELL"
order.orderType = "STP"
order.totalQuantity = Decimal("100")
order.auxPrice = 145.00        # stop trigger price
app.placeOrder(app.nextValidOrderId, c, order)

# Stop Limit
order = Order()
order.action = "SELL"
order.orderType = "STP LMT"
order.totalQuantity = Decimal("100")
order.lmtPrice = 144.50
order.auxPrice = 145.00        # stop trigger price
app.placeOrder(app.nextValidOrderId, c, order)

# Trailing Stop
order = Order()
order.action = "SELL"
order.orderType = "TRAIL"
order.totalQuantity = Decimal("100")
order.trailingPercent = 2.0    # 2% trailing; OR use auxPrice for fixed offset
app.placeOrder(app.nextValidOrderId, c, order)

# Bracket Order (parent + TP + SL must be linked)
def make_bracket(parent_id, action, qty, entry, tp_price, sl_price):
    parent = Order()
    parent.orderId = parent_id
    parent.action = action
    parent.orderType = "LMT"
    parent.totalQuantity = qty
    parent.lmtPrice = entry
    parent.transmit = False  # hold, don't send yet

    tp = Order()
    tp.orderId = parent_id + 1
    tp.action = "SELL" if action == "BUY" else "BUY"
    tp.orderType = "LMT"
    tp.totalQuantity = qty
    tp.lmtPrice = tp_price
    tp.parentId = parent_id
    tp.transmit = False

    sl = Order()
    sl.orderId = parent_id + 2
    sl.action = "SELL" if action == "BUY" else "BUY"
    sl.orderType = "STP"
    sl.auxPrice = sl_price
    sl.totalQuantity = qty
    sl.parentId = parent_id
    sl.transmit = True   # transmit=True on last leg sends all three

    return [parent, tp, sl]

bracket = make_bracket(app.nextValidOrderId, "BUY", Decimal("100"), 150.0, 160.0, 145.0)
for leg in bracket:
    app.placeOrder(leg.orderId, c, leg)
app.nextValidOrderId += 3
```

### Order Modification & Cancellation

```python
# Modify: re-submit same orderId with updated fields
order.lmtPrice = 151.00
app.placeOrder(existing_order_id, c, order)

# Cancel single order
app.cancelOrder(orderId, manualOrderCancelTime="")

# Cancel ALL open orders
app.reqGlobalCancel()
```

### Order Status Callback

```python
def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
    # status: "PendingSubmit" | "PreSubmitted" | "Submitted" |
    #         "Filled" | "Cancelled" | "Inactive" | "PendingCancel"
    pass

def execDetails(self, reqId, contract, execution):
    # execution.execId, execution.time, execution.acctNumber,
    # execution.exchange, execution.side, execution.shares,
    # execution.price, execution.permId, execution.clientId,
    # execution.orderId, execution.liquidation, execution.cumQty,
    # execution.avgPrice, execution.orderRef, execution.evRule,
    # execution.evMultiplier, execution.modelCode, execution.lastLiquidity
    pass
```

### Account & Portfolio (raw ibapi)

```python
# Account updates (one account at a time, updates every ~3 minutes)
app.reqAccountUpdates(True, "U1234567")
# Callbacks: updateAccountValue, updatePortfolio, updateAccountTime, accountDownloadEnd

# Account summary (max 2 active subscriptions)
app.reqAccountSummary(9001, "All",
    "NetLiquidation,TotalCashValue,BuyingPower,MaintMarginReq,AvailableFunds")
app.cancelAccountSummary(9001)
# Callback: accountSummary(reqId, account, tag, value, currency)

# All positions across all accounts
app.reqPositions()
app.cancelPositions()
# Callback: position(account, contract, position, avgCost)

# P&L subscription
app.reqPnL(reqId=8001, account="U1234567", modelCode="")
app.cancelPnL(8001)
# Callback: pnl(reqId, dailyPnL, unrealizedPnL, realizedPnL)

app.reqPnLSingle(reqId=8002, account="U1234567", modelCode="", conId=265598)
app.cancelPnLSingle(8002)
# Callback: pnlSingle(reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value)
```

### Key TWS API Limits

| Limit | Value |
|---|---|
| Max messages per second | 50 (across all clients on one TWS) |
| Max simultaneous client connections | 32 per TWS instance |
| Max active account summary subscriptions | 2 |
| Max tick data lines (default) | 100 |
| Historical data throttle | ~60 requests per 10 minutes |
| Tick-by-tick simultaneous subscriptions | Limited by data subscription |

---

## 15. Key Import Paths

```python
# NautilusTrader IB adapter
from nautilus_trader.adapters.interactive_brokers.common import (
    IBContract, IBOrderTags, IB, IB_VENUE,
)
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
    DockerizedIBGatewayConfig,
    IBMarketDataTypeEnum,
    SymbologyMethod,
)
from nautilus_trader.adapters.interactive_brokers.gateway import DockerizedIBGateway
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.adapters.interactive_brokers.historical.client import (
    HistoricInteractiveBrokersClient,
)

# NautilusTrader core
from nautilus_trader.config import (
    RoutingConfig,
    LiveDataEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
    StrategyConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import (
    InstrumentId, AccountId, ClientId,
)
from nautilus_trader.model.objects import Price, Quantity

# Raw ibapi (use when you need something the adapter doesn't expose)
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import MarketDataTypeEnum
```

---

## 16. Common Errors & Fixes

| Error Code | Message | Cause | Fix |
|---|---|---|---|
| 162 | Historical market data service error | Pacing violation (too many historical requests) | Add delay between `reqHistoricalData` calls; max ~60/10 min |
| 200 | No security definition found | Bad contract definition | Check `secType`, `exchange`, `symbol`, `currency` |
| 201 | Order rejected | Invalid order type/TIF combo for the instrument | Check allowed order types via `reqContractDetails` |
| 202 | Order cancelled | TWS cancelled the order | Check order validity; check margin |
| 321 | Error validating request | clientId already in use | Use a different `clientId` |
| 501 | Already connected | Duplicate connection attempt | Check for leftover connections |
| 502 | Couldn't connect to TWS | Socket refused | Verify TWS is running, port is correct, API enabled |
| 504 | Not connected | Sent messages before `nextValidId` fired | Wait for `nextValidId` callback before sending requests |
| 10182 | Failed to request live updates | `keepUpToDate=True` requires `endDateTime=""` | Set `endDateTime=""` |

### NautilusTrader Specific

| Symptom | Cause | Fix |
|---|---|---|
| Instrument not found in cache | `load_ids` / `load_contracts` not populated before strategy starts | Ensure instrument provider config lists all needed instruments |
| Orders silently dropped | `transmit=False` on last bracket leg | Set `transmit=True` on the last leg of a bracket |
| No bar data during off-hours | `use_regular_trading_hours=True` | Set to `False` for forex/futures 24h markets |
| `account_id` mismatch | Account ID not matching between exec client and IB account | Verify with `reqManagedAccts()` or set via `TWS_ACCOUNT` env var |
| Reconnect loop on startup | Docker gateway not ready | Increase `connection_timeout` and `timeout_connection` |
| `Cannot start bar aggregation: no instrument found for X.EXCHANGE` | `IBContract` missing `primaryExchange` — adapter generates `X.SMART` instead of `X.NYSE` | Set `primaryExchange` on every `IBContract`; instrument ID = `{localSymbol}.{primaryExchange}` |
| Error 10189 — "No market data permissions for NYSE/ISLAND STK" | Using `INTERNAL` bar types triggers tick-by-tick subscription; paper accounts lack permissions | Switch to `EXTERNAL` bars: `SYMBOL.EXCH-5-MINUTE-LAST-EXTERNAL` |
| `AttributeError: 'TradingNode' object has no attribute 'add_strategy'` | Calling `node.add_strategy()` — method does not exist on `TradingNode` | Use `node.trader.add_strategy(strategy)` |
| IB rejects market order: "Order type MKT not valid with TIF GTC" | Market orders cannot use `TimeInForce.GTC` on IB | Use `TimeInForce.DAY` for market entry; `GTC` is valid for limit/stop legs only |
| Two separate TCP connections to IB | Data client and exec client have different `ibg_client_id` | Use the same `ibg_client_id` for both — adapter reuses a single socket |
| Client factories not matched / engines not connecting | Using string key `"IBKR"` or `"IB"` literal instead of canonical `IB` constant | `from nautilus_trader.adapters.interactive_brokers.common import IB`; use `IB` as the key |
| Strategies receive no data after `node.run()` | `node.trader.add_strategy()` called before `node.build()` | Always call `node.build()` first, then add strategies |
| `no execution client configured for NASDAQ/NYSE or client_id None` | Exec client registered under `INTERACTIVE_BROKERS` venue but orders route by instrument venue (`NASDAQ`, `NYSE`) | Set `routing=RoutingConfig(default=True)` on `InteractiveBrokersExecClientConfig` |
| `TypeError: 'currency' argument was None` in `account.balance_total()` | IB margin accounts have no `base_currency`; calling `balance_total()` with no argument fails | Call `balance_total(currencies[0])` where `currencies = list(account.balances().keys())` |
| `on_timer` callback never fires | `on_timer` does not exist in NautilusTrader ≥1.200. Timer events dispatch to `on_event` | Override `on_event(self, event)` and check `hasattr(event, "name") and event.name == "my_timer"` |
| Error 321 — "API interface is currently in Read-Only mode" | TWS/Gateway has Read-Only API enabled | In TWS/Gateway: Edit → Global Configuration → API → Settings → uncheck **Read-Only API** |
| Error 326 — "Unable to connect as the client id is already in use" | A previous strategy process is still holding the same `ibg_client_id` socket | Kill old Python processes; use distinct `ibg_client_id` per running node (e.g. ORB=20, test=22+) |
| `close_all_positions()` silently does nothing | Nautilus position state corrupted when an order went `CANCELED → ACCEPTED → FILLED` at startup reconciliation | Submit explicit counter-side market order instead: `BUY` to close a `SELL`, `SELL` to close a `BUY` |

### Order Lifecycle — Placing and Closing Positions (Verified Live)

Complete patterns confirmed working against IB paper and live accounts (NautilusTrader 1.221.0).

#### Single Market Order (Entry)

```python
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity

order = self.order_factory.market(
    instrument_id=self.instrument_id,
    order_side=OrderSide.BUY,        # or OrderSide.SELL
    quantity=Quantity.from_int(1),
    time_in_force=TimeInForce.DAY,   # MUST be DAY for market orders on IB — GTC is rejected
)
self.submit_order(order)
```

#### Bracket Order (Entry + SL + TP)

```python
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Price, Quantity

bracket = self.order_factory.bracket(
    instrument_id=self.instrument_id,
    order_side=OrderSide.BUY,
    quantity=qty,
    sl_trigger_price=Price(sl_price, precision=precision),
    tp_price=Price(tp_price, precision=precision),
    entry_order_type=OrderType.MARKET,
    entry_time_in_force=TimeInForce.DAY,  # DAY for market entry
    tp_time_in_force=TimeInForce.GTC,     # GTC OK for limit/stop legs
    sl_time_in_force=TimeInForce.GTC,
)
self.submit_order_list(bracket)   # use submit_order_list, NOT submit_order
```

> **Important:** `submit_order_list` creates proper OTO+OCO linkage. Never submit three independent orders manually — the SL and TP will not be linked.

#### Closing a Position Reliably

**Do not use `close_all_positions()`** when order state may be inconsistent (e.g. after startup reconciliation of pre-existing positions). Submit an explicit counter-side order instead:

```python
# Closing a LONG (BUY entry → SELL to close)
close_order = self.order_factory.market(
    instrument_id=self.instrument_id,
    order_side=OrderSide.SELL,
    quantity=Quantity.from_int(qty),
    time_in_force=TimeInForce.DAY,
)
self.submit_order(close_order)

# Closing a SHORT (SELL entry → BUY to close)
close_order = self.order_factory.market(
    instrument_id=self.instrument_id,
    order_side=OrderSide.BUY,
    quantity=Quantity.from_int(qty),
    time_in_force=TimeInForce.DAY,
)
self.submit_order(close_order)
```

#### Timer-Based Close (60-second hold)

`on_timer` does **not** exist in NautilusTrader ≥1.200. Use `on_event` to handle time alerts:

```python
def on_order_filled(self, event):
    if event.client_order_id == self.entry_order_id:
        self.clock.set_time_alert(
            name="close_position",
            alert_time=self.clock.utc_now() + pd.Timedelta(seconds=60),
        )

def on_event(self, event):
    if hasattr(event, "name") and event.name == "close_position":
        # submit explicit counter-side close order here
        ...
```

#### Account Equity (for position sizing)

IB margin accounts have no `base_currency`. Do **not** call `account.balance_total()` with no argument — it raises `TypeError: 'currency' argument was None`. Use:

```python
accounts = self.cache.accounts()
account = accounts[0]
currencies = list(account.balances().keys())   # e.g. [Currency.GBP]
equity = float(account.balance_total(currencies[0]).as_double())
```

#### Exec Client Routing

Orders are routed by instrument venue (`NASDAQ`, `NYSE`). The IB exec client registers as `INTERACTIVE_BROKERS`. Without `routing=True` it won't handle non-IB-venue instruments:

```python
from nautilus_trader.live.config import RoutingConfig  # NOT from nautilus_trader.config

exec_config = InteractiveBrokersExecClientConfig(
    ...
    routing=RoutingConfig(default=True),   # required for NYSE/NASDAQ instruments
)
```

> **Import note:** `RoutingConfig` lives in `nautilus_trader.live.config`, not `nautilus_trader.config`, despite being listed in both — use `from nautilus_trader.live.config import RoutingConfig` to be safe.

---

### IB Exchange Routing Notes (Equities)

Some tickers route to unexpected primary exchanges. Always verify with IB's contract API:

| Ticker | Expected | IB `primaryExchange` |
|---|---|---|
| WMT | NYSE | **NASDAQ** (tradingClass=NMS) |
| UNH | NYSE | NYSE |
| CAT | NYSE | NYSE |
| TMO | NYSE | NYSE |
| AMAT | NASDAQ | NASDAQ |
| TXN | NASDAQ | NASDAQ |
| INTC | NASDAQ | NASDAQ |

The adapter generates `instrument_id = "{localSymbol}.{primaryExchange}"`. Strategy config `instrument_id`, bar type strings, and `IBContract.primaryExchange` must all agree.
