"""test_orders.py
----------------

Quick order lifecycle test.
Connects to IBKR, places a random BUY or SELL market order for 1 share of INTC,
holds for 60 seconds, then closes the position.

Verifies the full execution pipeline: connect: order: fill: close: stop.

Usage:
    python scripts/test_orders.py
"""

import os
import random
import signal
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from nautilus_trader.adapters.interactive_brokers.common import IB, IBContract
from nautilus_trader.adapters.interactive_brokers.config import (
    IBMarketDataTypeEnum,
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import StrategyConfig, TradingNodeConfig
from nautilus_trader.live.config import RoutingConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

# Single cheap instrument for testing
TICKER = "INTC"
EXCHANGE = "NASDAQ"
INSTRUMENT_ID = f"{TICKER}.{EXCHANGE}"


class OrderTestConfig(StrategyConfig, frozen=True):
    instrument_id: str
    order_side: str  # "BUY" or "SELL"


class OrderTestStrategy(Strategy):
    """
    Places 1 market order on start, waits 60s via timer, then closes and stops.
    """

    def __init__(self, config: OrderTestConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.order_side = OrderSide.BUY if config.order_side == "BUY" else OrderSide.SELL
        self.entry_order_id = None

    def on_start(self):
        self.log.info(f"=== ORDER TEST START: {self.order_side.name} 1 {self.instrument_id} ===")

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            available = [str(i) for i in self.cache.instrument_ids()]
            self.log.error(f"Instrument {self.instrument_id} not in cache. Available: {available}")
            self.stop()
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=self.order_side,
            quantity=Quantity.from_int(1),
            time_in_force=TimeInForce.DAY,  # MUST be DAY for market orders on IB
        )
        self.entry_order_id = order.client_order_id
        self.submit_order(order)
        self.log.info(f"Submitted {self.order_side.name} order (id={order.client_order_id})")

    def on_order_filled(self, event):
        self.log.info(
            f"FILLED: {event.client_order_id} | side={event.order_side.name} "
            f"qty={event.last_qty} @ {event.last_px}"
        )
        if event.client_order_id == self.entry_order_id:
            self.log.info("Entry filled. Setting 60-second close timer...")
            self.clock.set_time_alert(
                name="close_position",
                alert_time=self.clock.utc_now() + pd.Timedelta(seconds=60),
            )

    def on_order_rejected(self, event):
        self.log.error(f"ORDER REJECTED: {event.client_order_id} — {event.reason}")
        self.stop()

    def on_event(self, event):
        if hasattr(event, "name") and event.name == "close_position":
            self.log.info("60 seconds elapsed — submitting close order...")
            # Submit explicit counter-side order — more reliable than close_all_positions
            # when order state machine went through an unexpected path (e.g. CANCELED->FILLED)
            close_side = OrderSide.BUY if self.order_side == OrderSide.SELL else OrderSide.SELL
            close_order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=close_side,
                quantity=Quantity.from_int(1),
                time_in_force=TimeInForce.DAY,
            )
            self.submit_order(close_order)
            self.log.info(f"Close order submitted: {close_side.name} 1 {self.instrument_id}")

    def on_position_closed(self, event):
        self.log.info(f"POSITION CLOSED: realized_pnl={event.realized_pnl} | ==> TEST COMPLETE <==")
        os.kill(os.getpid(), signal.SIGINT)


def main():
    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = 24  # hardcoded to avoid conflict with ORB (uses IBKR_CLIENT_ID from .env)
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        print("ERROR: IBKR_ACCOUNT_ID not set in .env")
        sys.exit(1)

    side = random.choice(["BUY", "SELL"])
    print(f"Connecting to IBKR {ib_host}:{ib_port}  (client_id={ib_client_id})")
    print(f"Test: {side} 1 share of {INSTRUMENT_ID} — will close after 60s")

    contract = IBContract(
        secType="STK",
        symbol=TICKER,
        exchange="SMART",
        primaryExchange=EXCHANGE,  # critical: determines instrument_id = TICKER.EXCHANGE
        currency="USD",
    )
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=frozenset([contract]),
    )

    # Data and exec share same client_id — adapter reuses a single socket
    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )
    data_config = InteractiveBrokersDataClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        market_data_type=mkt_data_type,
        instrument_provider=inst_config,
    )
    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        account_id=ib_account_id,
        instrument_provider=inst_config,
        routing=RoutingConfig(default=True),  # route orders for any venue through this IB client
    )

    # Use canonical IB key — string literals like "IBKR" cause factory lookup failures
    node_config = TradingNodeConfig(
        trader_id="TEST-ORDERS",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)

    # build() BEFORE add_strategy — instruments and engines must be ready first
    node.build()

    strategy = OrderTestStrategy(OrderTestConfig(instrument_id=INSTRUMENT_ID, order_side=side))
    node.trader.add_strategy(strategy)

    def stop_node(*args):
        print("\nStopping test node...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    try:
        node.run()
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
