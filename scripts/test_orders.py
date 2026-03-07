"""test_orders.py
----------------

Pipeline verification script.
Connects to IBKR, places a BUY 1 share on each of the 7 ORB tickers
(UNH, AMAT, TXN, INTC, CAT, WMT, TMO), waits 60 seconds, then closes
all positions to confirm the full order lifecycle works.

Usage:
    uv run python scripts/test_orders.py
"""

import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
from nautilus_trader.adapters.interactive_brokers.common import IBContract
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import StrategyConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

# The 7 tickers from the ORB strategy universe
TEST_TICKERS = ["UNH", "AMAT", "TXN", "INTC", "CAT", "WMT", "TMO"]


class TestOrderStrategy(Strategy):
    """
    Places BUY 1 share on each ticker on start, waits 60 seconds,
    then sells all to verify the full execution pipeline.
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.instrument_ids = {
            ticker: InstrumentId.from_str(f"{ticker}.USD.IBKR") for ticker in TEST_TICKERS
        }
        self.stage = 0  # 0=opening, 1=waiting, 2=closing, 3=done
        self.orders_in_flight: set = set()

    def on_start(self):
        self.log.info("=== PIPELINE TEST STARTED ===")
        self.log.info(f"Testing {len(TEST_TICKERS)} tickers: {TEST_TICKERS}")

        # Verify all instruments are in cache
        missing = []
        for ticker, inst_id in self.instrument_ids.items():
            if not self.cache.instrument(inst_id):
                missing.append(ticker)

        if missing:
            available = [str(i) for i in self.cache.instrument_ids()]
            self.log.error(
                f"Instruments NOT in cache: {missing}. "
                f"Available ({len(available)}): {available[:20]}..."
            )
            return

        self.log.info("All instruments confirmed in cache. Submitting BUY orders...")

        for ticker, inst_id in self.instrument_ids.items():
            order = self.order_factory.market(
                instrument_id=inst_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(1),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.orders_in_flight.add(order.client_order_id)
            self.log.info(f"  BUY 1 {ticker}  (id={order.client_order_id})")

    def on_order_filled(self, event):
        self.log.info(f"FILL: {event.client_order_id} @ {event.last_px}")
        self.orders_in_flight.discard(event.client_order_id)

        if self.stage == 0 and not self.orders_in_flight:
            self.stage = 1
            self.log.info(
                f"All {len(TEST_TICKERS)} opening orders filled! Setting 60-second close timer..."
            )
            self.clock.set_time_alert(
                name="close_positions",
                alert_time=self.clock.utc_now() + pd.Timedelta(seconds=60),
            )

        elif self.stage == 2 and not self.orders_in_flight:
            self.log.info(
                f"=== ALL {len(TEST_TICKERS)} CLOSING ORDERS FILLED. PIPELINE TEST COMPLETE! ==="
            )
            self.stage = 3
            os.kill(os.getpid(), signal.SIGINT)

    def on_timer(self, event):
        if event.name == "close_positions":
            self.stage = 2
            self.log.info("60 seconds elapsed. Submitting SELL orders to close all...")

            for ticker, inst_id in self.instrument_ids.items():
                order = self.order_factory.market(
                    instrument_id=inst_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(1),
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
                self.orders_in_flight.add(order.client_order_id)
                self.log.info(f"  SELL 1 {ticker} (close)")


def main():
    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 10))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    print(f"Connecting to IBKR on {ib_host}:{ib_port}  Account: {ib_account_id}")
    print(f"Test tickers: {TEST_TICKERS}")

    # Tell IBKR provider to load specific stock contracts on startup.
    contracts = [
        IBContract(secType="STK", symbol=t, exchange="SMART", currency="USD") for t in TEST_TICKERS
    ]
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=frozenset(contracts),
    )

    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        account_id=ib_account_id,
        instrument_provider=inst_config,
    )
    data_config = InteractiveBrokersDataClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id + 1,
        instrument_provider=inst_config,
    )

    node_config = TradingNodeConfig(
        trader_id="TEST-NODE",
        exec_clients={"IBKR": exec_config},
        data_clients={"IBKR": data_config},
    )
    node = TradingNode(config=node_config)
    node.add_exec_client_factory("IBKR", InteractiveBrokersLiveExecClientFactory)
    node.add_data_client_factory("IBKR", InteractiveBrokersLiveDataClientFactory)

    # Rely on load_all=True to populate instruments from IBKR during node startup.
    node.trader.add_strategy(TestOrderStrategy(StrategyConfig()))

    def stop_node(*args):
        print("\nStopping Test Node...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)

    node.build()
    node.run()


if __name__ == "__main__":
    main()
