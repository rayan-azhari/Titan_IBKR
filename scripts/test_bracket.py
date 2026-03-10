"""test_bracket.py
-----------------

Bracket order lifecycle test.
Connects to IBKR and places a BUY or SELL bracket order (market entry +
stop loss + take profit) for 1 share of INTC.

Uses a configurable REFERENCE_PRICE instead of live price discovery.
IB paper accounts (DELAYED_FROZEN) do not support quote tick subscriptions
mid-session (error 10089), and EXTERNAL bar subscriptions only stream reliably
on connections established before market open.

Set REFERENCE_PRICE below to INTC's approximate current price before running.
TP is $0.10 away from the reference — tight enough to fill quickly.
SL is $5.00 away — wide safety net, should never fire during the test.

Verifies the full bracket lifecycle:
  connect → bracket submit (3 orders) → entry fill → TP/SL fill → OCO cancel → close

Fallback: if neither TP nor SL fires within FALLBACK_TIMEOUT_SECS (5 min),
cancel all orders and close manually with a counter-side market order.

Usage:
    python scripts/test_bracket.py
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
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

TICKER = "INTC"
EXCHANGE = "NASDAQ"
INSTRUMENT_ID = f"{TICKER}.{EXCHANGE}"

# ── CONFIGURE BEFORE RUNNING ─────────────────────────────────────────────────
# Set this to INTC's approximate current price. Check TWS or recent fills.
# Recent fills (Mar 10 2026): $47.19–$47.21
REFERENCE_PRICE = 47.20

# TP: $0.10 from entry — should fill within a few minutes as stock ticks.
# SL: $5.00 from entry — wide safety net, should not fire during test.
TP_OFFSET = 0.10
SL_OFFSET = 5.00

# How long to wait after connect before placing the bracket.
# Gives the instrument provider and account state time to fully load.
ENTRY_DELAY_SECS = 10

# Fallback close if neither TP nor SL fires within this window.
FALLBACK_TIMEOUT_SECS = 300  # 5 minutes
# ─────────────────────────────────────────────────────────────────────────────


class BracketTestConfig(StrategyConfig, frozen=True):
    instrument_id: str
    order_side: str  # "BUY" or "SELL"
    reference_price: float


class BracketTestStrategy(Strategy):
    """
    Places a bracket order ENTRY_DELAY_SECS after startup using a
    pre-configured reference price. Does not require live market data
    streaming — works on paper accounts with DELAYED_FROZEN data.

    Logs every order and position event.
    """

    def __init__(self, config: BracketTestConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.order_side = OrderSide.BUY if config.order_side == "BUY" else OrderSide.SELL
        self.reference_price = config.reference_price
        self.bracket_submitted = False
        self.entry_filled = False
        self.test_complete = False

    def on_start(self):
        self.log.info(
            f"=== BRACKET TEST START: {self.order_side.name} 1 {self.instrument_id} ==="
        )
        self.log.info(
            f"    Reference price: ${self.reference_price:.2f}  "
            f"TP offset: ${TP_OFFSET:.2f}  SL offset: ${SL_OFFSET:.2f}"
        )

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            available = [str(i) for i in self.cache.instrument_ids()]
            self.log.error(
                f"Instrument {self.instrument_id} not in cache. Available: {available}"
            )
            self.stop()
            return

        self.log.info(
            f"Instrument loaded. Placing bracket in {ENTRY_DELAY_SECS}s..."
        )
        self.clock.set_time_alert(
            name="place_bracket",
            alert_time=self.clock.utc_now() + pd.Timedelta(seconds=ENTRY_DELAY_SECS),
        )

    def on_event(self, event):
        if not hasattr(event, "name"):
            return

        if event.name == "place_bracket":
            if not self.bracket_submitted:
                self._place_bracket(self.reference_price)

        elif event.name == "fallback_close":
            self.log.warning(
                f"Fallback timer fired after {FALLBACK_TIMEOUT_SECS}s — "
                "neither TP nor SL filled. Cancelling orders and closing manually..."
            )
            self.cancel_all_orders(self.instrument_id)
            close_side = OrderSide.SELL if self.order_side == OrderSide.BUY else OrderSide.BUY
            close_order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=close_side,
                quantity=Quantity.from_int(1),
                time_in_force=TimeInForce.DAY,
            )
            self.submit_order(close_order)
            self.log.info(f"Manual close submitted: {close_side.name} 1 {self.instrument_id}")

    def _place_bracket(self, estimated_price: float):
        instrument = self.cache.instrument(self.instrument_id)
        precision = instrument.price_precision if instrument else 2

        if self.order_side == OrderSide.BUY:
            tp_price = round(estimated_price + TP_OFFSET, precision)
            sl_price = round(estimated_price - SL_OFFSET, precision)
        else:
            tp_price = round(estimated_price - TP_OFFSET, precision)
            sl_price = round(estimated_price + SL_OFFSET, precision)

        self.log.info(
            f"Placing {self.order_side.name} bracket: "
            f"entry~{estimated_price:.2f}  TP={tp_price:.2f}  SL={sl_price:.2f}"
        )

        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=self.order_side,
            quantity=Quantity.from_int(1),
            sl_trigger_price=Price(sl_price, precision=precision),
            tp_price=Price(tp_price, precision=precision),
            entry_order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,  # entry TIF — named 'time_in_force' not 'entry_time_in_force'
            tp_post_only=False,  # default True causes IB to reject TP limit orders in brackets
            tp_time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.bracket_submitted = True
        self.submit_order_list(bracket)

        # Fallback safety timer
        self.clock.set_time_alert(
            name="fallback_close",
            alert_time=self.clock.utc_now() + pd.Timedelta(seconds=FALLBACK_TIMEOUT_SECS),
        )
        self.log.info(
            f"Bracket submitted. Fallback close in {FALLBACK_TIMEOUT_SECS}s if needed."
        )

    # ── Order Events ──────────────────────────────────────────────────────────

    def on_order_submitted(self, event):
        self.log.info(f"ORDER SUBMITTED: {event.client_order_id}")

    def on_order_accepted(self, event):
        self.log.info(
            f"ORDER ACCEPTED:  {event.client_order_id}  venue_id={event.venue_order_id}"
        )

    def on_order_rejected(self, event):
        self.log.error(f"ORDER REJECTED:  {event.client_order_id} — {event.reason}")
        self._finish()

    def on_order_filled(self, event):
        self.log.info(
            f"ORDER FILLED:    {event.client_order_id}  "
            f"side={event.order_side.name}  qty={event.last_qty}  px={event.last_px}  "
            f"commission={event.commission}"
        )
        if not self.entry_filled:
            self.entry_filled = True
            self.log.info("Entry filled — waiting for TP or SL to fire...")

    def on_order_canceled(self, event):
        self.log.info(f"ORDER CANCELLED: {event.client_order_id}  (OCO counterpart or manual)")

    def on_order_expired(self, event):
        self.log.warning(f"ORDER EXPIRED:   {event.client_order_id}")

    # ── Position Events ───────────────────────────────────────────────────────

    def on_position_opened(self, event):
        self.log.info(
            f"POSITION OPENED: {event.position_id}  "
            f"side={event.entry.name}  qty={event.quantity}  avg_px={event.avg_px_open:.2f}"
        )

    def on_position_changed(self, event):
        self.log.info(
            f"POSITION CHANGED: {event.position_id}  "
            f"qty={event.quantity}  unrealized_pnl={event.unrealized_pnl}"
        )

    def on_position_closed(self, event):
        self.log.info(
            f"POSITION CLOSED: {event.position_id}  "
            f"realized_pnl={event.realized_pnl}  duration={event.duration_ns // 1_000_000_000}s"
        )
        pnl_str = str(event.realized_pnl).split()[0]
        result = "WIN" if float(pnl_str) > 0 else "LOSS"
        self.log.info(f"=== BRACKET TEST COMPLETE — {result} ===")
        self._finish()

    def _finish(self):
        if not self.test_complete:
            self.test_complete = True
            os.kill(os.getpid(), signal.SIGINT)


def main():
    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = 25  # hardcoded — distinct from ORB (env var) and test_orders (24)
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        print("ERROR: IBKR_ACCOUNT_ID not set in .env")
        sys.exit(1)

    side = random.choice(["BUY", "SELL"])
    print(f"Connecting to IBKR {ib_host}:{ib_port}  (client_id={ib_client_id})")
    print(f"Test: {side} bracket on {INSTRUMENT_ID}")
    print(f"      Reference price: ${REFERENCE_PRICE:.2f}")
    print(f"      TP offset: ${TP_OFFSET:.2f}  SL offset: ${SL_OFFSET:.2f}")
    print(f"      Bracket places in {ENTRY_DELAY_SECS}s after connect")

    contract = IBContract(
        secType="STK",
        symbol=TICKER,
        exchange="SMART",
        primaryExchange=EXCHANGE,
        currency="USD",
    )
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=frozenset([contract]),
    )

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
        routing=RoutingConfig(default=True),
    )

    node_config = TradingNodeConfig(
        trader_id="TEST-BRACKET",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)

    node.build()

    strategy = BracketTestStrategy(
        BracketTestConfig(
            instrument_id=INSTRUMENT_ID,
            order_side=side,
            reference_price=REFERENCE_PRICE,
        )
    )
    node.trader.add_strategy(strategy)

    def stop_node(*args):
        print("\nStopping bracket test node...")
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
