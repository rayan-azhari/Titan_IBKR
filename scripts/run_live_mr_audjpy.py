"""run_live_mr_audjpy.py -- Live runner for AUD/JPY MR + Confluence Regime Filter.

Intraday mean reversion on AUD/JPY using VWAP deviation grid with
multi-scale confluence disagreement as regime filter.

Paper trading port:  4002  (IBKR Gateway paper)
Live trading port:   4001  (IBKR Gateway live)

Prerequisites:
  1. IBKR Gateway running with the account logged in.
  2. data/AUD_JPY_H1.parquet present (warmup data).
  3. IBKR_ACCOUNT_ID set in .env.

Note: This runner requires the AUD/JPY MR strategy to be deployed
in titan/strategies/mr_audjpy/. Until then, it will fail to import.

Usage:
    uv run python scripts/run_live_mr_audjpy.py
"""

import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

from dotenv import load_dotenv

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
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.config import RoutingConfig
from nautilus_trader.live.node import TradingNode

LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mr_audjpy_live_{date_str}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    return logging.getLogger("titan.mr_audjpy")


def main() -> None:
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 15))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env. Aborting.")
        sys.exit(1)

    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )

    logger.info("=" * 60)
    logger.info("  AUD/JPY MR + CONFLUENCE REGIME -- IBKR GATEWAY")
    logger.info(f"  Host: {ib_host}:{ib_port}  |  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}  |  ClientID: {ib_client_id}")
    logger.info("=" * 60)

    audjpy_contract = IBContract(secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="JPY")
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False, load_contracts=frozenset([audjpy_contract])
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
        trader_id="TITAN-MR-AUDJPY",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    try:
        from titan.strategies.mr_audjpy.strategy import (
            MRAUDJPYConfig,
            MRAUDJPYStrategy,
        )
    except ImportError:
        logger.error(
            "AUD/JPY MR strategy not yet deployed to titan/strategies/mr_audjpy/. "
            "See directives/Strategy Deployment Guide.md for deployment steps."
        )
        sys.exit(1)

    strat_config = MRAUDJPYConfig(
        instrument_id="AUD/JPY.IDEALPRO",
        bar_type_h1="AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL",
        ticker="AUD_JPY",
    )
    strategy = MRAUDJPYStrategy(strat_config)
    node.trader.add_strategy(strategy)
    logger.info("AUD/JPY MR + Confluence Strategy attached for AUD/JPY.IDEALPRO (H1).")

    def _stop(*_args):
        print("\nStopping AUD/JPY MR Node...")
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.run()
    except Exception:
        logger.exception("Fatal runtime error in AUD/JPY MR node")
        sys.exit(1)


if __name__ == "__main__":
    main()
