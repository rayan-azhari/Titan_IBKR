"""run_live_mr_fx.py -- Live runner for MR FX (VWAP Mean Reversion) Strategy.

Intraday mean reversion on EUR/USD using session-anchored VWAP deviation
with tiered grid entries. London core session (07:00-12:00 UTC).

Paper trading port:  4002  (IBKR Gateway paper)
Live trading port:   4001  (IBKR Gateway live)

Prerequisites:
  1. IBKR Gateway running with the account logged in.
  2. data/EUR_USD_M5.parquet present (warmup data).
  3. config/mr_fx_eurusd.toml present.
  4. IBKR_ACCOUNT_ID set in .env.

Usage:
    uv run python scripts/run_live_mr_fx.py
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

from titan.strategies.mr_fx.strategy import MRFXConfig, MRFXStrategy

LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mr_fx_live_{date_str}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    return logging.getLogger("titan.mr_fx")


def main() -> None:
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 17))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env. Aborting.")
        sys.exit(1)

    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )

    logger.info("=" * 60)
    logger.info("  MR FX (VWAP MEAN REVERSION) -- IBKR GATEWAY")
    logger.info(f"  Host: {ib_host}:{ib_port}  |  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}  |  ClientID: {ib_client_id}")
    logger.info("=" * 60)

    eurusd_contract = IBContract(secType="CASH", symbol="EUR", exchange="IDEALPRO", currency="USD")
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False, load_contracts=frozenset([eurusd_contract])
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
        trader_id="TITAN-MR-FX",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    strat_config = MRFXConfig(
        instrument_id="EUR/USD.IDEALPRO",
        bar_type_m5="EUR/USD.IDEALPRO-5-MINUTE-MID-EXTERNAL",
    )
    strategy = MRFXStrategy(strat_config)
    node.trader.add_strategy(strategy)
    logger.info("MR FX Strategy attached for EUR/USD.IDEALPRO (M5).")

    def _stop(*_args):
        print("\nStopping MR FX Node...")
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.run()
    except Exception:
        logger.exception("Fatal runtime error in MR FX node")
        sys.exit(1)


if __name__ == "__main__":
    main()
