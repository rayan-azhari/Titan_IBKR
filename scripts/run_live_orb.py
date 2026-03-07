"""run_live_orb.py
-----------------

Live trading runner for the Opening Range Breakout (ORB) Strategy.
Connects to IBKR, loads instruments for the 7 top tickers, and launches the strategy.
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

from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode

# Import our Strategy
from titan.strategies.orb.strategy import ORBConfig, ORBStrategy

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"orb_live_{date_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    return logging.getLogger("titan.nautilus.orb")

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 2))  # ID=2 for ORB 
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR credentials not found. Set IBKR_ACCOUNT_ID in .env.")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("  ORB STRATEGY LIVE — IBKR GATEWAY")
    logger.info("=" * 50)

    # Note: We rely on the pre-existing data gathered from `scripts/download_data.py`
    # for the Daily and 5M history files used in warmup.

    # 1. Configure IBKR Adapter
    inst_config = InteractiveBrokersInstrumentProviderConfig(load_all=True)

    data_config = InteractiveBrokersDataClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        instrument_provider=inst_config,
    )

    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        account_id=ib_account_id,
        instrument_provider=inst_config,
    )

    # 2. Configure Node
    node_config = TradingNodeConfig(
        trader_id="TITAN-ORB",
        data_clients={"IBKR": data_config},
        exec_clients={"IBKR": exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory("IBKR", InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory("IBKR", InteractiveBrokersLiveExecClientFactory)

    # 3. Add Strategy Instances for Top 7 Tickers
    TICKERS = ["UNH", "AMAT", "TXN", "INTC", "CAT", "WMT", "TMO"]

    for ticker in TICKERS:
        # Construct Nautilus IDs
        inst_id = f"{ticker}.USD.IBKR"
        
        # IBKR periodicity strings: "1-DAY", "5-MINUTE"
        bar_5m = f"{inst_id}-5-MINUTE-LAST-INTERNAL" 
        bar_1d = f"{inst_id}-1-DAY-LAST-INTERNAL"
        
        strat_config = ORBConfig(
            instrument_id=inst_id,
            bar_type_5m=bar_5m,
            bar_type_1d=bar_1d,
            config_path="config/orb_live.toml",
            risk_pct=0.01,
            leverage_cap=4.0,
            warmup_bars_1d=60,
            warmup_bars_5m=200,  # Gaussian channel needs 144+ bars to initialize
        )

        strategy = ORBStrategy(strat_config)
        node.trader.add_strategy(strategy)
        logger.info(f"Added ORB Strategy for {ticker}.")

    # 4. Run Node
    print("\n🚀 Starting Trading Node...")

    def stop_node(*args):
        print("\n🛑 Stopping ORB Node...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    try:
        node.build()
        node.run()
    except Exception as e:
        logger.exception("Fatal Runtime Error")
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
