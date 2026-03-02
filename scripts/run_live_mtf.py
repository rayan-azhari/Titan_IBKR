"""run_live_mtf.py
-----------------

Live trading runner for the Multi-Timeframe Confluence Strategy.
Connects to OANDA (Practice/Live), loads instruments, and launches the strategy.
"""

import logging
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

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
from titan.strategies.mtf.strategy import MTFConfluenceConfig, MTFConfluenceStrategy

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    """Configure file + console logging for Root Logger."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mtf_live_{date_str}.log"

    # Configure Root Logger to capture EVERYTHING (Strategy, Nautilus, Titan)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    return logging.getLogger("titan.nautilus")


def main():
    """Run the MTF Strategy live."""
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 1))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error(
            "IBKR credentials not found. Set IBKR_ACCOUNT_ID and connection settings in .env."
        )
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("  MTF CONFLUENCE LIVE — IBKR GATEWAY")

    # 1. Download Data
    print("📥 Checking for latest data...")
    try:
        download_script = PROJECT_ROOT / "scripts" / "download_data.py"
        # Download data for instrument (H1, H4, D, W are defaults or handled in config)
        subprocess.check_call([sys.executable, str(download_script), "--instrument", "EUR_USD"])
        print("✅ Data download finished.")
    except Exception as e:
        logger.error(f"Data download warning: {e}")
        print("⚠️ Data download encounterd an issue. Proceeding with existing data if available.")

    # 2. Configure Adapter
    inst_config = InteractiveBrokersInstrumentProviderConfig(load_all=False)

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

    # 3. Load Instruments
    # Note: InteractiveBrokersInstrumentProvider relies on a live client connection.
    # The client is created during node build, so we will handle instrument loads
    # inside the node or configure the provider to fetch on demand.
    print("⏳ Instruments will be loaded dynamically by the IBKR client...")

    # 4. Configure Node with Client Configs
    node_config = TradingNodeConfig(
        trader_id="TITAN-MTF",
        data_clients={"IBKR": data_config},
        exec_clients={"IBKR": exec_config},
    )
    node = TradingNode(config=node_config)

    # 5. Register Clients using Nautilus built-in factories
    node.add_data_client_factory("IBKR", InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory("IBKR", InteractiveBrokersLiveExecClientFactory)

    # 5. Configure Strategy
    strat_config = MTFConfluenceConfig(
        instrument_id="EUR.USD.IBKR",
        bar_types={
            "H1": "EUR.USD.IBKR-1-HOUR-MID-INTERNAL",
            "H4": "EUR.USD.IBKR-4-HOUR-MID-INTERNAL",
            "D": "EUR.USD.IBKR-1-DAY-MID-INTERNAL",
            "W": "EUR.USD.IBKR-1-WEEK-MID-INTERNAL",
        },
        risk_pct=0.01,
        leverage_cap=5.0,
        warmup_bars=1000,
    )

    strategy = MTFConfluenceStrategy(strat_config)
    node.trader.add_strategy(strategy)

    logger.info("Strategy Added. Subscriptions:")
    for tf, bt in strat_config.bar_types.items():
        logger.info(f"  {tf}: {bt}")

    # 6. Run
    print("🚀 Starting Trading Node...")

    def stop_node(*args):
        print("\n🛑 Stopping...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    print(f"DEBUG Node: {[d for d in dir(node) if 'data' in d or 'client' in d]}")
    try:
        print(f"DEBUG Trader: {[d for d in dir(node.trader) if 'data' in d or 'client' in d]}")
    except Exception:
        pass

    try:
        node.build()
        node.run()
    except Exception as e:
        logger.exception("Fatal Runtime Error")
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
