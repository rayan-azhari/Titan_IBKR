"""run_live_mtf.py
-----------------

Live trading runner for the Multi-Timeframe Confluence Strategy on EUR/USD.
Connects to IBKR (Paper/Live), downloads fresh H1/H4/D/W candles via
scripts/download_data_mtf.py, then launches the strategy.

Config  : config/mtf.toml  (Round 3 validated — OOS Sharpe 2.936)
TFs     : H1, H4, D, W
MA type : SMA
Stop    : 2.5× ATR(14, H1) hard STOP_MARKET

Last updated: 2026-03-15
"""

import logging
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

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

from titan.strategies.mtf.strategy import MTFConfluenceConfig, MTFConfluenceStrategy

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mtf_live_{date_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    return logging.getLogger("titan.nautilus.mtf")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 3))  # ID=3 for MTF (ORB uses 2)
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR credentials not found. Set IBKR_ACCOUNT_ID in .env.")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("  MTF CONFLUENCE LIVE (M5) — IBKR GATEWAY")
    logger.info("=" * 50)

    # 1. Download latest EUR/USD warmup data (H1, H4, D, W)
    print("Checking for latest data...")
    try:
        download_script = PROJECT_ROOT / "scripts" / "download_data_mtf.py"
        subprocess.check_call([sys.executable, str(download_script)])
        print("Data download finished.")
    except Exception as e:
        logger.warning(f"Data download issue: {e}. Proceeding with existing data.")

    # 2. Configure EUR/USD Forex contract
    # CASH secType + IDEALPRO exchange → instrument_id = EUR/USD.IDEALPRO
    eur_usd_contract = IBContract(
        secType="CASH",
        symbol="EUR",
        currency="USD",
        exchange="IDEALPRO",
    )
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=frozenset([eur_usd_contract]),
    )

    # Paper: port 4002/7497 → DELAYED_FROZEN. Live: 4001/7496 → REALTIME.
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

    # 3. Configure Node
    node_config = TradingNodeConfig(
        trader_id="TITAN-MTF",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)

    # 4. Build node first — instrument provider must be ready before strategies attach
    node.build()

    # 5. Add Strategy — must be after node.build()
    # instrument_id matches IBContract CASH EUR/USD on IDEALPRO
    INSTRUMENT_ID = "EUR/USD.IDEALPRO"

    # config/mtf.toml: H1+H4+D+W, SMA, threshold=0.10 — validated in Round 3 backtest
    # (OOS Sharpe 2.73 with full friction: slippage, carry, ATR stop, next-bar fills)
    strat_config = MTFConfluenceConfig(
        instrument_id=INSTRUMENT_ID,
        bar_types={
            "H1": f"{INSTRUMENT_ID}-1-HOUR-MID-EXTERNAL",
            "H4": f"{INSTRUMENT_ID}-4-HOUR-MID-EXTERNAL",
            "D":  f"{INSTRUMENT_ID}-1-DAY-MID-EXTERNAL",
            "W":  f"{INSTRUMENT_ID}-1-WEEK-MID-EXTERNAL",
        },
        config_path="config/mtf.toml",
        risk_pct=0.01,
        leverage_cap=5.0,
        warmup_bars=1000,
    )

    strategy = MTFConfluenceStrategy(strat_config)
    node.trader.add_strategy(strategy)
    logger.info("Added MTF Strategy for EUR/USD.")

    for tf, bt in strat_config.bar_types.items():
        logger.info(f"  {tf}: {bt}")

    # 6. Run Node
    print("\nStarting Trading Node Loop...")

    def stop_node(*args):
        print("\nStopping MTF Node...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    try:
        node.run()
    except Exception as e:
        logger.exception("Fatal Runtime Error")
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
