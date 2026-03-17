"""run_live_etf_trend.py — Live runner for the ETF Trend Strategy.

Connects to IBKR Gateway / TWS, loads SPY.ARCA, and starts the daily
trend-following strategy. Submits MOC orders at 15:30 ET.

Paper trading port:  4002  (IBKR Gateway paper)
Live trading port:   4001  (IBKR Gateway live)

Prerequisites:
  1. IBKR Gateway running with the account logged in.
  2. data/SPY_D.parquet present (run scripts/download_data_databento.py first).
  3. config/etf_trend_spy.toml present (run the full pipeline first).
  4. IBKR_ACCOUNT_ID set in .env.

Usage:
    uv run python scripts/run_live_etf_trend.py           # paper (port 4002)
    IBKR_PORT=4001 uv run python scripts/run_live_etf_trend.py  # live

Pre-flight checklist (MANDATORY before live):
  - All 6 pipeline stages passed all quality gates.
  - 60-day paper trade completed and reviewed.
  - config/etf_trend_spy.toml reflects Stage 4 locked params.
  - data/SPY_D.parquet is fresh (rerun download script if >1 day old).
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

from titan.strategies.etf_trend.strategy import ETFTrendConfig, ETFTrendStrategy

# ── Logging ──────────────────────────────────────────────────────────────────

LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"etf_trend_live_{date_str}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("titan.etf_trend")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 3))  # ID=3 for ETF Trend
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env. Aborting.")
        sys.exit(1)

    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )

    logger.info("=" * 60)
    logger.info("  ETF TREND STRATEGY — IBKR GATEWAY")
    logger.info(f"  Host: {ib_host}:{ib_port}  |  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}  |  ClientID: {ib_client_id}")
    logger.info("=" * 60)

    if is_live:
        logger.warning("=" * 60)
        logger.warning("  *** LIVE TRADING MODE — REAL MONEY AT RISK ***")
        logger.warning("  Confirm all quality gates passed before proceeding.")
        logger.warning("=" * 60)

    # ── IBKR Instrument ───────────────────────────────────────────────────────
    # SPY trades on NYSE ARCA. SMART routing resolves to ARCA for AMEX-listed ETFs.
    # primaryExchange=ARCA ensures instrument_id resolves to "SPY.ARCA".
    spy_contract = IBContract(
        secType="STK",
        symbol="SPY",
        exchange="SMART",
        primaryExchange="ARCA",
        currency="USD",
    )
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=frozenset([spy_contract]),
    )

    # Data and exec share the same client_id (single socket to IBKR Gateway).
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
        trader_id="TITAN-ETF-TREND",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    # ── Strategy Instance ─────────────────────────────────────────────────────
    # Instrument ID must match {localSymbol}.{primaryExchange} from IBKR.
    # For SPY this resolves to "SPY.ARCA".
    strat_config = ETFTrendConfig(
        instrument_id="SPY.ARCA",
        bar_type_1d="SPY.ARCA-1-DAY-LAST-EXTERNAL",
        config_path="config/etf_trend_spy.toml",
        warmup_bars=350,
    )
    strategy = ETFTrendStrategy(strat_config)
    node.trader.add_strategy(strategy)
    logger.info("ETF Trend Strategy attached for SPY.ARCA.")

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\nStarting ETF Trend Trading Node...")

    def _stop(*_args) -> None:
        print("\nStopping ETF Trend Node...")
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.run()
    except Exception:
        logger.exception("Fatal runtime error in ETF Trend node")
        sys.exit(1)


if __name__ == "__main__":
    main()
