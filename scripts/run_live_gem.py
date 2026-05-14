"""run_live_gem.py -- Live runner for GEM Dual Momentum (C12 production cell).

Cross-asset momentum strategy: monthly SPY / EFA / IEF rotation with
multi-speed blend (3,6,12), continuous vol-target overlay, and defensive
switch into IEF when risk assets underperform.

Selected production cell C12 expects max_leverage=2.0. Two execution modes:

  * ``etf``  -- trade SPY/EFA/IEF as ETFs (caps effective leverage at 1.0).
               Use this for paper trading and the first 30+ days live.

  * ``mes``  -- trade MES futures for the SPY leg (full 2x leverage),
               ETFs for EFA/IEF. Requires CME futures permissions and
               an active front-month contract id in the TOML config.

Paper trading port: 4002 (IBKR Gateway paper)
Live trading port:  4001 (IBKR Gateway live)

Prerequisites:
  1. IBKR Gateway running with the account logged in.
  2. data/{SPY,EFA,IEF}_D.parquet present for warmup.
  3. data/VIX_D.parquet + data/HYG_D.parquet for optional stress signals.
  4. IBKR_ACCOUNT_ID set in .env.

Usage:
    uv run python scripts/run_live_gem.py
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from nautilus_trader.adapters.interactive_brokers.common import IB, IBContract  # noqa: E402
from nautilus_trader.adapters.interactive_brokers.config import (  # noqa: E402
    IBMarketDataTypeEnum,
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (  # noqa: E402
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import TradingNodeConfig  # noqa: E402
from nautilus_trader.live.config import RoutingConfig  # noqa: E402
from nautilus_trader.live.node import TradingNode  # noqa: E402

LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"gem_live_{date_str}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    return logging.getLogger("titan.gem")


def main() -> None:
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID_GEM", os.getenv("IBKR_CLIENT_ID", 21)))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env. Aborting.")
        sys.exit(1)

    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )

    logger.info("=" * 60)
    logger.info("  GEM DUAL MOMENTUM (C12 production cell) -- IBKR GATEWAY")
    logger.info(f"  Host: {ib_host}:{ib_port}  |  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}  |  ClientID: {ib_client_id}")
    logger.info("=" * 60)

    # Required instruments: SPY, EFA, IEF (all US ETFs on ARCA).
    spy_contract = IBContract(
        secType="STK", symbol="SPY", exchange="SMART", primaryExchange="ARCA", currency="USD"
    )
    efa_contract = IBContract(
        secType="STK", symbol="EFA", exchange="SMART", primaryExchange="ARCA", currency="USD"
    )
    ief_contract = IBContract(
        secType="STK", symbol="IEF", exchange="SMART", primaryExchange="ARCA", currency="USD"
    )
    contracts = [spy_contract, efa_contract, ief_contract]

    # Optional regime instruments.
    vix_contract = IBContract(secType="IND", symbol="VIX", exchange="CBOE", currency="USD")
    hyg_contract = IBContract(
        secType="STK", symbol="HYG", exchange="SMART", primaryExchange="ARCA", currency="USD"
    )
    contracts.extend([vix_contract, hyg_contract])

    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False, load_contracts=frozenset(contracts)
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
        trader_id="TITAN-GEM",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    try:
        from titan.strategies.gem.config import GemStrategyConfig
        from titan.strategies.gem.strategy import GemStrategy
    except ImportError as e:
        logger.error(f"GEM strategy import failed: {e}")
        sys.exit(1)

    # Load runtime params from TOML.
    import tomllib

    cfg_path = PROJECT_ROOT / "config" / "gem_voltarget_lev2.toml"
    with cfg_path.open("rb") as fh:
        toml_cfg = tomllib.load(fh)

    strat_config = GemStrategyConfig(
        # Instrument ids + bar types
        spy_instrument_id="SPY.ARCA",
        efa_instrument_id="EFA.ARCA",
        ief_instrument_id="IEF.ARCA",
        vix_instrument_id="VIX.CBOE",
        hyg_instrument_id="HYG.ARCA",
        spy_bar_type_d="SPY.ARCA-1-DAY-LAST-EXTERNAL",
        efa_bar_type_d="EFA.ARCA-1-DAY-LAST-EXTERNAL",
        ief_bar_type_d="IEF.ARCA-1-DAY-LAST-EXTERNAL",
        vix_bar_type_d="VIX.CBOE-1-DAY-LAST-EXTERNAL",
        hyg_bar_type_d="HYG.ARCA-1-DAY-LAST-EXTERNAL",
        # Spread remaining params from TOML
        **{k: v for k, v in toml_cfg.items() if k in GemStrategyConfig.__annotations__},
    )

    strategy = GemStrategy(strat_config)
    node.trader.add_strategy(strategy)
    logger.info(
        f"GEM Strategy attached "
        f"(execution_mode={strat_config.execution_mode}, "
        f"max_leverage={strat_config.max_leverage})."
    )

    def _stop(*_args):
        print("\nStopping GEM Node...")
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.run()
    except Exception:
        logger.exception("Fatal runtime error in GEM node")
        sys.exit(1)


if __name__ == "__main__":
    main()
