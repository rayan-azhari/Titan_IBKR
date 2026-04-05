"""run_portfolio.py -- Unified Portfolio Runner.

Runs all strategies in a single TradingNode with shared IBKR connection.
PortfolioRiskManager and PortfolioAllocator manage cross-strategy risk
and allocation automatically.

Paper trading port:  4002 (Gateway) or 7497 (TWS)
Live trading port:   4001 (Gateway) or 7496 (TWS)

Prerequisites:
  1. IBKR Gateway/TWS running with account logged in.
  2. All warmup data files present in data/.
  3. IBKR_ACCOUNT_ID set in .env.

Usage:
    uv run python scripts/run_portfolio.py
    uv run python scripts/run_portfolio.py --strategies all
    uv run python scripts/run_portfolio.py --strategies daily_only
    uv run python scripts/run_portfolio.py --strategies gld_confluence bond_gold

Strategy sets:
    all          -- all available strategies
    daily_only   -- daily-timeframe strategies only (lower bar volume)
    gold_core    -- gold-focused strategies (confluence + macro + bond_gold)
    custom       -- specify individual strategy names
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


# ── Strategy Registry ────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    # Daily strategies (low bar volume)
    "etf_trend_spy": {
        "module": "titan.strategies.etf_trend.strategy",
        "config_cls": "ETFTrendConfig",
        "strategy_cls": "ETFTrendStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="SPY",
                exchange="SMART",
                primaryExchange="ARCA",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "SPY.ARCA",
            "bar_type": "SPY.ARCA-1-DAY-LAST-EXTERNAL",
            "ticker": "SPY",
        },
    },
    "gold_macro": {
        "module": "titan.strategies.gold_macro.strategy",
        "config_cls": "GoldMacroConfig",
        "strategy_cls": "GoldMacroStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="GLD",
                exchange="SMART",
                primaryExchange="ARCA",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "GLD.ARCA",
            "bar_type_d": "GLD.ARCA-1-DAY-LAST-EXTERNAL",
            "ticker": "GLD",
        },
    },
    "bond_gold": {
        "module": "titan.strategies.bond_gold.strategy",
        "config_cls": "BondGoldConfig",
        "strategy_cls": "BondGoldStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="GLD",
                exchange="SMART",
                primaryExchange="ARCA",
                currency="USD",
            ),
            IBContract(
                secType="STK",
                symbol="IEF",
                exchange="SMART",
                primaryExchange="ARCA",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "GLD.ARCA",
            "signal_instrument_id": "IEF.ARCA",
            "bar_type_d": "GLD.ARCA-1-DAY-LAST-EXTERNAL",
            "signal_bar_type_d": "IEF.ARCA-1-DAY-LAST-EXTERNAL",
            "ticker_gld": "GLD",
            "ticker_ief": "IEF",
        },
    },
    "fx_carry_audjpy": {
        "module": "titan.strategies.fx_carry.strategy",
        "config_cls": "FXCarryConfig",
        "strategy_cls": "FXCarryStrategy",
        "contracts": [
            IBContract(secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="JPY"),
        ],
        "config_kwargs": {
            "instrument_id": "AUD/JPY.IDEALPRO",
            "bar_type_d": "AUD/JPY.IDEALPRO-1-DAY-MID-EXTERNAL",
            "ticker": "AUD_JPY",
        },
    },
    # H1 strategies
    "gld_confluence": {
        "module": "titan.strategies.gld_confluence.strategy",
        "config_cls": "GLDConfluenceConfig",
        "strategy_cls": "GLDConfluenceStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="GLD",
                exchange="SMART",
                primaryExchange="ARCA",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "GLD.ARCA",
            "bar_type_h1": "GLD.ARCA-1-HOUR-LAST-EXTERNAL",
            "ticker": "GLD",
        },
    },
    "mr_audjpy": {
        "module": "titan.strategies.mr_audjpy.strategy",
        "config_cls": "MRAUDJPYConfig",
        "strategy_cls": "MRAUDJPYStrategy",
        "contracts": [
            IBContract(secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="JPY"),
        ],
        "config_kwargs": {
            "instrument_id": "AUD/JPY.IDEALPRO",
            "bar_type_h1": "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL",
            "ticker": "AUD_JPY",
        },
    },
    # IC Equity (example -- add more as needed)
    "ic_equity_noc": {
        "module": "titan.strategies.ic_equity_daily.strategy",
        "config_cls": "ICEquityDailyConfig",
        "strategy_cls": "ICEquityDailyStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="NOC",
                exchange="SMART",
                primaryExchange="NYSE",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "NOC.NYSE",
            "bar_type_d": "NOC.NYSE-1-DAY-LAST-EXTERNAL",
            "ticker": "NOC",
        },
    },
}

# Pre-defined strategy sets
STRATEGY_SETS = {
    "all": list(STRATEGY_REGISTRY.keys()),
    "daily_only": [
        "etf_trend_spy",
        "gold_macro",
        "bond_gold",
        "fx_carry_audjpy",
        "ic_equity_noc",
    ],
    "gold_core": ["gld_confluence", "gold_macro", "bond_gold"],
    "h1_only": ["gld_confluence", "mr_audjpy"],
}


# ── Setup ────────────────────────────────────────────────────────────────────


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"portfolio_live_{date_str}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    return logging.getLogger("titan.portfolio")


def _import_class(module_path: str, class_name: str):
    """Dynamically import a class from a module path."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Portfolio Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all"],
        help="Strategy names or set names (all, daily_only, gold_core, h1_only)",
    )
    args = parser.parse_args()

    logger = _setup_logging()

    # Resolve strategy list
    selected = []
    for s in args.strategies:
        if s in STRATEGY_SETS:
            selected.extend(STRATEGY_SETS[s])
        elif s in STRATEGY_REGISTRY:
            selected.append(s)
        else:
            logger.error(f"Unknown strategy: {s}")
            logger.info(f"Available: {list(STRATEGY_REGISTRY.keys())}")
            logger.info(f"Sets: {list(STRATEGY_SETS.keys())}")
            sys.exit(1)
    selected = list(dict.fromkeys(selected))  # deduplicate preserving order

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 1))
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env. Aborting.")
        sys.exit(1)

    is_live = ib_port in (4001, 7496)
    mkt_data_type = (
        IBMarketDataTypeEnum.REALTIME if is_live else IBMarketDataTypeEnum.DELAYED_FROZEN
    )

    logger.info("=" * 60)
    logger.info("  TITAN PORTFOLIO RUNNER")
    logger.info(f"  Host: {ib_host}:{ib_port}")
    logger.info(f"  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}")
    logger.info(f"  Strategies: {len(selected)}")
    for s in selected:
        logger.info(f"    - {s}")
    logger.info("=" * 60)

    # Collect all contracts from selected strategies
    all_contracts = set()
    for name in selected:
        entry = STRATEGY_REGISTRY[name]
        for c in entry["contracts"]:
            all_contracts.add(c)

    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False, load_contracts=frozenset(all_contracts)
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
        trader_id="TITAN-PORTFOLIO",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    # Instantiate and add each strategy
    for name in selected:
        entry = STRATEGY_REGISTRY[name]
        try:
            config_cls = _import_class(entry["module"], entry["config_cls"])
            strategy_cls = _import_class(entry["module"], entry["strategy_cls"])
            config = config_cls(**entry["config_kwargs"])
            strategy = strategy_cls(config)
            node.trader.add_strategy(strategy)
            logger.info(f"  Attached: {name}")
        except Exception as e:
            logger.error(f"  FAILED to attach {name}: {e}")
            continue

    logger.info(f"\n  {len(selected)} strategies attached. Starting node...")

    def _stop(*_args):
        print("\nStopping Portfolio Node...")
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.run()
    except Exception:
        logger.exception("Fatal runtime error in Portfolio node")
        sys.exit(1)


if __name__ == "__main__":
    main()
