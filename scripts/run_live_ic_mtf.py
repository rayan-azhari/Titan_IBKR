"""run_live_ic_mtf.py
--------------------

Live trading runner for the IC MTF Strategy.
Connects to IBKR (Paper/Live), downloads fresh H1/H4/D/W candles, then
launches one ICMTFStrategy instance per configured instrument.

Validated pipeline (2026-03-19): Phases 1–5 all pass for 6 pairs.
  OOS Sharpe  : EUR/USD 7.71 | GBP/USD 8.28 | USD/JPY 7.35
                AUD/USD 6.84 | AUD/JPY 7.33 | USD/CHF 7.34
  Robustness  : MC, top-N, 3x slippage, WFO folds — all PASS

Config       : config/ic_mtf.toml  (per-instrument thresholds)
Signals      : accel_rsi14 + accel_stoch_k across W, D, H4, H1
Entry        : composite_z crosses ±threshold
Exit         : composite_z crosses zero
Sizing       : 1% risk / (1.5 × ATR14_H1)

Client IDs   : 4 reserved for IC MTF (ORB=2, MTF=3)
"""

import logging
import signal
import subprocess
import sys
import tomllib
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

from titan.strategies.ic_mtf.strategy import ICMTFConfig, ICMTFStrategy

# ---------------------------------------------------------------------------
# Instruments to trade (subset by commenting out)
# ---------------------------------------------------------------------------
INSTRUMENTS = [
    # (symbol, currency, exchange, instrument_id, TF thresholds from ic_mtf.toml)
    ("EUR", "USD", "IDEALPRO", "EUR/USD.IDEALPRO", "EUR_USD"),
    ("GBP", "USD", "IDEALPRO", "GBP/USD.IDEALPRO", "GBP_USD"),
    ("USD", "JPY", "IDEALPRO", "USD/JPY.IDEALPRO", "USD_JPY"),
    ("AUD", "USD", "IDEALPRO", "AUD/USD.IDEALPRO", "AUD_USD"),
    ("AUD", "JPY", "IDEALPRO", "AUD/JPY.IDEALPRO", "AUD_JPY"),
    ("USD", "CHF", "IDEALPRO", "USD/CHF.IDEALPRO", "USD_CHF"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ic_mtf_live_{date_str}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("titan.nautilus.ic_mtf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger = _setup_logging()

    ib_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ib_port = int(os.getenv("IBKR_PORT", 4002))
    ib_client_id = int(os.getenv("IBKR_CLIENT_ID", 4))  # ID=4 for IC MTF
    ib_account_id = os.getenv("IBKR_ACCOUNT_ID")

    if not ib_account_id:
        logger.error("IBKR_ACCOUNT_ID not set in .env")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  IC MTF LIVE — IBKR GATEWAY")
    logger.info(f"  Instruments: {len(INSTRUMENTS)}")
    logger.info("=" * 60)

    # 1. Download latest data for all pairs
    print("Downloading latest warmup data...")
    try:
        dl = PROJECT_ROOT / "scripts" / "download_data_databento.py"
        subprocess.check_call([sys.executable, str(dl)])
        print("Download complete.")
    except Exception as e:
        logger.warning(f"Download issue: {e} — proceeding with existing data.")

    # 2. Load ic_mtf.toml for per-instrument config
    cfg_path = PROJECT_ROOT / "config" / "ic_mtf.toml"
    with open(cfg_path, "rb") as f:
        ic_cfg = tomllib.load(f)

    # 3. Build IBContracts and instrument provider
    contracts = frozenset(
        IBContract(secType="CASH", symbol=sym, currency=ccy, exchange=exch)
        for sym, ccy, exch, _, _ in INSTRUMENTS
    )
    inst_config = InteractiveBrokersInstrumentProviderConfig(
        load_all=False,
        load_contracts=contracts,
    )

    # 4. Data / exec client config
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
        connection_timeout=60,
    )
    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=ib_client_id,
        account_id=ib_account_id,
        instrument_provider=inst_config,
        routing=RoutingConfig(default=True),
        connection_timeout=60,
    )

    # 5. Build trading node
    node_config = TradingNodeConfig(
        trader_id="TITAN-IC-MTF",
        data_clients={IB: data_config},
        exec_clients={IB: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    # 6. Add one strategy per instrument
    for sym, ccy, exch, inst_id, cfg_key in INSTRUMENTS:
        pair_cfg = ic_cfg.get(cfg_key, {})
        strat_config = ICMTFConfig(
            instrument_id=inst_id,
            bar_types={
                "H1": f"{inst_id}-1-HOUR-MID-EXTERNAL",
                "H4": f"{inst_id}-4-HOUR-MID-EXTERNAL",
                "D": f"{inst_id}-1-DAY-MID-EXTERNAL",
                "W": f"{inst_id}-1-WEEK-MID-EXTERNAL",
            },
            threshold=float(pair_cfg.get("threshold", 0.75)),
            risk_pct=float(pair_cfg.get("risk_pct", 0.01)),
            stop_atr_mult=float(pair_cfg.get("stop_atr_mult", 1.5)),
            leverage_cap=float(pair_cfg.get("leverage_cap", 20.0)),
            warmup_bars=int(pair_cfg.get("warmup_bars", 1000)),
        )
        strategy = ICMTFStrategy(strat_config)
        node.trader.add_strategy(strategy)
        logger.info(
            f"Added ICMTFStrategy for {inst_id}"
            f" | threshold=±{strat_config.threshold}z"
        )

    # 7. Run
    print("\nStarting IC MTF Trading Node...")

    def stop_node(*args) -> None:
        print("\nStopping IC MTF Node...")
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
