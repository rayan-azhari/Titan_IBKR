"""run_live_gem.py -- Live runner for GEM Dual Momentum.

Production cell (since 2026-05-16): **J5 P_hl60_vt05**
    vol_estimator_kind="ewma", vol_estimator_halflife=60, ann_vol_target=0.05.
    Replaces J4 A1_ewma_hl40 via L52 hybrid-framework re-audit.
    See `directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md` +
    `docs/strategies/gem-dual-momentum.md` for the full lineage.

Cross-asset momentum strategy: monthly SPY / EFA / IEF rotation with
multi-speed blend (3,6,12), continuous vol-target overlay, and defensive
switch into IEF when risk assets underperform.

Production cell expects max_leverage=2.0 (rarely binds at the J5
vol_target=0.05 operating point — L57). Two execution modes:

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

    universe = os.getenv("UNIVERSE", "us").lower()

    logger.info("=" * 60)
    logger.info("  GEM DUAL MOMENTUM (J5 P_hl60_vt05 production cell) -- IBKR GATEWAY")
    logger.info(f"  Host: {ib_host}:{ib_port}  |  Mode: {'LIVE' if is_live else 'PAPER'}")
    logger.info(f"  Account: {ib_account_id}  |  ClientID: {ib_client_id}")
    logger.info(f"  Universe: {universe.upper()}")
    logger.info("=" * 60)

    if universe == "uk":
        # UK UCITS substitutes (LSEETF). Required because UK retail
        # paper accounts are restricted from trading US-listed ETFs
        # under PRIIPs/KID rules. Mapping:
        #   SPY -> CSPX (iShares Core S&P 500 UCITS, USD, LSE)
        #   EFA -> IWDA (iShares Core MSCI World UCITS, USD, LSE)
        #          NOTE: IWDA includes ~65% US; not a pure EFA substitute.
        #          The audit on this UK universe shows the strategy is mostly
        #          CSPX-or-cash; the EFA leg rarely wins under this mapping.
        #   IEF -> IDTM (iShares $ Treasury Bond 7-10y UCITS, USD, LSEETF).
        #          yfinance ticker is IBTM.L; IBKR's broker symbol is IDTM.
        #          The IBKR ticker "IBTM" resolves to an unrelated US fund.
        spy_contract = IBContract(secType="STK", symbol="CSPX", exchange="LSEETF", currency="USD")
        efa_contract = IBContract(secType="STK", symbol="IWDA", exchange="LSEETF", currency="USD")
        # NB: yfinance ticker is "IBTM.L" but IBKR lists the same fund as
        # "IDTM" on LSEETF (ConId=68489992, "ISHARES USD TREASURY 7-10Y").
        # Symbol "IBTM" on IBKR resolves to an unrelated US fund.
        ief_contract = IBContract(secType="STK", symbol="IDTM", exchange="LSEETF", currency="USD")
    else:
        # US-listed primary universe. Used for non-UK paper / live accounts.
        spy_contract = IBContract(
            secType="STK", symbol="SPY", exchange="SMART", primaryExchange="ARCA", currency="USD"
        )
        efa_contract = IBContract(
            secType="STK", symbol="EFA", exchange="SMART", primaryExchange="ARCA", currency="USD"
        )
        # IEF's primary exchange is NASDAQ (not ARCA).
        ief_contract = IBContract(
            secType="STK", symbol="IEF", exchange="SMART", primaryExchange="NASDAQ", currency="USD"
        )

    contracts = [spy_contract, efa_contract, ief_contract]

    # Optional regime instruments (VIX index + HYG ETF). HYG is US-listed
    # so it WILL hit PRIIPs on UK retail -- but the strategy only reads its
    # bars, never trades it, so the subscription succeeds and the rejection
    # only matters if we try to trade. Skipped entirely under UK universe
    # to keep the subscription clean.
    vix_contract = IBContract(secType="IND", symbol="VIX", exchange="CBOE", currency="USD")
    contracts.append(vix_contract)
    if universe != "uk":
        hyg_contract = IBContract(
            secType="STK", symbol="HYG", exchange="SMART", primaryExchange="ARCA", currency="USD"
        )
        contracts.append(hyg_contract)

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

    # Build instrument ids + bar types per universe. Strategy roles stay
    # logical (spy_*, efa_*, ief_*) — only the underlying broker symbol
    # changes. NT instrument-id suffixes:
    #   - US: ".ARCA" for SPY/EFA/HYG, ".NASDAQ" for IEF (primary listing).
    #   - UK: ".LSEETF" for CSPX/IWDA/IBTM (LSE-listed UCITS USD class).
    #   - VIX is the same index in both ("^VIX.CBOE").
    #   - HYG (US-only) only used as a regime signal; under UK we point
    #     the role at a benign placeholder and the strategy ignores it
    #     because hyg_contract is not subscribed.
    if universe == "uk":
        spy_iid, efa_iid, ief_iid = "CSPX.LSEETF", "IWDA.LSEETF", "IDTM.LSEETF"
        hyg_iid: str | None = None  # HYG not subscribed under UK (PRIIPs-blocked)
        # Warmup parquet is data/IDTM_D.parquet (a copy of IBTM_D — yfinance
        # ticker IBTM.L == IBKR IDTM, same iShares USD Treasury 7-10y fund).
        ticker_spy, ticker_efa, ticker_ief = "CSPX", "IWDA", "IDTM"
    else:
        spy_iid, efa_iid, ief_iid = "SPY.ARCA", "EFA.ARCA", "IEF.NASDAQ"
        hyg_iid = "HYG.ARCA"
        ticker_spy, ticker_efa, ticker_ief = "SPY", "EFA", "IEF"

    strat_config = GemStrategyConfig(
        spy_instrument_id=spy_iid,
        efa_instrument_id=efa_iid,
        ief_instrument_id=ief_iid,
        vix_instrument_id="^VIX.CBOE",
        hyg_instrument_id=hyg_iid,
        spy_bar_type_d=f"{spy_iid}-1-DAY-LAST-EXTERNAL",
        efa_bar_type_d=f"{efa_iid}-1-DAY-LAST-EXTERNAL",
        ief_bar_type_d=f"{ief_iid}-1-DAY-LAST-EXTERNAL",
        vix_bar_type_d="^VIX.CBOE-1-DAY-LAST-EXTERNAL",
        hyg_bar_type_d=(f"{hyg_iid}-1-DAY-LAST-EXTERNAL" if hyg_iid else None),
        # Physical parquet filenames for warmup (data/{ticker}_D.parquet).
        ticker_spy=ticker_spy,
        ticker_efa=ticker_efa,
        ticker_ief=ticker_ief,
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
