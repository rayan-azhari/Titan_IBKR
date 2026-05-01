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
    uv run python scripts/run_portfolio.py --strategies bond_gold mr_audjpy

Strategy sets:
    all          -- all available strategies
    daily_only   -- daily-timeframe strategies only (lower bar volume)
    gold_core    -- gold-focused strategies (gold_macro + bond_gold)
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
DATA_DIR = PROJECT_ROOT / "data"


# ── Warmup file map (per strategy → list of required parquet basenames) ──
# Used by `_check_data_freshness` to warn if cron-refreshed parquets have
# gone stale. Add an entry for any new strategy that reads from data/ at
# startup. A strategy without an entry here is silently skipped (no warn).

_STRATEGY_WARMUP_FILES: dict[str, list[str]] = {
    "mr_audjpy": [
        "AUD_JPY_H1.parquet",
        "AUD_JPY_H4.parquet",
        "AUD_JPY_D.parquet",
        "AUD_JPY_W.parquet",
    ],
    "mr_audusd": [
        "AUD_USD_H1.parquet",
        "AUD_USD_H4.parquet",
        "AUD_USD_D.parquet",
        "AUD_USD_W.parquet",
    ],
    "bond_equity_ihyu_cspx": [
        "CSPX_D.parquet",
        "IHYU_D.parquet",
    ],
    "bond_equity_ihyg_vusd": [
        "VUSD_D.parquet",
        "IHYG_D.parquet",
    ],
    "bond_equity_ihyg_eimi": [
        "EIMI_D.parquet",
        "IHYG_D.parquet",
    ],
}


def _check_data_freshness(logger: logging.Logger, selected: list[str]) -> None:
    """Inspect warmup parquets for selected strategies; warn if stale.

    Non-blocking — only logs. Strategy startup proceeds either way (a
    missing or stale file just means cold-start indicators).

    Thresholds (against the file's most recent bar timestamp):
      *  <= 2 days old : INFO  (fresh)
      *  3-7 days      : WARN  (cron may have failed once)
      *  > 7 days      : ERROR (consider refreshing before trading)
      *  missing       : WARN  (strategy will boot with cold indicators)
    """
    import pandas as pd

    now = datetime.now(timezone.utc)
    files_seen: set[str] = set()
    for strategy_name in selected:
        files = _STRATEGY_WARMUP_FILES.get(strategy_name)
        if not files:
            logger.debug(f"  [freshness] no warmup mapping for '{strategy_name}'")
            continue
        for fname in files:
            if fname in files_seen:
                continue
            files_seen.add(fname)
            path = DATA_DIR / fname
            if not path.exists():
                logger.warning(
                    f"  [freshness] {fname:30s} MISSING — strategy will boot with cold indicators"
                )
                continue
            try:
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    last_ts = pd.to_datetime(df["timestamp"]).max()
                else:
                    last_ts = pd.to_datetime(df.index).max()
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                else:
                    last_ts = last_ts.tz_convert("UTC")
                age_days = (now - last_ts.to_pydatetime()).days
            except Exception as e:
                logger.warning(f"  [freshness] {fname:30s} could not be read: {e}")
                continue
            line = f"  [freshness] {fname:30s} age={age_days:>3}d  last_bar={last_ts.date()}"
            if age_days <= 2:
                logger.info(f"{line}  fresh")
            elif age_days <= 7:
                logger.warning(f"{line}  stale (cron may have failed)")
            else:
                logger.error(f"{line}  VERY stale — refresh before trading")


# ── Account-equity auto-allocation ──────────────────────────────────────────
# Each strategy's BondGoldConfig / MRAUDJPYConfig defaults to
# initial_equity=$10,000 (a placeholder seed used for vol-targeted sizing).
# When the actual account is smaller (or larger) than N × $10k, sizing is
# mis-calibrated. We fix this at startup by querying the broker's
# NetLiquidation, expressing it in USD (the strategies' implicit base ccy),
# and dividing equally among the active TRADING strategies.


def _query_account_equity_usd(
    host: str,
    port: int,
    account_id: str,
    timeout: float = 12.0,
) -> float | None:
    """Open a brief IBKR connection (client_id=96) and return total
    NetLiquidation expressed in USD.

    Returns ``None`` on any failure — caller falls back to env override or
    registry defaults. Adds at most ``timeout`` seconds to startup.
    """
    import threading
    import time

    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ImportError:
        return None

    class _AcctApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.ready = False
            self.done = False
            self.base_nlv: float | None = None
            self.fx_usd_to_base: float | None = None

        def nextValidId(self, oid: int) -> None:  # type: ignore[override]
            self.ready = True

        def updateAccountValue(  # type: ignore[override]
            self, key: str, val: str, currency: str, accountName: str
        ) -> None:
            if key == "NetLiquidation" and currency == "BASE":
                try:
                    self.base_nlv = float(val)
                except (TypeError, ValueError):
                    pass
            elif key == "ExchangeRate" and currency == "USD":
                try:
                    self.fx_usd_to_base = float(val)
                except (TypeError, ValueError):
                    pass

        def accountDownloadEnd(self, accountName: str) -> None:  # type: ignore[override]
            self.done = True

        def error(  # type: ignore[override]
            self, reqId, errorCode, errorString, advancedOrderRejectJson=""
        ) -> None:
            # Suppress 21xx info messages; everything else is a noisy debug.
            return None

    app = _AcctApp()
    try:
        app.connect(host, port, 96)
    except Exception:
        return None

    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.time() + timeout
    while not app.ready and time.time() < deadline:
        time.sleep(0.05)
    if not app.ready:
        try:
            app.disconnect()
        except Exception:
            pass
        return None

    try:
        app.reqAccountUpdates(True, account_id)
    except Exception:
        try:
            app.disconnect()
        except Exception:
            pass
        return None

    deadline = time.time() + timeout
    while not app.done and time.time() < deadline:
        time.sleep(0.1)

    try:
        app.reqAccountUpdates(False, account_id)
    except Exception:
        pass
    time.sleep(0.3)
    try:
        app.disconnect()
    except Exception:
        pass

    if app.base_nlv is None or app.base_nlv <= 0:
        return None
    # If FX rate is missing (rare; happens when base ccy is already USD),
    # the BASE NLV is already a USD figure.
    if app.fx_usd_to_base is None or app.fx_usd_to_base <= 0:
        return float(app.base_nlv)
    return float(app.base_nlv) / float(app.fx_usd_to_base)


def _auto_allocate_initial_equity(
    selected: list[str],
    host: str,
    port: int,
    account_id: str,
    logger: logging.Logger,
) -> None:
    """Override each trading strategy's ``initial_equity`` based on actual
    account NLV. Uses ``TITAN_PORTFOLIO_USD_EQUITY`` env override if set,
    else queries the broker. On any failure, registry defaults are kept.

    Mutates ``STRATEGY_REGISTRY[name]["config_kwargs"]["initial_equity"]``
    in place for each strategy ``name`` flagged ``trading=True``.
    """
    trading = [s for s in selected if STRATEGY_REGISTRY.get(s, {}).get("trading", True)]
    n = len(trading)
    if n == 0:
        logger.info("  Auto-equity: no trading strategies in selection; skipped")
        return

    total_usd: float | None = None
    override = os.getenv("TITAN_PORTFOLIO_USD_EQUITY")
    if override:
        try:
            total_usd = float(override)
            logger.info(
                f"  Auto-equity: using env override TITAN_PORTFOLIO_USD_EQUITY=${total_usd:,.2f}"
            )
        except ValueError:
            logger.warning(
                f"  Auto-equity: invalid TITAN_PORTFOLIO_USD_EQUITY={override!r}; "
                "ignoring and querying broker"
            )

    if total_usd is None:
        if not account_id:
            logger.warning(
                "  Auto-equity: IBKR_ACCOUNT_ID not set; falling back to "
                "registry-default initial_equity. Strategies will be sized "
                "as if each had its own $10k."
            )
            return
        logger.info(f"  Auto-equity: querying NLV from {host}:{port}...")
        total_usd = _query_account_equity_usd(host, port, account_id)
        if total_usd is None:
            logger.warning(
                "  Auto-equity: broker query failed or timed out; falling back "
                "to registry-default initial_equity. Set "
                "TITAN_PORTFOLIO_USD_EQUITY in .env.docker to override."
            )
            return

    per_strategy = total_usd / n
    logger.info(
        f"  Auto-equity: account NLV = ${total_usd:,.2f} (USD), "
        f"divided across {n} trading strategies = ${per_strategy:,.2f} each"
    )
    for name in trading:
        entry = STRATEGY_REGISTRY[name]
        old = entry["config_kwargs"].get("initial_equity")
        entry["config_kwargs"]["initial_equity"] = per_strategy
        old_str = f"${old:,.2f}" if isinstance(old, (int, float)) else "(default)"
        logger.info(f"    {name:<30} initial_equity {old_str} → ${per_strategy:,.2f}")


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
    # ``gld_confluence`` removed from registry on 2026-05-01 after fresh-sweep
    # re-validation: full-history WFO Sharpe collapsed to +0.14 (documented
    # +1.46 not reproducible). Best of 16 sweep combos was Sharpe +0.35 with
    # 34% positive folds and -27% DD — well below the deployment gate. Code
    # stays in titan/strategies/gld_confluence/ for future reference; do not
    # add back without re-validating signal selection. See memory:
    # project_gld_confluence_uk_substitute.md.
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
            # Explicit JPY -> USD rate. The strategy now refuses to start with
            # the old default 1.0 when quote_ccy != base_ccy. Operator must
            # keep this value current (~0.0065 at 2026-04; monitor and update
            # when JPY/USD moves >5% to avoid sizing drift).
            "fx_rate_quote_to_base": 0.0065,
        },
    },
    "mr_audusd": {
        # Reuses generic MRAUDJPYStrategy class with AUD/USD config.
        # Class name is legacy; logic is asset-agnostic FX H1 MR + Donchian regime.
        "module": "titan.strategies.mr_audjpy.strategy",
        "config_cls": "MRAUDJPYConfig",
        "strategy_cls": "MRAUDJPYStrategy",
        "contracts": [
            IBContract(secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="USD"),
        ],
        "config_kwargs": {
            "instrument_id": "AUD/USD.IDEALPRO",
            "bar_type_h1": "AUD/USD.IDEALPRO-1-HOUR-MID-EXTERNAL",
            "ticker": "AUD_USD",  # loads data/AUD_USD_H1.parquet
            "vwap_anchor": 36,  # AUD/USD research champion (NOT 46 like AUD/JPY)
            "max_leverage": 2.0,  # Paper: 2x; ramp post-validation
        },
    },
    "bond_equity_ihyu_cspx": {
        # IHYU.LSEETF -> CSPX.LSEETF cross-asset (UCITS substitute for HYG -> IWB).
        # Original HYG/IWB blocked by EU/UK PRIIPs (no KID for US-domiciled ETFs).
        # WFO validated: Sharpe +1.638, 84% positive folds (25 folds 2013-2026).
        # Reuses generic BondGoldStrategy class; ticker_gld/ticker_ief vars are legacy names.
        "module": "titan.strategies.bond_gold.strategy",
        "config_cls": "BondGoldConfig",
        "strategy_cls": "BondGoldStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="CSPX",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="USD",
            ),
            IBContract(
                secType="STK",
                symbol="IHYU",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="USD",
            ),
        ],
        "config_kwargs": {
            # Note: contract uses exchange="SMART" + primaryExchange="LSEETF" so IBKR
            # smart-routes the order, but the instrument_id NautilusTrader registers
            # is built from primaryExchange = LSEETF.
            "instrument_id": "CSPX.LSEETF",
            "signal_instrument_id": "IHYU.LSEETF",
            "bar_type_d": "CSPX.LSEETF-1-DAY-LAST-EXTERNAL",
            "signal_bar_type_d": "IHYU.LSEETF-1-DAY-LAST-EXTERNAL",
            "ticker_gld": "CSPX",  # warmup reads data/CSPX_D.parquet (legacy var name)
            "ticker_ief": "IHYU",  # warmup reads data/IHYU_D.parquet (legacy var name)
            "lookback": 10,  # research champion: 10d (NOT bond_gold's 60d)
            "threshold": 0.50,
            "hold_days": 10,  # research champion: 10d (NOT bond_gold's 20d)
            "max_leverage": 2.0,
        },
    },
    "bond_equity_ihyg_vusd": {
        # IHYG (€ HY credit UCITS) -> VUSD (Vanguard S&P 500 UCITS, USD line)
        # cross-asset. Discovered May 2026 with CSPX target (pre-sanctuary
        # Sharpe +1.17, sanctuary +1.33). Switched to VUSD target so the
        # strategy holds a broker position distinct from the live
        # bond_equity_ihyu_cspx strategy (which trades CSPX) — avoids
        # position-attribution conflict.
        # VUSD tracks the same S&P 500 underlying as CSPX (different issuer).
        # The Vanguard LSE-listed S&P 500 UCITS has a GBP line (VUSA) and a
        # USD line (VUSD); we use VUSD so the strategy stays USD-quoted, just
        # like CSPX. IBKR rejects `symbol=VUSA, currency=USD` because that
        # specific line doesn't exist.
        # Re-validated on VUSD: Sharpe +1.16, CI lo +0.47, 94% pos folds, DD -13.2%.
        # See: project_ihyg_cspx_discovery.md.
        "module": "titan.strategies.bond_gold.strategy",
        "config_cls": "BondGoldConfig",
        "strategy_cls": "BondGoldStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="VUSD",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="USD",
            ),
            IBContract(
                secType="STK",
                symbol="IHYG",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="EUR",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "VUSD.LSEETF",
            "signal_instrument_id": "IHYG.LSEETF",
            "bar_type_d": "VUSD.LSEETF-1-DAY-LAST-EXTERNAL",
            "signal_bar_type_d": "IHYG.LSEETF-1-DAY-LAST-EXTERNAL",
            "ticker_gld": "VUSD",
            "ticker_ief": "IHYG",  # warmup reads data/IHYG_D.parquet
            "lookback": 5,
            "threshold": 0.25,
            "hold_days": 5,
            "max_leverage": 2.0,
        },
    },
    "bond_equity_ihyg_eimi": {
        # IHYG (€ HY credit UCITS) -> EIMI (iShares Core MSCI EM IMI UCITS,
        # USD line on LSE) cross-asset. Same EU-credit signal as
        # bond_equity_ihyg_vusd, routed to emerging-markets equity instead
        # of US large-cap. Discovered May 1 2026 in the post-VUSD
        # diversification hunt: 5/27 sweep cells have CI lo > 0.
        # Re-validated on EIMI (USD line; original sweep used EMIM=GBP-pence
        # data which carried a GBP/USD overlay — IBKR rejects symbol=EMIM
        # currency=USD because that line is GBP, just like VUSA→VUSD lesson):
        # champion (lb=5, hold=5, th=0.25 — same as VUSD) gives Sharpe
        # **+1.09, CI lo +0.32**, 94% pos folds, DD -10.8%. CI lo is now
        # ABOVE the strict Bonferroni gate (+0.30 required). Sanctuary
        # 2025+ on EMIM showed Sharpe +1.97 — strong out-of-sample.
        # Correlation with deployed IHYG -> VUSD ≈ 0.52 (meaningful
        # diversification).
        # See: project_ihyg_emim_discovery.md.
        "module": "titan.strategies.bond_gold.strategy",
        "config_cls": "BondGoldConfig",
        "strategy_cls": "BondGoldStrategy",
        "contracts": [
            IBContract(
                secType="STK",
                symbol="EIMI",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="USD",
            ),
            IBContract(
                secType="STK",
                symbol="IHYG",
                exchange="SMART",
                primaryExchange="LSEETF",
                currency="EUR",
            ),
        ],
        "config_kwargs": {
            "instrument_id": "EIMI.LSEETF",
            "signal_instrument_id": "IHYG.LSEETF",
            "bar_type_d": "EIMI.LSEETF-1-DAY-LAST-EXTERNAL",
            "signal_bar_type_d": "IHYG.LSEETF-1-DAY-LAST-EXTERNAL",
            "ticker_gld": "EIMI",
            "ticker_ief": "IHYG",  # warmup reads data/IHYG_D.parquet
            "lookback": 5,
            "threshold": 0.25,
            "hold_days": 5,
            "max_leverage": 2.0,
        },
    },
    # IC Equity (example -- add more as needed)
    "daily_summary": {
        # Passive strategy: posts a daily Slack/Telegram rollup at the
        # configured time-of-day. Doesn't trade. Subscribes to AUD/JPY H1
        # bars (already in champion portfolio) as a clock-tick source.
        "trading": False,  # passive — excluded from auto-equity allocation
        "module": "titan.strategies.daily_summary.strategy",
        "config_cls": "DailySummaryConfig",
        "strategy_cls": "DailySummaryStrategy",
        "contracts": [
            IBContract(secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="JPY"),
        ],
        "config_kwargs": {
            "bar_type": "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL",
            "summary_hour": int(os.getenv("DAILY_SUMMARY_HOUR", "9")),
            "summary_minute": int(os.getenv("DAILY_SUMMARY_MINUTE", "0")),
            "summary_tz": os.getenv("DAILY_SUMMARY_TZ", "Europe/London"),
        },
    },
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
    # gld_confluence dropped 2026-05-01 (see comment near registry).
    "gold_core": ["gold_macro", "bond_gold"],
    "h1_only": ["mr_audjpy"],
    # AUD/USD MR removed 2026-04-21 after post-remediation re-validation:
    # CI_lo = -0.180 < 0 fails the deployment gate. See directives/Deprecated
    # Strategies.md.
    "champion_portfolio": [
        "mr_audjpy",
        "bond_equity_ihyu_cspx",
        "bond_equity_ihyg_vusd",  # added 2026-05-01 (see project_ihyg_cspx_discovery.md)
        "bond_equity_ihyg_eimi",  # added 2026-05-01 (see project_ihyg_emim_discovery.md)
        "daily_summary",
    ],
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

    logger.info("  Warmup data freshness:")
    _check_data_freshness(logger, selected)
    logger.info("=" * 60)

    # Auto-allocate initial_equity across active trading strategies, sized
    # to the actual broker account NLV (in USD). Set
    # TITAN_PORTFOLIO_USD_EQUITY=<value> to bypass the broker query.
    logger.info("  Per-strategy seed equity (auto-allocated from account NLV):")
    _auto_allocate_initial_equity(selected, ib_host, ib_port, ib_account_id, logger)
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
