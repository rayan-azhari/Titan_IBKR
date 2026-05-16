"""download_ig_markets.py -- Pull IG Markets historical DAY bars for the
B2 EWMAC universe-expansion path.

Companion to ``scripts/ig_scout_catalog.py`` (epic discovery).  This
script does ONE additional thing: hit ``/prices/{epic}?resolution=DAY``
for each B2 instrument and save the result as a parquet under
``data/ig_markets/{TICKER}_DAY.parquet``.

Strictly READ-ONLY. Mutating endpoints touched: ``POST /session`` and
``DELETE /session`` only (login + logout). No order/working-order calls.

Auth
----
Reads the same dotenv as the scout — defaults to ``titan/.env.ig`` or
``./.env.ig`` in repo root. Required keys:

    IG_USERNAME
    IG_PASSWORD
    IG_API_KEY
    IG_ACCOUNT_ID
    IG_BASE_URL  (e.g. https://api.ig.com/gateway/deal for LIVE)

Universe
--------
B2 EWMAC ensemble — same 24-commodity universe as IBKR plus the
universe-expansion sleeves (bonds, FX, equity indices) per pre-reg B2
§4.8 follow-up. Each instrument has a list of candidate search terms;
the first match returning a DFB or MONTH1 epic is used. Filter rejects
leveraged / inverse / option products.

Rate limits
-----------
IG non-trading endpoints cap ~30-60 req/min. We sleep 1.5s between
calls. Historical-prices endpoint also has a monthly cap (10k points
per week on basic accounts; check your subscription).

Usage
-----
    uv run python scripts/download_ig_markets.py --sleeves all
    uv run python scripts/download_ig_markets.py --sleeves equity_index,bond
    uv run python scripts/download_ig_markets.py --years 10

Output
------
``data/ig_markets/{TICKER}_DAY.parquet`` columns:
    timestamp (UTC, normalised to date)
    open, high, low, close   (mid price = (bid + ask) / 2 from snapshot)
    volume                   (IG returns last-traded volume; may be 0
                              for spread bets — kept for compatibility)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "ig_markets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ENV_PATHS = [
    # LIVE key preferred when present — much higher rate limits than DEMO.
    # IG T&Cs: personal use only, no data redistribution. Compliance with
    # FCA leverage caps applies when extending this to ORDER endpoints.
    PROJECT_ROOT / ".env.ig.live",
    PROJECT_ROOT / "titan" / ".env.ig.live",
    PROJECT_ROOT / ".env.ig",
    PROJECT_ROOT / "titan" / ".env.ig",
]

REQUEST_SLEEP_S = 1.5  # ~40 req/min, safely under IG cap


# ---------------------------------------------------------------------------
# B2 universe — instrument label + search terms (first hit with acceptable
# expiry wins). Epics are NOT hardcoded because IG epic IDs differ between
# DEMO and LIVE accounts.
# ---------------------------------------------------------------------------

UNIVERSE: dict[str, dict[str, list[str]]] = {
    # Sleeves are processed independently so partial download is OK.
    "commodity_energy": {
        "CL": ["Oil - US Crude", "US Crude Oil"],
        "BZ": ["Oil - Brent Crude", "Brent Crude"],
        "NG": ["Natural Gas"],
        "HO": ["Heating Oil", "NY Harbor ULSD"],
        "RB": ["Gasoline", "RBOB Gasoline"],
    },
    "commodity_metals": {
        "GC": ["Spot Gold", "Gold"],
        "SI": ["Spot Silver", "Silver"],
        "HG": ["High Grade Copper", "Copper"],
        "PL": ["Platinum"],
        "PA": ["Palladium"],
    },
    "commodity_grains": {
        "ZC": ["Corn"],
        "ZW": ["Chicago Wheat", "Wheat"],
        "ZS": ["Soybeans"],
        "ZL": ["Soybean Oil"],
        "ZM": ["Soybean Meal"],
    },
    "commodity_softs": {
        "KC": ["Coffee", "Coffee Arabica"],
        "CC": ["Cocoa"],
        "SB": ["Sugar No. 11", "Sugar"],
        "CT": ["Cotton No. 2", "Cotton"],
        "OJ": ["Orange Juice"],
    },
    "fx_major": {
        "EURUSD": ["EUR/USD"],
        "GBPUSD": ["GBP/USD"],
        "USDJPY": ["USD/JPY"],
        "USDCHF": ["USD/CHF"],
        "AUDUSD": ["AUD/USD"],
        "USDCAD": ["USD/CAD"],
        "NZDUSD": ["NZD/USD"],
        "DXY": ["US Dollar Index", "Dollar Basket"],
    },
    "bond": {
        # IG's UK SB catalogue labels bonds variably; including the most
        # common alternates. The sleeve-aware filter restricts to BONDS
        # or INDICES inst types so commodity false-positives are filtered.
        "US10Y": ["US T-Note", "10 Year T-Note", "US 10 Year", "10Y T-Note"],
        "US30Y": ["US T-Bond", "30 Year T-Bond", "US Long Bond Future"],
        "BUND": ["Bund", "German 10 Year", "Euro Bund"],
        "GILT": ["Long Gilt", "UK Long Gilt", "Gilt Future"],
    },
    "equity_index": {
        "SPX": ["US 500", "S&P 500"],
        "NDX": ["US Tech 100", "Nasdaq 100"],
        "DJI": ["Wall Street", "Dow Jones"],
        "RUT": ["US Russell 2000"],
        "FTSE": ["FTSE 100"],
        "DAX": ["Germany 40", "DAX"],
        "NIKKEI": ["Japan 225", "Nikkei 225"],
        "EUROSTOXX": ["EU Stocks 50", "Euro Stoxx 50"],
    },
    # ── Gap-fillers added 2026-05-15 PM after IG-live unlock ─────────────
    "volatility_index": {
        # Useful for E1 VRP re-audit + regime-aware overlays. IG offers
        # the VIX as a spread-bet "Volatility Index"; depth ~10y on DAY.
        "VIX": ["Volatility Index", "VIX"],
        "VSTOXX": ["Euro Volatility", "VSTOXX"],
    },
    "lse_ucits_etf": {
        # Bond_equity strategies' UCITS underlyings. Previously catalogued
        # in 2026-05-12 scout (KA.D.* epics). Pulling daily history lets us
        # re-audit the bond_equity champion under IG's price feed without
        # touching IBKR. Note: these are SHARES instrument type on IG so
        # the sleeve filter allows that — see SLEEVE_TYPE_PREFERENCES.
        "CSPX": ["CSPX", "iShares Core S&P 500"],
        "VUSD": ["VUSD", "Vanguard S&P 500"],
        "EIMI": ["EIMI", "iShares Emerging Markets IMI"],
        "IHYG": ["IHYG", "iShares EUR High Yield"],
        "IHYU": ["IHYU", "iShares USD High Yield"],
        "IGLT": ["IGLT", "iShares Core UK Gilts"],
    },
}


# ---------------------------------------------------------------------------
# Minimal IG REST client (read-only; copies the scout's auth pattern).
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _resolve_env_file(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg)
        if not p.exists():
            raise FileNotFoundError(f"--env-file not found: {p}")
        return p
    for candidate in DEFAULT_ENV_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No .env.ig found. Searched: {[str(p) for p in DEFAULT_ENV_PATHS]}.")


@dataclass
class IGClient:
    base_url: str
    api_key: str
    username: str
    password: str
    account_id: str
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("ig.dl"))
    cst: str | None = None
    x_security_token: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        version: str = "1",
        body: dict | None = None,
        authed: bool = True,
    ) -> tuple[int, dict, dict[str, str]]:
        url = self.base_url.rstrip("/") + path
        headers = {
            "X-IG-API-KEY": self.api_key,
            "VERSION": version,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
        }
        if authed:
            if not (self.cst and self.x_security_token):
                raise RuntimeError("Not authenticated — call login() first")
            headers["CST"] = self.cst
            headers["X-SECURITY-TOKEN"] = self.x_security_token
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = request.Request(url=url, data=data, method=method, headers=headers)
        time.sleep(REQUEST_SLEEP_S)
        try:
            with request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                status = resp.getcode()
                resp_headers = {k: v for k, v in resp.getheaders()}
        except error.HTTPError as e:
            raw = e.read()
            status = e.code
            resp_headers = dict(e.headers.items()) if e.headers else {}
        body_json: dict = {}
        if raw:
            try:
                body_json = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body_json = {"_raw": raw.decode("utf-8", errors="replace")}
        return status, body_json, resp_headers

    def login(self) -> dict:
        status, body, h = self._request(
            "POST",
            "/session",
            version="2",
            body={"identifier": self.username, "password": self.password},
            authed=False,
        )
        if status != 200:
            raise RuntimeError(f"Login failed: HTTP {status} body={body}")
        self.cst = h.get("CST") or h.get("cst")
        self.x_security_token = h.get("X-SECURITY-TOKEN") or h.get("x-security-token")
        if not (self.cst and self.x_security_token):
            raise RuntimeError("Login returned 200 but no session tokens")
        current = body.get("currentAccountId")
        if current and current != self.account_id:
            self.logger.warning(
                f"Logged in to {current} but IG_ACCOUNT_ID={self.account_id}; switching"
            )
            self._switch_account(self.account_id)
        return body

    def _switch_account(self, account_id: str) -> None:
        status, body, _ = self._request(
            "PUT",
            "/session",
            version="1",
            body={"accountId": account_id, "defaultAccount": False},
        )
        if status != 200:
            raise RuntimeError(f"Account switch failed: HTTP {status} body={body}")

    def logout(self) -> None:
        if not (self.cst and self.x_security_token):
            return
        try:
            self._request("DELETE", "/session", version="1")
        finally:
            self.cst = None
            self.x_security_token = None

    def search_markets(self, term: str) -> list[dict[str, Any]]:
        status, body, _ = self._request(
            "GET", f"/markets?searchTerm={parse.quote(term)}", version="1"
        )
        if status != 200:
            self.logger.warning(f"  search '{term}' failed: HTTP {status}")
            return []
        return body.get("markets", []) or []

    def prices(
        self,
        epic: str,
        *,
        resolution: str = "DAY",
        from_date: str,
        to_date: str,
        max_points: int = 10000,
    ) -> tuple[list[dict[str, Any]], dict]:
        """GET /prices/{epic} -- returns (prices, allowance_meta).

        ``from_date`` / ``to_date`` format: ``YYYY-MM-DDTHH:MM:SS`` (UTC).
        """
        q = parse.urlencode(
            {
                "resolution": resolution,
                "from": from_date,
                "to": to_date,
                "max": str(max_points),
                "pageSize": "0",
            }
        )
        status, body, _ = self._request("GET", f"/prices/{epic}?{q}", version="3")
        if status != 200:
            self.logger.warning(f"  prices({epic}) HTTP {status}: {body}")
            return [], {}
        return body.get("prices") or [], body.get("metadata") or {}


# ---------------------------------------------------------------------------
# Search / epic resolution
# ---------------------------------------------------------------------------


SLEEVE_TYPE_PREFERENCES: dict[str, tuple[str, ...]] = {
    # First-match-wins acceptable inst types per sleeve. Tightens the
    # search filter so "US Long Bond" doesn't accidentally match
    # "London Sugar" (COMMODITY).
    "commodity_energy": ("COMMODITIES",),
    "commodity_metals": ("COMMODITIES",),
    "commodity_grains": ("COMMODITIES",),
    "commodity_softs": ("COMMODITIES",),
    "fx_major": ("CURRENCIES",),
    "bond": ("BONDS", "INDICES"),
    "equity_index": ("INDICES",),
    "volatility_index": ("INDICES",),
    # UCITS ETFs come back as SHARES on IG SB; allow that here.
    "lse_ucits_etf": ("SHARES",),
}


def _is_acceptable(market: dict[str, Any], sleeve: str | None = None) -> bool:
    """Reject leveraged / inverse / option / share products. Keep INDICES,
    COMMODITIES, CURRENCIES, BONDS. Prefer DFB or MONTH1 expiry. When a
    ``sleeve`` is provided, restricts to the sleeve's preferred inst types
    so cross-sleeve false positives are filtered out."""
    inst_type = (market.get("instrumentType") or "").upper()
    name = (market.get("instrumentName") or "").lower()
    expiry = (market.get("expiry") or "").upper()
    allowed_types = (
        SLEEVE_TYPE_PREFERENCES.get(sleeve, ("INDICES", "COMMODITIES", "CURRENCIES", "BONDS"))
        if sleeve
        else ("INDICES", "COMMODITIES", "CURRENCIES", "BONDS")
    )
    if inst_type not in allowed_types:
        return False
    bad_terms = ("leveraged", "x leverage", "inverse", "option", "binary", "weekly")
    if any(t in name for t in bad_terms):
        return False
    # Tradeable status helps but isn't always set on search results.
    status = (market.get("marketStatus") or "TRADEABLE").upper()
    if status in {"CLOSED", "SUSPENDED"}:
        return False
    # Accept DFB (continuous) or quarterly expiries; reject hourly/daily-rolling odd-balls.
    if expiry not in {"DFB", "-", ""} and "-" not in expiry:
        return False
    return True


def _epic_priority(market: dict[str, Any]) -> tuple[int, int]:
    """Lower is better. (expiry_class, type_class).

    expiry_class: 0 = DFB (continuous), 1 = nearest dated future, 2 = other.
    type_class:   0 = INDICES, 1 = COMMODITIES, 2 = BONDS, 3 = CURRENCIES, 4 = other.
    """
    expiry = (market.get("expiry") or "").upper()
    if expiry == "DFB":
        ec = 0
    elif "-" in expiry:
        ec = 1
    else:
        ec = 2
    type_map = {"INDICES": 0, "COMMODITIES": 1, "BONDS": 2, "CURRENCIES": 3}
    tc = type_map.get((market.get("instrumentType") or "").upper(), 4)
    return (ec, tc)


def resolve_epic(
    client: IGClient,
    search_terms: list[str],
    logger: logging.Logger,
    sleeve: str | None = None,
) -> dict[str, Any] | None:
    """Search each term until we find an acceptable epic; return best by priority.

    ``sleeve`` constrains the inst-type filter so a search like "US Long
    Bond" doesn't false-positive against "London Sugar"."""
    candidates: list[dict[str, Any]] = []
    for term in search_terms:
        markets = client.search_markets(term)
        acceptable = [m for m in markets if _is_acceptable(m, sleeve=sleeve)]
        candidates.extend(acceptable)
        if acceptable:
            break  # first matching search term wins
    if not candidates:
        return None
    candidates.sort(key=_epic_priority)
    return candidates[0]


# ---------------------------------------------------------------------------
# Price-data normalisation
# ---------------------------------------------------------------------------


def _prices_to_df(prices: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert IG /prices payload into our standard OHLCV parquet schema."""
    rows = []
    for p in prices:
        ts = p.get("snapshotTime") or p.get("snapshotTimeUTC")
        if not ts:
            continue
        try:
            t = (
                datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                if "T" in ts
                else datetime.strptime(ts[:19], "%Y/%m/%d %H:%M:%S")
            )
        except ValueError:
            continue
        bid = p.get("closePrice", {}).get("bid")
        ask = p.get("closePrice", {}).get("ask")
        if bid is None or ask is None:
            continue
        close = (bid + ask) / 2.0

        def mid(side: str) -> float | None:
            d = p.get(f"{side}Price") or {}
            b, a = d.get("bid"), d.get("ask")
            if b is None or a is None:
                return None
            return (b + a) / 2.0

        rows.append(
            {
                "timestamp": t,
                "open": mid("open") or close,
                "high": mid("high") or close,
                "low": mid("low") or close,
                "close": close,
                "volume": float(p.get("lastTradedVolume") or 0),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = df.index.normalize()
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Main download driver
# ---------------------------------------------------------------------------


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ig.dl")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=None)
    parser.add_argument(
        "--sleeves",
        default="all",
        help=f"Comma-separated sleeves to download (or 'all'). Valid: {','.join(UNIVERSE.keys())}",
    )
    parser.add_argument(
        "--instruments",
        default=None,
        help="Comma-separated instrument labels to filter (e.g. CL,GC,EURUSD)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=15,
        help="Years of history to request (default 15; IG caps shorter on many epics)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve epics + show planned downloads but do not pull prices",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download instruments even when a cached parquet exists",
    )
    args = parser.parse_args()

    logger = _setup_logger()
    env_path = _resolve_env_file(args.env_file)
    env = _load_dotenv(env_path)
    logger.info(f"Loaded env: {env_path}  keys={list(env.keys())}")

    for required in ("IG_USERNAME", "IG_PASSWORD", "IG_API_KEY", "IG_ACCOUNT_ID", "IG_BASE_URL"):
        if required not in env or not env[required]:
            logger.error(f"Missing required env key: {required}")
            return 1

    if args.sleeves.lower() == "all":
        sleeves = list(UNIVERSE.keys())
    else:
        sleeves = [s.strip() for s in args.sleeves.split(",") if s.strip()]
        unknown = [s for s in sleeves if s not in UNIVERSE]
        if unknown:
            logger.error(f"Unknown sleeves: {unknown}. Valid: {list(UNIVERSE.keys())}")
            return 2

    instrument_filter: set[str] | None = (
        {s.strip().upper() for s in args.instruments.split(",")} if args.instruments else None
    )

    # Build the work list.
    work: list[tuple[str, str, list[str]]] = []  # (sleeve, instrument, search_terms)
    for sleeve in sleeves:
        for inst, terms in UNIVERSE[sleeve].items():
            if instrument_filter and inst.upper() not in instrument_filter:
                continue
            work.append((sleeve, inst, terms))
    logger.info(f"Planned downloads: {len(work)} instruments across {len(sleeves)} sleeve(s)")

    succeeded: list[tuple[str, str, str, int]] = []
    failed: list[tuple[str, str, str]] = []
    resolution_log: list[dict[str, Any]] = []
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    from_date = (datetime.now(timezone.utc) - timedelta(days=int(365 * args.years))).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    # IG demo API keys can flag a key after long sessions with many calls.
    # Run ONE SLEEVE per session, logout between, sleep 30s, re-login.
    # Skip instruments whose cache file already exists (idempotent re-run).
    sleeves_to_process = list({s for s, _, _ in work})
    sleeves_to_process.sort(key=lambda s: list(UNIVERSE.keys()).index(s))

    def _new_client() -> IGClient:
        return IGClient(
            base_url=env["IG_BASE_URL"],
            api_key=env["IG_API_KEY"],
            username=env["IG_USERNAME"],
            password=env["IG_PASSWORD"],
            account_id=env["IG_ACCOUNT_ID"],
            logger=logger,
        )

    # SINGLE-SESSION model — one login at start, one logout at end. Per
    # observed IG behaviour, the rate limit is on session-token churn
    # (login frequency), not on per-session call volume.
    all_work: list[tuple[str, str, list[str]]] = []
    for sleeve in sleeves_to_process:
        for s, i, t in work:
            if s != sleeve:
                continue
            if (DATA_DIR / f"{i}_DAY.parquet").exists() and not args.force_redownload:
                continue
            all_work.append((s, i, t))
    if not all_work:
        logger.info("All requested instruments already cached -- nothing to do.")
        return 0
    logger.info(
        f"Single-session download: {len(all_work)} instruments to fetch "
        f"({len(sleeves_to_process)} sleeves)"
    )

    client = _new_client()
    try:
        session_info = client.login()
        logger.info(
            f"  authenticated: account={session_info.get('currentAccountId')} "
            f"currency={session_info.get('currencyIsoCode')}"
        )
        _download_sleeve(
            client,
            all_work,
            from_date,
            to_date,
            args.dry_run,
            succeeded,
            failed,
            resolution_log,
            logger,
        )
    except RuntimeError as exc:
        logger.error(f"session error: {exc}")
        for s, i, _ in all_work:
            if not any(succ_s == s and succ_i == i for succ_s, succ_i, _, _ in succeeded):
                failed.append((s, i, "session_error"))
    finally:
        try:
            client.logout()
            logger.info("  session destroyed")
        except Exception:  # noqa: BLE001
            pass

    # Write resolution log + manifest.
    (DATA_DIR / "_resolution.json").write_text(
        json.dumps(resolution_log, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print(f"  IG download: {len(succeeded)} succeeded, {len(failed)} failed")
    print("=" * 70)
    for sleeve, inst, epic, n in succeeded:
        print(f"  [{sleeve:>16}] {inst:<8} {epic:<32} {n} bars")
    for sleeve, inst, reason in failed:
        print(f"  [FAIL {sleeve:>11}] {inst:<8} reason={reason}")
    return 0 if not failed else 2


def _download_sleeve(
    client: "IGClient",
    sleeve_work: list[tuple[str, str, list[str]]],
    from_date: str,
    to_date: str,
    dry_run: bool,
    succeeded: list[tuple[str, str, str, int]],
    failed: list[tuple[str, str, str]],
    resolution_log: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Process all instruments in one sleeve under a single session."""
    for sleeve, inst, terms in sleeve_work:
        logger.info(f"\n=== [{sleeve}] {inst} ===")
        chosen = resolve_epic(client, terms, logger, sleeve=sleeve)
        if chosen is None:
            logger.warning(f"  no acceptable epic for {inst} (terms={terms})")
            failed.append((sleeve, inst, "no_epic"))
            resolution_log.append(
                {"sleeve": sleeve, "instrument": inst, "epic": None, "reason": "no_match"}
            )
            continue
        epic = chosen.get("epic")
        inst_name = chosen.get("instrumentName")
        inst_type = chosen.get("instrumentType")
        expiry = chosen.get("expiry")
        logger.info(
            f"  resolved -> epic={epic!r} name={inst_name!r} type={inst_type} expiry={expiry}"
        )
        resolution_log.append(
            {
                "sleeve": sleeve,
                "instrument": inst,
                "epic": epic,
                "name": inst_name,
                "type": inst_type,
                "expiry": expiry,
            }
        )
        if dry_run:
            continue
        prices, meta = client.prices(epic, from_date=from_date, to_date=to_date)
        df = _prices_to_df(prices)
        if df.empty:
            logger.warning(f"  no bars returned for {inst}@{epic} (allowance={meta})")
            failed.append((sleeve, inst, "no_bars"))
            continue
        out_path = DATA_DIR / f"{inst}_DAY.parquet"
        df.to_parquet(out_path)
        first = df.index[0].date()
        last = df.index[-1].date()
        logger.info(f"  saved {len(df)} bars  {first} .. {last}  -> {out_path.name}")
        allowance_used = (meta or {}).get("allowance", {}).get("remainingAllowance")
        if allowance_used is not None:
            logger.info(f"  IG allowance remaining: {allowance_used}")
        succeeded.append((sleeve, inst, epic, len(df)))


if __name__ == "__main__":
    sys.exit(main())
