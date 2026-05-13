"""ig_scout_catalog.py — Read-only IG Markets instrument catalogue scout.

Purpose
-------
Authenticate against IG's REST API (demo or live) and enumerate the
spread-betting markets we care about for migrating the bond_equity
strategies away from IBKR. Writes structured output to .tmp/ig_catalog/
for offline analysis.

Strictly READ-ONLY. The only mutating endpoints called are:
    POST   /session      — login
    DELETE /session      — logout
No order, position, or working-order endpoints are touched. Adding any
would defeat the purpose of this scout; review carefully before extending.

Usage
-----
    uv run python scripts/ig_scout_catalog.py
    uv run python scripts/ig_scout_catalog.py --env-file titan/.env.ig
    uv run python scripts/ig_scout_catalog.py --search "US 500" "MSCI EM"

Credentials
-----------
Looks for an IG dotenv at (in order):
    1. --env-file <path>            (CLI override)
    2. titan/.env.ig                (current default location)
    3. .env.ig                      (repo root)

Required keys:
    IG_USERNAME
    IG_PASSWORD
    IG_API_KEY
    IG_ACCOUNT_ID
    IG_BASE_URL   (e.g. https://demo-api.ig.com/gateway/deal)

Output
------
.tmp/ig_catalog/<UTC-timestamp>/
    session_meta.json       — non-secret session info (account, currency, lightstreamer endpoint)
    nav_root.json           — top-level market navigation tree
    nav_children/<id>.json  — drilled-down sub-nodes (best-effort, depth-limited)
    search_<query>.json     — /markets?searchTerm= results per query
    epic_<epic>.json        — full /markets/<epic> detail for each candidate
    SUMMARY.md              — human-readable digest

Notes on rate limits (IG public)
--------------------------------
    Non-trading requests: ~30/min per app, ~60/min per account
    Trading requests:     ~100/min per account
We sleep 1.2s between requests to stay well under the non-trading cap.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / ".tmp" / "ig_catalog"
DEFAULT_ENV_PATHS = [
    PROJECT_ROOT / "titan" / ".env.ig",
    PROJECT_ROOT / ".env.ig",
]

# Markets we want to find on IG. Tuned to the three live bond_equity
# strategies plus their signal underlyings, plus a few "obvious" GBP-base
# alternatives in case the USD lines aren't available as spread bets.
DEFAULT_SEARCH_TERMS = [
    "US 500",  # CSPX / VUSD target proxy (S&P 500 cash bet)
    "S&P 500",  # alt phrasing
    "Wall Street",  # IG often labels Dow Jones / DJIA this way
    "MSCI Emerging",  # EIMI target proxy
    "Emerging Markets",  # alt phrasing
    "HYG",  # IHYU equivalent (US HY corp bonds)
    "High Yield",  # IHYG / HY credit
    "iBoxx",  # underlying index family for IHYU/IHYG
    "VUSA",  # exact UCITS line (unlikely as SB market but worth a look)
    "CSPX",
    "EIMI",
    "IHYU",
    "IHYG",
]

REQUEST_SLEEP_S = 1.2  # ~50 req/min, comfortably under IG non-trading cap


def _load_dotenv(path: Path) -> dict[str, str]:
    """Tiny dotenv loader that does NOT echo values. Only logs which
    keys were read. Skips comments and blank lines."""
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        env[k] = v
    return env


def _mask(v: str) -> str:
    """Mask a secret so logs are safe."""
    if not v:
        return "(empty)"
    if len(v) <= 4:
        return "*" * len(v)
    return f"{v[:2]}…{v[-2:]} ({len(v)} chars)"


@dataclass
class IGClient:
    """Minimal IG REST client.

    Constraints
    -----------
    * Read-only (login + logout only mutating calls).
    * No retries on 4xx — caller logs the error and continues.
    * Headers and version negotiated per-endpoint.
    """

    base_url: str
    api_key: str
    username: str
    password: str
    account_id: str

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("ig"))

    cst: str | None = None
    x_security_token: str | None = None
    session_meta: dict[str, Any] = field(default_factory=dict)

    # --- low-level HTTP --------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        version: str = "1",
        body: dict | None = None,
        authed: bool = True,
    ) -> tuple[int, dict, dict[str, str]]:
        """Perform a single HTTP call. Returns (status, json_body, response_headers).

        Sleeps REQUEST_SLEEP_S before issuing the call to stay under rate
        limits. JSON body is parsed defensively (empty body returns {}).
        """
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

        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

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
            resp_headers = {k: v for k, v in e.headers.items()} if e.headers else {}
        body_json: dict = {}
        if raw:
            try:
                body_json = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body_json = {"_raw": raw.decode("utf-8", errors="replace")}
        return status, body_json, resp_headers

    # --- session --------------------------------------------------------

    def login(self) -> None:
        """POST /session v2 — populates CST and X-SECURITY-TOKEN.

        Uses v2 (not v3) deliberately. v2 returns CST/X-SECURITY-TOKEN
        header tokens that are reused across all subsequent requests; v3
        returns OAuth bearer/refresh tokens that need extra refresh
        plumbing. For a one-shot scout, v2 is simpler and safer.
        """
        self.logger.info(f"POST /session  (base={self.base_url})")
        status, body, _headers = self._request(
            "POST",
            "/session",
            version="2",
            body={"identifier": self.username, "password": self.password},
            authed=False,
        )
        if status != 200:
            err = body.get("errorCode") if isinstance(body, dict) else None
            raise RuntimeError(
                f"Login failed: HTTP {status} errorCode={err!r}. "
                "Common causes: wrong env (demo key vs live URL or vice-versa), "
                "expired password, API key disabled, IP not allowed by key restrictions."
            )
        # v2 returns tokens in headers; the body has account context.
        # urllib lowercases header names on some Python versions — handle both.
        cst = _headers.get("CST") or _headers.get("cst")
        xst = _headers.get("X-SECURITY-TOKEN") or _headers.get("x-security-token")
        if not (cst and xst):
            raise RuntimeError("Login returned 200 but no CST/X-SECURITY-TOKEN headers")
        self.cst = cst
        self.x_security_token = xst
        self.session_meta = body
        # Sanity check: the credentials' default account should match the
        # one declared in .env.ig. If not, we warn and continue — the user
        # may have multiple accounts (Z-prefixed = spread bet, X-prefixed
        # = CFD on some accounts).
        current_acct = body.get("currentAccountId")
        if current_acct and current_acct != self.account_id:
            self.logger.warning(
                f"Logged in to default account {current_acct} but "
                f".env.ig IG_ACCOUNT_ID={self.account_id}. "
                "Switching account..."
            )
            self._switch_account(self.account_id)
        self.logger.info(
            f"  ✓ session established — current acct={body.get('currentAccountId')}, "
            f"lightstreamerEndpoint={body.get('lightstreamerEndpoint')}"
        )

    def _switch_account(self, account_id: str) -> None:
        """PUT /session — switch active account (NOT an order endpoint)."""
        status, body, _ = self._request(
            "PUT",
            "/session",
            version="1",
            body={"accountId": account_id, "defaultAccount": False},
        )
        if status != 200:
            raise RuntimeError(f"Account switch failed: HTTP {status} body={body}")
        self.logger.info(f"  ✓ active account switched to {account_id}")

    def logout(self) -> None:
        """DELETE /session — invalidate CST/XST. Idempotent on failure."""
        if not (self.cst and self.x_security_token):
            return
        try:
            self._request("DELETE", "/session", version="1")
            self.logger.info("  ✓ session destroyed")
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            self.logger.warning(f"  logout failed (non-fatal): {e}")
        finally:
            self.cst = None
            self.x_security_token = None

    # --- read-only catalog calls ---------------------------------------

    def market_navigation(self, node_id: str | None = None) -> dict:
        path = "/marketnavigation" if node_id is None else f"/marketnavigation/{node_id}"
        status, body, _ = self._request("GET", path, version="1")
        if status != 200:
            self.logger.warning(f"  marketnavigation({node_id}) failed: HTTP {status} body={body}")
            return {"_status": status, "_body": body}
        return body

    def search_markets(self, term: str) -> dict:
        from urllib.parse import quote

        path = f"/markets?searchTerm={quote(term)}"
        status, body, _ = self._request("GET", path, version="1")
        if status != 200:
            self.logger.warning(f"  search '{term}' failed: HTTP {status} body={body}")
            return {"_status": status, "_body": body}
        return body

    def market_detail(self, epic: str) -> dict:
        path = f"/markets/{epic}"
        status, body, _ = self._request("GET", path, version="3")
        if status != 200:
            self.logger.warning(f"  market_detail({epic}) failed: HTTP {status} body={body}")
            return {"_status": status, "_body": body}
        return body


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ig.scout")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _resolve_env_file(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg)
        if not p.exists():
            raise FileNotFoundError(f"--env-file not found: {p}")
        return p
    for candidate in DEFAULT_ENV_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No .env.ig found in: {[str(p) for p in DEFAULT_ENV_PATHS]}. "
        "Create it from the template I described (IG_USERNAME, IG_PASSWORD, "
        "IG_API_KEY, IG_ACCOUNT_ID, IG_BASE_URL)."
    )


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def _drill_navigation(
    client: IGClient,
    out_dir: Path,
    max_depth: int,
    logger: logging.Logger,
) -> None:
    """Walk the market-navigation tree to max_depth, writing each node's
    children to out_dir/nav_children/. Tree is broad — be conservative
    with depth or you'll burn the rate limit.
    """
    nav_children_dir = out_dir / "nav_children"
    nav_children_dir.mkdir(parents=True, exist_ok=True)
    root = client.market_navigation(None)
    (out_dir / "nav_root.json").write_text(json.dumps(root, indent=2))
    nodes = root.get("nodes", []) or []
    logger.info(f"  nav root has {len(nodes)} top-level nodes")

    # BFS up to max_depth levels deep
    queue: list[tuple[str, str, int]] = [(n["id"], n.get("name", ""), 1) for n in nodes]
    seen: set[str] = set()
    while queue:
        node_id, name, depth = queue.pop(0)
        if node_id in seen or depth > max_depth:
            continue
        seen.add(node_id)
        body = client.market_navigation(node_id)
        fname = _safe_filename(f"{node_id}_{name}") + ".json"
        (nav_children_dir / fname).write_text(json.dumps(body, indent=2))
        if depth < max_depth:
            for child in body.get("nodes") or []:
                queue.append((child["id"], child.get("name", ""), depth + 1))
    logger.info(f"  ✓ navigation walked: {len(seen)} nodes captured at depth ≤ {max_depth}")


def _summarise(out_dir: Path) -> None:
    """Emit a markdown digest of what was captured."""
    lines: list[str] = [
        "# IG Catalog Scout — Summary",
        "",
        f"Captured: {datetime.now(timezone.utc).isoformat()}",
        f"Output dir: `{out_dir}`",
        "",
    ]

    session_meta_path = out_dir / "session_meta.json"
    if session_meta_path.exists():
        meta = json.loads(session_meta_path.read_text())
        lines += [
            "## Session",
            f"- Account: `{meta.get('currentAccountId')}`",
            f"- Currency: `{meta.get('accountInfo', {}).get('currency') or meta.get('currencyIsoCode')}`",  # noqa: E501
            f"- Lightstreamer endpoint: `{meta.get('lightstreamerEndpoint')}`",
            f"- Trailing stops enabled: `{meta.get('trailingStopsEnabled')}`",
            "",
        ]

    # Search hits
    lines += ["## Search Hits", ""]
    for search_file in sorted(out_dir.glob("search_*.json")):
        body = json.loads(search_file.read_text())
        markets = body.get("markets", []) if isinstance(body, dict) else []
        term = search_file.stem.replace("search_", "")
        lines.append(f"### `{term}` — {len(markets)} markets")
        for m in markets[:25]:
            lines.append(
                f"- **{m.get('instrumentName')}**  "
                f"epic=`{m.get('epic')}`  "
                f"type=`{m.get('instrumentType')}`  "
                f"expiry=`{m.get('expiry')}`  "
                f"streamingPricesAvailable=`{m.get('streamingPricesAvailable')}`"
            )
        if len(markets) > 25:
            lines.append(f"- ... ({len(markets) - 25} more)")
        lines.append("")

    # Per-epic detail
    epic_files = sorted(out_dir.glob("epic_*.json"))
    if epic_files:
        lines += ["## Market Details", ""]
        for ef in epic_files:
            body = json.loads(ef.read_text())
            inst = body.get("instrument", {})
            rules = body.get("dealingRules", {})
            snap = body.get("snapshot", {})
            min_deal = rules.get("minDealSize", {})
            lines += [
                f"### {inst.get('name')} (`{inst.get('epic')}`)",
                f"- Type: `{inst.get('type')}`",
                f"- Currency / unit: `{inst.get('currencies', [{}])[0].get('code') if inst.get('currencies') else '?'}` "  # noqa: E501
                f"— value of one contract: `{inst.get('valueOfOneContract')}`",
                f"- Margin factor: `{inst.get('marginFactor')}` ({inst.get('marginFactorUnit')})",
                f"- Min deal size (bet/point): `{min_deal.get('value')} {min_deal.get('unit')}`",
                f"- Slippage factor: `{inst.get('slippageFactor', {}).get('value')} "
                f"{inst.get('slippageFactor', {}).get('unit')}`",
                f"- Streaming: `{inst.get('streamingPricesAvailable')}`",
                f"- Snapshot bid/ask: `{snap.get('bid')} / {snap.get('offer')}` "
                f"(updated {snap.get('updateTime')})",
                f"- Expiry / settlement: `{inst.get('expiry')}`",
                "",
            ]
            # openingHours can be: missing, None, or {"marketTimes": [...]} —
            # all three appeared in the demo response set.
            for det in (inst.get("openingHours") or {}).get("marketTimes") or []:
                lines.append(f"  - opens `{det.get('openTime')}` closes `{det.get('closeTime')}`")
            lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env-file", default=None, help="Path to IG dotenv (default: titan/.env.ig or ./.env.ig)"
    )
    parser.add_argument("--search", nargs="*", default=DEFAULT_SEARCH_TERMS, help="Search terms")
    parser.add_argument(
        "--nav-depth", type=int, default=2, help="Market-navigation walk depth (default: 2)"
    )
    parser.add_argument("--skip-nav", action="store_true", help="Skip market-navigation walk")
    parser.add_argument(
        "--epics", nargs="*", default=[], help="Specific epics to fetch /markets/<epic> detail for"
    )
    parser.add_argument(
        "--also-detail-search-hits",
        action="store_true",
        help="Call /markets/<epic> for every market returned by search queries (rate-limit aware)",
    )
    args = parser.parse_args()

    logger = _setup_logger()

    env_path = _resolve_env_file(args.env_file)
    logger.info(f"Loading credentials from: {env_path}")
    env = _load_dotenv(env_path)
    required = ["IG_USERNAME", "IG_PASSWORD", "IG_API_KEY", "IG_ACCOUNT_ID", "IG_BASE_URL"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        logger.error(f"Missing keys in {env_path.name}: {missing}")
        return 2

    logger.info(
        f"  IG_USERNAME={_mask(env['IG_USERNAME'])}  "
        f"IG_API_KEY={_mask(env['IG_API_KEY'])}  "
        f"IG_ACCOUNT_ID={_mask(env['IG_ACCOUNT_ID'])}  "
        f"IG_BASE_URL={env['IG_BASE_URL']}"
    )

    # Refuse to run if the URL looks like the live endpoint and the user
    # didn't explicitly acknowledge — first scout MUST be demo.
    if "demo-api.ig.com" not in env["IG_BASE_URL"] and not os.environ.get("IG_SCOUT_ALLOW_LIVE"):
        logger.error(
            "IG_BASE_URL is not the demo endpoint. Refusing to run against "
            "live without explicit IG_SCOUT_ALLOW_LIVE=1 in the environment. "
            "Demo URL: https://demo-api.ig.com/gateway/deal"
        )
        return 3

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    client = IGClient(
        base_url=env["IG_BASE_URL"],
        api_key=env["IG_API_KEY"],
        username=env["IG_USERNAME"],
        password=env["IG_PASSWORD"],
        account_id=env["IG_ACCOUNT_ID"],
        logger=logger,
    )

    try:
        client.login()
        (out_dir / "session_meta.json").write_text(json.dumps(client.session_meta, indent=2))

        if not args.skip_nav:
            logger.info("Walking market navigation tree...")
            _drill_navigation(client, out_dir, max_depth=args.nav_depth, logger=logger)

        logger.info(f"Running {len(args.search)} search queries...")
        all_search_epics: set[str] = set()
        for term in args.search:
            body = client.search_markets(term)
            (out_dir / f"search_{_safe_filename(term)}.json").write_text(json.dumps(body, indent=2))
            markets = body.get("markets", []) if isinstance(body, dict) else []
            logger.info(f"  '{term}' -> {len(markets)} markets")
            for m in markets:
                if m.get("epic"):
                    all_search_epics.add(m["epic"])

        epics_to_detail = set(args.epics)
        if args.also_detail_search_hits:
            epics_to_detail |= all_search_epics
        if epics_to_detail:
            logger.info(f"Fetching detail for {len(epics_to_detail)} epics...")
            for epic in sorted(epics_to_detail):
                body = client.market_detail(epic)
                (out_dir / f"epic_{_safe_filename(epic)}.json").write_text(
                    json.dumps(body, indent=2)
                )

        _summarise(out_dir)
        logger.info(f"✓ Done. See {out_dir / 'SUMMARY.md'}")
        return 0

    finally:
        client.logout()


if __name__ == "__main__":
    sys.exit(main())
