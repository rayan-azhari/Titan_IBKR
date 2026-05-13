"""ig_map_underlyings.py — Offline analyser for IG scout output.

Reads the cached search_*.json and epic_*.json files from the latest
.tmp/ig_catalog/ run and produces a markdown mapping of:

    Strategy underlying (CSPX, VUSD, EIMI, IHYU, IHYG)
        -> Candidate IG epics, with key dealing parameters

This is OFFLINE — no API calls. Run any time after ig_scout_catalog.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / ".tmp" / "ig_catalog"

# What we want to find on IG, ranked by preference. Each entry lists the
# search-result files to mine and a "preferred substring" used to rank the
# hits (substring-in-instrumentName, case-insensitive). The empty preferred
# list means "list everything; you decide".
TARGETS: dict[str, dict] = {
    "CSPX (iShares Core S&P 500 UCITS, USD)": {
        "search_files": ["search_CSPX.json", "search_US_500.json", "search_S_P_500.json"],
        "preferred": ["s&p 500", "us 500", "ishares core s&p 500", "cspx"],
        "exclude": ["option", "spread bet (futures)", "vix"],
    },
    "VUSD (Vanguard S&P 500 UCITS, USD line)": {
        "search_files": ["search_VUSA.json", "search_US_500.json", "search_S_P_500.json"],
        "preferred": ["vanguard s&p 500", "vusa", "vusd"],
        "exclude": ["option"],
    },
    "EIMI (iShares Core MSCI EM IMI UCITS)": {
        "search_files": [
            "search_EIMI.json",
            "search_MSCI_Emerging.json",
            "search_Emerging_Markets.json",
        ],
        "preferred": ["msci em imi", "core msci em", "msci emerging", "eimi", "ishares msci em"],
        "exclude": ["option", "leveraged", "short"],
    },
    "IHYU (iShares $ HY Corp Bond UCITS)": {
        "search_files": ["search_IHYU.json", "search_HYG.json", "search_High_Yield.json"],
        "preferred": ["ihyu", "$ high yield", "usd high yield", "hyg", "ishares iboxx high yield"],
        "exclude": ["short", "leveraged", "fallen angels", "option"],
    },
    "IHYG (iShares € HY Corp Bond UCITS)": {
        "search_files": ["search_IHYG.json", "search_High_Yield.json"],
        "preferred": ["ihyg", "€ high yield", "euro high yield", "eur high yield"],
        "exclude": ["short", "leveraged", "fallen angels", "option"],
    },
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _rank_market(market: dict, preferred: list[str], exclude: list[str]) -> tuple[int, int, str]:
    """Lower tuple sorts first. (exclude_hit, -preferred_hits, name)."""
    name = (market.get("instrumentName") or "").lower()
    excluded = 1 if any(e in name for e in exclude) else 0
    pref_hits = sum(1 for p in preferred if p in name)
    return (excluded, -pref_hits, name)


def _market_snapshot_line(m: dict) -> str:
    return (
        f"`{m.get('epic')}` — **{m.get('instrumentName')}**  "
        f"type=`{m.get('instrumentType')}`  "
        f"expiry=`{m.get('expiry')}`  "
        f"bid/ask=`{m.get('bid')}/{m.get('offer')}`  "
        f"market=`{m.get('marketStatus')}`"
    )


def _epic_detail_block(epic: str, out_dir: Path) -> list[str]:
    """Read epic_<epic>.json and emit detail lines if the file is present
    and contains a full payload (some 403s wrote 106-byte error stubs)."""
    p = out_dir / f"epic_{epic.replace('/', '_')}.json"
    if not p.exists():
        return [f"      _no /markets detail captured (epic file missing: {p.name})_"]
    body = _load_json(p)
    if "instrument" not in body:
        err = body.get("errorCode") or body.get("_status")
        return [f"      _detail call failed: {err}_"]
    inst = body.get("instrument", {})
    rules = body.get("dealingRules", {})
    snap = body.get("snapshot", {})
    min_deal = rules.get("minDealSize", {}) or {}
    max_deal = rules.get("maxDealSize", {}) or {}
    min_stop_dist = rules.get("minNormalStopOrLimitDistance", {}) or {}
    min_step = rules.get("minStepDistance", {}) or {}
    slip = inst.get("slippageFactor", {}) or {}
    expiry_details = inst.get("expiryDetails") or {}
    rollover = inst.get("rolloverDetails") or {}
    currencies = inst.get("currencies") or []
    cur_codes = ",".join(c.get("code", "?") for c in currencies)
    # IG returns marginFactor as either % or bps depending on instrument
    mf = inst.get("marginFactor")
    mf_unit = inst.get("marginFactorUnit")
    # `valueOfOneContract` is the £/point or $/point bet unit (key for sizing)
    val1c = inst.get("valueOfOneContract")
    lot = inst.get("lotSize")
    one_pip = inst.get("onePipMeans")
    sb = inst.get("type")  # SHARES / INDICES / BUNGEE_CAPPED / etc.

    lines = [
        f"      - instrumentType: `{sb}`",
        f"      - currency / unit: `{cur_codes}`  valueOfOneContract: `{val1c}`  "
        f"lotSize: `{lot}`  onePipMeans: `{one_pip}`",
        f"      - marginFactor: `{mf}{mf_unit if mf_unit else ''}` "
        f"slippageFactor: `{slip.get('value')}{slip.get('unit', '')}`",
        f"      - min/max deal: `{min_deal.get('value')}{min_deal.get('unit', '')}` / "
        f"`{max_deal.get('value')}{max_deal.get('unit', '')}`",
        f"      - min stop dist: `{min_stop_dist.get('value')}{min_stop_dist.get('unit', '')}` "
        f"min step: `{min_step.get('value')}{min_step.get('unit', '')}`",
        f"      - snapshot bid/ask: `{snap.get('bid')}/{snap.get('offer')}` "
        f"updated=`{snap.get('updateTime')}` status=`{snap.get('marketStatus')}`",
        f"      - expiry: `{inst.get('expiry')}` "
        f"lastDealingDate: `{expiry_details.get('lastDealingDate')}` "
        f"settlementInfo: `{expiry_details.get('settlementInfo')}`",
        f"      - controlled-risk allowed: `{inst.get('controlledRiskAllowed')}` "
        f"forceOpenAllowed: `{inst.get('forceOpenAllowed')}` "
        f"streaming: `{inst.get('streamingPricesAvailable')}`",
    ]
    if rollover:
        lines.append(
            f"      - rolloverDetails: lastRolloverTime=`{rollover.get('lastRolloverTime')}` "
            f"rolloverInfo=`{rollover.get('rolloverInfo')}`"
        )
    return lines


def main() -> int:
    if not OUT_ROOT.exists():
        print(f"No scout output at {OUT_ROOT}. Run scripts/ig_scout_catalog.py first.")
        return 1
    runs = sorted(p for p in OUT_ROOT.glob("*") if p.is_dir())
    if not runs:
        print(f"No runs under {OUT_ROOT}.")
        return 1
    latest = runs[-1]
    print(f"# IG Underlying Mapping — `{latest.name}`\n")

    # Session header
    meta = _load_json(latest / "session_meta.json")
    if meta:
        print(
            f"- session: account=`{meta.get('currentAccountId')}` "
            f"currency=`{meta.get('currencyIsoCode')}` "
            f"lightstreamer=`{meta.get('lightstreamerEndpoint')}` "
            f"timezoneOffset=`{meta.get('timezoneOffset')}`\n"
        )

    # Top-N per target
    TOP_N = 10
    for label, cfg in TARGETS.items():
        print(f"## {label}\n")
        # Collect & dedupe markets across all search files
        seen_epics: dict[str, dict] = {}
        for fname in cfg["search_files"]:
            body = _load_json(latest / fname)
            for m in body.get("markets") or []:
                e = m.get("epic")
                if e and e not in seen_epics:
                    seen_epics[e] = m

        if not seen_epics:
            print(f"_No hits across {cfg['search_files']}_\n")
            continue

        def _key(m: dict, c: dict = cfg) -> tuple[int, int, str]:
            return _rank_market(m, c["preferred"], c["exclude"])

        ranked = sorted(seen_epics.values(), key=_key)
        # filter out excluded (those with rank tuple starting with 1) at print time
        included = [m for m in ranked if _rank_market(m, cfg["preferred"], cfg["exclude"])[0] == 0]
        excluded = [m for m in ranked if _rank_market(m, cfg["preferred"], cfg["exclude"])[0] == 1]

        print(
            f"**{len(included)} included candidates** "
            f"(plus {len(excluded)} filtered — options/leveraged/short).\n"
        )
        for i, m in enumerate(included[:TOP_N], 1):
            print(f"{i}. {_market_snapshot_line(m)}")
            for ln in _epic_detail_block(m.get("epic", ""), latest):
                print(ln)
        if len(included) > TOP_N:
            print(f"\n_(... {len(included) - TOP_N} more candidates omitted)_\n")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
