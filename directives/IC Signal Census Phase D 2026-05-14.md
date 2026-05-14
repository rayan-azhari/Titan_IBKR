# IC Signal Census — Phase D pre-registration (paid feeds, future scope)

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect
**Status:** **PRE-REGISTRATION — NO EXECUTION.** Pure documentation of
the next-phase universe expansion. Charges require explicit user
$-approval before any feed contract is taken.
**Parent:** `directives/IC Signal Census Phase C 2026-05-13.md` §6.7 item 4.

---

## 0. Why this exists

Phases A + B + C surfaced exactly two TIER_B candidates and many
sample-limited near-misses. The remaining data-gated mechanisms are:

* **Eurex equity-futures intraday** (FESX, FDAX, FMDX) — Databento's
  `XEUR.EOBI` only carries 2 months of history. Useless for IC research
  until either (a) the EOBI series accumulates more history (~2027-2028
  before reaching 2y minimum) or (b) we buy a different paid feed
  (Refinitiv, Bloomberg, ICE Connect, or a Norgate / Pinnacle continuous
  futures series).
* **CFE VIX futures intraday** (VX1, VX2 continuous) — not on Databento's
  public catalogue. Requires a paid CFE feed (Eze / CBOE Direct / Pinnacle).
* **SPX breadth panel** — needs the panel of all S&P 500 constituents'
  D-frequency OHLCV plus a constituent-membership history. Polygon.io,
  Norgate, or Sharadar all sell this.
* **OIS-Fed-Funds spread** — needs the OIS rate curve and Fed Funds
  effective rate at daily granularity. FRED + a paid OIS feed (TraditionData
  or Bloomberg).
* **Cross-region lead-lag with proper futures basis** — would benefit
  from the Eurex futures intraday once it lands.

Phase D pre-registers the universe expansion and signal classes that
become testable once the data is acquired. This directive does NOT
trigger any feed contracts or charges.

---

## 1. Pre-registered universe additions

### 1.1 Eurex equity-futures (paid feed required)

| Local | Symbol | Source candidates | Cost estimate |
|---|---|---|---|
| FESX_D, FESX_H1 | EURO STOXX 50 future continuous | Norgate, Pinnacle, Refinitiv, IB historical | $200-500 one-shot |
| FDAX_D, FDAX_H1 | DAX 40 future continuous | same | $200-500 |
| FMDX_D, FMDX_H1 | MDAX (mid-cap) future continuous | same — Pinnacle has it | $200-500 |
| FTSE_FUT_D, _H1 | FTSE 100 future continuous (LIFFE) | same | $200-500 |

### 1.2 CFE VIX futures (paid feed required)

| Local | Symbol | Source | Cost estimate |
|---|---|---|---|
| VX1_D, VX1_H1 | VIX future front-month continuous | CBOE Direct ($), Pinnacle, Norgate | $100-300 |
| VX2_D, VX2_H1 | VIX future M2 continuous | same | $100-300 |

### 1.3 SPX breadth panel (paid feed required)

| Local | Symbol | Source | Cost estimate |
|---|---|---|---|
| SPX_breadth | Daily count: number of SPX constituents above their 200-day SMA | Polygon.io ($) or Norgate | Polygon $99/mo single user; Norgate ~$240/yr |
| SPX_members | Historical constituent membership panel | Polygon, Norgate, Sharadar | bundled with the above |

### 1.4 Macro flow (FRED free + paid OIS)

| Local | Source |
|---|---|
| OIS_FF_spread | FRED (free) for FedFunds; OIS via Bloomberg / TraditionData ($) |
| TIPS_real_yield_proxy | Already have via TIP/TLT ratio from Phase A (free) — no new buy |

---

## 2. Pre-registered signal classes (Phase D-only)

All H1 unless noted. Inherits all gates from Phase C unchanged.

### 2.1 Cross-region lead-lag (futures-vs-futures)

| Signal | Externals | Param | Cells |
|---|---|---|---|
| `mes_to_fesx_h1` | `[MES]` | window (H1 bars) | `1, 6, 24` |
| `mes_to_ftse_fut_h1` | `[MES]` | window | `1, 6, 24` |
| `mes_to_fdax_h1` | `[MES]` | window | `1, 6, 24` |

Mechanism: with both sides as actual futures (rather than Phase B's
spot proxies), the lead-lag mechanism is testable without spot-vs-future
basis noise. Migrate.md's §2.8 a priori was "yesterday's MES sign
predicts today's FESX". Phase B with index spot found the OPPOSITE
sign (mean-reversion). Phase D re-tests on futures.

### 2.2 VIX-futures basis

| Signal | Externals | Param | Cells |
|---|---|---|---|
| `vx_basis_z` | `[VX1, VX2]` | smoothing | `1, 5, 21` |

Mechanism: `(VX1 - VX2) / VX2` is the canonical contango/backwardation
signal that the Phase B spot-VIX ratios were proxying. The actual
basis carries roll-yield information that the spot ratios miss.

### 2.3 Breadth-conditional sizing

| Signal | Externals | Param | Cells |
|---|---|---|---|
| `breadth_above_200d` | `[SPX_breadth]` | smoothing (D bars) | `1, 21, 60` |
| `breadth_change` | `[SPX_breadth]` | window | `1, 21, 60` |

Mechanism: when fewer SPX names trade above their 200d SMA, market is
weakening before the index itself falls. Classic breadth divergence.

### 2.4 OIS-Fed-Funds dislocation

| Signal | Externals | Param | Cells |
|---|---|---|---|
| `ois_ff_z` | `[OIS_FF_spread]` | lookback | `21, 60, 252` |

Mechanism: persistently elevated OIS-FF spread indicates funding stress;
predicts risk-off in equity markets at 1-21 day horizon.

---

## 3. Gate overrides

Inherits from Phase C unchanged. **Likely consequence at Phase D N**:
the combined Phase A + B + C + D cell count likely exceeds 25,000.
That engages `dsr_t_floor_combined = 5.0` (tighter than the current 4.5).
This directive pre-commits to that tighter floor.

---

## 4. Pre-charge protocol (V3.1 + budget discipline)

Identical to the Phase C protocol that worked well there:

1. This directive on `main` first.
2. For each paid feed under consideration, compute itemised cost via
   that vendor's metadata API (Databento `metadata.get_cost`,
   Polygon `/v2/aggs/...` rate cards, etc.) or vendor quote.
3. Print itemised cost table.
4. **Pause for explicit user $-approval before any contract or charge.**
5. After approval, run the download with `--confirm` gate.
6. Run scan, append result log.

The Phase C cost surprise ($170-290 directive estimate → $2.76 actual)
illustrates why the per-symbol `metadata.get_cost` check matters.
Phase D's vendor mix is heterogeneous (Databento + Polygon + Norgate +
Bloomberg / TraditionData) so cost surprises in either direction are
possible.

---

## 5. Estimated total

| Source | Approx cost |
|---|---|
| Eurex futures continuous (4 instruments × D + H1) | $800-2000 one-shot |
| CFE VIX futures (2 instruments × D + H1) | $200-600 one-shot |
| SPX breadth panel | $99/mo (Polygon) or $240/yr (Norgate) — subscription |
| OIS feed | $50-200/mo (Bloomberg / TraditionData) or free via FRED OIS series — verify before buy |
| **One-shot total** | **$1,000-2,500** |
| **Subscription run-rate** | **~$10-25/mo if subscriptions sustained** |

Substantially more than Phase C's $5.50. Justifies the explicit pre-charge
approval step.

---

## 6. What this directive does NOT propose

- **Does not authorise any spend.** This is V3.1 pre-registration only.
- **Does not run any scan.** Each feed acquisition triggers a separate
  scan invocation with the result-log appended to this directive's §7.
- **Does not commit to the deferred Phase B items.** If after seeing
  the cost itemisation some items are cancelled, that's recorded in
  §7 as part of the as-built universe — same V3.6 pattern as Phase C §1.5.

---

## 7. Result log

To be appended as each feed lands and each Phase D sub-scan runs. Each
sub-scan documents its own as-built universe + outcome inline, in the
order of arrival.

> _Pending vendor selection and $-approval — appended below after first feed lands._

---

## 8. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial Phase D pre-registration. No execution. |
