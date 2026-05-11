# Rehydration Bug Post-Mortem — 2026-05-11

## What happened

Live IBKR paper account had **238 VUSD shares + 210 EIMI shares** ($44,782
USD notional total) when the strategies' internal accounting believed they
held only **23 VUSD + 18 EIMI** (today's session entries). The
**~$40,500 of phantom overhang** accumulated over ~3 prior container
restarts.

## Root cause

`BondGoldStrategy._rehydrate_position_from_broker` and
`MRAUDJPYStrategy._rehydrate_position_from_broker` both filter the cache
for positions whose `strategy_id` matches the current instance:

```python
positions = self.cache.positions(
    instrument_id=self.instrument_id,
    strategy_id=self.id,
)
```

But on every container start, NautilusTrader's `ExecEngine` reconciles
existing broker positions and tags them with `position_id=*-EXTERNAL`
(no strategy_id, because the prior session's strategy_id was lost
on shutdown). The filter excludes EXTERNAL positions, so the strategy
thinks it's flat and either:

- (BondGoldStrategy) submits a fresh BUY on top of the existing inventory
- (MRAUDJPYStrategy) leaves the inventory unmanaged and refuses to enter
  any new tier

Either way, the strategy's accounting drifts away from the broker's truth.

## Why the filter exists

Commit `b4ff9fc fix(strategy): isolate position rehydration per
strategy_id (CSPX→VUSA pivot)` tightened the filter to prevent a newly
deployed strategy from adopting positions left behind by a decommissioned
strategy that was historically targeting the same instrument (specifically
the CSPX → VUSA pivot in May 2026).

The fix was over-restrictive: it solved the cross-strategy attribution
problem but introduced this restart-amnesia problem.

## The fix

Adopt EXTERNAL positions for our instrument in addition to strategy-tagged
ones. Each live `bond_equity_*` strategy has a unique `instrument_id`
(CSPX, VUSD, EIMI — all different) so adoption is unambiguous.

For the historic CSPX→VUSA pivot scenario, the proper procedure is to
flatten the dead strategy's positions BEFORE deploying the replacement
— the runbook should handle this rather than the code defending against
the operator skipping it.

## Cost impact

Per restart × per instrument × $4 IBKR minimum commission =
**$8+ pure-waste commissions per restart**. Today (May 11) saw at least
3 restarts × 2 instruments = ~$24 in wasted commissions, with the
phantom inventory growing each time.

## One-off remediation

Before the code fix takes effect, the existing $40,500 phantom inventory
needs to be flattened. Use the same surgical-close pattern as
[scripts/close_cspx_orphan.py](../scripts/close_cspx_orphan.py) but
generalized for VUSD and EIMI overhang.

## Long-term

Once the code fix is deployed:

1. On next restart, the strategy will adopt the entire broker position
   as its own.
2. Any subsequent exits (signal flip, NY-close hard-flat for mr_audjpy,
   z<=threshold for bond_equity) will close the FULL position cleanly.
3. Subsequent entries will be sized correctly without phantom layering.
