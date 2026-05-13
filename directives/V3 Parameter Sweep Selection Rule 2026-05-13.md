# V3 Layered-Defense Parameter Sweep — Plateau-Seeking Selection Rule

**Created:** 2026-05-13
**Status:** Pre-committed BEFORE any sweep results are observed.
**Operator decision:** "find an optimal REGION of parameters, not a single
point. Reject configurations where small perturbations change the outcome
significantly."

This directive is committed to git before the sweep runs so the
selection logic cannot be retro-fitted to favour a specific cell.

---

## 1. Why plateau-seeking

A parameter configuration that produces excellent MC results but
**only at that exact point** is almost certainly overfit. The real
distribution of returns we care about — live trading — will not respect
the precise threshold we picked. If `dd_kill=0.15` gives `P(MaxDD>50%)
= 0.5%` but `dd_kill=0.13` gives 30% and `dd_kill=0.17` gives 25%, the
0.5% is **noise**, not a real safety margin.

The audit's A5 finding (multiple-testing without DSR) was the
quantitative form of this concern. Plateau-seeking is the structural
form: we want the result to be **robust** to parameter perturbations,
not optimised to a single grid point.

This is also Carver's "fit the model, not the noise" principle.

---

## 2. Selection rule (mechanical, applied AFTER sweep completes)

For any candidate cell `c` with parameters `(θ₁, θ₂, …, θₖ)` to be
deemed deployable, ALL of the following must hold:

### 2.1 Hard gates (each cell must individually pass)

1. **MC `P(MaxDD > 50%)` < 1%** — the RoR deployment gate from V3
   design directive §10.
2. **Sharpe CI95 lo > 0** — bootstrap-significant positive Sharpe.
3. **MC P(CAGR < 0) over 19 years < 10%** — strategy must reliably
   compound positively across the resampled futures.

### 2.2 Plateau gates (the cell AND its grid neighbours must pass)

A cell `c` is a **plateau cell** iff:

4. **All grid-immediate-neighbours of `c`** (cells differing in exactly
   one parameter by exactly one grid step) ALSO pass the hard gates.
5. **Across the ±1-step neighbourhood**, the headline metric
   (`mc_calmar_p025`) varies by less than 30% of the cell's own value.
   Math: `max_over_neighbours(|metric(neighbour) − metric(c)|) / |metric(c)| < 0.30`.

Cells without all neighbours present in the grid (edge cells) are
**INELIGIBLE** for selection — we don't trust extrapolation outside
the swept range.

### 2.3 Final selection

Among plateau cells, pick the one with the **highest mean
`mc_calmar_p025`** over the cell and its ±1-step neighbours (rewards
the cell whose neighbourhood is uniformly good, not the local maximum).

Tie-break by parsimony: lower `dd_kill` (less reliant on the failsafe),
lower `score_enter_threshold` (less restrictive), simpler config wins.

---

## 3. Sweep design (committed prior to running)

Modest grid to keep DSR threshold manageable:

| Layer | Parameter | Grid | Levels |
|---|---|---|---:|
| C | `dd_kill` | {0.10, 0.12, 0.15, 0.20, 0.25} | 5 |
| C | `dd_re_entry_score` | {0.55, 0.70, 0.80} | 3 |
| B (optional) | `dd_velocity_threshold` | {-0.10, -0.05, -0.02} or OFF | 4 |
| A (optional) | `score_enter_threshold` | {0.50, 0.60, 0.70, 0.80} | 4 |

Strategy is committed at L=2 only (Phase 5 finding: L=2 dominates higher
leverage on Calmar CI lo).

**Total grid: 5 × 3 × 4 × 4 = 240 cells**. With 100 MC paths per cell
and ~10s per path (HMM refit dominates), that's ~6.6 hours of compute.

The DSR correction at N=240 will tighten the Sharpe CI gate — but the
sweep is wide enough that a true plateau will survive even after DSR.

---

## 4. Adaptive sweep (executed in this order)

The full grid is 240 cells. Instead of running it all blindly, do a
hierarchical sweep that exits early if the simpler config passes:

### Phase α — single-layer sweeps (5 + 3 = 8 cells)
Run only Layer C sweeps (5 dd_kill × 3 dd_re_entry_score = 15 cells)
with Layers A and B at their pre-registered values. Apply the plateau
rule to that subspace.

**Exit condition:** if at least one Layer-C-only plateau cell passes
all hard gates, declare it a candidate champion. Move to Phase γ.

**Otherwise:** advance to Phase β.

### Phase β — two-layer sweep (15 × 3 = 45 cells)
Add Layer B sweep (`dd_velocity_threshold`) to Layer C. Apply plateau
rule.

**Exit condition:** if a Layer C+B plateau cell passes, declare it a
candidate champion. Move to Phase γ.

**Otherwise:** advance to Phase γ.

### Phase γ — full grid (240 cells)
Add Layer A sweep. Final plateau evaluation across the full hypothesis
space.

---

## 5. Reporting requirements

For every sweep, save:

- Full results CSV with all 240 cells (even those that failed).
- A 2D heatmap of `mc_p_dd_gt_50` across `(dd_kill, score_enter_threshold)`
  for each fixed B/C-row value — visualises plateau structure.
- A table of all cells that passed the hard gates (whether or not they
  passed the plateau gates).
- A table of the plateau-selected candidate(s).
- The final-selected cell with full diagnostics.

---

## 6. Sanity rules to enforce

To prevent gaming the rule retroactively:

- **No additional gates** added after seeing results. The five rules in
  §2 are it.
- **No threshold tweaks** to admit a specific cell. If no cell passes,
  the answer is "no plateau exists in this grid; expand the search or
  abandon the layer".
- **DSR correction** applied to the final-selected cell's Sharpe at
  N=240 trials. If DSR-adjusted significance fails, the cell is not
  deployable.

---

## 7. Where this directive lives

This is a commit to the V3 work in progress, not the deployed live
strategy. Live V3 is still under audit-remediation embargo until a
plateau cell passes ALL gates (hard + plateau + DSR).

If no plateau cell exists, the verdict is: **V3 fails — go back to V2
(currently deployed) and improve V2 instead**, or abandon the layered-
defense approach altogether and reconsider the architecture from
first principles again.
