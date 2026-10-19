# T400: rigour for the certified rare-event bound (mass bounds + exact certificate checks)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · lane: [event_cone README](../event_cone/README.md)
(branch `origin/claude/t373-event-cone`, T373/T390) · lead: T398 · GPU engine: T399

## Question
The certified bound on P(E(t0)) is P(E) <= sum over surviving step-10 regions of their event mass + the loss from
the normalisation. Here E(t0) = {pick = position of x_(3), g23 = x_(2) - x_(3) >= t0}, and the pruned cells are
proved event-free. Two parts are not yet rigorous:
- (a) the event mass of a surviving region, a Gaussian mass of a polyhedral cone cut by g23 >= t0, needs a
  deterministic rigorous upper bound;
- (b) floating-point LP pruning needs exact verification.

(c) The normalisation loss must be proved.

## Pre-registration (2026-10-19, written and pushed before any tightness or speed number is scored)

### Definitions
- Region = (tri, gates).
  - gates is an int array of shape (10,16): 1 = on, 0 = off, -1 = free (an unsplit neuron).
  - Every constraint is rebuilt in **exact rational arithmetic** (python-flint `fmpq`) from the float64 weights of
    `m16/common.py`, which are exact dyadic rationals. Float rows from any engine are never trusted.
  - Gate rows are closed half-spaces (z >= 0 if on, z <= 0 if off).
  - The event in a full region is {Wo A10 x: logit_p3 >= logit_j for all j}, inside the ordering cone
    x_p1 >= x_p2 >= x_p3 >= every low.
  - Ties and gate boundaries have measure zero, so closed regions cover R^10 up to a null set.
- Normalisation N (T390's, DELTA = 0.1): g23 >= 0.1 ||x||_inf.

### (c) The normalisation loses < 1e-49 (proof)
- Points of E(t0) outside N satisfy t0 <= g23 < 0.1 ||x||_inf, so ||x||_inf > 10 t0 = 15 at t0 = 1.5.
- Hence P(E \ N) <= P(||x||_inf > 15) <= 10 · 2 Φ(-15) <= 20 φ(15)/15 (Mills ratio).
  - φ(15)/15 = e^{-112.5} / (15 √(2π)).
  - So the bound is 20 φ(15)/15 = 7.3e-50 < 1e-49. `rigour.py` evaluates it in arb ball arithmetic and prints it.
- The bound holds for any t0 >= 1.5. For a general t0 it is 20 φ(10 t0)/(10 t0).
- Every pruned cell is proved empty on N (part b). Every survivor's mass is bounded on N (part a). So:
  **P(E) <= Σ_survivors ub(region) + 7.3e-50**.
  - No double counting: regions are disjoint up to null sets.
  - Nothing is missed: pruned cells ∪ survivors cover the event cone.
  - The cover is the lead's obligation; the verifier checks only each certificate.

### (a) Mass upper bound, method M1 ("exact slice triangulation")
- **Slice.** Every event point of a region has g23 > 0. Write x = z · m(y) with z = g23 and m(y) the point of the
  slice g23 = 1.
  - y lists the 9 coordinates other than x_p2, and x_p2 = 1 + y_p3. The Jacobian is exactly z^9.
- Integrating z from t0 to ∞ in closed form gives:

  γ(C ∩ {g23 >= t0} ∩ N) = (12/π^5) ∫_Q |m(y)|^-10 S10(t0 |m(y)|) dy,

  where:
  - S10(r) = P(χ_10 >= r) = e^{-r²/2} Σ_{k=0..4} (r²/2)^k / k!;
  - Q is the 9-D polytope {y : m(y) in C, |m_j(y)| <= 10} (N becomes the box ||m||_inf <= 10).
  - The integrand is a decreasing function F(|m(y)|²).
- **Exact polytope.**
  - Q's vertices come from exact (fmpq) pivoting: breadth-first search over the polytope graph, from a vertex found
    by a float LP and re-solved exactly. The graph is connected (Balinski), so BFS closure gives the complete
    vertex set.
  - Incidences are exact.
  - The triangulation is the exact pulling triangulation (recursion over faces, from exact incidences and exact
    ranks). Simplex volumes are |det|/9! in exact arithmetic.
  - An empty or lower-dimensional Q gives mass 0. "Empty" needs a certificate (part b's safe Farkas check).
- **Per-simplex bounds.**
  - The sup of F is at a rigorous lower bound on min |m|² over the simplex. That bound is a separating
    hyperplane through a float-QP projection point, evaluated exactly over the vertices.
  - The inf of F is at the max of |m|² over the vertices; this is exact, since |m|² is convex.
  - F is evaluated in arb ball arithmetic.
  - ub = Σ vol · sup F and lb = Σ vol · inf F, so the true mass lies in [lb, ub] rigorously.
  - Simplices with a large sup/inf ratio are bisected at the midpoint of their longest edge (exact), until the
    total ub/lb <= 1.05 or a budget of 20,000 simplices per region is used.
- **Cheap first pass M0 (enclosing box).**
  - Take Q's bounding box from 18 LPs, with each bound made rigorous by the safe dual bound below.
  - M0 = vol(box) · F(min |m|² over the box), which is separable and exact.
  - This is a rigorous but loose superset bound.

### (b) Certificate verifier
- The verifier proves cell ∩ N ∩ {argmax_10 = p3} = ∅ for a step-k cell.
- **Safe dual bound** (Neumaier–Shcherbina 2004):
  - Setting: an LP min c·v subject to A v >= b and lo <= v <= hi, with any float multipliers y >= 0.
  - The bound: min >= y·b + Σ_i min(r_i lo_i, r_i hi_i), where r = c - Aᵀy.
  - Every term is computed with rigorous error bounds: exact fmpq, or float64 with a γ_n error term, rounded
    outward.
  - Infeasibility is the case c = 0 with a positive bound. It uses phase-1 duals, so no Farkas ray from the solver
    is needed.
  - All variables are boxed: x in [-1,1]^10, and h in [0, u] with u rigorously certified.
- **Certificate kind 'lp'** (T390's relax.py relaxation; the cell's rows are recomputed with rigorous error bounds):
  - The verifier rebuilds the triangle relaxation step by step.
  - Each pre-activation bound l, u is the safe dual bound of its LP: the engine's duals if supplied, else
    re-solved by HiGHS. Each is rounded outward.
  - The final event LP must have a positive safe phase-1 bound.
- **Certificate kind 'crown':** the same linear-bound pass as T398's engine, re-run in outward-rounded arithmetic
  with the engine's free parameters. Its definition is agreed with T398 and added here before it is scored.

### Acceptance tests (pass/fail, fixed now)
- **A1 exactness on known cones.**
  - Cases:
    - the orthant-type cone {x_p2 >= x_p3 >= all others} ∩ g23 >= t0;
    - the full ordering cone x_1 >= ... >= x_10 (mass 1/10! at t0 = 0, through a t0 -> 0 limit check of the
      formula);
    - random simplicial cones.
  - The reference is high-precision quadrature or 1e8-sample MC.
  - Pass iff truth ∈ [lb, ub] in every case (MC truth with ±4σ).
- **A2 soundness and tightness on real regions.** The test set is the step-10 event regions of (1,8,2) harvested by
  `harvest.py` (line walk, seed 400 + ...; 993 regions from 2e5 lines). The MC check is independent code: the forward
  pass `common.run`, not my constraint algebra.
  - The MC draws y uniformly in M0's box and z from the exact radial law, then tests membership by running the
    network.
  - Pass iff ub >= MC - 4σ on every region with ≥ 30 MC hits, AND lb <= MC + 4σ on every such region.
  - Tightness is reported as ub/MC and ub/lb (median, p90, max) for M1 and M0.
  - **Target, not a pass condition:** M1 median ub/lb <= 1.1 within the budget.
- **A3 cost.** CPU-seconds per region for M1 and M0 (median, p90), on the same test set.
- **B1 soundness.** No certificate may verify on a cell that contains a known event point (the line-walk event
  points of (1,8,2) and their prefixes). Also, deliberately falsified certificates (claimed bounds tightened by 10%
  of their width) must not verify unless they still verify honestly.
  - Pass iff 0 false verifications.
- **B2 coverage.** Of T390's float-pruned cells on (1,8,2) at k = 1..5 (regenerated with relax.py's
  rule), report the fraction the verifier confirms.
  - **Target 99%.** Unconfirmed cells are not errors; they become survivors.
- **B3 speed.** CPU-ms per certificate (median, p90), with and without supplied duals.

## Deviations from the pre-registration (declared before the tables below were scored)
1. **M1 is too expensive to score.**
   - The exact slice triangulation of a k = 10 event region has 750-2,100 vertices.
   - Its pulling triangulation has 3.3e5-2.4e6 simplices: 0.7-1.8e6 at NORM 5, and 10-40 s just to triangulate.
   - That is before any refinement.
   - So M1 is kept as code (`mass_bounds`) but **not scored**. Measured instead:
     - M2 (orthant/tilt bound on a row subset, `orthant_bound`);
     - M3 (Lawrence volume distribution, `mass_lawrence`);
     - M4 (M3 on an adaptive exact partition, `mass_split`);
     - M5 (`mass_cell`): an exact marginal reduction for prefix cells, added after T398 asked for a cheap
       call on unfinished cells (k < 10, event ignored) and moved the target t0 to 2.25-2.5.
2. **The MC reference changed.**
   - k = 10 regions: importance sampling at the float dominating point (`mc.py`).
   - k = 4 cells: plain MC over all cells at once (`mc_cells.py`).
   - Both decide membership by the forward pass.
3. **The verifier sets were adapted to T398's benchmark.** B1 uses T398's 47 event-feasible k = 4 survivors
   (probe_best > 0) instead of line-walk prefixes. B2 uses the k = 5 children of the first 46 benchmark
   survivors, classified by T398's float relax2.prune_stage.
4. **DELTA is a parameter.** T398 uses DELTA = 1/3 at t0 = 2.5. The part-(c) bound is then
   20 Φ(-t0/DELTA) = 20 Φ(-7.5) = 6.4e-13.

## Pre-registration 2 (2026-10-19 ~08:30, after the DRI's call; pushed before any M5-tangent or probabilistic number is scored)

### (1) Headline, deterministic: M5 with exact per-simplex integrals ("M5t")
- **Method.** M5 as before, with each (simplex, z-bin) bound replaced as follows.
  - The exact identity s = s_c + τ + |m - m_c|², with τ affine, is integrated exactly over the simplex
    (divided differences / Hermite–Genocchi, in arb).
  - The upper bound drops |m - m_c|². The lower bound replaces it by its vertex maximum.
  - The low-order factor Φ(z y_p3)^m is bounded by the tangent (upper) or the chord (lower) of the concave
    log Φ, which makes it exp-linear in y as well.
- **Test set.** The k = 4 benchmark cells of (1,8,2), scored against the plain MC of `mc_cells.py`:
  - t0 = 2.25: the existing 2e8-draw MC, cells with >= 30 hits;
  - t0 = 2.5: a new MC with 1e9 draws, same rule.
- **Targets.**
  - Pass line: median ub/MC <= 3 and p90 <= 10, at <= 30 CPU-s per cell (median).
  - Soundness: lb <= MC + 4 s.e. and ub >= MC - 4 s.e. on every scored cell.
- **Deliverable for T398 (i).** M5t on all k = 6 survivors of (1,8,2) at t0 = 2.5, DELTA = 1/3:
  - Σ ub, Σ lb, and time per cell;
  - sharded by the DRI if it needs more than 2 processes.

### (2) Secondary, probabilistic (sampling-class; reported separately, never merged with (1))
- **Statement.**
  - For a verified prune set, the survivors' union S is decidable by a forward pass, so
    P(E) <= P(S ∩ far gap ∩ N) + loss(N) exactly.
  - The first term is estimated by plain MC: sorted draws with g23 >= t0, as in `mc_cells.py`, counting the
    draws whose gate prefix is a survivor.
  - The reported bound is the one-sided **Clopper–Pearson upper bound at confidence 1 - 1e-6** on that
    probability, times the exactly known prior factor P(g23 >= t0)/720.
  - P(g23 >= t0) also comes from the same draws. It enters through a joint Clopper–Pearson on the final count,
    so the prior is not used separately.
- **Label.** This bound is **probabilistic**: it holds with probability >= 1 - 1e-6 over the MC seed. It is a
  different class from a deterministic certificate.

## Pre-registration 3 (2026-10-19 ~08:40; two-tier charging, DRI request; before any tiered number is scored)
- **Tier (i).** M2 (`mass_upper`, with N_delta = {g23 >= delta ||x||_inf}) on every survivor of a slice
  (tri, k).
- **Tier (ii).**
  - Repeatedly take the cell with the largest current ub and move it to the next M5t budget: 200, then 2,000,
    then 8,000 simplices. Its ub becomes min(old, new).
  - Stop when the largest remaining cell is < 1% of Σub, or the slice's CPU budget is spent: 2 CPU-h per shard
    for (1,8,2), k = 6.
- Every tier is rigorous, so Σub is a guaranteed bound on Σ_cells γ(cell ∩ N_delta ∩ {g23 >= t0}) at every
  stage.
- **Report** for (1,8,2), k = 6, t0 = 2.5, delta = 1/3:
  - Σub after tier (i), and after tier (ii);
  - the number of cells refined at each budget;
  - CPU-h;
  - a projection to 720 slices (scaled by survivor counts).
- **Comparison, not a pass line:** the truth for the same quantity, by plain MC (`mc_cells.py` at k = 6).

## Method
- `rigour.py` (library).
  - `exact_rows`: exact fmpq constraints from the gate pattern.
  - `slice_polytope`, `Poly` (exact BFS vertex enumeration), `pulling_triangulation`.
  - Mass methods:
    - `mass_bounds` (M1);
    - `mass_upper`/`orthant_bound` (M2, the cheap call; any k);
    - `mass_lawrence` (M3);
    - `mass_split` (M4);
    - `mass_cell` (M5, k < 10);
    - `norm_loss` (part c).
- `verify.py` (part b).
  - `verify_prune(tri, gates, k, event_aware, delta)`, `verify_leaves` (stage-3 splits),
    `verify_certfile(path)` (T398 certificate format v1: tri, k, stage, start, leaves, delta).
  - Every pre-activation bound is a Neumaier–Shcherbina safe dual bound:
    - float duals from HiGHS;
    - the residual r = c - Aᵀy absorbed by the variable box;
    - explicit γ_n error terms for the float data (cell rows recomputed from the weights, with propagated error
      bounds) and for every float operation, × 2.
  - The final certificate is a positive safe lower bound on min over the relaxation of
    max_j (L_j - L_p3). If that LP is infeasible, a phase-1 bound with τ on every row is used instead.
  - The triangle chord is made valid by raising its intercept.
  - Floats only propose; nothing is decided by a float LP status.
- Drivers:
  - `harvest.py`: k = 10 event regions from line-walk points;
  - `mc.py`, `mc_ref.py`, `mc_cells.py`: MC references;
  - `a_cells.py`, `a_m5.py`: mass scoring on T398's bench;
  - `b_test.py`, `b_bench.py`: verifier scoring.

## Status
Running. The verifier (b) and the proof (c) are done. The mass bound (a) is rigorous but not yet tight enough
to be useful: the best options are M5 at 3-23x (median 6.9) for 25 s per k = 4 cell, or M2 at 12-5e4x (median 365) for 30 ms.

## Results

### (c) Normalisation loss [measured, arb]
- `norm_loss(1.5)` = 7.374e-50 (< 1e-49) with NORM = 10.
- At T398's t0 = 2.5 with DELTA = 1/3, the bound is 20 Φ(-7.5) = 6.4e-13.

### (b) Verifier (T398 benchmark (1,8,2), k = 4 → 5; `python b_bench.py <dir with relax2.py, cone_count.py> 46 0`)
| test | result |
|---|---|
| B1 (47 cells containing a probe event point, event-aware) | **0 / 47 verified** (must be 0) |
| B2 (all 717 float-pruned k = 5 children of the first 46 survivors) | **717 / 717 verified**. By stage: 0: 668/668, 1: 21/21, 2: 26/26, 3: 2/2 (36 and 30 leaves) |
| B3 cost, stage 0 | median 5.3 ms (float 3.5 ms) |
| B3 cost, stages 1-2 | 0.6-0.8 s (float 0.5-0.6 s) |
| B3 cost, stage 3 | 1.2-1.3 s per cell (float 0.9 s) |

Not run: the deliberately falsified certificates of B1 (pre-registered). The verifier takes no bounds from the
certificate, so there is nothing to falsify. Instead, B1 holds cells that are known to be event-feasible.

**T398's first certificate set.** Files: `certs/cert_d0.333_1-8-2_k{2..6}.npz` (branch claude/t373-event-cone
2a82be07; DELTA = 1/3). Command: `python cert_run.py data/certs/*.npz`.

| k | cells | verified | stage 0 / 1 / 2 / 3 | total s | median s |
|---|---|---|---|---|---|
| 2 | 2 | 2 | 2/0/0/0 | 0.03 | 0.015 |
| 3 | 40 | 40 | 39/1/0/0 | 1.2 | 0.005 |
| 4 | 398 | 398 | 368/17/11/2 | 31 | 0.005 |
| 5 | 2,030 | 2,030 | 1,664/113/188/65 | 303 | 0.005 |
| 6 | 3,513 | 3,513 | 3,115/160/113/125 | 254 | 0.005 |
| **all** | **5,983** | **5,983 (0 failures)** | | **590** | |

Median seconds per stage-1/2/3 cell: 0.45-1.0 / 0.6-1.1 / 0.7-1.3.


### Survivor-mass curve C_true(k) for (1,8,2), t0 = 2.5, delta = 1/3 [measured]
Plain MC with 1e9 draws (`python surv_curve.py 1,8,2 2.5 1e9 6`; output in `results_surv_curve_1-8-2_2.5.txt`).
The slab mass P(top three at (1,8,2), g23 >= 2.5, N) is 5.6e-8.

| k | 1-3 | 4 | 5 | 6 |
|---|---|---|---|---|
| survivors | 2 / 18 / 39 | 146 | 275 | 781 |
| C_true(k) | 5.6e-8 (100%) | 4.5e-8 | 4.2e-8 | **2.5e-8 (45%)** |

- No event draw occurred in the 1e9 draws: P(E_(1,8,2)) < ~1.4e-8 at 1 - 1e-6. The event-in-survivors check
  is therefore vacuous here.
- [inferred] Charging at k = 6 costs at least C_true(6) = 2.5e-8 for this one triple, even with a perfect mass
  bound. That is more than 25x the target truth (< 1e-9 over all 720 triples).
- So the charge depth must be close to k = 10, where the event rows apply. This is T398's point.


### Two-tier charging, (1,8,2) k = 6, t0 = 2.5 [measured; pre-registration 3; workers W1-W4]
Six of the 8 shards were completed. Shards 98-196 and 490-588 were stopped once k = 6 charging was known to be
futile (the C_true curve).

| shard | Σub, M2 (tier i) | Σub, tiered (tier ii) | MC truth of the same cells | refined (200 / 2k / 8k) | CPU-h |
|---|---|---|---|---|---|
| 0-98 | 3.5e-5 | 2.4e-9 | 2.2e-10 | 49 / 7 / 7 | 0.59 |
| 196-294 | 2.0e-5 | 4.2e-8 | 2.7e-9 | 60 / 7 / 4 | 0.28 |
| 294-392 | 8.1e-6 | 4.2e-8 | 1.6e-9 | 48 / 4 / 3 | 0.19 |
| 392-490 | 1.2e-5 | 6.0e-7 | 2.0e-8 | 67 / 6 / 6 | 0.52 |
| 588-686 | 4.0 (vacuous) | 1.4e-16 | 0 | 17 / 2 / 2 | 0.11 |
| 686-781 | 9.4e-15 | 8.0e-25 | 0 | 61 / 9 / 9 | 0.52 |
| **6 shards** | | **6.9e-7** | **2.5e-8** | | **2.2** |

- **Tightness.** The tiered Σub is about 28x the MC truth. Cells with >= 30 MC hits (15 cells, all at the top
  tier): ub/MC median 13.5, p90 26, max 31.
- **Soundness.** No violation.
- **Reading** [inferred]:
  - At k = 6 the reduced dimension is 7, and M5t's 8,000-simplex budget leaves 10-30x on the dominant cells,
    against 1.5x at k = 4 (dimension 5).
  - M2 alone is useless here: 1e2-1e5x, and vacuous on one shard.
  - Even an exact k = 6 charge would cost C_true(6) = 2.5e-8, above the target.


### T398's option (c), event-projection cuts: float pre-test [measured; `python cut_test.py 1,8,2 6 2.5 2e8`]
- **Method.** For each of the 781 k = 6 survivors, I computed homogeneous cuts ±x_i <= β g23 with
  β = max over the event-aware LP relaxation of ±x_i/g23 (a Charnes–Cooper LP).
- **Result.** The cuts keep **3,628 of 3,629** MC slab points in the survivors (C_true 2.52e-8 → 2.519e-8).
- **Reading** [inferred]. The relaxation's projection onto x is essentially the whole cell, since the triangle
  relaxation of 4 remaining steps is loose. So these cuts cannot shrink the charge.
- **Not tried:**
  - other directions (e.g. the LP's own extreme directions);
  - cuts on split leaves (stage-3 style), which give tighter projections.

### (a) Mass bounds: tightness against MC, and cost

**k = 10 event regions, t0 = 1.5** ((1,8,2), regions 0-5 of `data/regions_1-8-2.npz`).
The MC is an importance sampler with 1e7 points per region; relative s.e. 16-72%.

| region | MC mass | M2 ub/MC (0.2-1 s) | M3 ub/MC (2-450 s) | M4 ub/MC |
|---|---|---|---|---|
| 0 (752 vertices) | 2.5e-11 | 210 | 850 | - |
| 1 (2,122 vertices) | 3.9e-11 | 44 | 2,760 | - |
| 2 | 2.5e-12 | 1.6e9 | - | - |
| 3 (142 vertices) | 9.1e-12 | 5.1e4 | 9.7 | 9.7 (563 s, 9 pieces) |
| 4 | 6.2e-11 | 480 | - | - |
| 5 | 9.3e-11 | 6.5 | - | - |

**k = 4 cells, event ignored (T398's cheap call), t0 = 2.25.** All 273 benchmark survivors. The plain MC uses
2e8 draws, with 32,088 points in the triple's far-gap set.
- The MC total over the 273 cells is 2.23e-7, which equals the triple's whole far-gap mass.
- Only 39 cells have any MC hit; 27 have >= 30 hits.
- `python a_cells.py 2.25` (M2) and `python a_m5.py 2.25 4000` (M5):

| method | ub/MC median | p90 | max | min | cost per cell | Σ ub over 273 cells |
|---|---|---|---|---|---|---|
| M2 (orthant/tilt) | 365 | 5,000 | 4.7e4 | 12 | 27 ms (p90 43 ms) | 5.6e-4 (vs 2.2e-7) |
| M5 (exact marginal, 4,000 simplices) | 6.9 | 20 | 23 | 3.3 | 25 s (p90 28 s) | (27 scored cells) |

| **M5t** (exact per-simplex integrals, 2,000 simplices; pre-registration 2) | **1.55** | **2.07** | **2.93** | 1.20 | 23 s (p90 25 s) | Σ ub 3.9e-7 vs Σ MC 2.2e-7; rigorous lb/MC median 0.50 |

**At t0 = 2.5** (28 cells, 1e9-draw MC; `python a_m5.py 2.5 2000 t`), M5t has ub/MC median 1.49, p90 2.04, max 3.23, min 1.08, and lb/MC median 0.50. Cost is 22 s per cell. Σub is 9.6e-8 against Σ MC 5.6e-8, with no violation. M2 on the same cells has median 387. **M5t passes its pre-registered line** at both t0 = 2.5 and t0 = 2.25 (median <= 3, p90 <= 10, <= 30 s), with no soundness violation (`python a_m5.py 2.25 2000 t`). On the same 27 cells M2 has median 365. The M5 lower bounds are ≤ 0.49 of MC: rigorous [lb, ub] brackets of width 7-70x.

No ub fell below its MC value minus 4 s.e. on any scored cell or region.

## Compute accounting
- LP solves are HiGHS via scipy. Exact arithmetic is python-flint fmpq/arb.
- Verifier: about 1.1-1.4x the float LP time of the cell it verifies (table above).
- MC:
  - 6e7 forward passes for the k = 10 regions;
  - 2e8 draws (32k forward passes after the gap filter) for the k = 4 cells.
- About 4 CPU-h in total so far, including the abandoned M1/M4 runs (about 1.5 CPU-h).

## Claims
1. [measured] The verifier accepted none of the 47 cells that contain T398's probe event points, so 0 false
   verifications. Before two bugs were fixed (a sign error in the final margin rows; missing ordering-cone
   rows), it accepted 16/47. B1 caught this.
2. [measured] The verifier confirms T398's float prunes on the benchmark children (numbers in the table).
   Its bounds match relax2's to about 1e-9 (safe-bound slack). Its margins match relax2.max_margin to about 1e-8.
3. [inferred, argued in verify.py's docstring] Soundness of the safe bounds.
   - It rests on weak duality with box absorption, plus the γ_n rounding model of IEEE double.
   - Not proved: that the error-bound computation's own rounding is covered by SAFE = 2 (it is, by a factor
     of ~1e13, but this is not formalised).
4. [measured] The cheap rigorous mass bound M2 is 12-5e4x above MC on k = 4 cells, and 6-1e9x on k = 10 regions.
   M5 (exact marginal reduction) is 7-30x at a 4,000-simplex budget. No method tried is within 2x at a cost
   usable for charge-and-stop.
5. [speculative] The obstacle for k = 10 regions is intrinsic.
   - The event regions are 10-D polyhedra with ~20-30 facets and up to 2,000 slice vertices.
   - Their mass (~1e-11) comes from thinness in many directions.
   - So any bound that keeps < 10 rows (M2), or one direction of the quadratic (M3), loses orders of magnitude.

## Alternatives not ruled out
- **Better per-simplex bounds in M5.** For example:
  - exact exponential-linear integration over each simplex (Brion) after a tangent bound of |m|²;
  - finer z-bins.
  These could cut M5's refinement need by orders of magnitude. Not tried.
- **A probabilistic certificate for the survivor mass.**
  - Once pruning is verified, the survivor set S is a decidable set.
  - P(E) <= P(S ∩ far gap) can be bounded above by MC with a Clopper–Pearson bound at confidence 1 - α.
  - This is not deterministic. The owner/DRI must decide whether it is acceptable.
- **Charging at a coarser level** (k < 10 cells with M5, where the dimension is k + 1 for (1,8,2)) rather than
  per k = 10 region.
- The MC references themselves:
  - k = 10 regions: relative s.e. up to 72%.
  - k = 4 cells: < 30 hits for 12 of the 39 cells with hits, so those cells are not scored.

## Failure modes
- `Poly.enumerate` is exact but slow in Python: 30 ms/vertex at 700-bit rationals. Degenerate vertices
  enumerate edge directions combinatorially; their cost is unbounded in principle.
- M3/M4 need a simple polytope (checked) and a generic direction (re-drawn up to 6 times).
- Verifier:
  - It re-solves every bound LP; certificate duals are not used yet.
  - Cells whose float margin is below about 1e-9 cannot be verified.
  - It relies on HiGHS returning usable duals (status 0). Otherwise it answers "not proven", never "pruned".
