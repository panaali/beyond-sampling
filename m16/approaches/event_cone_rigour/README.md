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

## Method
(filled in as built) `rigour.py` (library), `harvest.py` (test regions), `test_rigour.py` (A1, B1), `score.py` (A2, A3, B2, B3).

## Status
Running: pre-registered; implementation started.

## Results
(none yet)

## Compute accounting
(none yet)

## Claims
(none yet)

## Alternatives not ruled out
(to fill)

## Failure modes
(to fill)
