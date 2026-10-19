# walk_bound (T449): can the weights bound the walk quadrature's line-to-line error?

↑ [TASKS](../../../TASKS.md) · [INDEX H-E16, H-E17](../../../INDEX.md) · [OWNER_QA 4.7, 4.13, 4.14](../../../OWNER_QA.md) ·
frozen recipe: `claude/t429-walk-budget` @ 79ecae63 (`m16/approaches/walk_budget/frozen/`, copied verbatim to `frozen/`, hashes asserted)

## Question
The frozen T10 walk-quadrature estimate (H-E16, H-E17) is `1 - sum_h M_h mean_{u in P_h} g_h(u)`: for each of 5 sign
strata h, a fixed scrambled-Sobol node set P_h in [0,1]^10 (u_0 picks the tie configuration, the rest the face point),
and g_h(u) = (stratum weight) x (Student-t mass of the wrong stretch of the line through the face point), computed
exactly from the weights. What is missing is a bound, derived from the weights, on
`|sum_h M_h (mean_{P_h} g_h - int g_h)|`, the error of averaging over the fixed lines instead of all of them.
Is there a rigorous bound of that kind within ~10x of the actual error?

## Pre-registration (written 2026-10-19, before any bound or measurement is computed)

**Reference error (the thing a bound must be close to).** Per-model quadrature error of the frozen T10 recipe
(RULE_SEED 31415). On DEV3 (100 models) T429 measured RMS error .102 pp at 64k FPE, .0202 pp at 1e6, .0090 pp at 4e6
(i.e. 1.0e-3, 2.0e-4, 9.0e-5); sampling's sd at 1e6 FPE is ~2.6e-4. Re-measured here on zoo seeds 0-4 and a DEV3
subset (first 5 models by name order, dev3_s500-s504) against a reference integral from a 4x larger, independent
Sobol rule (RULE_SEED 27182) at the top budget, and against the 1e8 DEV3 truths.

**Candidates (each a rigorous inequality |error| <= bound; the theorem used is named).**
- **(a) Lipschitz x transport.** For L-Lipschitz g_h (Euclidean metric on [0,1]^10), |mean_P g - int g| <=
  L * W1(uniform, empirical(P)) (Kantorovich-Rubinstein duality). W1 >= d/(d+1) (N V_d)^{-1/d} for any N points
  (each point can own at most volume 1/N; the ball minimises the mean distance), so the best case of this bound is
  computable from L alone. L is bounded below by the largest finite-difference slope found, and its weight-derived form
  is the switch-point sensitivity |d r / d u| = |grad_u g| / |d g / d t| at the endpoints of the wrong stretch.
- **(b) Discrepancy x variation.** Koksma-Hlawka on the cube: |error| <= V_HK(g_h) D*_N(P_h) (Hlawka 1961; Niederreiter
  1992 Thm 2.11). Sphere form: worst-case error of a node set in the Sobolev space H^s(S^d), s > d/2, times
  ||g||_{H^s} (Brauchart, Saff, Sloan, Womersley 2014, Math. Comp. 83, "QMC designs"; the spherical-cap L2 discrepancy
  is the s = (d+1)/2 case by Stolarsky's invariance principle). Checked: is V_HK (resp. ||g||_{H^s}) finite?
- **(c) Cell enclosures.** Partition each stratum's cube into cells (images of cones around the lines), enclose
  min and max of g_h on each cell (T398's LP / gate-pattern machinery at shallow depth); then
  int g in [sum m_i min_i, sum m_i max_i]. Any enclosure method, however tight, has width >= sum m_i (sup_i g - inf_i g)
  (the oracle width); a dense sample inside each cell bounds the oracle width from below. So (c) is killed if the
  oracle width is > 100x the actual error at every cell count reachable in this task, and its extrapolated cell count
  is recorded.

**Measurements first (step 1, not bounds):** per stratum, 2,048 nodes of the frozen rule; random perturbations of size
delta in {1e-1, ..., 1e-6} in u (u_0 held inside its configuration cell); quantiles and max of |Delta g| and
|Delta g| / delta; Hoelder exponent of the median and the 99th percentile; attribution of the largest slopes (smallest
|dg/dt| at a wrong-stretch endpoint = a line grazing a region boundary; window endpoints; configuration jumps).

**Decision rule (per budget B in {64k, 1e6, 4e6} FPE; bound compared with the RMS actual error at that B):**
- **GO:** some candidate gives a rigorous bound <= 10x the actual error at bound compute <= 10 B FPE-equivalents.
- **NO-GO:** every candidate is infinite, or > 100x the actual error at every compute reachable here (a few CPU-h);
  for a family killed by a lower bound on its best case (a: W1 floor; c: oracle width), that lower bound counts.
- **In between** (10x-100x, or <= 10x only at compute > 10 B): a scoped note on what would close the gap.
- If NO-GO is for compute reasons, an estimate goes to docs/COMPUTE_FRONTIER.md (AGENTS standing rule).

**Prediction (written now):** NO-GO. (a) and (b) infinite because g has creases (kinks) along non-axis surfaces and
unbounded root sensitivity where a line grazes a boundary; (c) finite but > 100x at reachable cell counts, because in
9-10 effective dimensions a cell's diameter shrinks only as N^{-1/9}.

**Not used:** the sealed HP5 panel; Fable; any contact with ARC.

## Method
- `lib.py`: loads the frozen T10 recipe (`frozen/`, verbatim from 79ecae63; the five sha256 prefixes T429 asserted are
  asserted again), the lazy Sobol rule and T393's compiled walk (T429's gated execution-only swaps), the panel and the
  per-stratum integrand g_h(U) = `estimator.run_points`. Gate: `est(dev3_s500, 1e6)` reproduces T429's saved row
  bitwise (0.9107139956907699).
- `measure.py err | mod | vitali | oracle MODEL...`: the actual error, the modulus of continuity, the 3-D Vitali
  variation, and the oracle enclosure width (definitions in its docstring and the pre-registration above).
- `sens.py`: an independent exact line walk (different code: per-line gate-pattern Jacobians) that returns every switch
  point's weight-derived sensitivity rho = |c_perp| / |c . d|. Its per-line masses equal the frozen walk's to
  <= 2e-13 on all 51,200 lines (check column in `analysis.log`).
- `analyse.py` -> `analysis.log`, `analysis.json`. Panel: zoo seeds 0-4 and dev3_s500-s504 (10 models).

## Status
**Done (2026-10-19). Verdict by the pre-registered rule: NO-GO.** Every candidate is infinite or more than 100x the
actual error at every compute reached; the closest, (c) with oracle enclosures, is still >= 330x at 8e13 cells.
Pre-registration 73a3a3ce. Nothing was changed after it was written.

## Results
All numbers are from `analysis.log` (`python analyse.py`, after the `measure.py` and `sens.py` runs listed under
Compute accounting).

**Actual error** (`measure.py err`). Bounds are compared with the larger of these, T429's DEV3 RMS (1.0e-3 / 2.0e-4 /
9.0e-5 at 64k / 1e6 / 4e6 FPE), which favours the bounds.

| B (FPE) | RMS vs reference rule (10 models) | max | DEV3 5, vs 1e8 truths | T429 DEV3-100 |
|---|---|---|---|---|
| 64k | 6.7e-4 | 1.0e-3 | 6.5e-4 | 1.0e-3 |
| 1e6 | 9.5e-5 | 1.9e-4 | 7.1e-5 | 2.0e-4 |
| 4e6 | 9.0e-5 * | 1.4e-4 | 7.9e-5 | 9.0e-5 |

\* At 4e6 FPE the reference (1.6e7 FPE, rule 27182) is itself off by 3e-5 to 1.2e-4 from the DEV3 truths, so this row
is the reference's own error.

**Smoothness of g** (`measure.py mod`: 10 models x 5 strata x 2,048 frozen nodes, delta = 1e-1 .. 1e-6)
- g > 0 on every node line (the Student-t radial weight reaches a wrong stretch on every line), so no cell is free.
- Typical points are smooth: the median and 99th percentile of |Delta g| scale as delta^1.0 (Hoelder exponent fits
  .94-1.01). The maximum does not: its exponent is .61-.90, and the largest finite-difference slope keeps growing as
  delta shrinks (stratum 3, dev3_s500: 24 at 1e-4, 52 at 1e-6).
- The largest slopes per stratum (lower bounds on any Lipschitz constant) are 9.5 / 15 / 50 / 84 / 14. The mean slopes
  are .016-.20. Hill tail indices of the slope distribution are .94-2.1, a heavy tail: index <= 2 means g's gradient
  has unbounded variance.
- Where the large slopes come from (`sens.py`, the weight-derived form): the sensitivity rho = |c_perp| / |c . d| of a
  wrong-stretch endpoint has median .06-1, but a 99th percentile of 2.2-8 and a maximum of 15-78 in 1,024 lines per
  model. For gate switch points the maximum is 140-2,000. The tail indices (endpoints 1.2-2.4, gates 1.3-1.7) match
  c . d having a positive density at 0: rho = 1/|c . d| is unbounded, and it diverges exactly where a line grazes a
  region or decision boundary. The largest k(r) rho at a node, a local |grad g|, is 17-91.

**(a) Lipschitz x transport: infinite (no finite L); finite floor already vacuous.** With L set to the largest
slope found (a lower bound on the true L, which is unbounded) and W1 at its floor for N <= B lines per stratum, split
over the 360 / 45 configurations: bound >= 112 / 83 / 71 at 64k / 1e6 / 4e6, i.e. 1.7e5x / 8.7e5x / 7.9e5x the error.
That is vacuous, since g's integral is < 1. Using the mean slope instead of L (not a valid bound) still gives .29 / .21
/ .18 (420-2,200x). Cause: first-order smoothness only buys N^{-1/9} in 9 dimensions.

**(b) Discrepancy x variation: infinite.** Koksma-Hlawka needs a finite Hardy-Krause variation V_HK. The 3-D Vitali
variation sum |Delta^3 g| on (n+1)^3 grids keeps growing as the grid is refined: exponent e3 = 1.06-2.25 on the full cube
(27/27 slices, n = 16-64). On the interior box [.25, .75]^3 it grows with e3 = .61-1.92 in 23/27 slices; the other 4
have e3 = .10-.13, i.e. they miss the creases. A crease along a non-axis surface gives e3 = 1, as theory predicts (Owen 2005,
"Multidimensional variation for quasi-Monte Carlo": kinks give infinite V_HK for d >= 3). e3 > 1 adds the grazing
singularities above, and on the full cube the inverse-CDF transforms at the faces. The 2-D Vitali variation grows
slowly in the interior (e2 mostly < .3), so the obstruction is the 3-D and higher mixed terms. Sphere form: the
QMC-design bound needs g in H^s(S^d) with s > d/2 = 4 to 4.5 (face directions on S^8 or S^9). A function with creases is
not in H^s for any s >= 3/2, and the heavy gradient tail puts even H^1 in doubt. So ||g||_{H^s} = infinity
[inferred from the measured creases and theory].

**(c) Cell enclosures: finite, >= 330x at 8e13 cells.** These are oracle widths (sampled range per cell, a lower bound
on any enclosure's width, T398's LPs included), summed over strata. Each figure is the minimum over the 10 models of
half-width / 2.0e-4:

| cells (total) | 1.2e3 | 9.4e3 | 7.5e4 | 6.0e5 | 4.8e6 | 3.1e8 | 1.6e11 | 8.0e13 |
|---|---|---|---|---|---|---|---|---|
| width (mean over models) | .358 | .322 | .309 | .294 | .262 | .223 | .158 | .098 |
| half-width / error at 1e6 | 1,210x | 1,085x | 1,072x | 995x | 916x | 763x | 541x | 330x |

At bound compute <= 10 B (<= 1e7 cells at B = 1e6, each cell costing >= 1 walk) the ratio is >= 900x. Width falls as
cells^-0.066 (the last three levels; geometric dimension 9 gives -0.11). Reaching 10x at 1e6 FPE would take ~2e29 cells
(N^{-1/9}) to ~6e39 (fit). Adapting the partition to the gradient would gain only 1.1-1.2x (optimal adaptive meshing
for first-order enclosures, computed from the slope distribution).

**Decision (pre-registered): NO-GO.** (a) and (b) are infinite. (c) is >= 330x everywhere it was reached, and >= 900x
within 10 B. The prediction written in advance (NO-GO; (a), (b) infinite; (c) > 100x) holds.

## Compute accounting
- What counts: exact line walks (FPE as charged by the frozen walk) and the independent walk of `sens.py`; no sampled
  model inputs are used in any bound. The random perturbation directions and in-cell points are measurement only.
- Per model: `err` ~42 s (64k + 1e6 + 4e6 + the 1.6e7 reference ≈ 2.1e7 FPE); `mod` 2-4 s (~3e5 lines); `oracle` 22-36 s
  (5 strata x 8 levels x 48k lines ≈ 1.9e6 lines); `vitali` ~40 s (3 strata x 3 anchors x 2 boxes x 5 grids, 4e5 lines);
  `sens` 4-9 s (5,120 lines, pure numpy).
- Total ≈ 20 CPU-min, 2 processes, each under `ulimit -v 2500000` (peak RSS ~0.56 GB).
- Reproduce (from this folder, `source ../../../.venv/bin/activate`; DEV3 weights and truths in `data/dev3/` from
  `claude/t192-gpu-cost-law` `m16/approaches/gpu_cost_law/dev3/`, hashes asserted; zoo seeds via `alg_zoo`):
  `M="seed0 seed1 seed2 seed3 seed4 dev3_s500 dev3_s501 dev3_s502 dev3_s503 dev3_s504"`;
  `python measure.py err $M`; `python measure.py mod $M`; `python measure.py oracle $M`;
  `python measure.py vitali seed0 dev3_s500 dev3_s503`; `python sens.py $M`; `python analyse.py`.

## Claims
- [measured] On 10 M_16,10-class models, the walk integrand g is continuous and positive on every line. It is Lipschitz
  at typical points, but its largest slopes keep growing as the step shrinks, and the weight-derived switch-point
  sensitivity |c_perp| / |c . d| has a heavy tail (index ~1.2-2.4) that diverges where lines graze boundaries.
- [measured] The 3-D Vitali variation of g grows without sign of saturation (n^.6-2.3) as the grid is refined, in
  50 of 54 slices (3 models x 3 strata x 3 anchors x 2 boxes). [inferred] V_HK(g) = infinity, so Koksma-Hlawka gives no bound; the Sobolev / spherical-cap form
  also needs smoothness (s > 4) that g lacks (creases cap it below 3/2).
- [measured] A Lipschitz x transport bound is >= 71 (vacuous, >= 1.7e5x the error) even with the smallest admissible L.
- [measured] Any cell-enclosure bound on box cells is >= 330x the actual error at 8e13 cells, and >= 900x at <= 10 B
  compute. [inferred] Reaching 10x needs >= ~1e29 cells.
- [inferred] Within the families tried (first-order smoothness, mixed variation, Sobolev on the sphere, oracle
  enclosures), the walk quadrature cannot be made deductive by a weight-derived bound on its node-set error. The
  observed QMC accuracy comes from cancellation that no worst-case bound over g's smoothness class certifies.

## Alternatives not ruled out
- **Bounds that know the crease geometry.** Splitting the cube along g's crease surfaces (the projections of the
  network's linear regions and decision boundaries) would leave smooth pieces where (b) and higher-order (c) work.
  Enumerating those surfaces is the exact-region problem (COMPUTE_FRONTIER items 1-2: 2e13-1e14 regions), not a cheaper
  route.
- **Weaker smoothness classes.** Fractional mixed smoothness (r < 1/2) or Walsh-coefficient decay classes may contain
  g with a finite norm. Their QMC error constants grow like (log N)^{d-1} in d = 9, and certifying class membership from
  the weights needs the same crease information. [speculative] looser than (c) at reachable N; not computed.
- **Statistical bounds** (randomised QMC with a CI, or spread over rule seeds) are cheap and tight but inductive. They
  are exactly what this task tries to replace, so they are out of scope.
- **The panel**: 10 models (5 zoo seeds, 5 DEV3 retrains). The margins are 330x to infinite, so more models would not
  change the verdict; noisy copies and the target model were not run.
- **Oracle-width estimate**: 1,500 random cells x 32 points per level. The sampled range is a lower bound per cell; the
  cell-subsampling CI is reported (CI-lo .058 at 8e13 cells, still >= 290x).

## Failure modes
- Finite differences at delta <= 1e-6 approach float resolution in u (the walk itself is exact to ~1e-14). The slope
  maxima are lower bounds and only strengthen the NO-GO.
- The Vitali growth exponents are fitted on n = 16-64. A few interior slices are flat because they miss creases, which
  does not make V_HK finite: one diverging slice is enough.
- W1's floor uses N <= B lines per stratum, an over-count that makes the floor smaller, i.e. generous to (a).

## Exhaustion checklist
Tried:
- (a) Lipschitz x W1 (Kantorovich-Rubinstein), with the largest-slope L and the mean-slope heuristic, plus the
  weight-derived switch-point sensitivity (endpoints and gates).
- (b) Koksma-Hlawka on the cube (V_HK via 3-D Vitali growth, full cube and interior), and the sphere QMC-design /
  Stolarsky cap-discrepancy form (by smoothness order).
- (c) Oracle cell enclosures on uniform dyadic box partitions (1.2e3-8e13 cells), and an adaptive-partition gain
  estimate.

Not tried, with reasons:
- Real T398 LP enclosures per cell: dominated by the oracle width, which is already > 100x.
- Isotropic discrepancy (Harman's variation): Schmidt's lower bound J_N >= c N^{-2/(d+1)} is slower than (c)'s rate,
  and its constant is not computable.
- Crease-aware partitions: the exact-region problem (above).
- Fractional mixed smoothness or Walsh-decay classes (above).
- Lattice rules: a different node set. The question here is a bound for the frozen set, and the smoothness obstruction
  is the same for any equal-weight rule.
- HP5: sealed, not used.

## Verdict for the method's class
Unchanged from H-E16 / H-E17. The walk quadrature is a deterministic fixed-node (QMC-class) quadrature whose
node-set error has no weight-derived bound within 100x. Within the families above, this task finds that such a bound
cannot be had at any reachable compute. Making the method clearly deductive needs structure-chosen lines, with error
regions handled exactly (T46 hybrid, T389), not a bound on the present line set.
