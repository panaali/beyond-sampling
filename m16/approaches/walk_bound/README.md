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
_(after the measurements)_

## Status
Running (pre-registered).
