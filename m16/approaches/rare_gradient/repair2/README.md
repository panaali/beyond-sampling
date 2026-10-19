# T411: repair of 'any wrong pick with g23 > t0' (not just the rank-3 pick)

↑ [T402 repair](../repair/README.md) · [T372](../README.md) · [TASKS](../../../../TASKS.md)

## Question
T402's edit made E1(2.0) (rank-3 pick, g23 > 2) 16.8x rarer but raised largest-input picks at g23 > 2 by 9.3%: it
partly converted the rare mistake into a common one. Can the same method (weight-gradient + accuracy Hessian, frozen
before truths) repair the whole region, i.e. make **W(t0) = {argmax != position of the 2nd largest, g23 > t0}** rarer,
at t0 = 2.0, without that conversion?

## Method (pre-registered 2026-10-19, before any T411 design run)
Base model: M_16,10 (`orig`), original weights (not T402's edit). t0 = 2.0. Base P(W(2.0)) ~ 1.4e-5 (from T402's
pick split: 99.95% of it is largest-input picks at a 1st-vs-2nd near-tie), P(E1(2.0)) = 6.87e-9.

**Design tools (method estimates; no truths).**
- P_W and g_W = dP_W/dW: line-CMC along **v1** lines under the law given g23 > t0 (given the rest, v1 has density phi
  on (v2, inf), so the ranks are fixed on the line); per line, wrong mass = window mass - mass{argmax = p2}; boundary
  terms as in T372. Gradient: 2e6 lines, seed 411001. (Lines along v1 cut the 1st-vs-2nd near-tie boundary, where
  W's mass is.)
- P_E, g_E: T402's line-CMC along v3 (g_E from T402's `gE_r1.json`; design P_E: 5e6 lines, seed 411004).
- Accuracy Hessian on S: paired plain MC, h = .005, N = 1e6 inputs, seed 411005 (as T402 R3).
- Design accuracy change: paired plain MC, 1e7 inputs, seed 411006. Design P_W: 1e6 v1 lines, seed 411003. All design
  estimates use the same draws for every candidate.

**Edit rule (fixed now).**
1. S = the 12 largest |g_W| entries with |g_W| > 3 s.e., plus the 6 largest |g_E| entries (T402's top 6) not already
   included: at most 18 weights.
2. Objective: both logs at once, f = log P_W + log P_E; direction d = -M^-1 (g_W / P_W + g_E / P_E) restricted to S,
   M = -H (accuracy Hessian on S, symmetrised, eigenvalues clipped below at .05 x the largest), unit L2 norm.
3. Step eps on the grid .00025 x k, k = 1..200 (to the budget .05). Qualifying: design P_W <= P_W0 / 10 (2x margin on
   the 5x bar) AND design P_E <= P_E0 / 20 AND design accuracy drop <= .01 pp (2x margin on .02 pp). Search: the
   smallest k meeting both P conditions by doubling then bisection (P monotone along the path, checked from the rows),
   then the accuracy check there; if it fails, no grid point on this path qualifies.
4. Fallback: re-linearise (new g_W, g_E, H, new seeds +10 per round) at the k with the lowest design f among those
   meeting the accuracy condition; at most 3 rounds, total L2 <= .05. If none qualifies, freeze the best accuracy-
   feasible point and report it as a design failure (verification still run).
5. Freeze to `repair2/edit.json` (sha256) and push before any verification run.

**Verification (once, on the frozen edit; base and edit on the same draws; new seeds 411101-411110).**
- W(2.0) and E1(2.0): conditional-law MC given g23 > 2.0 (T364 sampler), 3e8 draws, full pick-rank split.
- Accuracy: plain MC, 1e9 inputs.
- E1(t0) at 1.0, 1.25, 1.5, 1.75, 2.25 (2e7; 1e8 for 2.25); E2(.5), E2(.75) (2e7); W(t0) at 1.0 and 1.5 (2e7).

**Bars (both reported, fixed now).**
- **Bar T411 (the task's):** P_W(2.0) ratio <= .2 (95% upper <= .25) AND accuracy drop <= .02 pp.
- **Bar T402 (as before):** P_E(2.0) ratio <= .1 (95% upper <= .15) AND accuracy drop <= .05 pp AND no other listed
  event's P up by > 2x at 3 s.e.
- Also reported: the pick-rank split at g23 > 2 (no conversion), cost, the edit and its neurons, and the
  equal-compute sampling arithmetic.

**Alternatives to keep in mind.** (a) W's mass is a common near-tie error; a 5x cut there may need sharper 1st-vs-2nd
comparison everywhere, which the accuracy budget may forbid: then a scoped negative. (b) The combined objective could
trade one log against the other; both are reported separately. (c) v1-line CMC could be biased at edited weights: the
truths are independent.

## Status
Pre-registered; design not started.
