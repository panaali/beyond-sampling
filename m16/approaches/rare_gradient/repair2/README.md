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

### Tool check (before any design run)
`python test_wline.py` (orig, W(1.0)): v1-line P_W 5.989e-4 +- 1.3e-6 vs plain conditional MC 6.022e-4 +- 1.2e-6
(-0.5%, z = -1.9); along T372's top direction dP_W/du 5.9e-3 +- 5.2e-4 vs a paired FD (h = .003) 4.7e-3 +- 2.7e-4
(1.27x, z = 2.1). Possible small bias at the 1-2 s.e. level, not resolved; acceptable for a design tool with 2x margins
(the verification is independent).

### Deviation D1 (2026-10-19, after the round-1 design rows; before any truth)
Round 1 did not qualify: the combined direction lowers design P_E fast (x.024 at eps .004) but RAISES design P_W
(predicted d log P_W / d eps = +17.7; rows x1.13 at .004 ... x6.6 at .05). The E1 log-gradient (~950) dominates W's
(~120). Second-order bound on S (16 weights; 12 + 4, two overlap): the cheapest direction for a 5x cut of P_W costs
c^2 / (2 g' M^-1 g) = 7-12 pp of accuracy (M clipped at .05 x max or floored at .5), vs .02 pp allowed: unreachable by
> 300x, so re-linearised rounds 2-3 cannot qualify on this set. They are skipped. The pure-W direction is evaluated with
design estimates (`wonly.py`) to measure that, and, per rule 4, the best accuracy-feasible round-1 point (k = 16,
eps .004, lowest design f) is frozen and verified against both bars.

## Status
**Done. T411 bar: NO-GO (scoped negative, measured).** No edit found or possible near the trained weights on these 16
weights: P(W(2.0)) has a local minimum ~x.97 along the best direction. The frozen fallback point passes **T402's bar**
(E1(2.0) x.034, accuracy +.016 pp, no event up > 2x) but raises W by 12%.

## Results
Design: `python design2.py grad 1 && python design2.py hess 1 && python design2.py search 1`; diagnostics `wonly.py`,
`check_dW.py`, `path_W.py`; verification `python verify.py ...` (results/ver.log, seeds 411101-411109).

**Design.** P_W(2.0) = 1.397e-5 +- 2e-8 (v1 lines; 99.95% largest-input picks). |d log P_W / dW| = 118 (all 432),
92 on S, against ~950 for E1(2.0). S (16): Wh[n7<-n14], Wh[n7<-n2], Wh[n1<-n14], Wh[n2<-n10], Wh[n7<-n0],
Wh[n1<-n2], Wh[n2<-n11], Wh[n6<-n2], Wh[n1<-n0], Wh[n6<-n10], Wh[n2<-n2], Wh[n2<-n7] (top |g_W|) + Wh[n12<-n10],
Wh[n12<-n2], Wh[n11<-n2], Wh[n6<-n7] (T402's top |g_E|). Combined direction rows (design ratios):

| eps | P_W | P_E | d acc (pp) |
|---|---|---|---|
| .001 | 1.024 | .66 | +.024 |
| .002 | 1.053 | .30 | +.033 |
| **.004 (frozen fallback)** | **1.127** | **.024** | **+.014** |
| .008 / .016 / .05 | 1.31 / 2.50 / 6.62 | .009 / .007 / .001 | -.20 / -1.67 / -10.8 |

**Why it fails [measured].** Along the pure-W second-order direction (`wonly.py`, `path_W.py`, paired plain conditional
MC 1e7): the v1-line slope is right (d log P_W / d eps: line -47.9, paired FD -48.6 +- 1.3 at h = .00025, -46.9 +- .7
at .001), but P_W turns at once: ratios .980 / .974 / .999 / 1.065 / 1.242 / 1.818 at eps .0005 / .001 / .002 /
.003 / .005 / .01, while E1 and the accuracy get worse (eps .01: P_E x18, accuracy -1.19 pp). So the trained network
sits within ~3% of a local minimum of P(W(2.0)) in this 16-weight subspace. Second-order bound (accuracy Hessian):
a 5x cut of W costs ~7-12 pp of accuracy.

**Verification of the frozen fallback point (eps .004, 12 of 16 weights move by up to .0022) [measured]:**

| event | N | base | edit | ratio |
|---|---|---|---|---|
| **W(2.0)** | 3e8 | 1.399e-5 | 1.571e-5 | **1.123 +- .0002** (T411 bar .2: FAIL) |
| **E1(2.0)** | 3e8 | 7.05e-9 | 2.39e-10 | **.034 +- .003** (T402 bar .1: PASS) |
| accuracy | 1e9 | .95311 | .95327 | **+.0158 +- .0003 pp** (both bars: PASS) |
| E1(1.0 / 1.25 / 1.5 / 1.75 / 2.25) | 2e7 (1e8) | | | .41 / .28 / .15 / .060 / .016 |
| W(1.0 / 1.25 / 1.5 / 1.75 / 2.25) | 2e7 (1e8) | | | 1.056 / 1.071 / 1.087 / 1.105 / 1.139 |
| E2(.5 / .75) | 2e7 | | | .996 / .969 |

Pick split at g23 > 2 (3e8 draws): rank-3..7 picks 3661/766/113/17/2 -> 124/36/6/0/0; largest-input picks
7,261,446 -> 8,161,061 (+12.4%). Design vs verification: P_E .024 vs .034, P_W 1.127 vs 1.123, accuracy +.014 vs
+.016 pp.

## Compute accounting
Design: g_W 3.9e7 FPE (2e6 v1 lines), g_E reused from T402 (2.1e8), Hessian 1e6 x 529 forward passes, 9 candidates
x (1e6 v1 lines + 5e6 v3 lines + 2e7 accuracy passes). Diagnostics ~4e8 passes. Verification ~3.1e9 forward passes
(not part of the method). Single-threaded, <= 2 processes.

## Claims
- [measured] On M_16,10, P(any wrong pick, g23 > 2) cannot be cut 5x (nor even 5%) by small edits of the 16 most
  relevant weights: along the best second-order direction it has a local minimum at ~x.97. **T411 bar: NO-GO**,
  scoped to this subspace, L2 <= .05, the line-CMC gradient + accuracy Hessian method. Not a proof for all 432 weights
  or for large edits.
- [inferred] The contrast with T402 is the finding: the common near-tie error (W, mostly largest-input picks) is
  already at a training optimum, while the rare far-gap error E1 is not (training never sees it), so only the rare
  error has a large exploitable gradient. Repairing the rare error by small edits necessarily moves mass into the
  common near-tie error (T402 +9.3%, this point +12.4%), which is optimised and cannot absorb a downward push.
- [measured] The fallback point passes T402's bar more strongly than T402's own edit (E1(2.0) x.034 vs x.059, accuracy
  +.016 pp vs -.012 pp), at the price of more conversion (W +12.3% vs +9.3%).

## Alternatives not ruled out
- Larger edits, all 432 weights, or retraining with E1 examples could find a different basin where W is lower; this
  work only shows there is no nearby one in this subspace.
- A different objective weighting (e.g. log P_W alone with an E1 constraint) gives the pure-W path above, which
  already fails; constraint-based schemes would not do better than that path's minimum.
- The v1-line P_W had a ~0.5% low bias at W(1.0) (tool check); irrelevant at the scale of these effects.

## Failure modes
- Two targets on very different log scales: a sum of log-gradients is dominated by the rarer one.
- First-order predictions for a common, already-optimised error hold only for eps <~ .001.
