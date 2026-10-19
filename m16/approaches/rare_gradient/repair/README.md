# T402: targeted repair of M_16,10's rarest far-gap error

↑ [T372 rare_gradient](../README.md) · [TASKS](../../../../TASKS.md) · [OWNER_QA 4.10](../../../../OWNER_QA.md#q4-10)

## Question
The bridge-engineer test (OWNER_QA 4.10): can we change a few weights of M_16,10, chosen from a calculation (the
weight-gradient of a rare failure, T372) and not from observed failures, so that a failure too rare for sampling
becomes >= 10x rarer, without hurting accuracy and without creating another rare failure?

## Method (pre-registered 2026-10-19, before any edit is computed)

**Target.** E1(2.0) on M_16,10 (`orig`): the network picks the rank-3 position while g23 = v2 - v3 > 2.0. Truth
6.81e-9 +- 1.1e-10 (T364 truthc, 3e8 conditional draws). Too rare for sampling at this method's cost: the design
below costs ~1-3e8 FPE in total, where plain MC expects p x C ~ 0.7-2 failures (and 0 with probability ~e^-1).
K3 fails here (triple-tie slivers, T364), so the design uses only the unbiased line-CMC (T372 secondary).

**Design tools (the method; sampling-class but conditional, no observed failures used).**
- g_E = dP(E1(2.0))/dW: `cmcgrad.py` (exact v3 lines under the conditional law given g23 > 2.0), 2e7 lines, seed 402001.
- g_A = d(accuracy)/dW: line-CMC along a uniformly random coordinate j over the whole real line (target = position
  of the current 2nd largest, piecewise constant between the other 9 values; same boundary terms), 2e6 lines,
  seed 402002.
- Design estimates of P(E1(2.0)) at edited weights: line-CMC P (2e7 lines, seed 402003, the same draws for every
  candidate = common random numbers); design accuracy change: plain MC 1e7 inputs, seed 402004, CRN (accuracy is not
  rare, so sampling is the right tool for it).

**Edit rule (fixed now).**
1. S = the k = 12 weights with the largest |g_E| among those with |g_E| > 3 s.e. (few weights: 12 of 432).
2. Direction d = -(g_E restricted to S) with its component along (g_A restricted to S) removed, unit L2 norm.
3. Step: eps on the grid 0.0025 x {1, 2, ..., 20} (L2, max 0.05 = 0.12% of ||W|| = 40.3). Take the smallest eps whose
   design estimates give P(E1(2.0)) <= P0 / 20 (a 2x margin over the go line) AND design accuracy drop <= .02 pp.
4. If no eps <= 0.05 qualifies: re-linearise at the best grid point (the one with the lowest design P among those
   meeting the accuracy limit), recompute g_E and g_A there (new seeds), and repeat 1-3 from that point with the
   remaining budget; at most 3 rounds, total ||Delta W|| <= 0.05. If still none: freeze the best point and report.
5. **Freeze** only after T385 (independent check of the n6/n7 reading) reports; if T385 contradicts the gradient
   (sign pattern of the top neurons), stop and report instead of editing. The frozen weights go to
   `repair/edit.json` with their sha256, pushed before any verification run.

**Verification (independent; different code paths and seeds; run once on the frozen edit).**
- P(E1(2.0)): conditional-law MC (T364 `truthc_copy.py` sampler, own forward), base and edited on the same draws,
  N = 3e8, seed 402101. Ratio and its s.e. from the paired counts.
- Accuracy: plain MC, 1e9 i.i.d. N(0, I) inputs, base and edited on the same inputs, seed 402102; Delta acc and its
  paired s.e.
- Other far-gap events, base vs edited on the same conditional draws: E1(t0) for t0 in {1.0, 1.25, 1.5, 1.75, 2.25}
  (N 2e7 each; 1e8 for 2.25) and E2(t) = {wrong, g12 >= t, g23 >= t} for t in {0.5, 0.75} (N 2e7, own conditional
  sampler: v2 ~ 90 phi(v2)(1 - Phi(v2 + t)) Phi(v2 - t)^8, v1 | v2 > v2 + t, v3, lows as in truthc).

**Go (all three, fixed now).**
1. P_edit(E1(2.0)) / P_base <= 0.1, with the upper end of its 95% interval <= 0.15.
2. Accuracy drop <= .05 pp (Delta acc >= -5e-4; an accuracy gain is not a failure), judged on the point estimate with
   its s.e. reported (1e9 paired inputs give s.e. ~1e-5).
3. No new rare failure: no other listed event's P rises by more than 2x with the rise significant at 3 s.e.

**Also reported.** The edit (which weights, which neurons, T241 roles), the design-vs-verification agreement, the
cost (FPE) of the design and of the verification, and whether random-sample fine-tuning at equal compute could have
seen the failure: expected number of E1(2.0) examples among C random inputs = p C, and the samples needed to verify a
10x reduction by plain MC.

**Alternatives to keep in mind.** (a) The edit could lower P(E1(2.0)) by moving the failure to a nearby event
(e.g. E1 at lower thresholds or a different wrong pick): covered by go 3 and by reporting the full wrong-pick split at
g23 > 2.0. (b) A global change (e.g. effectively rescaling the comparators) could do it with any small direction:
compared against a random 12-weight edit of the same norm (control, design estimates only). (c) The line-CMC design
estimate could be biased at the edited weights: the verification truths are independent.

### Amendment R1 (2026-10-19, before any design run; cost)
Design P estimates use 5e6 lines per candidate (not 2e7; one candidate costs ~10 CPU-min), on the same draws for every
candidate. The eps search is doubling (0.0025, .005, .01, .02, .04, then .05) followed by bisection on the 0.0025 grid,
which finds the smallest qualifying grid eps if qualification is monotone in eps (assumed; reported).

### Amendment R2 (2026-10-19, after the tool check only; no design run yet)
The line-CMC accuracy gradient is too noisy for the projection (check `test_lcmc.py`: along a test direction
0.000 +- ~0.06 from 2e5 lines vs a paired plain-MC FD of 0.020 +- 0.0035). Accuracy is not rare, so g_A on the 12
weights of S is instead a paired central FD of plain-MC accuracy: h = 0.01 per weight, N = 1e7 inputs, seed 402002.
Processes run with OMP_NUM_THREADS = 1 (the CPU rule).

### Amendment R3 (2026-10-19, after the first two design candidates; design estimates only, no truths)
Design result: along the R2 direction (first-order accuracy gradient projected out), eps = .0025 gives design
P ratio .032 but design accuracy drop .078 pp; eps = .005: .0072 and .43 pp. The trained weights sit at an accuracy
maximum, so the loss is second order (~eps^2.5) and a first-order projection cannot control it. Change:
- Accuracy Hessian on S, H (12 x 12), by paired plain MC: four-point central differences for off-diagonals, three-point
  for diagonals, h = .005, N = 2e6 inputs, seed 402005 (accuracy is not rare: sampling is the right tool).
- Direction d proportional to -M^-1 g_E,S with M = -H symmetrised, eigenvalues clipped below at .05 x the largest:
  the step that minimises the predicted second-order accuracy loss for a given first-order fall of P.
- Grid eps = .00025 x {1, ..., 40} (to .01), same doubling + bisection; qualification unchanged (design P <= P0/20,
  design accuracy drop <= .02 pp); rounds and the .05 budget unchanged. S is unchanged.

### Note R4 (2026-10-19, implementation of the R1 search; rule unchanged)
R1's doubling + bisection assumed qualification is monotone in eps. It is not (the P condition needs a large eps, the
accuracy condition a small one): eps = .002 fails on P (design ratio .11), .004 on accuracy (-.039 pp), and the
search fell back without looking in between. Per the rule ("the smallest qualifying grid eps"), the bracket is now
searched: bisection on the P condition alone between k = 8 and k = 16 (P is monotone along the path in the design
rows), then the accuracy condition is checked at the smallest P-qualifying k. If it fails there, no grid point
qualifies (accuracy only gets worse at larger eps) and the R1 fallback (round 2 from the best point, k = 8) applies.

## Status
**Done. GO by the pre-registered line (all three conditions), with one trade-off reported below.** The edit was
frozen (edit.json, sha256 6c9ad06a...) after T385 confirmed the gradient and before any verification run.

## Results
Reproduce: `python design.py grad 1 && python design.py select 1 && python design.py hess 1 && python design.py search 1
&& python design.py bracket 1` (design), then `python verify.py ...` (commands in results/ver_events.log; seeds
402101-402109, 402102 for accuracy).

**Design (method estimates only; no truths).** g_E at E1(2.0): line-CMC P 7.24e-9 +- 4.2e-10 (truth 6.81e-9);
|d log P / dW| = 949, 297/432 entries resolved. S = the incoming weights of n6, n12, n11, n7 from n10, n2, n7, n11
(top 12 |g_E|). R2 direction (first-order accuracy projection): eps .0025 -> design P x.032 but accuracy -.078 pp
(fails). R3 direction (accuracy Hessian; -H positive definite, eigenvalues 4.3-1227): design rows

| eps | design P ratio | design d acc (pp) |
|---|---|---|
| .00025 / .0005 / .001 | .65 / .50 / .33 | +.003 / +.006 / +.008 |
| .002 | .11 | +.005 |
| .0025 / .00275 | .063 / .051 | -.003 / -.007 |
| **.003 (chosen, smallest qualifying)** | **.041 +- .016** | **-.011** |
| .004 / .008 / .01 | .012 / .008 / .008 | -.039 / -.29 / -.50 |

**The edit (L2 .003; 12 of 432 weights; largest absolute change .0014):** Wh[n6<-n10] -.00038, Wh[n12<-n10] +.00138,
Wh[n6<-n2] -.00048, Wh[n12<-n2] +.00063, Wh[n11<-n2] +.00103, Wh[n6<-n7] -.00015, Wh[n12<-n11] +.00119,
Wh[n11<-n10] +.00139, Wh[n12<-n7] +.00042, Wh[n6<-n11] -.00113, Wh[n11<-n7] +.00067, Wh[n7<-n10] +.00005.
Relative changes are <= 1.4% except three small weights: Wh[n6<-n11] -6%, Wh[n12<-n11] +18%, Wh[n11<-n10] +32%.
In words [inferred]: weaken comparator n6's drive from the clock n10, the running max n2 and the comparator n7, and
strengthen the clocks n12/n11's drive from n10, n2, n11, n7.

**Verification (independent conditional-law / plain MC, base and edit on the same draws) [measured]:**

| event | N | P base | P edit | ratio | go |
|---|---|---|---|---|---|
| **E1(2.0)** (target) | 3e8 | 6.87e-9 | 4.08e-10 | **.059 +- .004** (95% upper .067) | 1: ratio <= .1 PASS |
| accuracy | 1e9 | .95311 | .95300 | diff **-.0116 +- .0002 pp** | 2: drop <= .05 pp PASS |
| E1(1.0) | 2e7 | 5.53e-6 | 2.63e-6 | .475 +- .010 | 3 |
| E1(1.25) | 2e7 | 1.09e-6 | 3.74e-7 | .342 +- .013 | 3 |
| E1(1.5) | 2e7 | 2.10e-7 | 4.04e-8 | .193 +- .015 | 3 |
| E1(1.75) | 2e7 | 3.70e-8 | 3.38e-9 | .091 +- .015 | 3 |
| E1(2.25) | 1e8 | 1.06e-9 | 3.98e-11 | .037 +- .007 | 3 |
| E2(0.5) | 2e7 | 9.42e-5 | 9.24e-5 | .980 +- .002 | 3 |
| E2(0.75) | 2e7 | 5.15e-6 | 4.99e-6 | .969 +- .005 | 3: no event up > 2x PASS |

Design vs verification: design ratio .041 +- .016 vs verified .059 +- .004 (1.1 s.e.); design accuracy -.011 pp vs
verified -.0116 pp. The repair grows with the threshold (2x at t0 = 1, 27x at 2.25), as expected for a far-gap fix.

**Trade-off (pre-registered check (a), the full wrong-pick split at g23 > 2.0, 3e8 draws) [measured]:** picks of
rank 3/4/5/6/7: 3569/795/116/16/2 -> 212/65/4/0/0 (all rarer), but picks of the LARGEST input rose 7,263,321 ->
7,942,165 (+9.3%). Those are the common 1st-vs-2nd near-tie errors (g12 small) that happen to have g23 > 2: about
+1.3e-6 of probability, against the -6.5e-9 removed from the target. So the edit partly converts the rare rank-3
mistake into the common rank-1 mistake in the same region. It is not a new rare failure (it is 2.4% of the conditional
draws, a known near-tie error) and it sits inside the measured accuracy change (-1.16e-4 overall), but a bridge
engineer would call it "reinforced one member by loading another". Not covered by the go line; reported.

**Could random-sample fine-tuning at equal compute have seen the failure? [measured/inferred]** Design cost: E1 parts
8.9e8 FPE (gradient 2.1e8 + 13 candidate estimates 6.8e8) plus 1.1e9 accuracy forward passes, 2.0e9 in total. Plain
random inputs at that cost contain p C = 6.87e-9 x 2.0e9 = ~14 E1(2.0) examples (~6 at the E1-only cost, ~1.4 at the
gradient's), among 2e9 inputs: a fine-tuning loss would weight them at ~7e-9 of the batch, and no gradient estimate
from 14 indicator events resolves 12 weights. Verifying the 10x reduction by plain MC needs ~30 events on the edited
model, ~30 / 4.1e-10 = 7e10 inputs (35x the whole design cost); our verification needed 6e8 forward passes because it
samples from the law given the gap, i.e. it uses the knowledge of where the failure lives. [inferred] So sampling at
equal cost sees a handful of failures but can neither steer nor verify the repair.

## Compute accounting
Design: line-CMC E1 (2.1e8 gradient + 6.8e8 candidates, FPE as executed by walk.py + backward passes); accuracy plain
MC 1.1e9 forward passes (select 2.5e8, Hessian 5.8e8, candidates 2.6e8). Verification (not part of the method): E1(2.0)
6e8, accuracy 2e9, other events 4.4e8 forward passes. All single-threaded (OMP_NUM_THREADS = 1), <= 2 processes.

## Claims
- [measured] A 12-weight edit of L2 norm .003 chosen from the weight-gradient of E1(2.0) and the accuracy curvature (no
  truths used) makes E1(2.0) 16.8x rarer (.059 +- .004) at an accuracy cost of .0116 pp, and makes every other tested
  far-gap event rarer. GO by the pre-registered line. Unverified (one agent; the truths are independent of the design).
- [measured] The first-order version (accuracy gradient projected out) failed the accuracy limit: the trained weights
  are an accuracy maximum, so the repair needs second-order information on accuracy (cheap to sample) together with the
  first-order rare-event gradient (not available from sampling).
- [measured] The repair trades part of the rare rank-3 error for the common rank-1 near-tie error in the same region
  (+9.3% of those picks at g23 > 2).
- [inferred] The mechanism of the edit matches H-R2: shift the balance away from comparator n6 towards the clocks n12/n11.

## Alternatives not ruled out
- The random 12-weight control of the same norm (pre-registered as a design-only comparison) was not run: whether any
  small edit in this neighbourhood has a similar effect is not excluded by a control (the R3 design rows show a strong,
  graded, direction-specific response, but that is not a control). Cheap follow-up.
- The trade-off shows the gradient targets "pick rank 3" specifically; an objective "any wrong pick with g23 > t0"
  would likely give a different, more honest repair. Not tried.
- Only one model, one target, one edit; no check on other input distributions or on the noised copies (T371).

## Failure modes
- Accuracy is at a maximum: any edit costs accuracy quadratically; the first-order projection is insufficient.
- Relative changes to small weights are large (+32%), even though the absolute edit is tiny; a weight-relative budget
  would select a different edit.
- The search rule must handle two opposing conditions (note R4).
