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

## Status
Pre-registered (R1, R2); design not started.

## Results
(none yet)

## Compute accounting
(design and verification FPE to be reported)

## Claims
(none yet)

## Alternatives not ruled out
See above.

## Failure modes
(to be filled)
