# T436: dominating-point importance sampling for the far-gap rare event E1(t0) of M_16,10

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [novelty review s4](../../../lit/novelty_review.md)

## Question

Can importance sampling centred on the event's dominating points (Bai, Huang, Lam & Zhao 2020, "Rare-event
simulation for neural network and random forest predictors") estimate P(E1(2.5)) ~ 1.5e-10 for the target
M_16,10 to <= 10% relative s.e. at <= 1e8 forward passes, agreeing with T400's plain-MC 1.55e-10 +- .37e-10?
**Class: sampling-based** (unbiased IS with a sampled s.e.; not deductive, not a certified bound).

## Method

**Event** (as T400 `prob_bound.py`): x ~ N(0, I_10); E1(t0) = {argmax of the step-10 logits = position of x_(3)}
∩ {x_(2) - x_(3) >= t0}. Triple = (position of x_(1), x_(2), x_(3)). Code: [net.py](net.py).

**Scale invariance (used throughout).** The net has no biases, so K = {argmax logits = position of x_(3)} is a cone
and E1(t0) = K ∩ {g23 >= t0}. The dominating points of E1(t0) are t0 times the minimisers of ||x||^2 over
K ∩ {g23 >= 1}; one search serves every t0.

**Global MIQP (attempted first, not used).** [milp.py](milp.py): min ||x||^2 over K ∩ {g23 >= 1} ∩ [-4, 4]^10,
160 ReLU binaries (big-M from LP bound tightening: |pre-activation| bounds grow to 3475 by step 10), triple-free
encoding (20 more binaries), SCIP 10.x via pyscipopt: **no feasible point in 900 s** [measured]. The weak big-M is
the obstacle (as E19/T414 found for bound propagation). It is kept only as an optional per-triple check (below).

## Pre-registration (2026-10-19, before any IS estimate)

Dominating points (code [dp.py](dp.py)), all at the t0 = 1 normalisation:
1. **Pilot:** n_p = 1e7 draws x ~ N(0, I) (seed 1), one forward pass each; keep cone points; rank by
   kappa = g23 / ||x||; seeds = the top 200.
2. **Local descent from each seed:** in the seed's linear region (fixed gate pattern, fixed triple), solve the
   convex QP min ||x||^2 s.t. gate signs, x_a >= x_b, x_b - x_c >= 1, x_c >= x_k (k outside the triple), and
   logit_c >= logit_j (SLSQP); if gate rows are active at the optimum, flip them and re-solve in the neighbouring
   region; stop when ||x||^2 no longer decreases. Accept the end point if the QP constraints hold to 1e-7.
3. **Point set:** merge end points closer than 0.05; order by r^2 = ||x||^2; keep every point with
   r^2 <= r_min^2 + 2 ln(1000) / t0^2 (density within 1e-3 of the best at threshold t0), at most 30.
   (This is the "point-adding rule"; it replaces Bai et al.'s MIQP sequence, which did not solve here; the
   halfspace-cover guarantee is therefore NOT claimed.)

Mixture proposal at threshold t0: q(x) = a0 N(x; 0, I) + (1 - a0) sum_i w_i N(x; t0 x_i, I), a0 = 0.05
(defensive, caps the weight at 1/a0 = 20), w_i ∝ exp(-t0^2 r_i^2 / 2), normalised.

Estimator: n i.i.d. draws X_k ~ q; p_hat = mean(1_E(X_k) phi(X_k) / q(X_k)); s.e. = sample sd / sqrt(n); 95% CI
p_hat +- 1.96 s.e. Fixed n = 1e6 per t0 (seed 2); if the relative s.e. exceeds 10%, ONE extension to n = 1e7
(seed 3, pooled). The network runs only on draws with g23 >= t0 (same free prefilter as T400's MC).

Cost: C = pilot forward passes + IS forward passes + descent CPU time converted at the measured 4.0 us per forward
pass (single core, numpy). F = MSE x C / (p (1 - p)) with MSE = s.e.^2 (unbiased) and p = p_hat: **strict bar
F <= 1, lenient bar F <= 10** (DRI handbook). Also reported: F of T400's prefiltered plain MC, which pays one forward
pass only per draw with g23 >= t0 (its F equals that pass rate).

**Validation first: E1(1.5)**, truth = own plain MC, 4e9 draws (mc.py, seed 1, running before any IS run; ~1000
hits). Pass if |p_hat - p_MC| <= 2 sqrt(se_IS^2 + se_MC^2).
**Main: E1(2.5)**, pass line: relative s.e. <= 10% at C <= 1e8 AND |p_hat - 1.55e-10| <= 2 sqrt(se_IS^2 + (.37e-10)^2).
Report: efficiency gain = 1/F vs plain MC; F against both bars; the triples of the dominating points and of the IS
hits (T400: all 17 MC hits on (9,3,2)).

Optional check (not scored): a per-triple MIQP for the best point's triple, warm-started with it, 1800 s limit; its
dual bound says whether a lower-norm point can exist in that triple.

Design-stage facts seen before this pre-registration (2e6-draw pilot, 40 descents, no IS): best r^2 = 2.846 on
triple (9,3,2); other local minima at 2.86-3.34 on (9,3,2) and ~4.9 on (1,8,2)/(8,1,2); many optima lie on event
boundaries (argmax ties or x_(1) = x_(2)).

## Status

Done (2026-10-19). Both pre-registered lines pass (validation at t0 = 1.5; main at t0 = 2.5).

## Results

Commands (in order): `python mc.py 1.5 4e9 1` (validation truth), `python run_is.py points`, `python run_is.py is 1.5`,
`python run_is.py is 2.5`; comparator rate `gap_rate.json`; optional `python miqp_check.py 1800`.

**Dominating points** ([points.json](points.json)): the 1e7-draw pilot (42 s) and 200 descents (2.7 s) give 128 distinct
end points. Best r^2 = 2.8455 at t0 = 1 on triple **(9,3,2)**, the same value as the design-stage pilot. The next
eleven are also on (9,3,2) (2.858-3.31), except (3,9,2) at 3.271. Then come (1,8,2)/(8,1,2) at ~4.9 and
(0,9,7)/(9,0,7) at ~5.1-5.4. Many optima lie on event boundaries (argmax ties, or x_(1) = x_(2): the triple-tie
sliver). At both thresholds the 30-point cap was reached.

| | E1(1.5) validation | **E1(2.5) main** |
|---|---|---|
| IS estimate | 1.849e-7 +- .207e-7 (rel. s.e. 11.2%) | **1.454e-10 +- .115e-10 (rel. s.e. 7.9%)** |
| reference | own MC, 4e9 draws, 821 hits: 2.052e-7 +- .072e-7 | T400 MC, 1.1e11 draws, 17 hits: 1.55e-10 +- .37e-10 |
| agreement | z = -0.93 (tolerance 2): **pass** | difference -0.10e-10, tolerance 0.78e-10: **pass** |
| IS draws / network passes (after the gap prefilter) / hits | 1.1e7 (extended) / 5.30e5 / 1,671 | 1.1e7 (extended) / 5.57e5 / 2,387 |
| cost C (pilot 1e7 + IS passes + descents 6.9e5 FPE) | 1.12e7 | **1.12e7 <= 1e8: pass** |
| F = s.e.^2 C / p(1-p) | 0.026 (gain 38x) | **1.0e-5 (gain 9.8e4x)** |
| F counting every IS draw as one pass | 0.050 | 2.0e-5 |
| strict F <= 1 / lenient F <= 10 | pass / pass | pass / pass |
| comparator: T400's prefiltered MC, network passes only (F = P(g23 >= t0)) | 5.9e-3 (IS 4.4x worse) | 4.1e-5 (IS 4.0x better) |
| same, charging each MC draw + sort its measured 0.19 FPE (vs IS charged a full pass per draw) | ~0.19 (IS 3.8x better) | ~0.19 (IS ~1e4x better) |
| IS hits by triple | (9,3,2) 1501, (3,9,2) 115, (8,1,2) 33, (1,8,2) 11, (2,9,3) 6, (9,2,3) 4, (0,9,7) 1 | (9,3,2) 2244, (3,9,2) 135, (8,1,2) 6, (1,8,2) 2 |
| MC hits by triple | (9,3,2) 499, (8,1,2) 122, (0,9,7) 66, (9,0,7) 63, (1,8,2) 32, (3,9,2) 21, ... | all 17 on (9,3,2) (T400) |

Both runs needed the pre-registered extension (n = 1e6 gave rel. s.e. > 10%). The hit rate under the proposal is low
(1.5-2.2e-4 per draw; only ~5% of draws pass the gap prefilter): the dominating points sit on thin boundary slivers,
so a unit-variance mean shift puts most draws outside the event.

**Optional MIQP check** ([miqp_check.json](miqp_check.json)): (9,3,2) only, order constraints in the LP bounds (max
|bound| 2000 vs 3475 without), 159 binaries, 1800 s: **no primal solution; dual bound 1.60 vs the descent's 2.8455**.
Inconclusive: a lower-norm (9,3,2) point is not excluded. Defect: the script computes the warm start but never passes
it to SCIP, so this was a cold start.

## Compute accounting

Per threshold: pilot 1e7 forward passes (42 s; shared by both thresholds but charged to each); descents 2.7 s
(= 6.9e5 FPE at 4.0 us/FPE); IS 1.1e7 draws, 5.3-5.6e5 network passes, ~90 s. Validation truth: 4e9 draws, 2.4e7
network passes, 2351 s. Global MIQP 900 s and per-triple MIQP 1800 s (not part of the estimator). All single-core.

## Claims

- [measured] Seeded local dominating points plus mixture IS estimate P(E1(2.5)) = 1.454e-10 +- .115e-10 at 1.12e7
  forward-pass equivalents (rel. s.e. 7.9%). This agrees with T400's plain MC (1.1e11 draws) and has F = 1.0e-5,
  about 1e5x more efficient than plain MC. Sampling-class.
- [measured] Validated at t0 = 1.5 against an independent 4e9-draw MC (z = -0.93).
- [measured] The best dominating point is on triple (9,3,2), where all 17 of T400's MC hits are; 94% of IS hits at
  t0 = 2.5 are there too, and the rest are mostly on its mirror (3,9,2).
- [measured] Most of the gain is the event geometry: T400's free gap prefilter alone gives F = 4.1e-5 in network
  passes at t0 = 2.5. IS is 4x better on that count, ~1e4x when draws are charged.
- [inferred] The point search is local: the halfspace-cover property of Bai et al. is not established (global
  MIQP: no incumbent in 900 s; per-triple MIQP dual bound 1.60 < 2.85).

## Alternatives not ruled out

- At t0 = 1.5 the 30-point cap was filled by (9,3,2) neighbours. Triples (0,9,7)/(9,0,7) (16% of MC hits) got 1 IS
  hit and (8,1,2) (15% of MC hits) only 33, so they were covered only by distant components and the defensive part.
  The estimate stayed unbiased within its s.e. but its variance came mostly from there. A per-triple cap (e.g. <= 5
  points per triple) would likely help; not tested.
- Agreement at t0 = 2.5 rests on T400's 17-hit MC (24% s.e.): a bias up to ~50% would not be detected there.

- Missing dominating points (no global search) can make IS biased low with an over-confident s.e.; the validation
  at t0 = 1.5 and the comparison with T400 test this only at those two thresholds.

## Failure modes

- Thin slivers (triple ties) at the dominating point give a low hit rate (~2e-4) under a mean-shifted unit Gaussian.
  Anisotropic proposals (covariance shrunk across the sliver) are the obvious next step.
- A missed dominating region far from every component is sampled only by the defensive 5% N(0, I). At p ~ 1e-10 that
  part almost never hits, so such a region would bias the estimate low, with a s.e. that does not show it.
