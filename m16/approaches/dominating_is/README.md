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

Pre-registered; running.

## Results

Pending.

## Compute accounting

Pending (pilot FPE, IS FPE, descent CPU-s, MIQP CPU-s).

## Claims

None yet.

## Alternatives not ruled out

- Missing dominating points (no global search) can make IS biased low with an over-confident s.e.; the validation
  at t0 = 1.5 and the comparison with T400 test this only at those two thresholds.

## Failure modes

- Thin slivers (triple ties) at the dominating point give a low hit rate under a mean-shifted unit Gaussian.
