# T385: independent re-check of T372's rare-event weight-gradient and its n6/n7 mechanistic reading

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · checked: [T372](../rare_gradient/README.md) (branch `claude/t372-rare-gradient` 1973bfb9)

## Question
T372 reports (i) an unbiased sampling-class estimator of dP(E1)/dW (boundary terms along exact v3 lines) that is
within 2x of finite-difference truths on 11/12 directions, and (ii) a mechanistic reading on M_16,10 at
E1(t0 = 1.25) (P ~ 1.1e-6): the rare error is set by the balance of the comparator pair n6/n7, with neuron gain
elasticities n6 +112 ± 12, n12 -44 ± 5, n7 -43 ± 10, n8 +38 ± 7, n11 -30 ± 3, and top weights n2->n6 (+),
n2->n7 (-), the n6<->n7 coupling, n10->n6. Do both hold up under a different agent's own code, other line
directions, other seeds, independent truths, and an intervention test?

Event: E1(t0) = {argmax of the logits = position of the rank-3 input, and g23 = v2 - v3 > t0}.

## Method (pre-registered 2026-10-19, before any scored run; nothing in this folder has been compared with a truth)

**Estimator (sampling-class; own code, T372's estimator not read beyond its README).** `lines.py`.
Rank-r coordinate-line conditional MC. Draw x from the exact law given g23 > t0 (`core.CondLaw`, own
implementation; p_gap matches brute-force rejection, `check_law.py`). Take k = position of the rank-r value and
replace x_k by a free coordinate s on the interval I_r where x_k stays rank r and g23 > t0 (ends are order
statistics of the other nine and t0; model-free). Then P = p_gap E[Z/A], Z = exact phi-mass of {pick = p3} on the
line, A = phi-mass of I_r. Z is computed by walking the ReLU cells of the line exactly (numpy; one forward + one
tangent pass per cell). dZ/dW = sum over event-boundary points e inside cells (roots of the binding margin
L_p3 - L_j) of phi(e) (dm/dW) / |dm/ds|; dm/dW by torch autograd (independent of the walk code).
- **Line family:** rank r = 1 (lines along the LARGEST input, I_1 = (u1, u1 + 9)) is the scored estimator. T372
  used rank-3 lines (along v3); rank-1 lines are a different line family with different variance structure.
  Secondary (reported, not gating): rank r = 2 lines.
- **Implementation check (done, `python check_fd.py orig 1e5`, results/check_fd_orig.json):** on fixed lines the
  analytic gradient of the estimate equals a central FD of the same estimate to rel. error <= 2e-7 (4 directions,
  delta 1e-5..1e-7).
- **Pilot (done, estimator only, seeds 385900+ / 385950+, `pilot.py`, `pilot2.py`, results/pilot2_*.log; no truths
  computed or seen):** used only to choose the rank and N. Hit fraction per line on orig: r1 4.7e-3, r2 3.5e-4,
  r3 3.6e-4, r4 1.3e-4; at equal CPU time r = 1 gives the smallest s.e. on both models. The pilot's own
  elasticities (r1, 1e6 lines) were already printed: they are development values, not scored.
- **Scored runs:** `python est.py orig 1 2e7`, `python est.py dev3_s528 1 1e7` (seeds 385210 / 385211);
  secondary `python est.py <model> 2 5e6` (seeds 385220 / 385221). s.e. from per-line values (ratio quantities by the
  delta method).

**Quantities compared.** For a weight-space direction d, D(d) = d log P / d eps of P(W + eps d) = (d . grad P) / P.
- Neuron gain elasticity E(i): d = w restricted to neuron i's incoming weights (Wi[i], Wh[i, :]) (T372's E_in).
- Random directions: 3 unit Gaussian directions per model, `default_rng(385000 + 10 * model index + j)`.
- Interventions (orig only): d = w restricted to a weight set (scaling it by 1 + eps): coupling n6<->n7
  (Wh[6,7] and Wh[7,6] together), n2->n6 (Wh[6,2]), n2->n7 (Wh[7,2]); also reported: joint n2->n6 & n2->n7,
  Wh[6,7] alone, Wh[7,6] alone. (Wh[i, j] = weight from neuron j into neuron i.) All in `dirs.py`.

**Truths (sampling, truth only, own code `truth.py`).** Conditional-law MC with common random numbers (the same
draws feed W and every W ± eps d), eps ladder {.03, .01, .003, .001}. Truth of D(d) = symmetric log-difference
(log H+ - log H-) / (2 eps) at the largest level whose realised changes satisfy |log(P±/P)| <= .3 (the smallest
level if none does); s.e. by the delta method on the paired indicators. N: orig 4e7 draws (seed 385100),
dev3_s528 2e7 (seed 385101). Inconclusive rule below if too few items resolve. The log-difference (rather than
T372's difference of P) is chosen because P is log-convex in W; it is exact for P proportional to exp(linear).

**Go lines (fixed now).** An item is *resolved* if |truth| > 3 s.e. It *passes* if sign(estimate) = sign(truth)
and 0.5 <= estimate / truth <= 2.
- **G1 (gradient estimator, own):** items = 16 gain elasticities + 3 random directions on each of orig E1(1.25)
  and dev3_s528 E1(2.75) (38 items). GO if >= 80% of the resolved items pass, with at least 19 resolved (else
  inconclusive: more N).
- **G1-T372 (T372's published neuron map):** T372's five named elasticities on orig (n6 +112, n12 -44, n7 -43,
  n8 +38, n11 -30) vs the truths. REPRODUCED if >= 4/5 pass and every resolved one has T372's sign. Also reported:
  all 16 of T372's orig elasticities vs the truths, and T372 vs this estimator (z of the difference).
- **G2 (intervention, mechanistic claim):** prediction = eps E_set with E_set = (d . grad P)/P from T372's
  published orig gradient (`cmc_orig.json`: coupling +7.4, n2->n6 +81.2, n2->n7 -85.7, joint -4.4, Wh[6,7] -18.8,
  Wh[7,6] +26.2); observed = (log P(1 + eps) - log P(1 - eps)) / 2 at a fixed eps chosen NOW as the ladder level
  where the predicted |eps E_set| is closest to .2: n2->n6 .003, n2->n7 .003, coupling .03, Wh[6,7] .01,
  Wh[7,6] .01, joint .03. GO if the three gated interventions (coupling, n2->n6, n2->n7) all pass (sign and within
  2x of the prediction; an unresolved gated item makes G2 inconclusive for it). Also reported: the same with this
  estimator's prediction; each side's ratio P(1 ± eps)/P against exp(± eps E_set); every ladder level.
- **Balance test (reported, not gating):** if n6/n7 balance is what matters, scaling n2->n6 and n2->n7 together
  should nearly cancel: predicted -4.4 vs +81 / -86 separately. Passes if |observed joint| at eps .03 is
  <= .25 x min(|observed n2->n6|, |observed n2->n7|) (both per unit eps).

**Alternatives to keep in mind (rule 3).** (a) Agreement could come from both estimators sharing a bias of the
boundary formula (e.g. a missed boundary type): separated by independent truths that use only indicators.
(b) The truth FD could be biased by curvature (log-convex P): the ladder shows convergence; the log-difference is
exact for log-linear P. (c) A large elasticity could reflect a global scale effect on logits rather than a
specific circuit: the intervention on single weights and the joint/balance test separate these. (d) Heavy tails
of the per-line gradient (1/|dm/ds| near tangencies) could make the s.e. optimistic: reported via the share of the
largest line in each projection.

## Status
Pre-registered; scored runs not started.

## Results
(pending)

## Compute accounting
(pending) Counted: estimator walk cells (2.3 FPE each: forward + tangent + root search) and endpoint
forward+backward (3 FPE each); model-free sampling not counted. Truth forward passes reported separately.

## Claims
(pending)

## Alternatives not ruled out
(pending)

## Failure modes
(pending)
