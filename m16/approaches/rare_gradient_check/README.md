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

## Method (pre-registered 2026-10-19, commit ce551d16, before any scored run; unchanged since)

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
**Done.** All three pre-registered go lines pass: **G1 GO** (own estimator, 30/34), **G1-T372 REPRODUCED** (5/5),
**G2 GO** (3/3 interventions). Pre-registered N reached on every run; no amendments.

## Results
Full tables: `results/score.txt` (`python score.py`), from `results/truth_<model>.json` (`python truth.py orig 4e7`,
`python truth.py dev3_s528 2e7`) and `data/t385/est_<model>_r<rank>.npz` (`python est.py orig 1 2e7`,
`python est.py dev3_s528 1 1e7`, `python est.py <model> 2 5e6`). Values are D = d log P / d eps.

**P itself [measured]:** orig E1(1.25): truth 1.0895e-6 ± 2.1e-8 (4e7 draws), own r1 1.0923e-6 ± 1.4e-8, r2
1.064e-6 ± 4.3e-8 (T372: 1.143e-6 truth, 1.083e-6 line-CMC). dev3_s528 E1(2.75): truth 2.7005e-9 ± 3.6e-11, r1
2.7078e-9 ± 2.9e-11, r2 2.655e-9 ± 5.4e-11.

**Go lines [measured]:**
- **G1 (own rank-1 line estimator): 30/34 resolved pass (88%) -> GO.** orig 14/16, dev3_s528 16/18. The 4 failures
  (orig n3, n4; s528 n4, n5) are all small elasticities (|D| <= 3.5) whose truth was taken at the eps = .03 level,
  where curvature biases the symmetric difference: at eps .003 / .001 every one of them agrees with the estimate
  (e.g. s528 n4: truth 3.5 at .03, 8.2 / 7.9 / 10.5 at .01 / .003 / .001, estimate 8.3). Post hoc (not
  pre-registered, `python posthoc.py`): against the inverse-variance mean of the .003 and .001 levels, **31/31**
  resolved items pass.
- **G1-T372: 5/5 pass, all signs agree -> REPRODUCED.**

| neuron (T241 role) | T372 | own r1 | own r2 | truth (eps) | T372/truth |
|---|---|---|---|---|---|
| n6 (x9 comparator) | +112 ± 12 | +125.5 ± 1.8 | +140 ± 13 | +130.2 ± 5.8 (.001) | .86 |
| n12 (clock) | -44 ± 5 | -44.6 ± 0.7 | -49 ± 5 | -46.2 ± 1.8 (.003) | .94 |
| n7 (x9 comparator) | -43 ± 10 | -45.3 ± 2.1 | -60 ± 14 | -52.9 ± 2.8 (.003) | .82 |
| n8 (x9 comparator) | +38 ± 7 | +36.4 ± 1.4 | +45 ± 10 | +38.8 ± 2.2 (.003) | .97 |
| n11 (clock) | -30 ± 3 | -32.3 ± 0.6 | -37 ± 5 | -30.8 ± 1.4 (.003) | .98 |
| n1 (x9 comparator) | -19 ± 6 | -18.1 ± 0.6 | -19 ± 3 | -14.3 ± 0.9 (.01) | 1.33 |
| n13 (4.32 mode) | -16 ± 2 | -16.8 ± 0.3 | -19 ± 2 | -15.7 ± 0.6 (.01) | .99 |
| n0 (clock) | -14 ± 7 | -15.9 ± 0.8 | -12 ± 6 | -14.0 ± 1.0 (.01) | 1.03 |
| n2 (running max) | -44 ± 42 (unres.) | **-15.4 ± 2.6** | -0.5 ± 15 | -0.1 ± 0.5 (.03, curvature); -11.5 ± 1.5 (.01), -12.4 ± 3.9 (.003) | - |

  All 16 of T372's orig gain elasticities: 12/14 resolved pass (the 2 misses are n3 and n4, |D| <= 1.5, the same
  curvature-level effect as above).
- **G2 (interventions on orig, T372's predictions): 3/3 -> GO.**

| scaling (eps fixed in advance) | observed half log-ratio | T372 pred. | own pred. | P(1+eps)/P | P(1-eps)/P |
|---|---|---|---|---|---|
| coupling Wh[6,7] & Wh[7,6], eps .03 (gated) | +0.232 ± 0.009 | +0.221 (.95x) | +0.224 | 1.39 | 0.88 |
| n2->n6 Wh[6,2], eps .003 (gated) | +0.260 ± 0.008 | +0.244 (.94x) | +0.259 | 1.28 | 0.76 |
| n2->n7 Wh[7,2], eps .003 (gated) | -0.266 ± 0.009 | -0.257 (.97x) | -0.251 | 0.77 | 1.30 |
| n6<-n7 Wh[6,7] only, eps .01 | -0.206 ± 0.007 | -0.188 | -0.202 | 0.81 | 1.22 |
| n7<-n6 Wh[7,6] only, eps .01 | +0.289 ± 0.008 | +0.262 | +0.277 | 1.34 | 0.75 |
| n2->n6 & n2->n7 jointly, eps .03 | -0.176 ± 0.004 | -0.133 | +0.088 | **12.5** | **17.8** |

- **Balance test (not gating): passes as pre-registered** (joint -5.9 vs +63 / -32 per unit eps at .03), but the
  eps = .03 point is non-perturbative: scaling n2->n6 and n2->n7 together by ±3% raises P 12-18x on BOTH sides,
  while at ±1% it changes P by only 7% each way (D = 0.4 ± 1.0 vs +87 / -89 for each weight alone at .003).
  The cleaner reading is the small-step one: the two n2 inputs into the comparators have equal and opposite
  effects that cancel to first order when scaled together.

### Notes
- The truths and both estimators agree on the coupling: each n6<->n7 weight matters (-21 / +29 elasticity;
  d log P / dw ~ +150 / -145 in T372), but scaling the mutual inhibition as a whole matters ~3x less (+8.6), because
  the two directions oppose. "The n6<->n7 coupling" is a top weight pair individually, not as a block [measured].
- Mild tension at the 2.5-3.3 s.e. level on orig n7 (own -45.3 ± 2.1 vs truth -53.5 ± 2.5 small-step) and n1
  (-18.1 ± 0.6 vs -12.8 ± 1.5); both are within 1.4x. My per-line s.e. may be optimistic (heavy tails), or the
  estimator has a small bias there; not resolved.
- The implementation check (`results/check_fd_orig.json`) shows the analytic gradient is the exact derivative of
  my estimate (rel. err <= 2e-7). Rank-1, rank-2 and (pilot) rank-3 lines are unbiased for the same quantity and
  agree with each other within their s.e.

## Compute accounting
FPE = 5948 FLOPs (one M_16,10 forward pass). Counted: the walk's cells (2.3 FPE each: forward + tangent + root
search) and the endpoint forward + backward passes (3 FPE each); model-free sampling not counted.
- Estimator, rank 1: orig 2e7 lines, 96k endpoints, **4.3e8 FPE** (51 min on one process); s528 1e7 lines, 18k
  endpoints, **1.3e8 FPE** (15 min). Rank 2 (5e6 lines): 5.4e7 / 3.6e7 FPE. All 432 derivatives come from one run.
  Per FPE this is ~3x more lines than T372's 5e6 lines at 5.8e7 FPE, and the s.e. are 3-7x smaller on orig
  (n6 ± 1.8 vs ± 12) because rank-1 lines hit the event 13x more often (4.7e-3 vs 3.6e-4 per line).
- Truths (sampling, truth only): orig 4e7 draws x 201 passes = 8.0e9 forward passes (4.7 h); s528 2e7 x 153 =
  3.1e9 (3.0 h). Per direction, CRN FD needs 2 x 4 passes per draw.
- Total ~9 process-hours (2 at a time, numpy BLAS threads uncapped for the two truth runs), within the ~30 CPU-h cap.

## Claims
- [measured] An independent rank-1 coordinate-line conditional-MC estimator (own code, different line family and
  seeds) estimates d log P(E1) / d(direction) within 2x and with the right sign on 30/34 resolved items over 2 models
  (pre-registered G1 GO); 31/31 against small-step truths (post hoc). Sampling-class.
- [measured] T372's five named neuron gain elasticities on M_16,10 E1(1.25) (n6 +, n12 -, n7 -, n8 +, n11 -) are
  reproduced in sign and within 1.25x of independent CRN finite-difference truths; ordering n6 >> |n7| ~ |n12| > n8
  > |n11| holds (truth: 130, -53, -46, 39, -31).
- [measured] T372's gradient predicts the response of P(E1) to scaling the n6<->n7 coupling, the n2->n6 weight and
  the n2->n7 weight to within 0.94-0.97x (G2 GO).
- [measured] n2's gain elasticity, unresolved in T372 (-44 ± 42), is about -11 to -15 (own -15.4 ± 2.6; truth at
  small steps -11.5 ± 1.5 / -12.4 ± 3.9), and P is strongly curved along it (falls ~11% for ±3% gain).
- [measured] The n2->n6 and n2->n7 elasticities are equal and opposite (+87 / -89) and cancel when scaled together
  (0.4 ± 1.0), consistent with T372's "balance" reading. [inferred] The error depends on the difference of n2's
  drive into n6 and n7 rather than its overall size. [speculative] The 12-18x jump in P at ±3% joint scaling is a
  different error mode switching on, not curvature of the same one.
- [inferred] T372's mechanistic reading (comparators n6/n7/n8 and clocks n11/n12 set the far-gap error rate, with
  n6 vs n7 balance central) survives this check at the level of first-order sensitivities; it is not by itself a
  causal circuit account (gains and single-weight scalings, not path patching).

## Alternatives not ruled out
- Both line estimators share the boundary-term formula; agreement with indicator-only truths rules out a shared
  first-order bias at the resolved items, not at small items (|D| < 2) that the truths do not resolve.
- The truths share the conditional-law sampler formula with T372 (own implementation; p_gap checked by brute force).
- Sensitivities are local. Finite interventions are non-linear (n6 +3%: P x12.8; joint n2 ±3%: P x12-18), so the
  elasticities do not predict the effect of larger edits.
- Only 2 models and one threshold each; T372's s526 (dev) was not re-checked.
- The n7/n1 tensions (2.5-3.3 s.e.) may indicate optimistic per-line s.e. or a small bias; not chased.
- The pre-registered level rule (|log ratio| <= .3) picks eps = .03 for small elasticities, which is where it fails;
  a rule that also required agreement with the next level would have avoided the 4 G1 failures (post hoc).

## Failure modes
- FD truths on log-convex/curved P: the largest-acceptable-step rule picks steps where the symmetric difference is
  biased for small-|D| directions (n2 orig: 0 at .03 vs -12 at smaller steps). Ladders must be read whole.
- Line estimator: cost scales like 1/(hit fraction); lines along low-rank inputs hit rarely (r4: 1.3e-4 per line),
  so the line family matters (r1 is ~13x better than r3 on orig here; model- and event-dependent).
- Per-line gradient terms carry 1/|dm/ds| (tangent crossings): heavy tails possible; the largest single line's
  share of any reported projection is at most 0.28% (orig) and 0.05% (s528) here (`score.est_dirs` field top).
