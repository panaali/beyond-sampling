# T405: independent re-check of T386's family-T rare-event gradient (T5/gl20, frozen) on fresh models

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · checks [T386](https://github.com/panaali/clean-neurons/tree/claude/t386-gradient-dealias/m16/approaches/rare_gradient_dealias) · truths as [T372](https://github.com/panaali/clean-neurons/tree/claude/t372-rare-gradient/m16/approaches/rare_gradient)

## Question
T386's secondary result: family T, the K3 closure estimand integrated along lines in g12 (the top input) with
T372's exact boundary-term derivative, config **T5/gl20**, gave a deterministic weight-gradient of P(E1) that was
within 2x with the right sign on 10/12 directions (s528 + orig) at 2.8-3.7x the K3 walk. Family T was added after
s526 dev data had been seen, and one agent with one implementation ran it. **Does the same frozen rule, unchanged,
pass T372's go line on fresh models and thresholds, against truths from independent code?**

Event: E1(t0) = {argmax of the logits = position of the rank-3 input, and g23 = v2 - v3 > t0}.

## Method (pre-registered 2026-10-19, before any T5, K3 or truth run on the models below)

**Estimator (frozen, not retuned).** `transversal.est_grad(model, t0, 5, "gl20")` from claude/t386-gradient-dealias
@d708a676, imported read-only from a checkout; `frozen.py` asserts the sha256 of the six files that define it
(transversal.py, dealias.py, t372_k3src.py, t372_lgrad.py, t372_walk.py, t372_rnn.py), and T386's t372_k3src.py
in turn asserts the upstream K3 / walk files. The DEV3 weights file is asserted too. No wrapper changes the
computation; T386's own chunking (45k rows) is kept. Only paths are set (env variables T386 already reads).
pred = g . u is the absolute directional derivative, as in T372 / T386.

**Models and thresholds (fixed now).** dev3_s529, dev3_s530, dev3_s531 (DEV3 retrains, seeds 529-531; not HP5).
None appears in any rare-gradient INDEX row (T372, T385, T386, T402). Thresholds t0 = 2.35 and 2.75 for each:
T366's model-free thresholds. Disclosed: these models and thresholds were used for *value* estimation in T366 / T368
(K3 est / truth .88-1.23, P 1.2e-8 to 7.2e-7, not triple-tie-dominated). This was a deliberate choice: the test is
of the gradient rule, so the models are ones where the closure value bias is known to be small (< 1.25x). A failure
here cannot be blamed on a known closure failure. No gradient of any kind has been computed on them before this
commit. 6 cases, 36 directions.

**Directions (T372's definition).** Per case (index ci = 0..5 in the order s529/2.35, s529/2.75, s530/2.35, ...):
5 random unit directions in the 432-weight space (numpy default_rng(405000 + ci), standard normal, normalised;
order Wi (16), Wh (16x16 row-major), Wo (10x16 row-major)) plus the top direction g_K3 / |g_K3|, with g_K3 the
exact gradient of the K3 estimate (`dealias.est_grad(model, t0)`, which T386 showed reproduces T372's K3 bit for
bit). Step delta_u = 0.3 P_K3 / |g_K3 . u|, capped at 1.0. `make_cases.py`.

**Truth (own code, sampling-class).** `sampler.py` + `truth.py`, written for T405 and sharing no code with T364's
truthc or T372's truth_fd2. Exact conditional law of x given g23 > t0 (positions a uniform permutation; v2 by
inverse CDF of 90 phi(v) (1 - Phi(v)) Phi(v - t0)^8 on a fine grid; v1 truncated above v2; the 8 lows iid truncated
below v2 - t0), own batched RNN forward. Central finite differences with common random numbers (the same draws for
the base model and W +- delta u for every direction and every level), ladder delta_u x {1, 1/3, 1/10, 1/30, 1/100},
the FD picked by T372's A1 rule (largest level with |P+/P - 1| <= .5 and |P-/P - 1| <= .5; smallest if none).
s.e. from the per-draw paired differences. **N = 2e7 conditional draws per case**, seeds 405100 + ci. Resolved:
|FD| > 3 s.e.

Checks of the truth code (done before this commit, on a non-scored model): the sampler's v2 / v3 means and p_gap
match quadrature (t0 = 1: E v2 1.64246 vs 1.64244, s.e. 1.4e-4; E v3 .392667 vs .392675, s.e. 8e-5; p_gap equal to
1e-13); the forward equals T372's on 20k inputs. `check_code.py` (results/check_code.txt): base P on dev3_s528
E1(2.75) = 2.711e-9 +- 5.0e-11 vs T372's truth 2.72e-9 +- 5e-11; frozen T5 reproduces T386's stored s528 est and 432-gradient exactly (max diff 0).

**Go line (T372's, pooled over the 36 directions).** GO if at least 80% of the resolved directions have
sign(pred) = sign(FD) and 0.5 <= pred / FD <= 2, and at least 24 of 36 are resolved. If fewer are resolved, the
unresolved cases are rerun at N = 6e7 (same seeds, longer stream) before any verdict; if still fewer, INCONCLUSIVE.
Otherwise NO-GO. Reported, not gating: per-case and per-model pass counts; the top direction alone; K3's own
gradient on the same directions (aliasing reference); value est / truth P for T5 and K3.

**Cost.** Executed FPE (5948 FLOPs) as T386 counts them: T5 walk + endpoint backward passes, against the K3 walk
on the same case (both from the frozen code's counters).

**F against both bars (Directive 8).**
- Value: F_P = (est_T5 - P)^2 C_T5 / (P (1 - P)), against plain sampling at equal FPE (1 FPE per sample). Known to
  be degenerate for rare P (a zero estimate also passes when P C << 1); reported for completeness.
- Gradient (the quantity under test): sampling-class baseline = the same CRN central difference with **plain**
  (unconditional) sampling at the selected step, 2 FPE per draw, N_eq = C_T5 / 2 draws. Its variance per
  direction from the truth run's paired-difference moments: V = p_gap E_c[d^2] - (p_gap E_c[d])^2, divided by
  (2 delta)^2 N_eq (d = hit+ - hit- per conditional draw). F_g = sum over directions (pred - FD)^2 / sum V, per
  case and pooled; strict F_g <= 1, lenient F_g <= 10. The MSE numerator includes the truth's own s.e.^2 (it is
  reported) and the sampling side's curvature bias is ignored (both favour sampling). Also reported: the same
  with a *conditional-law* sampler (model-free exact law of g23 > t0 is known), which is the stronger baseline.

**Alternatives to keep in mind (rule 3).** (a) A pass could come from one dominant direction (top) carrying
most of the signal: the random directions are scored separately. (b) A failure could be the closure bias (K3's
estimand), not the T rule: T386 measured T's ratios 1.1-1.6 tracking value bias 1.14-1.47x; the value ratio
est_T5 / P is reported per case, and these models have small K3 value bias. (c) A pass/failure could be a truth
artefact (curvature at the chosen ladder level): the whole ladder is reported, and the K3 gradient (known to alias)
serves as a negative control: if K3 also "passes" broadly, the test does not discriminate.

## Status
**Done.** Pre-registered verdict: **NO-GO** (T5/gl20 passes 17/35 resolved directions, 49%; go needs >= 80%).

## Results
All from `python score.py` (-> `results/score.txt`, `results/score.json`) on `results/case_*.json`
(`python make_cases.py dev3_s529,dev3_s530,dev3_s531`) and `results/truth_*.json` (`python truth.py <model> <t0> 2e7`).
Values are d log P per unit step along u, (g . u) / P_truth; truth = CRN FD at the A1-selected level, N = 2e7.

| case | truth P (± s.e.) | T5/P | K3/P | T5 cost / K3 walk | T5 pass | K3 pass | T5 top: pred / truth |
|---|---|---|---|---|---|---|---|
| s529 E1(2.35) | 1.433e-7 ± 8e-10 | 1.24 | 1.09 | 3.86x | 3/6 | 1/6 | 45.2 / 57.7 ✓ |
| s529 E1(2.75) | 1.166e-8 ± 7e-11 | 1.14 | .87 | 4.33x | 3/6 | 2/6 | 20.2 / 43.8 ✗ (.46) |
| s530 E1(2.35) | 7.145e-7 ± 2e-9 | 1.16 | 1.24 | 2.12x | **0/5** | 2/5 | 2573 / 243 ✗ (10.6) |
| s530 E1(2.75) | 7.030e-8 ± 2e-10 | 1.27 | 1.23 | 2.32x | **1/6** | 2/6 | 1212 / 262 ✗ (4.6) |
| s531 E1(2.35) | 2.415e-7 ± 1e-9 | 1.11 | 1.14 | 2.78x | 4/6 | 2/6 | 373 / 284 ✓ |
| s531 E1(2.75) | 1.922e-8 ± 1e-10 | 1.06 | 1.23 | 3.04x | 6/6 | 5/6 | 303 / 328 ✓ |
| **pooled** | | 1.06-1.27 | .87-1.24 | 2.1-4.3x | **17/35 NO-GO** | 14/35 | 3/6 top |

- 35 of 36 directions resolved (|z| 5-247; the unresolved one is s530/2.35 dir 4, truth ~0). Per model: s529 6/12,
  s530 1/11, s531 10/12. Random directions alone: 14/29. Per-direction table: `results/score.txt`.
- The truths are not the failure [measured]. P agrees with T366's GPU truths (1.43e-7, 1.16e-8, 7.16e-7, 7.02e-8,
  2.43e-7, 1.91e-8) within 0.1-1.3%. On s530 the FD ladder is flat over a 100x range of steps (top: 243 / 247 / 248 /
  254 / 248, s.e. 1-11), so the selected truth is not curvature-biased there. T5 overshoots by 5-11x.
- **Diagnostic (post hoc, not pre-registered; `python diag_fd.py`, `results/diag_fd.log`) [measured].** The central
  FD of the frozen T5 *value* along s530 / 2.35 top: steps 1e-5 / 1e-4 / 7e-4 / 3e-3 give 1485 / 428 / 323 / 210
  (truth 243, analytic T5 2573). s530 / 2.75 dir 3: steps 1e-5 to 1e-2 give -113 / -72 / -38 / -25 (truth -17.5,
  analytic -123). T5's value tracks P (1.16-1.27x), and its slope over a finite step approaches the truth. The
  pointwise derivative does not. This is the aliasing signature T372 found for K3 along g23, now with lines along
  g12 on this model.
- Value: T5 est / truth 1.06-1.27 (all high, as T386 saw); K3 .87-1.24.

**Cost** (executed FPE; walk + endpoint backward): T5 0.76M-1.81M FPE, **2.1-4.3x the K3 walk** (0.32-0.47M). T386
reported 2.8-3.7x. Truths: 2e7 draws x 61 forwards per case (1.2e9 forward passes per case, 7.3e9 in total).

**F against both bars (pre-registered definitions).**
| | s529/2.35 | s529/2.75 | s530/2.35 | s530/2.75 | s531/2.35 | s531/2.75 | pooled |
|---|---|---|---|---|---|---|---|
| F_g vs plain CRN-FD sampling at equal FPE | .0021 | .0005 | 7.6 | .11 | .0018 | 1.6e-5 | **4.1** (strict fail, lenient pass) |
| F_g vs conditional-law sampling | 23 | 54 | 8.2e4 | 1.2e4 | 20 | 1.7 | **4.8e4** (both fail) |
| F_P (value) vs plain sampling | .014 | 3.8e-4 | .013 | .0038 | .0030 | 6.5e-5 | degenerate (P C << 1) |

Against plain sampling, which sees almost no flips at T5's cost, T5 wins on 5/6 cases. The pooled ratio of 4.1
comes from s530. Against sampling that uses the model-free exact law of g23 > t0, a baseline anyone could build,
T5 loses by 1.7x to 8e4x.

## Compute accounting
FPE = 5948 FLOPs. Estimators as counted by the frozen code: T5 0.76-1.81M FPE per case (all 432 derivatives);
K3 0.32-0.47M. Truth: 7.3e9 RNN forward passes (6 cases x 2e7 draws x 61 weight sets) plus the 1e7-draw sampler
check; about 14 CPU-min per case. Diagnostic: 16 T5 value runs on s530. Model-free tables (v2 inverse-CDF grid, 2e5
cells) are not charged.

## Claims
- [measured] **Pre-registered: NO-GO.** The frozen T5/gl20 gradient (T386, sha256-asserted, unchanged) passes
  T372's go line on 17/35 resolved directions on 3 fresh DEV3 models x 2 thresholds (go >= 80%). T386's 10/12 does
  not replicate as a general property of the rule. Scope: E1, P 1.2e-8 to 7.1e-7, non-triple-tie models, T5/gl20 only
  (T8/mid20x4 not tested).
- [measured] The failure is model-dependent. s531 passes 10/12 and s529 6/12. s530 passes 1/11, with ratios
  4.6-11 on the top direction, although T5's value there is within 1.27x.
- [measured, post hoc] On s530, T5's value differences approach the truth as the step grows while its pointwise
  slope is 5-10x too large: the transversal lines alias too.
- [inferred] T386's claim that the g12 orientation removes aliasing holds on the models used to develop it (s526,
  s528) and on s531. It is not a property of the rule: whether lines along g12 cross the islands transversally
  depends on each model's island geometry.
- [measured] K3's own gradient (negative control) passes 14/35, so this test does discriminate. T5 does somewhat
  better than K3 (17 vs 14) but far from 80%.

## Alternatives not ruled out
- The truth for single directions at level >= 1 on s529 (where P moves 2-40x at the largest step) might carry some
  curvature, but the A1 rule picks levels with P changes <= 50%, and s530 (the main failure) has a flat ladder.
- A finer T config (T8/mid20x4) or an adaptive rule might pass. Not tested: this re-check freezes T5/gl20 as asked.
- The s530 failure was diagnosed on two directions only. Which endpoints dominate (near-tangent |dm/dg12|) was not
  isolated.
- These models were used for value estimation in T366/T368. They were not chosen by gradient behaviour (no
  gradient had been computed on them), but they are not a random draw from DEV3.

## Failure modes
- A deterministic fixed-line boundary gradient can alias in any line orientation. Its value and finite-step slope
  can be fine while the pointwise derivative is 5-10x off (s530). Before trusting such a gradient on a new model,
  compare it with a finite difference of its own value at a step comparable to the line spacing (`diag_fd.py`).
- T's value is one-signed high (1.06-1.27x, closure bias), so even a de-aliased T gradient runs about 1.1-1.3x
  high.
