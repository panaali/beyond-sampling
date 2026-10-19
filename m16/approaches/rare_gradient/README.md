# T372: exact weight-gradient of a rare-event probability along the exact walks

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question
Which of M_16,10's 432 weights (and which neurons) drive a rare error? Concretely: can the exact gate-cell walks
that estimate P(E1) (H-R1, T340c / T354 / T364) also give dP(E1)/dW for all 432 weights at once, correctly, at the
cost of about one backward pass per wall crossing? Plain Monte Carlo on an indicator has no finite-variance unbiased
weight-derivative (the indicator's derivative is 0 a.e.; finite differences need the flipping samples), so this is a
question a mechanistic method can answer and sampling cannot answer directly.

Event: **E1(t0) = {argmax of the logits = position of the rank-3 input, and g23 = v2 - v3 > t0}**.

## Method (pre-registered 2026-10-18, before any validation run)

**Estimator to differentiate (primary, deductive).** The frozen K3 estimator of T364: T340c (`t340c.est_c` lines,
K = 3 Gauss nodes in g12 | v2) with the T354 segment start (`t354.est_seg`, walk from t = t0) and the T354 walk fix
(`walkfix.walk3_fixed`), T_hi = 6. Read-only code: `claude/t218-tie-tail-labels` r1_cont/{t340c,t354,walkfix}.py and
`claude/t286-cheap-frontier @620eb47b` (rare_events/rare.py, closure_floor/cont_g.py), checked out in a scratch
worktree (hashes asserted against r1_cont/t366_freeze_sha256.txt). Reproduced before this pre-registration (no
gradient code yet): dev3_s526 E1(3.25) 1.2100e-9, dev3_s528 E1(2.75) 3.577e-9, orig E1(1.25) 1.342e-6, equal to the
published K3 values.

The K3 estimate is a fixed linear combination of line masses,
`est = sum_rows w_row sum_pieces [Phi(v2 - a)^8 - Phi(v2 - b)^8]`, where each line is x(t) = x0 + t dx (t = g23),
the weights w_row and the lines are model-free, and [a, b] are the intervals on which the rank-3 position is the
argmax (a = max(lo, t0)). Gate flips inside a merged interval cancel (the network is continuous), t0 and T_hi are
model-free, so the only weight-dependence is through the **interval endpoints that are roots of an argmax margin**
m(t) = L_p3(x(t)) - L_j(x(t)) = 0. By the implicit-function theorem, de/dW = -(dm/dW)/(dm/dt) at the root, so

`d est / dW = sum_rows w_row sum_endpoints e  f(e; v2) * (dm/dW)(x(e)) / |dm/dt (e)|`,  f(e; v2) = 8 Phi(v2-e)^7 phi(v2-e)

(the Reynolds-transport / boundary term restricted to the walk lines; the sign is automatic: raising the margin at
either end widens the interval). dm/dW for all 432 weights and dm/dx (hence dm/dt = dm/dx . dx) come from one
per-endpoint backward pass through the RNN (gates fixed at the endpoint). This is the **exact derivative of the
deductive K3 estimate**; it inherits K3's quadrature/closure bias (H-R1: K3 within ~1.4x on non-triple-tie models),
which is what the validation below measures.

**Secondary estimator (sampling-class, unbiased, not gating).** Conditional MC along exact v3 lines (T362 / Fable L):
draw x from the exact law given g23 > t0 (the T364 truth sampler, model-free), drop v3, and walk the line x_p3 = v
over v in (v4, v2 - t0) exactly with my own walker (`walk.py`, written independently of rare.walk3). Z = exact mass of
{pick = p3} on the line under phi restricted to the window, P = P(g23 > t0) E[Z], and the same boundary term gives
dZ/dW. Unbiased for dP/dW with finite variance; reported with its s.e. next to the primary.

**Implementation checks (before validation).** (i) Tiny brute-force case: a random ReLU RNN with 2 inputs, H = 3,
C = 3, event {argmax = 0}; P(W) by a fine 2-D grid, its central finite differences vs the line-boundary gradient
(lines in x1, Gauss nodes in x2) with my walker and backward code. (ii) On the real models: the analytic directional
derivative of the K3 estimate vs central finite differences of the K3 estimate itself (small delta), along 3 random
directions. Both must agree to < 1% relative (else the code is wrong and is fixed before validation).

**Models and thresholds (fixed now).** M_16,10 (`orig`) at t0 = 1.25 (truth 1.09e-6, K3 1.24x); dev3_s526 at
t0 = 3.25 (truth 1.01e-9, K3 1.20x); dev3_s528 at t0 = 2.75 (truth 2.72e-9, K3 1.31x). All three are in H-R1's
within-2x scope; s526 and s528 are not triple-tie-dominated (T364/T366).

**Directions.** Weight vector w = concat(Wi (16), Wh (16x16, row-major), Wo (10x16, row-major)), 432 entries. Per
model: 5 random unit directions (numpy default_rng(372000 + model index 0/1/2), standard normal, normalised) plus the
top-gradient direction u* = g_K3 / |g_K3|: 18 directions in total.

**Step.** delta_u = 0.3 P_K3 / |g_K3 . u| (predicted |dP| / P = 30% per side, inside the 20-50% band), capped at
delta_max = 1.0 (unit direction, L2 in weight space).

**Truth.** Conditional-law MC (T364 `truthc.py` sampler, copied unchanged into this folder) with common random
numbers: the same draws x feed the base model and W +- delta_u u for every direction. FD_u = p_gap (hits+ - hits-)
/ (2 delta_u N); its s.e. from the per-sample paired differences. N: orig 1e8, s526 2e7, s528 5e7 (seeds 372100 +
model index). The base truth P is also recorded (sanity vs the published truths).

**Go line (fixed in advance).** A direction is *resolved* if |FD_u| > 3 s.e. **GO** if, pooled over the 18
directions, at least 80% of the resolved directions have sign(g_K3 . u) = sign(FD_u) AND 0.5 <= (g_K3 . u) / FD_u
<= 2; and at least 12 of 18 are resolved (otherwise **inconclusive**: rerun with more N, not a verdict). The same
table is computed for the secondary estimator (reported, not gating), and per model.

**Reported regardless of verdict.** Cost (FPE, executed: the walk's cost C plus endpoints x the backward cost),
number of endpoints; top-20 weights by |g| and by elasticity w dP/dw / P; per-neuron elasticities
(E_in(i) = sum of w dP/dw over weights into neuron i, E_out(i) = over weights out of i: d log P / d log of neuron
i's outgoing gain); mapped onto the T241 neuron roles for M_16,10 (running max n2, record detector n4, x9 comparators
n1/6/7/8, clocks n0/10/11/12/14/15, late-x8 n5/n9, period-4.32 mode n3/n13/n15).

**Alternatives to keep in mind (rule 3).** (a) A K3 gradient could agree with FD because both are dominated by a
global scale effect (e.g. Wo rows of p3 vs p2), not because the walk resolves the mechanism: separated by the random
directions (which mostly miss the top direction) and by the per-weight table vs a direct FD on single weights.
(b) A failure could be the closure (lows pinned at mu(v4), T369) rather than the boundary formula: separated by the
secondary unbiased line estimator, which has no closure. (c) Large delta curvature: central FD at 30% changes has
O(delta^2) error; reported via the base truth vs the mean of the +-delta truths.

### Amendment A1 (2026-10-19, after the first truth run on dev3_s526 only, before any other truth run)
The first literal run (truth_fd.py, dev3_s526, the pre-registered delta rule) showed that the rule's linearity
assumption is badly wrong: at steps predicted to change P by 30%, the realised P(W + delta u) / P and P(W - delta u) / P
were 2.25 / .25, 21.4 / 14.7, 6.2 / 21.3, 1.93 / .76, .20 / 1.91, 2.03 / .11 (P rises on BOTH sides along two random
directions: deep-tail P is strongly log-convex in W). A central difference over such a step is not a derivative, so
this run cannot test a gradient; it is kept (results/truth_fd_dev3_s526.json) and scored literally (signs 6/6, within
2x 0/6). Amendment: each frozen direction is probed at the ladder delta_u x {1, 1/3, 1/10, 1/30, 1/100} on the same
draws (truth_fd2.py, new seeds 372110 + model index); the FD used for the go line is the one at the **largest** level
whose realised changes satisfy |P+/P - 1| <= .5 and |P-/P - 1| <= .5 (the smallest level if none does). Directions,
delta_u, models, the go line and the resolution rule (|FD| > 3 s.e.) are unchanged. N: s526 2e7, s528 5e7, orig 5e7
(orig lowered from 1e8 because the ladder costs 4.7x more forward passes per draw). The whole ladder is reported, so
the convergence of FD as delta -> 0 is visible.

### Amendment A2 (2026-10-19 00:45, DRI request; before any s528 or orig truth output was seen)
(a) dev3_s526 is **development**: A1 was written after its first truth run, so it does not count towards the go line.
The go line is scored on s528 + orig only (12 directions): GO if >= 80% of the resolved directions pass and at least
8 of 12 are resolved (2/3 of them, as the 12/18 rule had). s526 is reported in the same table, marked dev.
(b) The orig point: the event is E1(1.25) exactly as in T340 (rank-3 pick, g23 > 1.25); K3 = 1.342e-6 against the
T340 truth 1.086e-6 (1.24x), inside H-R1. (The 1.21e-9 in my message to the DRI was dev3_s526's K3, not orig's.)
(c) dev3_s526's truth_fd2 run stopped at N = 1e7 (pre-registered 2e7), resolved at |z| >= 25 on every direction, to
free a CPU slot for orig.
(d) A container restart (2026-10-19 ~01:30) ended the s528 and orig truth_fd2 runs at N = 1e7 (A1 named 5e7; s528
was started at 3e7). All 12 scored directions were already resolved (|z| >= 10), so they are scored at 1e7.

## Status
**Done.** Primary (exact gradient of the deductive K3 estimate): **NO-GO**, 2/12 resolved directions pass (go needs
>= 80%). Secondary (unbiased conditional MC along exact v3 lines, not gating): **GO-level**, 11/12 pass.
Diagnosis (post hoc, measured): K3's P(W) is accurate but its pointwise slope is erratic.

## Results
All numbers are in `results/score.txt` (`python score.py`), from `results/dirs_*.json` (`python make_dirs.py <model>`),
`results/truth_fd2_*.json` (`python truth_fd2.py <model> 1e7`), `results/cmc_*.json` (`python cmcgrad.py <model> 5e6`).
Values are d log P / d(step along u) = (dP/du) / P_truth. "lvl" = ladder level chosen by A1's rule.

| model | dir | FD/P truth (lvl) | K3/P | K3/truth | line-CMC/P | CMC/truth |
|---|---|---|---|---|---|---|
| orig E1(1.25) | 0 | 40.8 ± 2.6 (2) | -5.3 | -0.13 ✗ | 38.7 ± 5.4 | 0.95 ✓ |
| | 1 | -46.8 ± 2.7 (3) | -2.3 | 0.05 ✗ | -43.1 ± 7.1 | 0.92 ✓ |
| | 2 | 17.6 ± 1.6 (1) | 8.8 | 0.50 ✗ (0.497) | 21.6 ± 3.8 | 1.23 ✓ |
| | 3 | -34.0 ± 1.9 (4) | -0.16 | 0.00 ✗ | -31.0 ± 4.4 | 0.91 ✓ |
| | 4 | -28.5 ± 1.3 (1) | -7.9 | 0.28 ✗ | -27.6 ± 3.6 | 0.97 ✓ |
| | top | 390 ± 26 (1) | 217 | 0.56 ✓ | 411 ± 44 | 1.06 ✓ |
| dev3_s528 E1(2.75) | 0 | -17.5 ± 0.7 (2) | 4.0 | -0.23 ✗ | -18.4 ± 1.2 | 1.05 ✓ |
| | 1 | 3.4 ± 0.3 (0) | 11.7 | 3.46 ✗ | 0.3 ± 1.6 | 0.10 ✗ |
| | 2 | 20.2 ± 0.5 (1) | 7.1 | 0.35 ✗ | 22.3 ± 0.9 | 1.11 ✓ |
| | 3 | 24.3 ± 0.5 (1) | 7.8 | 0.32 ✗ | 26.1 ± 1.0 | 1.07 ✓ |
| | 4 | -12.2 ± 0.4 (0) | 13.8 | -1.13 ✗ | -17.5 ± 2.8 | 1.44 ✓ |
| | top | 423 ± 11 (0) | 454 | 1.07 ✓ | 404 ± 20 | 0.95 ✓ |
| *dev3_s526 E1(3.25), dev* | 0 | 15.8 ± 0.1 (1) | 4.2 | 0.27 ✗ | 19.9 ± 1.7 | 1.26 ✓ |
| | 1 | -0.82 ± 0.02 (1) | 0.87 | -1.05 ✗ | 1.4 ± 0.9 | -1.68 ✗ |
| | 2 | -1.57 ± 0.06 (2) | -0.88 | 0.56 ✓ | -2.9 ± 1.0 | 1.85 ✓ |
| | 3 | 13.5 ± 0.2 (2) | 3.2 | 0.24 ✗ | 15.7 ± 1.4 | 1.16 ✓ |
| | 4 | -38.4 ± 0.4 (2) | -6.8 | 0.18 ✗ | -41.3 ± 2.8 | 1.08 ✓ |
| | top | 298 ± 3 (2) | 58 | 0.19 ✗ | 320 ± 22 | 1.08 ✓ |

- **Go line (s528 + orig, A2): K3 2/12 pass -> NO-GO [measured].** Line-CMC 11/12 -> would be GO [measured] (not
  gating). The literal pre-registered delta on s526 (dev): K3 1/6. Against the not-pre-registered "converged" FD
  (inverse-variance mean of ladder levels 2-4): K3 1/11, CMC 10/11. orig dir 3 hit the delta cap (1.0), its ladder
  is not converged; it does not change either verdict.
- **P itself:** truths orig 1.143e-6 ± 4e-8, s526 1.008e-9 ± 6e-12, s528 2.72e-9 ± 5e-11; K3 1.17x / 1.20x / 1.31x;
  line-CMC 1.083e-6 ± 4.7e-8 / 1.007e-9 ± 8.6e-12 / 2.68e-9 ± 5.6e-11 (all within 1.4 s.e. of the truths) [measured].
- **Code checks [measured]:** tiny brute-force RNN, boundary gradient vs FD of the exact line mass: max rel. error
  3.8e-10, vs a brute-force grid FD within 1-4% (`python tiny_check.py`). Analytic K3 gradient vs central FD
  (delta 1e-5) of the K3 estimate: max rel. error 8e-5 over 9 directions on the 3 models; K3 values reproduce the
  published ones exactly (`python check_k3fd.py <models>`). Exact identities hold for both estimators: the sum of
  the Wo elasticities is 0 (argmax is invariant to scaling Wo), and each neuron's in- and out-elasticities are equal
  (ReLU positive homogeneity).
- **Steepness [measured]:** deep-tail P is extremely weight-sensitive and log-convex: |d log P / dW| = 693 (orig),
  330 (s526), 499 (s528) per unit weight-space step; P doubles for a step of 0.002-0.006 along the top direction,
  and rises on both sides along some random directions at steps ~0.4.

### Why K3's P is right but its gradient is not (post hoc diagnostics, not pre-registered)
- `python k3_moved.py <models>`: K3 at the moved weights W ± delta u_top tracks the truth P(W ± delta u_top) (ratios
  .84-1.75; s526: 1.00 / 1.39 / 1.11 / 1.00). K3's own FD at delta/3 on s526 gives d log P = 246 vs the truth's 254,
  while its derivative AT W is 48 [measured].
- `python k3_jitter.py dev3_s526`: the analytic K3 slope along u_top at W + s u_top for s = -1e-3, -3e-4, -1e-4, 0,
  1e-4, 3e-4, 1e-3 is d log P = 284, 430, 74, 48, 37, 32, 42 (truth ~300) [measured]. **K3's estimate is a sum of a
  finite number of line-island masses; as W moves, islands slide across the fixed quadrature lines, so the estimate is
  accurate in value but bumpy at the scale of the line spacing, and its pointwise derivative is dominated by the few
  lines that currently cut an island near its tip.** [inferred from the two diagnostics] The boundary formula itself
  is right (the line-CMC uses the same boundary terms with randomised lines and is unbiased), and the moving-node
  explanation does not apply (nodes, lines and closure are model-free).
- cos(g_K3, g_CMC): s526 .97, s528 .82, orig .63; K3's projection on the CMC gradient: .17x / .76x / .21x [measured].

### Mechanistic reading: which weights and neurons drive the rare error (line-CMC gradient; `python analyze.py`)
Neuron elasticity E(i) = d log P / d log(gain of neuron i) = sum of w dP/dw / P over neuron i's incoming weights
(equal to the outgoing sum by homogeneity). orig (M_16,10), E1(1.25), s.e. from the per-line projections
(`results/cmc_orig.json` neuron_elast):

| neuron (T241 role) | E(i) | | neuron | E(i) |
|---|---|---|---|---|
| n6 (x9 comparator) | **+112 ± 12** | | n1 (x9 comparator) | -19 ± 6 |
| n12 (clock) | **-44 ± 5** | | n13 (4.32 mode) | -16 ± 2 |
| n7 (x9 comparator) | **-43 ± 10** | | n0 (clock) | -14 ± 7 |
| n2 (running max) | -44 ± 42 (unresolved) | | n10 (clock) | -8 ± 2 |
| n8 (x9 comparator) | **+38 ± 7** | | n5, n9 (late-x8) | +5 ± 2, +3 ± 0.5 |
| n11 (clock) | **-30 ± 3** | | n4, n15, n14, n3 | \|E\| <= 5 (n4, n15 unresolved) |

- [measured] The rare far-gap error is controlled mostly by the **x9 comparators**, with opposite signs: to first order a 1%
  higher gain on n6 multiplies P(E1) by e^1.12 ~ 3 and n8 by e^.38 ~ 1.5, while n7 and n1 lower it. Then the **clocks n12, n11** (lowering it), and the 4.32 mode n13. The running max n2 has a large but
  unresolved elasticity; the record detector n4 and the late-x8 neurons n5/n9 barely matter.
- Top single weights (d log P / dw, ± s.e.): Wh[n6<-n10] +174 ± 18, Wh[n6<-n2] +152 ± 18, Wh[n6<-n7] +149 ± 18,
  Wh[n7<-n7] -147 ± 17, Wh[n7<-n6] -143 ± 17, Wh[n12<-n2] -134 ± 18, Wh[n7<-n2] -127 ± 18. By elasticity:
  Wh[n7<-n2] -86 ± 12, Wh[n6<-n2] +81 ± 10, Wh[n8<-n7] -58 ± 7, Wo[o2<-n8] -44 ± 8, Wh[n8<-n1] +44 ± 5. 275 of the
  432 gradient entries are resolved at 3 s.e. [measured]
- [inferred] Reading: the error needs the comparator pair n6/n7 to mis-rank the 2nd and 3rd values. The most sensitive
  weights are the running-max input into the comparators (n2 -> n6 positive, n2 -> n7 negative) and the n6 <-> n7
  mutual coupling: the error probability depends on the *balance* between n6 and n7, not on any one neuron's size.
  [speculative] A training signal that pushes this balance would trade this rare error against others.
- K3's neuron map agrees in places (n6 +28, n2 -33, n8 +18, n12 -18) but not in others (n7 +2 vs -43, n1 +10 vs -19):
  cos .63, too unreliable for a mechanistic reading.
- s526 / s528 (no role map): the drivers are model-specific (s526: n10 +48, n8 +27, n11 +27, n9 -25, n5 -23;
  s528: the incoming row of n3 dominates the top weights). See `results/mech.txt`.

## Compute accounting
FPE = 5948 FLOPs. Counted per method (executed FLOPs):
- **K3 gradient:** the walk C_walk = 574k / 325k / 388k FPE (orig / s526 / s528), plus C_grad = 5.0k / 2.4k / 3.3k FPE
  for 1687 / 797 / 1110 endpoints (~3 FPE per endpoint: one forward + one backward, `rnn.flops_per_grad`). So all 432
  derivatives cost +0.7-0.9% on top of the estimate [measured], but they are wrong (above).
- **Line-CMC (secondary):** 5e6 lines: 58M / 33M / 43M FPE (the walker's executed FLOPs plus endpoint backward passes;
  model-free sampling not counted). It gives all 432 derivatives at once, unbiased.
- **Truths (sampling, not part of either method):** 1e7 conditional-law draws x 61 forward passes = 6.1e8 passes per
  model (CRN FD for 6 directions x 5 steps x 2 signs). Per direction and step, CRN FD on the conditional law is
  cheaper than the line-CMC (orig top: s.e. 26 from 2e7 passes vs line-CMC s.e. 44 at 5.8e7 FPE), but it needs one
  run per direction (864 passes per draw for all 432 weights) and a step whose curvature bias had to be controlled by
  the ladder; plain MC on an unconditioned indicator would need ~1/P ~ 1e6-1e9 draws per hit.

## Claims
- [measured] The exact derivative of the deductive K3 estimate is computable for all 432 weights at <1% extra cost,
  and is the correct derivative of that estimate (FD to 8e-5).
- [measured] It fails as an estimate of dP(E1)/dW: 2/12 pass on s528 + orig, 3/18 overall; errors of both signs,
  magnitudes 0.0-3.5x. **Pre-registered verdict: NO-GO** (scoped: K3 = T340c/T354 lines with K = 3 g12 nodes, 20 v2
  nodes per region, T_hi 6, these 3 models and thresholds).
- [measured] The same boundary terms on randomised exact lines (conditional MC along v3) are accurate: 11/12 within 2x
  on s528 + orig (16/18 overall), all P estimates within 1.4 s.e. Unverified; one agent, one implementation.
- [inferred] The K3 failure is quadrature aliasing of the derivative (estimate bumpy at the line spacing), not the
  boundary formula or the closure; supported by the moved-weight diagnostic (3 models) and the jitter diagnostic (s526 only).
- [measured] On M_16,10, E1(1.25) is driven by the x9 comparators n6 (+) / n7 (-) / n8 (+) / n1 (-) and the clocks
  n12, n11 (-); the most sensitive weights are n2 -> n6, n2 -> n7 and the n6 <-> n7 coupling.

## Alternatives not ruled out
- The secondary estimator was not pre-registered as the gating test, and its s.e. are heavy-tailed (1/|dm/dv| near
  tangent crossings): a few lines may carry much of a projection. Its 11/12 is consistent with unbiasedness but
  should be re-checked by an independent agent with more lines and different seeds.
- The truth FD is a finite difference: level choice (A1) leaves some curvature bias (e.g. s526 dir 1: -0.82 at
  level 1 vs ~0.2 at small steps). Both verdicts are unchanged against the converged FD.
- A smoothed K3 gradient (K3 FD at delta ~ the line spacing, or K3 averaged over randomly shifted lines, i.e. RQMC)
  might fix the primary; not tried (it would cost 2 K3 runs per direction or many K3 runs for all 432 weights).
- More K3 lines (K5, T366 graded rule) may reduce the aliasing; not tried.
- The neuron elasticities are local (first order). Given the log-convexity, finite changes of a neuron's gain
  have different effects; no ablation was run.

## Failure modes
- Deterministic quadrature (K3): derivatives alias at the line spacing (above); any fixed-node exact estimator
  should be expected to have this problem for gradients, even where its values are accurate.
- Line-CMC: heavy-tailed per-line gradient terms near tangent crossings; only 0.03-0.3% of lines hit the event, so
  cost scales like 1/(hit fraction).
- FD truths: deep-tail P is strongly log-convex in W, so the step must be ~1/100 of a naive linear guess.
- Triple-tie-dominated models (not tested here) would add K3's known value failures on top.
