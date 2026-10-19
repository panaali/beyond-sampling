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

## Status
Pre-registered; amendment A1; truth runs in progress.

## Results
(none yet)

## Compute accounting
FPE = 5948 FLOPs (cf.FPE). Counted: walk cost (as rare.walk3 charges), the mass sums, and per endpoint one forward +
backward pass (counted FLOPs in `rgrad.py`). Truth MC forward passes are reported separately (not part of the method).

## Claims
(none yet)

## Alternatives not ruled out
See the pre-registration above.

## Failure modes
Expected: tangent crossings (dm/dt ~ 0) giving heavy-tailed terms; triple-tie islands that K3 cannot resolve (not in
this test's models); the closure bias.
