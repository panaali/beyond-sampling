# Fable review, round 6 (2026-10-18): deductive integration of closure c to RMS(est - E0) <= .15 pp at <= 3.3k FPE

Read (nothing run; small arithmetic only): INDEX rows T324-T357 and H-L1/H-S1 (ccr-6a61f25c-wn80ii); T356
exhaustion checklist and T357 re-check; my round-5 review; the T346 brainstorm seed; budget_t317d/README (T324, addenda
A/B, T347, T351, T353); r1_cont/README (T317-T335 cost table, T326 floors and tree note, T330, T331, T344); r1_walk/README
(T304/T306). **New arithmetic in this review** comes from the per-model paired outputs already on the branches:
`budget_t317d/results.json` + `results_e0.json` (80 DEV2, T324), `addendum.json`, `corner.json` (10 DEV2, T333),
`exhaustion/floors_dev3.json` (100 DEV3, T356) and `t356_recheck/e0e1.json` (100 DEV3, T357). Those numbers are labelled
**[measured; my arithmetic on the T324/T333/T356/T357 outputs]**; anyone can reproduce them in seconds. All child
results remain unverified unless INDEX says otherwise.

Conventions as in T356: strict means total RMS <= 100 sqrt(mean p(1-p) / C) with DEV3 mean p(1-p) = .0689; E0 = closure c
with the true v4, integrated exactly (DEV3 RMS .432, bias -.129, sd .414).

## 0. Summary of the verdict
**The target is not plausible for closure c, and existing paired data already say so.** The integration error of the
walks is a sum of four model-specific terms (tmax cut, line-set product, one transverse anchor, K = 1 v4 node) whose
model-to-model sd is .18-.37 pp **each**, on 80 DEV2 and (for the v4 node) on 100 DEV3. Each exceeds the whole .149 pp
allowance on its own; the only known fixes cost >= 1k FPE each; the cost ceiling with closure c is 3.7k FPE at perfect
integration, so the budget for all fixes together is 0-400 FPE. Two of the terms (K = 1 node and tmax cut) are
independent, cost-blocked kills. Below I still give the three best schemes with their arithmetic and pre-registerable
kill tests, say what each would and would not fix, and name the one cheap experiment that could move the ceiling
(a closure change, not an integration change).

## 1. The arithmetic that binds every scheme

### 1a. Cost ceiling [inferred from T356's measured E0]
100 sqrt(.0689 / C) = .432 at **C = 3,692 FPE**: no estimator of closure c costing more can be strict whatever its
integration. Allowance for the integration error (quadrature, errors uncorrelated as T356/T357 measured):

| C (FPE) | 2,200 | 2,500 | 2,900 | 3,307 (T317f) | 3,500 |
|---|---|---|---|---|---|
| allowance vs E0 (pp) | .356 | .298 | .226 | **.147** | .101 |

For the K = 1 node closure E1 (RMS .462) the ceiling is 3,228 FPE, already below T317f's cost: a perfect integrator of E1
is F 1.02 at 3.3k (T356). So the scheme must integrate the true v4 law.

### 1b. Component budget of T317d/f's integration error (new table)
Sequential switch-offs along T324's path, per model, paired on the same 2e5 inputs, pp; 80 DEV2 **[measured; my
arithmetic on results.json + results_e0.json]**. The last row is on 100 DEV3 **[floors_dev3.json]**.

| term (path step) | mean | sd across models | RMS | cheapest known fix | its cost |
|---|---|---|---|---|---|
| tmax cut (V1 - path1) | +.182 | **.225** | .290 | exact walk to tmax 1 (T331) | +52% C, dead; affine extension overshoots (-.40 bias) |
| blind g12 (path1 - path2) | +.097 | .048 | .108 | walk b-g12 (T306-A) | +2.4k, but sd is harmless |
| mu linearisation (path2 - path3) | -.018 | .072 | .074 | exact lows along line 23 | ~0, harmless |
| line-set product (path3 - path4 = J1) | +.332 | **.371** | .498 | T347 tilt removes the mean; residual sd .263 [inferred from corr .54] | ~.35k (T353 reuse) for the mean; sd needs 2-D sets (T333: >= 21k) |
| one transverse anchor (path4 - path5) | -.310 | **.177** | .357 | second anchor (J2-style) | doubles the walks (T306-B / T333) |
| K = 1 v4 node (E1 - E0), DEV2 | +.309 | **.173** | .354 | K = 2 nodes (T304 V2) or a v4 line | x2 cost / +1.3k |
| K = 1 v4 node (E1 - E0), **DEV3, 100 models** | +.302 | **.178** | .351 | same | same |
| **total V1 - E0** | +.593 | .446 | .742 | | |

Correlations between terms are small except tmax vs product (-.64): their sum has mean +.514, sd **.286**. With the anchor
and node terms removed *exactly* (what any 2-anchor, exact-v4 scheme would achieve), the remainder V1 - path4 has mean
+.593, sd **.310** [measured]. With a perfect mean correction on top, the sd floor of "walks + product at tmax .5" is
~.29-.31 pp, twice the allowance.

Three consequences, each sufficient on its own:
1. **Every scheme on the K = 1 node fails the DRI's overturning test by construction**: RMS(E1 - E0) = .35 on both panels,
   and even its sd (.17-.18) exceeds .149. A deductive v4 treatment (K >= 2 nodes, or a v4 line through the anchor at
   ~+1.3k) pushes C past the 3.7k ceiling. [measured + inferred]
2. **The tmax cut alone (sd .225) fails**, and its only accurate fix (walking further) costs +1.4-2.9k (T331). [measured]
3. **The product's model-specific part (sd .37 raw, ~.26 after the tilt) fails**, and resolving it needs the walls in the
   L-band, where cells are 10x denser than in the outer window (corner.json: 10.4 of 20.3 nb cells per triple lie in
   [0, .1]^2, 4% of the window). [measured]

The DEV3 integration error also scales with the model's error (T317f - E0: strong third +.39 +- .28, middle +.55 +- .43,
weak third +.75 +- .43; E1 - E0 has corr .64 with 1 - p on DEV3) [measured; e0e1.json, floors_dev3.json]. A deductive
scheme cannot condition on the truth; it can only see the wall mass its own walks find, which is what the tilt does.

### 1c. Why the walks' errors are expensive to remove: the per-triple view [inferred]
If per-triple errors were independent and zero-mean across the 720 triples, a total of .15 pp would allow a per-triple
error of .15 x sqrt(720) = 4 pp of conditional accuracy, a loose requirement. The walks' four terms are **systematic**
(same sign in every triple), so they add linearly; they are precise per triple and biased per model. Plain sampling at
N = 3,307 gets .456 pp with zero-mean errors. A deterministic scheme that made the anchor and node errors alternate in
sign across triples (a balanced assignment of GL v2 nodes and v4 nodes to triples) would remove their systematic part
at zero cost, but it is a rank-1 lattice over (triple x node) and so sampling-class under the DRI's ruling on T357; and
it leaves product + tmax (sd .286) untouched anyway. I mention it only so the checklist can list it as considered.

## 2. Three schemes (best available; none reaches .15 by the arithmetic above)

### S1. Wall tracking: 2-D continuity along the decision boundary, mass-ranked, funded by a shared anchor prefix tree
**What.** Both nb lines pass through the nb anchor x* = (v2* + g12*, v2*, v3*, v4*, mu(v4*)) (line 23 at t = g23*,
line 12 at t = g12*). (i) Build one prefix tree at x* with two tangent directions (g23 and g12) and walk each line from
the anchor outward (toward the tie and toward tmax) instead of from the tie: the cells crossed are the same, so the
estimates are preserved up to the g12 segment [.5, .53] (the anchor sits just outside the window; declare tmax12 = g12*).
Cost: one 2-tangent tree (2,815 nodes x ~1,780 FLOPs = 841 FPE) replaces two trees (1,152 FPE), saving **~310 FPE**
(~390 with the shared first-cell readout) [inferred from T326's node costs]. The C1 screen runs toward tmax from the
anchor cell as before. (ii) Spend the saving on tracking: at each wall crossing theta* > 0 found by a line, follow the
zero set of the binding margin m(g12, g23) = L_label - L_rival through the (g12, g23) plane toward the other axis. Inside a
cell the pre-activations are affine in (g12, g23), so the curve is a segment; its exit is the first root among the 160
gate functions and 8 rival switches along the segment (~700 FLOPs); at the exit toggle the gate and recompute the two
tangents for steps > s* (continuity, 2 x 560 per step) and the readout: **~1.0-1.1 FPE per boundary cell**. The tracked
rows replace the product by the exact set bounded by the tracked curves (fill from the far corner); islands that a line
finds as later wrong pieces are tracked as closed curves from their endpoints. Untracked rows keep the product plus the
T347 tilt.
**Arithmetic.** Wall rows: ~49% of 1,440 nb rows = ~706 (T347). Cells per wall curve: the curve spans g12 from .53 to 0
at g23 ~ theta* (.02-.1); from corner.json's densities (3.9 / 6.6 / 10.4 / 14.4 / 20.3 nb cells per triple inside
[0, d]^2 for d = .025 / .05 / .1 / .2 / .5) I estimate 7-10 cells per wall curve, i.e. **7-11 FPE per wall row, 5-7k FPE
for full coverage** (vs 1,145 FPE of later cells in all of T317f). With the ~350 FPE saving, **~35 of 706 walls** can be
tracked. [inferred]
**Why it would scale better than walks + product.** Tracking resolves every kink along the wall exactly, where the tilt
extrapolates from g12* = .53 to the corner across the kinks (T347's stated failure mode, corr .54 per model); and it
resolves only boundary cells, not the whole L-band (T333 B3 priced all corner cells at 4.0k FPE at d = .1).
**Relies on / contradicts.** Relies on the product's model-specific error being concentrated in few wall rows (ANAT-80:
~44 triples carry a weak model's error). T326's tree note says the *line deficit* is diffuse over most prefix nodes
(pruning rows below 1e-4 drops .7-.9 pp per line), which points the other way. Contradicts nothing measured.
**Kill test (10 DEV3 models, pre-registerable, ~15 CPU-min, no tracker built).** Per nb triple, evaluate the J1 joint
verdict (line-23 closure, as addendum A) on a 60 x 60 grid over [0, .6]^2 at the nb anchor (2.6M forwards per model).
From the grid, per row: the exact-law masses of the product set and of the joint set; the far-corner connected component
and its boundary's distinct gate patterns (cost proxy); island mass not touching either axis line; the rank statistic
m_r = wrong mass on the two axis slices (what the walks know). Kill if any of: (a) mean distinct patterns along the main
wall curve > 6 per wall row; (b) with the 35 highest-m_r rows replaced by their joint verdict and all others at the
product, the residual product error across the 10 models has sd > .10 pp; (c) islands touching neither line carry mass
with sd > .05 pp. Also report the DRI's overturning line for the whole scheme (it will fail: anchor + node terms
remain, sd .25, mean 0). **My prediction: (b) fails with residual sd .25-.30.**
**Most likely failure.** The product's model-specific error is diffuse over hundreds of wall rows, so 35 tracked walls
remove < 30% of its variance; and the boundary cells near the corner are finer than the census grid shows (the 200 x 200
check added +5%), so even 35 walls overrun the saving.

### S2. A homogeneous closure (the only free reduction of the integral's dimension; it changes E0)
**What.** The network has no biases, so verdict(c x) = verdict(x) for c > 0. Closure c breaks this (v4 = E[v4 | v3] and
mu(v4) are not degree-1 in the top values), and that non-homogeneity *is* the anchor error: the 2-D set at v2 differs
from the anchor's set by more than a rescaling. Define c_h: v4 = v3 - d v1, lows = v4 - c v1 (or the T327 spread
pattern (c -+ sigma/v1*) v1 by position), with d = (v3* - v4*)/v1* and c = (v4* - mu(v4*))/v1* fixed from the nb
anchor's exact-law values (model-free). The scale s = v1 is degree-1 and affine along both lines, so the walks stay
exactly piecewise linear. Then per triple the verdict depends only on (v2/v1, v3/v1): **the 3-D (v2, g12, g23) integral
becomes exactly 2-D** with a model-free density of the ratios (tabulated once), the GL-in-v2 integration becomes exact,
and the v4 node disappears (v4 is a function of the top-3). Cost: T317f's walks unchanged minus the Phi profiles (-.1k).
**Arithmetic.** Anchor (sd .177, mean -.31) and node (sd .173, mean +.31) terms vanish by construction. Remaining: product
+ tmax + blind, mean +.59, sd .31 [measured, V1 - path4]; the tilt takes the mean to ~+.15, leaving RMS ~.33-.35 vs
E0_h, at ~3.2k (or ~2.9k with S1's shared tree, allowance .23). **Fails .15; would give F ~1.3-1.5 only if floor_h <= .43.**
The real lever is the ceiling: a floor of .40 moves C_max to 4.3k and the allowance at 3.3k to .22; a floor of .30 gives
.345, which tilt-corrected walks are within 1.3x of. [inferred]
**Relies on / contradicts.** Relies on the lows' values mattering less than their spread and order (T327: spread lows
.343 vs .48; T338/T341 floors .31-.60). Against it: the true lows are O(1) whatever the top-3 scale (mu(v4) ~ -1.2 +- .2),
while c_h scales them with v1, so the ~9% of inputs with v1 < .8 get lows off by ~.8; closure floors in this project have
never gone below .31, and homogeneous forms are more constrained. Note T304/T306's lesson: a lower floor only pays if the
walks' cost does not rise (T339's spread raised C 12-23% and its variance cut did not replicate, T343).
**Kill test (10 DEV3 models, 1-3 CPU-min).** Paired MC as floors_dev3.py (2e5 inputs, same seeds): closure c (check:
reproduce E0 within .03), c_h mean-lows, c_h spread-lows; report RMS vs truth and bias. **Kill if min floor_h > .45**
(no gain on the ceiling). Go (stage 2) if <= .40: compute the walks' pointwise form under c_h (budget.py's product at the
anchor, no cuts, blind walked) vs the paired E0_h MC; kill if its sd > .30. **My prediction: floor_h .6-1.0, kill.**
**Most likely failure.** The floor: mis-scaled lows create spurious near-ties with v3 exactly on the inputs where the
network already errs.

### S3. Clipped first-order v4 and v2 corrections at the wall (the T347 construction in the transverse directions)
**What.** The K = 1 node integrates the verdict as if it were linear in v4; it is a step function, and the wall position
theta(v4) is clipped at 0 (51% of triple-lines have theta* = 0). Jensen on the clipped wall makes the node overstate
accuracy: the right sign and, plausibly, the right size (+.31). Correction: at each wall crossing (or at the tie for
theta* = 0 rows, where the walk's first cell is available), one v4-tangent gives s4 = d theta / d v4 (gates frozen);
D4 = E_{v4 ~ exact law given v3*}[Psi(clip(theta* + s4 (v4 - v4*)))] - Psi(theta*), with the law 7 phi Phi^6 / Phi(v3*)^7
tabulated (model-free) and a 40-node rule. The same with a v2-tangent gives the first-order anchor correction. Deductive
given the walks; no free parameter.
**Arithmetic.** ~1 FPE per wall row with T353's state reuse: ~.7k for both directions on ~700 rows, or ~.35k for v4 only.
It removes means only: T347's analogue removed the product's panel mean within .06-.12 but left per-model MAE .23 (corr .54).
Expected residual sds: node .17 -> ~.14, anchor .18 -> ~.15 [inferred by the T347 ratio]. **Fails .15 on its own and in
combination; it is the necessary mean fix for any K = 1 scheme to satisfy "mean within +-.1", not a route to the sd.**
**Relies on / contradicts.** Relies on the Jensen/clipping mechanism being the bulk of +.31 (unmeasured). The first-order
extrapolation across gate kinks in v4 (sd of v4 | v3 ~ .4) is the same failure mode as ADD, T292 lin, T319 and T347 itself.
**Kill test (100 DEV3 models, ~2 CPU-min; the measured side already exists).** D4_meas = E1 - E0 per model is in
floors_dev3.json. Compute D4_pred from T351's cont_h crossing state plus one v4-tangent. Kill if |mean D4_pred - mean
D4_meas| > .1 or corr(D4_pred, D4_meas) < .5 (over 100; also report the first 10). **Prediction: mean within .1, corr .4-.6.**
**Most likely failure.** The +.31 comes from v4-driven gate flips deep in the sequence (v4 sits at the latest free
position), not from the clipped wall's curvature, so the tangent at the node misses it: mean error > .1.

### Considered and killed by existing data (for the checklist)
- **Adaptive exact 2-D cells where the lines disagree.** The disagreement region is the L-band near the walls (91% of the
  net product error at d = .1, corner.json). Its cells cost >= 4.0k FPE at d = .1 for the corner alone (T333 B3), and
  **the part of the product error outside the band at d = .05 has sd .150 across models** [measured, corner.json], so a
  fix restricted to min(g) < .05 fails by itself; one that reaches d = .1 (outside residual .05 +- .06) costs >= 2x the
  whole remaining budget.
- **Boundary tracking in (v2, g12, g23) proper** (a 2-D surface): cells intersected ~ the 2-D census (~25 per triple-
  anchor) times the v2 extent, ~100+ cells per triple at ~1.5 FPE = ~100k FPE. Only 1-D curves at one anchor (S1) are
  affordable, and they leave the anchor term.
- **Exact 2-D polygon masses are cheap; enumeration is not.** Given v2, g12 and g23 are conditionally independent (g12 has
  density phi(v2 + g) / (1 - Phi(v2)), g23 the max-of-8 law), so a polygon's mass is a sum of 1-D Phi-type integrals, a
  few dozen FLOPs per edge. The honest 2-D cell price (two tangents, 160 half-planes, clipping) stays >= 2.5-4.4 FPE
  (T333 B2'), and ~25 cells per triple-anchor x 720 x 2 anchors is >= 90k FPE. The integrand's structure helps the mass,
  not the count.
- **Rigorous error control** (interval, norm, zonotope, CROWN) is dead at 1e-4 box widths because |Wh| compounds over 10
  steps (T330, T330b, T344); declared first-order control is the tilt, with residual sd .26.

## 3. Is the target plausible, and what is the cheapest experiment that tells?
**Not plausible [inferred from measured components].** The four systematic terms have sds .225 / .371 / .177 / .173; the
allowance is .147 at 3.3k and 0 at 3.7k; the known fixes cost +1.3k (v4), +1.4-2.9k (tmax), x2 (anchor), >= 21k
(product's sd), against a free budget of ~310-390 FPE (the shared anchor tree). Two independent single-term kills (node,
tmax) hold before any 2-D consideration. T332's information ceiling (.577 held-out for any fitted function of the walks'
own outputs, i.e. ~.40 integration-equivalent) says the same from the inductive side: the information is not in what
the walks compute, and new evaluations have no budget.

**Odds (subjective):** any deductive integration of closure c reaching RMS(est - E0) <= .15 at <= 3.3k: **< 2%**. Strict for
the T317 paradigm by a lower-floor closure plus tilt-corrected walks (the only route whose arithmetic is not closed above):
~5%. S1 passing its component kill test (b): 20%; S2 passing its floor test: 10%; S3 passing: 40% (and insufficient).

**Cheapest experiment that would tell.** For closure c, none is needed: the component table in 1b is the test, computed
from paired outputs already on the branches (T324's results.json + results_e0.json, T356's floors_dev3.json); I ask the
DRI to add it to T356's checklist and to T357's reproduction (T357 can recompute the DEV2 rows from its own e0e1 code in
minutes). If one new run is made, it should be **S2's floor test (1-3 CPU-min)**, because a closure with floor <= .40 is
the only thing that moves the cost ceiling; everything on the integration side is bounded by 1b. S1's grid census
(~15 CPU-min) is the second, to close the wall-tracking variant formally rather than by my cell-count estimate.

## 4. Suggestions for the ledger
- H-S1: the "untried: integration schemes exploiting structure the walks do not (T346)" item can be scoped by arithmetic:
  any one-anchor, K = 1 scheme fails the overturning test by RMS(E1 - E0) = .35 (both panels) and by sd(tmax) = .225; the
  remaining untried items are a homogeneous or otherwise lower-floor closure (lane B, a closure question) and the
  sampling-class balanced designs (ruled out by the rules, and insufficient).
- H-L1 dossier: the integration error grows with the model's error (DEV3 terciles +.39 / +.55 / +.75), consistent with the
  red-team's reading that the bias is a property of weak retrains' soft tails.
- T356 section 3: the quadrature statement sqrt(.413^2 + .414^2) = .585 matches T317f's sd .586, but the component sds
  above show the .413 is itself four nearly independent terms; the checklist should name them, since each has its own
  cost-blocked fix. That is the exhaustion argument in its strongest form.

## Claims
1. [measured; my arithmetic on T324/T356/T357 outputs] Component sds of the walk-level integration error on 80 DEV2:
   tmax .225, blind .048, mu-lin .072, product .371, anchor .177, K = 1 node .173 (DEV3, 100: .178); total .446
   (DEV3 .414). E1 - E0 RMS .354 (DEV2) / .351 (DEV3). V1 - path4 (anchor and node removed) sd .310, mean +.593.
2. [measured; corner.json] The product error outside the L-band at d = .05 has sd .150 across 10 DEV2 models; inside
   [0, .1]^2 lie 10.4 of 20.3 nb cells per triple.
3. [inferred] The cost ceiling for closure c is 3,692 FPE; the allowance at 3,307 is .147 pp; the shared anchor tree saves
   ~310-390 FPE exactly; wall tracking costs ~7-11 FPE per wall row, 5-7k at full coverage.
4. [speculative] S2's homogeneous closure has floor .6-1.0; S3's v4 correction gets the mean within .1 but corr ~.5.

## Alternatives not ruled out
- The component sds are DEV2 (80) except E1 - E0; the DEV2 -> DEV3 shift was large for T339's variance cut. The DEV3
  totals (T317f - E0 sd .414, E1 - E0 sd .178) match DEV2 closely, so I expect the components to transfer, but T357's
  code could confirm the DEV2 rows on DEV3 cheaply (budget.py's path on 100 DEV3 is ~30 CPU-min).
- A closure with a floor well below .4 would reopen the strict question for the walks; none has been found (best .31,
  unaffordable position, T338/T341).
- The per-triple view (section 1c) says a deductive scheme with sign-alternating per-triple errors would be far more
  forgiving; I found no deductive construction with that property that is not a lattice over triples.

## Failure modes of this review
- Cell counts along the wall curve are my estimate from the corner census, not a measurement; S1's kill test (a) measures it.
- The homogeneity argument assumes no biases anywhere in the network (FINDINGS: none) and a closure that is affine along
  the lines; a wrong constant in c_h changes its floor but not the argument.
