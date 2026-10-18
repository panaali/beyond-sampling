# Fable review, T346 step 2 (2026-10-18): cleverer exact methods, and exact computation as a component

Read (nothing run; small python arithmetic only): the seed ideas A-J (exact_brainstorm/README, ccr-6a61f25c-wn80ii);
OWNER_QA 6.1-6.5 and 4.6; INDEX rows H-E13, H-E15, H-S1, T241, T270, T290, T316, T324, T326, T330, T333, T340, T344,
T348, T354, T356, T357, T358, T361; TASKS rows T345/T346/T348/T352; the boundary/ and cone_enum/ READMEs; the T344
section of r1_cont/README (sub-cone counts, 3-D cell counts); my rounds 5 and 6. Child results are unverified unless
INDEX says otherwise. I did not read Astra's review.

Conventions. 1 FPE = one forward pass (5,760 FLOPs). Sampling to 0.1 pp on the accuracy: N = p(1-p)/1e-6 = 4.5e4 FPE
(the project's figure, with its conventions, is ~7.4e4; I use 7.4e4 below so ratios are conservative). Sampling on a
rare event of probability p to relative error eps: N = (1-p)/(p eps^2). Per-region exact costs: LP + cone split
30-40 FPE (cone_enum: LP 10-30, build 9); a 10-D cone mass 1.1e3 FPE (estT at 6.5 ms, k = 6, buggy geometry) to
3.5e4 FPE (T316 robust geometry, 0.2 s), taking 1 GFLOP/s per core.

## 0. Verdict in one paragraph

No exact method computes M_16,10's accuracy within 10x of sampling at 0.1 pp; the shortfall is >= 4e4x for the
mildest pseudo-exact variant (delta = 1e-3 coverage) and ~1e9x for the full enumeration, and the seed ideas A-D do not
change the exponent because the object they would enumerate (the decision boundary) is where the explosion lives
(H-E15: error-mass coverage grows 25-44x per step, faster than regions overall). B becomes real only under quantised
activations (T352). What survives is exact computation as a **component**, and the three components with the best
arithmetic are not on the seed list in this form: **L**, conditional Monte Carlo along exact gap lines for rare events
(predicted 10-50x over plain sampling, unbiased, sampling-class); **J/U**, exact weight-sensitivities of an event's
probability along the same lines (the only finite-variance unbiased estimator of d P/dW that exists for an indicator;
~1 FPE per wall crossing for all 432 weights at once); and **E re-scoped**: exact enumeration inside the event's own
cone, killed in T344 against a 1e6-region line that was the right line for a small-budget certificate but the wrong
yardstick for a rare event whose plain-sampling cost is itself 1e9-1e10 FPE; it is within 10x of sampling only for
p <~ 1e-8 and only if the sub-cone growth stays ~3x per step for every triple, which T344's own (9,5,1) triple (18.8x
at one step) says it may not. The one cheap experiment that every "dead" verdict leans on and that has not been run
is T348's late-step (k = 7..10) growth of **error-carrying** gate patterns.

## 1. The arithmetic that binds every exact idea

**1a. Count.** N6 ~ 2.0e7 (T270 estimate, biased low by the T316 defect; exact lower bound 1.55e7). N10 = 1.9e13
(x31 constant), 2.4e13 (x33), 3.3e13 (rising 33/35/37/39) [inferred from measured N5, N6]. Full enumeration at >= 30
FPE per region (no masses) is >= 6e14 FPE = 8e9 x sampling; with masses for the boundary regions, more.

**1b. The boundary is the explosion [inferred from H-E15, confirmed scoped k <= 4].** Idea A assumes the decision
boundary touches a small, slowly growing subset of regions. The measured facts point the other way:
- 99% error-mass coverage grows x24.9 / x44.3 per step vs x8.3 / x18.4 for correct mass (T 3 -> 4, two configs), i.e.
  error-carrying cells multiply 2.4-3x faster than correct-carrying cells, and faster than the all-region growth
  (x23-30 at those steps).
- Cells per unit mass near ties are 358k vs 2.9k far from ties at k = 4 (trained target), a 123x density ratio. With
  ~10% of the mass within the near-tie band, near-tie regions are 0.1 x 358k / (0.1 x 358k + 0.9 x 2.9k) = 93% of
  all regions. So "boundary-only" saves at most ~1.1x once, not a factor per step.
- Error-touching regions at T = 10: a few thousand at T = 4 (the 99%-coverage count) x (25-44)^6 = 7e11 to 7e13:
  the same order as all regions.
Any exact method has to produce the Gaussian measure of a set whose boundary has ~1e12 distinct affine pieces. No
symmetry makes those pieces cancel: the only exact invariance is scale (no biases), already used in the cone masses;
the network is not permutation-equivariant across positions. So the count is intrinsic to exactness, whatever the
enumeration order (forward, backward, boundary-first). [inferred]

**1c. Why it explodes: orderings x within-ordering fragmentation [inferred, partly speculative].** T344(b) counted
regions inside cones with the top-3 order fixed: 630 / 13,067 at k = 5 for two triples, with growth falling from x23
at step 2->3 to x2-3 per step once the top values have been read (the comparator/record gates are then determined
and only the magnitude-dependent neurons fragment). In the full space every step can bring a new record, and the
rank pattern of 10 inputs has 10! = 3.6e6 cells; 3.6e6 x (1e5-1e6 regions per ordering cone at k = 10, from 630-13k
at k = 5 growing 2-3x) = 4e11-4e12, consistent with 1a. Neither factor is attackable exactly: the lows' 7! orderings
are irrelevant to the truth but not to the network, and the only way to drop them is a closure of the lows, whose
floor is RMS .43 pp (E0; T356/T357). That is the T317 paradigm, and H-S1 is where it ends.

**1d. Pseudo-exact (cover 1 - delta of the mass, bound the rest).** T270: g_1e-3 >= 7.58 per step at k = 6. With
N_1e-3(6) between 3e4 and 1.6e5 (N_1e-5(5) is 153k), N_1e-3(10) >= 1e8-5e8 regions, >= 3e9-1.6e10 FPE at 30 FPE
each: **4e4-2e5 x sampling**, for a rigorous +-0.05 pp. The probability-weighted effective count (x5/step) is the
only slowly growing quantity, and it is not an enumeration count.

**1e. Exact integral of a closure.** The closure's 3-D (v2, g12, g23) has ~200-500 cells per triple, ~4e5 per model
(T344(a) grid census); at the honest 2-D/3-D cell price 2.5-4.4 FPE (T333 B2') that is 1.0-1.8e6 FPE = 14-24x
sampling, and the result inherits the closure floor RMS .43: F = 270-470. The walks (T317f/T353, 3-5k FPE, F 2.6-3.1)
are the right way to integrate a closure; integrating it exactly buys nothing.

**1f. Rare-event yardsticks (plain sampling, FPE).**

| p | eps = .1 | eps = .5 (~"within 2x") |
|---|---|---|
| 5.4e-6 (orig E1(1.0)) | 1.9e7 | 7.4e5 |
| 2.2e-7 (orig E1(1.5)) | 4.5e8 | 1.8e7 |
| 1e-8 | 1.0e10 | 4.0e8 |

The exact walks' K3/K5 (T340c) reach within 2x at 1.8-2.7M FPE, so against the eps = .5 column they win only for
p <~ 1e-7; against eps = .1 they win from p ~ 1e-6. This is the regime where the components below matter.

## 2. Ranking

| rank | idea | buys | arithmetic in one line | status |
|---|---|---|---|---|
| 1 | **L (new)**: conditional MC along exact gap lines, event-restricted | rare events (10-50x), truths | gain = 1/(c x mbar), c ~3 FPE/line, mbar = conditional mass per hitting line (.005-.02 for far-gap islands) | sampling-class; test in CPU-min |
| 2 | **J/U**: exact d P(event)/dW along the walks or L's lines | long-term goal, unique vs sampling | +1 FPE per wall crossing gives all 432 derivatives, unbiased; plain MC has none (indicator) | test in GPU-min |
| 3 | **E (re-scoped)**: exact enumeration in the event cone, p <~ 1e-8 | proof/certificate for rare events | 2e8-1e10 regions x 30 FPE = 6e9-3e11 FPE vs 1e10 sampling at p = 1e-8, eps = .1: 0.6-30x | marginal; CPU-h test |
| 4 | **I (T348)**: census, extended | kills or revives A/C/D in one run; calibrates every extrapolation | 15-30 GPU-min; same pass, four extra hashes | diagnostic only |
| 5 | **H**: shared affine products | cost per region, already 31.5% of T317's C as the B2 tree; r5: ~7-15% more available | cannot touch a 1e13 count | component only |
| 6 | **F**: exact 2-D/3-D slices | strict: dead by T333 (J2 >= 13-21k vs 1.2k allowance) and r6's four-term table | lenient: already in T353 | exhausted |
| 7 | **A**: boundary-only enumeration | none | boundary regions are ~93% of all and grow faster (1b) | dead on existing data |
| 8 | **D**: exact late, declared early | none | T241 recurrent variant: a formula state at t0 = 5/6/7 costs .34/.61/1.0 pp agreement; exact suffix from t0 <= 4 is the whole problem | dead on existing data |
| 9 | **C**: backward decomposition | none alone | backward leaves = forward boundary regions (1b); mass pruning needs the forward law | reorders, does not reduce |
| 10 | **B**: future-equivalence merging | none for the float model | exact classes ~ regions (T290: small weights matter); tolerance eps at step k is amplified 50^(10-k): 1e-6 at k = 5 -> .3 | alive only as T352 |
| 11 | **G**: truncated models as anchors | none | inductive by construction; the ladder E43-48 already shows the cost blow-up | dead by rule |

## 3. The top four in detail

### 3.1 L: conditional Monte Carlo along exact gap lines, restricted to the event (Rao-Blackwellised boundary/)
**What.** boundary/ already did this for the accuracy: sample a surface point y, walk the normal line (e_a - e_b)/sqrt2
exactly, integrate the exact 1-D Gaussian mass of the wrong set; unbiased; measured 2.5-3.2x MSE gain over sampling at
equal compute at ~3 FPE per line [measured, boundary/ README]. T340's variants A/B/K3 are the same walks with the
sampling replaced by GL nodes and a closure, and their failures are the quadrature's (four systematic terms, r6 1b),
not the walks'. For a rare event E = {pick = p3, g23 > t0} (or E2), sample y from the exact law of the 9-dim
complement, restricted by a **model-free** rejection to lines that can reach the event's input constraints (ranks,
g23 > t0), walk the (e_p2 - e_p3)/sqrt2 line exactly over the admissible t-interval, and accumulate the exact mass.
**Why it escapes the explosion.** It never enumerates; each line costs ~3 FPE and resolves its own cells exactly.
**Arithmetic [inferred].** If the hitting lines carry conditional mass ~mbar, Var(Z) = p(mbar - p) ~ p mbar against
p(1 - p) for the indicator, so the compute gain is G = 1/(c mbar) with c ~3 FPE per line. boundary/'s 2.5-3x implies
mbar ~ .11-.13 for the near-tie band (consistent: the adjacent wrong band carries a sizeable share of a hitting
line's mass). For far-gap islands the island width in g23 (.05-.2) times the g23 | v2 density near 1-1.5 (~.1 per unit)
gives mbar ~ .005-.02, G ~ 17-70x; with the model-free rejection removing lines that cannot reach g23 > t0, more. I
predict **10-50x** on E1/E2.
**Kill test (pre-registerable; orig; ~1e4 lines = ~3e4 FPE, 2-5 CPU-min with the fixed walker).** Events E1(1.0)
(truth 5.4e-6) and E2(.5) (9.4e-5), T192 truths. Report the mean, the empirical SE, C as executed (walk cells, rejection
not charged by convention but reported). **Go:** gain = Var_MC-equivalent / (empirical Var x C per line) >= 10 on both
events and the mean within 2 SE of the truth. **Stop:** gain < 3 (per-line cost or mbar kills it) or |bias| > 2 SE
(a walk defect of the T350 class: simultaneous gate crossings when an input crosses zero; L must use the fixed walker).
**Most likely failure.** Per-line cost in the far-gap region is 6-10 FPE rather than 3 (T354: most cells lie beyond
t0 up to t ~6), halving the gain; and E1's islands are thick in g23 (mbar ~ .05-.1), leaving G ~3-7x.
**What it buys.** Rare events at equal compute (10-50x), with no closure and no declared term, so "within 2x" is a
matter of SE; cheaper truths than 1e10 GPU samples (a 30x gain gives T340's truth precision at ~3e8 FPE). It does
**not** buy the strict or lenient bar: it is sampling-class under the project's rules. The DRI should rule whether
sampling-class estimators are in scope for the owner's rare-event lane (ARC's letter compares error at equal compute;
the lane's primary criterion is within 2x).

### 3.2 J/U: exact sensitivities d P(E)/dW along the lines
**What.** On a line through y, the event set is a union of intervals [a_i, b_i] whose endpoints are roots of affine
margins m(t) = L_p3(t) - L_j(t) inside a cell. The line mass is sum Phi(b_i) - Phi(a_i), and d a_i/dW =
-(dm/dW)/(dm/dt) at the root; dm/dW for all 432 weights is one backward pass through the frozen-gate linear network
(~1 FPE). So d m(y)/dW = sum phi(b_i) db_i/dW - phi(a_i) da_i/dW costs ~1 FPE per crossing, and E_y of it is an
unbiased estimate of the gradient of the event probability. [inferred]
**Why it is special.** Plain sampling has no gradient of an indicator (zero a.e.); finite differences with common
random numbers need the flipping samples (variance ~ 1/(delta N)), and a smoothed surrogate has a temperature bias.
The exact-line gradient is unbiased with finite variance and gets all weights at once. This is the one place where
exact computation strictly dominates sampling rather than trading with it.
**Arithmetic.** With L's lines: ~4 FPE per line for mass + gradient. For the accuracy's own gradient (owner Q&A 4.4
"which weights make a failure more likely"), the same holds along T317f's deterministic walks at ~+30% C, but then it
inherits the walks' systematic terms; the unbiased version is L's.
**Kill test (orig, E2(.5), p ~1e-4 so FD truths are cheap; CPU-min + a few GPU-min).** Pick 5 weights: the 3 largest
|W_hh| entries and 2 small ones (T290's point that small weights matter). (i) Line gradient from 2e4 lines (~8e4 FPE).
(ii) GPU truth: central differences with common random numbers, delta = 1e-3, 2e8 samples per side per weight
(2e9 passes, the T192 kernels). **Go:** all 5 within 2 combined SE, and (i)'s SE <= 1/3 of (ii)'s at <= 1/100 of its
cost. **Stop:** a sign disagreement outside 3 SE (a missed-crossing bias) or (i)'s variance within 3x of (ii)'s.
**Most likely failure.** Crossings where the margin is nearly tangent (dm/dt ~ 0) give heavy-tailed da/dW; the
estimator needs a tangent cut (declared) or a median-of-means, and the tangent cases may carry much of the gradient.
**What it buys.** The long-term goal only: a mechanistic, weight-level account of a rare failure. Not the accuracy bar.

### 3.3 E re-scoped: exact enumeration inside the event cone, for p <~ 1e-8
**What.** T344(b) counted exact regions inside a conservative far-gap proxy cone (g23 >= 2 g12, g23 >= 2 x every low
gap) for three triples: 630 / 13,067 / > 3,994 regions at k = 5 (the third stopped at k = 4), growth falling to
2.2-3.2x per step after step 3 but 18.8x at step 3->4 on (9,5,1). It was killed against a 1e6-region line, which was
right for a certificate meant to cost <~ 1e5 FPE. Against plain sampling on the event itself the yardstick is 1f.
**Arithmetic [inferred from T344(b)].** Regions needed: event-infeasible regions need only an LP (~30 FPE; the event
"pick = p3" adds 9 rows); event-feasible ones need the sub-polytope's 10-D mass (1.1e3-3.5e4 FPE) but are a small
fraction. Projected N10 over 720 triples: 2.1e8 (best triple's rate) to 9.6e9 ((3,6,9)'s rate), i.e. 6e9-3e11 FPE.
Against sampling at eps = .1: p = 2.2e-7 -> 14-640x; p = 1e-8 -> 0.6-29x. The cone shrinks with t0 (P(g23 > t0)
= .044 / .017 / .006 at 1 / 1.25 / 1.5) while p falls faster (5.4e-6 -> 2.2e-7), so each step in t0 improves the
ratio ~10x: E is within 10x of sampling only at p <~ 1e-8 and only at the favourable growth.
**Kill test (orig, t0 = 1.5; T344's cone_count.py on the TRUE event cone K = {x_p1 >= x_p2 >= x_p3 + t0 >= lows};
10 random triples, k = 2..6, cap 2e5 regions per triple; ~1-2 CPU-h).** Project N10 = sum over the 10 triples of
count_6 x (growth 5->6)^4, scaled by 72. **Go:** projected N10 <= 2e8 (cost <= 6e9 FPE, <= 13x sampling at eps = .1
for p = 2.2e-7, < 1x for p <= 1e-8) AND max-triple / median <= 10 (otherwise the worst triples dominate and the
projection is meaningless). Also report the fraction of event-feasible regions at k = 6 (if > 1e-2, the mass work
alone exceeds the LP work by 3x). **Stop** otherwise.
**Most likely failure.** Heterogeneity across triples (one triple at 18.8x in a single step; late positions p3 = 9
leave no time for the comparator gates to settle) and the true cone being larger than the proxy (it contains all
g12 and all low gaps). My prediction: projected N10 ~ 1e9-1e10, kill.
**What it buys.** The only idea on the list that yields a **rigorous** number (an exact probability or a certified
upper bound for the far-gap events), which is what the owner's certificate question (4.6, T344) wanted. Not the
accuracy bar, and not p >= 1e-7.

### 3.4 I: the T348 census, extended so one GPU pass decides A, C and D
**What.** T348 is reserved (>= 1e9 inputs, hashed gate words for k = 4..10, all / near-tie / error inputs, validated
against exact k = 4-6). Add four hashes to the same pass: (a) distinct prefixes among **error** inputs and among inputs
within .05 of a tie, per k, with Good-Turing missing mass (A); (b) distinct (gates of steps 10-j+1..10, decision) classes
among error inputs for j = 1..4 (C's backward branching; cone_enum's 1M-sample value at j = 1 is ~4.5k with 3.6%
singleton mass, nearly saturated); (c) distinct suffix words (steps k+1..10) over all inputs (H's sharing, and D); (d)
distinct patterns per unit error mass vs per unit correct mass by k (the H-E15 ratio carried to k = 10).
**Why it matters.** Every "dead" verdict (H-E13, OWNER_QA 6.2-6.3, this review's 1a-1b) extrapolates from k <= 6.
T344(b) shows growth can fall to 2-3x per step inside a fixed-order cone; if error-pattern growth in the full space
fell below ~3x by k = 8 (the late steps re-reading a settled order), N10 would be ~1e10-1e11, still dead for the
accuracy (3e11 FPE vs 7.4e4) but reviving E for all rare events and halving the exponent on A/C. H-E15 at k <= 4
points the other way (error growth above all-region growth). It is 15-30 GPU-min to know.
**Go lines.** A revives only if error-pattern growth <= 6x per step for k >= 7 (prediction: 15-30x, lower-bounded by
singletons); C only if the backward branching (b) is <= 6x per step for j = 1..4 (prediction: 10-20x at j = 2); D is
closed if (c) at k = 7 is >= 1e6 (prediction: it saturates at the sample size).
**Failure.** At k >= 8 most patterns are singletons, so the counts are lower bounds and growth is under-estimated;
a Good-Turing missing-mass estimate is needed, and it is itself a lower-bound-ish estimate. This cannot make the
verdict more positive than the raw counts, so a "revive" reading would be robust while a "kill" reading is not.
**What it buys.** Nothing by itself; it is the cheapest calibration of the extrapolation everything else rests on.

## 4. The seed ideas that existing data already answer (brief, with the kill test they would have had)

**A. Boundary-only.** See 1b. Kill test from T270's k = 4-6 data: count regions whose cone meets a decision hyperplane
L_i = L_j within its own cell (an LP per region at k = 4: 21k LPs, CPU-min) vs all regions. Predicted: 80-95% of all
regions at every k, growth equal to the total. Fails the <= 6x line by a factor ~4-5.

**B. Future-equivalence merging.** Exact equality of two prefix regions' 16 x k affine maps needs the differing
neuron's row contribution to be annihilated downstream, which dense W_hh with load-bearing small weights (T290) never
does; so exact classes = regions. Merging at tolerance eps costs an error that compounds by the measured ~50x per step
(T330/T344): eps at step 5 -> eps x 3e2 at the output; for 1e-3 pp over 1e13 regions eps ~1e-27. Kill test (k = 4,
exact arithmetic, CPU-min): distinct affine maps (exact, and at 1e-9 / 1e-6 on the readout-projected rows) vs 21,032
regions; go if classes < regions / 10. Predicted: classes = regions at 1e-9, a few % fewer at 1e-6. The idea is real
under quantised activations (T352, finite states): there it is the method.

**C. Backward.** The backward tree's leaves are the error-touching regions (1b: ~1e12), and its depth-1 count is
already ~4.5k; the branching from 4.5e3 to 1e12 over 9 levels averages >= 8x per step, lower than the forward 23-33x
only because it starts from a large root. Pruning branches by Gaussian mass needs the law of h_t, which is forward
information: C becomes a hybrid with a sampled or declared state law, i.e. the T317 family's closure problem again.
Kill test: census item (b).

**D. Exact in some steps.** The cheap exact steps (k <= 4, 21k regions) are the ones that do not need it; the decision
is made in steps 8-10, where T241's recurrent variant measures the cost of a declared state: agreement .9995 / .9985 /
.9966 / .9939 / .9900 for t0 = 3..7, so a declared state at t0 >= 5 already spends >= .34 pp of agreement (3x the
bar), and the exact suffix from t0 <= 4 is N10 / N4 ~ 1e9 regions per prefix cell summed over 21k cells = the full
problem. Oracle split by step (the seed's test) is unnecessary: T241 is that split.

**F. Exact low-dimensional slices.** Closed by r6 1b for strict (four systematic terms with sd .17-.37 each against a
.147 allowance; J2 >= 13-21k FPE); for lenient the lineage is at F 2.6 (T353). Exact 2-D polygon masses are cheap given
conditional independence (r6), but the cell count (~25 per triple-anchor) is the cost, >= 90k FPE for two anchors.

**G.** Inductive unless the T -> T + 1 recurrence is derived from W; E43-E48 already show that a derived recipe's cost
explodes along T. **H.** Exact and worth ~7-15% of C in the walk family (swap-pair sharing, parent-matvec sharing, r5
(c)); irrelevant to a count of 1e13.

## 5. Exact computation as a component: what is reusable even when the standalone method is dead

1. **Exact 1-D walks** along gap directions: the core of T317f/T353 (lenient F 2.6-3.1), of boundary/ (sampling,
   2.5-3x), of L and of J. Their defect is the integration across lines, never the walk itself (r6 1b).
2. **Exact label combinatorics** (prefix trees over gate words, 2,815 nodes per line; B2) and shared affine products
   (H): 31.5% of T317's C today, exact, a further ~10% available.
3. **Exact sensitivities** (J/U): a derivative component no sampling scheme has.
4. **Exact sub-cone enumeration** (E) as a certificate generator for p <~ 1e-8, and as an exact truth where sampling
   returns zero.
5. **Exact small-k enumeration** (k <= 4, 21k regions, confirmed counts) as the test bed and calibration for every
   extrapolation (and the census, I).
6. **Exact cone masses in 2-D/3-D** when cells are given (r6: a few dozen FLOPs per edge): a component for any future
   scheme that enumerates few cells, not a reason to enumerate.
7. **B under quantisation** (T352): the merging is exact and the DP is the estimator.
Not reusable: A, C, D (they produce no piece an estimator can call), G (inductive).

## 6. Bottom line

**Exact or exact-component within 10x of sampling at 0.1 pp precision on the accuracy?**
- Exact (zero-error): **no**. Full enumeration ~1e9x, pseudo-exact at delta = 1e-3 >= 4e4x, exact integral of a
  closure 14-24x in cost but F 270-470 because of the closure floor. [inferred from measured counts]
- Deductive estimators built from exact components (T317f/T353): F 2.6-3.1, i.e. within 10x of sampling's MSE x C at
  their own cost, but their RMS saturates at ~.4-.6 pp whatever the compute (closure floor .43, walk terms .3-.4): they
  **never reach 0.1 pp**, so "within 10x at 0.1 pp" is not a frame they can meet. [measured floors; inferred]
- Sampling-class with exact components (boundary/): **yes, measured 2.5-3.2x better than sampling** at 0.1 pp, the
  only route that reaches 0.1 pp at all; excluded from the bars by the project's deductive rule.
- Rare events: L predicted 10-50x (unbiased); E within 10x only at p <~ 1e-8 and favourable growth; J unique.

**Odds (subjective).** L passes its 10x test on E1 and E2: 55%. J passes: 60%. E passes (N10 <= 2e8 with max/median
<= 10): 15%. T348 shows error-pattern growth <= 3x by k = 8: 10%. Any zero-error accuracy computation of M_16,10 within
10x of sampling: < 1%. A deductive exact-component estimator reaching RMS 0.1 pp at any compute: < 3%.

**The single experiment that would most change my mind:** T348's error-input growth for k = 7..10 (item (a) of 3.4),
15-30 GPU-min. Every verdict in this review and in H-E13 extrapolates from k <= 6 at the all-input rate; the one
measured sub-cone (T344(b)) shows growth can collapse from 23x to 2-3x once the order is settled. If the error-carrying
patterns do the same in the full space by k = 8, the count drops two to three orders of magnitude: still dead for the
accuracy, but E becomes the rare-event method and A/C's exponents halve. If they keep growing at 15-30x, this review's
"intrinsic boundary complexity" reading is confirmed at the one place it could have failed. For the rare-event lane
specifically, L's 1e4-line test is the experiment (CPU-min), and it is positive-expectation.

## Claims
1. [measured, from the ledger; unverified unless INDEX marks confirmed] N5 632,394; N6 >= 1.55e7 exact, ~2e7 estimated;
   growth x23/27/30/31-33; g_1e-3 >= 7.58 at k = 6; error-mass 99% coverage growth x24.9/44.3 vs correct x8.3/18.4
   (T 3->4); near/far cell density 358k vs 2.9k per unit mass at k = 4; T344(b) sub-cone counts 630 / 13,067 / >3,994
   at k = 5/5/4 with growth 2.2-23x; 3-D closure cells ~4e5 per model; boundary/ gain 2.5-3.2x at ~3 FPE per line;
   T241 recurrent-variant agreement .9995-.9900 for t0 = 3-7; E0 floor RMS .432/.435; T340c within 2x at 1.8-2.7M FPE.
2. [inferred, my arithmetic on 1] N10 1.9-3.3e13; error-touching regions ~7e11-7e13 (same order as all); near-tie
   regions ~93% of all; pseudo-exact at delta 1e-3 >= 4e4x sampling; exact closure integral F 270-470; E's cost
   6e9-3e11 FPE vs sampling 4.5e8 (p 2.2e-7) / 1e10 (p 1e-8); L's gain G = 1/(c mbar) with boundary/'s 2.5-3x implying
   mbar ~.11-.13; J's gradient costs ~1 FPE per crossing for all 432 weights; B's tolerance must be ~1e-27.
3. [speculative] L reaches 10-50x on far-gap events (mbar .005-.02); E's true-cone projection is 1e9-1e10 (kill);
   the census shows error-pattern growth 15-30x at k = 7-8; J's tangent crossings are heavy-tailed.

## Alternatives not ruled out
- The N10 extrapolation (1a) and the error-region estimate (1b) assume the k <= 6 growth persists; T348 tests this.
- 1c's orderings decomposition is a reading of two sub-cone counts and is not measured in the full space.
- L's mbar for the real events is a guess from island widths; the kill test measures it directly.
- E's per-region costs are cone_enum's and estT's on CPU; a GPU LP batch could cut the 30 FPE by 10x, which would
  move E's "within 10x" threshold from p ~1e-8 to ~1e-7. I did not price that.
- An exact method aggregating across regions by an algebraic identity (a generating function over gate words) is not
  excluded by 1b in principle; I know of none for ReLU networks under Gaussian inputs, and the no-symmetry argument is
  a reason not to expect one, not a proof.

## Failure modes of this review
- I rank from arithmetic on others' counts; two of the four top ideas (L, J) have never been run on this model, and
  boundary/'s 2.5-3x is the only measured anchor for L.
- I treat "sampling-class" as excluded from the bars per the project's rules; if the DRI reads ARC's equal-compute
  letter differently for rare events, L moves from "component" to "candidate" and the ranking's top is unchanged.
- The T350 walk bug (simultaneous gate crossings at input zero-crossings) affects any new use of the long-line walker;
  L and J must be built on the fixed walker and verified against the T192 truths before their kill tests count.
