# How the 432-parameter AlgZoo model M_16,10 computes the 2nd argmax

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md) · [m16 index](README.md) · [Approaches](approaches/README.md)

Model: `alg_zoo.example_2nd_argmax()`. A 1-layer ReLU RNN with 16 hidden neurons and no biases. It reads
10 i.i.d. N(0,1) numbers and outputs the position of the 2nd largest. Accuracy is 95.3%; chance is 10%.
ARC's published analysis (Hilton, Jan 2026) explains neurons 1, 2, 4, 6, 7 as running-max and
leave-one-out-max features, and leaves the other 11 neurons, the mapping to an algorithm, and an
accuracy estimate open.

Every number below is from code in this directory, on fresh N(0,1) samples (seeds in the scripts).
Each claim says how it was verified and whether it **passed or failed**.
**[Corrected 2026-10-18 afternoon]** This holds for sections 1-7. Section 8 cites registry rows T317-T353 and H-L1 in
[`INDEX.md`](../INDEX.md); their code is on the agents' branches named there, and they are scored on panels of retrained
models, not on fresh samples of this one.

## 1. What the network computes (algorithm level)

The answer depends only on a small discrete state: p1 and p2, the positions of the largest and 2nd largest
of x0..x8, plus where x9 falls (above both, between, or below both).

- **The algorithm is right; the errors come from magnitudes.** Replace every input by the expected Gaussian
  order statistic for its rank (same order, typical spacing) and the model scores **99.96%**. All of the
  4.7% error comes from how the actual values are spaced. [`frozen.py`](frozen.py)
- **The state is sufficient.** A lookup over the 216 discrete states, each a linear map on (m1, m2, x9),
  reproduces the hidden state with median R^2 0.95 and scores **98.0%**, higher than the model.
  The model's errors are its departures from this clean structure.

## 2. How it computes it (mechanism, derived from the weights)

With the gates that stay on after a record (13 of 16 neurons), the recurrence's eigenvalues form
**an approximate harmonic series**: a damped discrete Fourier memory. [`modes.py`](modes.py)

> **Caveat (DRI null test, later):** this gate set is only the *most common* one, covering 16% of states at
> steps 6-10. Other common gate sets give shifted periods (7.7/4.7/3.1; 6.7/4.0/3.0). Random 13-neuron subsets
> of the same matrix show the 1 : 1/2 : 1/3 pattern only 3% of the time, so it is not a coincidence of the weights. But only 2 of 5
> independently trained seeds show a similar series. Treat "Fourier memory" as a picture of one common configuration.
> The phase-clock measurement below is direct evidence and stands on its own. See [`INDEX.md`](../INDEX.md), H-M2.

| mode | abs(eig) | period (steps) | neurons that carry it |
|---|---|---|---|
| integrator | 1.010 | - | n2 (running max), plus a share of every clock |
| slow memory | 0.973 | - | n0, n10-n12, n14 |
| **fundamental** | 0.951 | **8.42** | **n0, n10, n11, n12, n14, n15** |
| 2nd harmonic | 0.937 | 4.32 (ratio 1.95) | **n3, n13**, n15 |
| 3rd harmonic | 0.837 | 2.89 (ratio 2.91) | **n3** |
| Nyquist | 0.928 | 2.00 | n3, n13 (small) |
| current input only | 0.00 | - | **n1, n6, n7, n8** (x9 comparators) |

Neurons n4, n5, n9 sit outside this gate set: they switch on at records (n4: record increment, as ARC found)
or when x8 was a near-record (n5, n9).

- **The clock: PASSED.** The phase of the fundamental mode steps by **-42.7 degrees per position of the
  max**; the eigenvalue predicts 360/8.42 = 42.8. Concentration is 0.86-0.98. One complex coordinate of the
  hidden state is a clock that says how long ago the max arrived.
- **The "polysemantic" neurons: explained.** n3 and n13 were rated C (mixture) by the blinded investigator
  (R^2 0.56 and 0.69). [WITHDRAWN as stated, see INDEX H-M2/H-M3: attribution used non-orthogonal modes] They load on the **2nd and 3rd Fourier harmonics**: irregular in position space,
  coherent in frequency space. They are what give the position code enough resolution.
- **Position code: mostly PASSED.** The hidden state is close to a *sum of two position vectors*:
  h ~ m2 U[p2] + (m1 - m2) V[p1]. That is 18 parameters per neuron instead of 216 tables. Read through the
  real output weights, every U[p] row is a bump peaked at exactly position p, about +-2 positions wide.
  This separable code alone scores 90-91%, against the model's 95-96%, when x9 is below both. [`separable.py`](separable.py)
- **Causal roles: PASSED for 13/16 neurons (in groups; n2, n5, n9 not ablated; mean-ablation can push the net off-distribution, R1).** Knocking out each group (final-state mean ablation) breaks exactly the cases its
  role predicts:

| knocked out | x9 is max (answer p1) | x9 is 2nd (answer 9) | x9 below both (answer p2) |
|---|---|---|---|
| none | 0.935 | 0.953 | 0.955 |
| clocks n0,10,11,12,14,15 | **0.127** | 0.437 | **0.229** |
| x9 comparators n1,6,7,8 | 0.840 | **0.000** | **0.085** |
| record detector n4 | 0.875 | 0.823 | 0.961 |
| harmonics n3,n13 | 0.708 | 0.895 | 0.676 |

- **Gating is set by record history.** At the last step, each neuron's on/off state is 92-100%
  predictable from the sequence of record / new-2nd-place events alone. Applying the gates of the
  rank-canonical input to the real input (an exact linear map, nothing fitted) gives 93.3% accuracy and 92.6%
  agreement with the model.

## 3. Why it fails (exact error anatomy; total 4.69%)

| source | share of all inputs |
|---|---|
| global 2nd-largest <= 0: no biases + ReLU means the network is nearly blind (wrong 78% of the time there) | 0.83% |
| x9 below both, picks the 3rd largest | 2.17% |
| x9 below both, picks the max | 0.52% |
| x9 is the max or 2nd, mostly picks the 3rd | 0.96% |
| other | ~0.2% |

The 2nd-vs-3rd near-tie, the dominant error, decomposed **exactly**: each input's own gates make its
logits an exact linear map of x, so the margin splits into per-input contributions with no fitting.
[`exact.py`](exact.py)
- The m2 vs m3 comparator is almost balanced: gain ~78 on each. The gain ratio is 0.982 on correct cases
  and 0.999 on flips. 44% of flips are this gain mismatch beating a tiny value gap.
- The other 56%: normally the max's contribution *helps* the right answer (mean +3.5) and so does crosstalk
  from the other inputs (+0.5). On flips these collapse to +0.2 and -0.9. Flips have a smaller rest of the
  sequence relative to the max: m2/m1 is 0.63 vs 0.69, and m4, m5, m6 are lower too.

## 4. Hypotheses that FAILED (kept on purpose)

- *Clock neurons are linear filters of per-step features*: R^2 0.72-0.93, but the replacement scores only
  **32%**. The gating carries essential information.
- *p2 is a second clean phasor in the fundamental mode*: residual phases are irregular. p2 is spread
  across harmonics and gating.
- *Errors are positional crosstalk between neighbouring positions*: near-tie flip rates do not rise for
  adjacent positions (about 10% at distance 1-4; 18% at distance 8).
- *The model wrongly picks the max when m2/m1 is small, at a threshold computable from the separable code*:
  the real network almost never picks the max in that case (at most 1.3%). The separable code's bump at p1 is
  an artifact of fitting it linearly; the real network suppresses p1 through gating.
- *Errors are set by (m2-m3) relative to m1*: no better a predictor than the plain gap.
- *A linear margin in (m2-m3, m1, m2, x9) for each position triple predicts the flips*: R^2 0.95, but it
  catches only 16% of actual flips.
- *The max's "assist" toward p2 over p3 is a positional effect*: the (p1,p2,p3) triple explains R^2 0.05 of
  it, and it is positive in only 63% of cases (mean +8.3 per unit m1). It depends on values, through gating.
- *[Added 2026-10-18 afternoon]* *Late margins read the low inputs through their running max* (a reading offered
  after T338): refuted by T341. Putting the lows at their true running max, or at its expected profile, is worse than
  a fixed spread pattern (section 8.5).
- *[Added 2026-10-18 afternoon]* *Low positions chosen from the weights W beat plain recency*: no. T338's W-derived
  choice is indistinguishable from taking the latest low (section 8.5).

## 5. Attempt at ARC's challenge: a mechanistic accuracy estimate ([`estimate.py`](estimate.py))

The error model built from sections 1-3: accuracy = 1 - [blind region] - [2nd-vs-3rd near-tie flips]
- [1st-vs-2nd near-tie flips]. The probes are designed, not random:
- Blind region: P = 11/1024, exactly (measured 0.01071). Error there is argued to be 8/9 (the network
  cannot rank the rest). Measured: 0.889 with 0 positive inputs, 0.768 with 1.
- For each of the 720 (rank1, rank2, rank3) position triples, build a canonical input (expected order
  statistics, rest shuffled) and bisect the relative gap at which the network flips. The flip probability is
  then P(gap < threshold), using the *model-free* distribution of that gap.

| estimate | accuracy | error vs truth (0.9531) | model evals | sampling s.d. at same evals |
|---|---|---|---|---|
| blind + 2nd-vs-3rd ties | 0.9676 | +1.45 pp | 5,400 | 0.29 pp |
| same, 8 arrangements | 0.9680 | +1.49 pp | 43,174 | 0.10 pp |
| **+ 1st-vs-2nd ties** | **0.9605** | **+0.74 pp** | 32,141 | **0.12 pp** |
| + threshold spread from sensitivity probes | 0.9590 | +0.59 pp | 80,629 | 0.07 pp |

The first version's bias didn't shrink with more probes, so it was systematic. The missing errors
turned out to be 1st-vs-2nd near-ties (gap median 0.019, the max picked 60% of the time). Adding them halved
the bias. **The result: understanding the mechanism gets the accuracy to within 0.74 points, but at equal
compute this is 4-6x worse than plain random sampling. ARC's bar is not met.** The remaining bias is the soft
tail: a flip threshold depends on all the other values, and a canonical probe fixes them at their means.
Modelling that spread from first-order sensitivity probes trims the bias to +0.59 pp, but costs 2.5x the
probes, so the gap to sampling *widens* (about 8x). Capturing that dependence without in effect sampling inputs is the open problem.
**[Updated 2026-10-18 afternoon]** Partly answered for the cheap sampling-free estimator of section 8: fixing the low
inputs at an alternating spread pattern (mean plus or minus one s.d.) instead of at their means cuts the model-to-model
error spread by about a third, but it exposes a positive bias from the other approximations (T327, T339; section 8.5).

### Under ARC's actual protocol ([`arc_eval.py`](arc_eval.py))

ARC scores estimators on noisy copies of the model (noise on all weights), by RMSE against ground truth
(2M samples per copy) as a function of compute. Noise = Gaussian with s.d. sigma times each matrix's RMS;
12 copies, 4 repeats for sampling.

| estimator | sigma | copies' accuracy | model evals | RMSE |
|---|---|---|---|---|
| sampling | 0.005 | 0.818-0.931 | 1,000 / 4,000 / 16,000 / 64,000 | 0.95 / 0.49 / 0.25 / 0.12 pp |
| probe estimator | 0.005 | same | 10,455 | **2.79 pp** |
| sampling | 0.0025 | 0.915-0.946 | 1,000 / 4,000 / 16,000 / 64,000 | 0.75 / 0.38 / 0.19 / 0.10 pp |
| probe estimator | 0.0025 | same | 10,607 | **1.97 pp** |

At equal compute the probe estimator is about 10x worse in RMSE (about 100x in MSE) than sampling, and worse
than sampling with only 1,000 evals. On noisy copies the error structure moves away from the original
model's near-tie pattern, which the estimator was built around. **Not cracked.** The harness takes any
estimator `est(model, budget, rng) -> (estimate, evals)`, so another method can be compared on identical
noisy copies and ground truth.

### Thinking outside the box: three more estimators under ARC's protocol

1. **Exact line integration** ([`lineint.py`](lineint.py)). With no biases and ReLU, the network is exactly piecewise linear
   along any line that varies one input. Walking the ReLU breakpoints, solving where the argmax changes and
   integrating the Gaussian density over the correct intervals gives P(correct | other 9 inputs) **exactly**
   (validated against 1M-sample brute force on individual lines; unbiased overall: 0.9521 +- 0.0028). But
   each line costs ~33 forward passes and halves the variance, so at equal compute it is about **17x worse**
   than sampling. (An earlier "62x better" figure in my notes was an arithmetic error: I divided where I
   should have multiplied.)
2. **Randomised quasi-Monte Carlo** ([`qmc_test.py`](qmc_test.py)), on the raw indicator and on the line-integrated
   function. No faster convergence; RMSE ratio to sampling ~1 (raw) and ~3-4 (lines). The line-integrated
   function is still discontinuous in the other coordinates (ties among them, the blind region).
3. **Mechanism-informed stratified sampling** ([`stratified.py`](stratified.py)): **beats sampling.** The mechanism says
   errors live at near-ties of the top order statistics and in the blind region. Those are properties of
   the input, so stratum probabilities are model-free and exact, and the estimate is unbiased for any weights.
   Evaluations go where errors are (Neyman allocation from a pilot on the original model only).
   **[Corrected 2026-10-18, T263]** "Errors live at near-ties" is most, not all, of the error. Per input (T269), 14% of the
   target's error lies at gap >= .05 outside the blind region, and 19-63% on the zoo seeds and noisy copies
   ([SCALING_OBSTRUCTIONS](SCALING_OBSTRUCTIONS.md) §5).

| sigma | compute-adjusted MSE gain over sampling (budgets 1k / 4k / 16k / 64k) |
|---|---|
| 0 (original) | x2.62 / x1.68 / x2.21 / x1.94 |
| 0.0025 | x1.65 / x1.93 / x1.53 / x2.80 |
| 0.005 | x1.23 / x1.16 / x1.29 / x1.48 |

**Correction (R1):** the exact variance ratios for this design are 2.07x (original), 1.78x (sigma 0.0025, 1.44-2.00 per copy) and 1.36x (sigma 0.005, two copies below 1). The higher table values are Monte Carlo noise in the MSE estimate. Candidate generation (~12.9 rejected draws plus sorts per evaluation) costs about as much as an RNN pass and may erase the gain under FLOP accounting. The original evaluation was an inline script; see [`m16/stratified_eval.py`](stratified_eval.py) for a committed version.

Caveats: (a) it is still sampling. The mechanism decides *where* to sample, but this is not a deductive
mechanistic estimate in ARC's sense. (b) Compute counts model evaluations only. Rejection-sampling the rare
strata draws ~13 candidate inputs per evaluation; drawing and sorting 10 Gaussians is cheap next to a 10-step
RNN pass, but under strict FLOP accounting it would eat part of the gain. (c) The gain shrinks with noise,
because noisier copies make errors that are less confined to ties.

### Feynman-style second-order corrections ([`estimate.py`](estimate.py), v4)

Diagnosis first: v3's +0.59 pp bias came mostly from under-predicting 2nd-vs-3rd tie flips (about 0.5 pp)
and an unmodelled "picks the 4th or lower" mode (about 0.27 pp). v4 adds, for every configuration, the
curvature of the flip threshold with respect to each other input (central differences at +-0.5 s.d.),
combined analytically (the exact mean and variance of the quadratic form), plus a rank-4 probe. Still fully
deterministic. Prediction written before running: bias on the original model < 0.2 pp.

- Original model: bias **+0.59 -> +0.25 pp** (prediction narrowly missed). The rank-4 probe finds 0: at
  canonical spacing the net never picks rank 4; those errors come from atypical configurations.
- Noisy copies (6 per sigma): v4 halves the bias on 10 of 12 copies, but **fails badly on one** (-7.7 and
  -9.3 pp; the same noise direction at both sigmas). There some canonical probes are wrong outright, so the
  estimator predicts errors across whole configurations that are really rare.
- Cost about 167k evals. RMSE 3.2 / 3.9 pp vs sampling's 0.06 / 0.07 pp at that compute: **about 50x worse
  under ARC's protocol.**

Lesson: the perturbation series converges slowly (bias roughly halves per order) while cost roughly doubles
per order. Probes at "typical" points are fragile when a model's typical point is atypical. To beat sampling
at about 100k evals, a deterministic method needs bias below about 0.05-0.1 pp **on every model**, about
5-10x better than v4's best case, and with no blow-ups.

## 6. What is still not done

- ARC's challenge (section 5): a deductive estimate that matches sampling. **Update:** about 30 further attempts by
  agents ([`m16/approaches/*/`](approaches), registry E10-E27 in [`INDEX.md`](../INDEX.md)) also failed deductively. Best verified estimator:
  sampling-based srb, ~1.6-2.4x (leaderboard v2). See [`STATUS.md`](../STATUS.md) and [`INDEX.md`](../INDEX.md) section 4b.
  **[Updated 2026-10-18 afternoon]** Still open. The best sampling-free estimator is now T317f (section 8): on a fresh
  sealed panel of 100 retrains it scores F 2.63 [1.93, 3.39] (sampling scores ~1), a pass of the project's lenient bar
  (F <= 10) and a strict fail. Plain sampling at equal compute still has ~2.6x lower MSE, and the estimator fails on
  weight-noised copies. The srb figure above is from leaderboard v2 and is not re-scored here.
- The gating rules are measured (92-100% predictable from record history), not derived from the weights.
- The "assist" (why the max's contribution normally favours p2 over p3) is measured exactly per input; it
  is value-dependent through gating (positions explain R^2 0.05), with no closed form found.

## Neuron labels versus mechanism

One-sentence labels for all 16 neurons rebuild only 54% accuracy (pre-registered E4), and some
descriptions with R^2 0.9 still get the error mechanism wrong (section 4).

## 7. Correspondence to ARC's handcrafted 22-neuron model (ARC open item 2): partial

- The trained state linearly encodes ARC's leave-one-out maxima L_k well (R^2 0.91-0.99 per position), but
  not the per-position values relu(x_k) that the handcrafted delay line stores (R^2 0.56-0.80).
- Applying the handcrafted readout (relu(x_k) + 2 L_k) to the features decoded from the trained state gives
  71.8%; the trained model gets 95.3%. The trained network shares the handcrafted model's key features
  but **not its decision rule**. It decodes through its own gating, which no linear map from its state
  reproduces.

## 8. 2026-10-18 afternoon: a sampling-free estimator, where its error comes from, and rare events

Every number in this section comes from registry rows T317-T353, the panel row T284/T321/T328 and H-L1 in
[`INDEX.md`](../INDEX.md) (DEV3 and HP5 details from [`TASKS.md`](../TASKS.md); the FPE rate from T196). Sealed scores
are by the independent T281 scorer. Results are **unverified** unless marked, and tagged [measured], [inferred] or
[speculative] as in the registry. The full table, with every lane, is [SCALING_OBSTRUCTIONS.md](SCALING_OBSTRUCTIONS.md) §9.

**Terms.** FPE: the cost of one sampled forward pass (5,948 FLOPs); C is an estimator's charged cost in FPE. Panels are
retrains of the zoo configuration: DEV2 (80, open, used for development), the sealed HP2 / HP3 / HP4 (100 each, seeds
100-199 / 200-299 / 300-399), DEV3 (100, open, seeds 500-599; T342, in progress) and HP5 (reserved). Score:
F = mean(error^2) x max(C, 1k) / mean(p(1-p)), so plain sampling scores ~1. **Strict**: F <= 1, i.e. matching
sampling. **Lenient**: F <= 10, a project convention. A **candidate** also needs the upper 95% CI <= 10.

### 8.1 The lineage T317 -> T317f

The base estimator, R1W V1 (T304, T306), is sampling-free at runtime. For each of the 720 (max, 2nd, 3rd) position
triples it walks exactly along the two gap directions, g12 (1st minus 2nd) and g23 (2nd minus 3rd), through the
network's piecewise-linear ReLU-gate cells, with the low inputs held at a closure point. It then integrates the
Gaussian law of the gaps over the pieces where the answer is right. It rests on declared approximations: the closure c
(the lows at their conditional means), v4 (the 4th-largest input) at the triple's latest free position with one node
(K = 1), a walk cut at gap tmax = .5, a blind-region rule, and the product of the two lines' correct sets. These were
chosen and validated by Monte Carlo on DEV2. Each later step is cost-only or adds one declared approximation:

| step | change | estimates | C (FPE) | sealed F [95% CI] |
|---|---|---|---|---|
| R1W V1 (T304, T306) | the base | - | 12.1k (official) | HP2 14.92 [10.74, 19.38]: lenient fail |
| T317 | continuity restarts at gate flips; an executed prefix tree shared across triples | = V1 (6.7e-16) | 7,880 | HP2 9.71 [6.99, 12.61] |
| T317b | 20 Gauss-Legendre nodes | within 1.4e-6 pp | 6,405 | HP2 7.89 [5.68, 10.25] |
| T317c | merge touching correct pieces; model-free Phi tables | = T317b | 5,489 (HP2); 5,427 (HP3) | HP2 6.76 [4.87, 8.79]; **HP3 5.08 [3.42, 7.04]** |
| T317d | tables precomputed once and listed; exact-law transverse points | within .0115 pp of T317c | 5,431 (HP3) | **HP3 5.07 [3.41, 7.03]**; HP2 6.75 [4.86, 8.78] |
| T317e | self-contained, no random draws anywhere | bitwise = T317d | 5,490 (DEV2) | = T317d's: **HP3 5.07** |
| **T317f** (T335) | + a declared frozen-gate margin screen: a walk stops once the frozen-gate margin stays positive to tmax (T330 tier C1) | declared error vs T317e on DEV2: RMS .077 pp, mean +.045, max .34 | **3,302 (HP4)** | **HP4 2.63 [1.93, 3.39]** |

- **T317f on HP4 [measured, unverified].** Scored once, on 100 retrains never used before: RMS .749 pp, bias +.436 pp,
  n* 1,255 FPE. It is a **lenient pass and a strict FAIL** (the CI excludes 1). Plain sampling at equal compute still
  has ~2.6x lower MSE (H-L1). Other cost counts: 2.67 [1.96, 3.45] with the screen's charge audited (+1.5% C), and
  2.98 [2.19, 3.85] with stacked latitude (C 3,743). On the spent panels (secondary): HP3 3.22 [2.19, 4.42], HP2 4.41
  [3.21, 5.66]. HP4's plain-sampling control scores .83-1.25, with every CI covering 1.
- **Caveats [measured].** T317f was selected among ~6 DEV2 variants (forking paths). It makes no random draws, but its
  approximation scheme was chosen and validated by Monte Carlo on DEV2, so it is not deductive in Hilton's strict sense
  (T324's audit). Its scope is retrains only: the family fails on weight-noised copies (section 8.7).

### 8.2 Where the error comes from (T324, T334)

T324 rebuilt the estimator's pointwise form, V1(x) = s23(g23) s12(g12) (the product of the two lines' correct sets),
on exact-law inputs for 80 DEV2 models. It matches the estimator (RMS difference .056 pp against an MC s.e. of .058).
It then switched the approximations off one at a time. A second agent repeated the two largest terms with its own code
(T334, pre-registered).

| switched off | change in bias | change in MSE | independent check (T334) |
|---|---|---|---|
| line-set product (replaced by the joint set) | -.57 pp | **-40%** | -.562 pp [-.623, -.502]; -41% [-57%, -20%] |
| tmax (the .5 cut on the walk) | -.18 | **-18%** | -.183 [-.235, -.136]; -20% [-33%, -1%] |
| blind-region g12 rule | -.07 | -9% | - |
| mu-lin | +.02 | +2% | - |
| single transverse point (interacts with the product) | +.27 | +25% | - |

- Switched off one after another, RMS goes .720 (tmax) -> .685 (+ blind) -> .691 (+ mu-lin) -> .583 (+ product;
  bias -.154) -> .475 (+ transverse point; bias +.155) -> **.463** (+ the K = 1 v4 node; T304's floor). **The K = 1 v4
  node is a sixth declared approximation**, and it moves the bias by +-.31 pp [measured].
- **The split [measured]: the walk approximations carry 66% of the MSE (RMS .80 -> .46) and +.59 pp of bias; the
  closure c carries 34% and -.15 pp.** The product (~40%) and tmax (~20%) terms are **CONFIRMED** by the DRI after T334.
  The product's share of the MSE varies across models (20-57%).

### 8.3 Addendum A: the T306-B anomaly is a cancellation

T306's refinement B (two transverse points instead of one) should have helped. It made the estimate worse (bias
+.78 pp, DEV2 F 37.8). Addendum A (pre-registered; 10 DEV2 models x 1e6 inputs; residue against the closure c at
each model's own point, model-level RMS in pp) explains why [measured]:

| how the two lines are combined | one anchor | half points |
|---|---|---|
| product (AND) | .385 | .495 (= T306-B) |
| zero-cost alternatives: OR / L23 / L12 / AVG | 6.84 / 2.24 / 4.65 / 3.41 | - |
| exact 2-D correct set | .377 | **.087 (J2)** |

The product overstates accuracy by +.30-.34 pp and a single anchor understates it by -.34. At one anchor the two
cancel (per class the residue is 4.9 pp), so fixing either alone makes the estimate worse. Only the exact 2-D correct
set at >= 2 anchors per region works.

### 8.4 The product's overstatement (T333, T347)

- **Where it lives (T333 B3, 10 DEV2 models) [measured].** The net overstatement is +.574 pp. It does not track
  accuracy (corr with 1-p .001), while the gross over-mass does (corr .62). It sits in the **L-band near both tie
  walls**: min(g12, g23) < delta holds 66% / 91% / 101% of the net at delta .05 / .1 / .2, against 14% / 49% / 92% for
  the corner.
- **[Corrected 2026-10-18 afternoon, T347]** T333 called it "model-constant". Over 80 DEV2 models the overstatement
  (V1 minus the corner joint J1) has mean +.568 but **sd .275 per model** (range -.04 to 1.43; corr with 1-p .07). It
  is not constant; it only does not follow accuracy. The best constant still leaves a mean |error| of .207 [measured].
- **Every fix so far is too costly or not derived.** No zero-cost rule works (T333 B1: ADD -65x, OR -182x, L23 -33x,
  L12 -81x). The exact joint set J2 adds ~21k FPE per model even at the line-cell price, ~17x the ~1.2k strict
  allowance (T333 B2) [measured census, inferred verdict]. A corner-only joint J1 removes .08 / .28 / .53 pp for
  +2.6k / 4.0k / 5.6k FPE (go line not met). T347's first-principles wall-tilt formula (no free parameter) misses its
  pre-registered line (mean |predicted - measured| .230 vs .1; corr .54). Used post hoc as a correction, it cuts bias
  +.439 -> -.013 and RMS .796 -> .667 at ~1.4k FPE extra [measured, post hoc].
- **A no-compute review note on T347** (DRI branch) [inferred]: no model-free derivation of the offset exists. Its
  **sign** is model-free and its **magnitude** is not; its apparent constancy is a cancellation between the corner and
  the region outside it.

### 8.5 The closure: spreading the lows (T327, T339, T338, T341)

- **Spread lows, as an oracle (T327 arm B) [measured].** The low inputs go to a fixed alternating pattern,
  mu + (-1)^j sigma, instead of all sitting at their means. The closure floor falls .482 -> .343 pp (bias -.15 -> -.04)
  at no added cost, and the reverse pattern gives .255. The gain comes from breaking the lows' equality. T327 also
  reproduced T319 with independent code: Gaussian-smoothed closures are dead (**verified**, scoped).
- **Built into the estimator (T339): no freeze [measured].** With s = -1: RMS .884, bias +.758, F 4.04 at 3,748 FPE,
  against T317f's .827 / +.489 / 3.14. The decomposition is the finding. **The spread cuts the model-to-model error sd
  from .667 to .455** (corr with truth .978 -> .991; each model moves against its own T317f error, corr -.73), as the **[T343, 14:15: does NOT replicate on fresh DEV3: sd ratio .912 [.76, 1.11] vs DEV2 .68; the variance cut was DEV2-specific]**
  oracle said. **But it exposes bias:** the estimates move up +.27 pp, because the mean-point closure's negative bias
  had been cancelling part of the walk approximations' ~+.6 pp. Post hoc, the variance alone would give F ~1.06 at 3.7k
  FPE [inferred, information only]. **[Corrected 2026-10-18 afternoon]** T327's projection (DEV2 F ~3.8) did not hold
  once the spread was built.
- **Lows chosen from W equal recency (T338) [measured].** Choosing K = 2 lows from the weights gives .410 against the
  closure c's .473. That is indistinguishable from the latest low (.426) or adjacency (.423; difference -.017
  [-.051, +.017]). With no node budget it reaches .309 at 2.4x the nodes, still worse than the zero-cost spread. W ranks
  the *earliest* low positions as most important, which the prefix tree cannot afford.
- **The running-max reading is refuted (T341) [measured].** After T338 the reading [speculative] was that the late
  margins see the lows through order-statistic neurons (running max, leave-one-out max). The oracle floors say no: v5
  at its true position .314, at a fixed slot .483, the true running max .359 (bias +.31), the expected running-max
  profile .602. All are worse than the spread s- (.248; best gap +.066 [+.010, +.120]). v5 helps only at its true
  position. The spread's orientation matters, and s- has a parity-dependent bias (+.058 / -.044 by the parity of the
  latest free position). A sign-alternating recurrent mode of period ~2, responding to the lows' deviation pattern, is
  [speculative]. Section 2 lists a period-2 (Nyquist) mode; whether this is that mode is untested.

### 8.6 Combining the fixes (T351, T353)

- **T351 [measured]** (four pre-registered arms, selection by DEV2 MSE x C): T317f .827 pp at 3,326 FPE (F 3.14);
  spread .884 / 3,748 / 4.04; tilt .670 (bias +.038) / 6,124 / 3.79; spread + tilt **.609** (the family's best RMS;
  bias +.413) / 6,543 / 3.35. The selection keeps T317f: the tilt fixes the bias, but its crossing pass costs +2.8k FPE.
- **T353 [measured, unverified]** reimplements the spread + tilt arm cost-only, reusing the walk's crossing-cell state.
  Its estimates are bitwise equal to T351's arm on 80/80 models. **C 5,079 FPE, RMS .609, bias +.413, sd .451, DEV2 F
  2.60** against T317f's 3.14 (MSE x C 1,883 vs 2,276; go line 2,048). **GO on DEV2.** It is frozen. A pre-registered
  DEV3 check against T317f, with no retuning, is pending, and only if it confirms does T353 become the candidate for
  HP5's single submission. It is a lenient-track gain. Strict at ~5k FPE needs RMS <= ~.38 pp (T351), and the sd .451
  alone would give F ~1.4 [inferred, arithmetic].

### 8.7 Noise breakdown (T337)

T317f on open models (the target and zoo seeds 0-4, 4 copies per level; noise s.d. relative to each matrix's RMS)
[measured, unverified]:

| relative sigma | 0 | .0005 | .001 | .0025 | .005 |
|---|---|---|---|---|---|
| F | 3.3 | 3.1 | 4.0 | 12.0 | 32.4 |
| bias / RMS (pp) | +.58 / .85 | - | +.75 / 1.04 | +1.63 / 2.19 | +2.94 / 3.86 |

- **It breaks down between .1% and .25% relative noise**, tracking the accuracy drop (per-copy estimate error vs true
  error, r = .81-.84). The base model matters: at .0025 the target is off by +.28 pp, zoo1 and zoo2 by +3.3.
- **The errors move away from the tie faces**, which the walk geometry treats as the only wrong places. Interior errors
  with both gaps >= tmax grow from .09 to .76 pp (about half the bias), and errors that pick a rank-4-or-lower position
  grow from .97 to 3.2 pp. At .0025, a walk to tmax 6 cuts zoo0's error from +1.60 to +.93; the screen and the blind
  rule barely matter. The remaining ~+.9 pp is attributed to the closure and the product [inferred]. This fits section
  5 (on noisy copies the error moves away from the near-tie pattern) and the T263 correction there.

### 8.8 Rare events (T340, T344)

This is ARC's long-term goal and the owner's priority: probabilities too small for sampling at this compute to see.

- **T340: partial positive [measured, unverified].** Three models (the target, seed0, seed3), with truths from 1e10
  (target) and 1e9 (seeds) float64 samples. E1(t0) is a rank-3 pick with g23 > t0 (far-gap islands; t0 = 1, 1.25,
  1.5; target truths 5.4e-6 / 1.1e-6 / 2.1e-7). E2(t) is a wrong answer with both gaps >= t (t = .5, .75; target
  9.4e-5 / 5.2e-6). Variant B (uncut walks, per-node transverse points, 0.21-0.32M FPE) reaches .64-.89 of E1's truth
  on the seeds but only .20-.29 on the target (one-signed low; the cause is where the walk lines sit, not the closure).
  Variant A (T317's own transverse point, 11-19k FPE) misses the islands but gets E2 within 3x in 6 of 6 cases. At
  equal compute (post hoc, executed C): A wins 5, ties 7 and loses 3 of 15; B wins 5, ties 3, loses 7; the better
  variant per event family wins 10, ties 3, loses 2. **No single estimator beats sampling in a majority of cases.** An
  earlier "12 of 15" counted ties as wins and was corrected. An independent check on DEV3 (T350) is pending.
- **T344: certificates are infeasible here [measured, unverified; scoped to interval certificates and exact sub-cone
  enumeration].** Interval propagation certifies boxes only <= 1e-4 wide around far-gap correct points, even reduced to
  3-D (v2, g12, g23), because |Wh| compounds over 10 steps. Exact enumeration inside the far-gap sub-cone projects to
  ~2e8 regions at step 10 (kill line 1e6). The only rigorous bound is the trivial P(g23 > t0) (4.4e-2 / 1.7e-2 /
  5.9e-3), **8,200-27,500x looser** than the target's truths. A bound conditional on the declared closure was not
  pursued: it would not be rigorous and could undershoot.

**Update 14:15 (T349 / T340c, unverified):** integrating g12 at K exact-law Gauss nodes instead of its conditional mean fixes both accuracy problems: K3 est/truth on E1 is 8/9 within 2x (target 1.73 / 1.24 / .49; seeds 1.02-1.27), E2 6/6 within 2x (one-signed high), and K5 9/9 E1 within 2x, for probabilities from 2e-7 to 4e-4: the first deductive estimates in the project that close at ~1e-6 to 1e-7. The cost rises to 1.8-4.5M FPE, so at equal compute it wins only where p x C <~ 90 (4 wins / 3 ties / 8 losses of 15); the cost does not scale with 1/p, so the advantage grows as events get rarer. Cost lever: T354; independent check: T350.

### 8.9 What this changes

- [measured, unverified] A sampling-free estimator passes the project's lenient bar on a fresh sealed panel (HP4 F
  2.63 [1.93, 3.39]) and fails strict: sampling still has ~2.6x lower MSE at equal compute. The challenge stays open
  (section 6).
- [measured] Most of its error is in the walk approximations (66%; the line-set product alone ~40%). Each cheap fix
  trades one error for another: the spread exposes bias (T339); the exact joint set and the tilt cost too much or are
  not derived (T333, T347); combining them helps, so far on DEV2 only (T353).
- [inferred] Strict needs a deductive removal of a positive bias (+.41 pp in the best DEV2 arm, T353; ~+.6 pp from
  the walk approximations, T339) whose magnitude is model-specific (T347), and a further cut in the model-to-model
  spread.
- [measured] The geometry assumes errors sit at the tie faces. Weight noise breaks that (T337), and so does the far-gap
  rare-event regime (T340's variant A).
