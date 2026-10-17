# How the 432-parameter AlgZoo model M_16,10 computes the 2nd argmax

Model: `alg_zoo.example_2nd_argmax()`. A 1-layer ReLU RNN with 16 hidden neurons and no biases. It reads
10 i.i.d. N(0,1) numbers and outputs the position of the 2nd largest. Accuracy is 95.3%; chance is 10%.
ARC's published analysis (Hilton, Jan 2026) explains neurons 1, 2, 4, 6, 7 as running-max and
leave-one-out-max features, and leaves the other 11 neurons, the mapping to an algorithm, and an
accuracy estimate open.

Every number below is from code in this directory, on fresh N(0,1) samples (seeds in the scripts).
Each claim says how it was verified and whether it **passed or failed**.

## 1. What the network computes (algorithm level)

The answer depends only on a small discrete state: p1 and p2, the positions of the largest and 2nd largest
of x0..x8, plus where x9 falls (above both, between, or below both).

- **The algorithm is right; the errors come from magnitudes.** Replace every input by the expected Gaussian
  order statistic for its rank (same order, typical spacing) and the model scores **99.96%**. All of the
  4.7% error comes from how the actual values are spaced. `frozen.py`
- **The state is sufficient.** A lookup over the 216 discrete states, each a linear map on (m1, m2, x9),
  reproduces the hidden state with median R^2 0.95 and scores **98.0%**, higher than the model.
  The model's errors are its departures from this clean structure.

## 2. How it computes it (mechanism, derived from the weights)

With the gates that stay on after a record (13 of 16 neurons), the recurrence's eigenvalues form
**an approximate harmonic series**: a damped discrete Fourier memory. `modes.py`

> **Caveat (DRI null test, later):** this gate set is only the *most common* one, covering 16% of states at
> steps 6-10. Other common gate sets give shifted periods (7.7/4.7/3.1; 6.7/4.0/3.0). Random 13-neuron subsets
> of the same matrix show the 1 : 1/2 : 1/3 pattern only 3% of the time, so it is not a coincidence of the weights. But only 2 of 5
> independently trained seeds show a similar series. Treat "Fourier memory" as a picture of one common configuration.
> The phase-clock measurement below is direct evidence and stands on its own. See `INDEX.md`, H-M2.

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
  This separable code alone scores 90-91%, against the model's 95-96%, when x9 is below both. `separable.py`
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
`exact.py`
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

## 5. Attempt at ARC's challenge: a mechanistic accuracy estimate (`estimate.py`)

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

### Under ARC's actual protocol (`arc_eval.py`)

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

1. **Exact line integration** (`lineint.py`). With no biases and ReLU, the network is exactly piecewise linear
   along any line that varies one input. Walking the ReLU breakpoints, solving where the argmax changes and
   integrating the Gaussian density over the correct intervals gives P(correct | other 9 inputs) **exactly**
   (validated against 1M-sample brute force on individual lines; unbiased overall: 0.9521 +- 0.0028). But
   each line costs ~33 forward passes and halves the variance, so at equal compute it is about **17x worse**
   than sampling. (An earlier "62x better" figure in my notes was an arithmetic error: I divided where I
   should have multiplied.)
2. **Randomised quasi-Monte Carlo** (`qmc_test.py`), on the raw indicator and on the line-integrated
   function. No faster convergence; RMSE ratio to sampling ~1 (raw) and ~3-4 (lines). The line-integrated
   function is still discontinuous in the other coordinates (ties among them, the blind region).
3. **Mechanism-informed stratified sampling** (`stratified.py`): **beats sampling.** The mechanism says
   errors live at near-ties of the top order statistics and in the blind region. Those are properties of
   the input, so stratum probabilities are model-free and exact, and the estimate is unbiased for any weights.
   Evaluations go where errors are (Neyman allocation from a pilot on the original model only).

| sigma | compute-adjusted MSE gain over sampling (budgets 1k / 4k / 16k / 64k) |
|---|---|
| 0 (original) | x2.62 / x1.68 / x2.21 / x1.94 |
| 0.0025 | x1.65 / x1.93 / x1.53 / x2.80 |
| 0.005 | x1.23 / x1.16 / x1.29 / x1.48 |

**Correction (R1):** the exact variance ratios for this design are 2.07x (original), 1.78x (sigma 0.0025, 1.44-2.00 per copy) and 1.36x (sigma 0.005, two copies below 1). The higher table values are Monte Carlo noise in the MSE estimate. Candidate generation (~12.9 rejected draws plus sorts per evaluation) costs about as much as an RNN pass and may erase the gain under FLOP accounting. The original evaluation was an inline script; see `m16/stratified_eval.py` for a committed version.

Caveats: (a) it is still sampling. The mechanism decides *where* to sample, but this is not a deductive
mechanistic estimate in ARC's sense. (b) Compute counts model evaluations only. Rejection-sampling the rare
strata draws ~13 candidate inputs per evaluation; drawing and sorting 10 Gaussians is cheap next to a 10-step
RNN pass, but under strict FLOP accounting it would eat part of the gain. (c) The gain shrinks with noise,
because noisier copies make errors that are less confined to ties.

### Feynman-style second-order corrections (`estimate.py`, v4)

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
  agents (`m16/approaches/*/`, registry E10-E27 in `INDEX.md`) also failed deductively. Best verified estimator:
  sampling-based srb, ~1.6-2.4x (leaderboard v2). See `STATUS.md` and `INDEX.md` section 4b.
- The gating rules are measured (92-100% predictable from record history), not derived from the weights.
- The "assist" (why the max's contribution normally favours p2 over p3) is measured exactly per input; it
  is value-dependent through gating (positions explain R^2 0.05), with no closed form found.

## Relation to the essay and to Hilton

On this model the essay's empirical claim holds up: looked at the right way, most neurons have one job,
and the "mixed" ones load on faster rotating modes (the "harmonics" reading was later weakened; INDEX H-M2/H-M3). Hilton's distinction also holds:
one-sentence labels for all 16 neurons rebuild only 54% accuracy (pre-registered E4), and some
descriptions with R^2 0.9 still get the error mechanism wrong (section 4).

## 7. Correspondence to ARC's handcrafted 22-neuron model (ARC open item 2): partial

- The trained state linearly encodes ARC's leave-one-out maxima L_k well (R^2 0.91-0.99 per position), but
  not the per-position values relu(x_k) that the handcrafted delay line stores (R^2 0.56-0.80).
- Applying the handcrafted readout (relu(x_k) + 2 L_k) to the features decoded from the trained state gives
  71.8%; the trained model gets 95.3%. The trained network shares the handcrafted model's key features
  but **not its decision rule**. It decodes through its own gating, which no linear map from its state
  reproduces.
