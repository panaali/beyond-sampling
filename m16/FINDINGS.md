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
  (R^2 0.56 and 0.69). They are the **2nd and 3rd Fourier harmonics**: irregular in position space,
  coherent in frequency space. They are what give the position code enough resolution.
- **Position code: mostly PASSED.** The hidden state is close to a *sum of two position vectors*:
  h ~ m2 U[p2] + (m1 - m2) V[p1]. That is 18 parameters per neuron instead of 216 tables. Read through the
  real output weights, every U[p] row is a bump peaked at exactly position p, about +-2 positions wide.
  This separable code alone scores 90-91%, against the model's 95-96%, when x9 is below both. `separable.py`
- **Causal roles: PASSED.** Knocking out each group (final-state mean ablation) breaks exactly the cases its
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

The first version's bias didn't shrink with more probes, so it was systematic. The missing errors
turned out to be 1st-vs-2nd near-ties (gap median 0.019, the max picked 60% of the time). Adding them halved
the bias. **The result: understanding the mechanism gets the accuracy to within 0.74 points, but at equal
compute this is 4-6x worse than plain random sampling. ARC's bar is not met.** The remaining bias is the soft
tail: a flip threshold depends on all the other values, and a canonical probe fixes them at their means.
Capturing that dependence without in effect sampling inputs is the open problem.

## 6. What is still not done

- ARC's challenge (section 5): a deductive estimate that matches sampling.
- The gating rules are measured (92-100% predictable from record history), not derived from the weights.
- The "assist" (why the max's contribution normally favours p2 over p3) is measured exactly per input but
  not explained in closed form.

## Relation to the essay and to Hilton

On this model the essay's empirical claim holds up: looked at the right way, most neurons have one job,
and the "mixed" ones turned out to be higher Fourier harmonics. Hilton's distinction also holds:
one-sentence labels for all 16 neurons rebuild only 54% accuracy (pre-registered E4), and some
descriptions with R^2 0.9 still get the error mechanism wrong (section 4).
