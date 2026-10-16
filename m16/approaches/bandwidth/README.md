# bandwidth: does the misclassified band width at a 2-3 tie have a compact description?

## Question
Every deductive estimator so far failed for one measured reason: the errors sit in thin bands next to tie
hyperplanes, and the width of the misclassified band at a point on the tie surface varies irregularly with
the other inputs. A deductive estimate needs a compact description of this function
w(other inputs | tie location). Is there one, derived from the network's structure? If not, what exactly
is the obstruction?

## Method
**Measurement tool (`wtool.py`).** A surface point has positions a < b with x_a = x_b = v, one input above
(the max, at p1) and seven below, so the tie sits at ranks 2-3. Along the unit normal d = (e_a - e_b)/sqrt2
the target is a for t > 0 and b for t < 0. The tool walks the network's linear pieces exactly on the
rank-preserving segment |t| < tau and returns:
- the adjacent misclassified band on each side (wp, wn) and the **signed offset delta = wp - wn**. This is
  where the model's a-vs-b decision boundary actually sits relative to the true tie;
  w = |delta|;
- the exact Gaussian mass of all misclassified t on the segment (`mass`, the integrand of the error estimate);
- the exact gated Jacobian at the tie, which gives the margin m = L_a - L_b = Jab . y (exact by homogeneity),
  the slope s and the first-order offset delta_lin = -m/s.

It is the same algorithm as the validated `boundary/zones.py:walk_wrong`. Masses agree with it exactly
(max difference 0) and with dense-grid brute force to 3e-5. Band edges agree with brute force to 5e-5,
which is the grid resolution (`validate.py`).

**Data (`build_data.py`, `build_big.py`).** Surface points come from the exact model-free law of the 2-3
tie surface (as in `boundary/direct.py`). Sets: orig 60k; orig 300k (learning curve); 30k each for
`zoo_2nd_argmax(16,10,seed=0..4)`, `noisy_models(6, 0.0025/0.005, seed=0)[0..2]` and orig (`orig30k`), all on
the **same** surface points (paired). The analysis uses v > 0 (97%). For v <= 0 part of the segment is in
the blind region, which is a separate regime.

**Hypotheses, written down before any fitting**, each with the test meant to falsify it:

| id | hypothesis | falsifying test |
|---|---|---|
| a | clock/phase: w is set by positions (a, b, p1, time since max) | held-out R^2 of positional models (additive clock, full (a,b,p1) lookup) |
| b | gates: w is set by which comparator/clock gates are on at steps a, b | single-bit gate-flip intervention; canonical-gate freezing; gate bits at a, b, 9 as predictors |
| c | recency: w depends mainly on inputs after the tie positions | Sobol total-effect indices from resampling one input from its exact conditional law, grouped by position |
| d | assist: w is set by the max's contribution to the margin | exact margin decomposition m = pair + max + rest; R^2 of m on the max term; Sobol index of the max |
| e | chaos: w is sensitive to many gates, so no compact description exists | structure function under small input moves; gradient spectrum; learning curve of a flexible regressor; paired correlation across seeds and noisy copies |

## Status
Done. No compact description was found, and the obstruction is characterised (see Claims).

## Results
All commands run from this folder with the venv active.

**1. What w is** (`python validate.py`, quick stats in the session, `python edges.py orig`)
- delta has mean 0.0006 and s.d. 0.029, and is symmetric (44% of bands on the b side). |delta| has median
  0.009, p90 0.038, p99 0.11, CV 1.45. Mass has CV 1.44. The adjacent band carries 79% of the segment's error
  mass; phi(0)|delta| explains R^2 0.90 of the adjacent mass and 0.68 of the total.
- The band edge is always a logit crossing. In 59-64% of lines no gate changes between the tie and the edge,
  so delta = -m/s exactly from one pass at the tie. In 21% one gate flips, and in 15% two or more.
- No gate switches exactly at the tie (0.01% of lines): no neuron is an exact x_a - x_b comparator.

**2. Hypothesis tests on the original model**

| test | result | command |
|---|---|---|
| (a) additive clock model (~134 params) | held-out R^2 0.06 (delta), 0.13 (mass) | `python extra.py` |
| (a) full lookup (a,b,p1), 360 params | 0.19-0.20 (delta), 0.21-0.24 (mass) | `python fit.py orig`, `python transfer.py` |
| (b) gate-flip intervention: bits that individually move delta_lin by > max(0.5|delta|, 0.005) | 104 of 160 per line (median 106). Within an (a,b,p1) class, 32 bits both vary and matter; 0.91 distinct patterns of those bits per line | `python gates.py orig 4000` |
| (b) gates frozen at the canonical (expected order statistic) point of the full ordering, zero parameters | R^2 -0.79; E\|delta\| 0.0224 vs true 0.0165 (+36%). Bit agreement is 95%, but all 160 bits agree on only 7% of lines | `gates.py`, `transfer.py` |
| (b) lookup(a,b,p1) + per-(a,b) linear in the 48 gate bits at steps a, b, 9 (2565 params) | R^2 0.28 (delta), 0.32 (mass); values instead of gates (855 params): 0.23 / 0.26 | `python extra.py` |
| (c) Sobol total effect of one lower input, by position: before a / between / after b | 0.19 / 0.11 / 0.07 (orig); 0.22 / 0.16 / 0.08 (seed2). **The reverse of recency** | `python sens.py orig 3000` -> `sens_orig.out`, `sens_seed2.out` |
| (d) margin decomposition: s.d. of tie-pair / max / rest contributions vs s.d. of m | 38.8 / 34.0 / 18.8 vs **2.77**: a 20x cancellation. R^2 of m on the max term alone is 0.003 | `python cross.py` -> `cross.out` |
| (d) Sobol total effect of the max input / of v | 0.30-0.59 / 0.53. These are the largest single effects; the sum over all inputs is about 1.8, so interactions are strong | `sens_orig.out` |
| (e) structure function S(eps) = E[(delta(y+eps u) - delta(y))^2] / 2Var, random u | 0.002 @0.01, 0.005 @0.02, 0.04 @0.05, 0.08 @0.1, 0.19 @0.2, 0.36 @0.5, 0.56 @1. Locally smooth, not chaotic at the 0.02 scale | `sens_orig.out` |
| (e) gradient spectrum per (a,b,p1) class (exact gradient -Jab/s) | one direction carries 73% of the gradient energy (participation ratio 1.8 of 9). Gradient heterogeneity H = 0.97: the mean gradient is about 0, so the gradient's sign flips within a class | `python spectrum.py orig` -> `spectrum.out` |
| (e) scans of delta along v with the other inputs fixed | 3.4 local extrema, 1.5 sign changes and 9.3 tie-gate-pattern changes per unit v; autocorrelation 0.84 / 0.65 / 0.45 / 0.17 at lags 0.02 / 0.05 / 0.1 / 0.2 | `python vscan.py orig 120` -> `vscan.out` |

**3. Best compact model and learning curve** (`python fit.py orig` -> `fit_orig.out`; `python curve.py` -> `curve.out`)

| model (trained on orig, held-out test) | params | R^2 delta | R^2 \|delta\| | R^2 mass |
|---|---|---|---|---|
| lookup(a,b) | 45 | 0.04 | 0.06 | 0.05 |
| lookup(a,b,p1) | 360 | 0.20 | 0.23 | 0.22 |
| lookup(a,b,p1) + per-(a,b) linear in v, max, 7 sorted lower values | 810 | 0.25 | 0.32 | 0.28 |
| **per-(a,b,p1) linear in the 10 values** (best compact) | 3960 | **0.39** | **0.39** | **0.32** |
| per-(a,b) quadratic | 2970 | 0.27 | 0.27 | 0.22 |
| MLP 3x256, 39k training lines | 145k | 0.39 | 0.41 | 0.35 |
| MLP, 10k / 30k / 100k / 290k training lines | 145k-315k | 0.14 / 0.37 / 0.57 / CURVE290 | | |
| per-(a,b,p1) linear, 10k / 30k / 100k / 290k lines | 3960 | 0.24 / 0.38 / 0.44 / LIN290 | | |

**4. Transfer** (`python transfer.py` -> `transfer.out`; `python cross.py` -> `cross.out`). Held-out R^2 on
paired 30k sets:

| model | refit lookup(abp) delta | refit lin(abp) delta / mass | orig-fit applied, delta / mass | corr(delta, orig delta) same points | E[mass] true vs orig-fit | E\|delta\| true vs canonical-gate |
|---|---|---|---|---|---|---|
| orig | 0.19 | 0.34 / 0.32 | 0.34 / 0.32 | 1 | 0.0087 / 0.0089 | 0.0165 / 0.0224 |
| seeds 0-4 | 0.13-0.24 | 0.18-0.36 / 0.24-0.40 | -0.44 to -0.15 / -0.20 to -0.09 | -0.03 to +0.02 | 0.0097-0.0258 vs 0.0089 | 0.020-0.038 vs 0.031-0.043 |
| sigma 0.0025 (x3) | 0.24-0.40 | 0.38-0.55 / 0.30-0.35 | 0.14-0.28 / -0.01-0.21 | 0.57-0.80 | 0.0116-0.0167 vs 0.0089 | R^2 of canonical gates < 0 |
| sigma 0.005 (x3) | 0.34-0.52 | 0.49-0.61 / 0.29-0.46 | -0.03-0.11 / -0.26 to -0.08 | 0.27-0.50 | 0.0182-0.0373 vs 0.0089 | R^2 0.16-0.22 on two copies |

Margin cancellation factor (sqrt of the summed variances of pair, max and rest over s.d.(m)): 9.5-77 across
the five seeds, 9-18 on noisy copies. Weight noise raises s.d.(m) from 2.8 to 3.2-6.5, so the band offset
grows. The other gradient-spectrum and v-scan numbers are about the same on every seed and noisy copy
checked (`spectrum.out`, `vscan.out`).

## Compute accounting
Analysis only, not an estimator. One line costs about 12 linear pieces (one forward pass plus one JVP each)
on orig and 5-12 on the seeds. Datasets: about 0.66M lines in total (about 8M piece-passes). The Sobol and
structure-function runs add about 30k lines per model, and the v-scans 29k lines per model. MLP training is
CPU only. Nothing here is charged to an estimator.

## Claims
- [measured] The band offset delta is exactly -m/s at the tie in about 60% of lines; otherwise one or a few
  gates flip inside the band. The edge is always a logit crossing.
- [measured] m is a small residual of large cancelling contributions: per-input terms have s.d. 19-39 against
  s.d.(m) = 2.8 on orig, and the cancellation factor is 9.5-77 across the seeds. Freezing gates at the
  canonical point of the input's full ordering (95% of bits agree) breaks the cancellation: R^2 < 0, and
  E|delta| comes out +36% on orig and -12% to +55% across models.
- [measured] (a) Positions explain 20-24% of variance and an additive clock 6-13%: partial, small.
  (c) Recency is refuted: early inputs matter more than late ones. (d) The max and the tie value have the
  largest single effects, but no single term predicts m (R^2 0.003), so the assist matters only together
  with the cancellation. (b) Gates at steps a and b add a little over values, but every gate bit is
  individually relevant and the varying relevant bits form an almost unique pattern on every line.
- [measured] Hypothesis (e) holds in a qualified form: delta is not chaotic at small scales (continuous,
  S(0.01) = 0.002). It is a wiggly function: along its dominant direction (v) there is an extremum every
  ~0.3, a tie-gate change every ~0.1, and a sign-flipping gradient within each (a,b,p1) class. It also
  interacts with all 9 other values (total-effect indices sum to ~1.8).
- [measured] Best compact model: per-(a,b,p1) linear in the 10 values, 3960 params, held-out R^2 0.39
  (delta) and 0.32 (mass) on orig, 0.18-0.36 / 0.24-0.40 when refit on the other seeds. Fitted on orig, it
  does not transfer to the other seeds (R^2 < 0) and barely to noisy copies (delta R^2 0.14-0.28 at sigma
  0.0025, about 0 at 0.005).
- [measured] delta is model-specific: on identical surface points it is uncorrelated across trained seeds
  (|r| < 0.03), and correlation with orig drops to 0.57-0.80 at weight noise 0.25% and 0.27-0.50 at 0.5%.
- [measured] A flexible regressor keeps improving with data (MLP R^2 0.14 -> 0.57 from 10k to 100k lines), so
  delta is learnable but not compact: it costs more parameters than the ~1e5 samples sampling would need.
- [inferred] A deductive estimator built on a compact w-model is not plausible for this network. The integrand
  is a cancellation residual whose sign and size depend on many gates; coarse-class means differ by CV 0.69
  across the 360 (a,b,p1) classes (median within-class CV 0.97). Predicting them from the weights to ~5%
  would need an exact treatment of the cancellation, and zero-parameter structural predictors are off by
  tens of percent.
- [speculative] The cancellation is what training produces: the margin is driven toward 0 at ties, so what is
  left is a sum of 'leftovers' from every gate. That makes it look like a weight-dependent random field. A
  compact theory would then predict only the statistics of w (its scale per class), not w itself. Even its
  scale is not given by the frozen-gate model.

## Alternatives not ruled out
- A better feature basis may exist, for example the hidden state at step b, rank-coded leave-one-out maxima,
  or a model in the eigenmode/phase coordinates of FINDINGS section 2. I tested raw values, sorted values,
  positions and gate bits. The MLP saw positions and raw/sorted values but not hidden states.
- Mixtures of a few canonical gate patterns per class, weighted by exact polytope masses (the H-M8
  alternative), were not tried. The single-pattern version fails badly.
- The MLP learning curve may reflect a poor architecture rather than intrinsic complexity. A
  physics-informed network that reuses the RNN itself would trivially reach R^2 1, which is the point:
  the shortest known description of w is the network.
- Sobol indices use a clipped delta (|delta| <= 0.3), and for the max and v they drop invalid configurations
  (up to 50% for v), so the conditional law is slightly distorted.
- Only the 2-3 tie surface was studied. The 1-2 tie (zone B, about a third of the tie errors) was not.

## Failure modes
- The tool needs the rank-preserving segment; for v <= 0 the segment enters the blind region (excluded here).
- On 5% of lines the band runs to the end of the rank-preserving segment (|delta| = tau): the model is wrong on
  that whole side, so "width" there is set by the segment, not by the network. For variance metrics delta is
  clipped at 0.3 (0.06% of lines).
- Gate-flip "relevance" is a counterfactual on the linear map; it overstates how many bits natural input
  changes reach. That is why the canonical-gate and natural-variation numbers are reported too.
