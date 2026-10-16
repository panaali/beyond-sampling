# control_variate: "structure + randomness" estimators for the accuracy of M_16,10

## Question
Can a cheap surrogate S(x) with an expectation known without random model inputs (a control variate),
derived from the weights so that it transfers to ARC's noisy copies, estimate accuracy with lower
compute-adjusted MSE than plain sampling and than `m16/stratified.py`?

## Answer in one paragraph
The control variates asked for (surrogates of the correctness indicator) **mostly failed**. They are either
too weakly correlated with the model, or they cost as much as the model. What worked is a different
use of "structure": **exact Rao-Blackwellisation over the last input**. The last RNN step is
`h10 = relu(Wh h9 + Wi x9)`, so given x0..x8 the logits are exactly piecewise linear in x9, with at most
16 breakpoints. P(correct | x0..x8) = Q is therefore an exact sum of Gaussian interval masses. That is the
"structure" part, computed from the weights. The "randomness" part is sampling only the 9-dimensional prefix,
stratified on model-free near-tie strata. The result is the estimator **srb**: unbiased for any weights.
Measured asymptotic compute-adjusted gain: **x2.7 over sampling and x1.28 over stratified** on the
original model, **x2.2 / x1.25** at sigma=0.0025 and **x1.5 / x1.15** at sigma=0.005. It beats stratified
on every one of the 13 models. The gain is real but modest, and it is still a sampling method
(`DEDUCTIVE = False`).

## Method (files)
- `rb9.py`: `Q(model, x)` gives the exact P(correct | x0..x8). It runs 9 RNN steps, takes the breakpoints
  -a_i/Wi_i of the last step's 16 ReLUs plus the top-2 of the prefix (where the target changes), and on
  each piece gets the logits as A + B·x9. "argmax == target" is then an interval in x9, and we add up
  Phi(U) - Phi(L). Validated: mean Q equals mean Y within 1 s.e. on 400k samples for 3 models (`test_rb9.py`).
  There is also a `radial=True` option that additionally integrates the prefix's scale. It is exact,
  because the net has no biases, so 3·x9/|x0..8| ~ t_9. It gave no gain (x2.08 vs x2.06 variance reduction).
- `cv.py`: estimator factory. It has prefix strata (blind: m2<=0, else bins of the min relative top gap
  among x0..x8) with model-free probabilities (`precomp.npz`, from 20M Gaussian draws, no model). It also has
  pooled rejection drawing, and an optional cross-fitted regression control variate on model-free prefix
  features (`basis9.py`) with model-free per-stratum feature means (2-fold cross-fitting keeps the estimate
  unbiased, and the fitted coefficient goes to ~0 when the surrogate is useless).
- `pilot.py` -> `alloc.json`: allocation is half Neyman, half proportional, with Neyman from a pilot on
  the ORIGINAL model only. The same fixed design is used for every copy, the same convention as
  `stratified.py`.
- `estimator.py`: leaderboard entry (`srb`), `DEDUCTIVE = False`.
- `run_eval.py`, `summarize.py`: ARC-protocol runs (arc_eval noisy copies, truth n=4M per copy).
  `variance_calc.py`: precise per-model asymptotic variance per unit compute (pilots of 10-20k per stratum).
- `thr.py`, `feats.py`, `basis.py`, `explore*.py`: the control-variate experiments (mostly negative).

## Status
Done. srb is a modest, robust improvement. The surrogate control variates in the literal sense are a
negative result (below).

## Results
### Main: precise asymptotic gains (`python variance_calc.py <sigma> 10000` (20000 for sigma 0), then `python aggregate.py`).
Compute-adjusted = variance per sample x cost per sample. Averaged over the 6 copies as ARC's MSE does.

| sigma | stratified vs sampling | rb (no strata) vs sampling | **srb vs sampling** | **srb vs stratified** | srb/strat per copy |
|---|---|---|---|---|---|
| 0 (original) | x2.13 | x1.27 | **x2.73** | **x1.28** | 1.28 |
| 0.0025 | x1.75 | x1.22 | **x2.18** | **x1.25** | 1.12-1.36 |
| 0.005 | x1.31 | x1.11 | **x1.51** | **x1.15** | 1.09-1.26 |

(srb is charged 1.7/sample here, see Compute accounting. variance_calc used 1.6, so the table rescales by 1.6/1.7.)
Worst copy (sigma=0.005, acc 0.824): srb x1.05 vs sampling, stratified x0.96. srb degrades towards plain
sampling and does not blow up.

### Harness MSE (`python run_eval.py <sigma> <reps> 4000,16000,64000`; `python summarize.py`)
Original model: 16 repeats. Each sigma: 6 copies x 5 repeats. Ratio = compute-adjusted MSE ratio.
These are noisy (each ratio has about +-35% relative s.e.). The table above is the precise one.

| sigma | budget | srb vs sampling | srb vs stratified | srb_cvA vs sampling | rb_cvB vs sampling |
|---|---|---|---|---|---|
| 0 | 4k / 16k / 64k | x2.86 / x2.76 / x2.76 | x1.19 / x0.94 / x1.84 | x2.55 / x3.15 / x3.01 | x1.96 / x3.32 / x1.70 |
| 0.0025 | 4k / 16k / 64k | x3.82 / x1.51 / x1.67 | x2.07 / x0.75 / x0.84 | x3.17 / x1.64 / x1.94 | x1.64 / x1.18 / x2.09 |
| 0.005 | 4k / 16k / 64k | x1.78 / x1.14 / x1.54 | x1.83 / x1.02 / x1.16 | x1.56 / x1.40 / x1.40 | x1.05 / x1.63 / x1.62 |

Mean signed errors of all srb runs are within about +-0.03 pp (no detectable bias).

### Control-variate experiments (asymptotic variance-reduction factors before cost; >1 is better)
| surrogate S | E[S] how | cost | Var(Y)/Var(Y - beta S) | net verdict | script |
|---|---|---|---|---|---|
| original model's correctness (for noisy copies) | known (amortised) | +1 eval/sample | 1.06-2.46 | net <= x1.2, mostly <1 | explore1 |
| per-(p1,p2,p3) flip-threshold model (estimate.py style, probed on the copy's weights) | exact, model-free gap CDFs | 11.5-14.4k evals of probes | 1.38-1.59 (agreement 89-96%) | loses at <=64k; within strata only adds x1.07-1.38 | explore5, explore11 |
| best model-free cell means (p1p2p3 x gap bins), oracle fit on 1M | model-free | 0 | 1.53-2.22 (oracle, not attainable at n<=64k) | upper bound only | explore2 |
| linear regression on 33-100 model-free features of x | model-free | 0 | 1.19-1.50 | about what stratification already gives | explore6 |
| regression of Q on prefix features (basis9) | model-free | 0 | 1.17-1.54 overall, **1.03-1.14 within strata** | srb_cvA ~ srb | explore8, explore10 |
| frozen rank-canonical gates (FINDINGS) | not computable exactly | >=1 eval | agreement 92.6% < accuracy, so useless | not run | - |

Variance components of Q under prefix sharing (`explore7.py`): Var(E[Q | x0..x7]) = 0.56-0.62 Var(Q). So
also branching over x8 (tree sampling) gains at most about 8%. Not implemented.

## Compute accounting
- Unit: one RNN forward pass on one sequence, about 5.9e3 FLOPs.
- Q per prefix: 9 RNN steps (~5.0e3 FLOPs) + last-step pre-activations (0.5e3) + logits on the first piece
  (0.6e3) + rank-1 updates for <=16 gate toggles (0.6e3) + <=18 pieces x 10 linear inequalities (~1.1e3) +
  <=36 normal-CDF evaluations (~1e3). That is ~9.1e3 FLOPs = 1.53. **Charged 1.6.**
- Stratified input generation: pooled rejection needs ~2.6 candidate prefixes per kept sample, ~500 FLOPs,
  ~0.09. **Charged 0.1.** srb total: **1.7 per sample**.
- Model-free precomputation, reused across models and not charged: `precomp.npz` (stratum probabilities and
  feature means from 20M Gaussian prefixes, no model). `alloc.json` comes from a 140k-prefix pilot on the
  ORIGINAL model. That is a one-off design choice like `stratified_design.json`; strictly it uses the
  original model, so on the original model it is a small leak (allocation only, no bias).
- Comparison caveat in favour of `stratified`: the reference stratified estimator is charged 1 per eval and
  its rejection sampling (~13 candidates of 10 Gaussians per eval, about 0.4 forward-equivalents) is
  **not** charged. Under equal accounting, srb's margin over it would be larger.
- Wall-clock: the numpy implementation of Q is ~5x slower than a forward pass, because it loops over pieces and
  does not do the rank-1 updates. The FLOP count above is for the incremental algorithm. That algorithm is
  straightforward but not implemented.

## Claims
1. [measured] Q (exact integration over x9) is unbiased: mean Q matches mean Y within 1 s.e. on 400k samples
   for base, n05_1 and n025_0. Harness runs show no bias.
2. [measured] Var(Y)/Var(Q) = 2.05 (original), 1.59-1.91 (noisy copies). At a cost of 1.6 this alone gives
   only x1.0-1.3 over sampling.
3. [measured] srb (stratified prefix + Q) beats `stratified.py` asymptotically on all 13 models, by x1.09-1.36
   at the final 1.7/sample charge (x1.16-1.45 at 1.6), aggregate x1.15-1.28.
4. [measured] Surrogate control variates for the correctness indicator give at most x1.1-1.6 variance reduction
   at zero or positive cost, and almost nothing once stratification is in place. The weight-derived threshold
   surrogate does not pay for its probes at budgets <= 64k.
5. [inferred] The reason: within a near-tie stratum, which configurations flip depends on the exact values of
   all inputs through gating (FINDINGS section 4). So no low-dimensional or input-only surrogate captures
   it, and anything that does capture it costs about one forward pass.
6. [speculative] Exact integration over more than one input (2D polygon Gaussian integrals for (x8, x9)) could
   cut the variance by up to a further ~1.8x (Var(E[Q|x0..7])/Var(Q) = 0.56-0.62). That is worth it only if
   it costs < ~1 extra forward-equivalent.

## Alternatives not ruled out
- For claim 3 (srb's gain comes from the Rao-Blackwellisation, not from a better stratification):
  (a) the prefix strata may simply be better strata than stratified.py's. **Tested in part**: without Q
  the prefix strata cannot be used on Y alone in the same way, but unstratified rb is x1.1-1.3 by itself,
  and srb is about rb x stratification-gain, which is consistent with the gains combining multiplicatively.
  (b) the allocation tuned on the original model might favour the original. **Tested**: the srb/strat ratio
  is similar on the noisy copies (1.12-1.36 at sigma=0.0025). Oracle per-copy Neyman would add only 2-12%.
- For claim 4: a richer, nonlinear surrogate fitted per copy (e.g. a small classifier on sorted values and
  positions) might beat the linear and cell bases. The oracle cell-means bound (<= x2.2 before stratification)
  suggests the ceiling is low. Not tested.
- The harness MSE tables use only 16-30 runs per cell. Their per-budget ratios (e.g. 0.75 vs stratified at
  sigma=0.0025, 16k) are within noise of the asymptotic numbers but were not re-run with more seeds.

## Failure modes
- The gain shrinks as noise grows (errors spread into large-gap strata, which no input-only stratification
  targets). Worst copy: x1.05 vs sampling.
- Q needs the last step to be a single ReLU layer applied to an affine function of x9 and Gaussian x9. It is
  specific to this architecture (any bias-free or biased 1-layer ReLU RNN works, but not, e.g., a model
  whose readout has a nonlinearity after x9 enters).
- The CV variants (srb_cvA, rb_cvB) have small-sample regression overfitting. With 2-fold cross-fitting
  they stay unbiased but can be slightly worse than srb at 4k.
