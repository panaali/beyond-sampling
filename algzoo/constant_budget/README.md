# T7: frozen E5 completion and a constant-budget sensitivity check

## Question

Do the frozen M_32,10 draft labels reconstruct its behavior, and how does reconstruction change
with 1, 3, 5, 10, or 20 fitted scalar coefficients per neuron on M_16,10 and M_32,10?

## Method

The original M_32,10 formulas are frozen; a separate agent completes their missing category/text
labels before seeing evaluation results. This is retrospective completion, not a fresh blinded
investigator replication. The original scorer evaluates the wrapper on its unchanged 200,000-input
held-out seed. The M_16,10 formulas are scored as a reference.

For the sweep, use one predeclared nested dictionary for every neuron and both models: m1, m2,
x9; then eight real Fourier coordinates of the first-nine top-two gap indexed by the max position;
then eight Fourier coordinates of m2 indexed by the second-max position; then
m2*cos(2*pi*(p1-p2)/9). The Fourier coordinates are cos/sin pairs in frequency order 1..4.
A K-constant predictor is ReLU(Phi[:,:K] @ beta_i), with exactly K fitted coefficients per neuron.
The structural constants (sequence length 10, nine positions, frequency enumeration, pi, zero, one)
are fixed before fitting, shared by all neurons, and never selected using observations. There are
no learned feature indices, subsets, offsets, normalizers, or hidden per-neuron parameters. This
controls the *fitted scalar* budget, not total program length or arbitrary label complexity.

Training uses 12,000 Gaussian sequences (seed 71101); evaluation uses 200,000 fresh Gaussian
sequences (seed 71102), shared within each model for paired comparisons. Fit each K with
least-squares initialization followed by bounded-iteration L-BFGS of rectified squared error;
report solver status. No test-dependent model selection. The original output matrix is used
only for scoring the reconstructed activations, never for fitting. The script saves all fitted
coefficients for audit. This is activation regression, not a deductive accuracy estimator;
the ARC estimator harness therefore does not apply to this Q2 experiment.

### Alternatives declared before experiments

1. The five-scalar restriction, rather than poor feature choice, may explain failure. Cheapest
   check: compare nested 5/10/20-coefficient models on exactly the same held-out sequences.
2. The feature family may omit pairwise/low-ranked-input effects, so even 20 coefficients could
   fail without disproving compact descriptions. Cheapest check: compare frozen investigator
   formulas against the universal family; a stronger follow-up needs another feature family.
3. Apparent size differences may be due to investigator effort, model seed, or category choices.
   Cheapest check here: use the same family, fitting samples and evaluation protocol for both
   models; multiple trained seeds/fresh independent investigators remain outside this light run.
4. High activation R^2 may hide errors amplified by the output matrix. Cheapest check: report
   both activation fits and whole-network replacement accuracy, including original formulas.

## Status

Done as a bounded sensitivity study; frozen E5 scoring completed. No result was used to choose
this dictionary or the evaluation seeds. The original fresh-investigator E5 protocol is not
fully restored because its qualitative labels were missing. Results are supported, not confirmed.

## Results

The unchanged scorer reproduces M_16,10: **54.133% replacement accuracy, 14/16 legible**,
model accuracy 95.350%, 51.71% of the chance-to-model gap recovered.

Frozen M_32,10 formulas: **89.1925% replacement accuracy**, model 99.442%, **88.54% gap
recovered**, below the 90% criterion (replacement threshold 90.4978%). There are **24/32 neurons
with R² ≥ 0.8**, but only **20/32** satisfy both that condition and the retrospective A/B labels;
this conservative labeling fails the 75% legibility criterion. The original investigator's missing
categories cannot be recovered. Its formulas have 142 fitted scalars total, at most five each,
plus uncharged searched position sets and chosen functional forms. A one-neuron replacement
retains 98.711–99.460% accuracy, unlike the much more sensitive M_16,10 reconstruction.

The fresh-seed sweep below scores regression formulas, with **no new qualitative A/B ratings**;
R² counts alone are therefore not legibility verdicts. Source accuracies on this seed are 95.205%
(M16) and 99.4475% (M32).

| Model | Fitted scalars / neuron | Replacement accuracy | 95% Wilson interval | Gap recovered | Neurons with R² ≥ 0.8 |
|---|---:|---:|---:|---:|---:|
| M_16,10 | 1 | 10.030% | 9.899–10.162% | 0.04% | 1 |
| M_16,10 | 3 | 18.793% | 18.623–18.965% | 10.32% | 6 |
| M_16,10 | 5 | 21.371% | 21.192–21.552% | 13.35% | 8 |
| M_16,10 | 10 | 23.053% | 22.869–23.239% | 15.32% | 11 |
| M_16,10 | 20 | 80.779% | 80.606–80.952% | 83.07% | 12 |
| M_32,10 | 1 | 10.024% | 9.893–10.156% | 0.03% | 1 |
| M_32,10 | 3 | 19.716% | 19.542–19.891% | 10.86% | 8 |
| M_32,10 | 5 | 16.479% | 16.317–16.642% | 7.24% | 10 |
| M_32,10 | 10 | 22.168% | 21.987–22.351% | 13.60% | 16 |
| M_32,10 | 20 | 92.826% | 92.713–92.939% | 92.60% | 19 |

At 20 coefficients, **M32 passes** the reconstruction criterion (92.60% gap recovered);
**M16 fails** (83.07%). Both improve strongly over five coefficients in this family. This
supports budget sensitivity but cannot establish that five coefficients is the sole cause of
the original M16 failure. At five coefficients this generic family is substantially worse than
the investigator formulas; feature choice matters. M32's 3→5 accuracy decrease despite improved
activation loss also shows that activation fit and task accuracy need not move together.

Seven of ten fits hit the predeclared 80-iteration limit (M16 at 5/10/20; M32 at 3/5/10/20).
All improve their training loss from initialization; no global-optimum or capacity limit is claimed.
The achieved M32 threshold crossing remains an existence result for this fitted family. Confidence
intervals quantify held-out input sampling uncertainty conditional on these fits, excluding training,
seed, feature-selection and investigator variability. These outcomes concern reconstruction,
not the ARC mechanistic accuracy-estimation challenge.

Reproduce from the repository root (use a Python environment with numpy, scipy, torch, alg_zoo):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python algzoo/constant_budget/score_frozen.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python algzoo/constant_budget/sweep.py
```

The run used `/workspace/clean-neurons/.venv/bin/python`. Public checkpoint loading needs network
access; source model hashes are in `sweep_results.json`. `score_frozen.py` invokes the unchanged
shared `algzoo/score.py`; it retains the original 200,000 examples and seed 20261016. Its outputs
are `frozen_16_10.json` / `frozen_32_10.json`. The sweep records all neuron fits, binomial intervals,
paired changes between adjacent budgets, solver statuses and timing in `sweep_results.json`,
and every fitted coefficient in `sweep_results_coefficients.json`.


## Compute accounting

Single-threaded CPU; no training of source models. Count all model forward passes for training
and held-out targets, regression objective evaluations, and wall/process CPU time. Input
sampling, feature construction, regression, and matrix scoring are included in process timing.
The original model-loading/cache cost and independent original-scorer runs are reported separately.
No LP solves or GPU use. Cumulative local experiment budget: under 10 CPU-minutes.

Measured core execution: sweep **1.771 process CPU seconds / 34.882 wall seconds**;
frozen M16 scorer **2.403 / 21.046 seconds**; frozen M32 scorer **4.168 / 14.582 seconds**.
Combined recorded core CPU is **8.341 seconds**; interpreter startup, small file inspections,
static audit and one failed sandboxed network attempt are outside those timers. Network time
is included in wall time. All runs remain comfortably below the light-task budget.

The sweep performs **424,000 hidden-state sequence passes** (including all 24,000 training
sequences) plus **2,048 full-forward validation sequences**; fitting uses **665 training objective
calls** including initialization checks. Frozen scoring adds **400,000 hidden-state passes** and
**400,000 full-forward passes**, plus **9.6 million** single-neuron output-matrix scoring examples.
The combined recurrent sequence-pass count is **1,226,048**. No exact FLOP total is claimed;
regression/output matrix products and generation/sorting are timed, not treated as free.
Historical draft formula search and original network training are not repeated or known here.
Timing excludes Python import startup for the sweep and is not a comparative speed benchmark.

## Claims

- [measured] Frozen E5 reconstruction recovers 88.54% of the chance-to-model gap; the original
  threshold is missed, with the stated retrospective-category caveat.
- [measured] Increasing the universal family from 5 to 20 fitted coefficients raises reconstruction
  from 21.37% to 80.78% for M16 and 16.48% to 92.83% for M32.
- [measured] A 20-coefficient, fixed-feature M32 reconstruction recovers more than 90% of the gap.
- [inferred] These results do not support a monotonic deterioration of reconstruction with hidden
  size, and do not justify a general conclusion that legible neurons cannot yield compact formulas.
- [speculative] Better feature families or converged optimization may further improve M16.
- [inferred] A fixed dictionary makes fitted coefficient counts auditable, but it cannot exhaust
  the class of explanations permitted by the preregistration.
- [inferred] The original draft's searched position sets are extra model-specific structural
  information, even when each formula has at most five fitted real scalars.

## Alternatives not ruled out

Other feature dictionaries, nonlinear formulas, additional training effort, seeds, and independent
label raters could yield different outcomes. The fitted coefficients do not establish causal roles.

## Failure modes

A small universal family may underfit, and activation squared error does not optimize output
accuracy. L-BFGS may stop before a global optimum. The original E5 formulas include offsets and
therefore need not preserve the network's exact positive homogeneity. Retrospective labels do
not restore the original investigator blinding or justify strong cross-size comparisons.

## Follow-ups for the task board

- Repeat the sweep with converged fits, other *predeclared* feature orderings/families, and >=3
  source-model training seeds. This separates family/optimizer choices from the coefficient budget;
  done when a held-out curve with uncertainty reports both fitted and structural complexity.
- Obtain fresh independent M_32,10 investigators and category judges with equal effort per model;
  complete a blinded ladder before making claims about how neuron legibility scales with size.
- Audit original labels' searched position subsets and feature-selection cost under a written
  total-description-length rule. Five fitted floats is not five total model-specific constants.

## Independent implementation audit

A second agent wrote `audit.py` using Python sorting and complex roots of unity to independently
construct the same input features. It checked all saved coefficient counts, summary arithmetic,
and positive homogeneity at scales 0, 0.1, 0.5, 2, and 10 on 128 synthetic inputs. Feature values
match exactly; the largest homogeneity error is 2.50e-13. The audit uses zero source-model calls
and does not independently reproduce the held-out accuracies or confirm causal interpretations.
It also identifies an ordering caveat: K=10 ends on an unpaired cosine, so the intermediate
budget curve is sensitive to dictionary ordering. Reproduce with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python algzoo/constant_budget/audit.py
```
