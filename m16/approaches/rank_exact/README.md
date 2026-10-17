# rank_exact: deductive accuracy estimate over rank orderings (M_16,10)

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
Can the accuracy be computed deductively, class by class over rank orderings? The idea: fix the ReLU gates
for each class, which makes the logits linear in x. P(correct | class) is then the Gaussian measure of a
polyhedral cone. Does this match random sampling's RMSE at equal compute?

**Answer: no.** It is 8-14x worse in RMSE (about 70-200x in MSE).

## Method
The starting point is an exact identity:

    acc = sum_K P(K) * mean_orderings P(correct | ordering, K)

Here K is the number of positive inputs and P(K) = C(10,K)/1024. Every ordering is equally likely, and
the ordering is independent of K.

Approximations:
1. **Classes.** There are 7,920 classes. Each is one (rank1, rank2, rank3) position triple × K = 0..10. The
   other 7 ranks get one fixed, hash-chosen arrangement per triple.
2. **Gate model (the key assumption).** The gates are frozen at the class's canonical input: the expected
   order statistics given K, placed by rank. Then logits = A_c x exactly, where A_c comes from tangent
   propagation through the RNN with the gates frozen.
3. **Integration.** P(argmax(A_c x) = target | class) is integrated over the model-free law of x given
   (ordering, K). It uses a fixed scrambled-Sobol rule with Q points of sorted half-normals. This step
   makes no model calls.

The estimate is **deductive**: no random model inputs. `rng` is unused.

Key files:
- `rank_est.py`: the estimator.
- `estimator.py`: the leaderboard entry. `DEDUCTIVE = True`.
- `run_eval.py`: harness evaluation.
- `adf.py`: Gaussian moment propagation with exact ReLU moments. Tried and failed.
- `diag1.py` to `diag7.py`: diagnostics. These use samples, for diagnosis only.

## Status
**Done.** It does not beat sampling, and the reason is structural (see Claims). I recommend abandoning the
frozen-gate-per-class idea.

## Results
**Original model** (truth 0.953179). Command: `cd m16/approaches/rank_exact && python rank_est.py`

| Q | estimate | bias | cost (fwd-pass equiv.) | sampling s.d. at that cost |
|---|---|---|---|---|
| 16 | 0.9327 | -2.04 pp | 91k | 0.070 pp |
| 64 | 0.9414 | -1.18 pp | 104k | 0.066 pp |
| 256 | 0.9481 | -0.51 pp (quadrature luck, not convergence) | 156k | 0.054 pp |
| 1024 | 0.9430 | -1.01 pp | 361k | 0.035 pp |

**ARC protocol.** `noisy_models(6, sigma)` from `m16/arc_eval.py`, with `true_acc` at n = 4M.
Command: `python m16/approaches/rank_exact/run_eval.py` (from the repo root), writing `eval_results.json`
and `eval_log.txt`. Sampling RMSE is the analytic sqrt(p(1-p)/N) at the same cost.

| sigma | Q | RMSE | mean error | per-model errors (pp) | sampling RMSE at same cost |
|---|---|---|---|---|---|
| 0.0025 | 16 | 1.74 pp | -1.69 | -1.5 -1.6 -1.0 -1.9 -2.1 -2.1 | 0.081 pp |
| 0.0025 | 64 | **0.63 pp** | -0.52 | -0.4 -0.2 -0.0 -0.7 -0.9 -1.0 | 0.076 pp |
| 0.0025 | 1024 | 0.53 pp | -0.40 | -0.3 -0.0 +0.1 -0.6 -0.7 -0.9 | 0.041 pp |
| 0.005 | 16 | 1.14 pp | -0.76 | -0.9 +0.3 +0.4 -1.2 -1.7 -1.5 | 0.100 pp |
| 0.005 | 64 | **1.18 pp** | +0.67 | +0.6 +2.3 +1.6 +0.1 -0.2 -0.4 | 0.093 pp |
| 0.005 | 1024 | 1.20 pp | +0.76 | +0.7 +2.4 +1.6 +0.2 -0.1 -0.3 | 0.050 pp |

At Q=64 (about 104k evals) the RMSE is **8x (sigma 0.0025) and 13x (sigma 0.005) worse** than sampling at
the same compute. That is worse than sampling with only about 1-4k evals.

## Compute accounting
- One forward pass is about 6.1k FLOPs and counts as 1.
- Per class, the gate pass plus a 10-column tangent costs 11. Over 7,920 classes that is 87k.
- Quadrature costs Q × 200 FLOPs per class: 4k at Q=16, 17k at Q=64 and 274k at Q=1024.
- Model-free precomputation, reused across models and charged nothing:
  - order statistics given K, from 2M Gaussian draws;
  - the Sobol rule;
  - the ordering design.
- No random inputs are given to the model.

## Claims
1. [measured] Gates are far from a function of the ordering. 200k inputs fell into 138k distinct
   activation patterns (`diag1.py`). The modal pattern within each record-history class covers 9.5% of
   inputs. The DRI's independent count is 535k distinct patterns in 1M inputs.
2. [measured] Frozen-gate surrogates are biased even at the finest class level (`diag2.py`, `diag3.py`,
   `diag6.py`, `diag7.py`). Surrogate accuracy against a truth of 95.3%:

   | where the gates come from | surrogate accuracy |
   |---|---|
   | canonical point of the full ordering | 92.9% |
   | full ordering + K | 94.2% (about -1.1 pp) |
   | record pattern + K | 93.0% |
   | top-3 positions + K, other ranks fixed | 83-88% (lumping fails) |
   | 2-3 and 1-2 tie points (`diag2.py`) | 90.0%, worse |

3. [measured] Gate errors are spread out. Restoring the true gates of the best 5 neurons (n6, n8, n1, n5,
   n15) lifts agreement with the model only from 94.6% to 98.0% (`diag5.py`).
4. [measured] The estimator converges, as Q grows, to the surrogate limit: 0.943 on the original model, a
   -1.0 pp bias. The bias sign is not stable across models: -0.4 pp on average at sigma 0.0025, and +0.7 pp
   (up to +2.4) at sigma 0.005. Because it is not a constant offset, a fixed correction cannot remove it.
5. [measured] Gaussian moment propagation fails (`adf.py`, `diag4.py`). Given a Gaussian input with the
   exact per-ordering moments, it predicts about 0.49 against a truth of 0.96. Monte Carlo on the same
   Gaussian input gives 0.77. The Gaussian input itself breaks the ordering and puts too little density
   at ties.
6. [inferred] Any method that assigns one gate pattern per rank class has a bias of about 1 pp. Beating
   sampling at about 100k evals needs less than 0.07 pp, so this family cannot win without modelling
   gates as a function of the values.

**Main claim: the bias comes from value-dependent gating.** Alternatives considered:
- (a) Quadrature error. Tested: Q=1024 converges to the same 0.942-0.943 that a sample-based check of the
  surrogate gives (0.9417).
- (b) The fixed arrangement of the other ranks in the design. Tested: with the true ordering per sample
  (`diag6.py`, the "10 true" row) the limit is the same, 94.2%.
- (c) A poor choice of canonical point, as opposed to gating being value-dependent. Partly tested: four
  different gate points (full ordering, +K, record pattern +K, tie points) all give 90-94%. Not tested: an
  optimised point per class, or a small mixture of gate patterns per class with polytope weights.

## Alternatives not ruled out
- A per-class mixture of a few gate patterns, weighted by exact polytope masses (Genz or Botev), could
  shrink the bias. Claim 3 suggests it needs many patterns per class, so it would cost far more than
  sampling. Not tested.
- Treating only the x9-comparator gates (n1, n6, n7, n8) exactly, as hyperplane splits of the cone, would
  probably recover part of the bias. Not tested.
- The cancellation on some noisy copies (errors near 0) could be luck rather than accuracy.

## Failure modes
- Structural bias of about 1 pp whose sign changes with the weights. It does not shrink as Q grows.
- At small Q the quadrature error is correlated across classes, because every class shares the same
  points. Q=256 looks good on the original model by chance.
- The minimum cost is about 91k evals. `estimator.py` returns the Q=16 version for smaller budgets, and
  its cost then exceeds the budget.
- K = 0 or 1 (the blind region) relies on the frozen-gate surrogate where the network is nearly blind.
  Not separately validated.
