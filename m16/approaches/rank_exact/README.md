# Rank-ordering ("exact over orderings") estimator for M_16,10: does not beat sampling

## Idea
The target depends only on the rank ordering of x, all 10! orderings are equally likely, and given the
ReLU gates the logits are linear in x. So:

    acc = sum_K P(K) * mean_orderings P(correct | ordering, K),   K = number of positive inputs, P(K)=C(10,K)/2^10

This is exact. Conditioning on K matters because with no biases the sign of each input drives the gates.
The plan was: per class, fix the gates at the class's canonical input (expected order statistics given K,
placed by rank), get the exact linear map logits = A_c x, and integrate P(argmax(A_c x) = target) over
the model-free law of x | class.

## Estimator (`rank_est.py`, `est_rank(model, budget, rng)`, deterministic)
- 7,920 classes: every (rank1, rank2, rank3) position triple × K = 0..10. The other 7 ranks get a fixed
  arrangement, chosen by a hash of the triple.
- Gates come from one forward pass at the canonical point. A_c comes from tangent propagation with the
  gates frozen, costing about 11 forward passes.
- P(correct | class) is integrated with a fixed scrambled-Sobol quadrature (Q points) for sorted Gaussians
  given K. This step uses no model calls, and its FLOPs are charged at about 200 per point.
- Compute is in forward-pass equivalents (about 6.1k FLOPs each): about 91k (Q=16), 104k (Q=64) and
  361k (Q=1024).

## Results
On the original model (truth 0.953179):

| Q | estimate | bias | compute (evals) | sampling s.d. at that compute |
|---|---|---|---|---|
| 16 | 0.9327 | -2.04 pp | 91k | 0.070 pp |
| 64 | 0.9414 | -1.18 pp | 104k | 0.066 pp |
| 256 | 0.9481 | -0.51 pp (quadrature luck) | 156k | 0.054 pp |
| 1024 | 0.9430 | -1.01 pp | 361k | 0.035 pp |

The Q -> infinity limit is about 0.942. It matches a sample-based diagnostic of the same surrogate with
true orderings (0.9417).

Under ARC's protocol (noisy_models(6, sigma), truth from n = 4M): see `eval_results.json` / table below.

RESULTS_TABLE

## Why it fails: diagnostics (`diag*.py`; samples used only for diagnosis)
1. **Gates are not a function of the ordering.** 200k random inputs fell into 138k distinct activation
   patterns (16 neurons x 10 steps). The most common pattern within each record-history class covers only
   9.5% of inputs. Exact enumeration of polyhedral regions (region walking, Genz or Botev polytope masses)
   is hopeless: there is roughly one region per sample.
2. **Frozen-gate surrogates are biased by at least 1 pp even at full resolution.** Surrogate accuracy
   (truth 95.3%):

   | where the gates come from | surrogate accuracy |
   |---|---|
   | canonical point of the full ordering | 92.9% |
   | full ordering + K | 94.2% |
   | record pattern + K | 93.0% |
   | top-3 positions + K, other ranks fixed | 83-88% |

   Lumping orderings destroys it. Gate flips are spread over all neurons. Restoring the true gates of the
   5 worst neurons (n6, n8, n1: the x9 comparators; then n5, n15) only lifts agreement from 94.6% to 98.0%.
   The comparators compare values against ratios of the running max. That is information about values,
   not ranks.
3. **Gaussian moment propagation (ADF, `adf.py`, exact ReLU moments) fails.** With a Gaussian
   approximation of x | ordering, the per-ordering accuracy comes out about 0.49 against a truth of 0.96. A
   Gaussian input breaks the ordering: plain Monte Carlo on the Gaussian inputs alone gives 0.77. It also
   puts density about 0.7 instead of about 2.5 at a 2nd-vs-3rd tie, which is exactly where the errors are.

## Failure modes, summarised
- Structural bias of about -1 pp: frozen gates over-predict errors. It is the wrong sign for any simple
  fix and does not shrink with more quadrature.
- Cost: an exact linear map per class costs about 11 passes, so even 7,920 coarse classes cost about
  100k evals. At that budget sampling already reaches 0.07 pp.
- Small-Q quadrature error is correlated across classes, because every class uses the same points.
