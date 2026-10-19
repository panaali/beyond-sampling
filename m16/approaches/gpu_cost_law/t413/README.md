# t413: could an exact dominant-pattern estimator beat sampling at ARC's budget? (T413 design note, no scoring)

↑ [gpu_cost_law](../README.md) · data: [t395/census10.json](../t395/census10.json)

**Question.** T395 finds that the probability-weighted error-pattern growth falls to ~2.7x per step by k = 10. Could an
estimator that enumerates the dominant k = 10 error patterns exactly (from the weights) and handles the remainder beat
plain sampling on ARC's metric (MSE at equal forward-pass-equivalent compute, budgets 1k-256k FPE)? The deliverable is
a cost estimate and a GO / NO-GO for building it.

**Status.** DONE 2026-10-19: **NO-GO (unverified)**. CPU only, no scoring, no model inputs beyond T395's census. Owner:
Claude session_016BGedpds19gq95yjWmDMn5. The board claim has to be recorded by the DRI: this session does not push to
the shared branch.

## Method
`python design.py ../t395/census10.json` writes `design.json`. The best case for the design is a hybrid estimator:
- An oracle supplies the top-N k = 10 error patterns of the model and each pattern's exact error mass, at c FPE per
  pattern.
- The estimate is the sum of those masses plus plain sampling of the indicator 1[error and pattern not in the top N],
  with the remaining B - cN forward passes. Each sample's pattern comes free with its forward pass.
- This is unbiased, with variance q(1-q)/n, where q = (1 - s(N)) p, p = .04689 is the error rate, and s(N) is the
  error-mass share of the top N patterns (T395 plug-in).
- MSE ratio vs sampling: R(N) = q(1-q) B / (p(1-p)(B - cN)), minimised over N.

"Bound the remainder" instead of sampling it is strictly worse: a bound on the unenumerated error mass (1 - s)p is no
tighter than that mass. At B = 256k, sampling's s.e. is 4.2e-4, so a bound needs (1 - s) <= 9e-3, i.e. more than
1e7 k = 10 patterns (top 1e7 cover .912 of the error mass).

## Results ([measured] from the census, [inferred] for costs)
Top-N error-mass share at k = 10: 1e2 .045, 1e3 .122, 1e4 .269, 1e5 .491, 1e6 .733, 1e7 .912. The error mass is spread
thin; the most common k = 10 error pattern holds 0.27% of it.

Best MSE ratio (optimal N in parentheses) at k = 10, by per-pattern cost c:

| B (FPE) | c = 1 | c = 10 | c = 100 | c = 1000 |
|---|---|---|---|---|
| 1k | .995 (5) | 1.0 (0) | 1.0 (0) | 1.0 (0) |
| 4k | .981 (65) | 1.0 (1) | 1.0 (0) | 1.0 (0) |
| 16k | .940 (716) | .992 (13) | 1.0 (0) | 1.0 (0) |
| 64k | .860 (5,129) | .971 (159) | .998 (3) | 1.0 (0) |
| 256k | .729 (34,733) | .918 (1,408) | .987 (32) | 1.0 (0) |

Reference only, NOT exact: the same oracle at k = 7 prefixes would give .183 at 256k (c = 1). But a k = 7 prefix's
error mass needs the 3 remaining steps integrated, which is the closure + walk paradigm (H-S1, dead (verified), scoped).

**Cost of one pattern [inferred].** c >= 1 is a lower bound that ignores everything except reading the pattern out:
- *Discovering* the patterns from the weights: a noisy copy has its own patterns, and the census itself cost 1e10 FPE.
  A branch and bound over gate prefixes visits >= one node per error-carrying prefix (3.6e6 at k = 7 alone, more
  counting non-error branches). Each node needs an LP feasibility check: ~1e5-1e6 FLOPs, ~150-1500 FPE at ~680
  FLOPs/FPE (T196).
- *Exact error mass* of a k = 10 cell: a 10-dimensional Gaussian polyhedral mass, split by which position is 2nd
  largest and the 9 logit comparisons. It has no closed form; numerical cubature to the needed accuracy is >= 1e3
  integrand evaluations.

So realistic c is >= 1e2-1e3 FPE, and the design loses or ties at every ARC budget (R >= .987).

## Decision
**NO-GO.** Even the free-oracle bound (c = 1) only reaches R = .73 at 256k and gives nothing at 1k-16k. Quadrature on
the leaderboard (LB3) already reaches .03-.17 on the original and seed models. At realistic per-pattern costs the
estimator cannot beat sampling at any budget.

## Claims
- [measured] At k = 10, the top 1e4 / 1e5 / 1e6 error patterns carry 27% / 49% / 73% of M_16,10's error mass (1e10-input
  plug-in; the unseen mass is 5.7%, so these shares are slight overestimates).
- [inferred] A dominant-pattern hybrid cannot beat sampling by more than 1.37x at <= 256k FPE even with free exact
  pattern masses, and loses at realistic costs.

## Alternatives not ruled out
- Patterns defined on a coarser equivalence, e.g. merging cells that share logit-difference signs, could concentrate
  the mass more than raw gate patterns do. Not measured.
- A control-variate use of pattern masses, instead of stratification, has the same (1 - s) ceiling to first order.
- Noisy copies may concentrate error mass differently from M_16,10 (the census is of M_16,10 only).

## Failure modes
- s(N) is a plug-in from 1e10 samples, optimistic for N >~ 1e7.
- The FPE cost of LP and cubature is an order-of-magnitude estimate, not measured here.
