# T37: reference-error independence ceiling

## Question
Does the trained-minus-handcrafted output residual admit an input-independent law,
globally or conditional on ordered top-two positions, before building covariance propagation?

## Method: frozen protocol before model evaluation
Two DEV copies only: original checkpoint perturbed with relative sigma .0025 and
.02, copy0, noise seed1 as in the shared leaderboard. No original/test accuracy fitting.
Reference scores are g_i=x_i^+ +2 max(0,x_j:j!=i), with analytically known accuracy
.9904405381944444 at T10. Remove common logit shifts from trained/reference scores.
Fit ONE positive beta by least squares f≈beta*g on an independent 16,384-input
calibration set. Report it and reject a nonpositive fit. Define residual
eta=(P f/beta-P g)/s using exactly two predeclared scales:

- fixed s=1: finite second moments; keeps the theoretical low-dimensional endpoint
  factorization, but gives up positive homogeneity;
- radial s=||x||/sqrt(10): residual bounded on the sphere; respects homogeneity,
  but breaks that endpoint factorization. This is a diagnostic ceiling only.

The earlier top-two scale sqrt(M²+S²) is excluded: T19 proved its residual second
moment diverges, with an exact rational checkpoint witness.

Generate disjoint training residual and held-out input pools of 131,072 each,
plus calibration, using fixed seeds 37101/37102/37103 for both models. Evaluate
both models exactly on these pools. No new calibration after observing results.

1. **Global independence:** a random permutation matches each training residual
   to one held-out input, using each residual exactly once. Reconstruct
   P g(x)+s(x)*eta(x'). Compare correctness with actual trained correctness on the
   SAME held-out inputs. Report bias, paired standard error, disagreement and
   reference-only bias. This tests independence while preserving residual shape.
2. **Position-conditioned independence:** within each of the90 ordered top-two
   position cells, pair independent training and held-out samples WITHOUT
   replacement, taking the smaller cell count. Average cell biases with exact
   weight1/90; standard error is sqrt(sum(var(diff_cell)/n_cell)/90²).
   This preserves arbitrary position dependence at no extra continuous endpoint
   dimension. Report discarded surplus rows and minimum cell counts.
3. **Frozen stop/go:** if BOTH independent laws for a given scale/model have
   |bias|-3*SE >.001 (0.1pp), their independence premise is rejected; do not fit
   Gaussian shapes or factor ranks for those laws. More generally advance a law
   only when |bias|+3*SE<=.001. Ambiguous results stay inconclusive without reruns.
   Gaussian/full-covariance and rank0/1/2 shape tests are conditional follow-ups,
   not guaranteed work. If no law advances, stop after independence.

Uncertainty uses distinct inputs and distinct residuals within each comparison,
conditional on the independently fitted beta. It does not include beta-fit
uncertainty, does not establish universal performance, and is not a leaderboard
MSE ratio. Comparisons share pools, so their errors are correlated. Normal 3SE
bounds are descriptive, not simultaneous exact confidence guarantees.

### Alternatives recorded before evaluation
1. Global failure may merely reflect position heterogeneity: compare the exact
   position-conditioned shuffle using the same independently drawn pools.
2. Failure may be incompatible amplitude scaling: compare only the two fixed
   finite-moment scales, with no scale search or clipping.
3. Residual dependence may be small but Gaussian shape/factor rank may fail:
   separate those only if independence clears the frozen tolerance.

## Status
Protocol frozen. No model data evaluated for this task yet.

## Results
Pending. Exact reproduction command will accompany the implementation/results.

## Compute accounting
All computation is diagnostic and charged as such: two models times278,528
forward inputs =557,056 FPE before reference arithmetic, input generation,
least-squares fit, residual normalization, sorting and reconstruction. All fits
are optimistic diagnostic fits, not advertised as a mechanistic estimator.
One thread; maximum300 CPU seconds; no new truth sweep or network required
with the cached public checkpoint. Conditional shape work, if triggered, must be
charged separately and documented before execution.

## Claims
[speculative] A conditionally input-independent residual law is adequate.
[inferred] Both predeclared scales have finite moments; top-two normalization does not.

## Alternatives not ruled out
Richer value/gate conditioning, non-Gaussian state mixtures and different analytic
references may behave differently. No negative result proves T37 impossible.

## Failure modes
Residual/reference dependence, fitted-beta uncertainty, small position cells,
misreading a sampling diagnostic as a deductive estimator, and uncharged fitting.
