# T28 rank-law versus gate-map diagnostic

## Question
E30's original conditional Gaussian/frozen-gate estimator remains biased even
with an oracle correction of failure-event overlap. Are rank-law leakage and
gate linearization separable sources of that error? Claim2f63323 was pushed
before implementation. This is a DEV sampling diagnostic, not an estimator.

## Method and frozen scope
Use exactly the same two DEV copies as T37: public original checkpoint,
relative noise sigma.0025 and.02, copy0, default_rng seed1, per-matrix RMS
scaling in leaderboard._noisy. No original/test-model accuracy fitting.
Record checkpoint, model-array and source hashes.

Freeze second-largest quantile strata L=2, all90 ordered largest/second positions,
and128 independent samples per chamber/stratum (23040 rows per input law).
Generate canonical conditional inputs by q=Phi(second)~Beta(9,2), restricted
to each equal-probability stratum; draw maximum above second and the other8
inputs below it using inverse truncated-normal CDFs. Independently generate
Gaussian inputs with the same conditional mean/covariance from E30's numerical
moment table. Fixed RNG seeds28101 (rank uniforms) and28102 (Gaussian normals);
fresh rows per position/stratum, common input rows across the two DEV models.
Input draws are diagnostic compute, not claimed deductive inference.

Build the frozen-gate linear network map at each conditional mean, independently
in this folder, retaining the map through step8 and final readout. Compare the
2x2 combinations on the same chamber labels:

| Input law | Frozen linear gate map | Actual ReLU recurrence |
|---|---|---|
| Rank respecting | rank+linear | rank+network (conditional Monte Carlo reference) |
| Moment-matched Gaussian | Gaussian+linear | Gaussian+network |

For Gaussian inputs the chamber label is intentionally held fixed even when
the sampled input violates that rank chamber; report this leakage explicitly,
and do not call that pseudo-label probability standard network accuracy.
Record exact first-index argmax handling. Compare individual competitor failure
probabilities as well as the joint correctness event, so overlap and marginal
errors remain distinct. Independently compute analytic Gaussian marginal tails
from the same linear map and reproduce E30's summed-tail estimate/union ceiling.

Strongest simple gate rescue predeclared: on rank-respecting inputs, use the
frozen mean-gate map only through step8, then recompute the last two ReLU steps
exactly. Report intermediate negative coordinates from the linear map. This
tests whether a cheap terminal gate correction suffices; it does not repair an
invalid prefix-state distribution or invent a winning estimator.

Controls: rank inputs must have exactly the specified largest/second positions
and correct quantile stratum; theoretical moment table positive semidefinite;
conditional sample moment deviations recorded with no tuning; frozen map at
the mean must reproduce actual mean-state/logits; zero readout must give1/10
by balanced chamber labels for every input law/map. Report paired differences
and stratified SEs; Gaussian-versus-rank rows are independent but indexed as
independent pairs. All90x2 equal-probability cells have weight1/180. Normal
3SE intervals are descriptive, not simultaneous exact certificates. Differences
|bias|−3SE>.001 reject only that fixed approximation at the diagnostic tolerance;
|bias|+3SE<=.001 permits further study; everything else is inconclusive.

## Alternatives before execution
1. Gaussian shape/rank leakage creates wrong tails even under a correct local
   map: compare rank+linear to Gaussian+linear, including per-competitor tails.
2. Frozen gates dominate: compare rank+linear to rank+network on identical inputs.
3. Errors arise mostly in the last steps: test exact last-two-step rescue.
4. Overlap alone could explain a change: compare marginal-max union ceiling and
   sampled joint linear accuracy, retaining the known E30 oracle limitation.

## Status
Bounded diagnostic and predeclared final-two-gate rescue complete; numerical
independent check requested. <=20CPU review lane, one thread,2GB RSS, no new
grid/model/noise sweep. No shared/source edits. Independent conditional-law/SE
derivation review received from T55; Claude remains DRI.

## Results
`RESULTS.md` gives the full comparison and limitations; `results.json` retains
all cell marginal arrays and accounting. Gaussian law leaves the top-two
chamber42.76% of the time. Restoring rank support alone leaves frozen-gate
biases+3.086/+10.608pp; the exact final-two-step rescue leaves−1.363/+5.425pp.
All fixed biases exceed3SE plus.1pp. Total1.56614CPU including setup/imports,
293640KiB RSS. No result is a scored equal-compute estimator or a universal
rejection of conditional non-Gaussian/gate-aware propagation.

## Compute accounting
Charge numerical moment construction even though model free, public weights,
noise, all input generation, matrix factors, frozen gate/Jacobian maps, actual
and rescued network steps, margins/tails and statistics. Report CPU/RSS and
explicit counts, including loaded E30 numerical table source. No scalar-FPE
bound is inferred from elapsed CPU.

## Claims
[inferred] Conditional q~Beta(9,2) with independent upper/lower truncations
generates the exact iid-normal rank chamber in real arithmetic.
[speculative] A rank-respecting law or terminal gate repair may close the E30
gap; neither is assumed sufficient.

## Alternatives not ruled out
More informative chambers, early-gate mixtures, deterministic conditional
integration, exact propagated support, other nonlinear approximations and
other model families. This bounded DEV test does not examine all of them.

## Failure modes
Finite conditional sampling uncertainty; numerical inverse CDF/covariance and
moment-table error; rank leakage under Gaussian pseudo-label laws; frozen maps
can leave the nonnegative hidden cone; covariance matching alone does not
preserve higher moments. A successful diagnostic correction may still cost
more than sampling or be unavailable deductively from weights.
