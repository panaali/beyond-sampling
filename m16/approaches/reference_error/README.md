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
Done. The frozen independence premise is rejected for all eight model/scale/law
combinations. Gaussian-shape and factor-rank branches were not run, as required
by the stopping gate. This is a negative diagnostic for this aligned handcrafted
OUTPUT-reference construction, not an impossibility result for every T37 method.
Protocol was committed as58778ab before model evaluation.

## Results

Reproduce from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/reference_error/probe.py --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
```

`probe.py` uses the shared leaderboard forward recurrence and exact DEV perturbation
helper. `results.json` retains all aggregate observations, seeds, matching counts,
checkpoint hash, diagnostic accounting and gate decisions. It does not retain the
large per-input arrays. Checkpoint SHA256:
`498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`.

| DEV sigma | Scale | Residual law | Actual accuracy | Independent-residual accuracy | Bias (pp) | Paired SE (pp) |
|---|---|---|---:|---:|---:|---:|
| 0.0025 | fixed | global | 94.1116% | 45.5505% | -48.5611 | 0.1480 |
| 0.0025 | fixed | positions | 94.1008% | 46.6074% | -47.4934 | 0.1404 |
| 0.0025 | radial | global | 94.1116% | 45.4269% | -48.6847 | 0.1480 |
| 0.0025 | radial | positions | 94.1008% | 46.7856% | -47.3151 | 0.1400 |
| 0.02 | fixed | global | 62.2490% | 39.3364% | -22.9126 | 0.1778 |
| 0.02 | fixed | positions | 62.2067% | 39.6426% | -22.5641 | 0.1470 |
| 0.02 | radial | global | 62.2490% | 38.9259% | -23.3231 | 0.1773 |
| 0.02 | radial | positions | 62.2067% | 39.4996% | -22.7071 | 0.1451 |

All eight satisfy |bias|-3SE>0.1pp by a wide margin. The paired uncertainty includes
the independently drawn residual pool through one-to-one matching; it remains
conditional on the separate fitted beta. No MSE-versus-compute or mechanistic
estimator claim follows from this diagnostic.

Least-squares beta is37.8483165133 for sigma.0025 and36.6008653043 for sigma.02.
The fit used only the independent16,384 calibration inputs. The held-out
reference accuracy is98.9914% globally (analytic99.0441%); the raw reference-only
bias is+4.8798pp / +36.7424pp against the two noisy models. The residual transplant
therefore does not merely fail to improve the reference: destroying the residual's
relationship to its input creates many additional wrong decisions.

Position conditioning uses all90cells with exact weight1/90 and129,253 matched
pairs. Cell sizes range1353–1515;1819rows in each pool are unused because cell
counts differ. Discarding is random within each position cell. These are not
missing probability strata, and the exact cell weighting prevents the selected
row counts from changing the target mixture.

The fixed and radial scales both fail; position conditioning does not rescue
either. The empirical residual law retains its non-Gaussian shape and full joint
output dependence, so Gaussian fitting or factor compression cannot explain this
particular failure. Input/reference dependence, or mismatch of this chosen
reference/alignment, is essential. A different deliberately changed marginal law
could accidentally cancel the bias; this diagnostic does not logically rule out
all independent laws or all alternative reference models.

No Gaussian-shape, covariance or factor-rank calculation was performed. Main
process CPU time1.8192seconds, wall1.8200seconds, including checkpoint loading and
all diagnostics after argument parsing.


## Compute accounting
All computation is diagnostic and charged as such: two models times278,528
forward inputs =557,056 FPE before reference arithmetic, input generation,
least-squares fit, residual normalization, sorting and reconstruction. All fits
are optimistic diagnostic fits, not advertised as a mechanistic estimator.
One thread; maximum300 CPU seconds; no new truth sweep or network required
with the cached public checkpoint. No conditional shape work was triggered. Total exact model calls were557,056
as planned. Inputs are generated once and reused across the two models and both
scales:278,528ten-dimensional Gaussian vectors. Each law/scale reconstruction,
reference computation, beta fit and sorting is included in measured CPU time;
FLOPs for that host arithmetic are not instrumented and are NOT silently counted
as free estimator work. No estimator cost or leaderboard result is reported.

## Claims
[measured] Transplanting the empirical residual independently, globally or within
ordered top-two positions, underestimates accuracy by22.56–48.68pp on the two DEV
copies with the two predeclared finite-moment scales.
[inferred] This particular reference-output-error independence closure is inadequate;
fitting its Gaussian moments would address the wrong failure mode.
[inferred] Both predeclared scales have finite moments; top-two normalization does not.

## Alternatives not ruled out
Richer value/gate conditioning, non-Gaussian state mixtures and different analytic
references may behave differently. No negative result proves T37 impossible.

## Failure modes
Residual/reference dependence, fitted-beta uncertainty, small position cells,
misreading a sampling diagnostic as a deductive estimator, and uncharged fitting.

## DRI handoff

The parent reserved T37 before this work. Claude remains DRI for acceptance and
board/registry updates. Recommended disposition: record this narrow negative,
close this output-residual subroute, and retain broader T37 variants only with a
specific alternative reference/dependence mechanism. No implementation of hidden
reference-error propagation was attempted. The exact T19 gate-crossing remainder
identity remains valid independently of this failed independence closure.
