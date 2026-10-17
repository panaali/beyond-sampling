# Translation/scale quotient of Gaussian accuracy

## Question
Does removing the two continuous task symmetries exactly produce an easier error integral than generic line/circle integration?

## Method
For iid Gaussian inputs and e = ones(T)/sqrt(T), write X = R U + Z e,
where U is uniform on the unit sphere in e-perp, R has chi(T-1) law,
and Z is an independent standard Gaussian. The target second-argmax is unchanged
by adding a common shift or multiplying by a positive scalar. The bias-free ReLU
network is positively homogeneous. Therefore classification on X equals classification
on U + q e, q = Z/R. Its CDF is F(q) = StudentTCDF(sqrt(T-1) q, df=T-1).
For fixed U, the target is fixed along the ENTIRE real line. Within a network gate
cell all logits are affine in q; the correct q values are one interval (possibly
empty). Summing F(b)-F(a) over correct intervals computes p(U) exactly, up to
declared omitted tails and numerical gate walking. Accuracy is E_U[p(U)].

This eliminates radial scale and common translation analytically. It is distinct
from random-plane integration: the plane contains the fixed common-mode vector,
the target never changes, and angular mass is not uniform. It still leaves a
(T-2)-dimensional angular integral. Random U in the experiment below is diagnostic
sampling, not a deductive solution and not a submitted estimator.

### Preregistered alternatives and cheapest falsification (before prototype)
1. **Translation is not the main source of errors.** Even exact p(U) may have almost
all Bernoulli variance. Compare Var(p(U)) with p(1-p) on a fixed DEV sample.
2. **The eliminated variability is real but too expensive.** Count all gate cells,
charge two complete forward recurrences per cell, and compare cost * Var(p(U))
with p(1-p). If this loses severely, stop before larger quadrature.
3. **Apparent variance reduction is a walker or measure bug.** Check the t-law against
an independent Gaussian ratio; compare conditional masses with dense uniform-CDF
midpoints on the same directions. Exact toy linear classifiers provide a further check.

## Status
Done. Negative as a computational estimator. The exact symmetry reduction is retained
as a mathematical building block, but the DEV ceiling test rejects gate-walking it.
No test model tuning or scoring was done.

## Results

Command from repository root (needs public checkpoint access on first load):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/crossover_compute/theory/quotient_probe.py
```

`quotient_probe.py` uses the shared `leaderboard.models(dev=True)` and `forward`
functions, 256 fixed-seed diagnostic directions, and no scored TEST models. It is
not an estimator submission, so no `est(model,budget,rng)` or class label is asserted.
Raw data are in `quotient_results.json`.

| DEV model | conditional variance | raw variance reduction | gate cells/direction | optimistic cost-adjusted gain |
|---|---:|---:|---:|---:|
| sigma .0025, copy 0 | .036065 | 1.554x | 90.293 | .00861x |
| sigma .02, copy 0 | .201132 | 1.195x | 96.828 | .00617x |

The denominator is two COMPLETE recurrence/readout evaluations per visited gate
cell, matching this implementation. Thus it loses by 116x and 162x even before
root solving, CDF calls and input generation are charged. A perfect walker costing
one pass per direction would gain only about 1.2–1.6x on this pilot. This does not
bound what deterministic computation on the remaining angular variable could do.

Checks: the T=2 toy that predicts the minimum's position from the sign of the last
input yields the analytic accuracy .75; zero logits yield .5 including exact tie
handling. On eight directions/model the exact masses match independent 8192-point
CDF-midpoint grids within .0000436 and .0001154. These are grid errors, not certified
walker-error bounds. Direct 50k-input Gaussian checks agree with the 256-direction
integrals at z=.05 and -.79 using both Monte Carlo variances. The omitted q tails
have total mass at most 2e-10; q limits are ±10.2458 for T=10.

The initial restricted run passed the toy checks and stopped while downloading the
public model checkpoint; the rerun with network permission completed. No partial
model results were used to change the design. Total recorded main CPU time is
1.546 seconds, wall time 15.306 seconds (includes checkpoint loading).

## Compute accounting
Two full recurrence/readout passes per gate cell: 46,230 and 49,576 FPE,
respectively, for 95,806 total. Validation adds 115,536 forward inputs/model
(50,000 direct + 8*8192 grid), or 231,072 total. Grand total is 326,878
recurrence/readout FPE, plus tiny toy evaluations. Each model generates 2,560
Gaussian direction coordinates and 500,000 direct-input coordinates; midpoint
q construction uses 8192 t quantiles per model. Root comparisons, normalizations,
sorting, CDF calls and input generation are NOT included in the quoted optimistic
cost-adjusted gains, making the negative result conservative. Wall/CPU figures
include their actual execution. No fitting, LP solves or precomputed advice.

## Claims
- [inferred] The quotient identity is exact for bias-free positively homogeneous
networks under isotropic Gaussian input; no Gaussian margin closure is used.
- [measured] Integrating common translation removes only about 16–36% of Bernoulli
variance on this 256-direction, two-DEV-model pilot, despite removing all rank crossings.
- [measured] Full gate walking loses by at least 116x in this pilot under optimistic
cost accounting. The preregistered estimator route is rejected.
- [inferred] Centered angular configuration, rather than common translation alone,
retains most of the accuracy-estimation difficulty for these copies.

## Alternatives not ruled out
See preregistered tests above. A favorable diagnostic would still require a genuinely
mechanistic treatment of the remaining sphere and honest full-cost scoring.

## Failure modes
Gate enumeration can dominate. A small direction sample can miss rare angular error
patches. Positive homogeneity fails with biases; non-Gaussian inputs change the t law.

## Derivation and relation to existing circle methods

Orthogonal Gaussian decomposition gives Z = e^T X independent of V = X-Ze.
Rotational invariance within e-perp gives V = R U, with independent U and R,
R^2 distributed chi-square(T-1). Therefore sqrt(T-1) Z/R has Student t(T-1)
law. Positive homogeneity cancels R from the network logits; rank invariance
cancels both R and the common shift from the target. This proves the identity
without approximating gates, margins or input moments.

Equivalently q=tan(theta), theta in (-pi/2,pi/2), and the angular density is
Gamma(T/2)/(sqrt(pi)*Gamma((T-1)/2)) * cos(theta)^(T-2). It is NOT uniform arc
length. The earlier generic great-circle estimator averages both direction axes
randomly and walks target crossings; this quotient instead fixes e and integrates
the induced nonuniform angular density. The resulting reduction is mathematically
distinct, but the empirical obstruction remains gate cost and angular irregularity.

No external papers or post-January-2026 ARC sources were read for this work.

## Follow-up for the task board (parent/DRI to register)

A new task is justified only if another route supplies a cheap, accurate function
of the centered angular configuration U. The quotient above can then integrate
its common-mode component exactly. Optimizing this gate walker, or adding ordinary
outer QMC, is not justified by the measured variance ceiling alone. An independent
algebraic check of the quotient and toy construction is cheap; no confirmation is
self-awarded here.
