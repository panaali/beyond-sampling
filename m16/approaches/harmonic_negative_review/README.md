# T57 independent review of the harmonic negative

## Question
Does the tested isotropic finite-degree propagation fail because of its cut
implementation, or because correlated readout boundaries expose truncation
error? Which untested alternatives remain plausible? Parent reserved T57 in
afb1211; this folder is the sole write scope. T38 source stays read-only.

## Method and frozen first phase
Reimplement S² cuts using a different derivation: rotate a hemisphere to z>=0,
evaluate standard associated-Legendre spherical harmonics there, and integrate
products directly with Gauss-Legendre z quadrature and equally spaced azimuth.
No Gegenbauer indicator expansion, author's basis recursion or projection
routine is used. A product of degree<=D harmonics is a polynomial of degree
<=2D restricted to the sphere. After exact azimuth integration, odd transverse
powers vanish and the remaining z polynomial has degree<=2D. Thus D+1 Gauss
nodes on[0,1] and2D+1 azimuth nodes integrate it exactly in real arithmetic.
Floating evaluation remains numerical, not interval-certified.

Controls, fixed before execution: degree2/4/8, five deterministic normals,
constant hemisphere mass1/2, complement identity, polynomial output agreement
with the author after a change of basis, and repeated same-hemisphere readout.
For the latter derive the leakage independently using ordinary Legendre
polynomials: lambda_l=(P_(l-1)(0)-P_(l+1)(0))/(2(2l+1)) for odd l, lambda0=1/2.
The retained self-mass is sum_(l<=D)(2l+1)lambda_l². For D=2n its deficit from
1/2 is binomial(2n,n)²/(4*16^n), asymptotic1/(2piD). This is an analytic control
with deliberately aligned past and future halfspaces, not a network proof.

Then use2–3 fixed available trained T3 models, preferring H16 seed0 and a
protocol noisy copy if checkpoints are available, otherwise H2/H4 seed0.
Freeze exact model names/hashes before the numerical run. Compare independent
cut implementation versus original within the same read-only branch harness;
this independently checks cuts, not cone branching or the entire estimator.
Source-audit ties and operation accounting separately. No random inputs or
new truth sampling, no larger-T enumeration, no anisotropic implementation.

Alternatives and cheapest falsifiers: (1) normalization/aliasing error rather
than structural bias: compare directly integrated polynomial cuts in a
different basis; (2) incorrect readout/cone bookkeeping rather than projection:
all-zero/duplicate-logit controls with exact first-index truth; (3) only a
pre-asymptotic failure: exact hemisphere deficit proves a1/D obstruction for
that control but cannot prove all models/degrees fail. Keep this distinction.

## Status
Protocol frozen; implementation and bounded checks next. One numerical cycle
<=110CPU seconds,2GB, primary compute lock, one BLAS/OpenMP thread. Store source
hashes and setup/cut/query CPU/work. No per-model caches are free. This review
must not retire the whole harmonic family when a variant is untested.

## Results
Pending. Reproduction command and all failures will be recorded here.

## Compute accounting
Charge independent setup and all hemisphere basis evaluations/rotations and
contractions; report primitive counts and CPU, without inventing an audited
FPE charge for special functions. Audit the original FLOP expressions and
distinguish implemented cost from a hypothetical big-O implementation.

## Claims
[inferred] The hemisphere cubature computes the same finite-dimensional cut
operator by a different integration method. Numerical agreement remains to be
tested. [speculative] Correlated terminal walls explain much of the reported
bias; network controls are needed before drawing a scoped conclusion.

## Alternatives not ruled out
Tie-adapted anisotropy (T56, not this task), smoothing/filtering, exact terminal
readout correction, local rather than global spectral bases, and backward
future-function projection are distinct from merely increasing isotropic D.
These need separate costed falsifiers. A finite tested negative is not an
impossibility theorem; insufficient budget means inconclusive.

## Failure modes
This direct hemisphere implementation scales poorly in D and is a validation
reference. Sharing the author's cone harness leaves correlated geometry bugs
possible. Existing MC truths have uncertainty. Platform floating arithmetic
and special functions are not certified. Repeated exact-cone shortcuts may
change the sketch's treatment of leakage, so compare the actual branches.
