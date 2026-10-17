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

## Fixed model selection, frozen before the first run
H16T3 seed0 checkpoint SHA256
`d9e5245146436fa8ec847a9ec3c771c7dd959708cae64825d46b52d51ec35551`;
seed1 SHA256 `e425ad7db73384432163b51410772181d4efe18ffe0d71c71569a348e0b49a6b`.
Public source URLs are
`https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_16_3_1073741824_0.pth`
and the corresponding `_1.pth`. Root downloaded both. This uses checkpoint
weights only, not additional ARC research text. AlgZoo zoo.py fixes best seed
for(H16,T3) to1, and leaderboard.models uses that as its noisy-copy base.
Third model s0.02_0 is exactly the first copy from default_rng(0), drawing
standard normals in Wi,Wh,Wo order and adding .02 times each matrix RMS.
Compare all three at degrees2/4 against stored eval_T3.json and its existing
truth/se; do not draw new accuracy samples. Direct cut controls use degrees2/4/8.
The fixed all-zero and duplicate-readout models must have accuracy1/3 under
first-index argmax. A failure is recorded rather than silently repaired.

## Follow-up falsifier, frozen after first controls
The first run agrees with the author on all six trained estimates, but both
constant-readout controls return .9375 instead of1/3. Source inspection separates
two effects: zero margins are accepted for every candidate class; and the six
separately projected root cells have total mass .9375 at D2. Freeze a <=30CPU
follow-up at the same degrees/models: fix first-index zero margins in an isolated
in-memory copy, then additionally rescale each model-free root density to its
known1/6 mass. No shared/author edits. This is a diagnostic variant, not an
unreported replacement of the original estimator. Record both adjustments and
test constant readout again. Independently compute root total with Legendre
addition theorem:6*[1/4+sum_odd_l<=(D)(2l+1)lambda_l² P_l(-1/2)].

Record all encountered zero margin queries, gate-map branch counts and the
missing dense matrix-product arithmetic in original accounting. Any cost
correction remains partial: special functions/comparisons/norms and Python
overhead are not fully audited. A model-free root normalization can repair the
constant control without repairing subsequent truncation, so keep the observed
accuracy/bias and do not infer general validity from mass conservation.
