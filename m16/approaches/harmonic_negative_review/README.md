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

## Independent results and scoped verdict
[measured] Direct rotated-hemisphere integration agrees with the author's cut
coefficients within3.94e-14 at D2/4/8 on all five normals. Complement and exact
same-hemisphere controls pass; the largest self-readout discrepancy is5.56e-16.
The independent cut, used in the common branch harness, agrees on all six
trained-model estimates within4.89e-14. Original estimates exactly reproduce
the stored eval_T3.json values. This strongly disfavors a cut-normalization or
aliasing bug as the explanation for the tested negative. It does not
independently validate the shared floating cone implementation.

| Model | D | Existing truth | Original estimate | Tie fix + root normalization |
|---|---:|---:|---:|---:|
| H16T3 seed0 | 2 | .99937775 | .84023619 | .89625193 |
| H16T3 seed0 | 4 | .99937775 | .90568406 | .89735896 |
| H16T3 seed1 | 2 | .99950675 | .80517683 | .85885529 |
| H16T3 seed1 | 4 | .99950675 | .89439833 | .88617696 |
| s0.02_0 | 2 | .85481300 | .63935745 | .68198128 |
| s0.02_0 | 4 | .85481300 | .78213534 | .77494590 |

Truths and their uncertainty are reused from the original stored experiment;
no new independent truth was computed. Zero-margin readout corrections alone
leave all six trained estimates unchanged: no exact-zero classify queries are
encountered. Root normalization helps D2 and hurts D4. Even charging the
normalized variant only the original reported cost, its MSE/sampling ratios
range from449.7 to327314 on these six cells. This favorable comparison ignores
extra normalization work and the accounting omission below; neither correction
rescues this fixed tested implementation.

[measured] Both all-zero and duplicate-readout models incorrectly return .9375
at D2 instead of1/3. The source treats a zero logit margin as satisfied for every
candidate class. Fixing first-index ties alone gives .3125 at D2 and .33642578125
at D4, still wrong because the starting mass is not normalized. Fixing ties
and normalizing every root to its known1/6 gives1/3 to floating precision.
The six projected roots total15/16 at D2,2067/2048 at D4, .9958255738 at D8,
1.0005378404 at D16 and .9998474534 at D32. The exact rational Legendre formula
is saved in diagnostics.json. Therefore gate-split conservation is not a proof
that the whole approximation starts with total probability one.

[inferred] The original finite-degree formulation remains a substantial
negative on these three fixed models, after independently checking its cut and
two plausible repairs. The broad statements "no degree and no budget" or
"higher T is pointless" do not follow from finite-degree experiments. The
closed-form hemisphere deficit proves1/D error on a correlated toy, not on
every network, distribution, spectral basis or readout strategy. Higher-T
resource exhaustion is a limitation of the tested implementation.

## Source and cost audit
`hprop.py` computes Wh@A + outer(Wi,e_t) at every incoming gate branch but never
adds these matrix operations to the counter. On all three checked models the
branch counts sum126, so these actual dense operations add199584 scalar
multiplications/additions (110.88FPE at the source's1800FLOPs/FPE convention).
This is a partial recount, not a complete corrected charge. Branch comparisons,
normalizations, basis special functions and implementation overhead have not
all been audited. Correcting this omission only worsens the tested cost ratio.

The independent reference explicitly counts associated-Legendre/trigonometric
calls, basis values, node rotations and linear contractions; its special
functions have no audited scalar FLOP conversion. Do not treat its exposed
linear-arithmetic subtotal as a fair FPE estimate. Each model run reconstructs
the rule and root densities; no model-dependent memoization is reused.

The author's "Hilton accounting" mD² per cut is a hypothetical leading-order
cost proxy. A big-O expression without constants is not a mathematical lower
bound. The conditional scaling argument is sound *if* bias remains c/D and
cost remains proportional toD^(T+1), but those premises are not an all-degree
impossibility result. cone.classify also uses an absolute1e-11 threshold, so
its description as exact geometry is inaccurate; scale/near-degeneracy tests
are outside this phase. H2/H4 T4 rational certificates from T68 concern another
geometry implementation and do not certify this code.

## Alternative inventory after review
* **Positive spectral smoothing:** on the aligned-hemisphere control a filter
  0<=eta_l<=1 changes retained mass to sum eta_l(2l+1)lambda_l². Every term is
  nonnegative, so damping cannot improve that control's downward bias. It may
  reduce other oscillations; nonlinear/adaptive filters are not ruled out.
* **Tie-adapted anisotropy:** concentrating resolution normal to the correlated
  readout wall changes the state/cost tradeoff. T56 now owns the implementation;
  shared the exact toy, root normalization and first-index controls with its
  author. No negative verdict on this variant is justified yet.
* **Support-preserving local bases:** basis functions restricted to a true order
  cell avoid leakage outside that cell by construction. Their gate-cut cost and
  numerical integration must be paid; this is untested, not impossible.
* **Coupled/terminal readout correction:** measure thin error wedges directly
  or keep an exact discontinuity along the known tie face. The existing wedge
  variant does not exhaust these choices. T63's successful scoped boundary
  account is evidence that changed readout structure can matter.
* **Backward or two-sided sketches:** the cut matrices are self-adjoint. Moving
  a projection from past to future does not help a single identical-halfspace
  pair, but longer chains and adaptive future-specific subspaces can differ.
  State construction, cut integration and selection cost remain untested.
* **Larger isotropic degree:** the exact toy and existing D10–40 observations
  disfavor an easy rescue; they do not justify declaring all larger degrees or
  every model impossible. No higher-degree sweep was run in this review.

## Reproduction, cost and next handoff
From the worktree root, run review.py then diagnose.py in this folder with
`ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python SCRIPT`.
First phase3.76348CPU seconds; follow-up5.65238CPU seconds; total9.41586CPU seconds,
excluding interpreter imports and report serialization. Peak first-phase
RSS218128KiB. Results and diagnostics contain source hashes, model provenance,
all query/setup counters and preserved failed controls. No model-forward input
samples were drawn; the only random draws reproduce the frozen weight noise.

Status: bounded independent phase complete, negative scoped to the checked
isotropic formulation. Parent/DRI owns claim registration. Recommended next
eligible independent-review task is T81 (TT-cross): its weighted contraction
and interpolation can be checked against a completely enumerable small tensor,
separating algorithm error from the chosen face coordinates without repeating
another harmonic engine. Read-only source review may continue while the manager
reserves a distinct task. T56 anisotropic work remains a separate active lane.
