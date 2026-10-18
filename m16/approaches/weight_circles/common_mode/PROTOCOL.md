# Exact common-mode discriminator (T191 continuation9df81752)

## Hypothesis and scope
Test whether correctness is invariant under x -> x+c*ones for the two exact
stored-float H4T4/H8T4 models from T187. This preserves all input ranks. Failure
rejects only projection onto the natural zero-sum subspace (dimension3 here),
not every2D projection, conditional representation or mechanistic circle.
Finite nonfailures establish no global invariance. No Gaussian mass claim.

Pin T187 trained.json/source/protocol and independent3bb92013 evidence. Use all
768 existing strict witnesses, in their saved order, with no case selection.
No checkpoint/model forward, geometry/mass, random/reference or new noise run.
Reconstruct symbolic integer affine maps only from the saved weights/masks,
and require exact equality to every saved logit matrix/denominator. Charge
this reconstruction and all rational interval/probe work as research computation.

## Exact interval construction
At each step reconstruct preactivation coefficient rows B using saved gates,
with the same D^t denominator as T187. For y=x+c*ones, row B is
v+c*s, v=B*x, s=sum(B). A positive saved gate requires v+c*s>0;
a negative nonzero gate requires v+c*s<0. An identically zero row imposes no
constraint and must remain inactive. Every nonzero row is strict at c=0.
Intersect these open half-lines to an interval I containing0. Its endpoints
are exact rational or infinite; endpoints themselves are not gate-certified.
Retain all gate coefficient rows and verify saved histories at x.

Final logit rows L yield lines a_j+c*b_j, where a_j=L_j*x and b_j=sum(L_j).
Partition I at EVERY distinct pairwise line-intersection root in I, including
rival-rival roots. Evaluate each open subinterval at an exact rational midpoint
(or endpoint+-1 for a half-infinite interval;0 if fully infinite), and each
interior crossing separately. First-index maximum resolves every exact tie.
Target is the unique second-largest index of original x. Never infer correctness
from a single derivative or a rival crossing that leaves target status unchanged.

## Robust versus line-only changes
A probe has a robust full-space winner if every tied winning row is IDENTICAL
as a full input coefficient vector, not merely equal on this common-mode line.
For all other rows require strictly positive winner margin. Permanent identical
readouts retain first-index ownership. Otherwise label the probe nonrobust;
line-only or isolated tie changes do not reject an almost-everywhere reduction.

For robust probes compute an exact positive infinity-norm radius: minimum of1,
each nonzero gate slack/(2*L1 gate norm), each input-rank gap/4, and each
nonzero winning-row gap/(2*L1 row-difference norm). Identical rows impose no
margin constraint. The open cube preserves gates, input rank and prediction.
A pair of open-subinterval probes with different correctness and positive
radii certifies opposite correctness for a common open set of zero-sum
projections: shift BOTH cubes by the same sufficiently small perturbation.
Thus this is stronger than a single ambient-null line witness. Save the first
such pair by frozen interval order, plus all probes/crossings for every witness.
Do not call a finite absence of such pairs an impossibility result.

## Frozen controls before trained saved-map contraction
1. H2T3 Wi=(1,1), Wh=diag(1,2), x=(1,2,3),
Wo rows(2,-1),(0,0),(0,-1): all gates active, I=(-1,infinity), true label1,
logit0-logit1=1-c. Expected robust wrong/correct open intervals separated at1;
exact tie at1 belongs to earlier label0. Verify intervals/maps explicitly.
2. Same recurrence/input, Wo rows(1,0),(0,0),(0,-1): no correctness change.
3. Zero Wi/Wh/Wo at H1T3 and x=(2,1,3): all-space interval, label0 permanent
identical tie, correctly robust, no change.
4. Primitive tied-line control x=(1,2,3), L0=(0,0,0), L1=(1,-2,1), L2=(-1,0,0):
rows0/1 agree for all common shifts but differ as vectors. At c=0, label0
must NOT receive a robust winner certificate. No rank tie or numerical threshold.

Freeze code and hashes, obtain independent source/algebra gate, then run controls
only in one <=20CPU/2GB/thread review-lock process. Preserve failures and never
overwrite. After independent controls review, one separately gated <=20CPU
saved-map contraction may process all768 rows; fail durably if a cap or parity
check is hit. No wider numerical phase follows automatically.

## Alternatives and accounting
A nonzero slope may be irrelevant to correctness: all-root classification tests
this. A line-only tie may be nongeneric: full-row tie identity/radii test this.
Failure of common-mode invariance may coexist with another low-dimensional
representation or conditional circle decomposition; neither is excluded.
Counts are explicit big-integer/rational expression calls, not bit complexity,
complete runtime or estimator cost. Record integer bit sizes and wall/process
CPU; fraction gcd/allocation/comparison/control costs remain separate.
