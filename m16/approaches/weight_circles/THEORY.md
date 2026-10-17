# T191: measure before circle selection

Claim6d1f9cc9. This is a source/theory phase, not a trained estimator result.
The original Hopf source is pinned in manifest.json. Its generic root-mask
and fibre-continuity assertions already have counterexamples; neither is
assumed here. Local analytic controls may use only the separately reviewed
tile-mask repair, retaining the original source/artifacts unchanged.

## Orthogonal orientation preserves the measure

For fixed network weights, let Q be an orthogonal4x4 matrix determined without
using random model inputs. If X is standard Gaussian, QX has the same law.
For a positively homogeneous correctness indicator f, accuracy is its uniform
S3 integral. Consequently the original Hopf disintegration remains

 A = integral_0^1 integral_0^{2pi} F_Q(u,d) dd du / (2pi),
 F_Q(u,d) = integral_0^{2pi} f(Q[a(u,d)cos(phi)+b(u,d)sin(phi)]) dphi/(2pi).

The original a,b are orthonormal, and each generic fibre is a circle in a
Q-rotated complex line. No extra Jacobian occurs for an orthogonal Q. This
statement is conditional on the weights, so deriving Q from them does not
break rotational invariance. It does NOT say a finite midpoint sum is exact,
that F_Q is universally continuous, or that changing Q changes quadrature into
a mechanistic certificate. Selecting unrelated circles and averaging them
requires a new measure identity; the Hopf law cannot simply be reused.

## An exact sufficient condition for one circle

Let K be a two-dimensional linear subspace with orthonormal basis(a,b).
Suppose a structural derivation establishes f(x)=g(P_K x) for every x
whose projection lies outside a conical angular-null set E in K, and
g(cz)=g(z) for c>0. In particular the identity holds on the selected circle
away from an angular-null set. Ambient almost-everywhere agreement alone is
NOT sufficient: the whole selected circle has zero ambient measure and could
lie in an exceptional set. The projection P_K X is a standard isotropic Gaussian
on K. Its angle is uniform and independent of radius, so

 A = integral_0^{2pi} f(a cos(phi)+b sin(phi)) dphi/(2pi).

This needs the COMPLETE correctness indicator: both the network prediction
and the second-argmax truth. Prediction-only dependence is insufficient.
A decomposition into proven disjoint events can apply the theorem to each
appropriate measure separately, but conditional events do not automatically
inherit an isotropic projected law.

A weight-derived frame can make this exactness visible inside the Hopf family:
choose Q so its first complex line is K and its other line is K-perpendicular.
Projection of every nondegenerate fibre onto K is a circle of radius sqrt(1-u).
By homogeneity F_Q is constant in u,d; u=1 is a null endpoint. One fibre is then
sufficient. This is a real-arithmetic identity, not a floating integration bound.

For an event given by homogeneous halfspaces with all normals in K, the
condition is immediate. For two independent normals with angle theta, the
intersection mass is (pi-theta)/(2pi). The exact angular result supplies a
nontrivial analytic control independent of any outer quadrature.

## What the rank law prevents

All T-label pairwise input differences span the (T-1)-dimensional zero-sum
subspace. For T=4, a sufficient test retaining every rank wall already has
rank3; for T=5, rank4. Thus merely collecting all rank and readout normals
cannot certify the rank2 theorem beyond T=3. This is a limit of that sufficient
test, not impossibility: actual correctness events may have cancellations,
conditional structure, symmetry or a lower-dimensional representation not
captured by the union of all walls. No family is declared dead.

A concrete legal T4 control shows why network-only geometry is insufficient.
Use an allzero readout, so first-index prediction is always0 and true Gaussian
accuracy is1/4. Choose K={x0=0, x1+x2+x3=0}, a two-dimensional plane. On its
uniform circle, the truth is0 exactly when the other three coordinates have
one positive and two negative entries. The antipodal map exchanges that
case with two positives/one negative; coordinate ties occur only at finitely
many angles. The circle accuracy is therefore1/2. Nothing about a constant
prediction alone licenses replacing the Gaussian rank event by this circle.

## Valid recurrent controls in the rank2 regime

H4,T3 control: Wi=(1,-1,1,-1), only Wh[2,:2]=(-1,1) and
Wh[3,:2]=(1,-1) are nonzero. At t>=2 the hidden state is
(x_t positive, -x_t positive, (x_t-x_{t-1}) positive,
(x_{t-1}-x_t) positive). Readout0 selects the third channel, readout1 the
fourth, readout2 iszero. Prediction0 iff x2>x1, otherwise1 away from ties.
Correct orders are x1<x0<x2 and x2<x1<x0, so Gaussian accuracy1/3.
Its correctness depends only on the T3 zero-sum plane, although raw hidden
gates depend on absolute coordinates. A circle in that plane is exactly valid.

Optional stronger synthetic H10,T3 control (not a ladder claim): store signed
current input, signed previous input, signed current-minus-previous,
signed current-minus-two-back, and signed previous-minus-two-back. The last
pair is the prior step's current-minus-previous pair. Scores are negative
sums of the two absolute pairwise differences incident to each input index.
For three distinct numbers, the median uniquely minimizes that sum, so this
fixed RNN predicts the second argmax exactly. Correctness is1, and a cyclic
permutation of the output rows makes it0 almost everywhere. The recurrence
and readout matrices must be written explicitly and independently checked
before execution. These verify the source path and rank/tie semantics, not
trained performance or a generic procedure discovering K from weights.

## Alternatives and cost questions

1. Apparent circle gains may be outer quadrature orientation effects rather
than a structural reduction. The rank2 proof and T4 failure control separate
those claims without fitting a trained grid.
2. The fibre implementation may mis-handle zero coefficients or ties. Preserve
the old mixed-zero counterexample and use the reviewed local mask repair plus
analytic expectations; never treat source parity alone as correctness.
3. Near-rank2 is not exact rank2. A numerical SVD cutoff cannot establish the
identity; approximate residual directions need an explicit probability/error
bound and a separate scope. No such truncation is used in this phase.
4. Building all branch normals may cost full gate enumeration. No generic
rank test is assumed cheap; discovery, certification, frame construction and
fibre arithmetic must all be counted before a trained performance claim.

A useful next discriminator after these controls would be a provable
weight-derived low-dimensional correctness representation or a bound on the
remaining Hopf base integral. Arbitrary grid rotations alone remain quadrature.
