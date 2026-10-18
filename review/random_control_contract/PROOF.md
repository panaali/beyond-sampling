# T236: exact interpretation prerequisites for T232

Scope: bias-free scalar-input ReLU recurrence h_t=rho(W h_(t-1)+w x_t),
h_0=0, t=1..k. Regions mean distinct complete gate histories having nonempty
full-dimensional interior. Boundary-only histories and subdivisions of the
same history are not additional regions. This is exact-real algebra, not an
assertion that floating source enumeration has certified those invariants.

## A. Global matrix-norm rescaling preserves cardinality

Let w'=a w, W'=b W with a,b>0. For input sequence x define z_t=x_t/b^(t-1).
Induction and positive homogeneity give

    h'_t(x_1,...,x_t)=a b^(t-1) h_t(z_1,...,z_t).

The preactivations have the same positive scale, so all gates, including zero
ownership, correspond. The diagonal map x->z is invertible on R^k, preserves
dimension and maps open sets to open sets. It therefore bijects nonempty
full-dimensional gate histories: N_k(w',W')=N_k(w,W) for every finite k.
Readout changes do not affect these prefix gates.

Consequently positive Frobenius rescaling of each random matrix cannot change
its exact prefix-history count from the underlying draw. This remains true if
a,b are functions of the draw, provided finite and positive. Zero matrices or
zero target norms need explicit treatment; they are outside that statement.

The time-diagonal transformation is generally NOT orthogonal. Under iid
standard-Gaussian x, z has unequal variances b^(-2(t-1)); Gaussian region masses,
N_delta, and rank-label accuracy need not be preserved. Also the second-largest
position in x can differ from that in z. Positive global scaling of w alone
or of the whole readout preserves labels at the SAME x; recurrent scaling b
has no such general accuracy guarantee. This separates cardinality from mass.

Independent row normalization of W is not one global b and is not covered.
It can change counts. A full positive-diagonal state reparameterization
(w',W',Wo')=(D w,D W D^-1,Wo D^-1) instead preserves gates up to identical
coordinates and outputs at the SAME input, hence counts and Gaussian masses.
It is a different operation from separately matching row norms.

## B. Generic depth-two count and its exceptions

Assume H>=2, all w_i nonzero, and condition on w. For the two open first-input
halves write x_1=s u, s in{-1,+1}, u>0, and a_s=rho(s w). The second-step forms
are

    (W a_s)_i u + w_i x_2.

After section u=1, these gates change at scalar thresholds
r_i=−(W a_s)_i/w_i. If a_s is nonzero and the H thresholds are distinct, there
are exactly H+1 open intervals, all with different second-step gate words:
each coordinate flips once and distinct thresholds flip only one coordinate.
Each interval gives a nonempty full-dimensional history cone. The two first
halves have different first-step words and so their counts add.

If W has independent rows with an absolutely continuous joint density,
conditional on w and nonzero a_s each projection (W a_s)_i is continuous;
independent row projections make equality of any two thresholds probability
zero. Thus when w has both signs, N_2=2(H+1) almost surely over W.

If all w_i share one sign, one a_s is zero. On that half all second-step
thresholds coincide at zero and there are exactly TWO intervals. On the other
half the generic H+1 argument applies. Therefore N_2=H+3 almost surely.
Zero w_i or algebraically special W can give other counts; they are not covered
by the generic equality and are not evidence of a counting defect.

For independent centered Gaussian w_i, signs are independent fair signs. Hence

    P(N_2=2H+2)=1−2^(1−H), P(N_2=H+3)=2^(1−H).

At H=16 these counts are34 and19, with probabilities32767/32768 and1/32768.
These are ideal-distribution statements, not statistics from five saved draws.
The target's verified34 therefore equals the usual generic depth-two value;
it alone does not establish a training-created reduction in complexity.

Positive global norm matching preserves the counts by A. For row-matched
Gaussian W, assume every target row norm is positive. The independently
normalized rows have continuous directional laws on S^(H-1), whose projection
onto any fixed nonzero a_s has a continuous distribution for H>=2. The same
threshold-distinctness conclusion applies. Row-matched scalar w_i has random
sign and fixed NONZERO target magnitude; its independent fair signs give the
same two probabilities. These conclusions require the actual generator to
implement the stated independent Gaussian laws and do not certify a finite
pseudorandom program or any particular floating realization.

## C. A stronger deterministic arrangement upper bound

Assume at least one w_i is nonzero. For each full-dimensional history of length
t, its exact cone C has inward gate-inequality normals spanning R^t: select the
same nonzero-w_i neuron at each time. The selected forms are triangular by time
with nonzero newest-coordinate coefficient, even when earlier gates are OFF.
Thus C is pointed. Taking the sum of all inward normals supplies a linear
functional ell positive on C\{0}; each summand is nonnegative on C and full
span excludes simultaneous zero at a nonzero point.

Extrude C by the new scalar input. Outside the lower-dimensional past-origin
boundary, the section ell(past)=1 represents every ray positively. It is an
open convex domain of dimension t (a (t−1)-dimensional parent section times R).
The H new gates restrict to affine hyperplanes there. H affine hyperplanes in
R^t have at most sum_(j=0)^min(t,H) binom(H,j) open arrangement regions, by the
usual insertion recurrence: each added plane creates at most the number of
regions induced by earlier planes on it. Restricting to a convex domain cannot
split one arrangement region into disconnected pieces. Therefore

    N_k <= 2 product_(t=1)^(k−1) sum_(j=0)^min(t,H) binom(H,j).

This improves the author's valid but loose per-parent full-space central-
arrangement bound. In particular N_2<=2(H+1)=34 at H16. The generic equality
in B saturates it at depth2. No generic higher-depth equality or per-step growth
factor is asserted. If w=0 then all states vanish and one history remains;
it is a separately trivial case. Counts are not minimal affine maps, true
computational lower bounds, or necessary costs for algorithms aggregating
histories. This proof does not validate numerical ray/witness completeness.

## D. Controls that preserve or change the problem

A consistent hidden-neuron permutation P uses w'=P w, W'=P W P^-1 and
Wo'=Wo P^-1. It relabels histories bijectively and preserves outputs, masses,
accuracy and all N_delta exactly. It is an implementation identity control,
not a new random network. Permuting entries WITHIN each recurrent row or
randomly flipping individual recurrent entries, as the preregistration states,
is not such a similarity; those changes may alter all measured quantities.

Matching scalar/global norms cannot by itself explain a cardinality difference
by A, although active-neuron/mass statistics can change. Matching row norms
controls only that particular set of magnitudes, not singular structure,
alignment with input/readout, or a unique causal mechanism of training.

A randomly replaced readout also changes accuracy independently of hidden gate
counts. Row-normalized readout rows need not be exchangeable. Even an iid
exchangeable readout requires handling h_T=0 / first-index ties before asserting
an unconditional chance baseline; the all-OFF event can correlate with the
truth label. No chance-accuracy formula is assumed here.
