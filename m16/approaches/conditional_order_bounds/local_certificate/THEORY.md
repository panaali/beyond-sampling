# A local affine enclosure, independent of the old global envelope

This is a proposed bounded follow-up under DRI 01:05/02:10 and claim ffd7494a.
No local certificate has been evaluated yet. The two original H4T4/H8T4 seed0
models are exact rational stored weights from independently reviewed T187.

## Local recurrent enclosure

Let v=2x, so all proposed box endpoints are integers. We represent each hidden
coordinate as h_i=(a_i dot v + e_i)/S, where S is a positive integer common to
all coordinates and e_i lies in an integer interval [l_i,u_i]. The remainder
can depend on v; no independence is assumed. Initially h=0 and S=2. The model
has integer numerators Wi,Wh,Wo over positive denominator D.

At step t, the preactivation has denominator DS, coefficient vector
A_i=sum_j Wh_ij*a_j + Wi_i*(S/2)*unit_t and error interval obtained by the
sign-aware integer sum of Wh_ij*[l_j,u_j]. S remains divisible by2. An affine
range on the input box gives a valid [L,U]. Intersect that range with the
ordinary sign-aware interval recurrence using previously retained nonnegative
hidden ranges; both enclose the same actual preactivation. An empty intersection
is an implementation failure, never permission to discard the box.

After ReLU use denominator2DS. If L>=0, double the coefficients and error
interval; if U<=0, set both tozero. On L<0<U select a fixed slope multiplier
k from {0,1,2}; exactly

    2*ReLU(z) = k*z + r_k(z),
    r_0(z) in [0,2U], r_1(z) in [0,max(-L,U)], r_2(z) in [0,-2L].

Thus the new coefficient is k*A and new error interval is
[k*l, k*u + R_k]. Choose the smallest k minimizing k*(u-l)+R_k, a deterministic
local enclosure rule, not a fitted parameter. The actual hidden interval is
[max(0,L),max(0,U)]/(DS), represented with doubled endpoints at denominator2DS.
This separate interval and affine remainder both remain valid; there is no
claim all points in either enclosure are reachable.

A readout difference uses the exact signed row contrast of Wo on the enclosed
final hidden state. Intersect its affine and ordinary interval ranges. For
true class j, certify correct if every margin j-o has lower endpoint>0 for
o<j and >=0 for o>j. Certify wrong if some margin has upper endpoint<=0 for
o<j or <0 for o>j. Otherwise abstain. These rules retain first-index ties,
including identically zero readouts. Strict/non-strict endpoint distinctions
are conservative on the whole closed input box.

The enclosure is local: stable gates preserve the affine dependence exactly,
while uncertain gates incur box-specific remainders. It does not use the old
T187 absolute-weight derivative or the global-Chernoff mean lower bound.

## Full Gaussian partition and true-label ownership

Freeze boundaries (-4,-3,-2,-3/2,-1,-1/2,0,1/2,1,3/2,2,3,4), hence12 bins
and12^4=20736 Cartesian boxes inside [-4,4]^4. Boundary hyperplanes have zero
Gaussian mass. Everything outside this cube is abstention; no tail is dropped.

For each labeled bin assignment, determine the bin containing the second
largest input. If the highest occupied bin has m>=2 coordinates, it is that
bin; otherwise it is the next occupied bin. Let G be its m labeled coordinates.
Conditional on the Cartesian box, those coordinates are iid with the same
Gaussian distribution truncated to that one interval. Their internal strict
ranks are exchangeable. Therefore each j in G is the second-largest label
with conditional probability1/m. Values in other bins cannot affect this
ownership. No independence between the network output and that rank is assumed.

For each candidate j, apply the whole-box readout certificate. It consequently
holds on the smaller rank-conditioned event. Its mass is exactly boxmass/m;
if the whole-box certificate abstains, that rank event is counted as abstention
regardless of what the actual network does. Other labels have zero true-label
mass on the box. The m events partition the box almost surely.

## Exact probability enclosures and width

At the13 boundaries, bound Phi using the integrated odd/even Taylor polynomials
of exp(-x^2/2), degrees95/96, and the existing independently checked rational
pi/square-root enclosure. Taylor's remainder sign is valid for all nonnegative
arguments; it does not require decreasing term sizes. Reflect for negative x.
Round endpoints outward to multiples of2^-80 using integer floor/ceil. Each
bin mass is bounded by the difference of appropriate opposite CDF endpoints;
all lower bounds must be positive. The product of four bin-mass intervals is
valid by input independence. Divide bym exactly; m=1,2,3,4 can be accumulated
with common denominator12*2^320. Model-free probability setup is recorded,
not silently charged as model work or declared numerically exact.

Accumulate certified-correct, certified-wrong and abstention masses separately,
with rational lower/upper bounds. Tail mass is bounded by1-(Phi(4)-Phi(-4))^4
with opposite endpoints. The lower bound on total abstention is an actual
lower bound on THIS algorithm's unresolved event mass. It is not a lower bound
on classification error, MSE, or uncertainty of a stronger local method.
An accuracy enclosure is [correct_lower,1-wrong_lower]; normalization and
independent partition conservation checks must pass.

## Frozen sampling comparison

Use the existing T196 c_s=218/682 at H4/H8,T4. For N=floor(B/c_s), maximal
Bernoulli sampling SD is1/(2sqrt(N)). For positive abstention lower bound w,
width>10*SD iff N>25/w^2. Integer floor gives the exact smallest N and common
budget beyond which the strict inequality holds. Also report the predeclared
B=10^7 FLOP comparison by exact squared inequalities. Certificate construction
is granted free for this information-width comparison, just as in the prior
T189 bridge; feasibility/full-cost/MSE advantage is not asserted. Record all
actual named integer arithmetic categories and CPU separately.

The DRI stop rule applies only if this frozen valid local certificate still
has width>10*sampling SD at the stated comparison. An implementation/resource
failure or unproved bound is inconclusive. Finer partitions, other slopes,
reachable-polytope bounds, conditional integration and aggregation remain open;
no general impossibility or target-model T10 claim follows.
