# T66, T20, T26 and T81 independent source/algebra review

## Projective-circle geometry (T66)
For orthonormal U,V, the charts sigma*(U+tV)/sqrt(1+t²), sigma=+/-1,
cover the circle except null endpoints. Positive homogeneity removes the
normalization for gates, output argmax and input ordering. Interval angular
mass is (atan(hi)-atan(lo))/(2pi). Both charts have total mass1. For a+bt,
the right-limit active rule is current>=-a/b when b>0 and current<-a/b when
b<0; b=0 uses a>0. Identically zero gates stay off. All pairwise logit and
input roots suffice within a fixed gate cell. Identical readouts have
positive-measure first-index ties. Identical input forms on degenerate planes
need an explicit rank tie convention; the source uses stable index order.
Random outer planes remain a sampling method even with exact circle slices.

Read projective.cpp, native.py and fraction_oracle.py. The exact Fraction
oracle enumerates gate masks with exact affine constraints, then exact pair
roots and exact midpoint decisions; 70-digit atan computes mass. Its zero
form ownership, unbounded-interval points and stable ties are consistent.
It is independent of the native right-limit walker, not an independent
arbitrary-precision probability enclosure. Native caps raise, finite-root
failures raise, and no partial value is returned. Near-orthonormal input
checks and long-double coefficients remain numerical qualifications.

Incremental-cache proof: choose globally smallest future gate root and the
earliest step attaining it. Retain only states/roots strictly before that
step, and recompute that entire step and all later steps at the new right
limit. Earlier affine prefix states are valid until their own next root;
all simultaneous changes are included by recomputing the earliest affected
suffix. Retaining a later simultaneous step would be incorrect. This proves
exact-arithmetic caching, not rounded root ordering. Require exact piece and
interval counts plus values against full recomputation on fixtures.

Two residual numerical falsifiers were sent to the owner: a finite midpoint
can round to an endpoint between adjacent long-double roots; atan subtraction
can cancel on narrow large-t arcs. Neither is ruled out by1e-12-spaced roots.
No tolerance merging or finite probe should be introduced to hide them.

## Radial tube callback (T20)
Substitution t=k/sqrt2 gives A*(1+t²)^-5 dt with A=sqrt2*C_RAD. Its full-line
mass is2sqrt(pi), preserving the original coarea normalization rather than
normalizing to1. On nonnegative endpoints, use reversed beta variable
z=1/(1+t²), parameters(9/2,1/2), prefactor A*B/2. Reflect negative intervals;
across zero use signed z=t²/(1+t²), parameters(1/2,9/2).
The second derivative is10A*(11t²-1)/(1+t²)^7, absolute supremum10A;
the endpoint-based local upper bound in radial.py is valid. The algebraic
midpoint denominator1+a²+a*delta+delta²/4 avoids explicitly rounded midpoint
construction, but evaluation/selection rounding is not enclosed.

Read radial.py/RADIAL.md. The finite interval branches match these formulas.
One uncovered API boundary was reported: across infinite endpoints, z uses
inf/inf and can produce NaN. Same-sign semi-infinite intervals use1/(1+t²)
and are safe algebraically. Either enforce finite callback endpoints or set
limiting z=1 explicitly and test. The owner has the report; no production
edit here. Beta100 is an explicitly opaque scalar-operation proxy. Duplicate
shared CDF allowances are conservative but do not prove a portable bound.

## Reachable selective patches (T26)
Within a fixed gate history, final state perturbations are A*dx. A desired
isolated patch P*delta is locally feasible iff it lies in range(A) and the
input perturbation satisfies all gate-cell inequalities. Nonnegativity of
patched hidden state alone is insufficient. Closed inactive zero constraints
must not be replaced by strict slack requirements. Failure in one gate cell
is not global unreachability across all histories. The relevant ambient rank
is the active reachable subspace, not all16 neurons.

For orthonormal group columns Q and orthonormal range basis U, impose
Q^T U a=b. If feasible, a=(Q^T U)^+b gives the smallest ||d|| for d=Ua.
Since Q^T d=b is fixed, ||(I-QQ^T)d||²=||d||²-||b||², so this also minimizes
complementary leakage. dx=A^+d gives a minimum-input-norm witness. Shrinking
both target/control by a common positive alpha below every gate boundary
preserves the linear map; actual forwards must verify all intermediate gates.
Input order needs its own constraints if it is to be held fixed.

Read position_patch_review/reachable.py. Its bmaps store preactivation
Jacobians before masking, as required; it includes all10 steps although only
the first8 inputs change. Zero preactivation with positive change forces
alpha0. SVD cutoffs, least-squares tolerance and finite witness checks remain
numerical qualifications. High leakage/small alpha and lost readout-energy
matching support only mixed-group local sensitivity. They cannot establish
isolated mediation, complete donor swap, or reject the mechanism.

## Residual tensor-fiber enrichment (T81)
Read tensor_enrichment/enrich.py. If residual amplitude at pivot is a!=0,
a*product_k(residual_fiber_k/a) matches the old residual on each queried fiber
in exact arithmetic. Product-weight contraction is exact for this represented
rank1 term. With a configuration indicator as the first factor, the same
statement holds locally while avoiding extrapolation across configurations.
Later corrections need not preserve earlier interpolation or decrease global
error. Tiny pivots with large normalized fibers can amplify off-fiber errors;
product-weighted pivot selection does not bound this amplification.

Retain an unqueried off-center separable spike with zero central fibers as a
support-discovery falsifier. Add nearzero pivot/large fiber-ratio controls.
The ledger counts prediction diagnostics honestly but remains a partial
algebra proxy: sorting/set/cache construction, residual arrays, argmax and
some indexing work are not a complete portable cost certificate. Distinguish
same-grid tensor error from quadrature-to-network-reference error; reduced
first error can remove accidental cancellation and worsen the second.

Additive followup: correction sum_k fiber_k-(d-1)*central_residual exactly
matches central fibers without dividing by the pivot. Its simplified weighted
integral assumes every noncategorical weight vector sums1; otherwise include
other-axis mass factors explicitly. Independent exhaustive Fraction sums with
nonuniform normalized weights, two categories and scales1,-3,1e-6 match the
production integral to2.12e-22. A central spike has exact fibers but estimated
mass-17/30 versus true+.08 and off-fiber correction-2. This is a mathematical
limitation of the extension, not an implementation sign bug. Scalar arithmetic
.001584CPU excluding imports, no neural inputs. See check_additive_review.py.

## Exact supported lune (new T56 phase)
In an orthonormal frame, coordinates z in[-1,1] and phi in[pi/3,pi/2] have
area element dz*dphi, area pi/3 and whole-sphere mass1/12. Conditional measure
is dz/2*dphi/(pi/6). Thus sqrt(2n+1)*P_n(z) times integer Fourier modes on this
finite phi interval is orthonormal and represents the supported root constant
exactly. Generic halfspace arcs must be intersected with the entire finite
window, retaining all wrapped components. Zero transverse normals and
identically zero forms require explicit ownership; pole ambiguity is null.
This removes root support projection error, not later projection error.

## Conditional rank/gate factorial diagnostic (T28)
For10 iid continuous normal inputs, conditional on specified largest/second
positions, q=Phi(second) has Beta(9,2) density. Given q, the largest CDF value
is uniform(q,1), and the eight labeled lower CDF values are independent
uniform(0,q), independently of the largest. Equal-probability strata require
uniform sampling in Beta-CDF coordinates, then inverse Beta, not uniform q.
For balanced180 cells, stratified variance is sum_c(s_c²/n_c)/180²; paired
comparisons use within-row differences. A moment-matched Gaussian can leave
the rank chamber: scoring it against the original chamber label is a closure
diagnostic, not Gaussian second-argmax accuracy. Distribution versus frozen/
actual gates is a factorial approximation comparison, not causal attribution.

Independent T28 numerical check:8192 iid normal inputs at fixed seed28103,
DEV .02 copy0, assigned to actual observed top-two positions and Beta-CDF
halves. Only model-free conditional means are shared with the author. Direct
fixed-mask recurrence replaces their Jacobian maps, and unconditional sampling
replaces their conditional rank sampler. Actual accuracy .6207275391; frozen
accuracy .7302246094, paired bias .1094970703 with SE .0041663786. Recomputing
last2 gates yields .6752929688, bias .0545654297 with SE .0043136990. Both
corroborate the author's large high-noise biases; this is a finite closure
failure with a partially helpful rescue, not a rejection of all gate-aware
or conditional-distribution methods. No efficiency claim. The diagnostic uses
81920 normal variates and262144 hidden-step vectors in.534333CPU after imports.
Source/checkpoint hashes are asserted. Raw script/results are
check_rank_gate_review.py and rank_gate_review_results.json.

T66 stable helper followup: the owner reproduced both proposed numerical
falsifiers, including 100% loss of some large-t thin arc widths. Read
stable.cpp/endpoint_controls.py. Root-side ordering eliminates rounded
midpoints, while same-sign normalized atan2(delta,1+ab), reflected negative
intervals and semi-infinite complementary atan2 are algebraically correct.
The180-digit exact-float-ratio reference has adequate precision for1e100
adjacent endpoints. This improves two concrete failure modes without
certifying coefficient/root topology. Source angle_width recomputes acos(-1)/2;
the owner was asked to hoist or explicitly count it rather than assume compiler
folding. atan_calls includes both atan/atan2 and needs that label qualification.

T20 followup: owner b559882 chooses explicit finite-endpoint rejection and
adds three nonfinite controls, rather than extending the callback domain.
That resolves the reported inf/inf API edge within the intended finite tube
windows. Full-line normalization remains an analytic/external control.

T82 proposed network-aware slice: for bounded z/t domain, every fixed
ReLU/readout/target-sign branch is an affine polygon. Its vertical slice
bounds are affine in z, with changes only at projected vertex coordinates.
Therefore Gaussian t-mass is a sum of normal-CDF affine-bound differences,
smooth on each open vertex slab. Transforming z through q=Phi(z) introduces
no interior kinks. Include all window/exclusion/sign constraints and explicit
zero-form ownership. This is exact-arithmetic topology only; floating thin
polygons, complete enumeration, endpoint regularity and cost need controls.
Implementation remains the owner's responsibility after a pushed claim.

T56 exact-lune source review: lune_cycle/space.py uses Fourier difference j-i,
consistent with E[conj(b_i)b_j 1cut]. Three wrapped angular intervals suffice
for delta in[-pi,pi]; finite-window clipping preserves every component.
Nonzero-normal complement I-M(-v) is valid up to a null boundary. After
canonicalizing v_z>=0, projected wall knots -h/hypot(h,v_z) and full/empty
knots +/-r are correct for normalized v. Sine-map weights include dz/2.
Rounded poles have the proper limiting full/empty arcs. Normalization and
near-degenerate knot arithmetic remain floating, not certified support
geometry. Root support is exact in the mathematical construction; later
finite-basis indicator projection is still approximate.

T66 DD clip followup: compared dd_clip.cpp to exact_ladder/cones.py. The
+/-1e-11 tight rule, all-tight ownership, signed/tight/new ray ordering,
two-word common-bit adjacency and exactly-two-containing-rays test match.
Capacity k+floor(k²/4) covers every positive/negative candidate pair. Charges
match inherited formulas, not a complete portable cost certificate. Scalar
C++ summation can differ from NumPy near the tolerance, so finite corpus
agreement is not universal bitwise topology equivalence. The original
module's phrase 'exact tight sets' must not be promoted to a rigorous claim.

T45 independent association arithmetic: grouped the90 frozen joined rows
into18 five-seed(H,T) groups, using direct pair-comparison ranks and exact
Fraction covariance rather than scipy rankdata. Every feature/gain group
has five distinct values. Common centered-rank norm10 makes pooled
within-group correlation exactly the mean of per-group Spearman values.
Gate uncertainty: Sobol -7/90, iid -1/60. Boundary alignment: Sobol -1/90,
iid7/60. All four equal saved values exactly. This verifies point
associations only, not bootstrap coverage, historic gain/truth uncertainty,
or heldout predictive value. Signed rank normals point inward: max-second
for the larger competitor and second-lower for each lower competitor;
output-gradient alignment must use matching correct-side orientation.

T66 native lifecycle review: dgesvd packs selected ray rows column-major,
sets lda=max(1,m), JOBU=JOBVT=N and dummy leading dimensions1. Workspace
max(5min,3min+max) is the documented minimum for these flags. The interface
assumes LP64 LAPACK int32 ABI; source/export wrapper copies returned arrays
before freeing the C++ handle in a finally block. Singular values use the
same strict>1e-9 rank threshold. Native stable_sort on equal |Wi| may differ
from NumPy default argsort tie ordering; test/qualify duplicate magnitudes
before claiming ordered topology identity. No generalized dimension claim.

The owner retained a newly added hidden-map absolute-error assertion failure
3.64e-12 while ordered rays/bits/triangles passed the frozen1e-12 ray check.
These are distinct tests. A separately frozen map accumulation bound is a
legitimate alternative, not a silent relaxation of ray tolerance. Conditional
on identical gate histories, propagate each floating path's map error using
|Wh|E plus a conservative dot-product gamma bound, then sum the two path
bounds. Bound arithmetic/BLAS/underflow need explicit qualifications. Passing
that bound cannot certify unseen near-threshold topology.
