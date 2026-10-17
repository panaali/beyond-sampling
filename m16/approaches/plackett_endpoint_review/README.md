# T142: locate the Plackett endpoint sensitivity

## Question
Does the negative T137 fixture require an unresolved extremely narrow endpoint
layer, and can a cheap analytic endpoint bound constrain that explanation?

## Method
TASKS-only claim889e2bb precedes implementation. Use only fixture672 and its
pinned T137 wide12/power2 metadata. No new quadrature rules or accuracy run.
For the straight path C(t)=I+t(C−I), the derivative is a sum over six pairs:
Cij asin(partial_kl.ij(t))/(4 pi² sqrt(1−t² Cij²)). Consequently its integral
on [a,1] has absolute value at most
B(a)=sum_ij [asin(abs(Cij))−asin(a abs(Cij))]/(8 pi).
The bound holds for every positive definite C, including limits by continuity.
It bounds the true contribution, not the quadrature error.

Freeze a at the original largest t node, and at1−2^-16,1−2^-24,1−2^-32.
Evaluate B, the derivative at a and actual endpoint C, and eigenvalues of C.
Also inspect the six partial correlations at all original12 nodes and these
four cuts. Repeat exact binary-ray correlation calculations at80/100 digits.
No adaptive probes, eigenvalue-based selection, additional rule, or tuning.

The Gauss rule approximates the whole interval; its final-node gap is NOT
an omitted interval. A precise localization test is available only for the
last original node a: define g(t)=f(t) below a, f(a) above. All source nodes
see unchanged values, so Q(g)=Q(f). The continuous integral changes by at
most D(a)=B(a)+(1−a)abs(f(a)). If D is smaller than the known absolute rule
error, a constant extension past the last node cannot remove all the error.
If D is larger, this localization test is inconclusive. The other frozen cuts
only bound how much true integral can lie in progressively narrower tails;
no claim that their quadrature contributions are absent or unchanged.

Controls before fixture: identity and two independent correlated pairs. The
latter's exact orthant law along the entire path is the product of two
bivariate arcsine formulas. Check its analytic derivative independently by
differentiating that product and its exact tail against B at every cut.

## Alternatives and falsifiers
1. Last-node D exceeds observed error: this test cannot localize the source.
2. D is smaller: substantial quadrature error survives constant continuation.
3. Small eigenvalue alone does not establish a consequential endpoint layer;
   bounded tail magnitude must be reported alongside it.
4. Invalid partial correlations or precision instability block the diagnostic;
   preserve them, never clip or silently change covariance.

## Status
Protocol frozen; implementation and controls next. T55 reviewed the tail
bound and cautioned against interpreting GL's final node as interval omission.

## Results
Pending. This is source-error localization on one existing posthoc fixture.

## Compute accounting
One thread,2GB, review lock, <=20CPU per phase. Count every matrix inverse,
eigenproblem, partial correlation and arcsine. Include all setup and controls.
No input sampling, model fitting, geometry rerun or competitive-cost claim.

## Claims
[inferred] The tail bound holds in real arithmetic by the triangle inequality.
[speculative] Endpoint conditioning might explain the finite12-point defect.
No directed floating-point certificate is claimed from MP stability alone.

## Alternatives not ruled out
Interior branch shape, cancellation, different transforms, positive integral
representations, and controlled interior integration. T140 independently owns
uniform degree doubling; T141 owns analytic positive coordinate reduction.

## Failure modes
Mistaking a bound for observed error, treating GL nodes as interval endpoints,
changing covariance/branch/source, ignoring rounding, or extrapolating one
fixture to a corpus. All initial failures remain in version control.
