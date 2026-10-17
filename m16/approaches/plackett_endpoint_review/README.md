# T142: locate the Plackett endpoint sensitivity

## Question
Does the negative T137 fixture require an unresolved extremely narrow endpoint
layer, and can a cheap analytic endpoint bound constrain that explanation?

## Method
TASKS-only claim889e2bb precedes implementation. Use only fixture672 and its
pinned T137 wide12/power2 metadata. No new quadrature rules or accuracy run.
For the straight path C(t)=I+t(C−I), subtract the analytically integrated
base arcsine term. The remaining Plackett integrand f is a sum over six pairs:
Cij asin(partial_kl.ij(t))/(4 pi² sqrt(1−t² Cij²)). Consequently its integral
on [a,1] has absolute value at most
B(a)=sum_ij [asin(abs(Cij))−asin(a abs(Cij))]/(8 pi).
The bound holds for every positive definite C, including limits by continuity.
It bounds the true contribution, not the quadrature error.

Freeze a at the original largest t node, and at1−2^-16,1−2^-24,1−2^-32.
Evaluate B, the residual integrand at a and actual endpoint C, and eigenvalues of C.
Also inspect the six partial correlations at all original12 nodes and these
four cuts. Repeat exact binary-ray correlation calculations at80/100 digits.
No adaptive probes, eigenvalue-based selection, additional rule, or tuning.

The Gauss rule approximates the whole interval; its final-node gap is NOT
an omitted interval. A precise localization test is available only for the
last original node a: define g(t)=f(t) below a, f(a) above. All source nodes
see unchanged values, so Q(g)=Q(f). The continuous integral changes by at
most D(a)=B(a)+(1−a)abs(f(a)). If D is smaller than the known absolute rule
error for exact-ray C with the original rounded rule, a constant extension past the last node cannot remove all the error.
If D is larger, this localization test is inconclusive. The other frozen cuts
only bound how much true integral can lie in progressively narrower tails;
no claim that their quadrature contributions are absent or unchanged.

Controls before fixture: identity and two independent correlated pairs. The
latter's exact orthant law along the entire path is the product of two
bivariate arcsine formulas. Check its analytic derivative independently by
differentiating that product, subtracting the base-term derivative, and
checking the residual exact tail against B at every cut.

## Alternatives and falsifiers
1. Last-node D exceeds observed error: this test cannot localize the source.
2. D is smaller: substantial quadrature error survives constant continuation.
3. Small eigenvalue alone does not establish a consequential endpoint layer;
   bounded tail magnitude must be reported alongside it.
4. Invalid partial correlations or precision instability block the diagnostic;
   preserve them, never clip or silently change covariance.

## Status
Bounded diagnostic complete; independent result review pending. T55 reviewed
the bound and corrected-source logic before the fixture.

## Results
Exact-ray C has eigenvalues approximately4.02796e-8, .0996927, .445658,
3.45465 (condition number8.57668e7), despite maximum pair correlation.94185.
The residual integrand rises from.180752 at the last source node to.204920
at t=1. Several conditional correlations become almost−1; the raw maximum
pair threshold does not describe this multivariate conditioning.

| Tail width | Absolute true residual-tail bound B | B / known rule error |
| --- | ---: | ---: |
| Last-node gap8.50026e-5 | 3.48783409e-5 | 72.6039 |
| 2^-16 | 6.26198815e-6 | 13.0352 |
| 2^-24 | 2.44617302e-8 | .0509204 |
| 2^-32 | 9.55536463e-11 | .000198908 |

The constant-tail replacement bound at the last node is5.02427323e-5,
larger than the4.80391856e-7 known error. Therefore the frozen localization
test is **inconclusive**, not proof of endpoint cause or irrelevance. The
very narrow final2^-24 interval contributes at most5.1% of the known error
in absolute true residual integral; this is not its quadrature-error share.

Both80/100-digit calculations agree on recorded spectrum/tail quantities
within1.60e-77. Floating displays are not directed certificates. This is
one fixed posthoc fixture, with no new rule or whole-estimator result.

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
uniform degree doubling; T143 owns analytic positive coordinate reduction.

## Failure modes
Mistaking a bound for observed error, treating GL nodes as interval endpoints,
changing covariance/branch/source, ignoring rounding, or extrapolating one
fixture to a corpus. All initial failures remain in version control.

## Preserved control failure
Initial controls compared the residual integrand against the full orthant
derivative, omitting the derivative of the analytically integrated base
asin term. The block control failed before any trained fixture run; source
and failure are preserved in f2ec592. The correction subtracts the explicit
base derivative and base tail from the independent block-product formula.
The residual integrand, original task and tail-bound formula are unchanged;
wording now distinguishes the full derivative from the residual integrand.

Reproduction (one thread,2GB cap, review lock):
```sh
python m16/approaches/plackett_endpoint_review/endpoint.py controls
python m16/approaches/plackett_endpoint_review/endpoint.py fixture
```
Controls used.010301CPU and8 inversions. The fixture used.062767CPU,
two covariance inversions, two symmetric eigenproblems,34 transformed
inversions/positive-definiteness checks,204 partial correlations and300
arcsines. The initial failed control is retained; it used less than.14wall
seconds including interpreter startup, but CPU was not separately captured.
