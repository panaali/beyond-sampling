# T117: fixed-state Gaussian two-input terminal operator

## Question
Can exact final-two-input boundary representation replace one interpolated
terminal stage at fixed absolute hidden state, with acceptable local cost?

## Method
For H2 and an already inserted target j<=T-3, fix nonnegative hidden h and
threshold u. Enumerate first-input ReLU intervals and second-step masks;
retain original affine halfspaces in fresh inputs x,y, including readout
strictness against earlier indices. First-index ties have one owner; a
zero preactivation has one inactive branch. Verticalize each convex polygon
in the input box[-6,6]^2 without reconstructing lines from rounded vertices.
For rank sides a,b integrate phi(x)[Phi(upper(x))-Phi(lower(x))] positively
on complete affine-envelope pieces, clipping at x/y=u and adding line/u
crossings. Numerical one-dimensional quadrature has explicit diagnostics;
this is exact boundary representation, not certified floating integration.

Let Pab be unnormalized original Gaussian mass of correctness in sides
(a,b), with0 below and1 above u. With Z=Phi(6)-Phi(-6),
q=(Phi(u)-Phi(-6))/Z, return P00/(Zq)^2 when zero exceedances remain and
(P01+P10)/(2Z^2q(1-q)) when one remains. T55 independently verifies this
law and the fact that prior-state correlation does not invalidate the
Markov conditioning on(h,u,remaining count).

## Prior-work separation
T55 target_tail and T59 BatchTerminal integrate a chi-distributed prefix
radius, not a fixed absolute h. Reset_wedge conditions a discarded reset
prefix and has a different rank law. real_arc_tail has normalized spherical
coordinates. None is silently reused as the fixed-state Gaussian measure.
The original-halfspace verticalization idea is reused from T67; geometry is
locally implemented to remove its specialized y-versus-target coordinate.
No shared source or other owner's folder is changed.

## Frozen alternatives and falsifiers
1. Stage3 interpolation is costly in accuracy: exact two-input boundary
   values differ materially on selected physically reachable states.
2. The primitive is algebraically wrong: constant/duplicate outputs, Wh0,
   all-label partition and swapped-axis positive integrals falsify it.
3. Geometry cost erases the benefit: count construction, every mask/line/
   intersection/query/CDF/callback, full cold CPU; no free advice caching.
4. Floating thin regions/ties spoil positivity: include exact-zero gates,
   positive rescaling1e-16 and one narrow readout fixture; preserve failures.

Phase1 before trained data: H2,T4 toy controls with h=(0,0),(.3,.7),
u=-1,0,1, all four labels for partition validation (production insertion
contract remains j<=T-3). Constant, duplicate and Wh0 cases; independent
swapped-axis integration and direct scalar terminal integration on controls.
Tolerance1e-10 absolute conditional probability, no clipping/renormalizing.
One local trained primitive panel only after independent review: fixed H2T4
base, h from exact two-step inputs(-1,1),(1,-1),(1,1), thresholds-1,0,1,
labels0,1. No full transport/outer/noise sweep under this claim.

## Status
Claim ee320e9 pushed before implementation; protocol frozen now.

## Results
Pending. T67 component cancellation and failed outer-stability gates remain.

## Compute accounting
One thread,2GB, review lane<=20CPU controls, primary<=110CPU bounded phases.
Count original halfspace construction and all query work, SciPy quadrature
callbacks/returned errors, CDFs and full CPU. General special-function and
adaptive-library work remains opaque; no portable FLOP or sampling claim.
No random inputs, parameter fitting, learned advice or full challenge claim.

## Claims
[inferred] For fixed h and u, the terminal event is a finite affine-polygon
Gaussian integral and the rank normalizers above are exact in real arithmetic.
[speculative] Removing stage3 interpolation is useful after full cost/accuracy.

## Alternatives not ruled out
Earlier hidden-grid approximation, alternate axis ordering, analytic Gaussian
polygon formulas, positive bounded fixed rules, batching and native kernels.

## Failure modes
Affine offsets accidentally dropped, duplicate zero gates, readout ties,
near-coincident lines, insufficient thin-region error control, quadrature
error estimates mistaken for proofs, and wrong previous-prefix measure.

## Toy primitive checkpoint
`controls.py` completes36 model/state/threshold panels (all4 readout labels,
both rank needs) in .593CPU. All-label partition and independently swapped
integration-axis discrepancies are<=4.45e-16. A separate direct scalar
terminal recursion agrees on six fixed ordinary queries within1.06e-12;
positive hidden/input rescaling1e-16 agrees within1.12e-16. Constant,
duplicate, exact-zero gates, Wh0 and a1e-9 thin-readout fixture all pass.
No quadrature warning is ignored: a nonconverged QUADPACK return raises.
Quad errors are diagnostics, not rigorous floating bounds. The JSON stores
all values; the detailed ledger covers the36 main panels, while total CPU
also includes independent direct/scale checks whose event counters are not
combined into that ledger. This is not a complete arithmetic inventory.
Independent source review requested before the frozen trained primitive panel.

## Independent review correction
T55 source review passes chronology, original-halfspace geometry, strict
readout ties, swapped-axis transpose and the two rank normalizers in the
frozen limit6 regime. It independently found an API domain defect outside
that regime: limit40,target39 underflows the upper-side mass and returned
[1,NaN] for constant label0. `initial_underflow_failure.json` preserves the
reproduced failure. The local repair explicitly rejects nonpositive/nonfinite
rank normalizers, nonfinite affine maps, quadrature values/errors and final
probabilities/bounds. `domain_check.py` verifies underflow and affine overflow
rejection. Repeating the same36 controls after this material repair preserves
all numerical discrepancies (.449CPU); no favorable timing inference.
The trained crossed(h,u,label) panel is an extension-operator check: h is
reachable for some prefix but not necessarily jointly with each crossed u.
No actual rank-weighted full-accuracy inference will be made from that panel.

## Frozen trained primitive panel
`pilot.py` completes18 fixed H2T4 queries in .05763CPU including saved output
writes; six cold objects total .00570CPU geometry,36 primary/swapped queries
.03593CPU including swapped geometry construction. Both integration axes
agree within5.56e-16 with all probability guards satisfied. Every value,
conditional rank mass, numerical error estimate, joint-prefix consistency
flag and object counter is stored in pilot.json. These are extension-operator
controls, not an accuracy estimate or wall-time scaling claim. Only6 of the
18 crossed(h,u,j) cases use the actual target value of the defining prefix.
No full outer/spatial estimator has been run or declared competitive.

## Same-probe interpolation discriminator, frozen before measurement
TASKS scope update1d0362c published. Evaluate the original h9 stage2-to-stage3
continuation on the SAME18 saved(h,u,j) probes; use exactly the previous
squared-coordinate stage3 grid and original scalar Gaussian interpolation
operator. For each label, terminal grid values integrate the last fresh input
exactly. Apply the first fresh-input below/above matrices; with two inputs
remaining the resulting values are low[:,0] and
(low[:,1]+high[:,0])/2. No fitted combination or final-accuracy extrapolation.

Check the input states against the stage2 box and both scalar support
endpoints after the next transition against the stage3 box. Reject any
out-of-domain point explicitly, without clamping or extrapolation. Constant
and duplicate outputs and positive hidden/input rescaling must pass before
the trained comparison. Frozen metrics: max/mean absolute discrepancy and
per-label errors for each rank need, plus the six target-consistent probes
separately. Every probe remains visible; no selection on favorable error.
The reference is the already measured pair primitive, reused only for this
diagnostic, with its source hashes and prior CPU retained. New baseline
setup/operator/query CPU is recorded separately, not a fresh paired timing
experiment. No new grid/threshold/model/noise nodes or full transport run.
