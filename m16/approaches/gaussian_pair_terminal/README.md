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
