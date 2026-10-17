# Independent T56 and T30 primitive review

## Supported half-order harmonic basis (T56)

For one rank ordering write a=emax-etarget and b=etarget-emin. The two half
cells are (a·x>=0,(b-a)·x>=0) and (b·x>=0,(a-b)·x>=0). Each implies both order
inequalities and selects which gap is smaller. Their intersection is a
measure-zero plane. All six orders therefore give twelve disjoint half-lunes
up to null boundaries. With the first frame axis along the near wall and the
second along the orthogonal component of far-near, the latter normal is
proportional to(-sqrt3,1,0). The cross-product third axis makes a proper frame.

On its positive hemisphere let z∈[0,1] be the first coordinate, phi the azimuth.
Conditional uniform measure is dz dphi/(2pi), so
sqrt(2n+1)P_n(2z-1) exp(i m phi) is orthonormal. Whole-sphere mass is half the
constant coefficient. One half-lune has mass1/12. The code's root mass and
phase conventions follow from this measure, not from fitted normalization.

For normalized cut normal(az,r cos(delta),r sin(delta)), az>=0, the rejected
arc occurs only z<r, centered at delta+pi, with half-width
acos(az*z/(r*sqrt(1-z²))). Its Fourier integral at k=mj-mi is
exp(i*k*(delta+pi))*sin(k*halfwidth)/(pi*k), with k0=halfwidth/pi.
The sine-over-k factor is even in k; using blocks[abs(k)] and signed complex
phase is correct. The substitution z=r sin(psi) has Jacobian r cos(psi).
Negative az uses complementation, and az0 uses the half-circle analytic
matrix. Frame-transformed gate/readout normals use frame.T, as required.

Independent falsifier `check_harmonic_phase.py` evaluates four nonzero complex
off-diagonal entries by direct arc-coordinate GL32 plus adaptive direct-z
quadrature, without the bad-arc phase formula. The two mode pairs(−1,2) and
(1,−1), with radial degrees(1,3) and(2,4), are tested under both signs of
normal(.3,.4,.5). Maximum difference1.401e-15,102144 arc integrand evaluations,
.045100CPU, no trained-model evaluation. This complements the author's prior
independent m0 entry; results are in harmonic_phase_results.json.

Finite projections need not remain pointwise positive or exactly supported
on every added cone. Exact geometry classifications combined with projected
coefficients are an approximate hybrid, not an exact probability calculation.
Gate branch complementation preserves total coefficient mass algebraically,
but does not bound local projection error or justify trusting negative masses.
The reported nine-case RMSE improvement versus the corrected isotropic method
is a scoped measured benefit. CPU-adjusted error product is not a certified
FLOP comparison and does not establish the challenge's10x condition. Larger
normal/tangential spaces, adaptive frames and certified truncation are not
ruled out by any tested finite-space limitation.

## Exact-line and legacy tolerance controls (T30)

Read rational_line.py and check.py in walker_tolerance_review. Induction over
steps preserves exact affine state pieces: all gate roots are split with
Fraction arithmetic, masks are selected at strict interval midpoints, then
all readout intersections and the target switch0 are split. First-index argmax
is explicitly max(value,-index). Endpoint tie ownership is irrelevant to
Gaussian mass unless a pair is identically tied on an interval, which the
first-index rule handles. Piece caps raise instead of returning an incomplete
number. Final70-digit Gaussian CDF evaluation is a numerical reference, not
an outward enclosure; exact roots refer to actual binary input coefficients.

The reset fixture has h_final=1 and a single positive class1 logit, however
small. It predicts1 everywhere. With negative target1 and positive target0,
wrong mass is Phi(.25)-.5=.09870632568292373. An approximate tie rule cannot
replace a strictly positive gap by an exact tie without changing the problem.
For envelope0→9→8 at r1<r2<0, negative target9 and positive target8, only the
negative middle interval is correct. Wrong mass is
Phi(0)-Phi(-.25)-[Phi(r2)-Phi(r1)], with actual rational roots.

Read-only legacy source inspection finds the implicated mechanisms directly:
sci_tao/core.py treats zero-slope gaps within1e-12*(1+scale) as ties;
sci_feynman/core.py, wtool.py and boundary/zones.py select readout winners at
val+1e-7*slope. A middle envelope segment narrower than that lookahead can be
skipped. These explain the deterministic fixtures independently of a broad
Monte Carlo comparison. This review did not rerun every legacy implementation
or measure error frequency on archived trained models. The scoped defect is
unconditional exactness of those implementations, not face integration or
all possible walkers. Complete affine envelopes, exact/validated event
ordering, and explicit remainders remain possible remedies.

## T56 exact-lune final bounded handoff
Read lune_cycle/README.md, ensemble.py, cost_cycle.py, cost_floor.py and saved
ensemble rows. H2 cases are the prior base plus8 absolute .05 perturbations
at seed47001; H4 cases are exactly the first8 prior ARC .05 perturbations at
seed0. Each estimator constructs its own state, Q64 is frozen production,
Q128 is a separately charged refinement diagnostic. No copied reference
accuracy is used to select a degree/order or omit a case.

Independent arithmetic from individual saved rows confirms all17 absolute
errors improve. Mean(error²*CPU) new/old ratios are1.09412439 for9 H2 cases
and.03623102 for8 H4 cases. These are ratios of mean products, not products
of separately averaged errors and costs. Maximum Q64/Q128 changes are
3.5343e-10 and1.2530e-8. Thus root support is a useful finite representation
rescue for H4 compared with the older hemisphere code; H2 loses under this
CPU proxy. Neither is a sampling superiority or portable-FLOP result.

The dense-operation floor sums real Gram, complex Gram and complex matvec
multiply-add events, charging only one scalar multiplication per event. This
is deliberately below ordinary dense arithmetic cost, not a lower bound on
all possible algorithms for the represented matrices. Complex storage and
real degrees of freedom must not be equated with identical cost. The stated
network-forward comparison omits input generation; its above-one ratios
therefore do not establish universal sampling impossibility. Named opaque
kernels, geometry, full branch enumeration and setup remain separate.

Primitive/source proof was reviewed previously, including opposite-order
entry integration by the owner. Shared cone geometry is not an independent
end-to-end solver. Basis truncation at internal gates/readout remains an
uncontrolled approximation; quadrature stability does not bound that bias.
Real-only/localized bases, internal-gate support and certified allocation
remain open finite alternatives. The T3 lune chart does not automatically
extend to higher input dimension. No broad spectral-family rejection or
challenge-solved claim is supported.

Saved-result recomputation costs.001467CPU; no new engine/model-input calls.
See check_lune_summary_review.py and lune_summary_review_results.json.
