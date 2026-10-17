# T40 exact-law argmax endpoint review

## Question
How much error comes from the final cumulant-to-argmax approximation alone,
when conditional target/logit moments are supplied independently of recurrent
cumulant propagation? Claim61dd4ea precedes this work. Claude T80 owns upstream
propagation and structured-basis rescues; this folder implements neither.

## Source and independence
Read public ARC mlp_cumulant_propagation branch experiment/max-endpoint,
commit357c45578a40cb1807cd09bdc28e17d3d0277d0e (2026-07-16), on2026-10-17.
The source explicitly says ties are ignored, and applies a finite Edgeworth
operator expansion around independent marginal Gaussians. It differentiates
expected-max approximations with cumulants fixed. Post-January2026 source
dependence is explicit; source_log.json supplies URLs/hashes for DRI web_log.
Clone is read-only at /workspace/work/t40_sources. No training or external
cluster instructions from upstream documentation are followed.

## Frozen first-cycle method
Use the unmodified upstream argmax_endpoint_estimate with full dense cumulants
orders1..4 (HTensor radial index0), three outputs, CPU float64. Its five nested
raw estimators E0,E1cov,E2cov,E2k3,E2full are all retained, as are its separately
labeled simplex projections. No favorable order/projection selection. Evaluate
Q64/128 and Q128/256 to distinguish quadrature sensitivity from endpoint bias.
Record upstream FLOP proxies, CPU including imports/setup, all clamp/status and
simplex diagnostics. Count model-free moment quadrature separately; upstream
counts omit or approximate some setup/reverse-mode costs and are not certified.

Freeze four exact-law controls before numerical evaluation:

1. Independent zero-mean unit Gaussians: exact winner vector(1/3,1/3,1/3),
   K3=K4=0. All nested corrections should vanish.
2. Correlated zero-mean Gaussian with covariance
   [[1,.4,-.2],[.4,.81,.1],[-.2,.1,.64]], K3=K4=0. For each i, compare the two
   Gaussian differences Yi-Yj,Yi-Yk; exact winner probability is
   1/4+asin(correlation)/(2*pi). This isolates finite covariance expansion
   error even without atoms or higher cumulants. Check positive definiteness.
3. Bias-free H1T3 reset RNN Wi=1,Wh=0,Wo=(-1,.25,1). Its final h=ReLU(X2).
   Under iid normal inputs and target J=2 (zero-based last index), X2 has
   median density6*Phi(z)*(1-Phi(z))*phi(z). Conditional J0 or J1, X2 is the
   equal mixture of maximum and minimum, density
   1.5*(Phi(z)^2+(1-Phi(z))^2)*phi(z). These are the two remaining controls.
   Their laws are symmetric: h has atom1/2 atzero and winner vector(.5,0,.5)
   by first-index argmax. Overall accuracy is( q0|J0 + q1|J1 + q2|J2 )/3=1/3.
   Compute raw h moments1..4 as the exact-law positive integrals over z>0,
   evaluate with scipy.quad epsabs1e-12 and record error estimates. Formula is
   exact in real arithmetic; numerical evaluation is not an enclosure.
   Convert to cumulants and outer-product Wo for full dense logit cumulants.

Strongest predeclared rescue: retain atom mass and its first-index winner
explicitly, and preserve the one-dimensional positive readout ray. Its winner
is exactly2 because coefficients are ordered and h>0; atom winner0. This
weight-derived support calculation gives the exact toy answer and does not
fit truths. It tests whether a failure is tied to discarding support/tie
information, without claiming a general trained-network estimator. For the
Gaussian control, full bivariate covariance gives the analytic reference;
do not mistake a finite covariance series for the full Gaussian law.

## Alternatives before runs
1. Numerical quadrature failure: compare64/128/256 and independent Gaussian
   analytic references; retain lack of convergence explicitly.
2. Upstream propagation failure: supply independent exact-law moments, not
   moments propagated through the RNN. Thus any remaining mismatch is local
   to the endpoint/numerical implementation under these moment laws.
3. Positive-mass ties break differentiability: E[max(Y+a)] need not have an
   ordinary gradient at0. For a zero-logit atom it includes.5*max(a), which is
   nondifferentiable; smoothing does not automatically implement first-index
   ownership. Use the explicit atom/ray rescue rather than arbitrary jitter.
4. Strong covariance or omitted higher cumulants may dominate: retain the
   non-atomic Gaussian control and all correction orders. An atom alone is
   not established as the unique source of any measured error.

## Status
Controls, trained H2T3 reference, and gate/rank mixture rescue completed.
Independent moment conversion review passed (T55 commit66efd3e): a distinct
set-partition Mobius calculation matches all six conditional cumulant tensors
through order4 to2.97e-16 scaled. This does not certify moment quadrature or
the upstream endpoint implementation. T55 mixture aggregation, corrected
Gaussian endpoint source and support review are recorded in commitb9e52d5.
At most18CPU on review lane or100CPU on primary lock, single thread,<2GB;
no model sweep or large Monte Carlo. Current T3 phases are complete; T4/T5
extensions require their own frozen protocol and are not claimed done.

## Results
Unmodified upstream code passes iid Gaussian symmetry to2.22e-16. On the
non-atomic correlated Gaussian, maximum winner-probability error decreases
from.0677255 (E0) to.0149612 (E1) to.00577319 (E2 covariance and higher).
Higher cumulants are exactly zero in this control; the residual is not caused
by upstream propagation or atom ties. Full covariance arcsine calculation
provides the predeclared exact-law rescue.

On the reset toy, all five methods return normalized nonnegative vectors and
zero variance clamps, yet the conditional winner vectors are inaccurate:
E2full maximum error is.493151 on J=last and.434832 on J=other. Aggregate
accuracy partly cancels these errors and must not be used to hide them:

| Method | Accuracy estimate | Bias from exact1/3 |
| --- | ---: | ---: |
| E0 product Gaussian | .3293510111 | -.0039823223 |
| E1 covariance | .3289701840 | -.0043631494 |
| E2 covariance | .3297501852 | -.0035831481 |
| E2 plus third cumulant | .3255805086 | -.0077528247 |
| E2 full fourth cumulant | .3401413371 | +.0068080038 |

Simplex projection changes none materially. Across128/256 nodes all winner
differences are<=4.56e-15; this is a sensitivity check, not a rigorous error
bound. The exact atom/ray support rescue returns1/3 algebraically. Results
isolate a failure of these finite smooth endpoint summaries on the specified
laws; they do not rule out gate/atom-conditioned endpoint methods. Source
labels already ignore ties, so the atom test is an applicability limit, not a
claim that upstream promised exact tie support.

Successful run CPU1.785556451 including imports/compilation, RSS237824KiB.
Eight endpoint calls (four laws,two node settings), proxy21,233,664 endpoint
operations across all nested estimators; eight scalar moment integrals.
Each128/256-node call reports3,538,944 proxy operations. Upstream proxy counts
special functions cheaply and omits/approximates setup/backward work; no
competitive-compute claim follows. `controls.json` preserves all raw/projected
vectors, status, moments, quadrature sensitivity and operation metadata.
Reproduce with PYTHONPATH=/workspace/work/t40_deps and the repository interpreter:
`python controls.py`, one thread, shared lock and memory limit.

### Trained H2T3 seed0
Gate/rank/orthant geometry produces58 triangles and label masses within3.55e-14
of1/3. Accuracy from separately clipped solid angles is.4101712617268285,
within4.36e-14 of the earlier independent facet reference. GL16/32 joint
moments differ by at most1.573e-11 under the frozen normalization; both pass
mass controls. These are converged numerical moments of the exact gate/rank
law, not formally enclosed exact tensors. Positive h=0 atom has joint masses
(.18514742,.17204551,.24324000) by target label, total.60043293.

| Endpoint | Raw mixture accuracy | h=0-atom-separated accuracy |
| --- | ---: | ---: |
| E0 | .4961440974 | .4420537096 |
| E1 covariance | .4702369821 | .4219622607 |
| E2 covariance | .4985776685 | .4325032914 |
| E2 plus third cumulant | .3934887773 | .3827291332 |
| E2 full fourth cumulant | 1.5268572951 | .5010903210 |

The atom rescue improves E0/E1/E2cov/full but worsens E2k3; no order is selected.
The raw full estimate outside[0,1] is preserved. Its simplex-projected accuracy
is.54218576 (raw mixture) and.46332326 (atom-separated), still biased.
Largest per-conditional raw vector violations include-6.2301 and4.5928.
Q128/256 changes are<=4.18e-14, separating these discrepancies from observed
endpoint quadrature sensitivity. No nonzero near-zero margin was treated as
zero; geometry found22 exact zero comparisons and no nonzero-hidden/all-zero-
readout cell. Floating clipping/topology remains qualified.

Trained phase1.866418015CPU including imports, RSS238544KiB; geometry70794
declared proxy operations,.02442CPU;74240 moment nodes (16/32),.09937CPU.
Six endpoint calls report21,233,664 upstream proxy operations, excluding the
qualified setup/moment work. No sampling or training. `trained_t3.py` and
`trained_t3.json` retain all tensors, per-target results, projections and costs.
T55 independently checked the conditional reset densities, half-mass atom/ray
and Gaussian arcsine formula (e6378b2). Two initial import attempts stopped
before numerical work for missing einops/joblib. Dependencies are pinned in
runtime_requirements.txt and installed into /workspace/work/t40_deps only;
the shared interpreter and upstream source files are unchanged. The virtualenv
has no pip, so installation used uv with a workspace cache. Those failed
imports did not have process-CPU measurements; their overhead is not silently
included in a successful-run-only CPU figure. Upstream joblib may initialize
its own ignored data cache in the source clone; no kprop fitting is invoked.

## Compute accounting
Charge imports, source diagram compilation, Legendre construction, every
quadrature/autograd contraction, moment integrals, controls and projections.
Endpoint's backward2x-forward assumption remains a declared proxy, not an
observed scalar-instruction count. Ground-truth formulas use no random inputs.

## Claims
[inferred] Exact moments separate endpoint approximation from propagation.
[inferred] Positive-mass ties require special treatment for the derivative
identity. [speculative] Finite covariance/cumulant expansion can be inaccurate
for strongly correlated scalar-input outputs even when supplied true moments.

## Alternatives not ruled out
Gate/atom-conditioned mixtures, support-aware readouts, full covariance
references, higher-order expansions, different endpoints, and a hybrid exact
last step. Toy failures cannot retire the endpoint family or its trained use.

## Frozen same-model gate/rank mixture rescue
The preceding atom split retains a mixture of different linear maps inside
each target law. Test one stronger distinct representation on the SAME H2T3
seed0: retain each final gate cell and complete input-order chamber as a
separate component. Before moment/endpoint evaluation, test whether one output
wins throughout the component using logit differences on its extreme rays.
For candidate j, require every difference to every k to be nonnegative on all
rays; if k<j require at least one positive ray margin, so the full-dimensional
relative interior has strict margin. Identically zero forms obey first-index.
Use no outcome/error values or boundary-location fit in this support test.
It is a real-arithmetic cone criterion evaluated on floating geometry, not an
interval certificate. Preserve failed/near-zero tests without a tolerance
that silently declares a winner. This automatically handles h=0 and some
positive rays, but may also recognize higher-rank constant-winner components.

For remaining components compute their own conditional logit moments via the
same orthant triangles and GL16/32 formulas. Preserve every positive-mass
component, with no fitted merge or mass truncation. Compare the summed moment
tensors to the previous artifact and per-component mass/moment sensitivity.
Then all five raw/projected upstream endpoints atQ128/256; aggregate each
component with its analytic numerical solid-angle mass. Report constant-win
mass, residual mass/count, complete geometry/moment/endpoint cost and all
orders. No estimate clipping or best-order selection. This is the strongest
tested support/conditioning rescue, not proof that all other mixtures fail.
One<=100CPU primary cycle,<2GB, no new models or Monte Carlo.

## Frozen trained T3 phase (after control results, before trained moments)
Use ONLY public H2T3seed0 checkpoint SHA256
47aa8e4b38f6433242b13d89666f01add73923b34123a6d56345e2c210cec1bd.
Read-only exact_ladder/cones.py snapshot SHA256
fc006840adc9513ac711b6c208afe96f4d10c429068d087dac8c674e8e6ee948
provides final gate cells and DD clipping/triangulation; its floating topology
is explicitly not the exact-rational T68 certificate. Avoid its outcome tie
tolerance: local correctness clipping treats exactly zero margins by first
index and records other near-zero margins rather than declaring them zero.

Split every final gate cell by all six input orders and all eight orthants,
retaining the second index. Pulling triangulate each piece with row rays R.
For b=(u,(1-u)v,(1-u)(1-v)), define x=bR and y=x/||x||. Integrate the joint
logit moment of orderk using positive tensor GL on[0,1]^2:

    |detR|*(1-u)/(4*pi*||x||^3) * (L y)^tensor_k * E[chi3^k],
    E[chi3^k]=2^(k/2)*Gamma((3+k)/2)/Gamma(3/2), k=0..4.

This is an exact real-arithmetic moment identity, evaluated numerically.
Sum joint moments then divide by each label mass; no Gaussian approximation
to the input/rank law, no empirical moment fitting, no random model inputs.
Freeze GL16/32 moment sensitivity, analytic cone solid-angle mass comparison,
total mass1 and label masses1/3 controls. Accept a numerical moment reference
only if mass discrepancy<=1e-9 and normalized maximum moment difference<=1e-8
(scale=max(1,maxabs(moment32)) per tensor). Otherwise one predeclared GL64
refinement within the bounded budget, preserving coarse failures. These are
diagnostics, not rigorous moment enclosures. No thin-cone error bound asserted.

Compute true accuracy independently of endpoint moments by clipping each rank
piece to its target's winning logit margins and summing analytic3D solid angles.
Compare the known H2T3 reference. Record zero-margin ties explicitly.
Evaluate all five upstream endpoint orders with Q128/256 for each of three
conditional target laws. Preserve atom-free and h=0-atom-separated variants:
subtract only the exact h=0 cone mass, renormalize nonzero-law raw moments,
then add the atom's label0 contribution to accuracy. This scoped rescue does
not remove every possible duplicate-readout/annihilated-map tie; record those
structures separately. No output-truth fitting or selected expansion order.

One<=100CPU primary-lock cycle, one thread,<2GB, persist before endpoint if
near deadline. Charge all geometry, clipping, moment nodes/tensor arithmetic,
endpoint calls and setup; no new training or model sweep. T55 independently
checked the moment Jacobian/radial normalization and the scoped atom caveat.

## Failure modes
Singular logit covariance, positive-mass ties, variance clamps, finite expansion
nonpositivity, quadrature resolution and understated setup proxies. Dense
three-output controls are not evidence of scalable H16T10 performance.

## Gate/rank mixture result
[measured] The stronger support-aware rescue materially improves this SAME
trained H2T3 case. Of38 components,25 have a constant winning output and
carry.9708213908840764 probability. Thirteen residual components carry
.029178609115885115. Constant-correct mass is.39021612885431867, so even
without an endpoint, nonnegativity gives the numerical geometric interval
[.39021612885431867,.41939473797020377]. This is not a floating-error enclosure.
Thus most of the accuracy information already comes from the partition and
ray support test; the endpoint fills in a small residual.

| Frozen method | Raw accuracy | Bias | Projected accuracy |
| --- | ---: | ---: | ---: |
| E0 product Gaussian | .4104490077860451 | +.0002777460592166 | .4104490077860451 |
| E1 covariance | .4090293915614711 | -.0011418701653574 | .4090293915614710 |
| E2 covariance | .4093165118685115 | -.0008547498583170 | .4093024847202162 |
| E2 plus third cumulant | .4105399632214605 | +.0003687014946320 | .4104922748478612 |
| E2 full fourth cumulant | .4108912372917807 | +.0007199755649522 | .4107769722937375 |

Reference accuracy is.4101712617268285. All five orders are retained; E0's
closest estimate is not a selected estimator or evidence higher cumulants
are harmful generally. Residual component raw probabilities still range from
-.0281107584 to1.0204872368. Maximum128/256 endpoint sensitivity is7.11e-15;
component GL16/32 normalized moment difference is5.50e-11. Summed component
joint moments match the unsplit artifact to1.39e-17 normalized and masses sum
to.9999999999999618. These checks localize finite endpoint approximation but
do not certify moment accuracy, topology, or successful extrapolation.

The run used2.382443047CPU including imports,239220KiB peakRSS,74,240 moment
nodes,62,777 geometry proxy operations and46,006,272 endpoint proxy operations
(13 calls, all five orders each). Moments of constant components were also
computed for reconciliation and are fully charged. Rule generation, tensor
arithmetic and upstream reverse-mode cost remain counted only partially by
these proxies; they are not a portable FLOP certificate. For scale, the
previous full T3 correctness-cone reference used70,794 geometry proxy
operations total. This rescue provides an accuracy improvement, not evidence
of a compute advantage. Full gate enumeration is a likely scaling obstacle.

Reproduce from this worktree, after source/dependency setup above:

```sh
ulimit -v 2500000
PYTHONPATH=/workspace/work/t40_deps PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/endpoint_cumulant_review/gate_mixture.py
```

[inferred] Pooling disconnected gate/rank components is an important failure
mode in the tested raw and atom-only summaries; retaining component support
is a plausible rescue. It does not show that cumulants alone caused the
improvement or that this exact partition is the best representation. T4/T5,
coarser weight-derived mixtures, full-covariance references and scalable
mixture construction remain untested. This finite T3 result does not retire
the endpoint family or establish a challenge solution.

## Frozen full-covariance Gaussian reference rescue
Next bounded phase retains each saved law's mean and complete covariance and
replaces only the endpoint with exact-law Gaussian winner probabilities.
For three logits, each winner is a bivariate Gaussian difference orthant.
Standardize its two differences to means a,b and correlation rho and compute
integral[-a,infinity] phi(z)*Phi((b+rho*z)/sqrt(1-rho^2)) dz by deterministic
scipy.quad, split at the conditional transition when inside the domain.
For rho=1 the limit is Phi(min(a,b)); for rho=-1 it is the normal mass between
-a and b (zero if bounds reverse). Zero difference variance uses its constant
mean and first-index tie; all equal deterministic logits must predict0.

Freeze tau=64*machine-epsilon. For 1-|rho|<=tau use the singular limit and
record its maximum Gaussian orthant perturbation bound
acos(|rho|)/(2*pi) when |rho|<=1 (from Plackett's derivative bounded by
1/(2*pi*sqrt(1-rho^2))). If floating rho exceeds1 by<=tau record the correction
separately; it has no certified covariance-error bound. Larger excursions or
negative difference variance fail explicitly, no generic PSD projection.
These conventions do not repair non-Gaussian moments or model ties.

Controls: centered iid/correlated Gaussians against analytic arcsines,
constant first-index ties, perfect positive/negative correlation with
nonzero means, output scaling1e-8/1e8, and sum of winner probabilities1.
All saved reset and trained pooled/atom-split/gate-rank mixture laws are
reported; no favorable subset or model fitting. Retain conditional vectors,
all rule error estimates/evaluation counts, singular corrections, CPU/setup
and prior moment/geometry costs. Compare epsabs1e-10 and1e-12, preserve
nonconvergence. No random samples, new models, upstream cumulant propagation
or shared source edits; <=18CPU review lock,2GB. This is a Gaussian moment
closure reference, not an exact endpoint for non-Gaussian logit laws.

## Full-covariance Gaussian rescue result
[measured] Keeping the full Gaussian covariance fixes both genuinely Gaussian
controls to roundoff (maximum error below1e-15), including the non-atomic
control where the finite covariance series had.00577 error. Constant ties
and perfectly correlated/anticorrelated affine controls pass. This is a
useful stronger reference, not evidence of universal endpoint failure.

For the reset toy its target-last vector is(.24767997,0,.75232003) and
other-target vector(.23694831,0,.76305169); true vectors remain(.5,0,.5).
Aggregate accuracy.3297561133592037 still benefits from cancellation. The
explicit atom/ray rescue remains exactly1/3.

| Trained H2T3 representation | Full Gaussian accuracy | Bias |
| --- | ---: | ---: |
| Pooled conditional target | .4942131153907876 | +.0840418536639591 |
| Zero-atom separated | .4320903576237874 | +.0219190958969589 |
| Gate/rank support mixture | .4090643340795152 | -.0011069276473133 |

Thus full covariance repairs the Gaussian control, but does not explain away
the trained pooled/atom-only mismatch. With geometric conditioning it is
competitive in accuracy with the tested finite-cumulant endpoints on this
single model. It does not beat every order, and no best-order choice is made.
The Gaussian mixture endpoint uses simple one-dimensional quadrature rather
than dense fourth-cumulant operators; the expensive partition/moment
construction remains, so no total compute advantage is demonstrated.

The entire diagnostic is.318441991CPU including imports,66612KiB peakRSS.
It evaluates27 laws at four rule/scale settings:192 quadrature subintervals,
31,968 integrand evaluations,120 explicit singular limits. Maximum numerical
rule change1.11e-16; scaling mean/covariance by1e-8 or1e8 changes a probability
by<=2.78e-16; maximum probability-sum residual2.22e-16. Summed QUADPACK error
estimates8.53e-10 are diagnostics, not outward bounds. Singular snapping for
correlations inside[-1,1] has analytic Gaussian probability perturbation
bound at most1.83e-8. Floating correlations exceed the unit interval by up to
1e-14 in46 evaluations; these recorded corrections have NO certified
covariance-error bound. Prior geometry, moments and source setup must be
charged on a cold run. No complete scalar operation/FPE count is asserted.

Reproduce with one thread and the review lock:

```sh
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/endpoint_cumulant_review/full_covariance.py
```

[measured] T55 independently establishes a further general endpoint limit:
commitdf85381 gives two atom-free, full-covariance logit laws with identical
cumulants through4 but winner vectors differing by2/15. This is not a proof
about the restricted Gaussian-input RNN family or a bound on useful
approximation there. It explains why retaining support/distributional
structure can supply information absent from a finite cumulant list.

## Independent large-mean falsifier and frozen correction
T55 source review verifies the orthant formula and singular bound, but identifies
a numerical tail risk: infinite-interval QUADPACK can miss Gaussian bulk when
its lower limit is very negative. Preserve the9 independent-Gaussian tests
(a,b) in{-100,0,100}^2 against Phi(a)Phi(b), and original source hash in
full_covariance_large_mean_failure.json before changing the implementation.
Also reject malformed zero-variance covariance rows with nonzero cross entry.

Frozen rescue: integrate only the intersection of[-a,infinity] with[-12,12],
split at -8,0,8 and the conditional transition when inside. Dropped probability
is nonnegative and at most2Phi(-12), recorded for every orthant call; this is
an exact analytic truncation bound, not a roundoff/quadrature enclosure. Return
zero when lower limit>=12 with the same tail bound. Repeat the exact same
saved law/control panel and additional9 extreme-mean product controls. Preserve
original trained results, compare new values rather than silently overwrite
old evidence; no new models or selected tolerances. <=18CPU review lock.

### Correction outcome and independent review
The original routine returns0 instead of.5/1 for(a,b)=(100,0)/(100,100),
with reported quadrature error0. T55 independently reproduces the failure
already at a=40. This is an implementation defect, not a Gaussian-law failure.
The original code and artifact remain for audit; use
full_covariance_bounded.py / full_covariance_bounded.json for the corrected
reference. The fixed9 product controls pass to1e-12, malformed zero-variance
cross covariance raises, and all original27 laws are rerun unchanged.

Corrected trained pooled accuracy.49421311539078655 differs from the original
by1.05e-15; atom-split and mixture outputs are unchanged at printed precision.
The original trained conclusions therefore survive this particular fix.
Corrected phase cost is.431458914CPU including imports. The added finite-domain
omission bound is2Phi(-12)~3.55e-33 per nonsingular orthant; it does not certify
the numerical quadrature or covariance corrections. T55 independently checks
the formula, singular perturbation bound, and finite bulk truncation logic.
Reproduce by replacing the filename in the preceding review-lock command
with full_covariance_bounded.py. No external endpoint code was changed.

Independent mixture review: T55 recomputes the saved weighted sums in Decimal,
obtaining fixed mass.97082139088407668568; all five raw aggregates match within
3e-17, and bounded full-covariance aggregate is.4090643340795151832. The
support criterion is applied on full-dimensional cone interiors; a nonnegative
linear form with one positive extreme-ray value is strict there, so first-index
ownership is correct in real arithmetic. Floating geometry remains uncertified.

## Frozen support-conditioned hidden Gaussian endpoint
T40 continuation reserved in pushed363f328. Same H2T3 seed0 and existing38
conditional gate/rank components,13 unresolved. Recover hidden mean/covariance
from saved logit moments via the fixed full-column-rank Wo left inverse; check
conditioning, reconstruction residuals, symmetry and PSD. Keep25 constant
support winners exactly as before, including atoms/rays and first-index ties.
No upstream moment propagation or new model/T4 enumeration. A deterministic
replay of the SAME13-cell T3 geometry is necessary because old JSON omitted
rays; charge it, verify component identities/masses against saved artifact,
and retain hidden support rays this time. No new moment cubature.

Two predeclared closures: Gaussian conditioned only on h>=0, and Gaussian
conditioned on the narrower hidden cone image of its gate/rank component.
Compute each target winner intersection as a 2D angular wedge bounded by at
most two linear inequalities, using corrected bounded full-covariance normal
integration for numerator/denominator. Conditioning changes the Gaussian
moments; this is not exact recovery of the original conditional law. Reject
negative/invalid covariance beyond documented floating tolerance, singular
residual laws, missing support, or denominator below1e-10. Report all cases,
no covariance repair or selecting the best closure after truth comparison.

Discriminators: independent centered isotropic wedge angles; positive scaling
of hidden/logit units; full winner partition sums; rule tolerances1e-10/1e-12.
Support omission could explain residual error, or moment closure within that
support could still be wrong. Compare both fixed variants with previous full
Gaussian and geometric truth without retuning. Opposite/fixed/ray supports
are explicit controls, not silently treated as smooth full-rank laws. Numerical
quadrature estimates and tail bounds propagate through the ratio using its
actual denominator; they are not global floating/topology certificates.
One<=20CPU cycle, primary lock,2GB. Save source/input hashes, all setup,
geometry, integrator calls and costs. T55 independent review requested.

### Preserved thin-wedge primitive failure and frozen rescue
The first new centered-isotropic control [.1,.1001] returns0 from the previous
bounded Gaussian primitive, versus exact angular mass1.5915494309187782e-5;
reported quadrature error is0. Other three wider controls pass. Finite bulk
clipping fixed distant means but does not resolve a thin near-anticorrelated
CDF layer adjacent to the integration endpoint. No trained result ran yet.
The failed process CPU was not captured; diagnostic process elapsed1.6s.

Freeze local transition-aware adapter with additional knots at standardized
conditional-CDF arguments -12,-8,-4,0,4,8,12. The exact location is
z=(t*sqrt(1-rho^2)-b)/rho, clipped only to the integration interval. Retain
original bounded primitive and failure. Test isotropic wedge widths .01,
.0001,.000001,.0000001 with exact angle mass, all previous distant-mean
independent-product controls, and unchanged27-law old endpoint panel before
inference. Near-singular limit error bound remains explicit at width1e-7.
No tuning of knots after outcomes. Preserve all trained closure results.

The initial trained continuation correctly rejected a covariance with minimum
eigenvalue-4.845e-15; no Gaussian aggregate was produced. Seven of13 unresolved
components appear rank-one, missed by the old floating all-ray sign test.
Predeclared singular-support handling now uses an exact binary-rational gate
map at each stored triangle interior, requires every triangle's map to agree,
and computes exact matrix rank and first-index winner on its nonnegative ray.
This classifies the fixed map algebraically; completeness of the underlying
floating cone partition remains uncertified. Genuine rank2 support rays use
that direct map, avoiding the ill-conditioned logit inverse for geometry.
The moment inverse remains checked. Covariance symmetrization only averages
roundoff asymmetry and is explicitly reported; no eigenvalue clipping occurs.
Two assertion-stopped diagnostic runs and one failed CUDA-load diagnostic had
unmeasured process CPU; the corrected diagnostic elapsed1.70s. Their work is
not silently charged as zero. Subsequent successful cycle records full CPU.

## Support-conditioned result
[measured] Seven previously unresolved components have exact rank-one fixed
maps and constant first-index winners, mass .0108444780071776. Six genuine
rank2 components remain, mass .0183341311087075. Both frozen variants preserve
the other25 fixed-support components. Exact map checks use binary-rational
weights and all stored triangle interiors; floating partition completeness
is still an assumption. No covariance eigenvalues were clipped.

| Endpoint variant on same law | Accuracy | Bias versus .4101712617268285 |
| --- | ---: | ---: |
| Previous full-covariance gate mixture | .4090643340795152 | -.0011069276473133 |
| Exact rank1, Gaussian conditioned on nonnegative orthant | .4091993183303292 | -.0009719433964993 |
| Exact rank1, Gaussian conditioned on component support cone | .4103475373067133 | +.0001762755798848 |

This is a positive scoped support-conditioning result on one trained model,
not an exact method or broader performance claim. Both prespecified variants
remain reported. The second result has bias .01763 percentage points; the
previous finite E0 Gaussian-mixture endpoint had .02777 points, also retained.
A normal conditioned to a wedge generally no longer has the input moments.
The remaining non-Gaussian density inside each support cone is not modeled.

[measured] Thin-wedge transition knots repair the .0001-width missed-bulk
failure. The1e-7-width control uses the preexisting singular-limit branch:
value0 versus true1.59154943e-8, computed-correlation replacement bound
1.59091326e-8. Their6.36e-12 difference demonstrates why the latter is not a
floating error certificate for the intended real Gaussian. Controls accept
this only under their explicit1e-10 numerical tolerance plus conditional
bound; do not state all narrow wedges are integrated exactly. The unchanged27
prior Gaussian laws differ at most3.22e-15 after transition-knot refinement;
all9 large-mean independent-product controls pass. Source failure remains.

The12 trained endpoint laws (six components, two supports), each evaluated at
two tolerances and two extra scales, use728 adaptive subintervals and21,462
integrand evaluations. No outside-unit correlation correction occurs there.
Minimum support denominator .77520448; largest winner-partition residual
2.22e-16, scale change1.53e-14, rule change2.22e-16. Largest conditional ratio
numerical-error estimate8.97e-13 excludes geometry, moment and roundoff errors.
Hidden left-inverse condition number45.8253 is recorded. Tiny negative support
rays and covariance asymmetry are explicitly logged before their numerical
roundoff treatment. They prevent a certification claim.

Successful support cycle1.278519439CPU including imports,232404KiB RSS;
geometry replay.023564105CPU/62777 inherited proxy operations. Prior moment
construction remains a cold dependency and no portable endpoint operation
counter or efficiency win is claimed. The27-law/large-mean replay costs
.298766578CPU. Assertion-stopped/setup failures above are extra unmeasured CPU,
not excluded evidence. Reproduce support_conditioned.py with the same Python,
PYTHONPATH=/workspace/work/t40_deps, single-thread primary lock; replay
check_transition.py under the review lock. Source and prior input hashes are
in saved outputs. Independent T55 source/aggregation review requested.

[inferred] This supports retaining support-aware closures as a live direction;
full gate-cone enumeration is currently more work than direct T3 accuracy and
cannot by itself solve the scaling challenge. Untried plausible variants:
coarser merged support mixtures with certified mass, conditional higher moments
inside each wedge, and downstream use of upstream propagated moments. Those
are separate scoped tasks; no family is retired by the residual error here.

### Frozen independent polar cross-check
Use a genuinely different coordinate formula on the SAME six saved rank2
Gaussian laws: integrate their polar angular density over each support/winner
wedge after analytically integrating radius. For s=b/sqrt(a), radial factor
is exp(-c/2)/a times[1+s*sqrt(2pi)*exp(s^2/2)*Phi(s)]. Evaluate s>=0 with
combined exponent exp(-(c-s^2)/2), moderate negative s using erfcx, and s<-5
with a positive32/64-node Laguerre integral after z=-s*sqrt(a)*r. No new laws,
no extra fitted threshold; check branch switch and s=-20,-5.01,-5,-1,0,1,20
against independently integrated radial functions. Check narrow isotropic
wedge1e-7 directly by angle, then all trained numerator/denominator ratios.
No quadrature certification from positivity alone. <=20CPU review lock.

### Independent-coordinate result
[measured] polar_check.py reproduces all12 trained wedge winner vectors within
3.50e-14 and the final aggregates within5.55e-17. Its directly integrated
isotropic angular controls resolve every width down to1e-7 within1e-15 absolute,
without a near-correlation singular branch. Seven independent radial quadrature
controls validate the erfcx/Laguerre switch to relative1e-10. No negative
Cauchy-Schwarz exponent residual occurs. The positive Laguerre branch is used
only by controls in this panel; all trained angles use the other two stable
branches, so no broad tail-branch accuracy claim follows.

This independent coordinate implementation uses10,416 angular evaluations,
.322787868CPU including setup, and preserves32/64-node checks and all raw
results. Agreement validates the Gaussian wedge calculation, not the Gaussian
closure or shared moment/geometry inputs. It also isolates the very-thin-wedge
failure as correlation parameterization/integration sensitivity, not zero true
mass. No endpoint-law improvement is selected from a new model sweep.

Independent T55 source/algebra and Decimal aggregation review23deb0d passes:
component-support accuracy .41034753730671331252 and orthant-support accuracy
.40919931833032920559. Exact rank1 logic and wedge orientation are accepted
conditional on numerical cell geometry. The review preserves moment-inversion,
tiny-ray clipping, thin-layer failures and denominator qualifications. The
scoped T40 phase is complete; no DRI claim confirmation or scaled performance
claim is made. The worker proceeds to a distinct Ready task while support-aware
upstream/mixture extensions remain live alternatives.
