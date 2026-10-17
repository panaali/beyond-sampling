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
Controls and one trained H2T3 reference phase completed; independent moment
identity review passed, trained numerical artifact review requested.
At most18CPU on review lane or100CPU on primary lock, single thread,<2GB;
no model sweep or large Monte Carlo. Next trained T3 phase requires its own
frozen moment-integration protocol and cost after these controls.

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
