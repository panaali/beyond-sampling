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
Source/theory audit complete; first bounded control cycle frozen, not run.
At most18CPU on review lane or100CPU on primary lock, single thread,<2GB;
no model sweep or large Monte Carlo. Next trained T3 phase requires its own
frozen moment-integration protocol and cost after these controls.

## Results
Pending; no source performance claims are accepted as independent validation.

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

## Failure modes
Singular logit covariance, positive-mass ties, variance clamps, finite expansion
nonpositivity, quadrature resolution and understated setup proxies. Dense
three-output controls are not evidence of scalable H16T10 performance.
