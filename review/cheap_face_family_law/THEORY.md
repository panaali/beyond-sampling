# Ideal and numerical tie-height families

## 1. Ideal law and labels

Fix integerT>=3 and a tie-face kind k in{12,23}. Let Phi be the standard Gaussian CDF and e(u)=exp(-u²)/sqrt(pi). Define

    d12(u)=e(u)*Phi(u)^(T-2),
    d23(u)=e(u)*(1-Phi(u))*Phi(u)^(T-3),
    M_k=integral_R d_k(u)du,
    H_k(b)=integral_(-infinity,b] d_k(u)du / M_k.

Both densities are positive at every finiteu, bounded above by e, integrable, and have0<M_k<infinity. Let binsB0=(-infinity,0],B1=(0,1/4],B2=(1/4,1/2],B3=(1/2,infinity). Conditional-on-kind probabilities are

    p_{k,j}=H_k(b_j)-H_k(b_{j-1}),

with endpoints infinity interpreted byH(-infinity)=0,H(infinity)=1. All four are positive. These integral definitions are exact and model-independent; they are not evaluated numbers, closed-form elementary formulas or free setup. Let n12=T(T-1)/2,n23=n12(T-2),Z_k=n_k M_k. Assuming the ideal co-area decomposition and exact fibre integrand F_k from the inherited model, the error identity becomes

    error=sum_k Z_k sum_j p_{k,j} E[F_k(Y)|kind=k,u in B_j].

Here F_k is a Gaussian correctness integral along the appropriate fibre window, so0<=F_k<=1. The identity is conditioning, not a numerical accuracy theorem for the source walker. If kind is selected uniformly, the joint family probability is(1/2)p_{k,j}, and the unstratified identity is2E[Z_kind F]. Z is not this probability. The T=2 source has only kind12 and needs a separate one-kind mixture convention; it is excluded from the two-kind theorem.

For independent continuous V in(0,1), choose q=H_k(a)+(H_k(b)-H_k(a))*V and u=H_k^{-1}(q). This gives exactly the conditional height in(a,b]. Keep the original independent configuration and other uniform coordinates, then the ideal normal inverse maps conditional on u; the joint face law is the conditioned one by disintegration. Boundaries are null under these continuous laws. No rejection or network-derived label is needed. This is an existence/construction statement, not a claim that exact H/inverse evaluation is cheap or available in the program.

## 2. Fixed-grid real-arithmetic law

Consider fixed finite grid values x0<...<xN and fixed finite CDF values0=c0<=...<=cN=1. Treat their stored values as exact real constants. Define Q(q) by linear interpolation from(ci,xi) to(ci+1,xi+1) on each intervalci<q<ci+1. Equal-ci spans have no positive q-measure; choose any consistent monotone endpoint convention there. For a continuous uniformq, Q has no atoms because every positive-q segment has strictly positive x slope. It may have gaps when c is flat. Its CDF G is

    G(b)=0 for b<=x0; G(b)=1 for b>=xN;
    G(b)=ci+(ci+1-ci)*(b-xi)/(xi+1-xi)
        when xi<=b<=xi+1.

Proof: on a positive-CDF segment the monotone affine Q maps q<=G(b) to x<=b. Flat-CDF segments carry no q-measure, and both endpoint formulas agree. Summing all lower segments givesG(b). Thus bin masses areG(b)-G(a), requiring no network or new mass integral. If x,c,b are rational (including exact stored binary floats), these numbers are rational in real arithmetic. This does not make the c values correct Gaussian integrals.

If p=G(b)-G(a)>0, q=G(a)+p*V followed byQ constructs the conditioned grid-law height, while the other latent coordinates remain independent. If p=0, that stratum contributes zero to its grid-law expectation and no conditional generator is defined; do not divide by zero or assign a fictitious distribution. If the CDF/grid assumptions fail, this theorem is unavailable. Inversion at endpoints of flat CDF sections is harmless only under the continuous-uniform law; actual finite rules need explicit conventions.

For the source, ci is normalized(cumulative_i-cumulative_0)/(cumulative_N-cumulative_0), not cumulative_i/cumulative_N. Meanwhile itsZ uses cumulative_N. Therefore combining sourceZ with the real-interpolated grid law reproduces that particular formal numerical-law functional; it is not automatically the ideal co-area integral. Integration discretization, truncated[-8,8] domain, first-cell exclusion and floating operations remain separate.

## 3. The actual finite program and fixed rules

np.interp performs floating operations; U may come from a finite Sobol rule or finite RNG. Its output is not identical by theorem to the real Q above. A fixed rule's realized bin share is an empirical weighted count, not p_{k,j}, and finite coordinates can hit bin boundaries. In particular an affine conditional map withV=0 may produce an excluded lower endpoint. The continuous conditioning proof does not authorize an arbitrary epsilon shift, clipping, resampling, changed node count or iid confidence interval. Any implementation must freeze an endpoint policy, source/output law and accounting first.

If a separately proved uniform interpolation error bound |Q_fl(q)-Q(q)|<=eta holds, then a bin-boundary classification disagreement atb is contained in{|Q(q)-b|<=eta}. Its continuous-latent probability is at mostG(b+eta)-G(b-eta), and a union bound covers several boundaries. This gives a possible finite-arithmetic certification route but no eta is derived here. Errors in constructing c, finite latent probabilities and rounding the conditional affine map are additional terms. Exact rational evaluation of G only certifies the formal frozen-array law.

Quantile clipping of other coordinates and the tau cap do not change the definition of this input-height bin, but they change the joint numerical face/integrand from the ideal one. A network-independent label/probability does not remove those approximation errors or walker termination/tie errors.

## 4. What remains open

The simple family can be labeled before an expensive fibre walk; fullcoarse wt/flips cannot generally be known that way. This removes an identifiable circular prerequisite for a candidate partition, not the cost of computing its conditional expectations. Coarsening may erase nearly all useful predictive power. T230's total-variance/control-variate/oracle-allocation identities apply only under their stated known-law, independence, setup and cost conditions. Fixed Sobol/other deterministic quadrature needs an error analysis rather than an iid variance inference. Stochastic conditional sampling is a comparator, not a deductive ARC solution.

No new face tables, target samples, integrals, conditional moments, quadrature performance or trained accuracy were calculated. Next potential implementation would first freeze the law and endpoint convention, then analytic synthetic controls and probability/setup cost, then separately approved complete-family evidence. Existing selected-tail archives cannot substitute for that full-family information. Both useful and useless within-family behavior remain possible.
