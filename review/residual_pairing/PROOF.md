# Time-reversal pairing: exact contract and limits

Written/source-only T266. Fix T>=2 and X~N(0,I_T). Define R(x)_i=x_(T-1-i), an orthogonal involution; R pushes this Gaussian law to itself. This statement concerns the input law, not a weight symmetry or neural equivariance.

## Correctness and labels
Let f(x) be the fixed network's first-index winning output and tau(x) the second-largest input index, with a fully specified first-index tie order. Set Y(x)=1{f(x)=tau(x)}. With distinct input values, tau(Rx)=T-1-tau(x). Input ties have Gaussian probability zero, but this relabeling need not hold on tied synthetic fixtures: e.g. an all-equal vector has second index1 both before and after reversal, whereas reversal of index1 need not be1. An implementation must either restrict the label-permutation assertion to strict ranks or recompute tau(Rx) under the declared tie rule. Output ties must always retain first-index ownership. No claim f(Rx)=R(f(x)) is made; both network evaluations are required unless a separate theorem eliminates one.

## Signed residual identity
Fix a measurable surrogate S(x) in[0,1] and its independently justified integral mu_S=E[S(X)]. No training or answer-key calibration is implicit. Define r=Y-S and p=(r+r∘R)/2. Then p∘R=p and

E[p]=E[r], and A=E[Y]=mu_S+E[p].

This follows only from measure preservation and linearity. Pairing does NOT change the signed bias E[r] of the approximation mu_S obtained by dropping the residual. If mu_S is approximate with certified error e_S, that error must be added separately. If S or its expectation is fitted or chosen on scored inputs, additional statistical/deployment rules are needed.

For any real a,b,

(|a|+|b|-|a+b|)/2 = min(|a|,|b|) * 1{ab<0}.

Thus

E|p| = E|r| - E[min(|r|,|r∘R|) * 1{r(r∘R)<0}] <= E|r|.

For binary S, r is in{-1,0,1}, and the improvement equals P(r=-r∘R!=0). Cancellation requires opposite signs, not merely different gate histories or labels. For S=1, r=Y-1<=0, so E|p|=E|r|=1-A: this common optimistic baseline cannot improve the absolute residual budget through pairing. Likewise S=0 yields one-sided residuals. These are specific surrogate facts, not a negative theorem for other S or integration schemes.

The remaining absolute budget always satisfies E|p|>=|E[r]|. A smaller absolute bound does not itself establish a smaller actual signed error, which is invariant under pairing.

## Exact fundamental-domain representation
Let D={x:x0<x_(T-1)}. Its reversal is the opposite strict inequality, and the excluded equality hyperplane has Gaussian mass0. Therefore P(D)=1/2 and

E[r] = integral_D [r(x)+r(Rx)] dmu(x) = E[p(X) | X in D].

Also E|p|=E[|p(X)| | X in D]. Integrating only D removes duplicate orbit representatives, but each representative still requires two residual evaluations and the construction of their joint classification. A partition for r(x) alone is not automatically a partition for r(Rx); intersections/refinement and boundary ownership must be charged.

If a method integrates p on an R-invariant retained set K and drops the rest, a valid absolute residual error is integral_(K^c)|p| dmu plus all retained integration/certification errors. Alternatively use the normalized D formulation consistently. Selecting regions using signed point samples or independently inaccurate cone masses is not this certificate. No universal cell-count, boundary or quadrature-rate improvement follows.

## Sampling comparison is separate
For r in L2 with variance sigma^2>0 and rho=Cov(r,r∘R)/sigma^2,

Var(p)=sigma^2(1+rho)/2.

With n independent paired inputs versus2n independent single inputs, paired-mean variance divided by single-mean variance is1+rho. This comparison assumes EXACTLY twice the single-evaluation cost and negligible/equal setup. Positive correlation loses, negative correlation helps, zero correlation ties. If sigma^2=0 both variances vanish and the ratio is undefined.

For fixed deterministic single and paired costs c_s,c_p, budget B, m=floor(B/c_s)>=1 and n=floor(B/c_p)>=1, the variance ratio is (m/n)(1+rho)/2. Include construction, both labels/surrogates/networks, routing, integration and setup; variable runtimes or adaptive stopping require a separate budget argument. These variance identities do not prove a deterministic quadrature benefit or a FLOP speedup.

## Scope
The law is unconditional iid Gaussian. Conditioning on a non-invariant rank chamber or fixed prefix generally destroys reversal invariance; either pair its image stratum with correct mass or prove a new conditional identity. No arbitrary hidden-state pairing follows from this input involution. Exact label transformations, region feasibility, weights/provenance and full-cost accounting remain separate prerequisites. No model, array, RNG, CDF, geometry or mass evaluation occurs in this written phase.
