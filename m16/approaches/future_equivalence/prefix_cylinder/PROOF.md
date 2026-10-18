# Unrestricted fresh input prevents extra exact margin equalities

## Statement

Let d>=1, k>=1 and let C be a subset of R^d with nonempty interior. Write the fresh scalar coordinate as t and the prefix coordinates as z. Consider a homogeneous vector-valued function

F(z,t) = L(z,t) + sum_i v_i ReLU(a_i·z + b_i t),

where L is linear, v_i is in R^k and **every b_i is nonzero**. All coefficients and equalities here are exact real quantities. A difference of two finite one-step ReLU margin networks with nonzero input weights has this form. No convexity or conic assumption on C is necessary for this theorem.

**Theorem.** F vanishes everywhere on C×R if and only if F vanishes globally on R^(d+1). It is enough to assume vanishing on int(C)×R.

This concerns exact vector functions, not labels, probability, approximate agreement, point-dependent positive scaling or computational savings. The fresh coordinate must be unrestricted; a full-dimensional arbitrary subset of R^(d+1) is not enough.

## Canonical preparation

Use the reviewed T208 exact homogeneous ReLU normalization: for a nonzero form q and its first nonzero coordinate alpha, write u=q/alpha, so

ReLU(alpha*u·x) = |alpha| ReLU(u·x) + 1_(alpha<0) alpha*u·x.

Add the last term to the linear part, combine identical normalized forms, and remove zero vector coefficients. This gives

G(x) = A x + sum_(u in U) c_u ReLU(u·x),

with distinct normalized nonzero u and nonzero vector coefficients c_u. All surviving forms still have nonzero fresh-coordinate coefficient: normalization scales a nonzero coefficient and combining equal forms does not change that form. This represents F globally exactly, including boundaries.

## Every remaining hyperplane crosses the cylinder interior

Fix u=(a,b) with b!=0. Its hyperplane H_u is the graph

t = -a·z/b.

Over any nonempty open ball B contained in int(C), this graph is a nonempty relatively open subset of H_u. Its points lie in int(C)×R. The projection H_u→R^d onto z is an invertible linear map, so the graph patch has full dimension d within H_u.

Every other distinct normalized form v defines a different hyperplane. Its intersection with H_u is a proper linear subspace of H_u. A finite union of proper subspaces cannot cover a nonempty relatively open patch. Choose x0 on the patch away from all other remaining hyperplanes.

Because x0 lies in the open cylinder, a sufficiently small Euclidean ball about x0 is contained in that cylinder and meets no other remaining hyperplane. The two open sides of H_u in this ball are nonempty. On each side G is linear with constant Jacobian; all other hinge signs are unchanged. Their Jacobian difference is the nonzero outer product c_u*u.

If G vanished on the cylinder, both Jacobians would be zero. Therefore c_u*u=0, which forces c_u=0 since u!=0, contradicting its retention. Thus U must be empty. The remaining linear map A vanishes on the nonempty open cylinder, hence A=0. This proves local-to-global necessity. Global vanishing trivially implies cylinder vanishing.

The argument is direct hyperplane isolation, the same necessity mechanism independently reviewed in T208. It does not assume that C is finitely generated or that a floating H/V representation is complete.

## Consequence for one-step recurrent margins

For fixed input weights Wi, recurrent matrix Wh and readout Wo, a frozen prefix-state map M in R^(H×d) gives future hidden forms

(Wh*M)_i·z + Wi_i*t.

Margins relative to output0 are

f_M(z,t) = (Wo[1:]-Wo[0]) ReLU(Wh*M*z + Wi*t).

If every Wi_i!=0, the difference f_M-f_N satisfies the theorem. Hence equality on ANY C×R with nonempty interior(C) is equivalent to equality globally. This conclusion does not itself require Wh invertibility or a readout fingerprint condition.

The separately reviewed T11597 identifiability theorem adds the sufficient predicates:

- Wh is invertible;
- every Wi_i!=0;
- the 2^H subset sums of vectors abs(Wi_i)*(Wo[1:,i]-Wo[0,i]) are all distinct.

Under those predicates, global equality f_M=f_N forces M=N. Therefore two distinct states cannot admit exact vector-margin replacement on an entire full-dimensional **gate-prefix cylinder**. Equality on just one such domain already forces the same state; testing both domains adds no weaker possibility in this setting. No new model predicate computations are performed here. Existing fixed-model predicates are independently reviewed, but the cylinder premise still needs its own applicability proof.

## Critical distinction: gate-prefix versus rank-chart domains

An exact activation history before reading the final input depends only on prefix coordinates. Its gates-only domain is a cylinder C×R, and the theorem applies if C has nonempty interior and the stated weight predicates hold.

Our T188/T11596 stored cones ALSO carry an initial truth/rank-chart restriction. The target is second argmax of ALL inputs, so those restrictions can constrain the final input before its neural step is evaluated. Such represented cones need not be cylinders even when full-dimensional. **This proof does not automatically apply to any of those stored cones.** Their fixed128 inequality result instead uses independent exact rank witnesses, generator signs and the finite-cone T208 theorem.

Likewise, a domain after the final ReLU split usually restricts t. The corollary does not rule out additional exact equalities on such domains. Consolidating charts before terminal processing would need correct truth/rank bookkeeping and charged domain representation; neither follows from this theorem.

## Assumption counterexamples and alternatives

1. **A zero fresh coefficient can hide a hinge.** F(z,t)=ReLU(-z), C=(0,infinity), is zero on C×R but not globally. This violates b!=0. Thus allowing a genuinely input-disconnected unit changes the result.
2. **Restricting the fresh input permits distinct functions to agree.** ReLU(z+t) and ReLU(2z+t) have nonzero fresh coefficients and are globally different, but both vanish on the full-dimensional cone z>0,t<-2z. At z=1,t=0 their values are1 and2. Thus full dimension alone is insufficient; final-input chart restrictions remain a plausible source of domain-only equality.
3. **A lower-dimensional prefix can hide differences.** For C={0} in R, ReLU(z+t)-ReLU(t) vanishes on C×R but not globally (take z=1,t=0). Both fresh coefficients are nonzero.
4. **Equal winning labels do not require equal margins.** Scores (0,ReLU(t)) and (0,2ReLU(t)) have identical first-index predictions at every t, including ties, but unequal margins for t>0. The theorem does not exclude label-preserving or positive-scale strategies.

These examples are exact algebraic counterexamples, not numerical experiments. They separate the two main alternative explanations: a domain-only equality may rely on a constrained fresh coordinate or a deficient prefix span; an apparent merging success may preserve labels rather than exact margins. The theorem addresses only exact full-dimensional cylinders with nonzero fresh coefficients. It is not an exhaustion claim for every merger or estimator.

## Evidence and accounting

Claim d0e0cd94 is TASKS-only and precedes this artifact. T208 proof and T11597 proof are the only mathematical dependencies; source hashes are in provenance.json. Independent written review is delegated to recover_h4. No candidate or reference calls, model loads, synthetic evaluations, geometry, determinant/rank computations, random inputs, mass integration, or native-cost work occurred in this proof-only phase. Existing T11596 measurements remain separate and unchanged. Human/agent symbolic reasoning and document processing are not represented as zero-cost algorithmic preprocessing.
