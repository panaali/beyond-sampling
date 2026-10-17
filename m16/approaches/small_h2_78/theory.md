# Reset-prefix marginalization proof

Write Wi=p with p0,p1>0 and Wh=A. If A00,A01<=0, any negative input gives h0'=0 for every h>=0. If also A11<=0, a second negative input gives h0''=h1''=0. This is a sign proof and is unchanged by any positive scaling of Wi. A subsequent positive input gives h=p*x; its readout winner is computed using exact rational products of stored floating-point weights. Output ties choose the first index.

Partition inputs by the **last** negative-negative pair, ending at zero-based index tau-1. The first m=tau-2 inputs are unrestricted iid normals, the pair is iid negative-truncated normal, and the suffix starts positive and has no negative-negative pair. These conditions are independent; the pair contributes probability1/4. The zero hidden state at tau erases the discarded values only from the network computation, not from the target rank.

For u with v=Phi(u), an ordinary discarded input is below u with probability v, and a negative-truncated input is below u with probability G=min(1,2v). With a ordinary and b truncated variables, probabilities of zero/one exceedances are

P0=v^a G^b,
P1=a(1-v)v^(a-1)G^b + b(1-G)G^(b-1)v^a,

omitting zero-count terms. If the predicted target is in the suffix, its known rank within the suffix determines whether P0, P1 or0 applies.

If the prediction refers to a discarded index, integrate its unknown u as well. For an ordinary target its CDF density is dv and the other counts are a=m-1,b=2. For a reset-pair target its density is2dv on[0,1/2] and the other counts are a=m,b=1. Let M and S be the largest and second-largest suffix CDF values, using0 for absent entries. Above M, integrate P1; between S and M, integrate P0. Below S correctness is impossible. These are exact polynomial integrals, split at v=1/2. No approximation or rank-independence assumption is involved.

For v<=1/2, P0=2^b v^(a+b) and P1=[a*2^b+b*2^(b-1)]v^(a+b-1)−(a+b)*2^b v^(a+b). For v>=1/2, P0=v^a and P1=a(v^(a-1)−v^a). The b=0 coefficient is interpreted as zero in the first term involving b. All boundary terms can be evaluated with rational arithmetic when the CDF thresholds are rational; evaluating Phi of an actual suffix and floating polynomial arithmetic introduces numerical rounding.

The implementation is independent of a particular RNN after its reset theorem is checked. Constant-output networks provide a partition test: marginalizing all components must yield1/T. Exact last-reset sign masses at T8 leave55/256 of the distribution in the no-reset case. A truncated reset estimator must report this mass explicitly.
