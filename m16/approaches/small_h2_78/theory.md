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

## Exact elimination of the last suffix input

For fixed earlier suffix values, the final hidden preactivation is affine in the last input y. Split at its two ReLU zeros, then at pairwise output-logit intersections within each gate interval. Each resulting open interval has a fixed first-index output prediction. Isolated input endpoints have zero Gaussian mass; identical affine logits and a dead hidden state can occupy positive intervals and are handled by first-index argmax.

Change variable v=Phi(y). The analytically marginalized rank probability has polynomial degree at most m+2 between the fixed earlier-suffix CDF values and1/2. A predicted last-input target uses P0 or P1 directly. A predicted earlier-suffix target has a fixed value and only changes when y crosses it. A discarded target uses polynomial antiderivatives with variable upper/lower limits; the other-variable degree is m+1, so its antiderivative degree is m+2. Therefore an n-point Gauss-Legendre rule, n=ceil((m+3)/2), exactly integrates every polynomial piece in exact arithmetic. This rule evaluates only the known rank law, not the network at sampled inputs. Floating gate geometry, Phi and polynomial arithmetic remain numerical limitations.

This removes one more suffix dimension. Outer integration over earlier suffix inputs is still needed. It must not be confused with the exact polynomial rule: finite outer quadrature is approximate and its convergence differences are not rigorous error bounds.

## Opposite structural certificate for the local T7 pilot

The pilot has opposite-sign input weights. Let v=(abs(Wi[1]),abs(Wi[0])); then v>0 and v Wi=0 exactly. Its two entries of v Wh are strictly positive. Starting from any h>=0,h!=0, the next preactivation z=Wh h+Wi x satisfies v z=(v Wh)h>0, regardless of x. Thus at least one preactivation is positive, and the hidden state cannot become zero. The first nonzero input produces nonzero state because the two input weights have opposite signs. Gaussian zero inputs have probability zero. This certificate is evaluated on exact dyadic rationals representing the stored weights, not rounded printed coefficients.

The pilot's nonzero-state readout can select only indices3,5,6. Index0 appears only through the all-zero tie, now proved unreachable almost surely. Therefore its accuracy is at most3/7. This is a structural ceiling, not a precise accuracy estimate. The H2T8 reset decomposition cannot be transferred to this model; a generic conditional-state or gate-geometric method is needed.
