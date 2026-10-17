# A finite union of Gaussian strips bounds hidden-state sensitivity

Fix the target value u and its output index. Let Z=(x,y) have independent standard Gaussian coordinates. The network takes three steps from hidden state h, with inputs x,u,y. Its correctness indicator is I(h,Z), using the repository's first-index tie convention.

For each first/middle gate word, extend the final preactivation algebra to all inputs and hidden states:

    z = A x + B y + C h + E u.

There are sixteen such words for H=2. Retain each coordinate of z as a final-support boundary. For each nonempty final support S and each competing output j, also retain the scalar margin obtained by multiplying z by (Wo[target]-Wo[j]) times S. Every retained scalar form has the shape

    f(h,Z) = n dot Z + c dot h + e u.

If c=0, the form cannot move with hidden state and needs no strip allowance. If n=0 but c is nonzero, return **unknown**. Otherwise assign the rational conservative allowance

    k(f) = ||c||_1 / max(|n_x|, |n_y|).

Exactly proportional forms, including their u coefficients, define the same strip for every pair of hidden states and can be deduplicated. This is an exact union simplification.

## Why the strip union covers every change

Follow the straight hidden-state path from h0 to h1. For fixed Z, the finite ReLU network is a continuous piecewise-affine function of the path parameter. A change in the target's first-index correctness must encounter either a readout-margin boundary or a change in final support that changes positive-volume tie ownership. Earlier gate changes alone do not create a discontinuity in the logits. A transition from an identically tied margin piece is recorded by its nonconstant adjoining piece; if final support changes, its gate boundary is already retained.

Choose a retained fixed-word affine extension at such a crossing. Unless the boundary is a fixed zero-measure Gaussian hyperplane, its hidden coefficient is nonzero and its affine values at the two path endpoints have opposite signs. It does not matter whether that gate word is valid at either endpoint: the extension is affine on the whole path. Thus Z belongs to the strip between the two parallel extended boundary hyperplanes.

The strip width is |c dot (h1-h0)|/||n||_2. A standard Gaussian projected onto a unit normal has density at most 1/sqrt(2pi), so its unconditional mass is at most this width times 1/sqrt(2pi). Therefore

    P(I(h0,Z) != I(h1,Z))
        <= min(1, sum_f k(f) * ||h1-h0||_infinity).

Using max(|n_x|,|n_y|) instead of ||n||_2 and density<1 makes the displayed coefficient rational and conservative. The bound deliberately ignores reachability and facet restrictions, so it can remain loose.

## Conditioning on the actual rank domain

For x,y restricted to [-6,6], define L=Phi(u)-Phi(-6) and U=Phi(6)-Phi(u). The unnormalized Gaussian probabilities of zero, one or two values above u are L^2, 2LU and U^2. Intersecting a strip with one of these domains cannot increase its unconditional mass. Divide the numerator bound by the corresponding probability; the initial truncated-square normalization cancels.

For a single ordered off-diagonal side the denominator is LU. For the need1 union it is 2LU; summing two separately normalized side bounds unnecessarily loses a factor of two.

At u=1, elementary bounds avoid numerical CDF certification: L>2phi(1)>1/3 and U>phi(2)>1/27, since sqrt(2pi)<3 and e<3. Certified need0/1/2 coefficients are consequently 9K, (81/2)K and 729K. Floating CDF-normalized values are reported separately.

## Why the zero-normal guard is necessary

Take Wi=(1,1), Wh=I, target readout (1,-1) at index2 and all other readouts zero. At h=(2,2), the two hidden coordinates remain identical for every future input, so target2 never wins the first-index tie. At h=(2+epsilon,2), every x,y in (0,1), with fixed u=1, preserves the positive coordinate difference epsilon and makes target2 uniquely win. Thus the correctness probability jumps by at least the positive Gaussian mass of that rectangle as epsilon tends to zero. The full-active-word margin has n=(0,0) and c=(1,-1); the implementation returns unknown. This does not imply that every rejected model is discontinuous.

With Wh=diag(1,2), the same full-active margin instead has x coefficient -3 and y coefficient zero. Its moving boundary is a genuine Gaussian strip. The primitive joint criterion succeeds although the earlier final-y-only test is unknown. Neither toy conclusion claims a trained-model interpolation advantage.
