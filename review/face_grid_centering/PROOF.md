# Symmetric real-grid centering diagnostic

This is written algebra, not an evaluation of the NumPy face table. Its role is to test whether the *formal real-arithmetic construction* in the source preserves a known property of an even density. The output law is the continuous distribution induced by linearly interpolating the inverse CDF at a uniform latent coordinate, not discrete endpoint sampling.

## Setup and binding to the source

Fix A>0 and an even integerN=2m>=2. Let Delta=2A/N and x_j=-A+j Delta forj=0,...,N. Suppose positive ordinates d_j satisfy d_j=d_{N-j}. Define

    S=sum_{j=1}^N d_j,
    C_j=(sum_{r=0}^j d_r-d_0)/(sum_{r=0}^N d_r-d_0)
       =sum_{r=1}^j d_r/S.

Then C_0=0,C_N=1 and every increment is positive. This is exactly the real-arithmetic cumulative construction `c=cumsum(d)*dg; (c-c[0])/(c[-1]-c[0])` on a uniform grid. Its inverse interpolation Q maps each probability interval(C_{j-1},C_j) affinely onto(x_{j-1},x_j). Therefore Y=Q(V), with continuous uniformV, is uniform inside intervalj with total interval probability d_j/S. Equivalently its density there is d_j/(S Delta). Endpoint conventions have zero continuous-latent mass. Positivity avoids flat-CDF ambiguity in this diagnostic.

In frozen core.py, T=2 kind12 has analytic d(u)=exp(-u²)/sqrt(pi), which is even and strictly larger at0 than at+-A. The formal grid matching the source endpoints/count is A=8,N=160000,Delta=1/10000. This does not assert that the floating `linspace`, `exp`, cumulative sums or `np.interp` produce exact symmetric values. The finite-program law remains separate. No T=10 performance inference or source execution is made.

## Exact CDF bias at zero

Since x_m=0,

    P(Y<=0)=sum_{j=1}^m d_j/S.

Let L=sum_{j=1}^{m-1} d_j. Reflection pairs the interior right ordinates with these left ordinates, so

    S=2L+d_m+d_N,
    sum_{j=1}^m d_j=L+d_m,
    P(Y<=0)-1/2=(d_m-d_N)/(2S).

Thus an even density with a higher center than edge yields a strictly positive central CDF error despite an exactly symmetric grid. A normalized symmetric continuous target law has CDF(0)=1/2, including both full and symmetrically truncated Gaussian laws. The bounded test function1[u<=0] witnesses this law mismatch. It is not a trained fibre-integrand error witness and does not imply that every integrand has nonzero error; constants, for example, do not.

For the Gaussian formal grid,A=8,N=160000, every ordinate is at mostd_m=1/sqrt(pi). HenceS<=N/sqrt(pi), and

    P(Y<=0)-1/2 >= (1-exp(-64))/(2N) > 1/(4N) = 1/640000.

The final strict inequality uses exp(64)>2, requiring no numerical Gaussian evaluation. This is a conservative lower bound for the formal even Gaussian grid law. It is not a directed bound for the actual floating array. Total variation from either symmetric target law is at least this CDF gap under the sup-event convention. No numerical value of S was computed.

## Continuous interpolation mean versus endpoint mean

The conditional mean on intervalj is its midpoint x_j-Delta/2. Pairing the full symmetric endpoint sum gives sum_{j=0}^N x_j d_j=0; removing j=0 leaves A d_0=A d_N. Consequently

    E[Y]=sum_{j=1}^N (d_j/S)*(x_j-Delta/2)
        = A*d_N/S - Delta/2.

By contrast, a discrete law putting probability d_j/S directly at endpointx_j has mean A*d_N/S. Omitting the interpolation term would give the wrong source-law diagnosis. If every interior ordinate is at least the edge and at least one is strictly larger, then S>N*d_N, so E[Y]<0. The constant-ordinate case has S=N*d_N and E[Y]=0; the claimed defect does not apply to every symmetric grid density.

For any fixed continuous positive even d, as Delta->0 through even uniform grids, Delta*S->M_A=integral_{-A}^A d(u)du. Therefore

    CDF gap / Delta -> (d(0)-d(A))/(2M_A),
    E[Y] / Delta -> A*d(A)/M_A - 1/2.

These are asymptotic statements about a specified real-arithmetic family. They do not supply an error bound at a particular floating grid size or a target accuracy floor.

## A symmetry-preserving cumulative alternative

Define trapezoidal interval weights

    a_j=(d_{j-1}+d_j)/2,  j=1,...,N,
    W=sum_j a_j,
    C^trap_j=sum_{r=1}^j a_r/W.

Reflection gives a_j=a_{N+1-j}. Inverse-linear interpolation now yields equal probabilities on reflected intervals, with uniform conditional densities. Its law is symmetric, so its mean is exactly0 and its CDF at0 exactly1/2. For even endpoint data,d_0=d_N, W=S. This is an algebraic alternative, not an implemented or adopted source patch.

The symmetry property is much weaker than equality with the Gaussian law. A finite interpolated-CDF grid gives a piecewise-constant density, which cannot equal a nonconstant analytic Gaussian density on every positive-width interval. No total-variation, all-integrand, quadrature convergence or trained-error certificate follows from correcting centering alone. At T>2 the face-height density is generally asymmetric, so this particular symmetry benchmark does not apply as stated.

The source co-area weightZ uses Delta*sum_{j=0}^N d_j=Delta*(S+d_0), whereas the inverse-law normalizer excludesd_0. The trapezoidal integration normalizer is Delta*W=Delta*S in this even setup. Changing only the CDF does not automatically update or certifyZ. Both the grid's ideal integration error and finite numerical operations need independent treatment.

## Scope and plausible rescue

This provides a falsifiable analytic diagnostic for the source's formal grid construction and a narrow rescue of one symmetry property. It does not establish an actual trained floating-program failure, invalidate previous point estimates, or rule out the tie-face method. A future separately gated implementation could compare the original law and a symmetric alternative on fixed analytic controls, retaining cost and endpoint behavior. Any real face/table or target experiment remains outside this proof-only claim and requires its own scope and review. No numerical grid, network, face, random point, mass integration or fitting was executed here.
