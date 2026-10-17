# Why the small-model cone calculation works

This establishes the mathematical reduction and quadrature truncation bound. It does not certify the current floating-point clipping, Qhull, or roundoff.

## T3: two analytic conditional integrals
For one standard Gaussian input, write x0=R U with U uniform on {-1,+1} and R independent, chi_1. Homogeneity gives the prefix hidden state R h(U). Adding two fresh independent Gaussian inputs, v=(R,x1,x2)/||(R,x1,x2)|| has density1/(2pi) on the positive v0 hemisphere. Every fixed ReLU history, input-rank condition, and output comparison is a homogeneous halfspace in v. Thus the conditional accuracy is a sum of spherical polygon areas divided by2pi. Average the two prefix signs. T55 implements this reduction; the independent T31 reference instead partitions all R3 octants and three recurrent steps before rank/output clipping.

Readout ties need explicit ownership: when a comparison linear form is identically zero on a full-dimensional gate cone, the lower output index wins. A generic nonzero comparison equality has zero Gaussian measure. Input ranks also have zero-probability ties under the original iid continuous distribution.

## T4: enumerate cones, then integrate a positive function
For fixed weights, ReLU unrolling gives finitely many homogeneous polyhedral gate cones. The correct-answer event intersects them with output halfspaces and, for each candidate second-largest index, the conditions that precisely one other input is larger. Distinct maximum/second-maximum choices have disjoint interiors.

Intersect each cone with one signed-orthant cross section: sum_i |x_i|=1. Triangulate its three-dimensional convex polytope. A simplex has four ray columns R, all in one orthant and each with L1 norm1. A conic point is x=r R y, where r>=0, y_i>=0 and sum y_i=1. The Jacobian is |det R| r^3. Integrating the standard Gaussian radial variable gives

    P(cone simplex) = |det R|/(2 pi^2) * integral_Delta3 ||R y||^(-4) dy.

No sampled network inputs enter this calculation. Since R y remains in the same signed-orthant cross section, Cauchy-Schwarz gives1/4 <= s=||R y||^2 <=1. The integrand s^-2 is positive, smooth, and bounded between1 and16, including for very thin simplices. This avoids inverse-covariance cancellation in the earlier Plackett implementation.

## Explicit integration-only error bound
Map s=5/8+(3/8)z. Its Chebyshev expansion is

    1/s^2 = 5 + sum_{k>=1} (10+8k)(-1/3)^k T_k(z).

One derivation differentiates the standard expansion for1/(a+bz) with respect to a at a=5/8,b=3/8. Since |T_k(z)|<=1, truncating after N has uniform remainder at most

    tail_N = (12N+33)/3^(N+1).

Use the Duffy map y=[u,(1-u)v,(1-u)(1-v)w,(1-u)(1-v)(1-w)] with Jacobian(1-u)^2(1-v). A polynomial of degree N in s has degree at most2N in each cube coordinate before the Jacobian. Tensor n-point Gauss-Legendre is therefore exact for this truncated polynomial when N=n-2. Both integration and quadrature are positive linear functionals with simplex mass1/6, so their difference on the remainder is at most tail_N/3. Consequently the simplex probability error is at most

    |det R| * (12n+9)/(6 pi^2 3^(n-1)).

For disjoint simplices across the16 orthants, sum|det R|<=16. This yields a model-independent global bound8 tail_N/(3 pi^2). The implementation uses the tighter actual determinant sum for the correct region. The GL16 H2T4 seed0 bound is1.16789e-6; GL20 is1.78615e-8. These exclude floating geometry and arithmetic error.

## Compute and the remaining acceptance limit
At each quadrature node, the direct four-dimensional matrix/vector evaluation uses16 multiplications and12 additions, the squared norm4 multiplications and3 additions, the squared reciprocal two operations, and the weighted summation about two operations: approximately39 scalar arithmetic operations per node. Rule construction, determinants, clipping, hulls, triangulation, and other setup are additional work. Hence node count alone must not be advertised as full solver FLOPs. Our process CPU measurements include these implemented operations and overhead; CPU-equivalent sampling cost is hardware/batch dependent.

The base uses569 correct simplices. At GL16 it uses2,330,624 geometric nodes, about90.9 million scalar operations for the positive-integrand evaluations alone. The full numerical result is strongly supported by independent angular calculations, higher quadrature order, full gate mass/volume checks, affine-map checks, noise ensembles, and exact tie controls. Independent review is still required for acceptance; the integration bound cannot be relabeled as an end-to-end software certificate.
