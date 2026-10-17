# A stable four-dimensional simplicial cone mass and its quadrature bound

This concerns Gaussian volume integration after a cone is known. It is not a certificate for floating-point cone enumeration. The algebra was independently checked in conversation by the H2T3/T4 worker; executable identity and subdivision checks appear in refinement.json.

Let the columns of R be four rays in one signed coordinate orthant, normalized to L1 norm one. For standard Gaussian X in R4, substitute X=r R y, where r>=0 and y lies in the standard three-simplex (sum y_i=1, y_i>=0). The Jacobian is |det R|r^3. Integrating r gives

    P(X in cone R) = |det R|/(2*pi^2) * integral_simplex ||R y||^-4 dy.

The simplex measure is dy1 dy2 dy3 and has volume1/6. Every R y stays in the same orthant with L1 norm one. Thus s=||R y||^2 lies in[1/4,1], even if R is nearly singular. The determinant multiplies a bounded positive integral; no inverse covariance or subtraction of large orthant probabilities is needed.

Use the Duffy transformation

    y=(u,(1-u)v,(1-u)(1-v)w,(1-u)(1-v)(1-w)),
    Jacobian=(1-u)^2(1-v),  (u,v,w) in[0,1]^3.

Each y component is affine separately in each cube variable. Consequently s is degree2 in each variable. Tensor Gauss-Legendre with n nodes per variable integrates any polynomial in s of degree N=n-2 exactly, including the Duffy Jacobian.

For z=(s-5/8)/(3/8), the exact Chebyshev series is

    1/s^2 = 5 + sum_(k>=1) (10+8k)(-1/3)^k T_k(z).

One derivation differentiates the standard Chebyshev expansion of1/(a+bz) with respect to a at a=5/8,b=3/8, with sqrt(a^2-b^2)=1/2. Since |T_k(z)|<=1, truncating after N gives uniform error at most

    epsilon_N = sum_(k>N)(10+8k)/3^k = (12N+33)/3^(N+1).

Both the exact integral and the positive quadrature integrate the polynomial identically. Each remainder contributes at most epsilon_N times the simplex volume. Therefore a single cone's mass error is at most

    |det R| * epsilon_(n-2)/(6*pi^2).

For disjoint cones within the16 signed orthants, normalized cross-section volumes imply sum|det R|<=16. Thus the total accuracy integration error is at most

    8/(3*pi^2) * (12n+9)/3^(n-1).

This is approximately3.78e-6 at n16,5.79e-8 at n20, and8.52e-10 at n24. Actual per-model bounds can use its determinant sum instead of16. The bound covers ideal quadrature error only. Floating input weights, clipping tolerances, Qhull triangulation, omitted degenerate pieces, finite-precision determinants, and rounding in the quadrature still require separate scrutiny.
