# Positive tail and conditional enclosure argument

Write q=m+delta*x, x in[-1,1], 0<a=m-delta<=m+delta=b. For delta>0 set s=sqrt(ab), r=delta/(m+s) and z=m/s-1. Then 0<r<1 and z>=0. The Chebyshev expansion of q^-1 has coefficient2*(-r)^k/s for k>=1. The constant coefficient uses a different convention; it is never part of a remainder starting N=K+1>=1.

At fixed delta, s'=m/s=1+z, r'/r=-1/s, and z'=-(z²+2z)/s. Since q^-(nu+1)=-(1/nu)*d_m(q^-nu), differentiation gives the recurrence in README. Uniform analyticity on a neighborhood of[-1,1] permits coefficient differentiation for a>0. By induction all coefficients of P_nu(k,z) are nonnegative rational numbers. In particular P3=k²+3*(1+z)*k+2+6z+3z², agreeing with the independently derived inverse-cube expression.

For N>=1, expand k=N+j. The moment sum for j^0 is1/(1-r); for j^l, l>=1, it is r*A_l(r)/(1-r)^(l+1), where A_l has the nonnegative Eulerian coefficients. This identity follows by applying r*d_r repeatedly to1/(1-r), or the standard Eulerian recurrence. Consequently every operation in the finite tail expression can be positive. Since |T_k(x)|<=1, the sum of coefficient magnitudes bounds the uniform degree-K remainder. Stable evaluation uses r=(b-a)/(a+b+2s), z=(b-a)²/[2s(a+b+2s)], avoiding subtraction of nearly equal alpha and1. If a=b, the function is constant and every such remainder is exactlyzero.

## Positive quadrature factor

For d=2nu and vertex-column matrix V, Gaussian cone mass in simplex coordinates is C*|det V| times the integral of q^-nu, with C=(nu-1)!/(2*pi^nu) and simplex volume1/(2nu-1)!. Let a positive quadrature rule have the same constant mass and integrate the degree-K polynomial approximation in q exactly. Both the integral and quadrature errors from replacing q^-nu by that polynomial are at most simplex_volume*tail. Their difference is therefore bounded by

    |det V| * (nu-1)! / [pi^nu * (2nu-1)!] * tail.

Exact integration of inner coordinates is a positive operator and preserves this argument whenever the outer rule retains polynomial exactness. This primitive does not independently verify an arbitrary caller's quadrature degree, ray geometry or inner numerical approximation. In the existing T6 Jacobi planner the retained degree is K=n-1. Inner numerical error and omitted-cone mass require separate bounds.

## Exact radial setup for the stored fixture

Every binary64 stored ray coordinate is an exact rational number. For simplex weights y>=0 summing to1, q=y^T G y is a convex combination of entries of the exact Gram matrix G=V^T V. Thus min(G) is a lower bound. Convexity of squared Euclidean norm gives q<=max_i ||v_i||². For w=sum_i v_i with nonnegative minimum projection, w dot Vy>=min_i w dot v_i>=0; Cauchy gives q>=(min_i w dot v_i)²/(w dot w). Taking the larger lower bound remains valid. Exact Gaussian elimination yields the rational determinant. The fixture converts the lower radial endpoint downward, upper endpoint upward, and determinant to an interval before evaluating the tail.

This proves input validity for the exact stored simplices. It does not prove those simplices form a complete/disjoint partition of the original real-weight network event. Any floating geometry omission, rank decision or topology error remains outside the enclosure.

## Arithmetic enclosure

Directed Decimal basic operations enclose each exact operation. Decimal square root is correctly rounded to nearest even regardless of context direction, so the implementation explicitly uses adjacent decimal values below/above each rounded endpoint. Rational conversions are directed divisions of exact integer endpoints. Machin's identity uses alternating rational series bounds for atan(1/5) and atan(1/239); reversing the bounds on the negative coefficient encloses pi. All tail operands have nonnegative mathematical signs, and every denominator is checked positive. Repeated outward multiplication encloses integer powers. By structural induction through the expression and directed sums, the result encloses the analytic tail/bound under the Decimal runtime specification. Arithmetic overflow, unresolved denominators or configured exponent exceptions raise; they are not replaced by a finite guess.

The binary alternative instead assumes correctly rounded IEEE basic arithmetic and square root. One adjacent binary64 value in each direction encloses a rounded operation. Known nonnegativity allows clipping a negative outward lower neighbor tozero. Intermediate uncertainty involving nonfinite values is widened conservatively, and denominator intervals touchingzero have infinite upper bounds. Integer powers use the same outward operations. Its output may be[0,+inf] or extremely loose after underflow; the finite fixed corpus does not prove useful widths universally. Exact a=b and zero-determinant branches are applied after all scaling. There is no opaque floating power whose rounding is silently assumed.

Neither runtime has been formally verified here. The tests are independent numerical checks of the algebra and implementation under stated arithmetic assumptions, not a proof of the whole neural-network accuracy.
