# Positive Chebyshev coefficients and a generic tail bound

For f(x)=(m+delta*x)^(-nu), x in[-1,1], assume nu>=1 and m>delta>=0.
Let a=m-delta,b=m+delta,s=sqrt(a*b),

    r=(sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)),
    C=(sqrt(a)+sqrt(b))^2/4=(m+s)/2.

Then m+delta*cos(theta)=C(1+r*exp(i*theta))(1+r*exp(-i*theta)).
Expand both factors by the positive binomial series. In the convention
f(x)=c0/2+sum_{k>=1} c_k T_k(x), the nonconstant coefficients satisfy

    |c_k| = 2*C^-nu * r^k*(nu)_k/k! * H_k,
    H_k = 2F1(nu+k,nu;k+1;r^2),       k>=1.

Their signs are(-1)^k. H_k is positive and nonincreasing with k for nu>=1:
each coefficient ratio (nu+k)_j/(k+1)_j is a product of factors
1+(nu-1)/(k+1+i), each decreasing with k. At N=K+1>=1, write
A=(nu+N)/(N+1). Bounding every factor by A gives

    H_N <= sum_j (nu)_j/j! * (A*r^2)^j = (1-A*r^2)^(-nu).

The remaining base sequence r^k*(nu)_k/k! has successive ratios bounded by
rho=r*A for k>=N. If rho<1, then also A*r^2<1, and the uniform Chebyshev tail is

    ||f-[c0/2+sum_{k=1}^K c_k T_k]||_infinity
      <= 2*C^-nu * (nu)_N/N! * r^N
           / [(1-rho)*(1-A*r^2)^nu].

For r=0 the function is constant and this tail is zero. If rho>=1 this
particular bound is unavailable; increase K rather than inventing a finite
bound. Nu1 reduces exactly to the known inverse-linear geometric tail
2/s * r^N/(1-r). The proof uses positivity rather than cancellation and applies
to half-integer nu=2.5 as well as integer exponents. T48 independently checked
the coefficient monotonicity and the two ratio bounds before author validation.

For nu3, differentiation of inverse-linear coefficients gives the sharper exact
coefficient |c_k|=r^k/s^3*[k^2+3*alpha*k+3*alpha^2-1], alpha=m/s. Its summed
tail is

    r^N/[s^3*(1-r)] * [N^2+3*alpha*N+3*alpha^2-1
       +(2*N+3*alpha)*r/(1-r)+r*(1+r)/(1-r)^2].

T48 owns and validated this special-case bound. The generic bound is primarily
useful where that finite polynomial formula is unavailable. It can be evaluated
with three log-gamma calls, elementary logs and one exponential, with no model
inputs or integration nodes. Its floating evaluation remains uncertified.

For positive simplex quadrature, multiply the uniform function tail by twice
the relevant total weight (and geometric probability prefactors). Exact positive
segment/triangle elimination preserves the polynomial-degree argument in the
remaining coordinates. Separately retain all guarded inner-quadrature and
small-angle series errors; this bound does not certify cone geometry or rounding.
