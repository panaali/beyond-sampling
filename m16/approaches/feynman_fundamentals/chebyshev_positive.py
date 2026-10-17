"""Positive generic Chebyshev tail bound, with real-arithmetic assumptions."""
from math import exp,lgamma,log,log1p,sqrt


def positive_tail_bound(lower,upper,nu,degree):
    """Uniform tail after Chebyshev degree K for q^-nu on q∈[lower,upper].

    nu>=1, K>=0. Returns infinity when the coefficient-ratio bound is unavailable.
    Floating evaluation is not an interval-arithmetic certificate.
    """
    if lower<=0 or upper<lower or nu<1 or degree<0 or int(degree)!=degree:
        raise ValueError('need0<lower<=upper, nu>=1 and nonnegative integer degree')
    sa,sb=sqrt(lower),sqrt(upper)
    r=(sb-sa)/(sb+sa)
    if r==0:
        return 0.
    C=(sa+sb)**2/4
    N=int(degree)+1
    A=(nu+N)/(N+1)
    rho=r*A
    if rho>=1:
        return float('inf')
    value=(log(2)-nu*log(C)+lgamma(nu+N)-lgamma(nu)-lgamma(N+1)
           +N*log(r)-log1p(-rho)-nu*log1p(-A*r*r))
    return exp(value)
