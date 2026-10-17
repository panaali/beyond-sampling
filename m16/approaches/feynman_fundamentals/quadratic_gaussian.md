# Stable quadratic times Gaussian interval integration

T67 splits a scalar input interval at ReLU and next-grid crossings. Inside a
fixed cell, each next hidden coordinate is affine in x and bilinear grid
interpolation is therefore a quadratic polynomial P(x). Its integral against
the normal density can use moments

    M0=Phi(r)-Phi(l), M1=phi(l)-phi(r),
    M2=M0+l*phi(l)-r*phi(r).

These formulas are exact in real arithmetic but can cancel for narrow intervals.
Choose c=(l+r)/2,h=(r-l)/2,t=(x-c)/h. From P_left,P_mid,P_right obtain

    P(t)=A+B*t+C*t²,
    A=P_mid, B=(P_right-P_left)/2,
    C=(P_right+P_left)/2-P_mid.

This avoids forming large coefficients in the global x coordinate. Centered
moments can be written

    J1=phi(l)-phi(r)-c*M0,
    J2=(1+c²)*M0-r*phi(l)+l*phi(r).

Their necessary bounds are |J1|<=h*M0 and0<=J2<=h²*M0. Violations expose
roundoff, but passing these checks does not certify accuracy. Dividing by h or
h² to obtain t moments can amplify cancellation; a narrow-interval fallback
should not depend solely on detecting a negative answer.

## A positive fallback with a stated quadrature-only bound

Write phi(c+h*t)=phi(c)*exp(z(t)), where
z(t)=-c*h*t-h²*t²/2 and |z(t)|<=M=|c|h+h²/2. Let |P(t)|<=Pmax on[-1,1].
For n>=2, truncate exp(z) after N=n-2 powers. The product of P and this
polynomial has degree at most2N+2<=2n-1, so an n-point Gauss-Legendre rule is
exact for it. The remaining integrand is bounded uniformly by

    Pmax*exp(M)*M^(n-1)/(n-1)!.

Both the true interval integral and the positive quadrature have weight sum2
in t. Their two remainders therefore give

    |integral_l^r P(x)phi(x)dx - GL_n|
      <= 4*h*phi(c)*Pmax*exp(M)*M^(n-1)/(n-1)!.

The factor4 is essential: two from interval length in t and two from the
integral-plus-quadrature error. For a bilinear interpolation of node values in
[0,1], Pmax=1 because every point in the grid cell is a convex combination of
its node values. Otherwise use an explicit polynomial bound; endpoints alone
do not bound a general quadratic. If h=0 the contribution and bound are zero.

Evaluate P at quadrature nodes through its original positive bilinear weights
when available. Then the fallback avoids moment subtraction and preserves
nonnegativity. Normal CDF mass in a positive tail should use
Phi(-l)-Phi(-r), but this subtraction too can cancel on extremely short
intervals, which the positive fallback can handle directly.

For a conditional below/above-target transition, divide the interval contribution
and its error by the appropriate support mass, Z*q or Z*(1-q). Sum interval
bounds before doing so. Positive conditional transition and bilinear
interpolation operators have supnorm at most1, so a sum of per-stage maximum
integration errors bounds accumulated QUADRATURE error. It does not bound the
separate spatial interpolation approximation, grid clipping, root location
roundoff, or finite truncation error.

This Taylor/GL bound is a real-arithmetic statement. Float arithmetic can exceed
it when it is extremely small. The separate check preserves that distinction:
high-precision quadrature verifies the bound, while float errors are recorded
and may require an independently justified arithmetic allowance in an engine.
