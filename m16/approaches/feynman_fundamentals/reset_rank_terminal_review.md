# T49 reset-pair rank and terminal strata

This is a read-only algebra/source review of `reset_wedge/rank.py` and
`terminal.py` (terminal 65ee086). The earlier reset certificate and last-
event independence proof are in `reset_certificate_review.md`.

Let E={Y<=-k max(X,0)}, r=P(E), and G_l(t)=P(E, exactly l of X,Y>t).
For t>=0, G0=1/4+integral_0^t phi(x)Phi(-kx)dx,
G1=integral_t^infinity phi(x)Phi(-kx)dx and G2=0. For t<0 put
u=Phi(t), v=1/2-u, a=-t/k. Then

```
G0=u^2
G2=v^2+integral_0^a phi(x)[Phi(-kx)-u]dx
G1=2uv+u[Phi(a)-1/2]+integral_a^infinity phi(x)Phi(-kx)dx.
```

The separate k=0 and infinite-k limits in the source agree with the
corresponding rectangular events. The binomial convolution is correct
and returns unnormalized masses summing to r, not one. Conditioning on
the reset pair requires division by r; the terminal integrals correctly
use joint masses without that division.

For last reset ending at T, N=T-2 earlier ordinary inputs include target
index zero. Its marker integral is
`phi(t)[(N-1)Phi(-t)Phi(t)^(N-2)G0(t)+Phi(t)^(N-1)G1(t)]`.
Independently, exchangeability gives `(r-P(X second,E)-P(Y second,E))/N`.
Both forms in the source are correct.

For last reset ending at T-1, the final input must be positive. The exact
ray readout is computed from rational stored weights with first-index
ties. When its winner is the final index, n=T-3 older ordinary inputs
give the integral over v>0 of
`phi(v)[n Phi(-v)Phi(v)^(n-1)G0(v)+Phi(v)^n G1(v)]`.
Reversing integration uses `J(u)=n integral_u^1 z^(n-1)(1-z) dz` and
produces the stated beta-tail reference. The sum is a partial accuracy
contribution; unresolved event mass is 1-3r/2. It is not a full estimator.

Two numerical limitations are concrete. Adaptive quadrature initially
missed the k=1e8 boundary layer despite a zero reported error; splitting
at 1/k,3/k,8/k repairs the reported controls but is not certification.
The source also uses max(0,CDF difference) where the exact integrand is
nonnegative. Adjacent floating CDF evaluations can violate monotonicity,
so this is a floating safeguard, not exact positivity or an outward
bound. Tail allowances address truncation only. Future rare-event use
must retain these qualifications and charge any stable-interval repair.
