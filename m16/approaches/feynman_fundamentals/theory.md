# What probability are we calculating?

Inputs X_1,...,X_T are independent standard normals. A bias-free network computes
h_t=ReLU(A h_(t-1)+w X_t), starting at zero. Let J(h) be the first index attaining
max_j (W_o h)_j. The desired number is P[J(h_T)=index of second-largest X].
Poor task performance does not make accuracy estimation meaningless: estimating
an accuracy near 30% can still solve that particular estimation problem.

## Cycle 1A: a compact exact state

Fix a candidate target index j and call X_j=u. Process other inputs chronologically,
inserting u at its proper timestep. Carry h, u, and the count
K=min(2,number of processed non-target inputs exceeding u). At the end, the event
is K=1 and J(h)=j. Sum its probability over j. This uses H+1 continuous variables,
plus three discrete count states, rather than a T-dimensional input integral.
The T target-specific passes are real computational work.

The outer integral over u can be removed by Gaussian radial independence. Initially
take R=|u|, a=h/R=0, b=u/R=+1 or -1 with probability 1/2. After k independent
Gaussian coordinates have been introduced, let R be their Euclidean norm. All
tracked state and comparisons are homogeneous, so (a,b,K) depends only on their
direction and is independent of R~chi_k. For a fresh Gaussian input x, write
x/R=tan(theta). Independently of the normalized history,

    theta in (-pi/2,pi/2),
    density = Gamma((k+1)/2)/(sqrt(pi) Gamma(k/2)) * cos(theta)^(k-1).

The exact ordinary-step update is

    a' = ReLU(A a cos(theta) + w sin(theta)),
    b' = b cos(theta),
    K' = min(2,K + 1[sin(theta)>b cos(theta)]).

Increase k by one. At timestep j, instead use a'=ReLU(A a+w b), leaving b,K,k
unchanged. Count-2 mass can be discarded because it cannot become correct later.
The last random-angle integral is elementary after gate/rank/readout root splits.

For H2 this is a three-dimensional continuous state, independent of T. It is not
automatically easy: zero-state atoms, axis pieces and higher-dimensional pieces
coexist, and generic density smoothing can lose their masses. Absolute-magnitude
conditioning would destroy the independence proof. There is no claim that three
coordinates can be replaced by two: hidden magnitude relative to input radius
still influences the next step.

An alternative single-pass formulation carries h/R and the two largest prefix
values/R, together with their two position indices. It has H+2 continuous variables.
Omitting the position indices loses information needed by a position-specific
readout. Whether T passes in dimension H+1 beat one pass in dimension H+2 is open.

## Cycle 1B: two fresh inputs have an exact spherical tail

Let a normalized prefix state be fixed, with independent R~chi_k, k>=1. Leave two
fresh inputs x,y~N(0,1). The direction

    v = (R,x,y)/sqrt(R^2+x^2+y^2)

lies in v_0>0 and has surface density k/(2*pi) * v_0^(k-1). This follows by writing
the joint density as R^(k-1) exp[-(R^2+x^2+y^2)/2], then integrating the total
radius. Every future gate, rank and logit inequality is homogeneous linear in
(R,x,y) within a gate cell. Thus the relevant regions are spherical polygons.
This works if the fixed target input u=bR occurs before, between or after x,y.

For a spherical polygon D, write M_p=integral_D v_0^p dOmega. The spherical Laplacian
satisfies

    Delta(v_0^p)=p(p-1)v_0^(p-2)-p(p+1)v_0^p.

On a great-circle edge, the outward unit conormal nu is constant and tangent to
the sphere. Green's identity therefore gives, for p>=1,

    M_p = (p-1)/(p+1) M_(p-2)
          - 1/(p+1) sum_edges nu_0 integral_edge v_0^(p-1) ds.

At p=1 omit the first term entirely: M_1=-(1/2) sum nu_0 edge_length. The other
base is M_0, the ordinary spherical area. Along an edge v_0(s)=a cos(s)+b sin(s),
so its integer-power integrals are finite elementary trigonometric sums.
The desired probability is k M_(k-1)/(2*pi). Whole-hemisphere moments are
2*pi/(p+1); coordinate-octant moments are pi/[2(p+1)]. Both give the expected
probabilities for every k. k=0 is a separate circle case, not this formula.

The T47 agent independently checked the density, recurrence and orientation
algebra. This is not independent validation of a complete estimator. Positive-
volume ties, including a wholly dead hidden state, require first-index argmax;
they cannot be dismissed as boundary measure zero. Gaussian-weight perturbations
can create or remove such regions.

`moments.py` implements the analytic primitive. `check_moments.py` compares it to
the independent positive simplex integral

    |det V| integral_Delta (V y)_0^p / ||V y||^(p+3) dy.

The check also repeats the recurrence at 100-digit precision. Thin near-equatorial
polygons expose severe relative cancellation in double precision. `tail_mass`
rejects outputs outside a geometric upper bound; that check detects some failures
but is not a rigorous numerical error certificate. Random outer prefixes would
make this a Rao-Blackwellized sampling baseline, not a deductive solution.

## Cycle 1C: what can a two-neuron readout express?

For h_1>0, the predicted class is the upper envelope of T lines
W_o[j,0]+W_o[j,1] r, r=h_2/h_1>=0. Each reachable class owns at most one interval.
The h_1=0 axis and h=0 first-index tie need separate handling. Classes absent from
this envelope cannot be predicted; including the dead-state class gives the
bound accuracy <= number_of_reachable_classes/T. Adjacent envelope intersections
suffice for decision boundaries. This does not imply that the recurrent projective
state alone is Markov: its scale controls the effect of the next Gaussian input.

## Cycle 1D: exact use of a reset

The T49 agent found a two-consecutive-negative-input reset for its H2T8 checkpoint.
Assume that theorem holds for the model under examination; do not transfer it to
other weights without proof. Partition sign sequences by the last '--' ending
at tau. The suffix starts '+' unless empty and has no '--'. The m=tau-2 earlier
inputs are unrestricted IID normals; the reset pair are IID negative-truncated
normals. Their signs contribute a factor 1/4. Final hidden state depends only on
the suffix, but those discarded values still influence the correct target.

Their ranking contribution can be integrated analytically. Set F=Phi(u) and
G=min(1,2F). The probabilities that the discarded prefix has zero or one values
above u are

    P0=F^m G^2,
    P1=m(1-F)F^(m-1)G^2 + 2(1-G)G F^m,

with the first term omitted at m=0. For a predicted suffix index with value u,
use P1 if no other suffix value exceeds u, P0 if exactly one does, and zero
otherwise. If the output predicts a discarded index, integrate that index's u:
only u above the suffix maximum or between its top two values can contribute.
With v=Phi(u), the integrands are polynomials split at v=1/2 and the suffix's two
CDF thresholds. Reset-index targets use density 2 dv restricted to v<1/2;
ordinary discarded targets use dv. This is exact analytic marginalization.

The no-reset remainder must still be computed. At T10 it has probability
Fibonacci(12)/2^10=144/1024, too large to discard at the desired accuracy.
The proof and formulas are deductions; no reset estimator has been implemented.

## Cycle 1E: an all-length analytic sanity family

For H1, zero recurrent weight, and positive input weight, h_T=max(0,b X_T).
Let J uniquely maximize the output weights. The dead-state prediction is index0.
For T>=2:

- J=0: accuracy=1/T.
- J is neither0 nor the last index: accuracy=1/T by exchangeability.
- J is the last index:
  accuracy=1/T+[1/2-(T+1)2^(-T)]/(T-1).

Negative input weight reverses the correction's sign. The formula follows from
P[last input is second-largest and negative]=(T+1)2^(-T)/T. Conditional on the
last input being negative, all other target positions are exchangeable. This
family tests ties and rank semantics with no numerical integration, but does not
solve the trained H2 ladder.

A separate closed-form check for the T4 tail uses R~chi_2 and c=u/sqrt(1+u^2):

    E Phi(uR) = 1/2+c/2,
    E Phi(uR)^2 = 1/4+c/2+c atan(c)/pi.

The second identity follows by integration by parts in the Rayleigh density,
then the bivariate Gaussian wedge identity. With all output rows identical,
prediction is always index0. If its prefix value is the largest, correctness
probability is 2 E[Phi(uR)(1-Phi(uR))]; if it is second, E Phi(uR)^2; if lower,
zero. `check_rayleigh.py` checks four positive/negative prefix examples without
any numerical reference integration.

## Next cycle: transport must retain interior rays

The two-input enumerator is now implemented and independently checked on bounded
synthetic cases. For T4, a prefix of length2 has a uniform circle direction, so
the entire accuracy reduces to one angle integral with an analytic k=2 tail.
The H2T4 and H4T4 owners have this primitive for an independent numerical route.
Classification and full compute accounting remain DRI decisions.

At fixed target u, the raw hidden-density update has a useful structure: a linear
map, rank-one Gaussian convolution split at x=u for the count, followed by ReLU.
Finite-volume or FFT transport could avoid the previous moment-closure failures.
However, a smooth interior density plus axis masses is not a sufficient
representation. Even the simplest reset example creates an interior singular ray.

Take H2, w=(1,1), A=diag(1,2), h0=0 and two independent inputs x,y. Then

    h1=(max(x,0),max(x,0)),
    h2=(max(max(x,0)+y,0), max(2 max(x,0)+y,0)).

Its exact law has four disjoint strata:

- Atom at0: x<=0,y<=0, or x>0,y<=-2x. Mass
  1/4 + atan(1/2)/(2*pi).
- Interior ray(q,q), q>0: x<=0,y>0. Mass1/4, density (1/2)phi(q)dq.
- Axis(0,s), s>0: x>0,-2x<y<=-x. Mass
  [atan(2)-pi/4]/(2*pi), density phi(s/sqrt(5))/sqrt(5)*Phi(-3s/sqrt(5)) ds.
- Two-dimensional interior b>a>0: x=b-a,y=2a-b. Mass3/8 and
  density phi(b-a)phi(2a-b) da db.

The masses sum exactly to one. A representation containing only a smooth
interior, axes and origin would silently lose a quarter of the probability.
This is an exact limiting-case falsifier for proposed transport code, not a
claim about a trained checkpoint. A deterministic target insertion can also
produce shifted line strata. Repeated random input generally smooths a ray into
an interior density, but new reset atoms continually create new rays.

The next bounded T55 cycle should preserve these strata and counts in this toy
law before attempting any long-horizon density grid. The alternatives are that
explicit strata remain cheap, or that conditioning/rank constraints cause their
description to grow too quickly; mass conservation and this exact four-part law
are the first falsifiers. Separately, reset-prefix marginalization remains an
independent candidate. Root should track implementation/review follow-ups in the
backlog; T55 does not modify shared files or other owners' estimators.

## Cycle 3: positive fallback with a quadrature bound

For a triangle with ray matrix V, the moment integral is
|det V| integral_Delta a(y)^p q(y)^(-s)dy, where a=(Vy)_0,
q=||Vy||^2 and s=(p+3)/2. Scale each ray to L1 norm1. For rays in a common
orthant, q is between1/3 and1. More locally, for any unit separating vector c,
q>=(min_i c dot v_i)^2, while q<=max_i ||v_i||^2 by convexity.

Choose qlo,qhi, center C=(qlo+qhi)/2 and r=(qhi-qlo)/(qhi+qlo). Expand
(q/C)^(-s) as a binomial series in z=(q-C)/C. Its degree-K truncation has a
uniform remainder bounded by

    R <= t_N/(1-rho),
    N=K+1, t_N=(s)_N r^N/N!, rho=r(s+N)/(N+1)<1.

The successive absolute term ratios decrease because s>1. Multiplying by
amax^p C^(-s), where amax is the largest vertex coordinate0, bounds the
integrand error. Under the Duffy map y=(u,(1-u)v,(1-u)(1-v)), a polynomial
of simplex degreeD has coordinate degree at mostD; its Jacobian adds one.
Therefore n-point tensor Gauss-Legendre quadrature integrates the truncated
polynomial exactly when p+2K+1<=2n-1.

The positive quadrature weights sum to the simplex area1/2. Both the true
integral and quadrature incur at most that area times the uniform remainder.
Thus a bound on the moment quadrature error is

    |det V| amax^p C^(-s) t_N/(1-rho).

Multiply by k/(2*pi) for tail probability and sum bounds over polygon triangles.
The T48 agent independently checked the degree count, factor2, and ratio bound.
`positive.py` implements this route, using tensor orders no greater than128.

This is a quadrature-error bound in real arithmetic on the supplied polygon,
NOT an interval-arithmetic certificate. Roundoff can exceed that bound when the
series remainder is extremely small. The saved tests explicitly expose such
cases instead of inflating the mathematical bound or calling them certified.
Across the tested polygons actual errors remain below1.48e-15. The supplied
trained H4T4 failing triangle is recovered as1.1427937453027048e-16 versus
the 100-digit reference1.1427937452910193e-16, using16 quadrature nodes.
Its difference1.17e-27 exceeds the much smaller9.53e-34 quadrature bound,
demonstrating the separate floating-geometry/determinant error limitation.

`tail_mass_with_fallback` preserves the original recurrence/guard and invokes
positive integration only after an explicit ArithmeticError. It records fallback
count, quadrature nodes and the quadrature-only bound in caller statistics.
Passing the original geometric guard still does not prove total numerical error.

## Cycle 5: analytic inverse-sixth norm on a segment

Let A,B be nonzero vectors in a common closed orthant, r=||A||, s=||B||,
c=(A dot B)/(rs), theta=acos(c), and z=1-c^2. Thus 0<=theta<=pi/2.
Set

    a = integral_0^1 t^4 / sqrt(1-z*t^2) dt
      = [3*theta - sin(theta)*c*(3+2*z)]/[8*sin(theta)^5],
    b = 1/4 - c*a,
    d = (1-2*z/3)*a - c/6.

The desired integral is

    I6 = 1/(rs) * [(r^-4+s^-4)*a
          + 4*(r^-3*s^-1+r^-1*s^-3)*b + 6*r^-2*s^-2*d].

All three coefficients are positive. At theta=0 their limits are
(1/5,1/20,1/30), giving the correct collinear formula; for perpendicular
unit vectors I6=2+3*pi/4. To derive this, parameterize points on the segment
by polar angle phi from A. Their radius rho satisfies

    1/rho = [sin(theta-phi)/r + sin(phi)/s]/sin(theta),
    du = rho^2/(rs*sin(theta)) dphi.

Expand the fourth power of the bracket. The normalized angular coefficients
are J40/sin(theta)^5=a, J31/sin(theta)^5=b and J22/sin(theta)^5=d, where
Jij=integral_0^theta sin(theta-phi)^i sin(phi)^j dphi. Elementary integration
gives the stated formulas. The T48 owner independently derived all three.

For z<=1/4 use the positive expansion

    a = sum_{n>=0} [(1/2)_n/n!]*z^n/(2*n+5).

Each successive term ratio is less than z, so after N terms the remaining
sum is at most term_N/(1-z). The implementation uses28 terms. Multiplying
this bound by

    1/(rs) * [U + |c|*V + |1-2*z/3|*W],
    U=r^-4+s^-4, V=4*(r^-3*s^-1+r^-1*s^-3), W=6*r^-2*s^-2,

propagates it conservatively to I6. This excludes floating roundoff. The
closed-form branch reports no numerical error certificate. Endpoint scaling
by their largest absolute coordinate reduces avoidable overflow. The API
rejects zero endpoints and opposite signs in any coordinate.

In the simplex Duffy map the final coordinate has constant Jacobian factor.
Replacing its Gauss rule by this exact segment integral removes one quadrature
dimension. Integrating a polynomial in that coordinate does not increase its
degree in remaining coordinates, so the existing polynomial-approximation
quadrature bound can retain its degree argument. This does not itself certify
floating cone geometry or establish a favorable total cost for T6.
