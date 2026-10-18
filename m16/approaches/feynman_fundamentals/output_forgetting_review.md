# T105 output forgetting: independent theorem and rank review

Read-only review of `output_forgetting` through the positive direct-Gaussian
implementation 1f833fa. No author numerical functions are imported by the
independent selected-last check.

Write Wi=(-alpha,beta), Wh=[[a,-b],[c,-d]], with positive parameters,
bc>ad and a²<bc. For any nonnegative initial state (p,q), after one
positive input x let the coordinates be u,v. If u>0, then
p>(bq+alpha*x)/a, which implies v>0. Substitution gives

`a*u-b*v < b*(d-b*c/a)*q - (b*c*alpha/a+b*beta)*x < 0`.

If u=0 the same expression is nonpositive. Thus the next positive
input forces coordinate1 to zero. From this second-axis state, a third
positive input xC preserves coordinate1=0 and gives coordinate2 in
[0,beta*xC]. A fourth input xD>d*xC makes coordinate2 strictly positive.
Consequently the readout is the first-index maximizer of Wo[:,1],
independent of initial hidden amplitude. This is weaker than hidden-state
reset, so there is no conflict with the earlier five-input reset minimum.

The event probability is one quarter times the positive (xC,xD) wedge
angle divided by 2pi: `(1/4)*(1/4-atan(d)/(2pi))`. Earlier inputs are
not erased from the target rank. For selected last value u>0, exactly
one larger input contributes

`m*F^(m-1)*Q*G^3 + 2*F^m*G^2*Q + F^m*G^2*(Phi(u/d)-F)`,

where m=T-4, F=Phi(u), Q=1-F and G=F-1/2. The terms respectively
choose the exceeding input among the ordinary prefix, the two free
positive inputs, and constrained xC. This independently reconstructs
the source's polynomial coefficient. The other target formulas also
follow by excluding their own factor; any target at u<=0 cannot be
second largest with four positive suffix values. Pair g0/g1 partitions
are disjoint; when xC>u and xD<=u, xC lies in (u,u/d).

`check_output_event_review.py` directly integrates the displayed scalar
formula on four fixed Gaussian-coordinate intervals through u=9. It
gives .009222679058888647, versus the author's .009222679058888648.
The remaining selected-target tail is at most Phi(-9)=1.129e-19.
Adaptive error estimates do not certify quadrature or floating rounding.
Only .05536152383968549 of input mass is covered; this is a partial
correctness contribution, not a full accuracy estimate.

The owner's direct-Gaussian rescue replaces inverse-CDF endpoint
quadrature and clips inner xC at8. Per target, omitted configurations
are contained in {target>8} union {xC>8}; hence 2Phi(-8) is a valid
truncation bound independent of the rank polynomial. Summing T such
bounds is conservative. The saved 52-case controls test quadrature
accuracy separately; the earlier inverse-CDF failures remain evidence.

No sampler comparison, portable full cost, exhaustive event covering,
or uniform numerical certificate follows from this review. Overlapping
event unions, other sign patterns and full-estimator complements remain
open, not disproved.

## Four disjoint signed events and selected-target integration

The signed extension preserves the first two positive inputs. From
state(0,v), C+ gives(0,w) with0<=w<=beta*C; C- gives(w,0) with
0<=w<=alpha*abs(C). Hence D+ thresholds d*C or a*abs(C) force the
second axis; D- thresholds b*beta*C/alpha or c*alpha*abs(C)/beta force
the first axis. Strict signs and thresholds keep positive amplitude,
and the four last-sign pairs are disjoint. This checks the extended
event theorem independently of the projective fixtures.

For a free positive target A/B under the +- event, C is the only
possibly positive constrained partner. Given target CDF v, C above the
target supplies the sole exceeding value, leaving
f0(v)=v^m(v-1/2). When C lies below, the m ordinary prefix inputs and
the other free positive input supply exactly one exceeding value,
f1(v)=(1-v)[m*v^(m-1)(v-1/2)+v^m]. Swapping the C/target integrals
gives H(F)=integral_(1/2)^F f0 + integral_F^1 f1, weighted by
phi(C)Q(k*C). Both polynomials have degree at most m+1; Gaussian
quadrature of order ceil((m+2)/2) is exact for them in real arithmetic.
For m6 this isGL4. Under --, both constrained inputs lie below every
eligible positive target, so the contribution factors as pair-wedge
mass times integral_(1/2)^1 f1. Both negative positions are impossible
second-largest targets because the first two suffix values are positive.

The general signed `p0/p1` source correctly splits at C=u and C=u/k,
including k>1 and the mixed-sign cases. These are probability masses,
not conditionally normalized pair laws. The selected-target adapter
can remove unused all-target integration while retaining its controls;
it does not expand event coverage or justify a full-estimator claim.

## Readout-aware enlargement for a negative third input

For C<0, write s=abs(C) and hidden state(w,0), with0<=w<=alpha*s.
For D>0, q=D/s, the second-axis winner j has readout differences
Delta2>=0 against every competitor. Only Delta1<0 can threaten its
margin while the first coordinate is active. At w=alpha*s the margin
divided by s is
`alpha*(a*Delta1+c*Delta2)+q*(-alpha*Delta1+beta*Delta2)`.
The coefficient of q is positive for a threatening competitor. The
other endpoint of each affine branch has nonnegative margin. Thus the
maximum nonnegative root over competitors is a sufficient threshold,
bounded above by a.

For D<0, q=abs(D)/s, the first-axis winner has Delta1>=0. Only
Delta2<0 threatens; its maximal-w margin is
`alpha*(a*Delta1+c*Delta2)+q*(alpha*Delta1-beta*Delta2)`.
The corresponding root threshold is bounded above by c*alpha/beta.
Use strict q above these thresholds unless exact equality ownership is
handled separately. These are sharp thresholds over the stated interval
enclosure for w. Calling them maximal events for the actual network
would additionally require proving that all enclosed w values are
reachable for the fixed preceding inputs. This review does not assume
that reachability or claim a globally largest forgetting event.

The exact `readout_cone.py` certificate clips the finite(w,q) rectangle
at the final gate kink. Each hidden/output margin is affine on each
clipped face, so vertex checks suffice. A nonnegative affine margin
that vanishes at an interior point above the excluded lower boundary
must have a zero-margin vertex there as well; earlier-index ties are
therefore detected. Above the old axis threshold, strictly positive
axis support covers the unbounded remainder. Degenerate k=old leaves
only the excluded finite boundary, which is harmless. The caller's
positive-parameter assumptions remain required.

The selected -- adapter's exact Fraction antiderivative also checks:
f1(v)=(m+1)(v^m-v^(m+1))-(m/2)(v^(m-1)-v^m), with the second term
absent at m0. Integrating these monomials from1/2 to1 is the displayed
`exact_negative_pair_target_integral`. This removes a test-module import
without changing the mathematical integral.

## Negative-positive initial pair

A proposed larger first-pair event begins with A=-s<0. From arbitrary
(p,q)>=0 its new coordinates are
u=max(a*p-b*q+alpha*s,0), v=max(c*p-d*q-beta*s,0).
The supremum of a*u-b*v is a*(alpha+a*beta/c)*s. If u=0 the expression
is nonpositive. If v=0, substitute p<=(d*q+beta*s)/c to obtain
a*(a*d/c-b)*q+a*(alpha+a*beta/c)*s, bounded above by the proposed
supremum. If both are positive, the coefficient of p is a²-bc<0;
substituting the lower p bound yields the same expression. Equality
is achieved at q=0,p=beta*s/c. Thus a following positive input
B>kappa*s, kappa=a+a²*beta/(c*alpha), forces coordinate1 OFF for
every prior state. This is sharp for that uniform OFF property.

The surviving coordinate may be zero; later positive-output guarantees
must still be applied. For a final positive D, B,D provide two positive
values, so any selected second-largest target must be positive. The
initial negative-positive pair still has a coupled joint rank law;
its event probability must not be multiplied into that law a second
time. This derivation alone is not a complete event-mass implementation.

The implemented positive-final two-pair rank law has now been reviewed.
Negative A is always below a positive target; its coupled positive B
contributes joint p0/p1, multiplied with the other pair's polynomial
and the ordinary prefix factor. For selected final D, the C partner
supplies either zero or one exceeding value. No normalization by either
pair's event probability is needed. The three-tail union allowance
accounts for the target and either integrated partner beyond the cutoff.

`check_two_pair_review.py` imports no author functions. Adaptive scalar
quadrature with independent positive Gaussian first-pair integration
gives .007984888967696072 for C+, and .009743822453890811 for C-.
Their sum .017728711421586885 differs from the fixed-rule result by
about1.4e-16. CPU .0191 seconds. Adaptive error estimates are numerical,
not certified bounds; this remains a partial event contribution.

## Negative target sectors completing the signed event family

For A<0,B>0,C>0,D<0, a negative selected A cannot be second largest:
B,C are already larger. For A<0,B>0,C<0,D<0, selected A=-s requires
all m ordinary prefix inputs below -s, B>kappa*s, and both negative
partners below -s. The resulting integrand is
phi(s)Q(s)^m Q(kappa*s)J(s), where
J(s)=integral_s^infinity phi(r)Q(max(s,k*r))dr. At trained k0,
J(s)=Q(s)^2. With kappa0 also, the integral is exactly
1/[2^(m+4)(m+3)].

The `negative_rank.py` source correctly keeps positive target sectors
for ordinary prefix/B and negative sectors for prefix/A/C/D. The
negative-prefix sector uses m-1 other ordinary values and the joint
first-pair tail I(s); C uses Q(max(s,k*s)); D requires partner magnitude
in(s,s/k), hence is zero for k>=1. These are unnormalized event masses.
The split for J at s/k is correct, with its finite cutoff tail reported
separately. No negative A/C/D can enter the positive-target sectors.

`check_negative_target_review.py` independently uses scalar erfc and
adaptive integration, without author functions. The selected trained
negative-A contribution is .00010442263405448726, differing from the
fixed rule by1.79e-17. Zero-threshold controls at m0/2/6 reproduce
1/48,1/320,1/9216. CPU .000438 seconds. The completed eight signed
events cover about .32226979, leaving about .67773021 unresolved;
neither this contribution nor coverage is the full model accuracy.

## Exact four-suffix ambiguity complement

Independent check_complement_review.py rebuilds all25 saved nonzero affine
gate/readout inequalities from the saved rational model, without importing
author certificate functions. For suffix(-1,-1,+1,-1), initial h=(0,0)
and(256,0) produce labels0 and6. Every rebuilt inequality agrees exactly;
the common radius928873407584354211505/12871199735412830144098 keeps at
least half of every center slack throughout the box. Separate direct
Fraction recurrence at all32 state/vertex cases preserves both labels.

With |x_i|<1.25, sqrt(2*pi)<3 and e<3 imply phi(x_i)>1/9.
The exact positive mass lower bound(2*rho/9)^4 is6.614544537666294e-8.
Thus uniform classification for every arbitrary nonnegative prior hidden
state cannot cover this suffix box. This is not an accuracy obstruction
under the actual prefix law, a practical coverage ceiling, or a negative
about state-conditioned/longer-suffix integration. Actual reachable open
prefix boxes are a separate proposed extension, not checked here.

## Actual-prefix support extension

Independent check_prefix_support_review.py verifies both six-input prefix
centers reach the saved hidden states exactly, and recomputes the
Lipschitz constants M=||Wh||infinity and L6=||Wi||infinity*sum(M^k,k0..5).
Prefix perturbation epsilon changes gate step t by at mostM^(t+1)*L6*eps
and readout margin by ||deltaWo||1*M^4*L6*eps. Together with the existing
suffix half-slack bound all25 inequalities retain at least quarter slack.
All8 gates per history are represented. Omitted tied readouts are checked
identical on the whole stable final support, not merely at fixed h.

The rank intervals independently certify unique second input index0 for
both full boxes. Outputs remain0 and6, hence one box correct and one wrong.
Exact factored Gaussian lower bounds have denominator powers3^20 and
3^81411 respectively; positivity follows without trusting the high-precision
probability calculation. This establishes actual open support dependence
on the prefix. The second box is astronomically rare, so this gives no
practical error floor or broad obstacle to probabilistic integration.
