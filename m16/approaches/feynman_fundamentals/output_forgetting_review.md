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
