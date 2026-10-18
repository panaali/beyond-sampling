# Conditional uncertainty without imputing its effect

Fix the values and labeled positions of the largest k among T iid standard
normal inputs, all strict. Without imposing any ordering among the remaining
labels, their regular conditional law is independent normal restricted to
X<=b, where b is the kth-largest value. Imposing further ranks among omitted
inputs would destroy this product description and requires different handling.

For a fixed omitted-input anchor a, s>=0, and Phi the normal CDF,

    M_b,a(s) = exp(s^2/2) [ exp(sa) Phi(a+s)
                 + exp(-sa) (Phi(b-s)-Phi(a-s)) ] / Phi(b),  a<=b.

For a>b it is exp(sa+s^2/2) Phi(b+s)/Phi(b). Splitting the Gaussian integral
at a and completing the square proves both identities. At s=0 the result is1.
The expectation is [2phi(a)-phi(b)+a(2Phi(a)-Phi(b))]/Phi(b) for a<=b,
and a+phi(b)/Phi(b) otherwise. All finite real a,b,s have finite moments.

Reuse T187's reviewed global coordinate margin-Lipschitz theorem:
L_j=|Wo_truth-Wo_rival| |Wh|^(T-1-j) |Wi|. For the fixed retained coordinates,
|m(X)-m(anchor)|<=S=sum_j L_j|X_j-a_j|. This follows through every ReLU
breakpoint; no fixed-gate assumption or conditional-mean approximation occurs.
For each lambda>=0, P(S>=m)<=min(1,exp(-lambda*m)*product M_b,a_j(lambda L_j)).
If all L_j=0 the event is known exactly:1 for m<=0 and0 for m>0.

The log bound has value0 and derivative E[S]-m at lambda0, and is convex.
Thus this fixed-anchor/envelope family improves strictly below1 iff m>E[S].
When the precheck fails, searching lambda cannot help this bound. This is not
a lower bound on the true failure probability and does not exclude better
rank/range-restricted bounds, different anchors or extra retained coordinates.

For neural correctness, correct-anchor lower bounds require a union over all
rival margin crossing bounds. Incorrect-anchor upper bounds may use a single
rival that must be overcome. First-index tie inequalities, identically zero
margin/L and threshold-zero cases must be handled explicitly. This scalar
phase does not implement a network estimator or assert these later controls.

The source is a finite MP evaluation of real-arithmetic identities, NOT an
outward numerical probability certificate. Direct-density quadrature is an
independent numerical control, not an exact oracle. Normalization, cancellation
and rounded comparisons remain explicit. No new neural or random input runs.
