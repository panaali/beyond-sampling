# Whole-box uncertainty discriminator

This is a test of the fixed zero-anchor/global-absolute-Lipschitz Chernoff
bound, not a test excluding all weight-derived or rank-conditioned methods.
The two original biasless T4 models and every pairwise bound are reused from
independently reviewed T187. Input index k corresponds to age3-k. For contrast
m_jr, the reviewed global bound gives

    |m_jr(x)-m_jr(y)| <= sum_k L_jr,k |x_k-y_k|.

At the zero input all scores and margins vanish. For each ordered top-two
identity pair(i,j), freeze largest value s in[1,2], second value t in[0,1].
Anchor every omitted coordinate at zero. Then |m_jr(anchor)|<=2L_jr,i+L_jr,j.
The actual omitted coordinates, conditioned ONLY on these top-two labeled
values, are independent standard normals truncated above t. No omitted rank
ordering is imposed. Their density maximum is at most2phi(0)<1 because t>=0.
For any real density bounded by M, integrating
P(|X-a|>u)>=max(0,1-2Mu) gives E|X-a|>=1/(4M). Consequently E|X|>1/4 here.

Let S_jr=sum over the two omitted coordinates of L_jr,k. Require S_jr>0 and
2L_jr,i+L_jr,j<=S_jr/4. Then the expected absolute allowance strictly exceeds
the absolute anchor margin. Convexity of log MGF proves the corresponding
Chernoff family cannot improve below1 at ANY nonnegative rate. If this holds
for every rival, the fixed candidate's conditional accuracy interval is[0,1]
throughout the whole box, regardless of anchor prediction. Nonzero-weight
zero margins are also unknown. The S>0 guard deliberately does not classify
constant zero-allowance/tie cases as undecided. A failed sufficient inequality
is unresolved, not evidence of a useful bound. All72 tests are retained.

Each ordered-label box has unconditional Gaussian probability

    B=(Phi(2)-Phi(1))*(Phi(1)^3-Phi(0)^3)/3,

because its density in s,t is phi(s)phi(t)Phi(t)^2. Boxes for distinct ordered
labels are disjoint except null boundaries. The fixed-top-two and first-index
tie boundaries have zero input mass; identically tied readouts are not silently
discarded (the positive-allowance guard and original candidate rules apply).
If n boxes pass, integrating this candidate's conditional lower/upper bounds
over all top-two conditions leaves interval width at least nB. This is NOT a
lower bound on any point estimator's actual error, nor its MSE. It does not
exclude cancellation, another anchor, tighter reachable/domain/one-sided bounds,
or an adaptive retained-coordinate scheme.

Rational probability enclosure: for x>=0, Taylor polynomials of exp(-z) at
degree31 and32 bound exp(-z) below/above for every z>=0, by the Lagrange
remainder sign. Integrating z=u²/2 termwise gives exact rational bounds on
int_0^x exp(-u²/2)du. Multiply by the reviewed outward rational enclosure of
1/sqrt(2pi), then add1/2; use Phi(-x)=1-Phi(x). The primitive reuses the exact
Machin/alternating-atan and integer-sqrt implementation reviewed in T166.
All interval operations remain rational; MP erfc120 is only a control, not
the endpoint certificate. Monotone interval products/differences enclose B.

For a positive certified width lower endpoint w, N=floor(1/(4w²))+1 guarantees
the largest Bernoulli sampling standard error1/(2sqrt(N)) is smaller than w.
This is a certificate-width/sample-count comparison, not a matched-compute
victory/defeat or a confidence interval. No cost-growth claim follows from two
fixed T4 seed0 models. Actual full undecided mass may exceed this lower bound.

Frozen alternatives/confounds: loose global L may certify vacuity even when
classification is easy (the conclusion concerns the bound only); the sufficient
anchor bound may be too loose to certify a box (retain all failed cases); reused
setup is charged separately from new exact predicates and CDF controls. No model
forwards, training, random inputs, geometry, original mass engine or quadrature.
