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

## Next cycle

Use the weighted primitive inside a bounded two-input network-tail enumerator;
compare exact H2T3 or synthetic closed-form cases before integrating any long
prefixes. Charge all geometric partitioning, special functions and higher-
precision fallbacks. Coordinate the implementation with T50 to avoid duplication.
Separately, the normalized transport and reset marginalization remain distinct
candidate formulations; root should track implementation/review as backlog items.
