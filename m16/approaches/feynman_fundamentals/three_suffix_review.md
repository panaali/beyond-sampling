# T49 three surviving inputs: rank and precision review

Read-only review of `three_suffix_reset/integrate.py` and its exact
rank-subdivided topology. This extends the independently reviewed
reset-pair rank law; it does not independently reimplement the polygon
enumeration.

All surviving sign patterns have two positive suffix values. For a
specified older target value t, let S,L be suffix second-largest and
largest. The target cannot be second largest below S. For S<t<L,
exactly one suffix exceeds t, so every other ordinary prefix and both
reset-pair values must lie below: factor Phi(t)^(N-1)*g0(t). For t>L,
none of the suffix exceeds t; exactly one exceeding value comes from
N-1 other prefix inputs or the coupled reset pair. This gives
(N-1)Q(t)Phi(t)^(N-2)g0(t)+Phi(t)^(N-1)g1(t). There is no N multiplier
because label0 is one specified older position. Integrating against
phi(t) reproduces the implemented formula. Thus the five zero-hidden
winner polygons cannot be dropped as if every predicted label were a
positive suffix position.

Signed positive L1 coordinates y satisfy sum(y)=1. The map x=s*y has
Jacobian s² in simplex coordinates. With Euclidean radius r=s*norm(y),
the Gaussian volume factor becomes
r² exp(-r²/2)/(2pi)^(3/2)/norm(y)^3, times the ordinary triangle
Jacobian. This verifies the source's Duffy/angular/radial factor.
The pair-reset probability is already in g0/g1; multiplying correctness
by it again would be wrong. Event geometry does multiply by reset mass.

The independent precision intervention retains angular16 and the same
source geometry, but changes both radial32 and older-target32 to64 in
a local in-memory wrapper. Contribution becomes .015084632123278732,
only-2.43e-17 from the frozen baseline; per-label changes are at most
1.13e-17. It uses3.283 CPU seconds. Source hash and raw results are saved
in `recovered_precision_review_results.json`. This is a numerical
precision check, not a different topology implementation or a rigorous
quadrature certificate. An owner check launched concurrently was already
in flight; neither run was used to justify further repetition.

The chi3 tail beyond8 bounds missing suffix radii, multiplied by reset
mass; the Gaussian target tail similarly bounds omitted older targets.
Both bounds concern truncation only. Negative Owen-T rounding clamps,
quadrature accuracy and exact-input-to-floating conversion remain
qualifications. The result is one additional partial contribution and
event coverage, not the full H2T8 accuracy or a portable-cost advantage.
