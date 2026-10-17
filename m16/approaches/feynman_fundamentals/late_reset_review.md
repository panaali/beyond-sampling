# Structural reset factoring: independent code and closed-form check

The mathematical certificate is recorded in primitive_and_reachability_review.md.
Read-only inspection of both factored kernels confirms that zero[step+1] is
set from the conjunction of all current inactive gates, and recomputed from
the earliest changed stage after each selected root. Earlier zero flags
remain valid on the retained prefix. The strict earliest-root scan chooses
the earliest stage on simultaneous roots, so affected downstream flags are
not reused across a changed predecessor. Wi0 and constant-input branches
implement their exact sign cases. This avoids cancellation-based reset tests.

`check_late_reset_review.py` supplies a separate closed-form fixture without
using the author's Fraction gate enumerator. Let H3,T4, Wi=(.7,1.3,0), Wh=-I,
and circle input x=(0,sigma,0,sigma*t), sigma=+1 or-1. The third input resets
all neurons. Readout predicts3 exactly when the final input is positive,
otherwise0. Stable-index ordering of the two identically zero inputs gives
correct t in(0,1) on the positive chart and t>0 on the negative chart.
The total probability is (pi/4+pi/2)/(2*pi)=3/8.

Both kernels return .375 exactly on24 cases: four orthogonal phase frames,
three input-weight scales1,1e-12,1e12, and two precision variants. Every case
has full coverage and an observed late structural reset. This simultaneously
tests a late reset, identically zero input, zero input weight, positive-volume
input/output ties, and phase/scale invariance. Work took .003263414 CPU seconds
after imports. Source hashes and all counters are retained in the JSON.

The owner's unchanged trained-circle comparisons now match long/mixed piece
counts on all32 H2 and96 H16 planes. This corroborates the reset mechanism
on the frozen corpus; it is not a theorem that arbitrary floating gate-root
topology is certified, nor a new deductive estimator over random outer planes.
The initial missing-declaration compile failure remains in the owner's lane.
