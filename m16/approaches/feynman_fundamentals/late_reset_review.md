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

## Axis-readout structural followup
If the final structural active mask has exactly one neuron k, its hidden
value is positive throughout the open gate cell. The winner is therefore the
first index maximizing Wo[:,k], independent of the affine magnitude. With
no active neuron it is index0. Source diffs for both axis kernels implement
this rule using strict greater-than updates on the original readout column;
they retain input-rank roots while skipping only readout roots/products.
The final mask is recomputed on every suffix update, which always includes
the last stage. This is not a numerical near-zero or near-tie rule.

The independent check_axis_readout_review.py uses the same reset circle but
only one nonzero Wi. It checks the prior3/8 case and a new positive-volume
readout tie between rows2 and3. First-index winner2 then makes precisely the
negative chart correct, giving probability1/2. Both kernels return the exact
values on48 phase/scale/readout cases, .006530824 CPU seconds after imports.
Raw counters and source hashes are preserved. This tests an actual tie and
the structural simplification independently of the author's Fraction oracle.
It does not certify arbitrary multi-active floating topology or imply that
every classified-interval count must match an exact oracle on unseen planes.
