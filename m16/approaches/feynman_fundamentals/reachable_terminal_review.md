# T67 reachable terminal diagnostic: independent review

Read-only review covers reachable_cycle/diagnostic.py and the reused terminal
interval evaluator, possible-label filter, and grid bounds in engine.py.
The frozen H2T4 scope explicitly rejects a possibly reachable target-last
label. Dominated labels have exactly zero success probability under the
componentwise dominance/first-index filter; label0 remains possible at the
hidden origin.

Condition on target index j being second largest. Under the iid input law,
its marginal CDF value q has Beta(3,2) density12q^2(1-q). Of the three other
slots, the unique input above target is uniformly positioned. Conditional on
that position and q, other values are independent truncated normals above or
below target. The code's two prefix quantiles and above_slot index implement
this law correctly: the third non-target slot is the final fresh input for
every allowed j, so final exceedance flag1 means above_slot==2. The joint
weight is target-label prior1/4 times location prior1/3 times the Beta and
two product rules. The factor1/12 is therefore correct.

The production diagnostic uses the normal law truncated to [-6,6], with
q interpreted in that conditional CDF. Every prefix and terminal interval
uses the same normalization. The absolute difference from the original
bounded payoff is at most1-Z^4, reported as7.8927e-9. This bound does not
cover the finite q/prefix quadratures. The pooled mass .75 is correct because
the dominated label3 is omitted, not renormalized away.

Every mesh is evaluated at identical directly propagated reachable states.
Comparison with exact scalar terminal probabilities isolates terminal
interpolation at those probes, not errors of earlier backward-grid stages.
The terminal cache is valid: for nonfinal target indices, the readout's
correct final-input intervals are independent of target u; only truncation
above/below u changes. Direct probes use a new cache for their changing states.

The proposed distinct discriminator partitions these same probes into origin,
two axes, and interior; it adds no new rule or fitted mesh. Independent saved
arithmetic in check_reachable_summary.py reconstructs signed, absolute, and
squared totals to1.74e-18 and pooled mass .75. At h9, baseline absolute error
.00703559 consists primarily of .00627519 on h1=0,h0>0; curvature and gate
meshes reduce that axis contribution to .00071244 and .00066514. Origin
error is zero for every mesh. This establishes a physical axis-resolution
improvement in the fixed diagnostic, rather than an off-support probe artifact.

Final all-inactive gates create positive-volume ties. Intersecting each
preactivation<=0 constraint with the conditional final-input interval gives
the recorded tied mass .18892468. Distinct pairwise readout slopes for every
nonzero active mask suffice to exclude another interval of identical logits
for this model; isolated crossings have Gaussian measure zero. First-index
ownership remains essential at the origin. This distinctness check is
model-specific and is not a general no-ties theorem.

Signed residuals and the source accuracy still change with prefix order.
Absolute improvement at the terminal cannot imply improved full transport
bias: earlier interpolation changes the distribution weighting the terminal
error, and signed contributions can cancel differently. The prior complete
outer-panel calculation indeed ranks the tested meshes oppositely at h9.
Thus the finite axis improvement is supported, while its contribution to
the complete estimator remains unresolved. No additional q/h sweep, global
error certificate, or portable cost advantage is inferred.
