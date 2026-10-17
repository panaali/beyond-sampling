# T114 independent truth and comparator semantics

Source review of `training_axis_review/reference.py` and its frozen
control protocol. Ordered top-two and sign(x0) inequalities give the
same invertible three-ray cone charts established in the T110 review.
The first gate mask is sign(x0)*Wi>0; subsequent affine recurrence maps
are correct. Exact-zero gate forms are taken OFF. An identically zero
truth-minus-competitor output margin rejects the whole cone when the
competitor has an earlier index, and otherwise preserves it. Nonzero
margin boundaries have zero Gaussian mass, so weak clipping suffices
for the integral. The cyclic polygon fan and determinant/atan2 spherical
triangle formula are valid for the convex pointed charts.

This is an independent floating implementation, not a certified exact
real-arithmetic enclosure. Roundoff in zero tests, normalization and
nearly degenerate intersections remains a limitation. The saved zero
input and zero/identical-output fixtures exposing inherited T75/truth
defects must remain visible. The frozen evaluated recipe should not be
silently altered during the independently planned panel.

If each model's Monte Carlo comparator receives its estimator's actual
cost n_i in forward-pass units, the average comparator MSE is
mean[p_i(1-p_i)/n_i]. In contrast, mean[p_i(1-p_i)]/mean[n_i] describes
uniform Monte Carlo allocation at the average cost. Both can answer
well-defined questions, but they are not interchangeable when costs and
accuracies vary across models. Report the allocation convention beside
the ratio; shared setup and any reused counts need explicit attribution.

The frozen relative-noise control anchors each training amount to its
seed's global-RMS scale relative to the fully trained model. This is a
defensible separate perturbation protocol, not function-equivalent noise
or invariance under arbitrary hidden-unit rescaling. Keep the absolute
noise panel as its original comparison and distinguish the relative
panel. Neither by itself establishes intrinsic robustness as a universal
function of training amount.

## Tight absolute-noise panel arithmetic

`check_training_bulk_review.py` independently reads the256 frozen
k24,seed4 absolute-noise rows.60-digit arithmetic gives the per-model
allocation ratio .745153320481545068, versus own-counter uniform
average-budget ratio .747009609599737775. The historical-counter
uniform ratio is .730732685765223589. Independent histogram aggregation
of the same1000 frozen bootstrap resamples reproduces the percentile
interval [.5651172397240184,.9291922067425471]. All reference probabilities
are in(0,1), all costs positive. There are50 individual losses and the
largest absolute error is .0600192364653358.

This checks saved arithmetic, not the complete scalar instrumentation.
The interval is a pointwise noisy-copy bootstrap with finite-resampling
variation, not simultaneous coverage or training-seed generalization.
Individual losses do not themselves refute the ensemble-MSE criterion.
The reused R3/R4 cost conventions and floating reference qualifications
remain explicit; no additional estimator runs were made for this audit.

## Protocol-aligned label cost and fresh confirmation

Leaderboard rule4 permits reused model-free preprocessing free of the
charged model-dependent budget when declared as PRECOMPUTE_FLOPS.
Cold setup sensitivity is a separate metric, not a protocol blocker.
Rule5 also makes one input and its true label free. The R3 sample
source explicitly performs three input-label comparisons. Removing
these leaves56 multiplications,41 additions,12 ReLU comparisons,
two prediction-argmax comparisons and one correctness comparison:
112 scalar operations rather than115. Both ratios and their paired
bootstrap quantiles multiply by115/112. Historical fields stay visible.

The independent `check_label_cost_review.py` confirms the corrected
256-copy ratio .7651127844230150256, interval[.5802543086451974,
.9540812837088652], and51 individual losses. The separately frozen
512-copy confirmation gives .7647193222412240735, interval
[.6170097881466432,.9269774651104496], with86 individual losses.
These are pointwise finite-noise-panel results under the stated scalar
convention, not a universal robustness or full-challenge claim.

The fibre counter is not reconciled by this change. Its inherited
ops.n uses one unit per dot-product multiply-add, while ops.x adds
elementwise arithmetic and comparisons, and the sampling denominator72
uses the historical affine-only convention. This is a hybrid legacy
comparison, not the same all-scalar unit as T75/112. In addition,
`rows_at` executes dense Wh@Ag while its inherited charge scales with
active gates and used columns. A complete as-implemented audit must
resolve such gaps rather than assume a uniform conversion factor.
Preserve those results as qualified diagnostics; shared-cost recounting
is a separate task. No estimator outputs need rerunning for this
arithmetic correction.
