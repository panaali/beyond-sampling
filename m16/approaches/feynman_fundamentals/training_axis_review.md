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
