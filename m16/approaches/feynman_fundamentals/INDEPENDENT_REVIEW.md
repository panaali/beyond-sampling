# T47 agent's independent review of T55

Review delivered 2026-10-17 by `small_h2_34`. The agent read the implementation and
used separate direct formulas; it changed no T55 files. `independent_review.py`
preserves its supplied exact cases and script for reproduction. It was saved by
T55 after the review, without rerunning the experiments.

## Algebra
The reviewer independently verified the normalized Gaussian radius independence,
hemisphere density k/(2*pi) v0^(k-1), Green-identity recurrence, conormal sign,
and hemisphere/octant normalization. It checked sufficiency of the top-two
prefix records and first-index handling of positive-volume ties.

## Numeric falsifiers
Six all-identical-output-row cases use prefix dimensions 1,2,3,4,8 and positive,
negative, zero, largest, second and non-top-two index0 records. Because prediction
is identically0, their correctness is an elementary function of the independent
chi radius. Separate 1-D positive integration agrees with the polygon primitive
to 1.832e-15. All exact prefixes/models are preserved in the script.

An H1T5 model with output rows1 and2 tied at+1, other rows-1, zero recurrence and
positive input weight predicts1 for a positive last input and0 otherwise.
For prefix[-.9,.4,1.2], the primitive gives .10712475639961178. A separate direct
classification of262144 fresh conditional draws (RNG552204) gives
.10657501220703125, SE .0006026800033326281, disagreement .9122 SE.

The complete independent numeric check used .134475753 CPU seconds. Its random
draws are validation, not part of the analytic primitive. These costs are separate
from the author experiments in README.md.

## Limits
This is bounded verification of a conditional primitive, not confirmation of a
full estimator, a competitive compute budget, or a solved challenge. Floating
polygon clipping still uses tolerances. High-degree moments can cancel, including
positive incorrect values. A failed geometric upper-bound guard requires higher
precision or a justified error interval; passing it is not a certificate. Only
the DRI may confirm a project-level claim.
