# T107 exact-zero non-leading gate review

Read-only review of the locally pinned `prepare.py`, `fixed.cpp` and
`_source/dd_clip.cpp`; no rebuild or full geometry rerun here.

A zero computed affine form has ReLU output zero and is identically
tight on the whole cone. Keeping the cone, choosing the OFF branch and
adding its incidence bit to every ray is mathematically consistent.
The subsequent allmask includes this constraint, as required for the
lifted apex. This applies to coefficient-exact floating zero, not a
certification that computed coefficients equal the exact-real form.
Near-zero forms still enter the original clipping path.

The universal bit does not change which rays contain the common
incidence set for a candidate pair: all rays acquire that bit. It can
weaken the popcount prefilter, but the subsequent contained-ray test
still determines adjacency under the original complete-incidence
assumptions. Pulling triangulation skips the constraint because its
chosen vertex is incident. Thus a redundant universal constraint is
not incorrectly made a facet by these routines. Inherited tolerance,
rank and underflow limitations remain unchanged.

Independent exact Fraction integration in `check_output_event_review.py`
checks the Wi=(1,0), Wh=0 T5 readout fixture. Last input positive and
second largest contributes13/80. First input positive, second largest,
and last input negative contributes11/128. First input negative and
second largest with last input negative contributes19/640. Their sum
is89/320. Constant first-index readout accuracy is1/5 by exchangeability,
including positive-volume zero/duplicate-output ties.

The separate source controls compare trained geometry and preserve
unsupported leading-allzero-Wi rejection. Parity on those fixtures is
valuable but not a universal topology proof. Native cost, builds,
allzero-leading support and larger dimensions remain separate issues.
