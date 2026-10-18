# Frozen source-phase discriminator

## Data and fixed groups
Use every saved row of train, holdout1 and holdout2, in original chunk order.
Use original weights, lookup coefficients, original hidden/logit arrays and
saved lookup/network/true labels. No fitting, input generation or neural replay.
Do not use the separable predictor in this phase. Its case2-only scope remains
in the original evidence. There is no adaptive row, feature or neuron selection.

Every row belongs to exactly one of five label transitions:
1. both network and lookup correct;
2. network wrong, lookup correct (network error erased);
3. network correct, lookup wrong (error introduced);
4. both wrong, same predicted class;
5. both wrong, different predicted classes.

Keep the original positive-count bin (0,1,>=2), last-input case (0,1,2), and
original-network picked global rank (1..10) as exhaustive fixed context axes.
The existing table is inherited, not reported as a new discovery. This phase
adds transition and neuron-margin information. No rank threshold is learned.

## Competitor and algebra
Let y be truth, c the first-index argmax of the ORIGINAL stored logits among
classes other than y, h the stored network hidden state, B the stored lookup
coefficients and hhat its reconstructed hidden state. Set v_i=W[y,i]-W[c,i].
In real arithmetic for these stored vectors:

    m = sum_i h_i v_i
    mhat = sum_i hhat_i v_i
    mhat - m = sum_i (hhat_i-h_i) v_i.

There are exactly 16 terms. c is fixed by the original network for this
comparison. On a network error c is its chosen wrong class (including ties).
On correct rows it is the strongest other class. A positive target-versus-c
lookup margin alone does NOT prove lookup correctness: another competitor can
win, and equal margins require first-index semantics. Hence actual saved labels
and the five transition categories are authoritative.

The error-erased category may reveal that a fitted task summary suppresses the
network's error-producing residual. This is an association/algebraic accounting,
not evidence that the summary faithfully reconstructs that mechanism.

## Finite arithmetic outputs to freeze next
Reconstruct hhat with the original three-term einsum expression and lookup
logits with the original stored Wo matrix. Require exact saved-label parity
before advancing each panel/chunk. Preserve every parity failure and stop.
Compute all 16 floating terms and both signed/absolute reductions in a fixed
order; retain raw terms and candidate/original margins. Report

    ((lookup_logits[y]-lookup_logits[c])
     -(original_logits[y]-original_logits[c])) - sum(terms)

as an observed rounding/operation-order discrepancy, not zero by assertion or
an outward error certificate. A cancellation ratio uses null for a zero signed
sum; do not silently divide by zero or infer importance from relative error.
Signed and absolute term totals for every neuron and all five transitions are
reported. No top-k residual intervention or predictor correction is authorized.
Store class ties and exact first-index tie decisions explicitly.

## Validation boundary
Next phase must freeze finite analytic controls (including cancellation, equal
logits, correct-row competitor exclusion, different-wrong-class transitions,
zero term sum and malformed inputs) and a source/resource gate before execution.
Then a separately gated saved-data contraction may run under one20CPU/2GiB
review-lock process, with durable chunk progress, actual counters and a stop on
first mismatch. The independent saved audit must use distinct formulas/reductions
and raw bindings. No author forward, least-squares, RNG, native kernel audit,
new reference, geometry or mass call is part of this task.
