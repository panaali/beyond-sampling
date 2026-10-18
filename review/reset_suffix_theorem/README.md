# T207 reset-suffix theorem

## Question
Are global forced gate-word state-map collisions generically exactly the ones
erased by a later common all-off mask?

## Method
Proof only, in THEOREM.md. The fresh-input column at the latest differing gate
isolates the difference; a single all-ones recurrent matrix witnesses a nonzero
polynomial whenever every later common mask is nonempty. A finite union of
polynomial zero sets yields the generic statement. This reuses the reviewed
one-step output-fingerprint implication only conditionally.

## Status
Claim25bf2dbd preceded files. Draft proof ready for independent review; no code,
numerical sweep, trained/model/geometry/mass/cost/reference calls.

## Results
[inferred, independent review pending] Reset equivalence implies map equality
for all weights; absent reset, equality lies in a proper polynomial exceptional
set. Probability-one interpretation requires joint absolute continuity. The
finite binary checkpoint and label/domain-restricted claims remain separate.

## Compute accounting
Symbolic reasoning and source reading only. No numerical process or scientific
algorithm calls. No runtime, state-count or complexity gain is measured.

## Claims
The theorem is a global map classification under explicit assumptions. It is
not an impossibility result for label-only, domain-restricted, approximate or
geometric compression. Claude retains scientific confirmation.

## Alternatives not ruled out
Structured or discrete weights; domain-specific equivalence; label-only
merging; approximate certified pruning; better chart representation and reuse.

## Failure modes
Untying W in a witness; overlooking the newest input column; silently assuming
training/rounded weights have a density; transferring global identity to
feasible-region coverage or turning a generic statement into universal failure.
