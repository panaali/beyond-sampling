# T39 independent mechanism comparison

## Question
Which H-M1..H-M8 claims agree with, conflict with, or are not covered by Martin
Keane's third-party AlgZoo analysis? In particular, does the reported5.4-step
phase wheel conflict with the8.4-step gated mode, or refer to a different
operator/experimental regime? Task claim24ead14 was pushed before this work.

## Method and frozen scope
Read the current qualified project ledger and the primary third-party documents
and relevant source, without executing fetched code. Pin repository commit and
document hashes; record URLs/read dates and post-Jan-2026 source dependence in
an owned source log for root to integrate into lit/web_log.md. Cached project
summaries are leads, not substitutes for checking the primary text.

Build an eight-row claim comparison with exact source pointers and limits.
Independently recompute raw Wh and published common-gate restricted spectra
from the identical public example checkpoint, using at most a few tiny16x16
eigendecompositions. Distinguish eigenfrequency, gated impulse trajectory,
forced trajectory and regression slope. If impulse claims need checking, freeze
a finite explicit impulse schedule before any numerical run. Do not perform
new broad Gaussian sampling, intervention sweeps or parameter fitting.

Alternatives: (1) period discrepancy could reflect raw versus gated operators,
tested by independent eigenvalues; (2) different checkpoint or eigenmode
selection could explain it, tested by source/checkpoint provenance and the full
complex-pair list; (3) an impulse-only statement could fail under generic input
forcing, separated by source scope and an exact recurrence identity rather
than assuming transfer. Antisymmetric readout offsets and stable gate histories
are distinct claims, not evidence for every proposed neuron role.

## Status
Primary-source audit underway. <=110CPU/2GB per cycle, shared locks and
single thread for numerical work. No edits to author/shared files and no claim
self-confirmation. Claude remains DRI.

## Results and reproduction
Pending. Commands, source provenance and numerical checks will be retained.

## Compute accounting
Documentary reads and bounded matrix algebra; charge checkpoint loading, all
decompositions/forward trajectories and setup CPU. No accuracy-estimator or
equal-compute claim is implied by a mechanism comparison.

## Claims
[inferred] Comparing periods requires specifying the actual recurrent operator.
[speculative] The apparent5.4/8.4 discrepancy is definitional, pending direct
source and weight checks. A third-party qualitative account is not an
independent proof of Gaussian accuracy.

## Alternatives not ruled out
Checkpoint mismatch, gate/input-history dependence, nonnormal transients,
mode-selection ambiguity, and off-distribution impulse behavior. Uncovered
claims remain unverified rather than assigned agreement by analogy.

## Failure modes
External documents may be exploratory, superseded or contain unsupported
claims. Raw/gated eigenvalues do not prove causal encoding or predict an entire
forced trajectory. Public-source reads afterJanuary2026 are explicitly logged;
no independence claim based on a pre-publication cutoff is made.

## Frozen bounded algebra protocol (before execution)
Use the public example checkpoint SHA256
498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb,
converted to float64. Compute complete spectra of raw Wh and principal gated
matrices for masks 64975,63951,65519,62927,65007, plus the n10/11/12 block;
report every complex pair and select the largest-modulus complex pair explicitly.
No new input distribution sampling is involved.

Run all 90 ordered two-positive-impulse positions with M=1 and each fixed
S in {0.2,0.8,0.99}; all other inputs zero. Save final correctness, countdown
diagnostics after both impulses, and post-second gate histories. Compare histories
only over their overlapping lengths for fixed gap/direction, and distinguish
translation invariance within a magnitude ratio from changes between ratios.
At S=.8 additionally run each sequence at common scale3 to check homogeneity.

For S=.8, fit the same balanced position-additive offset model as the source
(F0=G0=0), solely to audit its claimed symmetry and R². This is diagnostic
fitting, not an estimator or a generalization claim. Repeat the symmetry check
on a fixed synthetic antisymmetric array constructed from sin(i+2j+1) in
three feature channels, with no learned weights. Compare least squares against
the independent row-sum formula F_i=(r_i-r_0)/n, G=-F. Antisymmetry alone
does not entail a readout discrimination sign; give an explicit counterexample.
Record all sequence/hidden-step/readout/decomposition/fit counts, CPU including
loading and imports, source and checkpoint hashes, and RSS. Numerical budget
<=20CPU on review lock, no follow-up sweep. Source code fetched from Keane is
read only and is not executed.
