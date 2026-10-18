# T259 margin-loss source and proof discriminator

## Question
Does replacing a common linear logit fit by unweighted pairwise squared-margin fitting create a genuinely different candidate?

## Method
PROOF.md derives output-linear ridge/readout commutation and the exact pair-loss normalization. SOURCE_REVIEW.md binds its assumptions to five text-only source snapshots at a9b09a14. PROTOCOL.md separates this written phase from any later numerical comparison. manifest.json supplies byte and Git provenance.

## Status
Written phase prepared for independent recover_34 review. No numerical phase or fitted outcome.

## Results
[inferred] With common features/routing and matched quadratic regularization, all-pair squared margin fitting produces the same exact class differences as ordinary ridge logit fitting. For unordered pair sums the data factor is K; ordered sums 2K; either pair mean 2/(K-1).
[inferred] The two shared-design fits in phaseC_diag.py commute with the readout in exact arithmetic. C2's per-neuron selection, ReLU and casts require separate treatment.
Reproduce the frozen source text with `git show a9b09a14:m16/approaches/neuron_account/encode.py` and the other manifest paths. The derivation is written, with no executable experiment command.

## Compute accounting
Zero model/checkpoint/array/fit/forward/RNG calls. Only source text reads, hashing, Git copying and documentation preparation. No numerical compute ceiling or experiment CPU measurement claimed.

## Claims
All mathematical/source conclusions above are [inferred], awaiting independent written review. No [measured] target performance result is added. A useful weighted/nonlinear or richer-feature alternative remains [speculative].

## Alternatives not ruled out
Different feature families, input-dependent or class-weighted losses, top-rival/hinge/cross-entropy objectives, ReLU constraints, recurrent losses, new regularization and more accurate computation. Feature-set insufficiency and objective mismatch remain separate explanations. No variants were run in this phase.

## Failure modes
Unmatched normalization/penalty, per-output feature choice, target-dependent refitting, different singular minimizers, nonlinear clipping and finite arithmetic can invalidate or limit transfer of the theorem. Successful shared source solves are assumed; pooled zero-design failure is not repaired here. Target source results and descriptions remain subject to independent T258/T260 scopes.
