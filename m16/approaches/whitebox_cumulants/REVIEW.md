# Independent source review

Reviewer: Codex child `whitebox_public_review`, 2026-10-21. Read-only; no model
experiments or file edits by the reviewer. Reviewed generated commit `d2dabade9`
against pinned original V25 and the frozen reproduction protocol.

No blocking defect was found in the row-blocked derivative. The reviewer checked
that output-side A/P/Z/Zf rows are sliced while right-side feedback factors,
dense source legs, transported K4 vectors and shared-basis factors stay full.
The two old-source tiers preserve their complete contractions. Rectangular
blocks are assembled before global diagonal clearing; nonlinear symmetrization
also follows full assembly. Scratch results are copied before reuse. No ranks,
coefficients, source ages, shape gates or terms were removed.

Conclusion: algebraically equivalent in exact arithmetic. Floating-point
regrouping and metered work can differ. The frozen finite-network parity tests
do not activate the official-shape K4 branch. Root subsequently added separate
synthetic controls for nonzero K4/feedback and ragged-block branches; these pass,
but they are not an independent rerun of a full model. Full-width original-versus-
derivative equality remains unmeasured because the original failed the memory
launch gate. Scores are author-measured, independently source-audited, not
independently numerically replicated.

The reviewer also verified the dataset seed protocol against installed
whestbench 0.16.2 and the published reconstruction helper. Root subsequently
checked all weights of one stored public network against both generators;
provenance and measurements are in [results](results/summary.json) and
[INDEX T474](../../../INDEX.md).
