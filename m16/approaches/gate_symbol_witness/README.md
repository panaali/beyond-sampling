# T138 exact gate-symbol sufficiency discriminator

## Question
On the original H16T10 target, can complete input rank and complete gate history, together with the values/indices of the largest three inputs, determine correctness? This is a bounded prerequisite to T3 rotating-symbol research, not T3's all-model ceiling experiment. T138 claim a678be7 was pushed first.

## Method
Freeze64 rational anchors before results. Top indices are (r,r+3,r+6) modulo10 for r=0..7; their values are1,1-d,1-2d for d=1/64 or1/16. The other seven indices in increasing time order receive equally spaced interior points (j/8,j=1..7) in one of four intervals: [-1/2,0], [0,1/2], [1/2,3/4], [3/4,7/8]. Every anchor has distinct inputs. Public checkpoint is the original example_checkpoint SHA256498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.

At each anchor derive its affine linear gate cell and readout map in floating point. With top3 values fixed, search by bounded LP for (a) a strictly correct point and (b) a point with some competitor strictly beating truth. Keep rank order and all nonzero gate inequalities strict; structural zero rows use exact inactive ownership. LP is a candidate generator, not a certificate. Complete all64 finite anchor searches, at most640 LP calls, preserving failures. In anchor order, attempt exact verification of at most the first8 positive pairs, stopping once one pair is certified. Rationalize lower coordinates on a fixed2^30 grid, leave the top3 exact, rebuild recurrence with exact binary-weight Fractions, and verify every rank/gate/output inequality. All distinct lower coordinates and inequalities must have positive exact slack. Record all potential zeros; identical-zero rows are separate algebraic equalities. A common open top3 neighborhood plus two open lower-coordinate neighborhoods then proves summary non-sufficiency beyond a single measure-zero conditioning slice.

Before target searches, analytic H2T4 fixture uses four positive inputs and identity recurrence, Wi=(1,1), Wo rows (1,0),(0,1),(0,0),(0,0): it intentionally has constant tied output0 on the positive orthant, so no false correctness-crossing witness is allowed. A constructive H2T4 control with the same top3 but a lower-coordinate-dependent winner is also required, derived and frozen before execution. Preserve all controls and negative searches.

Competing explanations: (1) E27 lower-input effects require gate switches; an exact same-cell pair would distinguish this. (2) A floating LP crossing is tolerance/roundoff or an input/gate tie; exact rational checks reject it. (3) Dependence is confined to isolated top3 values; strict inequalities give open persistence. No result here estimates probability or accuracy.

## Status
Protocol frozen; implementation next. Independent review required.

## Results
Pending.

## Compute accounting
No random model inputs or MC. Fixed deterministic anchors, LP proposals and exact proof arithmetic only. Count every LP, diagnostic forward and exact recurrence; record CPU including failed stages. One thread,2GB, bounded primary/review locks. This is a structural diagnostic, not estimator cost.

## Claims
[speculative] Top3 values plus complete rank/gate symbols may fail to determine correctness even locally. A finite null search is inconclusive.

## Alternatives not ruled out
Weighted lower-input accumulators, gate-conditioned affine margin coefficients, stochastic integration over discarded inputs, nonlinear state summaries, approximate rather than exact correctness, and other retained coordinates. Full gate history includes rotating-gate history, so a certified collision excludes only summaries which discard all additional continuous information. No family-wide symbolic-dynamics impossibility.

## Failure modes
Floating LP tolerances, exact-zero gates/readout ties, rational rounding crossing a facet, restricted anchors missing collisions, overly loose state reachability assumptions, or confusing a structural counterexample with a practical probability lower bound. This procedure works in actual input space, not arbitrary hidden states.
