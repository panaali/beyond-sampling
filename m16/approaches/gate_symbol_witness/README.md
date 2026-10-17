# T138 exact gate-symbol sufficiency discriminator

## Question
On the original H16T10 target, can complete input rank and complete gate history, together with the values/indices of the largest three inputs, determine correctness? This is a bounded prerequisite to T3 rotating-symbol research, not T3's all-model ceiling experiment. T138 claim a678be7 was pushed first.

## Method
Freeze64 rational anchors before results. Top indices are (r,r+3,r+6) modulo10 for r=0..7; their values are1,1-d,1-2d for d=1/64 or1/16. The other seven indices in increasing time order receive equally spaced interior points (j/8,j=1..7) in one of four intervals: [-1/2,0], [0,1/2], [1/2,3/4], [3/4,7/8]. Every anchor has distinct inputs. Public checkpoint is the original example_checkpoint SHA256498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.

At each anchor derive its affine linear gate cell and readout map in floating point. With top3 values fixed, search by bounded LP for (a) a strictly correct point and (b) a point with some competitor strictly beating truth. Keep rank order and all nonzero gate inequalities strict; structural zero rows use exact inactive ownership. LP is a candidate generator, not a certificate. Complete all64 finite anchor searches, at most640 LP calls, preserving failures. In anchor order, attempt exact verification of at most the first8 positive pairs, stopping once one pair is certified. Rationalize lower coordinates on a fixed2^30 grid, leave the top3 exact, rebuild recurrence with exact binary-weight Fractions, and verify every rank/gate/output inequality. All distinct lower coordinates and inequalities must have positive exact slack. Record all potential zeros; identical-zero rows are separate algebraic equalities. A common open top3 neighborhood plus two open lower-coordinate neighborhoods then proves summary non-sufficiency beyond a single measure-zero conditioning slice.

Before target searches, analytic H2T4 fixture uses four positive inputs and identity recurrence, Wi=(1,1), Wo rows (1,0),(0,1),(0,0),(0,0): it intentionally has constant tied output0 on the positive orthant, so no false correctness-crossing witness is allowed. Constructive H2T4 control: Wh=diag(1,0), Wi=(1,1), readout rows(1,-13),(0,0),(-1,0),(-1,0); inputs(3,2,1,u), u=1/4 or3/4, give h=(6+u,u), output0=6-12u, and truth1. Both have strictly positive gates and identical top3/ranks, while correctness flips. This is derived and frozen before execution. Preserve all controls and negative searches.

Competing explanations: (1) E27 lower-input effects require gate switches; an exact same-cell pair would distinguish this. (2) A floating LP crossing is tolerance/roundoff or an input/gate tie; exact rational checks reject it. (3) Dependence is confined to isolated top3 values; strict inequalities give open persistence. No result here estimates probability or accuracy.

## Status
Done as a bounded exact structural result. Independent T55 direct recurrence, global radius and reverse-adjoint margin reconstruction all pass; gate_symbol_review_results.json records the independent values. DRI registration requested.

## Results
All64 frozen anchors and640 LP calls completed in1.203938813CPU. Eight anchors have floating positive candidates:0,8,9,17,25,32,40,48; all64 statuses, including56 null searches, are retained in search_results.json. Per frozen rule, only the first positive pair needed exact certification; it passed with no failed exact attempts. This is not an estimate of how common ambiguity is.

The exact target pair fixes indices0,3,6 to1,63/64,31/32, respectively. Both have the same strict full input order and all160 gates, but winners are3(correct) and6(wrong). Exact minimum gate/rank/output slacks are in certificate.json. A common input sup-norm radius8.28669939158e-10 preserves both cases and permits shared perturbations of the top3 values. The tiny conservative radius certifies open persistence, not a useful probability bound. T55 independently reproduced the recurrence and radius.

The local constructive rescue also passes. Exact margin3-6 is affine in the seven lower inputs with coefficients approximately [.05032495,.03685721,.17661832,.27759050,.59516045,.19612536,.30373672] at time indices[1,2,4,5,7,8,9]. Its one zero on the segment is at parameter.6988823149957404; all other outputs lie at least2.43219660145 below there. Endpoint/crossing linearity proves exactly one switch on this segment. These coefficients are derived from weights/gates, not fitted. One weighted scalar thus restores local correctness along this segment; no global representation or cost claim follows.

Analytic H2T4 positive and constant-tie controls passed before the target search; post-search LP adapter controls also recover both true winners and reject tied truth. Reproduce each saved artifact with single-thread `/workspace/clean-neurons/.venv/bin/python m16/approaches/gate_symbol_witness/{search.py,certify.py,margin_rescue.py,lp_controls.py}` in that order, with2GB limit and project compute/review locks. Commands denote four separate script invocations.

## Compute accounting
No random model inputs or MC. Fixed deterministic anchors, LP proposals and exact proof arithmetic only. Count every LP, diagnostic forward and exact recurrence; record CPU including failed stages. One thread,2GB, bounded primary/review locks. This is a structural diagnostic, not estimator cost.

## Claims
[inferred from exact arithmetic, independently reproduced] Top3 values plus complete rank/gate history do not determine correctness for the original target model. [inferred, independently reviewed] A single derived affine lower-input coordinate restores correctness on the exhibited segment. Neither statement determines accuracy or the cost of global symbolic integration.

## Alternatives not ruled out
Weighted lower-input accumulators, gate-conditioned affine margin coefficients, stochastic integration over discarded inputs, nonlinear state summaries, approximate rather than exact correctness, and other retained coordinates. Full gate history includes rotating-gate history, so a certified collision excludes only summaries which discard all additional continuous information. No family-wide symbolic-dynamics impossibility.

## Failure modes
Floating LP tolerances, exact-zero gates/readout ties, rational rounding crossing a facet, restricted anchors missing collisions, overly loose state reachability assumptions, or confusing a structural counterexample with a practical probability lower bound. This procedure works in actual input space, not arbitrary hidden states.

### Frozen follow-up after exact witness
The first exact pair passes, with correct output3 and wrong output6. To preserve a constructive rescue, derive the exact affine margin3-6 on this fixed gate word. Evaluate its unique zero on the segment and check that all other outputs are strictly below both there. If so, linearity and endpoint margins prove this segment has exactly one classification switch, controlled by a weighted sum of the seven lower inputs. Record all coefficients, not a fitted feature. This is a local retained-coordinate rescue, not global compression or a new probability estimator; implementation begins after this protocol note.

### Exhaustion and scope checklist
Tested: frozen64 rank/top3/sign-range anchors,640 LP proposals, exact first positive pair, open neighborhood preservation, analytic synthetic positive/no-crossing controls, and one exact local weighted-margin rescue. Untested: other weights/noise, omitted56 finite-search nulls with alternative anchors, probability integration of the ambiguous cell, reduced histories augmented by continuous variables, and global complexity. The seven additional floating positive pairs were not certified because the predeclared existential stopping rule was met. These are unnecessary for the counterexample but cannot be reported as exact witnesses. No broader symbolic family is retired.

### Why this is more than a zero-probability collision
Use the common certified radius r for the three retained coordinates and for two separate seven-coordinate lower-input boxes. For every top-three vector in the common open box, both lower boxes have the same fixed rank/gate symbols and opposite correctness. Under independent standard Gaussian inputs, any measurable deterministic correctness function of just these summaries must therefore make errors with probability at least P(common top box) times min(P(correct lower box), P(incorrect lower box)), which is strictly positive. All boxes have positive volume and finite coordinates. This is only an existence bound; its very small conservative radius supplies no useful global accuracy floor. It does not prevent computing accuracy by integrating the conditional distribution of discarded inputs.
